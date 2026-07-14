"""Teacher-facing Stage D deployment router.

The teacher loads ``StudentAgent`` through its fixed ``act(obs) -> int``
contract.  This adapter receives the raw-state observation, routes by the
runtime layout and physical player index, and delegates to the validated Stage
D specialists without any absolute local paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from policies.basic_policies import GreedyFullTaskPolicy
from policies.scenario2_guided import StudentAgent as Scenario2GuidedAgent


_POLICIES = Path(__file__).resolve().parent
_ARTIFACTS = {
    "asymmetric_advantages": _POLICIES / "asymmetric_advantages_distilled.pt",
    "counter_circuit": _POLICIES / "counter_circuit_distilled.pt",
}
_MODELS: dict[Path, "_ActorCritic"] = {}


class _ActorCritic(nn.Module):
    """Minimal inference-only reader for the existing Stage D artifacts."""

    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], activation: str):
        super().__init__()
        layer_type = nn.Tanh if activation == "tanh" else nn.ReLU
        layers: list[nn.Module] = []
        previous = int(input_size)
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, int(hidden)), layer_type()))
            previous = int(hidden)
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(previous, 6)
        self.critic = nn.Linear(previous, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.backbone(observation))


def _load_model(path: Path) -> _ActorCritic:
    """Load one checked-in inference artifact once per Python process."""
    if path in _MODELS:
        return _MODELS[path]
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage D artifact is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model_spec = payload.get("model", {})
    parameters = model_spec.get("parameters", {})
    if (
        payload.get("profile") != "inference"
        or model_spec.get("architecture") != "mlp_actor_critic"
        or int(model_spec.get("num_actions", 0)) != 6
    ):
        raise ValueError(f"Incompatible Stage D inference artifact: {path}")
    model = _ActorCritic(
        int(model_spec["input_size"]),
        tuple(int(size) for size in parameters["hidden_sizes"]),
        str(parameters.get("activation", "tanh")),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    _MODELS[path] = model
    return model


class _Scenario4FixedPotB(GreedyFullTaskPolicy):
    """The selected deterministic Scenario 4 specialist."""

    def _pots_that_can_accept_ingredients(self, state, pot_states):
        candidates = super()._pots_that_can_accept_ingredients(state, pot_states)
        pots = sorted(self.mdp.get_pot_locations())
        return [pot for pot in candidates if pots and pot == pots[-1]]


class _AsymmetricReachableFullTask(GreedyFullTaskPolicy):
    """Full-task teacher that rejects unreachable cross-wall targets."""

    def _nearest(self, origin, positions):
        valid_positions = set(self.mdp.get_valid_player_positions())
        reachable: list[tuple[int, tuple[int, int]]] = []
        for target in positions:
            goals = {
                position
                for position in self._adjacent_positions(target)
                if position in valid_positions
            }
            path = (
                self._bfs_shortest_path(
                    origin,
                    goals,
                    valid_positions,
                    blocked=set(),
                )
                if goals
                else None
            )
            if path is not None:
                reachable.append((len(path), target))
        return min(reachable)[1] if reachable else None


class _CounterCircuitMixedRecipe(GreedyFullTaskPolicy):
    """Tomato specialist that completes the onion partner's valid recipes."""

    def __init__(self):
        super().__init__(ingredient="tomato", avoid_teammate=True)

    @staticmethod
    def _ingredient_names(soup) -> list[str]:
        ingredients = getattr(
            soup,
            "ingredients",
            getattr(soup, "_ingredients", ()),
        )
        return [item if isinstance(item, str) else item.name for item in ingredients]

    @staticmethod
    def _can_extend_active_order(ingredients: list[str], orders) -> bool:
        for order in orders:
            remaining = list(order)
            for ingredient in ingredients:
                if ingredient not in remaining:
                    break
                remaining.remove(ingredient)
            else:
                return True
        return False

    def _pots_that_can_accept_ingredients(self, state, pot_states):
        candidates = super()._pots_that_can_accept_ingredients(state, pot_states)
        valid: list[tuple[int, int]] = []
        for position in candidates:
            soup = state.objects.get(position)
            if soup is None or soup.name != "soup":
                continue
            current = self._ingredient_names(soup)
            # Wait for the disclosed onion partner to seed a pot. This avoids
            # creating single- or triple-tomato soups when the partner is blocked.
            if "onion" not in current:
                continue
            if self._can_extend_active_order(current + ["tomato"], state.all_orders):
                valid.append(position)
        return valid


class StudentAgent:
    """Route the teacher-facing policy to the validated Stage D specialist."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.agent_index = 0
        self.mdp = None
        self._feature_env: OvercookedEnv | None = None
        self._scripted: GreedyFullTaskPolicy | None = None
        self._scripted_route: str | None = None
        self._scenario2 = Scenario2GuidedAgent(
            {
                "stochastic": bool(self.config.get("scenario2_stochastic", True)),
                "stage": "p3",
                "drop_after": 5,
                "seed": self.config.get("seed"),
            }
        )

    def set_agent_index(self, agent_index: int) -> None:
        self.agent_index = int(agent_index)

    def set_mdp(self, mdp) -> None:
        """Initialize raw-state feature extraction before timed action calls."""
        self.mdp = mdp
        self._feature_env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
        # Building the medium-level planner can be slow, but happens during
        # teacher setup rather than inside the 100 ms per-action wrapper.
        _ = self._feature_env.mlam
        layout = str(mdp.layout_name)
        if layout in _ARTIFACTS:
            _load_model(_ARTIFACTS[layout])
        self._scripted = None
        self._scripted_route = None

    def reset(self) -> None:
        self._scenario2.reset()
        if self._scripted is not None:
            self._scripted.reset()

    def act(self, obs: Any) -> int:
        if not isinstance(obs, dict) or "state" not in obs or "mdp" not in obs:
            raise ValueError("Stage D submission requires observation.type: state")
        state = obs["state"]
        mdp = obs["mdp"]
        agent_index = int(obs.get("agent_index", self.agent_index))
        if self.mdp is not mdp:
            self.set_mdp(mdp)
        self.agent_index = agent_index
        layout = str(mdp.layout_name)
        if layout == "coordination_ring":
            return int(self._scenario2.act(obs))
        if layout in _ARTIFACTS:
            return self._rl_action(layout, state, agent_index)
        return self._scripted_action(layout, state, mdp, agent_index)

    def _rl_action(self, layout: str, state, agent_index: int) -> int:
        if self._feature_env is None:
            raise RuntimeError("Stage D feature environment was not initialized")
        features = np.asarray(
            self._feature_env.featurize_state_mdp(state)[agent_index],
            dtype=np.float32,
        ).reshape(-1)
        encoded = np.concatenate(
            (features, np.eye(2, dtype=np.float32)[agent_index]),
        )
        model = _load_model(_ARTIFACTS[layout])
        with torch.inference_mode():
            logits = model(torch.as_tensor(encoded).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())

    def _scripted_action(self, layout: str, state, mdp, agent_index: int) -> int:
        route = (
            "scenario4"
            if layout == "scenario_4"
            else "counter_circuit"
            if layout == "counter_circuit"
            else "generic"
        )
        if self._scripted is None or self._scripted_route != route:
            if route == "scenario4":
                self._scripted = _Scenario4FixedPotB(
                    ingredient="onion", avoid_teammate=False
                )
            elif route == "counter_circuit":
                self._scripted = _CounterCircuitMixedRecipe()
            else:
                self._scripted = GreedyFullTaskPolicy(
                    ingredient="onion", avoid_teammate=True
                )
            self._scripted_route = route
        self._scripted.set_agent_index(agent_index)
        self._scripted.set_mdp(mdp)
        action, _info = self._scripted.action(state)
        from src.constants import overcooked_action_to_index

        return int(overcooked_action_to_index(action))
