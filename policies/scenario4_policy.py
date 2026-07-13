"""Small deterministic specialists for the disclosed Scenario 4 layout."""

from __future__ import annotations

from typing import Iterable

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

from policies.basic_policies import GreedyFullTaskPolicy


class Scenario4PlannerPolicy(GreedyFullTaskPolicy):
    """Greedy soup planner with deterministic pot assignment and recovery.

    The policy deliberately reuses the base policy's collision-aware BFS and
    interaction geometry.  Its only Scenario 4-specific choice is which pot to
    feed when the random-motion partner obstructs the central corridor.
    """

    def __init__(self, config: dict | None = None) -> None:
        config = {} if config is None else dict(config)
        pot_strategy = str(config.get("pot_strategy", "nearest"))
        blocked_threshold = int(config.get("blocked_threshold", 3))
        avoid_teammate = bool(config.get("avoid_teammate", True))
        seed = config.get("seed")
        super().__init__(ingredient="onion", avoid_teammate=avoid_teammate, seed=seed)
        if pot_strategy not in {"fixed_a", "fixed_b", "nearest", "conservative", "two_pot"}:
            raise ValueError(f"Unsupported Scenario 4 pot strategy: {pot_strategy}")
        self.pot_strategy = pot_strategy
        self.blocked_threshold = max(1, int(blocked_threshold))
        self._blocked_steps = 0
        self._alternate = False

    def reset(self) -> None:
        super().reset()
        self._blocked_steps = 0
        self._alternate = False

    def action(self, state):
        target = self._choose_target(state)
        action, info = super().action(state)
        blocked = target is not None and action == Action.STAY
        self._blocked_steps = self._blocked_steps + 1 if blocked else 0
        if self._blocked_steps >= self.blocked_threshold:
            self._alternate = not self._alternate
            self._blocked_steps = 0
        info.update(
            {
                "policy_name": f"scenario4_{self.pot_strategy}",
                "scenario4_blocked": blocked,
                "scenario4_recovery": self._alternate,
            }
        )
        return action, info

    def _assigned_pots(self, state, pot_states) -> list[tuple[int, int]]:
        pots = sorted(self.mdp.get_pot_locations())
        if not pots:
            return []
        if self.pot_strategy == "fixed_a":
            return pots[:1]
        if self.pot_strategy == "fixed_b":
            return pots[-1:]
        if self.pot_strategy == "conservative":
            active = [p for p in pots if p not in pot_states.get("empty", [])]
            return active[:1] or pots[:1]
        if self.pot_strategy == "two_pot":
            return pots
        # nearest switches its preferred pot after a blocked-target recovery.
        return list(reversed(pots)) if self._alternate else pots

    def _choose_target(self, state) -> tuple[int, int] | None:
        player = state.players[self.agent_index]
        held = player.held_object
        pot_states = self.mdp.get_pot_states(state)
        assigned = set(self._assigned_pots(state, pot_states))

        def pick(candidates: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
            candidates = [p for p in candidates if p in assigned]
            return self._nearest(player.position, candidates)

        ready = pick(pot_states.get("ready", []))
        accepting = []
        for count in range(Recipe.MAX_NUM_INGREDIENTS):
            accepting.extend(pot_states.get("empty" if count == 0 else f"{count}_items", []))
        accepting = [p for p in accepting if p in assigned]

        if held is not None:
            if held.name == "soup":
                return self._nearest(player.position, self.mdp.get_serving_locations())
            if held.name == "dish":
                return ready
            if held.name == "onion":
                return self._nearest(player.position, accepting)
            return None

        if ready is not None:
            dishes = self._counter_objects_by_name(state, "dish") or list(
                self.mdp.get_dish_dispenser_locations()
            )
            return self._nearest(player.position, dishes)
        if accepting:
            onions = self._counter_objects_by_name(state, "onion") or list(
                self.mdp.get_onion_dispenser_locations()
            )
            return self._nearest(player.position, onions)
        return None
