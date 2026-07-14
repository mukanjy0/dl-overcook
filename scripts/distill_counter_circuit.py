"""Quickly repair the Counter Circuit PPO policy by dataset aggregation.

The existing feed-forward actor-critic is retained, but its actor is distilled
from the validated mixed-recipe specialist on states visited by both teacher
and learner.  Checkpoint selection uses only positive sparse reward, never the
recipe-agnostic soup-delivery event ledger.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PROJECT_ROOT / "final"

# Import project modules before making the self-contained final bundle's
# ``policies`` package importable. This keeps training utilities sourced from
# the main tree while using the exact deployed teacher implementation.
from src.constants import (  # noqa: E402
    action_index_to_overcooked_action,
    overcooked_action_to_index,
)
from src.environment import build_env  # noqa: E402
from src.models.actor_critic import ActorCritic, ActorCriticConfig  # noqa: E402
from src.models.interfaces import ObservationSpec  # noqa: E402
from src.observations import ObservationBuilder  # noqa: E402
sys.path.insert(0, str(FINAL_ROOT))
from policies.basic_policies import GreedyFullTaskPolicy  # noqa: E402
from policies.template import _CounterCircuitMixedRecipe  # noqa: E402

_wrapper_spec = importlib.util.spec_from_file_location(
    "final_policy_wrappers", FINAL_ROOT / "src" / "policy_wrappers.py"
)
if _wrapper_spec is None or _wrapper_spec.loader is None:
    raise RuntimeError("Could not load final/src/policy_wrappers.py")
_wrapper_module = importlib.util.module_from_spec(_wrapper_spec)
sys.modules[_wrapper_spec.name] = _wrapper_module
_wrapper_spec.loader.exec_module(_wrapper_module)
EpsilonActionWrapper = _wrapper_module.EpsilonActionWrapper


OFFICIAL_SEEDS = (67, 607, 6007, 60007)
ENVIRONMENT = {
    "layout_name": "counter_circuit",
    "layout_file": None,
    "horizon": 400,
    "old_dynamics": True,
}
PARTNER_RANDOM_ACTION_PROB = 0.05
PARTNER_STICKY_ACTION_PROB = 0.15
TEACHER_CLASS = _CounterCircuitMixedRecipe


def _load_actor(path: Path) -> tuple[dict, ActorCritic, ObservationSpec]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_spec = payload["model"]
    observation_spec = ObservationSpec.from_dict(payload["observation"])
    model = ActorCritic(
        input_size=int(model_spec["input_size"]),
        num_actions=int(model_spec["num_actions"]),
        config=ActorCriticConfig.from_dict(model_spec["parameters"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    return payload, model, observation_spec


def _new_partner(env, seed: int) -> EpsilonActionWrapper:
    partner = EpsilonActionWrapper(
        GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True, seed=seed),
        random_action_prob=PARTNER_RANDOM_ACTION_PROB,
        sticky_action_prob=PARTNER_STICKY_ACTION_PROB,
        seed=seed,
    )
    partner.reset()
    partner.set_agent_index(1)
    partner.set_mdp(env.mdp)
    return partner


def _teacher(env) -> GreedyFullTaskPolicy:
    teacher = TEACHER_CLASS()
    teacher.reset()
    teacher.set_agent_index(0)
    teacher.set_mdp(env.mdp)
    return teacher


def _encode(builder, spec: ObservationSpec, state) -> np.ndarray:
    return spec.encode(builder(state, 0))


def _model_action(model: ActorCritic, encoded: np.ndarray) -> int:
    observation = torch.as_tensor(encoded, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        return int(model.act_batch(observation, deterministic=True).actions.item())


def collect_episode(
    model: ActorCritic,
    spec: ObservationSpec,
    *,
    seed: int,
    teacher_probability: float,
) -> tuple[list[np.ndarray], list[int]]:
    """Collect learner-visited observations labelled by the safe teacher."""
    env = build_env(ENVIRONMENT)
    env.reset(regen_mdp=False)
    builder = ObservationBuilder(
        env, {"type": "featurized", "include_agent_index": True}
    )
    teacher = _teacher(env)
    partner = _new_partner(env, seed + 2000)
    rng = np.random.default_rng(seed + 3000)
    observations: list[np.ndarray] = []
    labels: list[int] = []
    done = False
    while not done:
        state = env.state
        encoded = _encode(builder, spec, state)
        teacher_action, _ = teacher.action(state)
        teacher_index = overcooked_action_to_index(teacher_action)
        observations.append(encoded)
        labels.append(teacher_index)
        ego_index = (
            teacher_index
            if rng.random() < teacher_probability
            else _model_action(model, encoded)
        )
        partner_action, partner_info = partner.action(state)
        _, _, done, _ = env.step(
            (action_index_to_overcooked_action(ego_index), partner_action),
            ({"policy_name": "distilled_ego"}, partner_info),
        )
    return observations, labels


def fit_actor(
    model: ActorCritic,
    observations: list[np.ndarray],
    labels: list[int],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> float:
    """Fit the shared network and actor head; the value head is intentionally idle."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.long)
    optimizer = torch.optim.Adam(
        [*model.backbone.parameters(), *model.actor.parameters()],
        lr=learning_rate,
    )
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(targets), 512):
            indices = permutation[start : start + 512]
            logits, _ = model(features[indices])
            loss = nn.functional.cross_entropy(logits, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.item())
    model.eval()
    return final_loss


def evaluate(model: ActorCritic, spec: ObservationSpec) -> dict[str, object]:
    """Evaluate with the teacher's disclosed noise and positive reward only."""
    rewards: list[float] = []
    soups: list[int] = []
    for seed in OFFICIAL_SEEDS:
        env = build_env(ENVIRONMENT)
        env.reset(regen_mdp=False)
        builder = ObservationBuilder(
            env, {"type": "featurized", "include_agent_index": True}
        )
        partner = _new_partner(env, seed + 2000)
        done = False
        sparse_return = 0.0
        while not done:
            state = env.state
            ego_index = _model_action(model, _encode(builder, spec, state))
            partner_action, partner_info = partner.action(state)
            _, reward, done, _ = env.step(
                (action_index_to_overcooked_action(ego_index), partner_action),
                ({"policy_name": "distilled_ego"}, partner_info),
            )
            sparse_return += float(reward)
        rewards.append(sparse_return)
        soups.append(int(round(sparse_return / 20.0)))
    return {
        "seeds": list(OFFICIAL_SEEDS),
        "soups": soups,
        "minimum_soups": min(soups),
        "mean_soups": float(np.mean(soups)),
        "mean_sparse_reward": float(np.mean(rewards)),
    }


def save_inference(payload: dict, model: ActorCritic, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(payload)
    result["model_state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    result["environment"] = dict(result.get("environment", {}))
    result["environment"]["layout_name"] = "counter_circuit"
    torch.save(result, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=FINAL_ROOT / "policies" / "stage_d_counter_circuit.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "counter_circuit_distilled"
        / "inference.pt",
    )
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--episodes-per-iteration", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    payload, model, spec = _load_actor(args.source)
    observations: list[np.ndarray] = []
    labels: list[int] = []
    best_model = copy.deepcopy(model.state_dict())
    best_metrics = evaluate(model, spec)
    history: list[dict[str, object]] = [{"iteration": 0, **best_metrics}]
    print(json.dumps(history[-1], sort_keys=True), flush=True)

    for iteration in range(1, args.iterations + 1):
        # Start mostly on-policy with the teacher, then expose the teacher to
        # learner-induced states so mistakes are explicitly corrected.
        teacher_probability = max(0.05, 0.65 * (0.45 ** (iteration - 1)))
        for episode in range(args.episodes_per_iteration):
            episode_seed = args.seed + iteration * 10_000 + episode
            episode_observations, episode_labels = collect_episode(
                model,
                spec,
                seed=episode_seed,
                teacher_probability=teacher_probability,
            )
            observations.extend(episode_observations)
            labels.extend(episode_labels)
        loss = fit_actor(
            model,
            observations,
            labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed + iteration,
        )
        metrics = evaluate(model, spec)
        record = {
            "iteration": iteration,
            "examples": len(labels),
            "teacher_probability": teacher_probability,
            "loss": loss,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        candidate_key = (int(metrics["minimum_soups"]), float(metrics["mean_soups"]))
        best_key = (
            int(best_metrics["minimum_soups"]),
            float(best_metrics["mean_soups"]),
        )
        if candidate_key > best_key:
            best_metrics = metrics
            best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    save_inference(payload, model, args.output)
    summary = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "selection": "minimum positive-reward soups, then mean positive-reward soups",
        "best": best_metrics,
        "history": history,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
