from __future__ import annotations

from pathlib import Path

import numpy as np

from src.constants import (
    NUM_ACTIONS,
    action_index_to_overcooked_action,
    overcooked_action_to_index,
)
from src.environment import build_env
from src.observations import ObservationBuilder


def test_builtin_environment_preserves_horizon_and_observations() -> None:
    env = build_env(
        {"layout_name": "cramped_room", "horizon": 7, "old_dynamics": True}
    )
    env.reset(regen_mdp=False)
    assert env.horizon == 7

    featurized = ObservationBuilder(
        env, {"type": "featurized", "include_agent_index": True}
    )(env.state, 0)
    lossless = ObservationBuilder(
        env, {"type": "lossless_grid", "include_agent_index": False}
    )(env.state, 1)
    raw_state = ObservationBuilder(
        env, {"type": "state", "include_agent_index": True}
    )(env.state, 0)
    assert isinstance(featurized["obs"], np.ndarray)
    assert featurized["agent_index"] == 0
    assert isinstance(lossless, np.ndarray)
    assert raw_state["state"] is env.state


def test_custom_layout_uses_existing_loader(project_root: Path) -> None:
    layout_path = project_root / "configs" / "layouts" / "custom_room.layout"
    env = build_env(
        {"layout_file": str(layout_path), "horizon": 5, "old_dynamics": True}
    )
    env.reset(regen_mdp=False)
    assert env.mdp.layout_name == "custom_room"
    assert env.horizon == 5


def test_optional_state_source_is_called_on_reset() -> None:
    class TrackingStateSource:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self, mdp, rng):
            del rng
            self.calls += 1
            return mdp.get_standard_start_state()

    source = TrackingStateSource()
    env = build_env(
        {"layout_name": "cramped_room", "horizon": 3, "old_dynamics": True},
        state_source=source,
        rng=np.random.default_rng(9),
    )
    env.reset(regen_mdp=False)
    assert source.calls >= 1


def test_exact_six_action_round_trip() -> None:
    assert NUM_ACTIONS == 6
    for action_index in range(NUM_ACTIONS):
        assert overcooked_action_to_index(
            action_index_to_overcooked_action(action_index)
        ) == action_index
