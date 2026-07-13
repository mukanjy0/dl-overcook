from __future__ import annotations

from types import SimpleNamespace

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction

from src.policy_wrappers import (
    EpsilonActionWrapper,
    StickyActionWrapper,
    wrap_agent,
)


class AlternatingAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    def reset(self) -> None:
        super().reset()
        self.index = 0

    def action(self, state):
        action = (Direction.NORTH, Direction.SOUTH)[self.index % 2]
        self.index += 1
        return action, {"proposed_action": action}


def test_sticky_wrapper_repeats_previous_executed_action() -> None:
    wrapper = StickyActionWrapper(AlternatingAgent(), sticky_action_prob=1.0, seed=7)
    wrapper.reset()
    first, first_info = wrapper.action(None)
    second, second_info = wrapper.action(None)
    assert first == Direction.NORTH
    assert first_info["sticky_action_repeated"] is False
    assert second == Direction.NORTH
    assert second_info["proposed_action"] == Direction.SOUTH
    assert second_info["sticky_action_repeated"] is True


def test_random_then_sticky_wrapper_order_tracks_final_action() -> None:
    wrapped = wrap_agent(
        AlternatingAgent(),
        {
            "max_action_time_ms": 0,
            "random_action_prob": 0.1,
            "sticky_action_prob": 0.2,
        },
        seed=11,
    )
    assert isinstance(wrapped, StickyActionWrapper)
    assert isinstance(wrapped.base_agent, EpsilonActionWrapper)
