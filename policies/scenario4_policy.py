"""Small deterministic specialists for the disclosed Scenario 4 layout."""

from __future__ import annotations

from overcooked_ai_py.mdp.actions import Action

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

    def _pots_that_can_accept_ingredients(self, state, pot_states) -> list[tuple[int, int]]:
        """Filter only ingredient assignment; retain the proven base task FSM."""
        candidates = super()._pots_that_can_accept_ingredients(state, pot_states)
        if self.pot_strategy in {"nearest", "two_pot"}:
            return candidates
        assigned = set(self._assigned_pots(state, pot_states))
        return [pot for pot in candidates if pot in assigned]
