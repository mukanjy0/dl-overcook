"""Distill the validated Scenario 4 policy for both physical positions."""

from __future__ import annotations

import sys

import distill_counter_circuit as distillation
from policies.basic_policies import RandomMotionPolicy
from policies.template import _Scenario4FixedPotB


distillation.ENVIRONMENT = {
    "layout_name": None,
    "layout_file": str(
        distillation.PROJECT_ROOT / "configs" / "layouts" / "scenario_4.layout"
    ),
    "horizon": 400,
    "old_dynamics": True,
}
distillation.PARTNER_RANDOM_ACTION_PROB = 0.05
distillation.PARTNER_STICKY_ACTION_PROB = 0.05
distillation.EGO_POSITIONS = (0, 1)
distillation.PARTNER_FACTORY = lambda seed: RandomMotionPolicy(seed=seed)
distillation.TEACHER_FACTORY = lambda: _Scenario4FixedPotB(
    ingredient="onion", avoid_teammate=False
)


def _set_default_argument(name: str, value: str) -> None:
    if name not in sys.argv:
        sys.argv.extend((name, value))


if __name__ == "__main__":
    # The architecture is layout-agnostic; the AA artifact is only a compatible
    # initialization and remains untouched.
    _set_default_argument(
        "--source",
        str(
            distillation.FINAL_ROOT
            / "policies"
            / "stage_d_aa_position0.pt"
        ),
    )
    _set_default_argument(
        "--output",
        str(
            distillation.PROJECT_ROOT
            / "outputs"
            / "scenario4_distilled"
            / "inference.pt"
        ),
    )
    distillation.main()
