"""Repair the Scenario 1 actor using the shared short distillation engine."""

from __future__ import annotations

import sys

import distill_counter_circuit as distillation
from policies.template import _AsymmetricReachableFullTask


distillation.ENVIRONMENT = {
    "layout_name": "asymmetric_advantages",
    "layout_file": None,
    "horizon": 400,
    "old_dynamics": True,
}
distillation.PARTNER_RANDOM_ACTION_PROB = 0.0
distillation.PARTNER_STICKY_ACTION_PROB = 0.0
distillation.TEACHER_FACTORY = _AsymmetricReachableFullTask


def _set_default_argument(name: str, value: str) -> None:
    if name not in sys.argv:
        sys.argv.extend((name, value))


if __name__ == "__main__":
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
            / "asymmetric_advantages_distilled"
            / "inference.pt"
        ),
    )
    distillation.main()
