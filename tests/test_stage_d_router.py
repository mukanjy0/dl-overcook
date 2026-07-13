from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action

from src.deployment.stage_d_router import (
    StageDDeploymentRouter,
    StageDRoutingError,
    select_specialist,
)


def _write_mapping(path: Path, *, checkpoint_path: str | None = None) -> Path:
    python_policy = (
        {
            "type": "python_class",
            "path": "../../policies/rl_policy.py",
            "class_name": "StudentAgent",
            "config": {"checkpoint_path": checkpoint_path},
        }
        if checkpoint_path is not None
        else {"type": "builtin", "name": "stay"}
    )
    mapping = {
        "fallback": {"id": "fallback", "policy": {"type": "builtin", "name": "stay"}},
        "layouts": {
            "asymmetric_advantages": {
                "positions": {
                    "0": {"id": "aa0", "policy": python_policy},
                    "1": {"id": "aa1", "policy": {"type": "builtin", "name": "stay"}},
                }
            },
            "coordination_ring": {
                "positions": {
                    "0": {"id": "cr", "policy": {"type": "builtin", "name": "stay"}},
                    "1": {"id": "cr", "policy": {"type": "builtin", "name": "stay"}},
                }
            },
            "counter_circuit": {
                "positions": {
                    "0": {"id": "cc", "policy": {"type": "builtin", "name": "stay"}},
                    "1": {"id": "cc", "policy": {"type": "builtin", "name": "stay"}},
                }
            },
        },
    }
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def test_stage_d_mapping_covers_known_layouts_and_explicit_fallback(project_root: Path) -> None:
    mapping = project_root / "configs/stage_d/specialists.yaml"
    expected = {
        ("asymmetric_advantages", 0): "aa_rl_position0_900096",
        ("asymmetric_advantages", 1): "aa_greedy_position1",
        ("coordination_ring", 0): "cr_p010_s11_step1050624",
        ("coordination_ring", 1): "cr_p010_s11_step1050624",
        ("counter_circuit", 0): "cc_exact_long_seed3_step1902592",
        ("counter_circuit", 1): "cc_exact_long_seed3_step1902592",
        ("unseen_layout", 0): "generic_greedy_fallback",
        ("unseen_layout", 1): "generic_greedy_fallback",
    }

    for route, specialist_id in expected.items():
        selection = select_specialist(mapping, *route)
        assert selection.specialist_id == specialist_id
        assert selection.physical_position == route[1]


class _CountingAgent(Agent):
    def __init__(self):
        self.reset_calls = 0
        super().__init__()

    def reset(self) -> None:
        self.reset_calls += 1
        super().reset()

    def action(self, state):
        return Action.STAY, {"policy_name": "counting"}


def test_stage_d_router_lazily_caches_and_resets_selected_policy(tmp_path: Path) -> None:
    mapping = _write_mapping(tmp_path / "specialists.yaml")
    built: list[_CountingAgent] = []

    def build_policy(_config: dict) -> _CountingAgent:
        agent = _CountingAgent()
        built.append(agent)
        return agent

    router = StageDDeploymentRouter(
        layout_name="asymmetric_advantages",
        mapping_path=mapping,
        policy_builder=build_policy,
    )
    router.set_agent_index(1)
    router.set_mdp(object())

    _, first_info = router.action(None)
    _, second_info = router.action(None)

    assert len(built) == 1
    assert first_info["stage_d_specialist"] == "aa1"
    assert second_info["stage_d_ego_index"] == 1
    resets_before = built[0].reset_calls
    router.reset()
    assert built[0].reset_calls == resets_before + 1


@pytest.mark.parametrize("artifact_exists", [False, True])
def test_stage_d_router_reports_missing_or_corrupt_checkpoint(
    tmp_path: Path,
    artifact_exists: bool,
) -> None:
    checkpoint = tmp_path / "artifact.pt"
    if artifact_exists:
        checkpoint.write_bytes(b"not an inference artifact")
    mapping = _write_mapping(tmp_path / "specialists.yaml", checkpoint_path=str(checkpoint))
    raw = yaml.safe_load(mapping.read_text(encoding="utf-8"))
    raw["layouts"]["asymmetric_advantages"]["positions"]["0"]["artifact_sha256"] = "0" * 64
    mapping.write_text(yaml.safe_dump(raw), encoding="utf-8")
    router = StageDDeploymentRouter(
        layout_name="asymmetric_advantages",
        mapping_path=mapping,
        policy_builder=lambda _config: _CountingAgent(),
    )
    router.set_agent_index(0)
    router.set_mdp(object())

    expected = "missing" if not artifact_exists else "hash mismatch"
    with pytest.raises(StageDRoutingError, match=expected):
        router.action(None)


def test_stage_d_clean_process_smoke_runs_every_mapped_specialist(
    tmp_path: Path,
    project_root: Path,
) -> None:
    required_artifacts = [
        project_root
        / "outputs/stage_a_asymmetric_seed67/selected/inference_step_000900096.pt",
        project_root / "outputs/stage_d_specialists/coordination_ring/inference.pt",
        project_root
        / "outputs/counter_circuit_exact_long_seed3_1m/checkpoint_evaluation/selected/inference.pt",
    ]
    if not all(path.is_file() for path in required_artifacts):
        pytest.skip("Stage D artifacts are generated locally and unavailable")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_stage_d_router.py",
            "--output-dir",
            str(tmp_path / "smoke"),
            "--horizon",
            "3",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "smoke" / "summary.json").read_text())
    assert summary["status"] == "complete", completed.stderr
    assert len(summary["routes"]) == 6
    assert all(route["timeouts"] == 0 for route in summary["routes"])
    assert all(route["invalid_actions"] == [0, 0] for route in summary["routes"])
