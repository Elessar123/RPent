# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from robots.libero.flash.replay import (
    execute,
    load,
    pick_succeeded,
    plans,
    replay,
)
from robots.libero.robot_spec import FLASH_SUITES, _add_cli_args, _parse_config


class _Toolkit:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result or {}
        self.state = SimpleNamespace(latest_step=0)
        self.calls = []

    def execute_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return SimpleNamespace(result=self.result)

    def solved(self) -> bool:
        return False


def test_execute_rejects_tool_error() -> None:
    with pytest.raises(RuntimeError, match="move_to failed: unreachable"):
        execute(_Toolkit({"error": "unreachable"}), "move_to", {})


def test_pick_succeeded_uses_pi0_pick_success_contract() -> None:
    wrapped = {"log": {"result": {"success": True, "peak_lift_m": 0.05}}}

    assert pick_succeeded(wrapped) is True
    assert (
        pick_succeeded(
            {
                "log": {
                    "result": {
                        "success": False,
                        "peak_lift_m": 0.10,
                        "final_gripper_opening": 0.01,
                    }
                }
            }
        )
        is False
    )


def test_replay_reuses_toolkit_opening_observation() -> None:
    toolkit = _Toolkit()
    toolkit.state.latest_step = 7

    result = replay(
        toolkit,
        molmo=SimpleNamespace(),
        program={"plan": [], "reference": {}, "locator_of": {}},
    )

    assert result == {"done": False, "anchors": 0, "plan": 0}


def test_replay_passes_legacy_pick_thresholds_to_pi0_pick() -> None:
    toolkit = _Toolkit({"success": True})

    replay(
        toolkit,
        molmo=SimpleNamespace(),
        program={
            "plan": [
                {
                    "action": "pi0_pick",
                    "arguments": {
                        "prompt": "pick up the bowl",
                        "lift_thresh": 0.08,
                    },
                }
            ],
            "reference": {},
            "locator_of": {},
        },
    )

    assert toolkit.calls == [
        (
            "pi0_pick",
            {
                "prompt": "pick up the bowl",
                "lift_thresh": 0.04,
                "gripper_closed_thresh": 0.07,
                "gripper_open_thresh": 0.003,
                "descent_thresh": 0.0,
            },
        )
    ]


def test_pick_retry_reuses_relocated_move_arguments() -> None:
    class RetryToolkit(_Toolkit):
        def execute_tool(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            if name == "segment":
                result = {"world_xyz": [0.2, 0.1, 0.0]}
            elif name == "pi0_pick":
                result = {"success": False}
            else:
                result = {}
            return SimpleNamespace(result=result)

    toolkit = RetryToolkit()
    replay(
        toolkit,
        molmo=SimpleNamespace(),
        program={
            "plan": [
                {
                    "action": "move_to",
                    "arguments": {"xyz": [0.0, 0.0, 0.7], "gripper": -1},
                    "anchor": "bowl",
                    "anchor_distance": 0.0,
                    "offset": [0.01, -0.02],
                },
                {"action": "pi0_pick", "arguments": {"prompt": "pick up the bowl"}},
            ],
            "reference": {"bowl": [0.0, 0.0]},
            "locator_of": {"bowl": "segment"},
        },
    )

    retried_moves = [args for name, args in toolkit.calls if name == "move_to"]
    assert len(retried_moves) == 3
    assert all(args["xyz"] == [0.21, 0.08, 0.7] for args in retried_moves)


def test_replay_stops_when_attached_anchor_is_not_located() -> None:
    toolkit = _Toolkit()
    notes = []

    result = replay(
        toolkit,
        molmo=SimpleNamespace(),
        program={
            "plan": [
                {
                    "action": "move_to",
                    "arguments": {"xyz": [0.3, 0.2, 0.7], "gripper": -1},
                    "anchor": "bowl",
                    "anchor_distance": 0.0,
                    "offset": [0.01, -0.02],
                },
                {"action": "release", "arguments": {}},
            ],
            "reference": {"bowl": [0.0, 0.0]},
            "locator_of": {"bowl": "segment"},
        },
        note=notes.append,
    )

    assert result["done"] is False
    assert [name for name, _ in toolkit.calls] == ["segment"]
    assert any("unavailable; stopping replay" in note for note in notes)


def test_replay_propagates_toolkit_exceptions() -> None:
    class FailingToolkit(_Toolkit):
        def execute_tool(self, name: str, arguments: dict):
            raise ConnectionError("RPC disconnected")

    with pytest.raises(ConnectionError, match="RPC disconnected"):
        replay(
            FailingToolkit(),
            molmo=SimpleNamespace(),
            program={
                "plan": [
                    {
                        "action": "move_to",
                        "arguments": {"xyz": [0.1, 0.1, 0.7], "gripper": -1},
                    }
                ],
                "reference": {},
                "locator_of": {},
            },
        )


def test_flash_supports_all_libero_pro_task_and_swap_suites() -> None:
    assert FLASH_SUITES == {
        f"libero_{family}_{regime}"
        for family in ("spatial", "object", "goal", "10")
        for regime in ("task", "swap")
    }


def test_non_flash_planner_rejects_molmo_endpoint() -> None:
    args = SimpleNamespace(
        suite="libero_object_swap",
        task=0,
        planner="api",
        molmo_endpoint="http://127.0.0.1:8115",
    )

    with pytest.raises(ValueError, match="requires --planner flash"):
        _parse_config(args)


def test_missing_plans_explains_where_to_download(monkeypatch, tmp_path) -> None:
    sync_calls = []
    monkeypatch.setattr(
        "rpent.memory.MemoryManager.sync",
        lambda *args, **kwargs: sync_calls.append(kwargs),
    )

    with pytest.raises(FileNotFoundError, match="RLinf/RPent-memory") as error:
        plans(tmp_path / "flash")

    assert str(tmp_path / "flash") in str(error.value)
    assert sync_calls == []


def test_plans_do_not_require_an_index(tmp_path) -> None:
    root = tmp_path / "flash"
    root.mkdir(parents=True)
    (root / "object_swap_t0_plan.json").write_text('{"plan": []}')

    assert plans(root) == root


def test_load_reads_only_runtime_plan_fields(tmp_path) -> None:
    (tmp_path / "object_swap_t0_plan.json").write_text('{"plan": []}')
    (tmp_path / "object_swap_t0_anchors.json").write_text(
        '{"anchors": [{"phrase": "bowl", "locator": "segment", '
        '"median_xy": [0.1, 0.2]}]}'
    )

    program = load(tmp_path, "object_swap_t0")

    assert program["plan"] == []
    assert program["reference"]["bowl"].tolist() == [0.1, 0.2]
    assert program["locator_of"] == {"bowl": "segment"}


def _local_flash_args(root):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--planner", default="flash")
    parser.add_argument("--memory-profile", default="local")
    parser.add_argument("--memory-dir", default=str(root))
    _add_cli_args(parser, use_dashboard=False)
    return parser.parse_args(
        [
            "--suite",
            "libero_spatial_task",
            "--task",
            "0",
            "--molmo-endpoint",
            "http://localhost:20703",
        ]
    )


def test_flash_uses_selected_memory_without_downloading(monkeypatch, tmp_path):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.planner.base import build_planner

    root = tmp_path / "custom-memory"
    flash = root / "flash"
    flash.mkdir(parents=True)
    (flash / "spatial_task_t0_plan.json").write_text(
        json.dumps(
            {"plan": [{"action": "move_to", "arguments": {"xyz": [0.12, 0.23, 0.7]}}]}
        )
    )
    (flash / "spatial_task_t0_anchors.json").write_text('{"anchors": []}')
    config = _parse_config(_local_flash_args(root))
    assert config.prompt_vars["memory_dir"] == str(root)

    def reject_sync(*args, **kwargs):
        pytest.fail("local Flash Mode must not download memory")

    monkeypatch.setattr(MemoryManager, "sync", reject_sync)
    toolkit = _Toolkit()
    toolkit.memory = MemoryManager(config.prompt_vars["memory_dir"])
    toolkit.primitives = SimpleNamespace(molmo_client=object())
    planner = build_planner(
        "flash",
        robot_name="libero",
        recipe_tag="spatial_task_t0_s9",
        output_dir=tmp_path,
        dashboard_events=NullDashboardEventSink(),
    )
    result = planner.solve(
        system_prompt="", user_message="", toolkit=toolkit, max_turns=0
    )
    assert result.error is None
    assert "spatial/task_t0" in result.finish_result["summary"]
    assert result.stats["total_input_tokens"] == 0
    assert toolkit.calls == [("move_to", {"xyz": [0.12, 0.23, 0.7]})]


@pytest.mark.parametrize("missing", ["plan", "anchors"])
def test_local_flash_rejects_incomplete_plan(tmp_path, missing):
    root = tmp_path / "flash"
    root.mkdir()
    present = "anchors" if missing == "plan" else "plan"
    (root / f"spatial_task_t0_{present}.json").write_text("{}")
    with pytest.raises(ValueError, match="no complete Flash plan"):
        _parse_config(_local_flash_args(tmp_path))


def test_flash_config_rejects_exploration(tmp_path):
    args = _local_flash_args(tmp_path)
    args.explore = True
    with pytest.raises(ValueError, match="evaluation-only"):
        _parse_config(args)
