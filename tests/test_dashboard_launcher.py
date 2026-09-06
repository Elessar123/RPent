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

from types import SimpleNamespace

from rpent.dashboard.launcher import apply_to_args, defaults_from_args


def test_task_card_dashboard_config_preserves_molmo_endpoint() -> None:
    args = SimpleNamespace(
        planner="task_card",
        model=None,
        cuda_device=None,
        max_turns=20,
        max_episode_steps=10000,
        planner_timeout_s=None,
        reasoning_effort="none",
        claude_code_max_budget_usd=None,
        no_images=False,
        molmo_endpoint="http://127.0.0.1:20703",
    )

    payload = defaults_from_args(args)
    apply_to_args(args, payload)

    assert args.planner == "task_card"
    assert args.molmo_endpoint == "http://127.0.0.1:20703"


def test_non_task_card_dashboard_config_discards_molmo_endpoint() -> None:
    args = SimpleNamespace(
        planner="api",
        model="openai:gpt-5.6-sol",
        cuda_device=None,
        max_turns=20,
        max_episode_steps=10000,
        planner_timeout_s=None,
        reasoning_effort="none",
        claude_code_max_budget_usd=None,
        no_images=False,
        molmo_endpoint=None,
    )
    payload = defaults_from_args(args)
    payload["molmo-endpoint"] = "http://127.0.0.1:20703"

    apply_to_args(args, payload)

    assert args.molmo_endpoint is None
