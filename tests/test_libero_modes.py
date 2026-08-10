from __future__ import annotations

import json

from robots.libero.toolkit import LiberoToolkit
from rpent.cli.main import _build_argparser
from rpent.envs.base import get_env_spec


def _parse(*extra: str):
    parser = _build_argparser()
    spec = get_env_spec("libero")
    spec.add_cli_args(parser, use_dashboard=False)
    return spec, parser.parse_args(
        ["--env", "libero", "--suite", "libero_10_task", "--task", "0", *extra]
    )


def test_legacy_eval_defaults_to_hf_single_session():
    spec, args = _parse("--planner", "claude_code")
    config = spec.parse_config(args)

    assert args.explore is False
    assert args.memory_profile == "hf"
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["memory_profile"] == "hf"


def test_layered_eval_uses_same_entrypoint_and_planner():
    spec, args = _parse(
        "--planner", "codex", "--memory-profile", "layered", "--seed", "3"
    )
    config = spec.parse_config(args)

    assert args.planner == "codex"
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["memory_profile"] == "layered"
    assert config.prompt_vars["reference_tag"] == "10_task_t0_s0"


def test_prompt_profiles_are_isolated():
    spec, hf_args = _parse("--memory-profile", "hf")
    hf_config = spec.parse_config(hf_args)
    hf_vars = {**hf_config.prompt_vars, "output_dir": hf_config.output_dir}
    hf_prompt = spec.prompts.render("system", variables=hf_vars)

    _, layered_args = _parse("--memory-profile", "layered")
    layered_config = spec.parse_config(layered_args)
    layered_prompt = spec.prompts.render(
        "system",
        variables={**layered_config.prompt_vars, "output_dir": layered_config.output_dir},
    )

    _, explore_args = _parse("--explore")
    explore_config = spec.parse_config(explore_args)
    explore_prompt = spec.prompts.render(
        "system",
        variables={**explore_config.prompt_vars, "output_dir": explore_config.output_dir},
    )

    assert "LOCAL SUITE + TASK + GLOBAL" not in hf_prompt
    assert "MULTI-ATTEMPT EXPLORE MODE" not in hf_prompt
    assert "LOCAL SUITE + TASK + GLOBAL" in layered_prompt
    assert "MULTI-ATTEMPT EXPLORE MODE" in explore_prompt


def test_explore_uses_same_api_planner_and_enables_auto_merge(tmp_path):
    spec, args = _parse(
        "--planner",
        "api",
        "--model",
        "anthropic:test-model",
        "--explore",
        "--memory-dir",
        str(tmp_path / "memory"),
    )
    config = spec.parse_config(args)

    assert args.planner == "api"
    assert args.auto_merge_memory is True
    assert config.prompt_vars["mode"] == "explore"
    assert config.prompt_vars["memory_profile"] == "layered"
    assert config.prompt_vars["memory_dir"] == str((tmp_path / "memory").resolve())
    assert spec.finalize_run is not None


def test_explore_finalizer_automatically_merges_task_pair(tmp_path):
    memory_dir = tmp_path / "memory"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    spec, args = _parse(
        "--planner",
        "codex",
        "--explore",
        "--memory-dir",
        str(memory_dir),
        "--output-dir",
        str(output_dir),
    )
    config = spec.parse_config(args)
    cell = config.recipe_tag
    (output_dir / f"{cell}.json").write_text(
        json.dumps({"libero_terminated": True})
    )
    (output_dir / f"recipe_{cell}.jsonl").write_text('{"action":"move_to"}\n')

    result = spec.finalize_run(args, config)

    assert result is not None and result["task"] == 1
    assert (memory_dir / "task" / f"{cell}.json").exists()


def test_explore_finish_requires_attempt_budget():
    toolkit = LiberoToolkit.__new__(LiberoToolkit)
    toolkit._attempts_per_session = 5
    toolkit._session_attempt = 3
    toolkit._solved = False

    result = toolkit._guarded_finish(lambda **kwargs: {"_finish": True})

    assert result["error"] == "finish refused"
    toolkit._solved = True
    assert toolkit._guarded_finish(lambda **kwargs: {"_finish": True}) == {
        "_finish": True
    }


def test_explore_reset_enforces_attempt_budget():
    toolkit = LiberoToolkit.__new__(LiberoToolkit)
    toolkit._attempts_per_session = 2
    toolkit._session_attempt = 1
    toolkit._attempt = 4
    toolkit._step = lambda *args, **kwargs: {"state": {}}

    result = toolkit._reset_episode("retry")

    assert result["attempt"] == 5
    assert toolkit._session_attempt == 2
    assert toolkit._reset_episode("retry")["error"] == "reset refused"
