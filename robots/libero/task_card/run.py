# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay task cards across a family's tasks and seeds, in a batch.

A plan recorded as absolute waypoints cannot follow an object that moved. The
same plan recorded together with what was localized, and how far each waypoint
sat from that reading, can: the offset is task logic and survives a change of
layout, the coordinate is not. A task card is a plan in that second form.

This drives many episodes without the CLI, starting an environment server per
episode and reusing models that are already serving. For a single episode
inside a normal RPent run, use ``--planner task_card`` instead.

Run manually with::

    python -m robots.libero.task_card.run --family object --tasks swap_t3 \
        --seeds 0 1 2 \
        --vla-endpoint http://127.0.0.1:20701 \
        --sam3-endpoint http://127.0.0.1:20702 \
        --molmo-endpoint http://127.0.0.1:20703
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from robots.libero.task_card.replay import CARDS, cards, load, replay
from rpent.robots.components.molmo_client import MolmoClient
from rpent.utils.config import get_repo_root
from rpent.utils.rpc import make_rpc_client


def catalogue(root: Path, family: str) -> dict[str, dict]:
    """The cards recorded for one family, keyed by task."""
    index = json.loads((root / "index.json").read_text())
    return {row["key"]: row for row in index if row["family"] == family}


def run(args: argparse.Namespace, key: str, chosen: dict, seed: int, note) -> dict:
    """Stand up one episode's runtime, replay the card into it, tear it down."""
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.robot_spec import get_toolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.robots.robot_spec import RunConfig
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import wait_for_ready
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    family = args.family
    card = load(args.cards / family / key)

    suite = f"libero_{family}_{chosen['suite']}"
    out = args.output_dir / family / key / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="env_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "env_server.py"),
            "--suite",
            suite,
            "--task",
            str(chosen["task"]),
            "--seed",
            str(seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--transport",
            "http",
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides={
            "LIBERO_TYPE": get_libero_type(),
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
        },
        log_path=str(out / "env_server.log"),
    )
    daemon.start()
    try:
        note(f"      env_server on port {port}")
        rpc = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(rpc, daemon=daemon, timeout_s=300.0)
        env = LiberoEnvClient(
            rpc,
            expected_meta={
                "suite": suite,
                "task": chosen["task"],
                "seed": seed,
                "max_episode_steps": args.max_episode_steps,
            },
        )
        # Built the way the CLI builds it, so the toolkit is configured
        # identically whether a card is replayed here or by --planner task_card.
        toolkit = get_toolkit(
            primitives_kwargs={
                "env": env,
                "model": Pi05VLAClient(make_rpc_client(args.vla_endpoint)),
                "sam3_client": Sam3Client(make_rpc_client(args.sam3_endpoint)),
            },
            dashboard_events=NullDashboardEventSink(),
            config=RunConfig(
                recipe_tag=f"{family}_{key}_s{seed}",
                output_dir=out,
                prompt_vars={},
                task_desc={"suite": suite, "task": chosen["task"], "seed": seed},
            ),
            state_output_dir=out,
        )
        molmo = MolmoClient(make_rpc_client(args.molmo_endpoint))
        outcome = replay(toolkit, molmo, card, note)
        return {**outcome, "card": chosen["source"]}
    finally:
        daemon.stop()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay task cards in a batch")
    parser.add_argument(
        "--family",
        default="object",
        choices=["10", "goal", "object", "spatial"],
        help="Perturbation family to replay.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task keys, e.g. swap_t3 task_t0. Default: all "
        "cards recorded for the family.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=list(range(10)),
        help="Seeds to replay each card against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_repo_root() / "logs" / "replay",
        help="Where per-episode output is written.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument(
        "--cards",
        type=Path,
        default=CARDS,
        help="Card corpus. Defaults to the one under resources/.",
    )
    parser.add_argument(
        "--vla-endpoint",
        default="http://127.0.0.1:8113",
        help="An already-serving Pi0.5 policy.",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default="http://127.0.0.1:8114",
        help="An already-serving SAM3 segmenter.",
    )
    parser.add_argument(
        "--molmo-endpoint",
        default="http://127.0.0.1:8115",
        help="An already-serving Molmo grounder.",
    )
    return parser


def main() -> int:
    """Replay every selected task of a family, one episode at a time."""
    args = _build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cards = cards(args.cards)
    available = catalogue(args.cards, args.family)
    if args.tasks:
        unknown = sorted(set(args.tasks) - set(available))
        if unknown:
            raise SystemExit(f"no card for {unknown} in family {args.family}")
        available = {key: available[key] for key in args.tasks}

    def note(message: str) -> None:
        print(message, flush=True)

    results: dict = {}
    for key, chosen in sorted(available.items()):
        print(
            f"\n=== {args.family}/{key}  card {chosen['source']}"
            f" ({chosen['actions']} actions, {chosen['anchors']} anchors) ===",
            flush=True,
        )
        for seed in args.seeds:
            try:
                outcome = run(args, key, chosen, seed, note)
            except Exception:
                traceback.print_exc()
                outcome = None
            results.setdefault(key, {})[seed] = outcome
            print(f"  seed {seed}: {outcome}", flush=True)
            (args.output_dir / f"summary_{args.family}.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2)
            )
    total = hit = 0
    print("\n=== summary ===")
    for key, seeds_result in sorted(results.items()):
        done = sum(1 for r in seeds_result.values() if r and r.get("done"))
        total += len(seeds_result)
        hit += done
        print(f"  {key:<10}{done}/{len(seeds_result)}")
    if total:
        print(f"  total {hit}/{total} = {hit / total * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
