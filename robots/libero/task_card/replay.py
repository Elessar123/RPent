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
"""One LIBERO task card: what it holds, and how it is replayed.

The card supplies the actions; perception supplies the coordinates. Each anchor
is re-read through the interface it was first read through -- a `segment` anchor
is re-segmented, because its offsets are relative to a mask centroid, and a wide
container seen at an angle has that centroid some way from where a pointing
model points. Only the pixels the recording planner chose by eye go to the
grounder.

Localization is two-stage, as the recording planner's was: a coarse survey of
the opening frame, then the arm parks over each hand-picked anchor and asks
again from the wrist, where the object fills the view. The close reading is kept
only when it agrees with the coarse one, so a wrist view that found something
else cannot overwrite a correct answer.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robots.libero import tools as libero_tools
from robots.libero.task_card.prompts import build as prompt_for
from rpent.robots.components.molmo_client import MolmoClient
from rpent.session.base import EnvState
from rpent.utils.config import get_resources_dir
from rpent.utils.rpc import make_rpc_client

#: ``<family>_<suite>_t<task>_s<seed>``, the tag the CLI builds per cell.
_CELL = re.compile(r"^(10|goal|object|spatial)_(task|swap)_t(\d+)_s(\d+)$")

#: The card corpus, under ``resources/libero/`` beside the curated memory and
#: the per-family reference results the prompts already read from there.
CARDS = get_resources_dir("libero") / "task_card"

#: A close reading further than this from the coarse one has found something
#: else, so the coarse one stands.
REFINE_ACCEPT = 0.05
#: Beyond this a waypoint was not written relative to any located object.
MAX_ATTACH = 0.20
#: The reachable workspace. A reading outside it is not a position.
REACH = 0.45
#: How far a held object can plausibly sit from the gripper holding it.
MAX_HELD = 0.06
#: A wrist reading of a held object lands short of its centre, further the
#: taller it is. Fitted against the simulator's own poses: 35 mm of error
#: becomes 2.5 mm.
PARALLAX = {"x": (0.0231, 0.0610), "y": (-0.0029, 0.2056)}
#: How many times a pick that did not take hold is retried by replaying its
#: approach. No reset is involved; the episode continues.
PICK_ATTEMPTS = 3


def profile(
    state: EnvState,
    step: int,
    camera: str,
    col: float,
    row: float,
    spread: int = 45,
    samples: int = 9,
):
    """A phrase's pixel, and what the depth map says along a line through it.

    One reading names a point on a surface; several down the object's own
    height say where its top is and whether the line stayed on one surface at
    all. Readings that fall far below the highest have left the object for the
    table behind it and are dropped.
    """
    points = []
    for offset in np.linspace(-spread, spread, samples):
        found = libero_tools.back_project(
            row=int(round(row + offset)),
            col=int(round(col)),
            step=step,
            camera=camera,
            resolution="high",
            state=state,
        )
        world = found.get("world_xyz") if isinstance(found, dict) else None
        if isinstance(world, list) and len(world) >= 3 and all(np.isfinite(world[:3])):
            points.append([float(v) for v in world[:3]])
    if not points:
        return None
    array = np.array(points)
    keep = array[array[:, 2] > array[:, 2].max() - 0.03]
    span = float(keep[:, 2].ptp())
    centre = np.median(keep[:, :2], 0)
    return {
        "xy": centre,
        "z_top": float(keep[:, 2].max()),
        "z_span": span,
        "corrected": centre
        - np.array(
            [
                PARALLAX["x"][0] + PARALLAX["x"][1] * span,
                PARALLAX["y"][0] + PARALLAX["y"][1] * span,
            ]
        ),
    }


def locate(molmo: MolmoClient, state: EnvState, step: int, camera: str, query: str):
    image = _image_bytes(state, step, camera)
    if image is None:
        return None
    found = molmo.ground(image, query)
    if not found.found:
        return None
    col, row = found.point_xy
    return profile(state, step, camera, col, row)


def held_body(molmo: MolmoClient, state: EnvState, step: int, query: str):
    """What is in the gripper, sampled on a grid because a line may miss it."""
    image = _image_bytes(state, step, "wrist")
    if image is None:
        return None
    found = molmo.ground(image, query)
    if not found.found:
        return None
    col, row = found.point_xy
    points = []
    for dc in (-40, 0, 40):
        for dr in (-40, 0, 40):
            got = libero_tools.back_project(
                row=int(round(row + dr)),
                col=int(round(col + dc)),
                step=step,
                camera="wrist",
                resolution="high",
                state=state,
            )
            world = got.get("world_xyz") if isinstance(got, dict) else None
            if (
                isinstance(world, list)
                and len(world) >= 3
                and all(np.isfinite(world[:3]))
            ):
                points.append([float(v) for v in world[:3]])
    if len(points) < 4:
        return None
    array = np.array(points)
    centre = np.median(array[:, :2], 0)
    near = array[np.linalg.norm(array[:, :2] - centre, axis=1) < 0.05]
    if len(near) < 3:
        return None
    span = float(array[:, 2].ptp())
    return {
        "xy": np.median(near[:, :2], 0)
        - np.array(
            [
                PARALLAX["x"][0] + PARALLAX["x"][1] * span,
                PARALLAX["y"][0] + PARALLAX["y"][1] * span,
            ]
        )
    }


def _image_bytes(state: EnvState, step: int, camera: str):
    name = "agentview_high.png" if camera == "agentview" else "wrist_high.png"
    try:
        return state.load_bytes(name, step=step)
    except Exception:
        return None


def action_result(raw):
    if not isinstance(raw, dict):
        return {}
    log = raw.get("log")
    if isinstance(log, dict) and isinstance(log.get("result"), dict):
        return log["result"]
    return raw


def pick_confirmed(raw) -> bool:
    """Did the pick take hold?

    The policy's own flag is a heuristic over its rollout, so a closed jaw that
    lifted counts too. Either is enough; neither is a failed pick, and that is
    what the guards key off.
    """
    result = action_result(raw)
    if str(result.get("success", "")).strip().lower() in {"true", "1", "yes"}:
        return True
    if (
        str((result.get("grasp_continuity") or {}).get("state", ""))
        == "GRASP_CONFIRMED"
    ):
        return True
    opening = result.get("final_gripper_opening")
    if opening is None:
        qpos = (result.get("state") or {}).get("robot0_gripper_qpos") or []
        if len(qpos) >= 2:
            opening = abs(float(qpos[0])) + abs(float(qpos[1]))
    lift = float(result.get("peak_lift_m", 0.0) or 0.0)
    return bool(opening is not None and 0.003 <= float(opening) < 0.07 and lift >= 0.04)


def cards(root: Path | None = None) -> Path:
    """The task-card corpus, synced the way every other resource is.

    Cards live under ``resources/libero/`` beside the curated memory, so they
    arrive either from the HuggingFace dataset or from a copy already on disk.
    A run that has neither is told which, rather than failing later on a
    missing index.
    """
    root = root or CARDS
    if not (root / "index.json").is_file():
        from rpent.robots.base import get_robot_spec
        from rpent.utils.resources import ensure_resources

        ensure_resources(get_robot_spec("libero"))
    if not (root / "index.json").is_file():
        raise SystemExit(
            f"no task cards under {root}; sync them with the rest of the "
            "LIBERO resources, or point --cards at a corpus."
        )
    return root


def load(folder: Path) -> dict:
    """Read one card: its plan, and the anchors its coordinates were written against."""
    plan = json.loads((folder / "plan.json").read_text())["plan"]
    anchors = json.loads((folder / "anchors.json").read_text())["anchors"]
    return {
        "plan": plan,
        "reference": {a["phrase"]: np.array(a["median_xy"]) for a in anchors},
        "source_of": {a["phrase"]: a["source"] for a in anchors},
    }


def replay(
    toolkit: Any,
    molmo: MolmoClient,
    card: dict,
    note: Callable[[str], None] = lambda _: None,
) -> dict:
    """Run one card against the scene the toolkit is holding open.

    The episode is single-attempt: a pick that does not take hold is retried in
    place by replaying its approach, and its carry is skipped if it still does
    not, but the environment is never reset back to a clean state.
    """
    primitives = toolkit._primitives
    env = primitives.env
    plan = card["plan"]
    reference = card["reference"]
    source_of = card["source_of"]

    state = toolkit.state

    step = [0]

    def look() -> int:
        libero_tools.dump_state(primitives, state)
        step[0] = state.latest_step
        return step[0]

    def finished() -> bool:
        return bool(toolkit.solved())

    # Record the scene this seed laid out, not the one the env was built
    # with: dumping before the reset gives every seed the same picture.
    primitives.set_obs(env.reset()[0])
    opening = look()

    live: dict[str, np.ndarray] = {}
    for phrase in reference:
        if source_of.get(phrase) == "segment":
            found = primitives.segment(
                prompt=phrase,
                camera="agentview",
                step=opening,
                min_score=0.2,
                state=state,
            )
            world = found.get("world_xyz") if isinstance(found, dict) else None
            xy = (
                np.array(world[:2], dtype=float)
                if isinstance(world, list) and len(world) >= 2
                else None
            )
        else:
            got = locate(
                molmo, state, opening, "agentview", prompt_for("survey", phrase)
            )
            xy = got["xy"] if got else None
        if xy is None or max(abs(xy[0]), abs(xy[1])) > REACH:
            note(f"      {phrase[:26]!r} not located, or out of reach")
            continue
        if any(np.linalg.norm(a - xy) < 0.03 for a in live.values()):
            continue
        live[phrase] = xy
    note(
        "    survey "
        + "  ".join(f"{p[:18]}=({a[0]:+.3f},{a[1]:+.3f})" for p, a in live.items())
    )

    # Park over each hand-picked anchor and read it again from close range,
    # keeping the second reading only when it agrees with the first.
    hover = max(
        (
            s["arguments"]["xyz"][2]
            for s in plan
            if s["action"] in {"move_to", "move_pose"} and s["arguments"].get("xyz")
        ),
        default=0.72,
    )
    for phrase in list(live):
        if source_of.get(phrase) == "segment":
            continue
        coarse = live[phrase]
        try:
            primitives.move_to(
                xyz=[
                    round(float(coarse[0]), 4),
                    round(float(coarse[1]), 4),
                    float(hover),
                ],
                gripper=-1,
                step_clip=0.02,
                max_steps=150,
                tol=0.012,
            )
        except Exception:
            continue
        close = locate(molmo, state, look(), "wrist", prompt_for("refine", phrase))
        if close is None:
            continue
        gap = float(np.linalg.norm(close["xy"] - coarse))
        if gap > REFINE_ACCEPT:
            note(f"      {phrase[:20]!r} close reading {gap:.3f} away, rejected")
            continue
        note(f"      {phrase[:20]!r} refined by {gap:.3f}")
        live[phrase] = close["xy"]

    offset = np.zeros(2)
    held_phrase = "object"
    recent: list[tuple[str, dict]] = []
    skip_suffix = False
    for entry in plan:
        if finished():
            break
        name = entry["action"]
        arguments = dict(entry["arguments"])
        if skip_suffix:
            # A pick that never took hold must not fall through into its
            # carry: an empty gripper crossing the workspace knocks over
            # whatever an earlier transaction put down.
            if name == "release":
                skip_suffix = False
                recent.clear()
                continue
            if name != "pi0_pick":
                continue
            skip_suffix = False
        try:
            if name in {"move_to", "move_pose"}:
                xyz = arguments.get("xyz") or []
                if len(xyz) != 3:
                    continue
                target = np.array(xyz[:2], dtype=float)
                phrase = entry.get("anchor")
                if phrase in live and entry.get("anchor_distance", 9) <= MAX_ATTACH:
                    target = live[phrase] + np.array(entry["offset"])
                held = arguments.get("gripper", -1) == 1
                if held:
                    target = target - offset
                if max(abs(target[0]), abs(target[1])) > REACH:
                    continue
                arguments["xyz"] = [
                    round(float(target[0]), 4),
                    round(float(target[1]), 4),
                    float(xyz[2]),
                ]
                toolkit.execute_tool(name, arguments)
                look()
            elif name in {"segment", "segment_point"}:
                continue
            elif name in {"pi0_pick", "pi0_doubled"}:
                prompt = str(arguments.get("prompt", ""))
                stripped = re.sub(r"^(pick up|grasp)\s+the\s+", "", prompt, flags=re.I)
                held_phrase = re.split(r"\b(on|in|into|inside|by|and)\b", stripped)[
                    0
                ].strip()
                raw = toolkit.execute_tool(name, arguments).result
                look()
                if name == "pi0_pick" and not pick_confirmed(raw):
                    for _ in range(PICK_ATTEMPTS - 1):
                        if finished():
                            break
                        for again, again_args in recent:
                            toolkit.execute_tool(again, dict(again_args))
                        raw = toolkit.execute_tool(name, dict(arguments)).result
                        look()
                        if pick_confirmed(raw):
                            break
                    if not pick_confirmed(raw):
                        note("      pick unconfirmed, skipping its carry")
                        skip_suffix = True
            elif name == "set_gripper":
                toolkit.execute_tool(name, arguments)
                body = held_body(molmo, state, look(), prompt_for("held", held_phrase))
                eef = np.asarray(primitives.env.raw_obs()["robot0_eef_pos"][:2])
                candidate = body["xy"] - eef if body is not None else None
                if candidate is not None and np.linalg.norm(candidate) <= MAX_HELD:
                    offset = candidate
                    note(f"      held offset ({offset[0]:+.4f},{offset[1]:+.4f})")
                else:
                    offset = np.zeros(2)
            elif name == "release":
                toolkit.execute_tool(name, arguments)
                offset = np.zeros(2)
                look()
            else:
                toolkit.execute_tool(name, arguments)
                look()
        except Exception as exc:
            note(f"      {name} raised {type(exc).__name__}: {str(exc)[:70]}")
            break
        if name in {"release", "pi0_pick", "pi0_doubled"}:
            recent.clear()
        elif name in {"move_to", "move_pose", "set_gripper", "rotate_wrist"}:
            recent.append((name, dict(entry["arguments"])))
            recent = recent[-6:]

    return {"done": finished(), "anchors": len(live), "plan": len(plan)}


def replay_card(
    toolkit: Any, cell_tag: str, note: Callable[[str], None] = lambda _: None
) -> dict:
    """Replay the card for one cell, named the way the CLI names its cells.

    The ``RobotSpec`` hook behind ``--planner task_card``: everything LIBERO
    knows about its own cards -- how a tag maps to a card, and where its
    grounder is -- stays here, so the planner needs to know none of it.
    """
    match = _CELL.match(cell_tag)
    if not match:
        raise ValueError(
            f"cannot read the cell tag {cell_tag!r}; "
            "expected <family>_<suite>_t<task>_s<seed>"
        )
    # The seed selects the layout to solve, not the plan used to solve it.
    family, suite, task, _ = match.groups()
    key = f"{suite}_t{task}"
    folder = cards() / family / key
    if not (folder / "plan.json").is_file():
        raise FileNotFoundError(f"no task card for {family}/{key} under {CARDS}")

    endpoint = os.environ.get("MOLMO_ENDPOINT", "http://127.0.0.1:8115")
    molmo = MolmoClient(make_rpc_client(endpoint))
    note(f"replaying the {family}/{key} card")
    return {**replay(toolkit, molmo, load(folder), note), "card": f"{family}/{key}"}
