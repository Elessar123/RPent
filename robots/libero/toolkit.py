"""LIBERO toolkit: common tools + LIBERO primitives.

Inherits the common file/IO tools from :class:`Toolkit` and registers the
LIBERO primitives (``move_to``, ``pi0_pick``, ``release``, ...) on top.
"""
from __future__ import annotations

import shutil
import time
from functools import partial
from typing import Any

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink, ToolResultEvent
from rpent.tools.toolkit import ToolCancelled, Toolkit
from rpent.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """Toolkit for the LIBERO environment."""

    # Tool schemas keyed by name (built once from the canonical ordered list
    # in libero_tools.TOOLS_SPEC) so each tool registers with its own spec.
    _SPECS = {spec["name"]: spec for spec in libero_tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        video_path: str | None = None,
        explore: bool = False,
        attempts_per_session: int = 0,
    ) -> None:
        super().__init__(dashboard_events=dashboard_events)
        self._next_step: int = 0
        self._video_path: str | None = video_path
        # Evaluation is single-episode; only exploration exposes reset.
        self._explore = explore
        self._solved: bool = False
        self._attempt: int = 1
        # Bound each planner session while allowing later sessions to continue.
        self._attempts_per_session: int = max(0, int(attempts_per_session))
        self._session_attempt: int = 1
        self.init_primitives_clean(primitives_kwargs=primitives_kwargs)
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        specs = self._SPECS
        # Inspection tools do not advance environment state. Most are stateless
        # module functions; segment is bound to the primitives-owned SAM3 client.
        inspection_handlers = {
            "view_driver_state": libero_tools.view_driver_state,
            "view_camera_meta": libero_tools.view_camera_meta,
            "back_project": libero_tools.back_project,
            "segment": self._primitives.segment,
        }
        for name, handler in inspection_handlers.items():
            self.add_tool(name, specs[name], handler)
        # Primitive tools: each goes through _step, which looks up the
        # matching primitive method via getattr at call time.
        for name in libero_tools.PRIMITIVE_TOOL_NAMES:
            self.add_tool(name, specs[name], partial(self._step, name))
        if self._explore:
            self.add_tool("reset", specs["reset"], self._reset_episode)
            finish_spec, finish_handler = self._tools["finish"]
            self.add_tool(
                "finish", finish_spec, partial(self._guarded_finish, finish_handler)
            )

    def _guarded_finish(
        self, inner: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Refuse to end an unsolved session while attempts remain."""
        budget = self._attempts_per_session
        if budget and not self.solved() and self._session_attempt < budget:
            remaining = budget - self._session_attempt
            return {
                "error": "finish refused",
                "reason": (
                    f"This session has {remaining} of its {budget} attempts left "
                    "and the task is not solved. Archive this attempt, call "
                    "`reset`, and try another approach."
                ),
            }
        return inner(**kwargs)

    def begin_session(self) -> None:
        """Start a fresh agent session: the per-session attempt budget refills."""
        # A continuation resets before acting; that reset starts attempt one.
        self._session_attempt = 0

    def _reset_episode(self, reason: str) -> dict:
        """Restart the episode (explore only) and dump the fresh scene.

        The step counter keeps advancing across attempts so ``states.json``
        holds the whole exploration history — that trace is what the DISTILL
        pass mines for failure modes. ``write_recipe_from_states`` skips
        everything up to the last reset, so the exported recipe stays replayable.
        """
        budget = self._attempts_per_session
        if budget and self._session_attempt >= budget:
            return {
                "error": "reset refused",
                "reason": (
                    f"This session's attempt budget is spent ({budget} attempts). "
                    "Archive the attempt, update the handoff notes, and call "
                    "`finish` so the next session can continue."
                ),
            }
        self._attempt += 1
        self._session_attempt += 1
        out = self._step("reset_episode", reason=reason, command_name="reset")
        out["attempt"] = self._attempt
        out["notice"] = (
            f"Episode restarted; this is attempt {self._attempt}. The original "
            "layout was restored. Re-run perception before acting."
        )
        return out

    def _step(self, name: str, command_name: str | None = None, **kwargs) -> dict:
        """Run ``self._primitives.<name>(**kwargs)``, dump the new step, and
        return the rendered state view + log.

        ``command_name`` overrides the action recorded in the trace when the
        primitive method and the tool the LLM called are named differently
        (``reset_episode`` vs the ``reset`` tool).
        """
        command = {"action": command_name or name, **kwargs}
        t0 = time.time()
        start_frame = self._primitives.recorded_frame_count()
        try:
            result = getattr(self._primitives, name)(**kwargs)
            self.raise_if_cancelled()
        except ToolCancelled as exc:
            result = {
                "error": str(exc),
                "code": "tool_cancelled",
                "interrupted": True,
            }
        elapsed = round(time.time() - t0, 2)

        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": result}
        self._solved |= result_dict.get("libero_terminated") is True

        self._next_step += 1
        step_idx = self._next_step
        output_dir = get_output_dir()
        if self._dashboard_events.enabled:
            video_dir = libero_tools.artifact_path(output_dir, "action_videos")
            video_path = video_dir / f"step_{step_idx:02d}_{name}.mp4"
            try:
                self._primitives.save_frame_slice(start_frame, str(video_path), fps=20)
            except Exception as e:
                get_logger("libero_toolkit").warning(
                    f"failed to save action clip to {video_path}: {e}"
                )
        libero_tools.dump_state(
            self._primitives,
            str(output_dir),
            step_idx=step_idx,
            log={"command": command, "result": result_dict, "elapsed_s": elapsed},
        )
        out = libero_tools.view_driver_state(step_idx)
        out["agent_elapsed_s"] = elapsed
        if result_dict.get("interrupted"):
            out.update(result_dict)
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """Wipe stale run artifacts, build the LiberoPrimitives, dump step 0."""
        out_dir = get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in libero_tools.ARTIFACT_DIRECTORIES:
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        for target in (
            libero_tools.artifact_path(out_dir, "states"),
            libero_tools.artifact_path(out_dir, "metadata", camera="agentview", resolution="low"),
            libero_tools.artifact_path(out_dir, "episode_video"),
        ):
            if target.exists():
                target.unlink()

        primitives = libero_tools.LiberoPrimitives(
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        libero_tools.dump_state(primitives, str(out_dir), step_idx=0, log=None)
        self._dashboard_events.emit(
            ToolResultEvent(
                name="view_driver_state",
                result=libero_tools.view_driver_state(0),
            )
        )

        self._primitives = primitives

    def close(self) -> None:
        """Flush the agent-side video buffer to disk (end-of-run).
        """
        if self._video_path is None:
            return
        try:
            self._primitives.stop_recording_and_save(self._video_path)
        except Exception as e:
            # The runner is in the cleanup path; never let a video save
            # abort it.
            get_logger("libero_toolkit").warning(
                f"failed to save video to {self._video_path}: {e}"
            )

    def solved(self) -> bool:
        """Return whether this run has completed the task."""
        return self._solved

    def write_recipe(self, recipe_tag: str) -> str:
        """Write the LIBERO recipe JSONL from the dumped state trace."""
        return libero_tools.write_recipe_from_states(str(get_output_dir()), recipe_tag)
