"""Operator replies must not be consumed as planner steering, or vice versa."""

import threading

import pytest

from rpent.cli.operator_input import OperatorInput
from rpent.tools.toolkit import ToolCancelled


def test_operator_replies_are_request_scoped_and_do_not_consume_steering(monkeypatch):
    broker = OperatorInput(interactive=True)
    ready = threading.Event()
    result = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: ready.set())
    worker = threading.Thread(
        target=lambda: result.append(broker("restore scene", lambda: None))
    )
    worker.start()
    assert ready.wait(2)
    request_id = broker._pending[0]
    assert not broker.route_line("success")
    assert broker.route_line("/operator stale done")
    assert not result
    assert broker.route_line(f"/operator {request_id} done")
    worker.join(2)
    assert result == ["done"] and not worker.is_alive()
    assert broker.route_line(f"/operator {request_id} success")
    assert broker._pending is None


def test_operator_eof_and_cancellation_release_pending_requests(monkeypatch):
    broker = OperatorInput(interactive=True)
    ready = threading.Event()
    result = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: ready.set())
    worker = threading.Thread(
        target=lambda: result.append(broker("verdict", lambda: None))
    )
    worker.start()
    assert ready.wait(2)
    broker.close()
    worker.join(2)
    assert result == [None] and broker._pending is None
    assert broker("late request", lambda: None) is None

    broker = OperatorInput(interactive=True)

    def cancelled():
        raise ToolCancelled("cancelled")

    with pytest.raises(ToolCancelled):
        broker("verdict", cancelled)
    assert broker._pending is None


def test_interactive_reader_routes_operator_input_without_stealing_steering(
    monkeypatch,
):
    import contextlib
    import queue
    from types import SimpleNamespace

    from rpent.cli import tui

    lines = iter(["initial task", "/operator request done", "move more slowly"])
    routed = []
    closed = []

    class Session:
        def __init__(self, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

    def route(line):
        if line.startswith("/operator "):
            routed.append(line)
            return True
        return False

    monkeypatch.setattr(tui, "PromptSession", Session)
    monkeypatch.setattr(
        tui.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
    )
    monkeypatch.setattr(tui, "_restore_tty_on_exit", lambda fd: None)
    monkeypatch.setattr(tui, "patch_stdout", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        tui, "_route_console_logs_to_current_stdout", contextlib.nullcontext
    )
    messages = queue.Queue()
    thread = tui.start_interactive_reader(
        messages, line_handler=route, on_close=lambda: closed.append(True)
    )
    thread.join(2)
    assert not thread.is_alive()
    assert [messages.get_nowait() for _ in range(3)] == [
        "initial task",
        "move more slowly",
        None,
    ]
    assert routed == ["/operator request done"] and closed == [True]
