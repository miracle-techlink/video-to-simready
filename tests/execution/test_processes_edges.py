"""Edge cases for cancellation, local cleanup, remote polling and supervision."""

import json
import sys
import time
from threading import Event

import pytest

from simready.core.artifacts import read_json
from simready.execution.processes.cancellation import ChildScope, parent_cancellation
from simready.execution.processes.local import (
    ProcessCancelledError,
    ProcessTimeoutError,
    run_process,
)
from simready.execution.processes.remote_polling import (
    RemoteJobError,
    RetryablePollError,
    poll_existing_job,
)
from simready.execution.processes.supervision import supervise_process


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


# ---------------------------------------------------------------------------
# Cancellation pipe edges
# ---------------------------------------------------------------------------


def test_child_scope_close_is_idempotent():
    scope = ChildScope()
    scope.close()
    scope.close()


def test_parent_cancellation_ignores_unknown_environment():
    """Unknown or missing cancellation env vars must not raise."""
    assert parent_cancellation(enabled=False) is None
    # No SIMREADY_CANCEL_FD set → returns None.
    assert parent_cancellation() is None


def test_parent_cancellation_rejects_invalid_fd(monkeypatch):
    monkeypatch.setenv("SIMREADY_CANCEL_FD", "not-a-fd:123")
    monkeypatch.setenv("SIMREADY_CANCEL_ACK_FD", "also-bad:456")
    with pytest.raises(ValueError, match="Invalid inherited cancellation"):
        parent_cancellation()


def test_request_stop_on_already_exited_process_is_noop(tmp_path):
    scope = ChildScope()
    run_process(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        timeout=5,
        stdout=tmp_path / "out",
        stderr=tmp_path / "err",
    )
    # Process has exited; request_stop must return immediately.
    scope.request_stop(type("P", (), {"poll": lambda self: 0})(), grace_s=1)
    scope.close()


# ---------------------------------------------------------------------------
# Local process cleanup edges
# ---------------------------------------------------------------------------


def test_timeout_kills_entire_process_group(tmp_path):
    """A child that spawns a grandchild must not leak the grandchild."""
    marker = tmp_path / "grandchild-ran"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        f"import time,pathlib; time.sleep(1.0); pathlib.Path({str(marker)!r}).touch()"
    )
    child = tmp_path / "child.py"
    child.write_text(
        f"import subprocess,sys,time; subprocess.Popen([sys.executable, {str(grandchild)!r}]); time.sleep(10)"
    )
    with pytest.raises(ProcessTimeoutError):
        run_process(
            [sys.executable, str(child)],
            cwd=tmp_path,
            timeout=0.3,
            stdout=tmp_path / "stdout",
            stderr=tmp_path / "stderr",
        )
    time.sleep(0.5)
    assert not marker.exists()


def test_cancel_event_checked_before_spawn(tmp_path):
    event = Event()
    event.set()
    with pytest.raises(ProcessCancelledError):
        run_process(
            [sys.executable, "-c", "raise AssertionError('must not run')"],
            cwd=tmp_path,
            timeout=5,
            stdout=tmp_path / "stdout",
            stderr=tmp_path / "stderr",
            cancel_event=event,
        )
    assert not (tmp_path / "stdout").exists()


def test_zero_exit_with_empty_stdout_is_success(tmp_path):
    result = run_process(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        timeout=5,
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
    )
    assert result.returncode == 0
    assert (tmp_path / "stdout").read_text() == ""


def test_large_stderr_does_not_deadlock(tmp_path):
    result = run_process(
        [sys.executable, "-c", "import sys; sys.stderr.write('x' * 500000)"],
        cwd=tmp_path,
        timeout=5,
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
    )
    assert result.returncode == 0
    assert len((tmp_path / "stderr").read_text()) == 500000


# ---------------------------------------------------------------------------
# Remote polling edges
# ---------------------------------------------------------------------------


def test_unknown_remote_state_is_fatal(tmp_path):
    def fetch(remaining):
        return {"status": "WEIRD_STATE"}

    with pytest.raises(ValueError, match="Unknown remote job state"):
        poll_existing_job(fetch, job_id="x", directory=tmp_path, timeout=10)


def test_transport_retry_backoff_caps_at_max_backoff(tmp_path):
    clock = Clock()
    calls = 0

    def fetch(remaining):
        nonlocal calls
        calls += 1
        raise RetryablePollError()

    with pytest.raises(TimeoutError):
        poll_existing_job(
            fetch,
            job_id="x",
            directory=tmp_path,
            timeout=100,
            interval=5,
            max_backoff=10,
            clock=clock,
            sleep=clock.sleep,
        )
    # backoff: 5, 10, 10, 10... should cap at 10
    assert calls >= 5


def test_poll_rejects_invalid_budget(tmp_path):
    with pytest.raises(ValueError):
        poll_existing_job(lambda r: {}, job_id="", directory=tmp_path, timeout=10)
    with pytest.raises(ValueError):
        poll_existing_job(lambda r: {}, job_id="x", directory=tmp_path, timeout=0)
    with pytest.raises(ValueError):
        poll_existing_job(
            lambda r: {}, job_id="x", directory=tmp_path, timeout=10, interval=0
        )


def test_poll_heartbeat_persists_across_restarts(tmp_path):
    clock = Clock()
    poll_existing_job(
        lambda r: {"status": "SUCCEEDED"},
        job_id="j1",
        directory=tmp_path,
        timeout=10,
        clock=clock,
        sleep=clock.sleep,
    )
    first = read_json(tmp_path / "heartbeat.json")
    assert first["status"] == "SUCCEEDED"
    with pytest.raises(RemoteJobError, match="FAILED"):
        poll_existing_job(
            lambda r: {"status": "FAILED"},
            job_id="j1",
            directory=tmp_path,
            timeout=10,
            clock=clock,
            sleep=clock.sleep,
        )
    second = read_json(tmp_path / "heartbeat.json")
    assert second["status"] == "FAILED"
    events = [
        json.loads(line)
        for line in (tmp_path / "poll_events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 2


# ---------------------------------------------------------------------------
# Supervision edges
# ---------------------------------------------------------------------------


def test_supervision_rejects_invalid_budget(tmp_path):
    with pytest.raises(ValueError, match="Invalid supervision budget"):
        supervise_process(
            ["true"], cwd=tmp_path, directory=tmp_path / "s", timeout=0, retry_delay=0
        )
    with pytest.raises(ValueError, match="Positive attempt count"):
        supervise_process(
            ["true"],
            cwd=tmp_path,
            directory=tmp_path / "s",
            timeout=10,
            max_attempts=0,
        )


def test_supervision_retry_delay_respects_deadline(tmp_path):
    Clock()
    attempts = 0

    def fail(argv, **kwargs):
        nonlocal attempts
        attempts += 1
        return type("R", (), {"returncode": 75, "seconds": 0.1})()

    import simready.execution.processes.supervision as supervision

    original_run = supervision.run_process
    supervision.run_process = fail
    try:
        report = supervise_process(
            ["worker"],
            cwd=tmp_path,
            directory=tmp_path / "s",
            timeout=0.5,
            max_attempts=5,
            retry_delay=10,
        )
    finally:
        supervision.run_process = original_run
    assert report["status"] == "TIMEOUT"
    assert attempts == 1


def test_supervision_attempt_logs_are_separate(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("import sys; print('attempt', len(sys.argv)); sys.exit(75)")
    report = supervise_process(
        [sys.executable, str(script)],
        cwd=tmp_path,
        directory=tmp_path / "s",
        timeout=10,
        max_attempts=2,
        retry_delay=0,
    )
    assert report["status"] == "FAILED"
    assert len(report["attempts"]) == 2
    assert (tmp_path / "s/attempt_0/stdout.log").is_file()
    assert (tmp_path / "s/attempt_1/stdout.log").is_file()


def test_supervision_non_75_nonzero_fails_immediately(tmp_path):
    report = supervise_process(
        [sys.executable, "-c", "raise SystemExit(1)"],
        cwd=tmp_path,
        directory=tmp_path / "s",
        timeout=10,
        retry_delay=0,
    )
    assert report["status"] == "FAILED"
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["returncode"] == 1
