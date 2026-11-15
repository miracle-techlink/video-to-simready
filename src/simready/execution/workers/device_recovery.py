"""Recheck an uncertain owned container before clearing its UUID lease quarantine."""

import fcntl
import os
import re
import time
from pathlib import Path

from simready.core.artifacts import loads_json, object_digest, write_json
from simready.core.validation import number
from simready.execution.processes.local import run_process
from simready.execution.resources.resource_pool import DEFAULT_LOCK_DIR
from simready.execution.workers.isaac_container import _cleanup_owned_container


def recover_device(uuid, *, directory, lock_dir=DEFAULT_LOCK_DIR, timeout_s=15):
    """Clear only after token-matched removal or confirmed absence of a known ID.

    A failed inspect, unknown creation identity or unreachable daemon cannot
    authorize reuse. Such cases retain quarantine and an actionable receipt.
    """
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
        raise ValueError("Recovery requires a physical GPU UUID")
    if number(timeout_s, "timeout_s") <= 0:
        raise ValueError("timeout_s must be positive")
    if timeout_s > 15:
        raise ValueError(
            "Recovery timeout_s must be <=15 to fit nested cancellation grace"
        )
    path = Path(lock_dir) / f"{uuid}.lease"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected an existing UUID lease file")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "gpu_uuid": uuid, "timeout_s": timeout_s}
    deadline = time.monotonic() + timeout_s
    try:
        with path.open("r+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report["status"] = "BUSY"
                return report
            text = handle.read()
            if not text:
                report["status"] = "NOT_QUARANTINED"
                return report
            previous = loads_json(text)
            report["previous_state_sha256"] = object_digest(previous)
            if previous.get("status") != "QUARANTINED":
                raise ValueError(
                    "Unknown device lease state; manual inspection required"
                )
            reason = previous["reason"]
            name, token = reason["name"], reason["ownership"]
            if not re.fullmatch(r"simready-[a-f0-9]{12}", name) or not re.fullmatch(
                r"[a-f0-9]{32}", token
            ):
                raise ValueError(
                    "Quarantine has no verified container ownership identity"
                )
            cleanup = {}
            report["cleanup"] = cleanup
            _cleanup_owned_container(
                name,
                token,
                timeout_s=timeout_s,
                prefix=directory / "recovery",
                report=cleanup,
            )
            confirmed = cleanup["status"] == "REMOVED"
            known_id = reason.get("cleanup", {}).get("container_id")
            known_id_valid = isinstance(known_id, str) and re.fullmatch(
                r"[a-f0-9]{64}", known_id
            )
            remaining = deadline - time.monotonic()
            if not confirmed and known_id_valid and remaining > 0:
                listing = directory / "known-id.stdout.log"
                result = run_process(
                    [
                        "docker",
                        "container",
                        "ls",
                        "--all",
                        "--no-trunc",
                        "--filter",
                        f"id={known_id}",
                        "--format",
                        "{{.ID}}",
                    ],
                    cwd=None,
                    timeout=remaining,
                    stdout=listing,
                    stderr=directory / "known-id.stderr.log",
                )
                confirmed = result.returncode == 0 and not listing.read_text().strip()
                report["known_id_absent"] = confirmed
            # With a VERIFIED container id, the name listing must never
            # authorize release: the container may still exist under a
            # different name (renamed), so an empty name listing cannot
            # prove it gone, and a failed ID query proves nothing either.
            # The name fallback is only sound for name-only quarantines
            # (container died under --rm before its ID was verified): there
            # a daemon-successful listing with no exact name match means the
            # container does not exist now. A failed listing (daemon down,
            # permission, timeout) proves nothing and keeps the quarantine;
            # an inspect rc=1 alone never did. Exact string compare on
            # {{.Names}} - neighbors sharing the prefix must not read as
            # present, and an existing same-name container with a foreign
            # ownership label stays NEEDS_REVIEW (removal already refused).
            if not confirmed and not known_id_valid:
                remaining = deadline - time.monotonic()
            if not confirmed and not known_id_valid and remaining > 0:
                listing = directory / "name-absence.stdout.log"
                result = run_process(
                    [
                        "docker",
                        "container",
                        "ls",
                        "--all",
                        "--no-trunc",
                        "--filter",
                        f"name=^/{name}$",
                        "--format",
                        "{{.Names}}",
                    ],
                    cwd=None,
                    timeout=remaining,
                    stdout=listing,
                    stderr=directory / "name-absence.stderr.log",
                )
                if result.returncode == 0:
                    listed = {
                        line.strip().lstrip("/")
                        for line in listing.read_text().splitlines()
                        if line.strip()
                    }
                    still_present = name in listed
                    confirmed = not still_present
                    report["name_absence_verified"] = confirmed
                    if still_present:
                        report["name_absence_reason"] = (
                            "Container name still present in the daemon listing"
                        )
                else:
                    report["name_absence_verified"] = False
                    report["name_absence_reason"] = (
                        "Docker listing failed (daemon unreachable, permission "
                        "or timeout); absence unconfirmed"
                    )
            if confirmed:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                report["status"] = "RECOVERED"
            else:
                report.update(
                    status="NEEDS_REVIEW",
                    reason="Container termination is unconfirmed; lease quarantine retained",
                )
    except BaseException as error:
        report.update(status="ERROR", error=type(error).__name__)
        raise
    finally:
        write_json(directory / "recovery.json", report)
    return report
