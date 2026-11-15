"""Shared synthetic inputs; no test collection or real device initialization."""

import os
import sys

import pytest


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "docker"
    script.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["FAKE_DOCKER_ROOT"])
mode = os.environ.get("FAKE_DOCKER_MODE", "fail")
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "run":
    labels = [args[i+1] for i,v in enumerate(args) if v == "--label"]
    token = next(x.split("=",1)[1] for x in labels if x.startswith("org.simready.launch="))
    (root / "token").write_text(token)
    print("fake worker", flush=True)
    if mode == "hang": time.sleep(30)
    sys.exit(0 if mode == "success" else 7)
if args[0] == "inspect":
    if mode == "inspect-hang": time.sleep(30)
    if mode in ("inspect-fail", "known-absent", "name-absent", "name-neighbor", "name-present", "daemon-error", "id-query-error"): sys.exit(1)
    token = "wrong-owner" if mode == "mismatch" else (root / "token").read_text()
    print("a" * 64 + " " + token)
    sys.exit(0)
if args[0] == "rm":
    sys.exit(9 if mode == "remove-fail" else 0)
if args[0] == "container":
    if mode == "daemon-error":
        print("Cannot connect to the Docker daemon at unix:///var/run/docker.sock", file=sys.stderr)
        sys.exit(1)
    fmt = args[args.index("--format") + 1]
    if fmt == "{{.ID}}":
        if mode == "id-query-error":
            print("docker: error during connect", file=sys.stderr)
            sys.exit(1)
        if mode != "known-absent": print("a" * 64)
        sys.exit(0)
    # Name queries mirror real docker: an existing container (same name,
    # whatever its ownership) is listed; a gone one is not.
    filt = args[args.index("--filter") + 1]
    name = filt[len("name=^/"):-1]
    if mode == "name-neighbor":
        print("simready-0123456789ab")
        sys.exit(0)
    if mode in ("name-present", "mismatch"):
        print(name)
        sys.exit(0)
    sys.exit(0)
sys.exit(99)
"""
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_DOCKER_ROOT", str(tmp_path))
    return tmp_path
