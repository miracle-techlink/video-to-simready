"""Benchmark observation worker: contract validation, receipt layout, failure semantics."""

import copy
import json

import pytest

from simready.core.artifacts import digest, write_json
from simready.core.benchmark_contract import (
    ContractError,
)


def _view_set_manifest(view_ids=("front", "side")):
    ids = tuple(view_ids)
    if ids == ("front", "side"):
        return _view_set_default_manifest()
    template = _view_set_default_manifest()["views"]
    views = []
    for index, view_id in enumerate(ids):
        view = copy.deepcopy(template[index % len(template)])
        view["view_id"] = view_id
        view["camera_evidence"]["path"] = f"cam_{view_id.replace(' ', '_')}.json"
        views.append(view)
    return {
        "kind": "view_set",
        "schema_version": 1,
        "view_set_id": "vs-1",
        "coordinate_frame": "world",
        "unit": "m",
        "views": views,
    }


def _view_set_default_manifest():
    return {
        "kind": "view_set",
        "schema_version": 1,
        "view_set_id": "vs-1",
        "coordinate_frame": "world",
        "unit": "m",
        "views": [
            {
                "view_id": "front",
                "resolution": {"width": 64, "height": 64},
                "intrinsics": [[50, 0, 31.5], [0, 50, 31.5], [0, 0, 1]],
                "camera_to_world": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 5],
                    [0, 0, 0, 1],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": {"path": "cam_front.json", "sha256": "a" * 64},
            },
            {
                "view_id": "side",
                "resolution": {"width": 64, "height": 64},
                "intrinsics": [[50, 0, 31.5], [0, 50, 31.5], [0, 0, 1]],
                "camera_to_world": [
                    [0, -1, 0, 5],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "preprocessing": "none",
                "camera_evidence": {"path": "cam_side.json", "sha256": "b" * 64},
            },
        ],
    }


def _write_view_set(tmp_path, manifest=None):
    if manifest is None:
        manifest = _view_set_manifest()
    for view in manifest["views"]:
        ev_path = tmp_path / view["camera_evidence"]["path"]
        ev_path.write_text(json.dumps({"camera": view["view_id"]}))
        view["camera_evidence"]["sha256"] = digest(ev_path)
    path = tmp_path / "view_set.json"
    write_json(path, manifest)
    return path


def _write_render_request(tmp_path, *, view_ids=("front", "side"), **overrides):
    """Write a minimal valid render request manifest."""
    scene_file = tmp_path / "scene.usda"
    scene_file.write_text("#usda 1.0\n")
    view_set_file = _write_view_set(tmp_path, _view_set_manifest(view_ids))
    reconstruction_file = tmp_path / "reconstruction.json"
    reconstruction_file.write_text(json.dumps({"schema_version": 1}))

    manifest = {
        "kind": "render_request",
        "schema_version": 1,
        "request_id": "req-1",
        "scene": {"path": "scene.usda", "sha256": digest(scene_file)},
        "reconstruction": {
            "path": "reconstruction.json",
            "sha256": digest(reconstruction_file),
        },
        "view_set": {"path": "view_set.json", "sha256": digest(view_set_file)},
        "renderer": {
            "name": "isaac-sim",
            "version": "6.1.0",
            "code_sha256": "a" * 64,
            "environment": {
                "image": "nvcr.io/nvidia/isaac-sim:6.1.0",
                "python": "3.10",
            },
        },
        "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
        "lighting": {"policy": "scene_default", "exposure": 0.0},
        "seed": 42,
        "budget": {"max_wall_seconds": 3600.0},
    }
    manifest.update(overrides)
    path = tmp_path / "render_request.json"
    write_json(path, manifest)
    return path


class TestPlanValidation:
    """plan_benchmark_observation validates the full input chain."""

    def test_valid_plan_accepted(self, tmp_path):
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        req_path = _write_render_request(tmp_path)
        vs_path = tmp_path / "view_set.json"
        plan = plan_benchmark_observation(req_path, vs_path, tmp_path / "out")
        assert len(plan["views"]) == 2
        assert plan["render_request"]["request_id"] == "req-1"

    def test_invalid_render_request_rejected(self, tmp_path):
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        req_path = _write_render_request(tmp_path, seed=-1)
        vs_path = tmp_path / "view_set.json"
        with pytest.raises(ContractError, match="seed"):
            plan_benchmark_observation(req_path, vs_path, tmp_path / "out")

    def test_invalid_view_set_rejected(self, tmp_path):
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        req_path = _write_render_request(tmp_path)
        # Overwrite view_set with duplicate view IDs
        bad_manifest = _view_set_manifest()
        bad_manifest["views"][1]["view_id"] = "front"
        vs_path = tmp_path / "view_set.json"
        for view in bad_manifest["views"]:
            ev_path = tmp_path / view["camera_evidence"]["path"]
            ev_path.write_text(json.dumps({"camera": view["view_id"]}))
            view["camera_evidence"]["sha256"] = digest(ev_path)
        write_json(vs_path, bad_manifest)
        # Update the render_request to match the new view_set hash
        req = json.loads(req_path.read_text())
        req["view_set"]["sha256"] = digest(vs_path)
        write_json(req_path, req)
        with pytest.raises(ContractError, match="Duplicate view"):
            plan_benchmark_observation(req_path, vs_path, tmp_path / "out")

    def test_nonsquare_pixel_rejected(self, tmp_path):
        """Views with fx != fy must be rejected for benchmark use."""
        from simready.simulation.benchmark_observation import plan_benchmark_observation

        bad_manifest = _view_set_manifest()
        bad_manifest["views"][0]["intrinsics"][1][1] = 45  # fy != fx
        for view in bad_manifest["views"]:
            ev_path = tmp_path / view["camera_evidence"]["path"]
            ev_path.write_text(json.dumps({"camera": view["view_id"]}))
            view["camera_evidence"]["sha256"] = digest(ev_path)
        vs_path = tmp_path / "view_set.json"
        write_json(vs_path, bad_manifest)

        # Build render_request manually to reference the modified view_set
        scene_file = tmp_path / "scene.usda"
        scene_file.write_text("#usda 1.0\n")
        reconstruction_file = tmp_path / "reconstruction.json"
        reconstruction_file.write_text(json.dumps({"schema_version": 1}))
        req_manifest = {
            "kind": "render_request",
            "schema_version": 1,
            "request_id": "req-1",
            "scene": {"path": "scene.usda", "sha256": digest(scene_file)},
            "reconstruction": {
                "path": "reconstruction.json",
                "sha256": digest(reconstruction_file),
            },
            "view_set": {"path": "view_set.json", "sha256": digest(vs_path)},
            "renderer": {
                "name": "isaac-sim",
                "version": "6.1.0",
                "code_sha256": "a" * 64,
                "environment": {
                    "image": "nvcr.io/nvidia/isaac-sim:6.1.0",
                    "python": "3.10",
                },
            },
            "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
            "lighting": {"policy": "scene_default", "exposure": 0.0},
            "seed": 42,
            "budget": {"max_wall_seconds": 3600.0},
        }
        req_path = tmp_path / "render_request.json"
        write_json(req_path, req_manifest)
        with pytest.raises(ValueError, match="square pixels"):
            plan_benchmark_observation(req_path, vs_path, tmp_path / "out")


class TestReceiptLayout:
    """Receipt must match the fake_render_worker layout for interchangeability."""

    def test_receipt_top_level_keys(self, tmp_path):
        from simready.simulation.benchmark_observation import _build_receipt

        receipt = _build_receipt(
            render_request={
                "seed": 42,
                "image_settings": {"width": 64, "height": 64, "mode": "rgb"},
                "scene": {"path": "scene.usda"},
            },
            view_set={"views": [{"view_id": "v0"}, {"view_id": "v1"}]},
            views_result=[
                {
                    "view_id": "v0",
                    "width": 64,
                    "height": 64,
                    "rgb_file": "v0.rgb.png",
                    "rgb_sha256": "a" * 64,
                    "depth_file": None,
                    "depth_sha256": None,
                },
                {
                    "view_id": "v1",
                    "width": 64,
                    "height": 64,
                    "rgb_file": "v1.rgb.png",
                    "rgb_sha256": "b" * 64,
                    "depth_file": None,
                    "depth_sha256": None,
                },
            ],
            started=0.0,
        )
        # Must match fake_render_worker receipt keys
        expected_keys = {
            "synthetic_fixture",
            "banner",
            "worker",
            "scene_id",
            "seed",
            "want_depth",
            "render_settings",
            "views",
            "completed",
            "expected",
            "duration_seconds",
        }
        assert expected_keys <= set(receipt.keys())
        assert receipt["synthetic_fixture"] is False
        assert receipt["completed"] == receipt["expected"] == 2

    def test_receipt_views_have_required_fields(self, tmp_path):
        from simready.simulation.benchmark_observation import _build_receipt

        receipt = _build_receipt(
            render_request={
                "seed": 0,
                "image_settings": {"width": 32, "height": 32, "mode": "rgbd"},
                "scene": {"path": "s.usda"},
            },
            view_set={"views": [{"view_id": "v0"}]},
            views_result=[
                {
                    "view_id": "v0",
                    "width": 32,
                    "height": 32,
                    "rgb_file": "v0.rgb.png",
                    "rgb_sha256": "c" * 64,
                    "depth_file": "v0.depth.npy",
                    "depth_sha256": "d" * 64,
                },
            ],
            started=0.0,
        )
        view = receipt["views"][0]
        assert view["view_id"] == "v0"
        assert view["rgb_sha256"] == "c" * 64
        assert view["depth_sha256"] == "d" * 64


class TestFailureSemantics:
    """Crash, timeout, cancel must not publish complete success receipts."""

    def test_rgb_mode_flag_matches_render_mode(self):
        """want_depth in receipt matches render_request mode."""
        from simready.simulation.benchmark_observation import _build_receipt

        for mode, expected in [("rgb", False), ("rgbd", True)]:
            receipt = _build_receipt(
                render_request={
                    "seed": 0,
                    "image_settings": {"width": 32, "height": 32, "mode": mode},
                    "scene": {"path": "s.usda"},
                },
                view_set={"views": []},
                views_result=[],
                started=0.0,
            )
            assert receipt["want_depth"] is expected

    def test_incomplete_views_reflected_in_receipt(self):
        """Missing views are recorded, not silently padded."""
        from simready.simulation.benchmark_observation import _build_receipt

        receipt = _build_receipt(
            render_request={
                "seed": 0,
                "image_settings": {"width": 32, "height": 32, "mode": "rgb"},
                "scene": {"path": "s.usda"},
            },
            view_set={
                "views": [{"view_id": "v0"}, {"view_id": "v1"}, {"view_id": "v2"}]
            },
            views_result=[
                {
                    "view_id": "v0",
                    "width": 32,
                    "height": 32,
                    "rgb_file": "v0.rgb.png",
                    "rgb_sha256": "a" * 64,
                    "depth_file": None,
                    "depth_sha256": None,
                },
            ],
            started=0.0,
        )
        assert receipt["completed"] == 1
        assert receipt["expected"] == 3


class _KitFinder:
    """Meta-path stub for isaacsim/omni/carb/pxr reproducing kit import rules.

    Kit-only modules raise ModuleNotFoundError when imported before a
    SimulationApp exists — the exact live failure mode of F-GPU-2 — and every
    served import is journaled so tests can assert import/creation order.
    """

    KIT_ONLY = {
        "carb",
        "omni.kit.app",
        "omni.timeline",
        "omni.usd",
        "omni.replicator.core",
        "pxr",
    }

    def __init__(self, state):
        self._state = state

    def find_spec(self, fullname, path=None, target=None):
        if not (
            fullname == "isaacsim"
            or fullname in {"carb", "pxr"}
            or fullname.startswith("omni")
        ):
            return None
        import importlib.machinery

        return importlib.machinery.ModuleSpec(
            fullname, _KitLoader(fullname, self._state), is_package=True
        )


class _KitLoader:
    def __init__(self, fullname, state):
        self._fullname = fullname
        self._state = state

    def create_module(self, spec):
        import types

        return types.ModuleType(self._fullname)

    def exec_module(self, module):
        state = self._state
        if self._fullname in _KitFinder.KIT_ONLY and not state["app_created"]:
            raise ModuleNotFoundError(
                f"No module named {self._fullname!r} "
                "(kit modules are importable only after SimulationApp)"
            )
        state["journal"].append(("import", self._fullname))
        module.__path__ = []
        _populate(self._fullname, module, state)


def _populate(name, module, state):
    import types

    class _Settings:
        def set(self, key, value):
            state["settings"].append((key, value))
            state["journal"].append(("settings:set", key))

    class _ExtensionManager:
        def set_extension_enabled_immediate(self, ext, enabled):
            state["journal"].append(("ext:enable", ext))

    class _App:
        def get_extension_manager(self):
            return _ExtensionManager()

    class _Timeline:
        def pause(self):
            state["journal"].append(("timeline:pause",))

    class _Context:
        def open_stage(self, path):
            state["journal"].append(("stage:open", path))
            return True

        def get_stage(self):
            return state["stage"]

    class _Stage:
        def SetEditTarget(self, layer):  # noqa: N802
            state["journal"].append(("stage:edit-target",))

        def GetSessionLayer(self):  # noqa: N802 USD API
            return object()

        def Traverse(self):  # noqa: N802
            return iter(())

    if name == "isaacsim":

        class _SimulationApp:
            def __init__(self, config, experience=None):
                if not config.get("headless"):
                    raise ValueError("stub requires headless config")
                state["app_created"] = True
                state["journal"].append(("app:create",))

            def close(self, exit_code=0):
                state["journal"].append(("app:close", exit_code))

        module.SimulationApp = _SimulationApp
    elif name == "carb":
        module.settings = types.SimpleNamespace(get_settings=lambda: _Settings())
    elif name == "omni.kit.app":
        module.get_app = lambda: _App()
    elif name == "omni.timeline":
        module.get_timeline_interface = lambda: _Timeline()
    elif name == "omni.usd":
        state["stage"] = _Stage()
        module.get_context = lambda: _Context()
    elif name == "pxr":

        def _define_dome_light(stage, path):  # noqa: N802-free USD mirror
            return types.SimpleNamespace()

        module.UsdGeom = types.SimpleNamespace(GetStageMetersPerUnit=lambda stage: 1.0)
        module.UsdLux = types.SimpleNamespace(
            DomeLight=types.SimpleNamespace(Define=_define_dome_light),
            LightAPI=object,
        )

        def _compute_all_dependencies(path):
            state["journal"].append(("deps:compute", path))
            return [], [], list(state["unresolved"])

        module.UsdUtils = types.SimpleNamespace(
            ComputeAllDependencies=_compute_all_dependencies
        )
    elif name == "omni.replicator.core":
        module.annotator = types.SimpleNamespace()


_STUB_MODULES = [
    "isaacsim",
    "carb",
    "omni",
    "omni.kit",
    "omni.kit.app",
    "omni.timeline",
    "omni.usd",
    "omni.replicator",
    "omni.replicator.core",
    "pxr",
]


def _fresh_state():
    return {
        "app_created": False,
        "journal": [],
        "settings": [],
        "unresolved": [],
        "stage": None,
    }


@pytest.fixture
def kit_env(monkeypatch):
    """Install kit import stubs; yield the shared journal/state."""
    import sys

    state = _fresh_state()
    finder = _KitFinder(state)
    sys.meta_path.insert(0, finder)
    for name in _STUB_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("EXP_PATH", "/stub/isaacsim")
    yield state
    sys.meta_path.remove(finder)
    # Purge any stub the import system cached during the test so later
    # tests resolve the real (or absent) modules again.
    for name in _STUB_MODULES:
        sys.modules.pop(name, None)


def _write_worker_plan(tmp_path, *, view_ids=("front", "side")):
    """Bundle manifests in their own root and write a schema-2 worker plan."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    req_path = _write_render_request(bundle, view_ids=view_ids)
    vs_path = bundle / "view_set.json"
    out = tmp_path / "worker-out"
    plan = {
        "schema_version": 2,
        "kind": "observation_plan",
        "request_root": str(bundle),
        "render_request": {
            "path": "render_request.json",
            "sha256": digest(req_path),
        },
        "view_set_root": str(bundle),
        "view_set": {"path": "view_set.json", "sha256": digest(vs_path)},
        "output_dir": str(out),
        "mode": "rgb",
        "want_depth": False,
        "embedded_render_request": json.loads(req_path.read_text()),
        "embedded_view_set": json.loads(vs_path.read_text()),
    }
    plan_path = tmp_path / "observation_plan.json"
    write_json(plan_path, plan)
    return plan_path


def _stub_capture(monkeypatch):
    import numpy as np

    import simready.simulation.benchmark_observation as bo

    def _camera(stage, settings, scope_path=None):
        return f"{scope_path}/cam"

    def _capture(
        rep,
        camera_path,
        width,
        height,
        want_depth,
        app,
        warmup,
        subframes,
        attempts,
        stage_units,
        capture_mode,
    ):
        rgb = np.full((height, width, 4), 127, dtype=np.uint8)
        return rgb, None, {}, 1

    monkeypatch.setattr(bo, "_create_observation_camera", _camera)
    monkeypatch.setattr(bo, "_capture_single_view", _capture)
    monkeypatch.setattr(
        bo, "_validate_capture_result", lambda rgb, depth, capture_mode: None
    )


class TestWorkerLaunchPath:
    """Worker subcommand/argv must reach the real worker (F-GPU-1)."""

    def test_worker_subcommand_launches_real_worker(
        self, tmp_path, kit_env, monkeypatch
    ):
        """`main(['worker', ...])` must parse and complete, not argparse-error."""
        from simready.simulation.benchmark_observation import main

        _stub_capture(monkeypatch)
        plan_path = _write_worker_plan(tmp_path)
        out = tmp_path / "worker-out"
        rc = main(["worker", "--plan", str(plan_path), "--out", str(out)])
        failure = out / "failure.json"
        assert rc == 0, failure.read_text() if failure.exists() else "no receipt"
        receipt = json.loads((out / "receipt.json").read_text())
        assert receipt["completed"] == 2
        assert receipt["synthetic_fixture"] is False

    def test_kit_imports_only_after_simulation_app(
        self, tmp_path, kit_env, monkeypatch
    ):
        """omni.*/carb/pxr imports must follow SimulationApp creation (F-GPU-2)."""
        from simready.simulation.benchmark_observation import main

        _stub_capture(monkeypatch)
        plan_path = _write_worker_plan(tmp_path)
        rc = main(
            ["worker", "--plan", str(plan_path), "--out", str(tmp_path / "worker-out")]
        )
        assert rc == 0
        journal = kit_env["journal"]
        app_at = next(i for i, e in enumerate(journal) if e[0] == "app:create")
        kit_imports = [
            i
            for i, e in enumerate(journal)
            if e[0] == "import" and e[1] in _KitFinder.KIT_ONLY
        ]
        assert kit_imports and min(kit_imports) > app_at, journal

    def test_orchestrated_worker_argv_parses_through_main(
        self, tmp_path, kit_env, monkeypatch
    ):
        """run_benchmark_observation's worker argv must parse via main()."""
        from simready.simulation.benchmark_observation import (
            main,
            plan_benchmark_observation,
            run_benchmark_observation,
        )

        req_path = _write_render_request(tmp_path)
        vs_path = tmp_path / "view_set.json"
        plan = plan_benchmark_observation(req_path, vs_path, tmp_path / "out")
        captured = {}

        def _fake_launch_worker(worker, **kwargs):
            captured["argv"] = list(worker)
            return {"returncode": 0}

        import simready.execution.workers.isaac_launcher as launcher

        monkeypatch.setattr(launcher, "launch_worker", _fake_launch_worker)
        run_benchmark_observation(plan, backend="local")
        argv = captured["argv"]
        assert argv[:3] == [
            "-m",
            "simready.simulation.benchmark_observation",
            "worker",
        ]
        # The exact argv the orchestrator builds must parse through main().
        # The plan validates references relative to its own directory, so the
        # frozen manifest files must sit next to observation_plan.json.
        _stub_capture(monkeypatch)
        out_dir = tmp_path / "out"
        for manifest_file in tmp_path.iterdir():
            if manifest_file.is_file() and manifest_file.name != "plan.json":
                import shutil

                shutil.copy(manifest_file, out_dir / manifest_file.name)
        rc = main(argv[2:])
        assert rc == 0, argv

    def test_gaussian_settings_and_dependency_check(
        self, tmp_path, kit_env, monkeypatch
    ):
        """F-GPU-3: splat render settings set; unresolved layers fail loudly."""
        from simready.simulation.benchmark_observation import main

        _stub_capture(monkeypatch)
        plan_path = _write_worker_plan(tmp_path)
        out = tmp_path / "worker-out"
        assert main(["worker", "--plan", str(plan_path), "--out", str(out)]) == 0
        applied = dict(kit_env["settings"])
        assert applied["/rtx/rtpt/gaussian/accumulatedDepth/enabled"] is True
        assert applied["/rtx/rtpt/gaussian/accumulatedAlbedo/enabled"] is True
        assert applied["/rtx/rtpt/gaussian/maxGaussiansToAccumulate"] == 48
        journal = kit_env["journal"]
        deps_at = next(i for i, e in enumerate(journal) if e[0] == "deps:compute")
        open_at = next(i for i, e in enumerate(journal) if e[0] == "stage:open")
        assert deps_at < open_at, journal

        # Negative: unresolved layer must abort before rendering.
        kit_env["unresolved"] = ["missing_cabinet.usd"]
        (out / "receipt.json").unlink(missing_ok=True)
        rc = main(["worker", "--plan", str(plan_path), "--out", str(out)])
        assert rc == 1
        failure = json.loads((out / "failure.json").read_text())
        assert "missing_cabinet.usd" in failure["error"]

    def test_cli_help_needs_no_isaac(self, tmp_path):
        """`--help` (and module import) must never require Isaac (CLI contract)."""
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        env = {"PYTHONPATH": str(repo / "src"), "PATH": "/usr/bin:/bin"}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "simready.simulation.benchmark_observation",
                "--help",
            ],
            capture_output=True,
            env=env,
            cwd=repo,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        assert b"worker" in proc.stdout


class TestWorkerManifestRootAndExitCodes:
    """F-GPU-6/7: manifest roots survive plan->worker; exit codes honest."""

    def test_production_chain_with_distinct_cwd_input_output(
        self, tmp_path, kit_env, monkeypatch
    ):
        """cwd != input != output: real plan->run->worker with only the Isaac
        execution boundary stubbed; receipts keep original view ids."""
        from simready.simulation.benchmark_observation import (
            main,
            plan_benchmark_observation,
            run_benchmark_observation,
        )

        repo = tmp_path / "repo"
        (repo / "bundle").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        bundle = repo / "bundle"
        req_path = _write_render_request(
            bundle, view_ids=("view-0", "view_0", "0start", "sp ace")
        )
        vs_path = bundle / "view_set.json"
        first_evidence = json.loads(vs_path.read_text())["views"][0]["camera_evidence"][
            "path"
        ]
        frozen = {
            "request": digest(req_path),
            "view_set": digest(vs_path),
            "scene": digest(bundle / "scene.usda"),
            "evidence": digest(bundle / first_evidence),
        }
        out_dir = tmp_path / "run-out"

        captured = {}

        def _capture_argv(worker, **kwargs):
            captured["argv"] = list(worker)
            return {"returncode": 0}

        import simready.execution.workers.isaac_launcher as launcher

        monkeypatch.setattr(launcher, "launch_worker", _capture_argv)

        plan = plan_benchmark_observation(req_path, vs_path, out_dir)
        run_benchmark_observation(plan, backend="local", repo=repo, assets=repo)

        # Execute the exact production argv through main() from a foreign cwd.
        _stub_capture(monkeypatch)
        rc = main(captured["argv"][2:])
        assert rc == 0, captured["argv"]
        receipt = json.loads((out_dir / "receipt.json").read_text())
        assert [v["view_id"] for v in receipt["views"]] == [
            "view-0",
            "view_0",
            "0start",
            "sp ace",
        ]
        assert receipt["completed"] == receipt["expected"] == 4
        # Nothing frozen was rewritten by planning/execution.
        assert digest(req_path) == frozen["request"]
        assert digest(vs_path) == frozen["view_set"]
        assert digest(bundle / "scene.usda") == frozen["scene"]
        assert digest(bundle / first_evidence) == frozen["evidence"]
        # app.close(0) only after the complete receipt.
        assert kit_env["journal"][-1] == ("app:close", 0)

    def test_docker_backend_maps_plan_and_roots_into_container(
        self, tmp_path, monkeypatch
    ):
        """Docker argv and plan roots use /workspace,/assets,/output mounts."""
        from simready.simulation.benchmark_observation import (
            plan_benchmark_observation,
            run_benchmark_observation,
        )

        repo = tmp_path / "repo"
        assets = tmp_path / "assets"
        out_dir = tmp_path / "out"
        assets.mkdir()
        out_dir.mkdir()
        bundle = repo / "bundle"
        bundle.mkdir(parents=True)
        req_path = _write_render_request(bundle)
        vs_path = bundle / "view_set.json"
        captured = {}

        def _capture_argv(worker, **kwargs):
            captured["argv"] = list(worker)
            return {"returncode": 0}

        import simready.execution.workers.isaac_launcher as launcher

        monkeypatch.setattr(launcher, "launch_worker", _capture_argv)
        plan = plan_benchmark_observation(req_path, vs_path, out_dir)
        run_benchmark_observation(plan, backend="docker", repo=repo, assets=assets)
        argv = captured["argv"]
        assert argv[argv.index("--plan") + 1] == "/output/observation_plan.json"
        assert argv[argv.index("--out") + 1] == "/output"
        plan_data = json.loads((out_dir / "observation_plan.json").read_text())
        assert plan_data["request_root"] == "/workspace/bundle"
        assert plan_data["view_set_root"] == "/workspace/bundle"
        assert plan_data["output_dir"] == "/output"

    def test_docker_backend_rejects_unmounted_worker_inputs(
        self, tmp_path, monkeypatch
    ):
        """Inputs outside repo/assets/output are invisible in-container: fail
        loudly at launch, not inside the container."""
        from simready.simulation.benchmark_observation import (
            plan_benchmark_observation,
            run_benchmark_observation,
        )

        repo = tmp_path / "repo"
        assets = tmp_path / "assets"
        out_dir = tmp_path / "out"
        repo.mkdir()
        assets.mkdir()
        out_dir.mkdir()
        stray = tmp_path / "stray"
        stray.mkdir()
        req_path = _write_render_request(stray)
        vs_path = stray / "view_set.json"
        plan = plan_benchmark_observation(req_path, vs_path, out_dir)
        with pytest.raises(ValueError, match="invisible to the docker"):
            run_benchmark_observation(plan, backend="docker", repo=repo, assets=assets)

    def test_v1_plan_without_roots_rejected(self, tmp_path, kit_env, monkeypatch):
        """Old embedded-only plans fail loudly instead of guessing a root."""
        from simready.simulation.benchmark_observation import main

        req_path = _write_render_request(tmp_path)
        vs_path = tmp_path / "view_set.json"
        legacy = {
            "render_request": json.loads(req_path.read_text()),
            "view_set": json.loads(vs_path.read_text()),
            "mode": "rgb",
            "want_depth": False,
        }
        plan_path = tmp_path / "legacy_plan.json"
        write_json(plan_path, legacy)
        with pytest.raises(ValueError, match="schema_version=2"):
            main(["worker", "--plan", str(plan_path), "--out", str(tmp_path / "o")])

    def test_manifest_drift_rejected(self, tmp_path, kit_env, monkeypatch):
        """Manifest file changing after planning must fail hash verification."""
        from simready.simulation.benchmark_observation import main

        plan_path = _write_worker_plan(tmp_path)
        # Drift the view set file after the plan froze its hash.
        vs = json.loads((tmp_path / "bundle" / "view_set.json").read_text())
        vs["views"][0]["view_id"] = "drifted"
        write_json(tmp_path / "bundle" / "view_set.json", vs)
        with pytest.raises(ValueError, match="hash drift"):
            main(
                [
                    "worker",
                    "--plan",
                    str(plan_path),
                    "--out",
                    str(tmp_path / "worker-out"),
                ]
            )

    def test_partial_capture_fails_with_nonzero_close_and_rc(
        self, tmp_path, kit_env, monkeypatch
    ):
        """A view failing mid-run must produce rc!=0, failure.json and a
        nonzero app.close even though earlier views succeeded."""
        import simready.simulation.benchmark_observation as bo
        from simready.simulation.benchmark_observation import main

        plan_path = _write_worker_plan(tmp_path)
        _stub_capture(monkeypatch)
        stub_capture = bo._capture_single_view
        calls = {"n": 0}

        def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("view 1 capture exploded")
            return stub_capture(*args, **kwargs)

        monkeypatch.setattr(bo, "_capture_single_view", _flaky)
        out = tmp_path / "worker-out"
        rc = main(["worker", "--plan", str(plan_path), "--out", str(out)])
        assert rc == 1
        failure = json.loads((out / "failure.json").read_text())
        assert "capture exploded" in failure["error"]
        assert not (out / "receipt.json").exists()
        assert kit_env["journal"][-1] == ("app:close", 1)

    def test_view_id_prim_mapping_legal_and_injective(
        self, tmp_path, kit_env, monkeypatch
    ):
        """Contract-legal ids (hyphens, leading digits, spaces) map to legal,
        distinct USD prims; receipts keep the original ids."""
        from simready.simulation.benchmark_observation import main

        plan_path = _write_worker_plan(
            tmp_path, view_ids=("view-0", "view_0", "0start", "sp ace")
        )
        scopes = []
        import numpy as np

        import simready.simulation.benchmark_observation as bo

        def _camera(stage, settings, scope_path=None):
            scopes.append(scope_path)
            return f"{scope_path}/cam"

        monkeypatch.setattr(bo, "_create_observation_camera", _camera)

        def _capture(
            rep,
            camera_path,
            width,
            height,
            want_depth,
            app,
            warmup,
            subframes,
            attempts,
            stage_units,
            capture_mode,
        ):
            return (np.full((height, width, 4), 90, dtype=np.uint8), None, {}, 1)

        monkeypatch.setattr(bo, "_capture_single_view", _capture)
        monkeypatch.setattr(
            bo, "_validate_capture_result", lambda rgb, depth, capture_mode: None
        )
        rc = main(
            ["worker", "--plan", str(plan_path), "--out", str(tmp_path / "worker-out")]
        )
        assert rc == 0
        import re

        for scope in scopes:
            prim = scope.rsplit("/", 1)[-1]
            assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prim), scope
        assert len(set(scopes)) == len(scopes), scopes
        receipt = json.loads((tmp_path / "worker-out" / "receipt.json").read_text())
        assert [v["view_id"] for v in receipt["views"]] == [
            "view-0",
            "view_0",
            "0start",
            "sp ace",
        ]
