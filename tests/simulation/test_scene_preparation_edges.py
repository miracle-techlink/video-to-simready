"""Scene preparation edge cases: lighting policy, frozen light detection."""

import numpy as np
import pytest

from tests.fixtures.scene_preparation import cloud, write_ply


class TestLightingPolicy:
    """Missing light source handling must be explicit, not automatic."""

    def test_default_policy_adds_light(self, tmp_path):
        """Default 'auto' policy adds DomeLight unconditionally (legacy behavior)."""
        pytest.importorskip("pxr")
        from pxr import Usd, UsdLux

        from simready.simulation.scene_preparation import prepare_scene

        output = tmp_path / "prepared"
        report = prepare_scene(
            write_ply(tmp_path / "input.ply", cloud()),
            output,
            {"world_up_axis": "Z"},
        )
        stage = Usd.Stage.Open(str(output / report["scene"]))
        lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
        assert len(lights) >= 1

    def test_none_policy_does_not_add_light(self, tmp_path):
        """Explicit 'none' policy must not silently add light."""
        pytest.importorskip("pxr")
        from pxr import Usd, UsdLux

        from simready.simulation.scene_preparation import prepare_scene

        output = tmp_path / "prepared"
        report = prepare_scene(
            write_ply(tmp_path / "input.ply", cloud()),
            output,
            {"world_up_axis": "Z"},
            lighting_policy="none",
        )
        stage = Usd.Stage.Open(str(output / report["scene"]))
        lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
        assert len(lights) == 0

    def test_frozen_policy_adds_light_when_missing(self, tmp_path):
        """'frozen' policy adds light only when scene has none."""
        pytest.importorskip("pxr")
        from pxr import Usd, UsdLux

        from simready.simulation.scene_preparation import prepare_scene

        output = tmp_path / "prepared"
        report = prepare_scene(
            write_ply(tmp_path / "input.ply", cloud()),
            output,
            {"world_up_axis": "Z"},
            lighting_policy="frozen",
        )
        stage = Usd.Stage.Open(str(output / report["scene"]))
        lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
        assert len(lights) >= 1

    def test_invalid_policy_rejected(self, tmp_path):
        """Unknown lighting policy values must be rejected."""
        pytest.importorskip("pxr")

        from simready.simulation.scene_preparation import prepare_scene

        output = tmp_path / "prepared"
        with pytest.raises(ValueError, match="lighting_policy"):
            prepare_scene(
                write_ply(tmp_path / "input.ply", cloud()),
                output,
                {"world_up_axis": "Z"},
                lighting_policy="magical",
            )

    def test_policy_is_keyword_only(self, tmp_path):
        """lighting_policy must be keyword-only to avoid positional confusion."""
        pytest.importorskip("pxr")
        import inspect

        from simready.simulation.scene_preparation import prepare_scene

        sig = inspect.signature(prepare_scene)
        param = sig.parameters.get("lighting_policy")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
