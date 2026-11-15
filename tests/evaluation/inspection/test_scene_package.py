"""Scene package inspection: loadability, bounds, Gaussian collision rejection."""

import pytest

from simready.evaluation.inspection.scene_package import inspect_scene


class TestInspectScene:
    """These tests verify the scene_package inspector's contract checks.

    The inspector requires actual USD files (via pxr), so we test the
    validation logic that runs before USD parsing: path checks, schema
    checks, and output constraints.
    """

    def test_missing_scene_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            inspect_scene(tmp_path)

    def test_output_must_not_exist(self, tmp_path):
        """The CLI entry point refuses to overwrite an existing output."""
        from simready.evaluation.inspection.scene_package import main

        out = tmp_path / "existing.json"
        out.write_text("{}")
        with pytest.raises(FileExistsError):
            main(["--scene-directory", str(tmp_path), "--out", str(out)])
