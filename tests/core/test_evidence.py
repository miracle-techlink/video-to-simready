"""Direct coverage for evidence reference resolution and tamper detection."""

import pytest

from simready.core import evidence
from simready.core.artifacts import digest


def _file(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_reference_records_absolute_path_and_digest(tmp_path):
    path = _file(tmp_path / "a.bin")
    ref = evidence.reference(path)
    assert ref["path"] == str(path.resolve())
    assert ref["sha256"] == digest(path)


def test_resolve_ref_accepts_matching_file(tmp_path):
    path = _file(tmp_path / "a.bin")
    resolved = evidence.resolve_ref({"path": "a.bin", "sha256": digest(path)}, tmp_path)
    assert resolved == path.resolve()


def test_resolve_ref_rejects_non_object(tmp_path):
    with pytest.raises(ValueError, match="must be an object"):
        evidence.resolve_ref(["a.bin"], tmp_path)


def test_resolve_ref_rejects_missing_path_field(tmp_path):
    with pytest.raises(ValueError, match="needs a path"):
        evidence.resolve_ref({"sha256": "0" * 64}, tmp_path)


def test_resolve_ref_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Missing file"):
        evidence.resolve_ref({"path": "absent.bin", "sha256": "0" * 64}, tmp_path)


def test_resolve_ref_rejects_hash_mismatch(tmp_path):
    path = _file(tmp_path / "a.bin")
    ref = {"path": "a.bin", "sha256": digest(path)}
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evidence.resolve_ref(ref, tmp_path)


def test_resolve_ref_accepts_absolute_path_inside_root(tmp_path):
    path = _file(tmp_path / "sub" / "b.bin")
    ref = {"path": str(path.resolve()), "sha256": digest(path)}
    assert evidence.resolve_ref(ref, tmp_path) == path.resolve()


def test_ensure_unchanged_accepts_untouched_references(tmp_path):
    refs = [evidence.reference(_file(tmp_path / name)) for name in ("a", "b")]
    evidence.ensure_unchanged(refs)


def test_ensure_unchanged_detects_tampering(tmp_path):
    path = _file(tmp_path / "a.bin")
    refs = [evidence.reference(path)]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Frozen campaign evidence changed"):
        evidence.ensure_unchanged(refs)


def test_ensure_unchanged_detects_missing_file(tmp_path):
    path = _file(tmp_path / "a.bin")
    refs = [evidence.reference(path)]
    path.unlink()
    with pytest.raises(OSError):
        evidence.ensure_unchanged(refs)
