"""GPU capability detection and resource-pool lease edge cases."""

import json
import os
import threading
import time

import pytest

from simready.execution.resources.gpu_capability import (
    create_capability,
    inherited_capability,
    lease_uuid,
)
from simready.execution.resources.resource_pool import ResourcePool


@pytest.fixture
def inventory():
    return {
        "schema_version": 1,
        "cpu_slots": 4,
        "ram_gb": 4,
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-test-0",
                "vram_gb": 24,
                "capabilities": ["cuda", "rtx"],
                "rtx_isolated": True,
            },
            {
                "index": 1,
                "uuid": "GPU-test-1",
                "vram_gb": 24,
                "capabilities": ["cuda"],
            },
        ],
    }


# ---------------------------------------------------------------------------
# lease_uuid / capability
# ---------------------------------------------------------------------------


def test_lease_uuid_uses_physical_uuid_when_present():
    gpu = {"uuid": "MIG-alias", "physical_uuid": "GPU-physical-0"}
    assert lease_uuid(gpu) == "GPU-physical-0"


def test_lease_uuid_rejects_missing_or_malformed_uuid():
    with pytest.raises(ValueError, match="physical UUID"):
        lease_uuid({"uuid": "not-a-gpu-uuid"})
    with pytest.raises(ValueError, match="physical UUID"):
        lease_uuid({"uuid": "GPU-ok", "physical_uuid": "bad"})


def test_capability_roundtrip(tmp_path):
    lock = tmp_path / "GPU-test-0.lease"
    lock.write_text("")
    gpu = {"uuid": "GPU-test-0", "vram_gb": 24, "capabilities": ["cuda"]}
    fd = os.open(str(lock), os.O_RDONLY)
    try:
        cap = create_capability(fd, lock, gpu)
        try:
            assert os.pread(cap, 1, 0)  # sealed and readable
        finally:
            os.close(cap)
    finally:
        os.close(fd)


def test_inherited_capability_rejects_unsealed_fd(monkeypatch, tmp_path):
    lock = tmp_path / "GPU-test-0.lease"
    lock.write_text("")
    fd = os.open(str(lock), os.O_RDONLY)
    monkeypatch.setenv("SIMREADY_GPU_LEASE_FD", str(fd))
    try:
        with pytest.raises(RuntimeError, match="Invalid inherited GPU lease"):
            inherited_capability()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# ResourcePool CPU-only edges
# ---------------------------------------------------------------------------


def test_cpu_lease_restores_resources_on_close():
    pool = ResourcePool({"cpu_slots": 2, "ram_gb": 2, "gpus": []}, "/tmp/locks", "")
    lease = pool.acquire({"kind": "cpu", "cpu": 2, "ram_gb": 2})
    assert pool.cpu == 0 and pool.ram == 0
    lease.close()
    assert pool.cpu == 2 and pool.ram == 2


def test_cpu_lease_refused_when_insufficient():
    pool = ResourcePool({"cpu_slots": 1, "ram_gb": 1, "gpus": []}, "/tmp/locks", "")
    assert pool.acquire({"kind": "cpu", "cpu": 2, "ram_gb": 1}) is None
    assert pool.acquire({"kind": "cpu", "cpu": 1, "ram_gb": 2}) is None


def test_lease_close_is_idempotent_and_restores_once():
    pool = ResourcePool({"cpu_slots": 2, "ram_gb": 2, "gpus": []}, "/tmp/locks", "")
    lease = pool.acquire({"kind": "cpu", "cpu": 2, "ram_gb": 2})
    lease.close()
    lease.close()
    assert pool.cpu == 2 and pool.ram == 2


def test_lease_resources_are_copied():
    pool = ResourcePool({"cpu_slots": 2, "ram_gb": 2, "gpus": []}, "/tmp/locks", "")
    resources = {"kind": "cpu", "cpu": 1, "ram_gb": 1}
    lease = pool.acquire(resources)
    resources["cpu"] = 99
    lease.close()
    assert pool.cpu == 2


# ---------------------------------------------------------------------------
# GPU lease edges
# ---------------------------------------------------------------------------


def test_gpu_lease_excludes_busy_uuid(tmp_path, inventory):
    pool = ResourcePool(inventory, tmp_path / "locks", "")
    first = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    assert first is not None
    second = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    assert second is not None
    assert first.gpu["uuid"] != second.gpu["uuid"]
    third = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    assert third is None
    first.close()
    third = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    assert third is not None
    assert third.gpu["uuid"] == first.gpu["uuid"]
    second.close()
    third.close()


def test_gpu_lease_respects_vram_requirement(tmp_path, inventory):
    inventory["gpus"][0]["vram_gb"] = 4
    pool = ResourcePool(inventory, tmp_path / "locks", "")
    lease = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 8})
    assert lease is not None
    assert lease.gpu["uuid"] == "GPU-test-1"
    lease.close()


def test_gpu_lease_capability_filter(tmp_path, inventory):
    pool = ResourcePool(inventory, tmp_path / "locks", "")
    lease = pool.acquire({"kind": "rtx", "ram_gb": 0.1, "vram_gb": 1})
    assert lease.gpu["uuid"] == "GPU-test-0"
    lease.close()


def test_concurrent_lease_acquisition_is_serialized(tmp_path, inventory):
    pool = ResourcePool(inventory, tmp_path / "locks", "")
    acquired = []

    def grab():
        lease = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
        if lease:
            acquired.append(lease.gpu["uuid"])
            time.sleep(0.05)
            lease.close()

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(acquired) == 2  # only two GPUs
    assert len(set(acquired)) == 2


def test_lease_quarantine_blocks_reacquisition(tmp_path, inventory):
    pool = ResourcePool(inventory, tmp_path / "locks", "")
    lease = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    lease.quarantine({"reason": "test"})
    lease.close()
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})


def test_quarantine_preserves_reason_in_lock_file(tmp_path, inventory):
    pool = ResourcePool(inventory, tmp_path / "locks", "")
    lease = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    lease.quarantine({"reason": "container unknown"})
    lease.close()
    lock = tmp_path / "locks" / "GPU-test-0.lease"
    payload = json.loads(lock.read_text())
    assert payload["status"] == "QUARANTINED"
    assert payload["reason"] == {"reason": "container unknown"}


def test_cloud_slots(tmp_path):
    pool = ResourcePool(
        {"cpu_slots": 4, "ram_gb": 4, "gpus": [], "cloud_slots": 1},
        tmp_path / "locks",
        "",
    )
    lease = pool.acquire({"kind": "cloud", "ram_gb": 0.1})
    assert lease is not None
    assert pool.cloud == 0
    assert pool.acquire({"kind": "cloud", "ram_gb": 0.1}) is None
    lease.close()
    assert pool.cloud == 1


def test_legacy_glob_blocks_gpu_acquisition(tmp_path, inventory):
    legacy = tmp_path / "old.gpu.lock"
    legacy.touch()
    pool = ResourcePool(inventory, tmp_path / "locks", str(tmp_path / "*.gpu.lock"))
    assert pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1}) is None
    legacy.unlink()
    lease = pool.acquire({"kind": "cuda", "ram_gb": 0.1, "vram_gb": 1})
    assert lease is not None
    lease.close()
