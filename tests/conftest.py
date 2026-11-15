"""Repository-wide pytest configuration.

Registers opt-in markers for tests that need real model weights or a GPU.
By default both categories are skipped; CI enables them explicitly via
``--run-real-model`` / ``--run-gpu``. See ``docs/TEST_DEBT.md`` §2.1 for
the full policy.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-model",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.real_model (requires SAM/Grounding DINO/NeRF2Physics weights)",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.gpu (requires GPU / Isaac Sim)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-real-model"):
        skip = pytest.mark.skip(reason="needs --run-real-model (real model weights required)")
        for item in items:
            if "real_model" in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--run-gpu"):
        skip = pytest.mark.skip(reason="needs --run-gpu (GPU / Isaac Sim required)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
