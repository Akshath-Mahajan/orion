"""Shared fixtures for matmul-encoding oracle tests.

Mirrors tests/oracle/test_bootstrap.py's backend-parameterized scheme
fixture so every test runs against both backends. Disagreements between
lattigo and desilo localize bugs to the binding layer; agreement means
the kernel itself is correct.
"""

from __future__ import annotations

import pytest

from benchmarks.matmul_encodings.context import Context


def _make_config(backend: str) -> dict:
    """Minimal CKKS config sized for the matmul-encoding kernels.

    LogN=13 -> 4096 slots (ConjugateInvariant). Five-level chain
    matches Negar's Lattigo runner default and gives 4 usable mult
    levels -- enough for a single CMult + Rescale per kernel call.
    """
    return {
        "ckks_params": {
            "LogN": 13,
            "LogQ": [29, 26, 26, 26, 26],
            "LogP": [29],
            "LogScale": 26,
            "H": 8192,
            "RingType": "ConjugateInvariant",
        },
        "orion": {
            "backend": backend,
            "io_mode": "none",
            "debug": False,
        },
    }


@pytest.fixture(scope="module", params=["lattigo", "desilo"])
def ctx(request) -> Context:
    """Backend-parameterized Context. Module-scoped so each backend's
    scheme is built once and reused across the tests in a module.
    """
    c = Context.from_config(_make_config(request.param))
    yield c
    c.scheme.delete_scheme()
