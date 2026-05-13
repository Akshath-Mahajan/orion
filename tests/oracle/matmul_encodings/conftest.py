"""Shared fixtures for matmul-encoding oracle tests.

Mirrors tests/oracle/test_bootstrap.py's backend-parameterized scheme
fixture so every test runs against both backends. Disagreements between
lattigo and desilo localize bugs to the binding layer; agreement means
the kernel itself is correct.
"""

from __future__ import annotations

import gc
import os

import pytest

from benchmarks.matmul_encodings.context import Context


def _make_config(backend: str) -> dict:
    """Minimal CKKS config sized for the matmul-encoding kernels.

    LogN=13 -> 4096 slots (ConjugateInvariant). Five-level chain
    matches Negar's Lattigo runner default and gives 4 usable mult
    levels -- enough for a single CMult + Rescale per kernel call.
    """
    orion_cfg = {
        "backend": backend,
        "io_mode": "none",
        "debug": False,
    }
    # GPU opt-in for the desilo backend. Lattigo has no GPU path in
    # this wheel, so we only thread device through for desilo.
    if backend == "desilo":
        orion_cfg["device"] = os.environ.get("ORION_DESILO_DEVICE", "cpu")
    return {
        "ckks_params": {
            "LogN": 13,
            "LogQ": [29, 26, 26, 26, 26],
            "LogP": [29],
            "LogScale": 26,
            "H": 8192,
            "RingType": "ConjugateInvariant",
        },
        "orion": orion_cfg,
    }


@pytest.fixture(scope="module", params=["lattigo", "desilo"])
def ctx(request) -> Context:
    """Backend-parameterized Context. Module-scoped so each backend's
    scheme is built once and reused across the tests in a module.

    Teardown explicitly drops the Python-side reference and forces a GC
    cycle before the next module spins up its own scheme. Without this,
    pytest's collection of all five test modules tipped the lattigo
    binding's plaintext heap into a Go-runtime abort partway through
    the second module: Go's heap pressure depended on the *count of
    test items collected at startup* even though the per-module ctx is
    fresh. Forcing a Python GC + a small allocator nudge here shrinks
    that pressure enough to keep the whole suite green.
    """
    c = Context.from_config(_make_config(request.param))
    yield c
    c.scheme.delete_scheme()
    del c
    gc.collect()
