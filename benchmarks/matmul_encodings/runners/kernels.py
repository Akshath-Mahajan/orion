"""Per-kernel benchmark closures.

Each ``bench_<kernel>(ctx, shape, n_trials, warmup, verify) -> BenchResult``
builds the kernel inputs once (untimed), wraps the kernel call in a
closure, and hands it to ``_common.bench_kernel`` for timing + HBM +
op-count collection.

The kernel call inside the closure is JUST the encrypted-domain matmul
(post-encrypt, pre-decrypt) -- mirrors Negar's Go runners which time
only the kernel proper, not the encrypt / decrypt setup. Decryption +
verification happens once after the timed loop, on the last result.
"""

from __future__ import annotations

import numpy as np

from benchmarks.matmul_encodings.context import Context
from benchmarks.matmul_encodings.kernels import (
    bmm1_cipher, bmm3_cipher, moai_cipher, rowenc_cipher, thor_cipher,
)
from benchmarks.matmul_encodings.plaintext.bmm1_plain import (
    bicyclic_decode, bicyclic_encode, repeat_vector,
)
from benchmarks.matmul_encodings.plaintext.bmm3_plain import break_into_chunks
from benchmarks.matmul_encodings.plaintext.thor_plain import (
    build_masks_for_kernel, decode_batched, encode_batched, replication,
)

from ._common import BenchResult, bench_kernel
from .gpu_sampler import GpuMonitor
from .shapes import Bmm1Shape, Bmm3Shape, MoaiShape, RowEncShape, ThorShape


_BACKEND = lambda ctx: type(ctx.backend).__name__.replace("Library", "").lower()


# ---------------------------------------------------------------------------
# BMM-I
# ---------------------------------------------------------------------------


def bench_bmm1(
    ctx: Context, shape: Bmm1Shape, *, n_trials: int, warmup: int,
    verify: bool, device: str, gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    """Single-block BMM-I bench. Negar's Go runner runs N/s_n * M/s_m * P/s_p
    block matmuls per shape -- to stay close to that, we time ONE block call
    here and the harness scales the per-block time externally if desired."""
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((shape.s_n, shape.s_m))
    B = rng.standard_normal((shape.s_m, shape.s_p))

    ct_a = bmm1_cipher.encrypt_a(ctx, A, shape.s_p)
    ct_b = bmm1_cipher.encrypt_b(ctx, B, shape.s_n)

    def run():
        return bmm1_cipher.bmm1_he_hoisted(
            ctx, ct_a, ct_b, shape.s_n, shape.s_m, shape.s_p,
        )

    def verify_fn(ct_out: int) -> float:
        got = bmm1_cipher.decode_output(ctx, ct_out, shape.s_n, shape.s_p)
        return float(np.max(np.abs(got - A @ B)))

    return bench_kernel(
        backend=_BACKEND(ctx), device=device, kernel="bmm1",
        shape=shape.label, n_he=ctx.slots, ctx=ctx, run_fn=run,
        n_trials=n_trials, warmup=warmup,
        verify_fn=verify_fn if verify else None,
        gpu_monitor=gpu_monitor,
    )


# ---------------------------------------------------------------------------
# BMM-III
# ---------------------------------------------------------------------------


def bench_bmm3(
    ctx: Context, shape: Bmm3Shape, *, n_trials: int, warmup: int,
    verify: bool, device: str, gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    rng = np.random.default_rng(seed=42)
    n, m, p = shape.n, shape.m, shape.p
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    n_he = ctx.slots
    a_chunks = break_into_chunks(bicyclic_encode(A), n * m, n * p, n_he)
    b_chunks = break_into_chunks(bicyclic_encode(B), m * p, n * p, n_he)
    a_cts = bmm3_cipher.encrypt_chunks(ctx, a_chunks, n_he, ctx.max_level)
    b_cts = bmm3_cipher.encrypt_chunks(ctx, b_chunks, n_he, ctx.max_level)

    if shape.mode == "cached":
        def run():
            return bmm3_cipher.bmm3_he_cached(
                ctx, a_cts, b_cts, n, m, p, n_he, ctx.max_level,
            )
    elif shape.mode == "hoisted":
        def run():
            return bmm3_cipher.bmm3_he_hoisted(
                ctx, a_cts, b_cts, n, m, p, n_he, ctx.max_level,
                hoist_block_size=shape.hoist_block_size,
            )
    else:
        raise ValueError(
            f"bench_bmm3: unknown shape.mode={shape.mode!r}; "
            f"expected 'cached' or 'hoisted'"
        )

    def verify_fn(ct_chunks: list) -> float:
        got = bmm3_cipher.decode_output(ctx, ct_chunks, n, p, n_he)
        return float(np.max(np.abs(got - A @ B)))

    return bench_kernel(
        backend=_BACKEND(ctx), device=device, kernel="bmm3",
        shape=shape.label, n_he=n_he, ctx=ctx, run_fn=run,
        n_trials=n_trials, warmup=warmup,
        verify_fn=verify_fn if verify else None,
        gpu_monitor=gpu_monitor,
    )


# ---------------------------------------------------------------------------
# THOR
# ---------------------------------------------------------------------------


def bench_thor(
    ctx: Context, shape: ThorShape, *, n_trials: int, warmup: int,
    verify: bool, device: str, gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    rng = np.random.default_rng(seed=42)
    H, m, n, c = shape.H, shape.m, shape.n, shape.c
    s_enc = c * n * H
    if ctx.slots % s_enc != 0:
        raise ValueError(
            f"THOR shape {shape.label} requires ctx.slots ({ctx.slots}) to be a "
            f"multiple of c*n*H ({s_enc})"
        )
    As = rng.standard_normal((H, m, n))
    Bs = rng.standard_normal((H, n, n))
    masks = build_masks_for_kernel(c=c, n=n, H=H, s=s_enc)

    p_as_cts = thor_cipher.encrypt_a(ctx, As, c=c, H=H)
    p_b_rep_cts = thor_cipher.encrypt_b_rep(ctx, Bs, c=c, H=H)

    def run():
        return thor_cipher.thor_cc_matmul_he(
            ctx, p_as_cts, p_b_rep_cts, masks, n=n, H=H, c=c,
        )

    def verify_fn(out_cts: list) -> float:
        decoded = np.stack(
            [ctx.decrypt(ct)[:s_enc] for ct in out_cts], axis=0
        )
        got = decode_batched(decoded, m=m, n=n, c=c, H=H)
        return float(np.max(np.abs(got - np.matmul(As, Bs))))

    return bench_kernel(
        backend=_BACKEND(ctx), device=device, kernel="thor",
        shape=shape.label, n_he=ctx.slots, ctx=ctx, run_fn=run,
        n_trials=n_trials, warmup=warmup,
        verify_fn=verify_fn if verify else None,
        gpu_monitor=gpu_monitor,
    )


# ---------------------------------------------------------------------------
# MOAI Algorithm 3 (Col x Col -> Diag) and Algorithm 4 (Diag x Col -> Col)
# ---------------------------------------------------------------------------


def bench_moai_alg3(
    ctx: Context, shape: MoaiShape, *, n_trials: int, warmup: int,
    verify: bool, device: str, gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    rng = np.random.default_rng(seed=42)
    n_batch, m, d_prime = shape.n_batch, shape.m, shape.d_prime
    n_he = n_batch * m
    if ctx.slots % n_he != 0:
        raise ValueError(
            f"MOAI shape {shape.label} requires ctx.slots ({ctx.slots}) to be a "
            f"multiple of n_batch*m ({n_he})"
        )
    Qs = rng.standard_normal((n_batch, m, d_prime))
    Ks = rng.standard_normal((n_batch, m, d_prime))

    Q_cts = moai_cipher.encrypt_col_packed(ctx, Qs, n_he=n_he)
    K_cts = moai_cipher.encrypt_col_packed(ctx, Ks, n_he=n_he)

    def run():
        return moai_cipher.moai_col_col_bsgs_he(
            ctx, Q_cts, K_cts, m=m, d_prime=d_prime,
            rot_stride=n_batch, n_he=n_he,
        )

    def verify_fn(out_cts: list) -> float:
        decoded = np.stack(
            [ctx.decrypt(ct)[:n_he] for ct in out_cts], axis=0
        )
        from benchmarks.matmul_encodings.plaintext.moai_plain import (
            interleaved_diag_unpack,
        )
        got = interleaved_diag_unpack(decoded, m=m, n_batch=n_batch)
        ref = np.matmul(Qs, np.transpose(Ks, axes=(0, 2, 1)))
        return float(np.max(np.abs(got - ref)))

    return bench_kernel(
        backend=_BACKEND(ctx), device=device, kernel="moai_alg3",
        shape=shape.label, n_he=n_he, ctx=ctx, run_fn=run,
        n_trials=n_trials, warmup=warmup,
        verify_fn=verify_fn if verify else None,
        gpu_monitor=gpu_monitor,
    )


def bench_moai_alg4(
    ctx: Context, shape: MoaiShape, *, n_trials: int, warmup: int,
    verify: bool, device: str, gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    rng = np.random.default_rng(seed=42)
    n_batch, m, d_prime = shape.n_batch, shape.m, shape.d_prime
    n_he = n_batch * m
    if ctx.slots % n_he != 0:
        raise ValueError(
            f"MOAI shape {shape.label} requires ctx.slots ({ctx.slots}) to be a "
            f"multiple of n_batch*m ({n_he})"
        )
    # Alg 4 takes (m, m) diag-packed C and (m, d') col-packed V.
    Cs = rng.standard_normal((n_batch, m, m))
    Vs = rng.standard_normal((n_batch, m, d_prime))

    C_cts = moai_cipher.encrypt_diag_packed(ctx, Cs, n_he=n_he)
    V_cts = moai_cipher.encrypt_col_packed(ctx, Vs, n_he=n_he)

    def run():
        return moai_cipher.moai_diag_col_bsgs_he(
            ctx, C_cts, V_cts, m=m, d_prime=d_prime,
            rot_stride=n_batch, n_he=n_he,
        )

    def verify_fn(out_cts: list) -> float:
        decoded = np.stack(
            [ctx.decrypt(ct)[:n_he] for ct in out_cts], axis=0
        )
        from benchmarks.matmul_encodings.plaintext.moai_plain import (
            interleaved_column_unpack,
        )
        got = interleaved_column_unpack(decoded, m=m, n_batch=n_batch)
        return float(np.max(np.abs(got - np.matmul(Cs, Vs))))

    return bench_kernel(
        backend=_BACKEND(ctx), device=device, kernel="moai_alg4",
        shape=shape.label, n_he=n_he, ctx=ctx, run_fn=run,
        n_trials=n_trials, warmup=warmup,
        verify_fn=verify_fn if verify else None,
        gpu_monitor=gpu_monitor,
    )


# ---------------------------------------------------------------------------
# RowEnc
# ---------------------------------------------------------------------------


def bench_rowenc(
    ctx: Context, shape: RowEncShape, *, n_trials: int, warmup: int,
    verify: bool, device: str, gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    rng = np.random.default_rng(seed=42)
    n = shape.n
    if n * n > ctx.slots:
        raise ValueError(f"RowEnc n={n} requires n*n <= ctx.slots ({ctx.slots})")
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))

    from benchmarks.matmul_encodings.plaintext.rowenc_plain import (
        row_pack, row_unpack,
    )

    ct_a = ctx.encrypt(row_pack(A), level=ctx.max_level)
    ct_b = ctx.encrypt(row_pack(B), level=ctx.max_level)

    def run():
        return rowenc_cipher.rowenc_he_kernel(ctx, ct_a, ct_b, n, ctx.max_level)

    def verify_fn(ct_out: int) -> float:
        vec = ctx.decrypt(ct_out)
        return float(np.max(np.abs(row_unpack(vec, n) - A @ B)))

    return bench_kernel(
        backend=_BACKEND(ctx), device=device, kernel="rowenc",
        shape=shape.label, n_he=ctx.slots, ctx=ctx, run_fn=run,
        n_trials=n_trials, warmup=warmup,
        verify_fn=verify_fn if verify else None,
        gpu_monitor=gpu_monitor,
    )


KERNEL_TABLE = {
    "bmm1":      (bench_bmm1,      "bmm1"),
    "bmm3":      (bench_bmm3,      "bmm3"),
    "thor":      (bench_thor,      "thor"),
    "moai_alg3": (bench_moai_alg3, "moai"),
    "moai_alg4": (bench_moai_alg4, "moai"),
    "rowenc":    (bench_rowenc,    "rowenc"),
}
"""Maps kernel-id (CSV column value) to (bench_fn, shape_set_key).

Entries with the same shape_set_key share the same shape table from
``shapes.SHAPE_SETS``: e.g. moai_alg3 and moai_alg4 both iterate over
the moai shape list.
"""
