"""Oracle tests for the plaintext matmul-encoding ports.

For each of the four modules (bmm1, bmm3, moai, thor), generate small random
matrices with a fixed RNG, run the plaintext kernel, and compare against the
naive numpy reference matmul (or the variant matmul each scheme computes).

These are oracles that the ciphertext kernels' correctness tests depend on,
so the tolerance is tight (atol=1e-10 since this is pure float64 math).
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.plaintext.bmm1_plain import (
    bicyclic_decode,
    bicyclic_encode,
    bmm1_matmul_plain,
    bmm1_plain,
    encode_blocks,
    repeat_vector,
    rotate_vec,
)
from benchmarks.matmul_encodings.plaintext.bmm3_plain import (
    bmm3_matmul_plain,
    bmm3_plain,
    break_into_chunks,
    long_rot_plain,
    smallest_r,
)
from benchmarks.matmul_encodings.plaintext.moai_plain import (
    interleaved_column_pack,
    interleaved_column_unpack,
    interleaved_diag_pack,
    interleaved_diag_unpack,
    moai_col_col_bsgs,
    moai_col_col_naive,
    moai_diag_col_bsgs,
)
from benchmarks.matmul_encodings.plaintext.op_counts import OpCounts
from benchmarks.matmul_encodings.plaintext.rowenc_plain import (
    row_pack,
    row_unpack,
    rowenc_matmul_plain,
    theoretical_rowenc_costs,
)
from benchmarks.matmul_encodings.plaintext.thor_plain import (
    build_masks_for_kernel,
    decode_batched,
    encode_batched,
    replication,
    thor_cc_matmul_plain,
)


# ---------------------------------------------------------------------------
# OpCounts sanity
# ---------------------------------------------------------------------------


def test_op_counts_add():
    a = OpCounts(rotations=1, ct_ct_muls=2, ct_pt_muls=3)
    b = OpCounts(rotations=10, ct_ct_muls=20, ct_pt_muls=30)
    c = a + b
    assert c.rotations == 11
    assert c.ct_ct_muls == 22
    assert c.ct_pt_muls == 33
    # Originals unchanged.
    assert a.rotations == 1 and b.rotations == 10


# ---------------------------------------------------------------------------
# Rotation direction sanity: Go rotateVec(v, k)[i] = v[(i+k) mod n].
# So rotate_vec([0,1,2,3,4], 1) should be [1,2,3,4,0] -- the element at
# position 1 has moved to position 0 (left rotation by 1).
# ---------------------------------------------------------------------------


def test_rotate_vec_direction():
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    got = rotate_vec(v, 1)
    np.testing.assert_array_equal(got, np.array([1.0, 2.0, 3.0, 4.0, 0.0]))

    got = rotate_vec(v, 2)
    np.testing.assert_array_equal(got, np.array([2.0, 3.0, 4.0, 0.0, 1.0]))

    # k = 0 is identity.
    np.testing.assert_array_equal(rotate_vec(v, 0), v)

    # Negative k wraps correctly.
    got = rotate_vec(v, -1)
    np.testing.assert_array_equal(got, np.array([4.0, 0.0, 1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# BMM-I
# ---------------------------------------------------------------------------


def test_bicyclic_encode_decode_roundtrip_coprime():
    """For pairwise-coprime n, m, p, encode then decode is identity on the
    encoded slot vector (and recovers the original matrix when n*m = n*p)."""
    rng = np.random.default_rng(seed=42)
    n, m = 5, 7
    M = rng.standard_normal((n, m))
    enc = bicyclic_encode(M)
    assert enc.shape == (n * m,)
    # Decode back at (n, m) should recover the matrix exactly when n,m coprime.
    rec = bicyclic_decode(enc, n, m)
    np.testing.assert_allclose(rec, M, atol=1e-12)


def test_bmm1_plain_single_block_coprime():
    """Single-block BMM-I with pairwise-coprime (n, m, p)."""
    rng = np.random.default_rng(seed=42)
    n, m, p = 5, 7, 11
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    a_enc = bicyclic_encode(A)
    b_enc = bicyclic_encode(B)

    counts = OpCounts()
    c_vec = bmm1_plain(a_enc, b_enc, n, m, p, counts)
    assert c_vec.shape == (n * p,)

    C_got = bicyclic_decode(c_vec, n, p)
    C_ref = A @ B
    np.testing.assert_allclose(C_got, C_ref, atol=1e-10)

    # Op counts: m-1 rotations on A, m-1 on B (i=0 has rot=0), and m mults.
    # The Go code increments rotations only when rot_a / rot_b != 0.
    assert counts.ct_ct_muls == m


def test_bmm1_matmul_plain_block_coprime():
    """Block BMM-I where the matrix dims are multiples of the (coprime) block
    dims. Use one block (degenerate case) and two blocks per axis."""
    rng = np.random.default_rng(seed=42)
    s_n, s_m, s_p = 5, 7, 11  # pairwise coprime
    # Try (N, M, P) = (s_n, s_m, s_p) (single block) and 2x in every axis.
    for mult in (1, 2):
        N, M, P = mult * s_n, mult * s_m, mult * s_p
        A = rng.standard_normal((N, M))
        B = rng.standard_normal((M, P))

        a_enc, b_enc = encode_blocks(A, B, s_n, s_m, s_p)
        C_got, counts = bmm1_matmul_plain(a_enc, b_enc, N, M, P, s_n, s_m, s_p)
        C_ref = A @ B
        np.testing.assert_allclose(C_got, C_ref, atol=1e-10)
        # Sanity: some rotations + mults happened.
        assert counts.ct_ct_muls > 0


def test_repeat_vector():
    v = np.array([1.0, 2.0, 3.0])
    out = repeat_vector(v, 3)
    np.testing.assert_array_equal(out, np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]))


# ---------------------------------------------------------------------------
# BMM-III helpers (no plaintext kernel exists; verify the helpers themselves)
# ---------------------------------------------------------------------------


def test_smallest_r_definition():
    """smallest_r(n, m, p) = smallest r >= 1 with (r*m - n) >= 0 and divisible
    by p. Only defined when gcd(m, p) divides n (BMM-III precondition);
    otherwise the function loops forever. The published algorithm uses
    pairwise-coprime block dimensions, so we test on those."""
    from math import gcd

    cases = []
    for n in range(1, 12):
        for m in range(1, 12):
            for p in range(1, 12):
                if gcd(m, p) == 1:  # always solvable when m and p are coprime
                    cases.append((n, m, p))

    for n, m, p in cases:
        r = smallest_r(n, m, p)
        s = r * m - n
        assert s >= 0, (n, m, p, r, s)
        assert s % p == 0, (n, m, p, r, s)
        # Confirm no smaller r works.
        for r_prime in range(1, r):
            s_prime = r_prime * m - n
            assert (s_prime < 0) or (s_prime % p != 0), (n, m, p, r, r_prime)


def test_break_into_chunks_shapes_and_content():
    """break_into_chunks tiles cyclically when enc_len < output_len + n_he and
    splits into ceil-many length-n_he chunks (last zero-padded)."""
    enc = np.arange(1, 11, dtype=np.float64)  # length 10
    n_he = 4
    output_len = 10
    chunks = break_into_chunks(enc, len(enc), output_len, n_he)

    # All chunks must be length n_he.
    for c in chunks:
        assert c.shape == (n_he,)

    # Concatenated, the first len(enc) values must be the original (cyclic
    # tile of) enc -- regardless of whether tiling kicked in.
    needed = output_len + n_he
    expected_full = np.tile(enc, max(1, ((needed // len(enc)) + 2)))
    flat = np.concatenate(chunks)
    np.testing.assert_array_equal(flat[: len(enc)], enc)
    # The tiled content (where present) should match the tile.
    L = min(len(flat), len(expected_full))
    # Last chunk may be zero-padded -- only check up through the last full
    # non-padded position.
    non_padded_len = (
        (len(chunks) - 1) * n_he
        + min(n_he, len(np.tile(enc, ((needed // len(enc)) + 2))) - (len(chunks) - 1) * n_he)
    )
    np.testing.assert_array_equal(flat[:non_padded_len], expected_full[:non_padded_len])


def test_break_into_chunks_no_tiling_needed():
    """When enc_len >= output_len + n_he no tiling occurs."""
    enc = np.arange(100, dtype=np.float64)
    n_he = 4
    output_len = 10
    chunks = break_into_chunks(enc, len(enc), output_len, n_he)
    flat = np.concatenate(chunks)
    np.testing.assert_array_equal(flat[: len(enc)], enc)


# ---------------------------------------------------------------------------
# BMM-III LongRot simulator -- the chunked rotation MUST equal a plain
# logical-vector rotation followed by chunking. This is the invariant that
# pins the LongRot algorithm to its specification, independent of any
# matmul wiring.
# ---------------------------------------------------------------------------


def _logical_long_rot_reference(
    chunks: list[np.ndarray], rot: int, output_len: int, enc_len: int, n_he: int
) -> np.ndarray:
    """The simplest possible LongRot reference: reconstruct the full logical
    encoded vector, rotate it by `rot`, take the first `output_len` slots."""
    work = np.concatenate(chunks)
    enc = work[:enc_len]
    needed = output_len + n_he
    if enc_len < needed:
        reps = ((needed + enc_len - 1) // enc_len) + 1
        enc = np.tile(enc, reps)
    rotated = np.concatenate([enc[rot % len(enc):], enc[: rot % len(enc)]])
    return rotated[:output_len]


@pytest.mark.parametrize(
    "enc_len, output_len, n_he, rot",
    [
        # Single-chunk case: enc_len <= n_he, identity rotation.
        (12, 12, 16, 0),
        (12, 12, 16, 5),
        # Multi-chunk, fits in a few tiles, mid-chunk rotation.
        (35, 55, 16, 0),    # rot=0 baseline
        (35, 55, 16, 7),    # v_tmp != 0, normal stitch
        (35, 55, 16, 16),   # v_tmp == 0, u-shift only
        (35, 55, 16, 17),   # v_tmp != 0 across chunk boundary
        (35, 55, 16, 30),   # large rot -- exercises Step 1 advance + 3-way
        # Bigger enc_len, no tiling, several output chunks.
        (77, 55, 16, 0),
        (77, 55, 16, 13),
        (77, 55, 16, 33),
        # Mixed: enc_len % n_he == 0 (no last-chunk padding).
        (32, 32, 16, 5),
        (32, 32, 16, 16),
    ],
)
def test_long_rot_plain_matches_logical_rotation(
    enc_len, output_len, n_he, rot
):
    """`long_rot_plain` must reproduce the logical vector rotation across
    every Step-1/2/3 branch we exercise."""
    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal(enc_len)
    chunks = break_into_chunks(raw, enc_len, output_len, n_he)

    out_chunks = long_rot_plain(chunks, rot, output_len, enc_len, n_he)
    got = np.concatenate(out_chunks)[:output_len]
    want = _logical_long_rot_reference(chunks, rot, output_len, enc_len, n_he)

    np.testing.assert_allclose(got, want, atol=1e-12)


# ---------------------------------------------------------------------------
# BMM-III plaintext kernel -- the encoding-aware matmul oracle that the
# CKKS BMM-III kernel will be checked against. Single-chunk (n*m,m*p<=n_he)
# AND multi-chunk shapes both have to land within float64 epsilon.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, m, p, n_he",
    [
        # Single-chunk fallback: n*m, m*p <= n_he (BMM-III reduces to BMM-I).
        (5, 7, 11, 128),
        # Multi-chunk: n*m and m*p exceed n_he -- the "real" BMM-III path.
        (5, 7, 11, 16),
        (4, 5, 7, 16),
        (8, 9, 11, 32),
        # Coprime triple with all three dims > 1 chunk after tiling.
        (7, 11, 13, 32),
    ],
)
def test_bmm3_matmul_plain_matches_numpy(n, m, p, n_he):
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    C_got, counts = bmm3_matmul_plain(A, B, n_he)
    C_ref = A @ B

    np.testing.assert_allclose(C_got, C_ref, atol=1e-10)

    # Sanity: the kernel did the right amount of multiplicative work.
    stop = (n * p + n_he - 1) // n_he
    assert counts.ct_ct_muls == m * stop


def test_bmm3_plain_op_counts_grow_as_chunks_shrink():
    """Smaller n_he -> more chunks -> strictly more LongRot work.

    break_into_chunks always tiles to support the worst-case rotation
    window, so genuine single-chunk fallback (w==1, rotations==0) is not
    reachable with the algorithm's encode_len + n_he window. What we can
    verify is monotonicity in the chunk count.
    """
    rng = np.random.default_rng(seed=7)
    n, m, p = 5, 7, 11
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    _, big_chunks = bmm3_matmul_plain(A, B, n_he=128)
    _, small_chunks = bmm3_matmul_plain(A, B, n_he=16)

    assert small_chunks.rotations  > big_chunks.rotations
    assert small_chunks.ct_pt_muls > big_chunks.ct_pt_muls
    # ct_ct_muls counts m * stop and stop = ceil(n*p / n_he); shrinking
    # n_he shouldn't decrease it either.
    assert small_chunks.ct_ct_muls >= big_chunks.ct_ct_muls


# ---------------------------------------------------------------------------
# MOAI -- pack/unpack roundtrip and matmul correctness.
#
# Note: MOAI Algorithm 3 (Col x Col -> Diag) requires n_he divisible by m,
# with n_batch = n_he / m matrices packed in lock-step. The Go runner uses
# n_he = 4096 with m in {128, 256, ...}. For tests we use n_he = 64 with
# m = 8 (so n_batch = 8) which is a power-of-two case.
# ---------------------------------------------------------------------------


def test_moai_column_pack_unpack_roundtrip():
    rng = np.random.default_rng(seed=42)
    m, d, n_batch = 8, 4, 3
    n_he = m * n_batch  # exactly divisible

    Xs = rng.standard_normal((n_batch, m, d))
    packed = interleaved_column_pack(Xs, n_he)
    rec = interleaved_column_unpack(packed, m, n_batch)
    np.testing.assert_allclose(rec, Xs, atol=1e-12)


def test_moai_diag_pack_unpack_roundtrip():
    rng = np.random.default_rng(seed=42)
    m, n_batch = 8, 3
    n_he = m * n_batch

    Cs = rng.standard_normal((n_batch, m, m))
    packed = interleaved_diag_pack(Cs, n_he)
    rec = interleaved_diag_unpack(packed, m, n_batch)
    np.testing.assert_allclose(rec, Cs, atol=1e-12)


def test_moai_col_col_naive_matches_qkt():
    """Algorithm 3 (Col x Col -> Diag) computes Q . K^T over a batch."""
    rng = np.random.default_rng(seed=42)
    m, d_prime, n_batch = 8, 4, 2
    n_he = m * n_batch
    rot_stride = n_batch  # keeps the interleaved packing intact

    Qs = rng.standard_normal((n_batch, m, d_prime))
    Ks = rng.standard_normal((n_batch, m, d_prime))

    enc_q = interleaved_column_pack(Qs, n_he)
    enc_k = interleaved_column_pack(Ks, n_he)

    enc_out, counts = moai_col_col_naive(enc_q, enc_k, m, d_prime, rot_stride)
    got = interleaved_diag_unpack(enc_out, m, n_batch)

    # Reference: Q . K^T per batch element.
    ref = np.matmul(Qs, np.transpose(Ks, axes=(0, 2, 1)))
    np.testing.assert_allclose(got, ref, atol=1e-10)
    # Naive theoretical: (m - 1) * d_prime rotations, m * d_prime mults.
    assert counts.rotations == (m - 1) * d_prime
    assert counts.ct_ct_muls == m * d_prime


def test_moai_col_col_bsgs_matches_qkt():
    """BSGS variant of Algorithm 3 must produce the same result as naive."""
    rng = np.random.default_rng(seed=42)
    m, d_prime, n_batch = 8, 4, 2
    n_he = m * n_batch
    rot_stride = n_batch

    Qs = rng.standard_normal((n_batch, m, d_prime))
    Ks = rng.standard_normal((n_batch, m, d_prime))

    enc_q = interleaved_column_pack(Qs, n_he)
    enc_k = interleaved_column_pack(Ks, n_he)

    enc_out, _counts = moai_col_col_bsgs(enc_q, enc_k, m, d_prime, rot_stride)
    got = interleaved_diag_unpack(enc_out, m, n_batch)

    ref = np.matmul(Qs, np.transpose(Ks, axes=(0, 2, 1)))
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_moai_diag_col_bsgs_matches_cv():
    """Algorithm 4 (Diag x Col -> Col) computes C . V over a batch.

    C is a square (m, m) matrix in diag-pack; V is (m, d_prime) in col-pack;
    output is (m, d_prime) in col-pack.
    """
    rng = np.random.default_rng(seed=42)
    m, d_prime, n_batch = 8, 4, 2
    n_he = m * n_batch
    rot_stride = n_batch

    Cs = rng.standard_normal((n_batch, m, m))
    Vs = rng.standard_normal((n_batch, m, d_prime))

    enc_c = interleaved_diag_pack(Cs, n_he)
    enc_v = interleaved_column_pack(Vs, n_he)

    enc_out, _counts = moai_diag_col_bsgs(enc_c, enc_v, m, d_prime, rot_stride)
    got = interleaved_column_unpack(enc_out, m, n_batch)

    ref = np.matmul(Cs, Vs)
    np.testing.assert_allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# THOR -- encode/decode roundtrip and Algorithm 2 correctness.
#
# THOR computes A . B at the layout (m, n) x (n, n) -> (m, n). For tests we
# use the smallest configuration from the Go runner: d=n=2, H=2, s=8 (which
# gives c = s/(n*H) = 2, m_c = d/c = 1).
# ---------------------------------------------------------------------------


def test_thor_encode_decode_roundtrip():
    rng = np.random.default_rng(seed=42)
    d, n, H = 4, 4, 2
    c = 2
    s = c * n * H  # = 16

    Ms = rng.standard_normal((H, d, n))
    packed = encode_batched(Ms, c, H)
    assert packed.shape == (d // c, s)
    rec = decode_batched(packed, d, n, c, H)
    np.testing.assert_allclose(rec, Ms, atol=1e-12)


def test_thor_cc_matmul_tiny():
    """Algorithm 2 on a tiny config (d=n=2, H=2, s=8) from the Go runner."""
    rng = np.random.default_rng(seed=42)
    d, n, H, s = 2, 2, 2, 8
    c = s // (n * H)  # = 2
    assert c == 2
    assert d % c == 0 and n % c == 0

    As = rng.standard_normal((H, d, n))
    Bs = rng.standard_normal((H, n, n))
    # Reference: A . B per head.
    ref = np.matmul(As, Bs)

    a_packed = encode_batched(As, c, H)
    b_packed = encode_batched(Bs, c, H)
    b_rep = replication(b_packed, n, c, H)

    masks = build_masks_for_kernel(c, n, H, s)
    c_vecs, _counts = thor_cc_matmul_plain(a_packed, b_rep, masks, d, n, H, s)

    got = decode_batched(c_vecs, d, n, c, H)
    np.testing.assert_allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Row-encoding -- the simplest of the four. Constraint: n must be a power of 2
# (the log2(n) replication steps need integer shift amounts). Tested at
# n in {2, 4, 8} which exercises the full algorithm at small enough sizes
# that everything stays in float64-tight tolerance.
# ---------------------------------------------------------------------------


def test_rowenc_pack_unpack_roundtrip():
    rng = np.random.default_rng(seed=42)
    n = 8
    M = rng.standard_normal((n, n))
    np.testing.assert_allclose(row_unpack(row_pack(M), n), M, atol=1e-12)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_rowenc_matmul_plain_matches_numpy(n):
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))

    C_got, counts = rowenc_matmul_plain(A, B)
    C_ref = A @ B
    np.testing.assert_allclose(C_got, C_ref, atol=1e-10)

    # Sanity: the kernel did n ct*ct multiplies and 2n ct*pt extractions.
    assert counts.ct_ct_muls == n
    assert counts.ct_pt_muls == 2 * n


def test_rowenc_op_counts_match_theoretical():
    """Multiplicative op counts match the closed-form formula exactly.

    Rotation counts are <= the theoretical bound: the runtime counter
    skips identity rotations (the i=0 initial-align in replicate_row, and
    the i=0 diagonal-align), saving exactly 2 rotations per matmul.
    """
    for n in (2, 4, 8):
        n_rot_th, n_pmult_th, n_mult_th, _, _ = theoretical_rowenc_costs(n)

        rng = np.random.default_rng(seed=7)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        _, counts = rowenc_matmul_plain(A, B)

        assert counts.ct_pt_muls == n_pmult_th, n
        assert counts.ct_ct_muls == n_mult_th, n
        # Runtime counter skips the two identity rotations at i=0
        # (replicate_row initial align, diagonal align). Everything else
        # matches the theoretical bound.
        assert counts.rotations == n_rot_th - 2, (
            n, counts.rotations, n_rot_th
        )


def test_thor_cc_matmul_bigger():
    """Algorithm 2 at a larger but still test-sized config: d=n=4, H=2, s=16
    (c = 2). Exercises the rotation/mask routing more thoroughly."""
    rng = np.random.default_rng(seed=42)
    d, n, H = 4, 4, 2
    c = 2
    s = c * n * H  # = 16
    assert d % c == 0 and n % c == 0

    As = rng.standard_normal((H, d, n))
    Bs = rng.standard_normal((H, n, n))
    ref = np.matmul(As, Bs)

    a_packed = encode_batched(As, c, H)
    b_packed = encode_batched(Bs, c, H)
    b_rep = replication(b_packed, n, c, H)

    masks = build_masks_for_kernel(c, n, H, s)
    c_vecs, _counts = thor_cc_matmul_plain(a_packed, b_rep, masks, d, n, H, s)

    got = decode_batched(c_vecs, d, n, c, H)
    np.testing.assert_allclose(got, ref, atol=1e-10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
