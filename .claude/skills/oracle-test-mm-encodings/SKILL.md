---
name: oracle-test-mm-encodings
description: Run kernel-level oracle tests for the matmul-encoding GPU ports (THOR / MOAI / BMM1 / BMM3). Tests each encoding's plaintext oracle against numpy and its CKKS kernel against the plaintext oracle, across both lattigo and desilo backends. Use after porting a kernel or changing the desilo bindings.
---

# Oracle tests for matmul-encoding kernels

Mirrors the pattern in `tests/oracle/` that validated the desilo backend. Three nested checks per encoding, parameterized over backends so kernel bugs and binding bugs separate cleanly.

## Layout

```
tests/oracle/matmul_encodings/
  conftest.py           # backend-parameterized scheme fixture (params=["lattigo", "desilo"])
  test_thor.py
  test_moai.py
  test_bmm1.py
  test_bmm3.py
```

`conftest.py` extends `tests/oracle/conftest.py` with a backend-parameterized variant of the `scheme` fixture (model: `tests/oracle/test_bootstrap.py:44`). Reuse `assert_fhe_close`, `encrypt_values`, `decrypt_values` from `tests/oracle/fhe_test_utils.py`.

## Test body template

For each encoding `X`, the test file has three checks:

```python
# 1. Plaintext oracle vs numpy — no crypto, validates encoding math
def test_X_plaintext_oracle():
    A = np.random.randn(n, m); B = np.random.randn(m, p)
    got = matmul_encodings.plaintext.X_plain(A, B)   # port of X_plain.go
    assert np.allclose(got, A @ B, atol=1e-10)

# 2. CKKS kernel decrypt vs plaintext oracle — runs on each backend
def test_X_ciphertext_matches_plaintext(scheme):     # backend-parameterized
    A = np.random.randn(n, m); B = np.random.randn(m, p)
    expected = matmul_encodings.plaintext.X_plain(A, B)
    out_ct = matmul_encodings.kernels.X_cipher(scheme, A, B)
    result = decrypt_values(scheme, out_ct)
    assert_fhe_close(result, expected, atol=1e-1,
                     msg=f"X on {scheme.config['orion']['backend']}")

# 3. (Optional) Cross-encoding equivalence — Bicycle vs THOR on same A,B → same matrix
```

The `params=["lattigo", "desilo"]` fixture runs check 2 twice; identical pass = both backends agree; lattigo-pass + desilo-fail = bug is in the desilo binding work (see `project-desilo-optimization-gaps` memory).

## Tolerance discipline

`assert_fhe_close` defaults to `atol=1e-2`. CKKS noise grows with multiplicative depth and rotation count, so:

- THOR / MOAI BSGS kernels: start `atol=1e-1`, widen to `5e-1` if noise demands.
- BMM3 (deep, many LongRot calls): may need `atol=1.0`. Log the chosen tolerance + reason in `tests/oracle/matmul_encodings/assumptions.md` (mirroring `tests/oracle/assumptions.md`).

If a test fails because of noise (not a logic bug), the `assert_fhe_close` error message shows `Max error: X at index Y` — use that to decide if it's slot-local noise (acceptable, widen tolerance) or systematic offset (likely an encoding bug).

## Running

```bash
# All matmul-encoding oracles, both backends:
pytest tests/oracle/matmul_encodings/ -v

# Single encoding, single backend:
pytest tests/oracle/matmul_encodings/test_thor.py -v -k "lattigo"

# Just plaintext oracles (no crypto, fast):
pytest tests/oracle/matmul_encodings/ -v -k "plaintext"
```

## When to use this skill

- After porting a new kernel from Negar's `matmul-encoding-material/MatMult/matmult/*.go`.
- After ANY change to `orion/backend/desilo/bindings.py` (the binding work for `MulNoRelin`, `Relinearize`, hoisted `RotateBatch`).
- Before running the benchmark harness for paper numbers.
- Pair with `/example-test-mm-encodings` for full coverage — oracle tests catch kernel-local bugs; example tests catch integration regressions.

## Cross-checking against Negar's Go runner (paper-grade)

When a numeric result is going into the paper, also run the equivalent Go runner on the same seed at `matmul-encoding-material/MatMult/matmult/` (`go run . -verify`) and confirm the decrypted matrices agree to CKKS precision. This catches backend bugs that pass numpy-oracle checks but disagree with the reference Lattigo build.
