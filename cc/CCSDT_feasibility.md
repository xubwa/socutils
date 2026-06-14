# Feasibility: full CCSDT (iterative triples) for the two-component spinor CC

Assessment of implementing **full, iterative CCSDT** (solving for the T3
amplitudes self-consistently, not the perturbative `(T)`) in socutils' `cc/`
module, following pyscf's CCSDT.

## TL;DR

- **A reference, in-core spinor CCSDT is moderate difficulty (~1 week).** The
  framework, integrals, intermediates, and — crucially — the machinery that
  builds the connected triples from T1/T2 already exist here. What is new is a
  bounded set of T3↔T3 and T3→T1/T2 contraction terms, a T3 amplitude array
  with its denominator, and hooking T3 into the DIIS/iteration loop.
- **The binding constraint is memory, not algebra.** A fully-antisymmetric
  complex T3 is `O(n_o^3 n_v^3)` and the iteration is `~O(n_o^3 n_v^5)`. With
  spinor (two-component) orbitals `n_o`/`n_v` count spin-orbitals (2× spatial)
  and amplitudes are complex (≈2× memory, ≈4× flops vs. a real spin-orbital
  code). So a naive in-core version is only usable for small active spaces.
- **An *efficient* CCSDT (packed/blocked/DF, à la pyscf's `rccsdt.py`) is a
  multi-week effort** — that is where pyscf spends 1600–3000 lines.

## What "following pyscf" means here

pyscf ships full triples as:

| module | role | lines |
|---|---|---|
| `cc/rccsdt_highm.py`, `cc/uccsdt_highm.py` | **reference**: store the full T3, residual as plain `einsum` (`compute_r3`, `r1r2_add_t3_`) | ~260 / ~380 |
| `cc/rccsdt.py`, `cc/uccsdt.py` | production: triangular packing of T3 (perm. symmetry), spin-summation patterns, blocking, out-of-core | ~1600 / ~2980 |

There is **no `gccsdt`** (generalized / spin-orbital). But socutils' CC *is*
generalized: `ZCCSD(gccsd.GCCSD)` with a single complex, fully-antisymmetric
`t2[i,j,a,b]`. The spin-orbital CCSDT amplitude is likewise a **single**
fully-antisymmetric `t3[i,j,k,a,b,c]` — i.e. the `aaa`-like block of
`uccsdt_highm` but spanning *all* spinors, with **no spin cases to sum**. So
the natural template is `uccsdt_highm.compute_r3` collapsed to one antisymmetric
tensor (equivalently the textbook spin-orbital CCSDT equations,
Scuseria–Schaefer / Crawford–Schaefer / Hirata).

## What already exists and is directly reusable

- **Base class + driver**: `ZCCSD(gccsd.GCCSD)` (cc/zccsd.py:88) inherits
  pyscf's CCSD kernel/DIIS. T3 just needs to join the amplitude vector.
- **Integrals / ERIS**: `_PhysicistsERIs` with antisymmetrized
  `oooo/ooov/oovv/ovov/ovvo/ovvv/vvvv` (cc/zccsd.py:148, `_make_eris_outcore`
  zccsd.py:213). The in-core variant already stores **vvvv** and **ovvv**,
  which the T3 ladder/ring terms need.
- **Intermediates** (cc/gintermediates.py): `cc_Foo/Fov/Fvv`,
  `cc_Woooo/Wvvvv/Wovvo`, plus `Wooov/Wvovv/Wovoo` (lines 195–236). The CCSDT
  residual reuses essentially all of these; only a couple need the small
  "λ→t" (no-disconnected) variants.
- **The connected triples builder is already written.** `gccsd_t.get_wv_abc`
  (cc/gccsd_t.py:54) / `get_wv_ijk` (gccsd_t.py:83) form, per occupied or
  virtual triple,
  `w = <bc||ei> t2 - <ma||jk> t2` (this is exactly the T2-driving part of the
  T3 residual, `W_vvvo·t2 + W_vooo·t2`) and the disconnected `v` from T1/Fock.
  Full CCSDT = **this `w`, plus the T3↔T3 terms, solved iteratively** instead of
  used once perturbatively.

## What is new (the actual work)

1. **T3 amplitude + denominator + storage.** `t3` of shape
   `(n_o,n_o,n_o,n_v,n_v,n_v)` complex; `D_ijkabc = f_ii+f_jj+f_kk -
   f_aa-f_bb-f_cc`; antisymmetry maintained (or packed).
2. **T3 residual `R3` (the T3↔T3 terms).** Beyond the reusable `w`:
   - `+ P(a) F_vv·t3 − P(i) F_oo·t3`
   - `+ ½ W_vvvv·t3` (particle–particle ladder, needs `vvvv`)  ← rate-limiting `~o^3 v^5`
   - `+ ½ W_oooo·t3`
   - `+ P(i)P(a) W_voov·t3` (ring)
   each with the cyclic antisymmetrizers `P(i/j/k)`, `P(a/b/c)`. ~10–15 terms.
3. **T3 → T1/T2 coupling** (`r1r2_add_t3_`, uccsdt_highm.py:81): T2 residual
   gains `<ov||vv>`/`<oo||ov>` contractions with T3; T1 gains `<oo||vv>·t3`.
   These promote CCSD's update to CCSDT.
4. **Wire T3 into the amplitude vector** for energy/DIIS/convergence
   (`amplitudes_to_vector`/`vector_to_amplitudes`, `update_amps`, `energy`).

Note the CCSDT *correlation energy* expression is unchanged from CCSD (T3 has no
direct energy term in canonical HF); T3 enters energy only through its effect on
T1/T2.

## Storage / scaling reality (spinor, complex)

- Memory `16 · n_o^3 n_v^3` bytes for T3 alone (complex128). With spinor
  (two-component) orbitals `n_o`/`n_v` count spin-orbitals (≈2× spatial):
  `n_o=8, n_v=40 → 0.5 GB`; `n_o=12, n_v=60 → 6 GB`;
  `n_o=16, n_v=120 → 110 GB`; `n_o=20, n_v=200 → ~1 TB`. In-core is for
  **small** systems only; it grows as the 6th power of system size.
- Flops `~O(n_o^3 n_v^5)` per iteration (the `W_vvvv·t3` ladder), ×~4 for
  complex. This is the genuine CCSDT wall.

## Recommended path

1. **Phase 1 — reference, in-core, full-T3 (validate first).** Build on
   `cc/zccsd.py` (it already has `vvvv`/`ovvv`). Transcribe the spin-orbital
   CCSDT residual (template: `uccsdt_highm` collapsed to one antisymmetric
   tensor), reusing `gintermediates` and `gccsd_t`'s `w`. Validate the energy on
   a small case against a known spin-orbital CCSDT (e.g. a non-relativistic
   GHF/GCCSDT reference, or pyscf `uccsdt_highm` on a closed-shell mol mapped to
   spin-orbitals). Target: correctness, not speed. **~1 week.**
2. **Phase 2 — make it usable.** Either (a) triangular packing of the
   antisymmetric T3 (the `tri2block`/permutation machinery is the bulk of
   pyscf's `rccsdt.py`), and/or (b) a **DF/Cholesky** residual reusing
   `cc/chol_zccsd.py` + `chol_zccsd_t.py` (which already rebuild `<bc||ei>`
   cheaply from `Lvv/Lvo`). The AO-direct path (`zccsd_direct.py`) is **not**
   suitable for the T3 ladder — its `vvvv` is rebuilt `O(n_ao^4)` per block,
   which is prohibitive when needed for every `t3` block. **Multi-week.**

## Risks / sharp edges

- Getting the **antisymmetrizer signs/permutations** exactly right across the
  spin-orbital residual is the most error-prone part; unit-test each term
  against a brute-force `einsum` on full antisymmetric tensors for a tiny case.
- Complex conjugation conventions (this is a *complex* spinor CC — the energy
  uses `v.conj()` already in `gccsd_t`); the T3 residual must keep them
  consistent.
- DIIS over a TB-scale T3 is itself a memory problem (store a few error
  vectors) — another reason Phase 1 stays small.

## Bottom line

Difficulty is **moderate for a correct reference implementation** because the
hard infrastructure (generalized complex CC, integrals incl. `vvvv`/`ovvv`,
intermediates, and the connected-triples builder) is already in place — full
CCSDT is largely "make the existing `(T)` `w` self-consistent and add the
T3↔T3 / T3→T1T2 terms." It becomes a **large effort only when made efficient**
(packing/blocking/DF), which is exactly the part pyscf devotes most of its
`rccsdt.py`/`uccsdt.py` to.
