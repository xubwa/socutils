# Spinor ADC — status

Status of the two-component (j-adapted, Kramers-unrestricted) post-HF code in
`socutils`, built on a non-relativistic / X2C `SpinorSCF` reference.

Branch: `claude/stoic-maxwell-996i5j`.

## What exists

| module | method | solver | status |
|--------|--------|--------|--------|
| `socutils/mp/spinor_mp2.py` | spinor MP2 | direct | done, exact |
| `socutils/adc/spinor_adc.py` | IP-ADC(2) | Davidson matvec | done, exact |
| | EA-ADC(2) | Davidson matvec | done, exact |
| | EE-ADC(2) | Davidson matvec | done, exact |
| | IP-ADC(2)-x | Davidson matvec | done, exact |
| | EA-ADC(2)-x | Davidson matvec | done, exact |
| | EE-ADC(2)-x | Davidson matvec | done, exact |
| `test_spinor_harness.py` | two-gate validation | — | MP2/IP/EA/EE/IP-x/EA-x/EE-x green |

Not done: ADC(3), spectroscopic factors / transition moments, CVS,
G0W0 (xfail in the harness), Kramers-symmetry exploitation.

## Validation

All quantities are checked two ways (see `test_spinor_harness.py`):

* **Gate 1 — physical correctness vs PySCF** on a non-relativistic
  Hamiltonian (Ne, HF, H2O / cc-pVDZ):
  - MP2 vs RMP2; IP/EA-ADC(2) vs RADC; EE-ADC(2) and IP-ADC(2)-x vs **UADC**.
  - Agreement to ~1e-6 or better (each spatial root carries its
    Kramers/Ms multiplicity, so compare the unique-value sets).
* **Gate 2 — complex-implementation correctness** (no external reference):
  every observable is invariant under a *legal* complex orbital rotation
  (per-orbital phase + unitary mix within an exactly-degenerate / Kramers
  block). All methods invariant to ~1e-13. This is what catches
  `.T`-vs-`.conj().T` bugs.

Current full harness: **36 passed, 6 xfailed** (xfail = G0W0 only).

## Key facts that were hard-won (do not re-derive blindly)

* **Conjugation / phase gauge.** Holes transform as `ph_i*`. The static
  self-energies and the EE `Lambda` term must use `t2.conj()` (not `t2`) for
  Gate-2 invariance; the real-valued result is unchanged either way, so only
  Gate 2 catches it.
* **EA coupling.** The external particle must sit in the *bra*:
  `<ai||cd>` (not `<cd||ai>`). Equal for real, different phase.
* **EE singles–doubles coupling.** The `ooov` part has the *opposite* sign to
  the `ovvv` part: `+d_ij<ak||bc> -d_ik<aj||bc> +d_ac<jk||ib> -d_ab<jk||ic>`.
* **EE ph-ph 2nd order** = CIS `(e_a-e_i)d + <aj||ib>`, minus the
  occupied/virtual self-energies, **plus** a non-separable term
  `Lambda_{ia,jb} = sum_{me} t_im^ae <jm||be>`. Without `Lambda` the triplets
  are wrong (it is the part the He triplet — which does not couple to doubles —
  pins down).
* **EE reference must be UADC, not RADC.** Spinor EE spans the full spin
  manifold (singlets + all Ms of triplets); RADC gives singlets only.
* **IP-ADC(2)-x reference must be UADC, not RADC.** ADC(2)-x splits the
  doublet/quartet 2h1p satellites; RADC keeps only the doublet ones. (Strict
  ADC(2) keeps the 2h1p block diagonal, so doublet/quartet stay degenerate and
  RADC agrees — which is why IP/EA-ADC(2) validate fine against RADC.)
* **IP-ADC(2)-x 2h1p block** (EOM-IP Hbar(2h1p,2h1p) with t=0):
  `(e_a-e_i-e_j) + 0.5<mn||ij> r2_mna + P(ij)<ma||ei> r2_mje`. Its exact
  diagonal (needed as Davidson preconditioner, else near-degenerate satellites
  are skipped) is `e_a-e_i-e_j + <ij||ij> - <ia||ia> - <ja||ja>`.
* **EA-ADC(2)-x 2p1h block** (particle-hole mirror of IP):
  `(e_a+e_b-e_i) + 0.5<ab||cd> r2_icd - P(ab)<jb||ic> r2_jac`, with
  `<jb||ic> = -voov[b,j,i,c]`.  Exact diagonal (preconditioner):
  `e_a+e_b-e_i + <ab||ab> - <ia||ia> - <ib||ib>`. Validated against UADC by
  the sorted spectrum (with multiplicity); the unique-set test fails spuriously
  because deep satellites depend on how many roots each side requests.
* **EE-ADC(2)-x 2p2h block** (pp + hh ladders + ph ring):
  `(e_a+e_b-e_i-e_j) + 0.5<ab||cd> r2_ijcd + 0.5<kl||ij> r2_klab
   + P(ij)P(ab)<kb||cj> r2_ikac`, with `<kb||cj> = ovvo[k,b,c,j]` and
  `P(ij)P(ab) f = f - f(i<->j) - f(a<->b) + f(both)`.  Exact diagonal:
  `... + <ab||ab> + <ij||ij> - <ia||ia> - <ib||ib> - <ja||ja> - <jb||jb>`.
  Compared as UNIQUE values vs UADC (spinor carries the extra Ms triplet
  components), coarse-rounded like IP-x.
* **ADC(2)-x: every DISTINCT excitation energy matches pyscf to ~1e-8**
  (IP-x 8/8, EA-x 8/8, EE-x 10/10 leading distinct energies on HF/cc-pVDZ,
  incl. the deep 2p1h/2h1p satellites).  The only difference is degeneracy
  *multiplicity*: the spinor (Kramers-unrestricted) carries the full
  Kramers/Ms manifold, so a given energy appears with more copies than in UADC
  on a spin-restricted reference (e.g. EA satellite 1.08468 is 8-fold here vs
  4-fold in UADC) -- the same reason strict EE is compared on unique values.
  Compare DISTINCT energies, never sorted-with-multiplicity (the latter
  misaligns once multiplicities differ, and truncating at a fixed nroots drops
  the higher distinct values of whichever side has larger multiplicity).
  The spinor 2p1h block was independently verified == a brute-force
  Slater-Condon CI 2p1h block (1e-13) with exact integrals (1e-15).
* PySCF's `adc` eris are **chemist** `(pq|rs)`. PySCF's `nrr_outcore` with
  `motype='j-spinor'` from `int2e_sph` reproduces `int2e_spinor` MO integrals
  to ~1e-14.
* The repo at `/home/user/pyscf` is older and lacks EE-ADC; the *installed*
  pyscf (2.13.1) has `radc_ee`/`uadc_ee`. Use the installed one as reference.

## Architecture

* `SpinorADC` reads `mo_coeff`/`mo_occ`/`mo_energy` off the mean field at
  construction, so injecting rotated orbitals (`mf.copy()` + overwrite
  `mo_coeff`) needs no re-diagonalisation; canonical `mo_energy` is reused
  verbatim (never recomputed).
* Integrals: `_SpinorADCERIs` builds antisymmetrised physicist blocks
  `<pq||rs>` **lazily, block by block**, via the C s4 driver
  `socutils.lib.ao2mo.nrr_fast` from `int2e_sph`.
  The full `n2c^4` tensor and the all-virtual `vvvv` block are never formed
  (no current method needs `vvvv`). Unique chemist sub-blocks are cached and
  `(pq|rs)=(rs|pq)` is reused by transpose.
* Solver: pyscf's ``lib.davidson_nosym1`` (``pick_real_eigs``, the solver
  pyscf's own ADC/EOM use) drives all IP/EA/EE methods via ``_davidson``;
  small satellite spaces fall back to a dense probe+eigh.  (The earlier
  hand-rolled block Davidson under-converged / missed near-degenerate
  satellites -- replacing it tightened EE-ADC(2)-x agreement with pyscf from
  ~1e-4 to ~1e-6.)  Doubles vectors are stored in the restricted
  antisymmetric basis (``i<j``, ``a<b``); EA/EE matvecs pack the antisymmetric
  pair to halve the dominant ``O(no^2 nv^3)`` contractions.

## Performance (H2O / cc-pVTZ, n2c=116, nv=106) after optimisation

| method | build+solve | earlier |
|--------|-------------|---------|
| IP | ~0.6 s | ~1.1 s |
| EA | ~2.9 s | ~12 s |
| EE | ~13 s  | ~34 s |

Optimisation history: dense O(N^9) -> Davidson O(N^5); full-tensor ->
block-selective lazy integrals (no `vvvv`); 10 -> 6 transforms via exchange
symmetry; antisymmetry packing in EA/EE matvecs; **C s4 AO->MO driver
(`socutils.lib.ao2mo.nrr_fast`, ~4x on the virtual blocks); matvec GEMMs via
explicit reshape+dot (lib.einsum hid a 47 MB transpose-copy per matvec, 8-23x
on the dominant 2p1h/2p2h couplings); precomputed conj'd blocks; preallocated
Davidson subspace with incremental conj(V).**

## Current bottlenecks (next work, ranked)

EE (~13 s) is now balanced: integral build ~4.6 s, matvec ~6 s, Davidson
overhead ~2 s. No single dominant lever remains. Next, ranked:

1. **Share the AO half-transform across EE blocks.** `voov`/`vovv` both need
   the `(v·|··)` bra half-transform; computing it once (or fusing the block
   builds) would cut the ~4.6 s integral build. Now the top lever.
2. **Fewer Davidson iterations** (still ~60-70 matvecs) via a better
   preconditioner / a deflated restart.
3. **Kramers (time-reversal) symmetry:** a further ~2x, deliberately skipped.

## Running the tests

The environment needs `numpy scipy h5py pyscf pytest`; `socutils` on the path.

    cd <dir-with-installed-pyscf>   # so `import pyscf` is the installed 2.13.1
    python -m pytest test_spinor_harness.py --import-mode=importlib -q
    python -m pytest test_spinor_harness.py -k mp2 -q     # subset
