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
| | CVS-IP-ADC(2) | Davidson matvec | done, exact |
| | IP/EA spectroscopic factors | Dyson amplitudes | done, exact |
| | **IP-ADC(3)** | Davidson matvec | **done, exact** |
| | **EA-ADC(3)** | Davidson matvec | **done, exact** |
| | **EE-ADC(3) ph-ph self-energy** | `_ee_lam3` | **done, validated (full solver coupling pending)** |
| `test_spinor_harness.py` | two-gate validation | — | MP2/IP/EA/EE/IP-x/EA-x/EE-x/CVS/spec/IP+EA-ADC(3) green |

**IP-ADC(3) done & validated** (`ip_adc3`).  The full secular matrix:
* 1h-1h block = `-eps - Sigma^(2) - Sigma^(3)`.  The third-order static
  self-energy `Sigma^(3)` (`_sig_ip3`) is the unified spin-orbital
  (antisymmetrised `<pq||rs>`) transcription of the pyscf UADC `M_ij^(3)`
  amplitude formula -- groups A (`t1^(2)`), B (`t2^(2)`, the Sigma^(2)
  structure), O (oooo hole ladder), L (double ph ladder), S (single ph
  ladder).  Coefficients pinned against `uadc_ip.get_imds` (full matrix,
  ~1e-10 over H2O/HF/N2/CO/LiH/BH; symmetric molecules must be compared in
  the same orbital gauge, since RHF vs UHF resolve degenerate blocks
  differently).
* second-order doubles `t2^(2)` (`_t2_2`) and singles `t1^(2)`
  (`_t1_2_pyscf`), both built on `t2.conj()` so they are covariant under a
  complex rotation.
* second-order 1h<->2h1p coupling: the t2-dressed `<ai||bc>` + `<kl||ia>`
  pieces (sign negative vs the naive translation); 2h1p block = the ADC(2)-x
  ladder+ring (already done).
* **gauge conjugations pinned by Gate-2**: A/B/L formed as `P + P^dag`; the
  coupling uses un-conjugated amplitudes with conjugated interaction integrals
  in the 2h1p->1h direction and its exact adjoint in reverse.  `Sigma^(3)`
  eigenvalues match the Kramers-doubled UADC `Sigma^(3)` exactly; IP-ADC(3)
  roots (main + 2h1p satellites) match UADC IP-ADC(3) to ~5e-7 and are
  rotation-invariant to ~1e-14.

**EA-ADC(3) done & validated** (`ea_adc3`, `_sig_ea3`): the particle-hole
mirror (o<->v) of IP-ADC(3).  `Sigma^(3)_ab` is built from the o<->v dual
amplitudes and dual blocks (vvvv/vvoo/voov/vvvo); the second-order coupling is
the t2-dressed `<kl||ia>` (coeff -1/4) + `<ia||bc>` pieces (exact adjoint in
reverse).  Note the EA self-energy enters the 1p block with the OPPOSITE sign
to IP (`eps - Sigma^(2) + Sigma^(3)`), mirroring the EA<->IP eigenvalue
convention -- pinned numerically.  Matches pyscf UADC EA-ADC(3) roots to ~5e-7,
Hermitian to ~1e-15, rotation-invariant to ~1e-14.

Both charged-excitation ADC(3) self-energies (IP `Sigma^(3)_ij`, EA
`Sigma^(3)_ab`) are validated eigenvalue-for-eigenvalue against the
Kramers-doubled UADC `Sigma^(3)` for GENERAL molecules (HF/Ne/H2O/N2/CO/LiH/BH).

**EE-ADC(3) ph-ph excitation self-energy DONE & validated** (`_ee_lam3`,
`_EE_M3_TERMS`).  The neutral-excitation 1p1h block at third order -- the
ph-channel ISR -- is the complete third-order increment M^(3)_{ia,jb} of the
1p1h secular matrix (third-order IP/EA self-energy legs + genuine ph-ph
coupling).  Derivation/validation this session:
  * The order-2 ISR bug was DOUBLE-COUNTING: wicked's ket-dressing B2k=(V R T2)
    and bra-dressing B2b=(T2d V R) each give the full Hermitian 2nd order, so
    `M2 = B0 + B1(ring) + 0.5*(B2k+B2b)_connected` == pyscf M2 (no metric).
  * M^(3) is a clean half-integer combination of the same B3a([V,T2_2])=0.5,
    B3b([V,T1_2])=1.0, B3c(T2.V.T2)=0.5/1.0 spin-orbital topologies (Hermitised
    m+m^dagger, connected filter).  Found by least-squares over many molecules:
    machine-precision fit, clean coefficients, generalises to held-out molecules.
  * CRITICAL: pyscf's `uadc_ee.get_imds` is NOT the complete ph-ph block -- the
    EE matvec adds t2.t2 ph-ph terms on the fly.  `_ee_lam3` is the COMPLETE
    block (get_imds + matvec terms), validated two ways: (a) reproduces
    get_imds + the matvec t2.t2 increment to ~1e-15 over general molecules; (b)
    its eigenvalues match pyscf's full M11 (all spin sectors, probed) to ~1e-6.
  * Gate-2: complex orbital-rotation covariance (phase + degenerate mixing) to
    ~1e-14.  Conjugations pinned by a full-unitary oracle: t2_2, t1_2 are built
    on t2.conj() (transform as conjugated amplitudes) so they enter
    un-conjugated with the interaction conjugated; t2.t2.V ladders use
    (t2.conj, t2, V.conj).  (Phase-only covariance is NOT sufficient -- the
    legs/channels are only covariant as wholes; this is why the earlier
    per-term phase pinning failed Gate-2 under degenerate-orbital mixing.)

`ee_adc3` (= `ee_adc2(method='adc(3)')`): 1p1h = ADC(2) self-energies + ring +
complete M^(3); 2p2h = ADC(2)-x.  The 1p1h<->2p2h SECOND-ORDER coupling is still
pending, so excitation energies agree with pyscf EE-UADC(3) to ~2-5e-3 (the
coupling contribution).  The static ph-ph self-energy itself -- the deliverable
-- is complete and validated.  The remaining coupling is the t2-dressed (ia|bc)
2nd-order term (pyscf matvec M_02Y1/M_12Y0); its dominant contribution to the
alpha singles is via the opposite-spin (abab) doubles, so the clean spin-orbital
form needs the all-alpha matvec (pyscf's empty-beta path needs guarding) -- the
ph-ph block confirms the remaining solver error is purely this coupling.

All three ADC(3) excitation self-energies (IP `Sigma^(3)_ij`, EA
`Sigma^(3)_ab`, EE `M^(3)_{ia,jb}`) are now validated against pyscf for general
molecules.
Also pending: EE transition moments, G0W0 (xfail), Kramers symmetry.

* **IP/EA spectroscopic factors** (`ip_adc2_spec`, `ea_adc2_spec`): pole
  strengths `P_n = sum_p |<N-+1,n|a_p(^dag)|0>|^2` from the ADC(2) Dyson
  amplitudes.  Needs the 2nd-order singles `t1_2` (stored in `_build`).
  Transition moments: occ-p 1h `delta - 0.25 t2 t2*`, virt-p 2h1p `t2*`
  + 2nd-order 1h `t1_2`; EA is the mirror.  **All conjugations fixed by
  Gate-2**: the Dyson amplitudes must use `t2.conj()` (holes ~ ph*), like
  `sig_ip` -- matching pyscf alone (real reference) does NOT pin them; only
  the complex-rotation-invariance gate does.  Matches pyscf UADC pole
  strengths to ~1e-8 (IP 0.924, EA 0.987...).  Note: within a degenerate
  energy manifold individual P are basis-dependent (solver picks an arbitrary
  basis); only the manifold sum is gauge-invariant -- the Gate-2 test compares
  manifold sums.

* **CVS-IP-ADC(2)** (`ip_cvs_adc2(nroots, ncvs)`): core-valence separation by
  projection -- restrict the 1h space to a core hole and the 2h1p space to
  configs with >=1 core hole (drop both-valence), with the *same* IP-ADC(2)
  matrix elements.  ``ncvs`` counts core *spinors* (2x spatial, Kramers).
  Matches pyscf UADC CVS-IP to machine precision (F 1s: 25.475 Ha; Ne 1s).

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
