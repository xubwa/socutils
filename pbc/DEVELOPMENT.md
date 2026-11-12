# Periodic J/K for all-electron, heavy-element crystals: development notes

Living document for the periodic Coulomb/exchange work in `socutils/pbc`.  It
records what was decided, why, what was measured, and what is still open.
Numbers quoted here come from `pbc/bench` and `pbc/test` (pyscf 2.14, Gamma
point, 4-core container) unless stated otherwise.

## 1. Goal

Target system: Eu doped on a Y site of Y2SiO5 (X2 phase), with nuclear-position
properties (W_NSM) computed from a CI wave function of the Eu 4f shell.

Constraints that drive everything else:

* **All-electron.** The X2CAMF Hamiltonian and the property at the nucleus need
  every electron; no pseudopotentials, no ECP + X2C mixing.
* **Compact functions.** Eu/Y cores and the property need tight (uncontracted,
  even-tempered) s/p primitives, exponents up to ~1e7-1e8.
* **Two-component, complex density matrices** (SOC); later Kramers-restricted.
* **Exchange is the leading cost.**  J is cheap in every scheme; K is not.

Physics route agreed on (Section 7 has the alternatives considered):

1. periodic 2c fractional-occupation (4f^6 averaged) DFT, semilocal functional
   (time-reversal-invariant density, so the XC is the closed-shell functional);
2. CASCI(6e, 14 spinors) on Eu with h_eff built from the periodic core;
   only K_core projected on the 4f spinors is needed, which is local;
3. W_NSM from core + CI density with picture-change correction;
4. later: periodic CAHF/cMF, which needs a full periodic K.

## 2. What was wrong with the available tools

Found while evaluating pyscf for this target; each item has a reproducer in
`pbc/bench` or below.

| item | evidence | consequence |
|---|---|---|
| pyscf `pbc/scf/ghf.py` (and `kghf.py`, `kghf_ksymm.py`) takes the beta-alpha block of the density for the alpha-beta exchange block (`dms[:, nao:, :nao]`) | random complex Hermitian GHF density vs explicit ERIs: K_ab error 1.13, equal to K_ba; fixed slice gives 1e-15 | every non-collinear (SOC) periodic GHF exchange is wrong; collinear tests cannot see it |
| RSJK default precision with Y (x2c-SVPall) | A-type Y2O3: J energy off by -0.085 Eh at prec 1e-8, -0.11 at 1e-6; correct (1e-9) only from 1e-10 | silent; lattice-sum estimators do not cover compact functions |
| RSJK cost | Y2O3 (128 AO) one J at prec 1e-10: 585 s; YSO primitive (692 AO) did not finish one J at prec 1e-6 in 9 min; omega = 0.1 on 2-atom ZnO allocates a 65 GB shell-pair index | not usable for production on CPUs |
| RSJK complex density | imaginary part below 1e-6 is dropped silently (CPU); the GPU version asserts a real density at Gamma | 2c Gamma-point SOC needs a wrapper (K is linear: split real/imag) |
| GDF with dense compact auxiliary sets in a dense lattice | Y2O3 + Lehtola auxgen (uncontracted, tau 1e-8): J +6.7 Eh, K -0.29 (variational bound violated); periodic j2c has negative eigenvalues; forcing eigh with any threshold leaves J +2.2e-3; same sets behave in vacuum | integrals of compact functions in dense lattices are not accurate; cannot be fixed by metric thresholds |
| GDF default (ETB) fitting error | quartz/Y2O3 J -3e-4/-2.6e-4, K 4.6e-4/6.3e-4; denser ETB (beta 1.3) breaks on Y2O3 | K needs much larger auxiliary sets than J |
| `cell.exp_to_discard` | crashes when it empties a whole shell | use `bench.cells.trim_basis` |
| molecular `pyscf.scf.ghf.get_jk` (module function) | default `jkbuild=hf.get_jk` is called positionally as `(mol, dm, hermi, with_j, with_k, omega)`, so `with_j` lands in `vhfopt` and it crashes | the GHF class method has its own jkbuild and is fine; pass `jkbuild` explicitly when using the function |
| FFTDF J for a non-Hermitian complex density | not linear in D (assumes Hermitian) | irrelevant for GHF (J uses diagonal blocks) |

Likely common cause of the silent failures (consistent with what we hit in
our own code, see 4.4): absolute integral thresholds applied to L2-normalised
compact functions, whose fit coefficients are large, plus a metric that is
used without checking positive definiteness.  Not yet traced line by line in
pyscf.

## 3. Validation: references that do not depend on any periodic code

Ordered by what they test.  All in `pbc/coulomb_reference.py`, unit-tested in
`pbc/test/test_coulomb_reference.py`.

1. **Analytic.** Madelung constants (NaCl, CsCl, ZnS) to 1e-12; lattices of
   Gaussian charges as a pure reciprocal sum and via an Ewald split (any eta)
   agree to 1e-13; point-charge limit.
2. **Pure reciprocal-space AO reference.** Gamma-point J, V_ne and energy of an
   AO density with Gaussian nuclei from analytic FTs of explicitly enumerated
   AO pairs; only the G cutoff and the pair radius truncate, both monotone.
   Smooth bases only (cost grows with the largest exponent): He/cc-pVDZ is
   converged to ~1e-13 and agrees with dense-mesh FFTDF to 3e-12.
3. **Invariances.** Translation, supercell replication (J of a fixed periodic
   density does not depend on the cell), choice of omega.
4. **Zero-dimensional limit.** A neutral dipole-free molecule in a box:
   E(L) - E_mol = A/L^5 and vj + vnuc shifted by -(2 pi/3V) tr2 S.  Works for
   compact functions but does not exercise dense lattice sums: pyscf GDF passes
   it with the same auxiliary sets that break in Y2O3.
5. **Dense lattice + compact functions, exact.**  One s function per atom and
   an auxiliary basis spanning chi^2 exactly: the fitted Coulomb energy must
   equal the Ewald energy of the Gaussian charges.  Covers exponents 1e4 and
   nuclei 1e6 at a 2-bohr Li-H spacing (`test_rsdf.test_exact_fit_tight_dense`).
6. **Cross-code.** pyscf RSJK at prec 1e-12 with an omega check (~1e-8 on
   Y2O3; slow) for realistic bases.  The LAPW HF-limit data of Gulans et al.
   (arXiv 2606.20116) test basis completeness and the exchange treatment at the
   thermodynamic limit, not an implementation at fixed basis.

## 4. RSDF: range-separated Gamma-point density fitting (`pbc/rsdf.py`)

### 4.1 Structure

Kernel: periodic Coulomb with G = 0 removed (pyscf exxdiv=None convention),
split into
* SR erfc(w r)/r in real space over explicit images,
* LR erf(w r)/r on G != 0 by analytic Fourier transforms,
* minus the G = 0 part of the SR term, pi/(w^2 V) per product of charges.

Every 2c/3c integral of the fit uses this kernel; the metric is the same
kernel.  Nuclei are Gaussian charges (finite nucleus; 1e12 mimics points)
treated as extra "auxiliary" shells, so J, V_ne, E_nn share one convention.

### 4.2 Short range (`lib/pbcdf/sr_int.c`)

* One libcint environment: AO image shells (cell 0 first) + auxiliary/nuclear
  image shells; env[PTR_RANGE_OMEGA] = -w.
* Pair bound from the most diffuse primitives:
  c_i c_j exp(-mu R^2) (pi/p)^1.5; triple bound times c_k (pi/q)^1.5 g(d), with
  g the exact SR interaction of two s Gaussians,
  g = [erf(gamma d) - erf(beta d)]/d, 1/beta^2 = 1/w^2 + 1/p + 1/q.
* Aux image shells binned on a 4-bohr grid; each pair visits only the bins
  inside its interaction sphere.  OpenMP over cell-0 shells (row ownership, no
  atomics).

### 4.3 Long range

AO pairs (overlap-screened, `ft_aopair(..., ovlp_mask=...)`) and auxiliary
functions transformed analytically; |G| up to 2w sqrt(ln(1/prec) + ln 4pi).
`rs_lr.LongRangeJK` is the same LR part as a direct J/K (no fitting), validated
against FFTDF with omega to 1e-13.

### 4.4 Lessons from the screening

**Bounds must be in the units the quantity is used in.**  First version:
errors of 1e-5..1e-6 in the exact-fit dense test, shrinking with precision.
libcint-normalised compact s shells have tiny charges, so quantities divided
by those charges afterwards (nuclei) or multiplied by large fit coefficients
(compact aux) amplify every screened-out integral (factor ~6e7 for a 1e6
nucleus).  Nuclei are now scaled to their charge Z inside the libcint
environment.

**Neither L2 nor Coulomb normalisation alone is a safe measure for the
auxiliary functions.**  Coulomb-normalised bounds ((pi/2q)^(1/4) per s
envelope) fix compact aux functions (exact-fit dense test 5e-11) but
under-estimate diffuse ones: on Y2O3 at prec 1e-10 the metric got a -4e-8
eigenvalue, J turned positive (+2.5e-5) and the omega dependence was 3.5e-4.
L2-normalised bounds have the opposite problem.  Either alone needs prec 1e-12
on Y2O3; the larger of the two converges at 1e-10 (default, `aux_bound='max'`).

| Y2O3, omega 0.8 | J dE | K dE | metric min eig | triples | build |
|---|---|---|---|---|---|
| aux bound L2, prec 1e-10 | -1.8e-5 | 5.5e-5 | +2.4e-11 | 2.35e7 | 161 s |
| aux bound Coulomb, prec 1e-12 | -1.8e-5 | 5.6e-5 | -5.7e-11 | 2.69e7 | 194 s |
| aux bound L2, prec 1e-12 | -1.8e-5 | 5.5e-5 | +2.4e-11 | 3.60e7 | 213 s |

**The metric must be checked.**  Loose screening shows up first as negative
metric eigenvalues and a positive J error, the same signature as the pyscf
failure: quartz at prec 1e-6 gives -6e-6 and J +5e-4; Y2O3 at prec 1e-6
gives -4e-4 and J +2e-2.  RSDF warns when the metric is not positive
semidefinite.

**Rigorous angular envelope (implemented).**  For any r,
r^l exp(-a r^2) <= (l/(a e))^(l/2) exp(-a r^2/2), so every primitive is bounded
pointwise by an s Gaussian of half the exponent (l = 0 keeps a).  The angular
amplitude per l is measured numerically from libcint (`_angular_amplitude`,
+5%), so libcint's normalisation is absorbed.  With the positive erfc kernel
the s-s interaction of the envelopes is a true upper bound; bin charges are
summed, g is taken at both ends of each p bin, and the cutoff is split over
the dropped primitive pairs and bins.  Aux shells pass envelope exponent and
envelope charge directly.  Result (prec 1e-10, 4 cores):

| | aux bound | quartz w-dep. J / K | Y2O3 w-dep. J / K | Y2O3 min eig (w 0.5) | Y2O3 triples / build (w 0.5) |
|---|---|---|---|---|---|
| old heuristic bound | max | 3.7e-9 / 1.3e-8 | 5.4e-6 / 1.1e-5 | -1.5e-10 | 3.6e7 / 166 s |
| rigorous envelope | max | 3.4e-10 / 2.1e-10 | 5.8e-7 / 1.2e-7 | -1.3e-10 | 1.5e8 / 404 s |
| rigorous envelope | l2 | 2.6e-10 / 2.1e-10 | 4.8e-7 / 1.2e-7 | -1.3e-10 | 1.3e8 / 396 s |
| rigorous envelope | coulomb | 6.7e-8 / 5.9e-8 | 3.9e-7 / 5.6e-7 | -4.6e-9 | 1.4e8 / 375 s |

(w-dep. = max matrix difference omega 0.5 vs 0.8.)  Errors against RSJK are
unchanged (fitting-limited: J dE -1.2e-5 / -1.8e-5, K dE 2.2e-4 / 5.5e-5).
On these cells L2 alone would do (cheapest); but the exact-fit tight-dense
test still needs the Coulomb measure for its compact aux functions (L2 alone:
energy error 2e-8 there), so the default stays `aux_bound='max'`.  The omega
dependence on Y2O3 dropped 10x (K 100x).  It costs 4x the triples and 2.4x the
build time, because halving the exponent is loose.

**Envelope family.**  For any 0 < t < 1,
r^l e^{-ar^2} <= (l/(2(1-t)ae))^(l/2) e^{-t a r^2} (t = 1/2 is the bound
above).  A larger t decays faster with a larger amplitude.  Every t is a
rigorous bound, so the kernel takes the smallest over t = 0.5, 0.75, 0.9.  It
also has three other parts:
* Primitive pairs are dropped smallest first while their summed bound stays
  below cutoff/2, and triples are screened at cutoff/2.
* The pair radius comes from bisection on the summed tail.
* A cheap exp-only bound (erfc x <= e^{-x^2}, g <= g(0)) runs before the erf one.

| prec 1e-10, w 0.5 | triples | SR 3c | build |
|---|---|---|---|
| quartz, old heuristic | 4.3e7 | | 65 s |
| quartz, t = 1/2 only | 1.5e8 | | 136 s |
| quartz, t in {.5,.75,.9} | 5.6e7 | 69 s | 103 s |
| Y2O3, old heuristic | 3.6e7 | | 166 s |
| Y2O3, t = 1/2 only | 1.5e8 | | 404 s |
| Y2O3, t in {.5,.75,.9} | 5.6e7 | 274 s | 303 s |

On Y2O3 the SR 3c integrals themselves dominate: about 20 us of CPU per
triple, with f AOs and aux up to l = 6.

**libcint's erfc-attenuated 2c integrals are not exact.**  For a one-centre Y
f-f pair (exponent 0.32), int2c2e with omega = -0.5 differs from
full - LR by 3.7e-10.  Both full and LR agree with Fourier quadrature
(Lebedev 5810 x 400 radial) to 1e-13, so the SR routine is the inaccurate
one.  The same check on 3c gives 3.5e-11.  The SR 2c metric is now evaluated
as full - LR (2c is cheap).  It did not remove the Y2O3 residual, but it
removes a known error source from the metric.

**The remaining omega dependence on Y2O3 is precision times conditioning,
not a bug.**  Tests that changed nothing (max matrix difference, w 0.5 vs 0.8):
* G cutoff, SR image radius or AO image radius x1.3: J 5.9e-7 in every case.
* The fitting metric differences between omega values are about 1e-10 on
  diffuse s functions (relative 6e-12).  The metric's condition number is
  1e13 (eigenvalues 2.4e-11..2.6e2), and its smallest eigenvalue is -1.2e-10
  at w 0.5 for both precisions.  That is about the round-off floor of the
  lattice sum.
* With lindep 1e-12..1e-8, K's omega dependence falls from 1.2e-7 to 1.7e-9,
  but J stays at 5-8e-7.  J's part therefore comes from the 3c integrals.

What does change it is the precision:

| Y2O3 | J w-dep. | K w-dep. | build w 0.5 / 0.8 |
|---|---|---|---|
| prec 1e-10 | 5.8e-7 | 1.7e-7 | 318 / 275 s |
| prec 1e-12 | 9.2e-8 | 1.1e-7 | 381 / 336 s |

The errors against RSJK are the same at both precisions (fitting-limited).

### 4.5 Results

Exact-fit tests (`pbc/test/test_rsdf.py`): energies and J/V_ne matrices vs the
references to ~1e-12 for omega 0.4-1.5; tight-dense test 5e-11 at prec 1e-12.

Realistic bases (x2c-SVPall trimmed at 0.1, auxgen uncontracted tau 1e-8),
defaults (prec 1e-10, `aux_bound='max'`), against RSJK prec 1e-10:

| system | omega | J dE | K dE | metric min eig | build (4 cores) |
|---|---|---|---|---|---|
| alpha-quartz (135 AO) | 0.5 | -1.2e-5 | 2.2e-4 | +1.7e-9 | 65 s |
| alpha-quartz | 0.8 | -1.2e-5 | 2.2e-4 | +1.7e-9 | 120 s |
| A-Y2O3 (128 AO) | 0.5 | -1.8e-5 | 5.5e-5 | -1.5e-10 | 166 s |
| A-Y2O3 | 0.8 | -1.8e-5 | 5.5e-5 | +2.4e-11 | 159 s |

Omega independence (max matrix difference, omega 0.5 vs 0.8): quartz J 3.7e-9,
K 1.3e-8; Y2O3 J 5.4e-6, K 1.1e-5 (an order of magnitude below the K fitting
error of this auxiliary basis, 2.7e-4; the rigorous envelope of 4.4 should
close it).

pyscf GDF with the same auxiliary basis: quartz identical errors (-1.2e-5 /
2.2e-4, build 63 s); Y2O3 J +6.7 Eh, K -0.29 (build 137 s).  So RSDF costs
about the same as pyscf GDF and is correct where GDF breaks.

### 4.6 Precision and omega

* prec 1e-6: unusable (metric non-PSD, positive J error) on both systems.
* prec 1e-8: quartz fine (J -7.7e-6), Y2O3 marginal (metric -3e-6).
* prec 1e-10: converged to the fitting error on both; default.
* omega: 0.5 is the faster choice on these cells (quartz 65 s vs 120 s at
  0.8; LR grows as omega^3, SR shrinks); omega 1.2 took 300+ s.

Build-time history (quartz, prec 1e-8): 152 s first version -> 38 s after
folding image AOs by a sum over translations, using one vector per +-G pair
and a GEMM for the metric transform (SR 3c 20 s, LR 15 s, SR 2c 1.5 s).

### 4.6b Blocked build and the YSO cell

The build is blocked over cell-0 AO shells (rows), with this order:
* The metric comes first: SR 2c as full - LR, LR 2c in G space, then the
  G = 0 removal and the eigendecomposition.
* Then, per row block:
  * SR 3c over the lower shell triangle.  It is a contiguous range of pyscf's
    packed pair index, and the diagonal shell blocks come out complete.
  * LR: pair FTs summed per translation, skipping translations whose
    lower-triangle overlap mask is empty, then two real GEMMs on the packed
    columns.
  * The G = 0 removal and the fit.
* The (nfit, nao, nao) tensor is never stored.

The vectors are stored packed (`cderi`, (naux_kept, nao(nao+1)/2)) in memory,
or in HDF5 when they exceed max_memory/2.  `loop()` and `get_jk` stream aux
blocks.  The old LR (one complex F over all image AOs, G batches of ~20 under
the memory limit) took 128-190 s per YSO block; the current one takes 7-47 s.

Y2SiO5 conventional cell (`bench/yso_build.py`): 32 atoms, x2c-SVPall
trimmed at 0.1, nao 692, auxgen naux 9912, prec 1e-10, w 0.5,
max_memory 10 GB, 4 cores.

| | |
|---|---|
| 2c metric (SR full - LR / LR) | 127 s / 22 s, eig [2.4e-11, 6.5e2], kept 9847 |
| SR 3c | 1994 s, 2.8e8 triples |
| LR 3c | 1418 s, 3263 G vectors |
| build total | 3960 s wall (7277 s CPU: only ~1.8 of 4 cores busy) |
| storage | 19 GB HDF5; peak RSS 7.4 GB |
| J / K per call (minao density) | 43 s / 156 s |

No reference exists for this cell (RSJK and GDF are out of reach here), so it
is a cost measurement only.  Correctness rests on the quartz/Y2O3 checks and
the tests.

Where the time goes:
* **Parallel efficiency.** The SR kernel is OpenMP over cell-0 shells, with
  4-8 shells per row block.  Blocks with one heavy Y shell leave the other
  threads idle (block 20: 4.6e6 triples in 113 s; block 30: 1.1e7 triples
  in 38 s).  The LR part is numpy with only partly threaded steps.
  Next: parallelize over (shell, image-shell) pairs with per-thread row
  buffers, and run SR and LR concurrently.
* **K.**  At 156 s per call, K dominates an SCF iteration: about 2 naux nao^3
  flops plus a 19 GB read.  occ-RI (K on nocc ~ 264 occupied orbitals) and
  local/robust fitting are the next steps for K, as planned.

### 4.7 Known limitations

* Build cost (4 cores, defaults): quartz ~1 min, Y2O3 ~3 min, comparable to
  pyscf GDF with the same auxiliary basis; SR 3c (2-4e7 shell triples with
  high-l uncontracted aux) and the LR transforms dominate.
* Storage is global (naux x nao^2/2, on disk when large): 19 GB for the YSO
  conventional cell.  Larger cells need local fitting.
* K accuracy is limited by the auxiliary basis (quartz 2e-4 with auxgen tau
  1e-8): K converges much more slowly than J.
* Gamma point only.  Complex densities: K is built as K(Re D) + i K(Im D)
  with real vectors.  Spin-orbital (2c) densities: `RSDF.get_jk_ghf` /
  `rsdf.ghf_get_jk` assemble the spin blocks with the correct alpha-beta slice
  (tested against pyscf's molecular GHF for Hermitian and non-Hermitian
  non-collinear densities).  No Kramers-symmetry savings yet.

## 4b. Spinor SCF and CASCI on RSDF (`pbc/spinor_scf.py`)

The molecular socutils machinery is reused as is, and RSDF is the DF object:
* `scf.spinor_hf._DFJHF` builds J and the MO-based K from `with_df.loop()`.
* `mcscf.zmc_ao2mo._CDERIS` (zcasci/zcasscf) builds cd_pa, (tu|vw), core K
  and core J in one pass over the vectors.

RSDF only needed adapters: pyscf's `get_jk(dm, hermi, with_j, with_k,
direct_scf_tol, omega)` argument order, and the `mol`, `blockdim` and
`_cderi` attributes.  `spinor_scf.periodic(mf, rsdf)` (or
`SpinorRSDF(cell, rsdf)`) replaces only the one-electron part:
* S and T are Gamma lattice sums.
* V_ne and E_nn come from RSDF (Gaussian nuclei, the same kernel as J).
* Periodic X2C/X2CAMF is not wired in yet (NotImplementedError).

**Exchange G = 0 term.**  The first molecule-in-box comparison was off by
1.40 Eh at L = 14 and 0.99 Eh at L = 20.  That is -nu_M N/2 with
nu_M = 2.837/L: the exchange-hole G = 0 term that exxdiv=None leaves out.
The earlier 0D tests checked only Coulomb energies, so they missed it.  The
G = 0 part of (pq|rs) is v delta_pq delta_rs.  For any N-electron
wavefunction it gives v N(N-1)/2, which leaves -nu_M N/2 after the
background.  So it is a constant: orbitals, CI vectors and correlation
energies are unaffected.  `exxdiv='ewald'` (default) adds it to
`energy_nuc()`; HF and CASCI both pick it up there, and for converged HF it
equals pyscf's ewald exchange correction.

What remains is the tin-foil dipole term, exactly A/L^3.  For N2/cc-pVDZ,
aug-ETB aux, CASCI(6e, 12 spinors), with the extrapolated value
E_inf = (E_b b^3 - E_a a^3)/(b^3 - a^3):

| L (bohr) | E_HF - E_mol | E_CASCI - E_mol |
|---|---|---|
| 14 | -1.38e-2 | -1.24e-2 |
| 20 | -4.73e-3 | -4.24e-3 |
| 26 | -2.15e-3 | -1.93e-3 |
| extrapolated (20, 26) | 5.0e-7 | 2.1e-7 |

A CAS of only occupied orbitals reproduces E_HF to 1e-13.
`pbc/test/test_spinor_casci.py` covers the extrapolation, the CAS = occupied
check and the Madelung term.

**One-centre X2CMP** (`OneCenterX2CMPHelper`, `mf.x2cmp('x2cmp_1c' or
'x2camf_1c')`).  *Superseded as the periodic X2CAMF by 4k:* the statement
below that dropping the other nuclei from the decoupling "is exactly what
X2CAMF neglects relative to X2CMP" is wrong -- molecular X2CAMF decouples the
full one-electron Dirac Hamiltonian (all nuclei, X2C-1e) and only its 2e
SOC/Gaunt/Breit part is atomic.  The one-centre scheme also makes the scalar
relativity one-centre.  Original notes:

  h = h_NR(cell) + sum_A [h_X2C(A) - h_NR(A)]

Each atomic block comes from the molecular socutils helper
(`SpinorX2CMPHelper`) on the isolated atom, in its own basis:
* flavor 'x2camf': X2C-1e plus the atomic mean-field SOC.
* flavor 'x2cmp_1c': the atomic 4c Fock decoupled.

The multi-centre relativistic terms are dropped: other nuclei inside the
decoupling, and the two-centre picture change.  That is exactly what X2CAMF
neglects relative to X2CMP, and it is what makes the method usable for a
crystal.  The interface follows the x2cmp one: `mf.x2camf()` or
`mf.x2cmp('x2cmp_1c')`; every other flavor (molecular 'x2cmp', 'sfmp', ...)
raises.

Checks (`pbc/test/test_periodic_x2camf.py`):
* **One-centre error on a molecule** (HI, x2c-SVPall), against the full
  molecular X2CAMF energy: 4.4e-5 Eh, or 4.9e-5 Eh with Gaunt + Breit.  The
  relativistic effect itself is 386 Eh.
* **Kr atom in a box**, where the one-centre form is exact: the
  extrapolation (22, 28) matches the molecular atom to 7e-7 Eh, for point
  and for Gaussian nuclei.
* **Nuclear-model mismatch.**  With RSDF zeta 1e12 against point nuclei the
  error was 1.5e-5 Eh; at the NR level point vs Gaussian 1e12 is already
  6e-6.  `check_nuclear_model` now enforces consistency: point cells want
  zeta >= 1e13 (warning otherwise); Gaussian cells (cell.nucmod 'G',
  Visscher-Dyall as in the x2camf code) require
  `nuc_zeta=nuclear_zeta(cell)`.  This matters later for W_NSM.

**x2camf C backend bug found on the way.**  `x2c_int_sph.c` and
`x2c_int_sph_gauge.c` kept the radial 2e lists (n^4 (L+1) 9 doubles) as
stack VLAs.  They overflowed the 8 MB default stack for uncontracted
heavy-atom shells (segfault for I and Eu with x2c-SVPall), so they now live
on the heap, as the Gaunt kernels already did.

YSO with the (row shell, column shell) task split: SR 3c 1994 -> 1073 s,
build 3960 -> 3105 s.  The K timings of that run are not usable (the tests
ran at the same time); occ-RI K = full K to 2.4e-13.

Next for the Eu target:
* Localised Eu 4f active spinors in a Gamma supercell (AVAS-type projection).
* A (tu|vw) accuracy check against exact integrals.
* Kramers pairs in J/K and in the transformation.
* ACE.

## 4c. Cost after the LR and load-balance changes (YSO conventional cell)

The run below had nothing else running alongside it (4 cores, w 0.5,
prec 1e-10, `bench/yso_build.py`):

| | before | now |
|---|---|---|
| SR 3c | 1994 s | 938 s (tasks over (row shell, column shell) pairs) |
| LR 3c | 1418 s | 538 s (pyscf's C lattice-sum pair FT, s2, real/imag) |
| fit W B | 343 s | 309 s |
| 2c metric | 150 s | 154 s |
| build | 3960 s | 2026 s |

**Calls on this run** (minao density, scalar `RSDF.get_jk`, vectors read
from disk):
* J: 39 s.
* Full K: 147 s.
* occ-RI K: 145 s, with nocc 312 of 692 (fractional minao occupations).
  It is no faster here because each block is unpacked and copied.

The 2c SCF does not take this path.  It uses `_DFJHF._get_k_mo`, which
half-transforms the packed vectors with the occupied spinors and now
transforms only the unbarred Kramers half.  Its cost still has to be
measured in an SCF.

**Omega scan** (single cell, the LR part now being cheap):
* Y2O3: w 0.8 is fastest (179 s vs 259 s at w 0.5).  The SR triple count
  falls only from 3.8e7 at w 0.8 to 2.8e7 at w 1.6, because the SR cost
  comes from the compact core functions whatever w is.
* Quartz: also fastest at w 0.8 (62 s vs 85 s at w 0.5).
* For a larger cell the LR part grows faster (N^2 V against N), so the best
  w shrinks with the cell.  For YSO, w 0.5 remains the choice.

**Scaling**, for a cell n times larger: SR 3c grows as n; the LR GEMM, the
fit W B and occ-RI K grow as n^4; storage grows as n^3.
* On 48 cores (about 10x the 4 used here) the estimates are: 1x1x1 build
  4 min / 19 GB; 1x1x2 (64 atoms) build 25 min / 150 GB, a few minutes per
  K; 128 atoms build 6 h / 1.2 TB.
* Beyond 64 atoms this needs local fitting for K (PARI / RI-LVL) and sparse
  storage for J.

## 4d. First Eu-doped run (workflow check, `bench/eu_casci.py`)

Eu:Y2O3 (A-type cell, one Y -> Eu; 5 atoms, n2c 320, 126 electrons; Gaussian
nuclei; 4 cores).  The workflow:
1. RSDF, w 0.8: 272 s.
2. Periodic KRHF + one-centre X2CAMF, with the 14 Eu 4f spinors at 6/14
   occupation (`projected_frac_occ`), K rebuilt every 3rd iteration: 77
   cycles, 28 exact K builds, 160 s.  This run predates the frozen-K
   convergence fix.
3. CASCI(6e, 14) (`cas_order`): 24 s.

What it shows:
* **4f selection.**  The selected open shell has Eu-f weights 0.94-0.99, in
  Kramers pairs.  The first version picked virtual f-polarisation orbitals
  at +1..+4.6 Eh, which is why the selection is now by energy among the
  candidates.
* **Multiplet structure.**  The CASCI levels (cm-1) are 0 (7F0); 98 and
  725 x2 (7F1, A + E); 1340 x2, 1598 and 2160 x2 (7F2, E + A + E).  That is
  the C3v crystal-field pattern of the site.  The barycentres (7F1 ~ 516,
  7F2 ~ 1700) lie above the free-ion values (370, 1000).  This is expected
  from HF orbitals with no dynamic correlation, an approximate structure,
  and a basis for which the atomic X2CAMF Dirac-HF of Eu does not converge.
  The run checks the workflow, not the physics.

## 4e. Local exchange: accuracy prototype (`pbc/pari.py`)

Pair-atomic RI (PARI): each lattice-summed AO pair (mu on A, nu on B) is
fitted with the lattice-summed aux functions of A and B, optionally plus
their first neighbours; K = C V C (non-robust).  This is the scheme of
Sodt/Subotnik/Head-Gordon 2006, PARI-K (Manzer et al. 2015) and RI-LVL
(Ihrig et al. 2015, FHI-aims periodic hybrids); the robust form is Dunlap's,
analysed for PARI by Merlot et al. 2013.  The prototype uses dense
intermediates (`RSDF.keep_raw`).  Exchange-energy error dE against global DF
(minao density):

| | Y2O3 (5 at.) | quartz (9 at.) |
|---|---|---|
| PARI, two-atom domains | -2.6e-4 | 2.0e-3 |
| robust PARI | 1.3e-3 | 1.8e-3 |
| PARI + first neighbours (4.5 bohr) | 2.4e-5 | 4.1e-4 |
| **SR-PARI (erfc) + exact LR**, two-atom, w 0.8 / 1.6 | -8.6e-5 / -8.2e-5 | 1.5e-3 / 6.2e-4 |
| **SR-PARI + neighbours + exact LR**, w 0.8 / 1.6 | 1.6e-5 / -3.6e-7 | 3.2e-4 / 9.4e-5 |
| (global DF vs RSJK, for scale) | 5.5e-5 | 2.2e-4 |

**SR-PARI + exact LR** means:
* K_SR uses PARI with the erfc kernel only, for both the fit and the
  metric, so everything in it is local.
* K_LR = sum_G (v_G/V) Re[rho(G) D rho(G)^+] - pi/(w^2 V) S D S, computed
  from the analytic lattice-summed pair FTs, with no fit.
* Splitting this way is exact: a global SR fit plus the exact LR reproduces
  global DF to 2e-6 (w 0.8).

Findings:
* Robust PARI does not help K: the two-atom fit residuals are too large for
  second-order cancellation to pay.
* The shorter the SR kernel (larger w), the more accurate the local fit.
* With neighbours, SR-PARI at w 1.6 is below the fitting error of the
  global DF itself.

**The open problem is the cost of K_LR.**  Densely it is ng nao^3 per K,
with ng ~ w^3 V:
* At a small w_K (0.2-0.3) a 128-atom cell needs ~10^3 G and roughly
  1e14 flop, i.e. minutes.  But SR-PARI loses accuracy there, because the
  SR kernel is long-ranged.
* At w 1.6, ng is ~10^5 and the dense form is out of reach.

The way out is the locality of exchange.  Two options:
* Sparse D (LinK-type screening of rho(G) D rho(G)^+).
* Localised occupied spinors.

Either one makes K_LR ~ ng N, so w can stay large.  Storage of the SR-PARI
coefficients is linear, and they need only the SR three-index integrals,
which are local and come from the existing SR kernel restricted to the
domain aux.  The LR part of the three-index tensor, the N^2 wall of any
explicit fit, is never formed.

## 4f. Production driver and dopant supercells

`examples/14-periodic_dopant_casci.py` runs RSDF -> KRHF + one-centre
X2CAMF (fractional dopant 4f) -> CASCI(4f) and keeps every stage in a work
directory:
* `cderi.h5`: `RSDF.cderi_file`, then `RSDF.load` on later runs.
* `scf.npz`: the SCF restart guess.
* `casci.npz`: the CASCI energies.

The SCF also builds J and the MO-based K in one pass over the vectors.  On
Eu:Y2O3 (4 cores):
* First run: RSDF 279 s, SCF 17 cycles in 92 s, CASCI 24 s.
* Second run: loads the vectors, the SCF converges in 1 cycle, and the
  CASCI levels are unchanged.
* The fractional SCF energy is 0.56 Eh above the earlier run.  That is the
  -nu tr(DSDS)/2 exchange term (section 4b), not a change in physics: the
  CASCI energy is the same to 1e-9.

**Dopant separation in Eu:YSO.**  The primitive cell (32 atoms) contains a
6.82 A lattice vector (-a1 + 2 a3, the conventional c), so doubling a single
cell vector does not separate the dopant images.  Over all Hermite normal
forms, the best Eu-Eu image distances are:

| supercell | atoms | Eu-Eu | vectors (global RSDF) |
|---|---|---|---|
| x1 | 32 | 6.82 A | 19 GB |
| x2 (`eu_yso_x2`) | 64 | 8.05 A | ~150 GB |
| x3 (`eu_yso_x3`) | 96 | 10.52 A | ~0.5 TB |
| x4 | 128 | 12.54 A | ~1.2 TB |

## 4g. Periodic spinor Kohn-Sham (`pbc/spinor_ks.py`)

`spinor_ks.periodic_ks(spinor_hf.KRHF(cell), rsdf, xc)` puts KS on top of the
periodic spinor SCF (one-centre X2CAMF, fractional occupation and CASCI work
unchanged):
* **Grids.**  `pyscf.pbc.dft.gen_grid.BeckeGrids`: atomic radial x angular
  grids, Becke partition over the lattice images.  The heavy-atom core needs
  atomic grids; a uniform FFT mesh cannot hold it.
* **Numerical integration.**  `pyscf.pbc.dft.numint2c` at Gamma, with
  lattice-summed AO values, on the scalar-AO x spin form of the spinor
  density, as `socutils.dft.numint_2c` does for molecules.  It is
  collinear by default; 'mcol' needs `mcfun`.
* **Coulomb and exchange.**  J comes from RSDF.  Hybrids take hyb x (K +
  nu S D S); pure functionals have no exchange G = 0 term.
  Range-separated hybrids are not supported yet.
* **Energy.**  tr(D h) + E_J + E_xc - hyb tr(D K')/2 + bare E_nn.

Test (`test_periodic_ks.py`): N2 in a box extrapolates to the molecular
SpinorDFT energy (same aux basis, grid level 4) to 2.5e-7 Eh for both PBE
and PBE0.

Eu:Y2O3 with `--xc PBE`: 17 cycles, 288 s on 4 cores, 4f weights 0.92-0.97.
CASCI on the PBE orbitals shows the same C3v pattern, with 7F1 at 496 and
918 x2 cm-1.

## 4h. CAHF stage (dopant driver) and three fixes in the CAHF/CASSCF code

The driver now runs reference SCF (HF or KS, fractional 4f) -> CAHF ->
CASCI.  CAHF is the CASSCF with the `zcahf.CAHF` ensemble solver; it uses
the second-order super-CI driver (`zmcscf.CASSCF.kernel`) by default, with
`cahf_diis.mc_diis` as an option.  Eu:Y2O3, 4 cores:
* Super-CI: monotonic descent to E = -14420.01818.  After 60 iterations
  dE is ~1e-8 and |g| 7e-5.  Active 4f weights 0.990-0.996.  590 s.
* CASCI on the CAHF orbitals: E0 = -14420.3724, 0.12 Eh below CASCI on the
  fractional-HF orbitals.  Levels (cm-1): 0; 153, 507 x2; 948 x2, 1175,
  1196 x2, ...
* 7F1 barycentre ~389 and 7F2 ~1090 against the free-ion ~370 / ~1000
  (with fractional-HF orbitals: ~516 / ~1700).
* mc_diis: on the molecular Eu3+ ion too, it does not converge in 30
  cycles (DIIS erratic after cycle 10).  The second-order driver converges
  there as well.

Fixes on the way:
* `cahf_diis.mc_diis` built h1eff and the core energy from the core J/K of
  the *previous* iteration.  tr(D_new h) + 1/2 tr(D_new (J-K)[D_old]) is not
  variational.  Periodic Eu:Y2O3 went 6 Eh below the CASCI ground state and
  diverged; the molecular code is affected the same way, only less visibly.
  It now uses the current core J/K, which `_CDERIS` already builds.
* `mc_diis`, periodic: the composite Fock matrices lacked the exchange
  G = 0 term nu S D S.  It is -nu D in the MO basis: no gradient, but the
  core/active level positions of the step were off by nu and the
  iterations went uphill.  It is added when the SCF has `madelung()`.
* `zmc_superci` checked the Kramers pairing with `mol.intor` overlaps,
  which for a cell lack the lattice sum: a false 8.85 deviation stopped the
  run.  It now uses `mc._scf.get_ovlp()`, identical for molecules.
* `addons.cas_order` moves Kramers pairs as units (the interleaved layout
  the drivers require).

Tests: `test_periodic_cahf` extrapolates the O atom in a box to the
molecular CAHF with both drivers.  The mcscf finite-difference and Hessian
tests pass.

## 4i. Integral-free CAHF (`mcscf/cahf_fock.py`)

Integrals should not go through one global cderi: the plan is AO-driven
J/K and integral-free wavefunction steps.  For one open shell (n electrons
in m spinors, Eu 4f^6) the configuration-averaged energy depends only on
D_c = C_c C_c^+ and D_o = f C_o C_o^+ (f = n/m):

    E = tr h (D_c + D_o) + 1/2 tr D_c G_c + tr D_o G_c + a/2 tr D_o G_o + E_nuc,
    a = (n-1) m / (n (m-1)),

with G_x = J(D_x) - K(D_x).  The gradient is g = [F_c, P_c] + f [F_o, P_o],
with F_c = h + G_c + G_o and F_o = h + G_c + a G_o.  Each energy plus
gradient costs two `mf.get_jk` calls (occ-RI, Kramers-paired).  There is no
integral transformation and no rdm2.  Optimiser: L-BFGS on kappa
(exponential retraction), preconditioned with the diagonal model Hessian
2[n_p(F^p_qq - F^p_pp) - n_q(F^q_qq - F^q_pp)], backtracking line search on
the exact energy.

* O atom (2p^4, cc-pVDZ, DF): 10 iterations, equal to super-CI to 1e-14.
  The finite-difference gradient matches.
* Eu:Y2O3 (5 atoms, 4 cores): 15 iterations, 56 s, |g| 4e-6.
  E = -14420.0181818019; super-CI gave -14420.0181817902 in 590 s, not
  converged.  The CASCI levels on top are identical.
* It is the default of the dopant driver (`--cahf-driver fock`).

Active-space CASCI without the global cderi: h1eff and E_core come from the
same core Fock build.  (tu|vw) needs only the active pair densities: the SR
part from a local fit in the domain of the dopant (4f coefficients are
local, the erfc kernel keeps the aux near the pair), the LR part exactly
from the pair FTs rho_tu(G).  The G = 0 term is a constant (4j).  To do.

## 4j. SR-only fit + exact LR rows (`RSDF(lr='gspace')`)

The kernel is (mn|ls)' = SR + LR(G != 0) - c0 S_mn S_ls, with
c0 = pi/(w^2 V) the G = 0 part of the erfc kernel.  With lr='gspace':
* the vectors are the fit of the SR kernel alone (SR metric, SR 3c, no
  G = 0 removal), plus the LR part as its exact plane-wave factorisation:
  one real row per cos/sin of each +-G pair, sqrt(v_G/V) Re/Im rho_mn(G);
* -c0 S S stays out of the vectors and out of RSDF.get_jk (with_g0=True
  adds it: J -c0 tr(DS) S, K -c0 S D S).  In an orthonormal MO basis it is -c0 delta_pq delta_rs, so
  for wavefunctions it is the constant -c0 N(N-1)/2, in energy_nuc next to
  the Madelung term.  spinor_scf/spinor_ks add it explicitly to the mean
  field (`g0_terms`, `k_shift`, `wavefunction_constant`).  Every consumer of
  `loop()` (_CDERIS, CASCI, CAHF) needs no change.
* V_ne and E_nn keep the full kernel (LR only for the nuclei columns).

N2 in a box, A/L^3 extrapolation against the exact (non-DF) spinor HF
(aug-etb beta 2 aux; the molecular DF error is -4.7e-5):

| | vectors (L = 26) | E_inf - exact |
|---|---|---|
| lr='fit', w 0.5 | 240 | -4.66e-5 |
| gspace, w 0.5 | 43962 | -3.18e-5 |
| gspace, w 0.3 | 9674 | -3.90e-5 |
| gspace, w 0.2 | 3040 | -4.24e-5 |

* The exact LR removes part of the fit error.  The SR fit covers more of
  the kernel as w goes to 0, and lr='gspace' becomes lr='fit' in that limit.
* The LR row count is ~V (2w)^3 (times ln 1/eps)^{3/2}: the boxes are the
  worst case.  For YSO-64 (V ~ 11400 bohr^3) that is ~25000 rows at w = 0.5
  and ~5000 at w = 0.3, against ~9000 aux.
* Stored rows are the first step.  Next: SR vectors from a local fit
  (sparse, linear storage), and LR J/K evaluated from the rho(G) on the fly
  (K_LR with D sparsity) instead of stored rows.

Tests: `pbc/test/test_rsdf_gspace.py` checks the LR rows against the dense
G-space exchange, the SR rows against the SR fit, the G = 0 constant (CAS
of occupied orbitals = HF) and the box limit.

## 4k. Periodic X2CAMF: range-separated X2C-1e (`RangeSeparatedX2CAMFHelper`)

`mf.x2camf(omega=1.0)`:

  h = h_X2C-1e[V_sr] + (V_ne - V_sr) + sum_A [h_AMF(A) + d_A]

* `V_sr = -Z erfc(omega r)/r` of every nucleus (RSDF Gaussian nuclei),
  lattice-summed in real space at Gamma (`sr_nuclear_integrals`: V, pVp and
  p x Vp from int3c2e / int3c2e_pvp1 / int3c2e_pvxp1 with the nuclei as
  s shells, omega < 0).  Absolutely convergent: no G = 0 term, no background.
  The modified Dirac equation with V_sr is decoupled in the uncontracted basis
  and contracted (C^T h C) as in the molecular code; it carries the scalar and
  1e SOC relativity of all nuclei within ~1/omega, multi-centre included.
* `V_ne - V_sr`: the long-range rest of the RSDF nuclear potential (G = 0
  free, with the background), untransformed like J.
* `h_AMF(A) = h_X2CAMF(A) - h_X2C-1e(A)` of the isolated atom (socutils
  `SpinorX2CMPHelper`, flavors 'x2camf' and '1e'): the atomic 2e SOC +
  Gaunt + Breit, exactly the atomic part of molecular X2CAMF.
* `d_A = h_X2C-1e[V_A] - h_X2C-1e[V_sr,A] - V_lr,A` of the isolated atom
  (`self_lr=True`, same integrals): the picture change of the atom's OWN
  long-range nuclear part.  That part is smooth at the atom; leaving it
  untransformed costs a Darwin-like term of the Gaussian countercharge,
  ~ (pi/2c^2) Z (omega^2/pi)^1.5 per electron (HI, omega = 1: 28 mEh in the
  total energy, 0.4 mEh in the I 1s energy).  d_A restores it exactly, so an
  isolated atom is X2C-1e(+AMF) and only the long-range parts of the other
  nuclei and the background stay untransformed.

Why not a full X2C-1e of the cell (as pyscf.pbc.x2c.sfx2c1e does): the Ewald
nuclear potential is that of the bare nuclei plus a uniform background
-rho = -sum Z / Omega.  X2C-1e picture-changes both, while the electrons
(J, with their own background) are untransformed; in 4c the two backgrounds
cancel.  The leftover is the Darwin term of the nuclear background,
-(pi/2c^2) rho per electron: O(c^-2), extensive, volume dependent
(computed with exact molecular integrals, qdhf branch
examples/qdhf/pbc/33-heavy_background_darwin.py: fcc Ar -0.11, Kr -0.34,
Xe -0.60, Au -4.4 mEh/atom; a spurious pressure of ~ -1 GPa for Au), plus
the multipole terms of the bare-nucleus field at low-symmetry sites and a
macroscopic field that grows with the thickness of a slab.  The one-centre
scheme (4b) avoids all of this but drops the multi-centre scalar relativity.

Checks (`pbc/test/test_periodic_x2camf.py`, dyall-v2z):
* omega -> 0, molecule: h equals the molecular X2CAMF hcore to 1e-8 (|h|
  1.5e3), energy to 2e-8 Eh (HI, Gaussian nuclei; checked with x2c-SVPall).
* HI against molecular X2CAMF (energy; I 1s orbital energy), dyall-v2z:
  rs1e omega = 1: -1.46e-5 Eh, +4.6e-6;  omega = 2: -5.8e-7, +2.9e-6;
  one-centre X2CAMF (4b): -5.8e-5, +7.7e-6.  Without d_A (omega = 1,
  x2c-SVPall): -2.8e-2, -4.1e-4.
* No background term: first-order relativistic one-electron energy of an Ar
  atom in a box minus the isolated atom, ~1e-12 Eh for L = 12-20, against
  -(pi/2c^2) Z^2/L^3 = -1.6e-5 .. -3.4e-6 Eh for a full Ewald decoupling.
* Kr atom in a box (L = 28): E(X2CAMF) - E(NR) equals the molecular atom's
  to 2.5e-8 Eh.  The total energies themselves miss the molecular atom by
  4.6e-5 Eh after the A/L^3 extrapolation (22, 28) already at the NR level
  (NR 4.64e-5, one-centre 4.48e-5, rs1e 4.47e-5): a periodic NR/RSDF
  accuracy issue of dyall-v2z with the aug_etb auxiliary basis, not of the
  relativistic part, so the test compares the relativistic difference.

omega: the Gaussian countercharges of neighbouring nuclei must not overlap an
atom (exp(-omega^2 R_nn^2) << 1), or their sum becomes a smooth background
inside the decoupling again; omega = 1 (R_nn >~ 4 bohr) is the default.  The
real-space sums are a Python loop over AO images (fine for the test cells;
for YSO-size supercells the SR kernel of `lib/pbcdf` is the place to move it).

## 5. Auxiliary bases (measured)

* ETB (pyscf default for custom bases): built from primitive exponent ranges,
  independent of contraction, l up to 6 -> large for all-electron heavy atoms.
* AutoAux: good for J, poor for K (quartz K 7.4e-4).
* Lehtola auxgen (BSE master, PRs #365/#367; needs sympy): any `size` preset
  forces SVD contraction and l pruning (shells of 22-31 primitives: half the
  functions of ETB, same build time); `contract=False` without a preset gives
  single-primitive shells, which suit rigorous screening.
* For K, all sets tested give 4e-5..7e-4 per small cell; a K-oriented
  auxiliary basis (or local/robust fitting) is an open item.

## 6. Roadmap

1. Done: rigorous envelopes, an envelope family, and the SR 2c metric as
   full - LR (4.4).  The omega dependence is now controlled by precision.
   Open: the auxgen tau = 1e-8 metric has condition number 1e13.  A less
   linearly dependent aux set, or pivoted-Cholesky pruning of its diffuse s
   functions, would make the fit robust.
2. Cost.  Done: blocked out-of-core build; YSO conventional cell in 66 min on
   4 cores.  Left: thread load balance (pair-level parallelism), omega
   choice, and SR/LR overlap.
3. GHF/2c interface: spin-block J/K with the correct alpha-beta slice,
   Kramers symmetry, X2CAMF hcore; `_CDERIS` through `loop()`.
4. K accuracy: auxiliary basis for exchange, robust/local fitting (also cuts
   storage), occ-RI for repeated K builds.
5. Fractional-occupation 2c DFT (needs all-electron XC grids), then CASCI with
   local K_core on the 4f spinors, W_NSM.
6. Later: k-points; periodic CAHF/cMF; optional GPU (gpu4pyscf RSJK passes only
   real Gamma densities).

## 7. Alternatives considered (short)

* k-point CASSCF: not size-extensive with a fixed active space; the dilute
  impurity makes a single local CAS physical.
* Embedded cluster: done before by the user's group, results unsatisfactory.
* GPU4PySCF RSJK: exact K, LinK-style screening, but same estimators and no
  complex Gamma densities; would need a real/imag split and accuracy checks.
* Sharma et al. (RPS + occ-RI + ISDF, multigrid ISDF): K-oriented and fast, but
  built for pseudopotentials on uniform grids; usable for the smooth part.
* Lee et al. all-electron FTC (arXiv 2509.16460): J only (SR Gaussian DF + LR
  plane-wave DF); same split as here.
* Ye et al. smoothed truncated Coulomb (arXiv 2609.00203): candidate for the
  exchange G = 0 treatment in all-electron calculations.

## 8. Practical notes

* Build: `make` (whole lib/); in a container without system BLAS pass
  `CMAKE_ARGS=-DBLAS_LIBRARIES=<pyscf>/lib/libopenblas*.so`.
* Tests: `python pbc/test/test_coulomb_reference.py` (~1.5 min),
  `python pbc/test/test_rsdf.py` (seconds, needs libsocpbcdf.so).
* Benchmarks: `pbc/bench/README.md`; reference J/K cached in `refjk_*.npz`.
* auxgen: `pip install git+https://github.com/MolSSI-BSE/basis_set_exchange.git sympy`.
