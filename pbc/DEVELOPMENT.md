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

**Next step for robustness:** a rigorous envelope.  For any r,
r^l exp(-a r^2) <= (l/(a e))^(l/2) exp(-a r^2/2), so every primitive is bounded
pointwise by an s Gaussian of half the exponent; with the positive erfc kernel
the s-s interaction of the envelopes is then a true upper bound (with bin
amplitudes summed rather than maximised).  The current bound has no angular
factor, which is the likely reason Y (f AOs, aux up to l = 6) needs the
conservative aux measure.

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

### 4.7 Known limitations

* Build cost (4 cores, defaults): quartz ~1 min, Y2O3 ~3 min, comparable to
  pyscf GDF with the same auxiliary basis; SR 3c (2-4e7 shell triples with
  high-l uncontracted aux) and the LR transforms dominate.
* Storage is dense global (naux x nao^2): fine up to the YSO conventional cell
  on a large-memory node, not beyond.
* K accuracy is limited by the auxiliary basis (quartz 2e-4 with auxgen tau
  1e-8): K converges much more slowly than J.
* Gamma point only.  Complex densities: K is built as K(Re D) + i K(Im D)
  with real vectors.  Spin-orbital (2c) densities: `RSDF.get_jk_ghf` /
  `rsdf.ghf_get_jk` assemble the spin blocks with the correct alpha-beta slice
  (tested against pyscf's molecular GHF for Hermitian and non-Hermitian
  non-collinear densities).  No Kramers-symmetry savings yet.

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

1. Rigorous angular envelope in the bounds (4.4); omega independence on Y2O3.
2. Cost: mu <-> nu symmetry in the 3c kernel, tighter bounds (angular factors),
   omega choice, LR blocking; benchmark against pyscf GDF/RSJK incl. YSO.
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
