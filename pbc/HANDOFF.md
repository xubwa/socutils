# Handoff: periodic two-component Eu:Y2SiO5 (Gamma point, X2CAMF)

State at commit `a4f7683` (branch `ccr-17f4d4cb-lw45cf`).  Details, numbers and
the reasoning behind each choice are in `pbc/DEVELOPMENT.md` (sections in
brackets below).

## Goal and fixed decisions

Goal: all-electron, two-component Eu 4f levels of an Eu dopant in Y2SiO5.
The pipeline is:
1. periodic reference SCF (HF or KS, 4f fractionally occupied);
2. CAHF;
3. CASCI(6e, 14 spinors);
4. later, W_NSM.

Decisions already taken with the user; do not reopen them:

* **Relativity.** X2CAMF with a range-separated X2C-1e (rs1e) scalar + 1e SOC
  part: `mf.x2camf()` (`RangeSeparatedX2CAMFHelper`) -- the short-range
  nuclear potential of the cell decoupled, the long-range rest untransformed,
  the atomic mean-field SOC/Gaunt/Breit one-centre (decision of 2026-10-15,
  replacing the one-centre-only scheme: X2CAMF decouples all nuclei, only its
  2e part is atomic; a full X2C-1e decoupling of the Ewald potential would
  picture-change the background of the nuclei, an O(c^-2) volume-dependent
  error, ~ -4 mEh/atom for fcc Au).  The one-centre approximations remain as
  `mf.x2cmp('x2cmp_1c')` and `mf.x2cmp('x2camf_1c')`.  Other flavors raise on
  purpose.
* **Symmetry and k-points.** No space-group symmetry: the doping breaks it.
  Gamma point only, on a supercell.
* **CAHF.** Single open shell only; no multi-open-shell CAHF.
* **Basis problems** (e.g. Eu atomic DHF not converging with some
  basis): fix them at run time.  The benchmark basis (x2c-SVPall with
  exponents < 0.1 dropped) is a placeholder.
* **Integral strategy.** Do not use one global cderi for everything.
  Target: AO/Fock-driven J/K, an SR-only (local) fit, the LR exact in
  G space, and integral-free wavefunction steps.

## What exists

| piece | file | status |
|---|---|---|
| RSDF: SR (C kernel, rigorous bounds) + LR (pair FTs), Gaussian nuclei | `pbc/rsdf.py`, `lib/pbcdf/sr_int.c` | validated (4, 4.5-4.6b) |
| `lr='gspace'`: SR-only fit + exact LR plane-wave rows, G = 0 term kept out | `pbc/rsdf.py` | validated (4j) |
| periodic spinor HF (KRHF, Kramers occ-RI K, Madelung / G = 0 handling) | `pbc/spinor_scf.py` | validated (4b) |
| rs1e X2CAMF (periodic X2CAMF, `mf.x2camf()`) | `pbc/spinor_scf.py:RangeSeparatedX2CAMFHelper` | validated (4k) |
| one-centre X2CMP / X2CAMF (`x2cmp_1c`, `x2camf_1c`) | `pbc/spinor_scf.py:OneCenterX2CMPHelper` | validated (4b) |
| periodic spinor KS (Becke grids, collinear) | `pbc/spinor_ks.py` | validated (4g) |
| 4f fractional occupation by character, CAS ordering | `scf/addons.py` | done |
| integral-free CAHF (J/K of D_core, D_open only) | `mcscf/cahf_fock.py` | validated (4i) |
| CASCI on RSDF (`_CDERIS`, global cderi) | `mcscf/zcasci.py` | validated |
| local-exchange prototype (SR-PARI + exact LR K) | `pbc/pari.py` | accuracy prototype (4e) |
| dopant supercells (HNF search, Eu-Eu image distance) | `pbc/bench/cells.py` | done (4f) |
| production driver with restartable stages | `examples/14-periodic_dopant_casci.py` | done |

## Running on Rockfish

**Install**

```bash
# checkout must be importable as `socutils`: its parent on PYTHONPATH
git clone -b ccr-17f4d4cb-lw45cf <socutils repo> socutils
pip install pyscf==2.14.0 h5py          # tested with numpy 2.5, scipy 1.18
cd socutils && make                      # all native libs into lib/
# without a system BLAS: make CMAKE_ARGS=-DBLAS_LIBRARIES=<pyscf>/lib/libopenblas*.so
export PYTHONPATH=$(dirname $PWD):$PYTHONPATH
```

The auxiliary basis comes from auxgen, which needs extra packages:
`pip install git+https://github.com/MolSSI-BSE/basis_set_exchange.git sympy`.

**Run**

```bash
export OMP_NUM_THREADS=48
python examples/14-periodic_dopant_casci.py eu_yso_x2 --workdir $SCRATCH/eu_yso_x2 \
       --max-memory 160000            # MB; add --xc PBE0 for KS, --lr gspace --omega 0.3
```

**Stages.** Each stage writes to `--workdir` and is reused on a rerun:

| stage | file |
|---|---|
| RSDF | `cderi.h5` |
| SCF | `scf_<xc>.npz` |
| CAHF | `cahf_<xc>.npz` (log in `cahf.log`) |
| CASCI | `casci_<xc>.npz` |

A job that dies is resumed by resubmitting the same command.  Delete a
stage's file to redo that stage and the ones after it.

**Systems** (`pbc/bench/cells.py`):

| name | atoms | best Eu-Eu image distance | note |
|---|---|---|---|
| `eu_yso` | 32 | 6.82 Å | |
| `eu_yso_x2` | 64 | 8.05 Å | production target first |
| `eu_yso_x3` | 96 | 10.52 Å | |

**Cost** (extrapolated, 4f 4.c/4.d).  The YSO-32 build took 2026 s on 4
cores.  Storage of the global cderi grows as n^3 and fit plus K as n^4.

| cell | cderi size | where it fits |
|---|---|---|
| 64 atoms | ~150 GB | in memory needs --max-memory > 2x that (big-memory node); otherwise streamed from `cderi.h5`, so put it on fast scratch |
| 96 atoms | ~0.5 TB | practical only after the open items below |

## Conventions to keep

The G = 0 bookkeeping is the subtle part (4b, 4j):
* `mf.get_jk` and `loop()` carry the vector integrals only.  That is the
  G = 0-free kernel; with `lr='gspace'` it also excludes -c0 S S.
* The mean field adds the missing terms explicitly through
  `spinor_scf.g0_terms`: J gets -c0 tr(DS) S, K gets (nu_M - c0) S D S.
* `energy_nuc()` carries the wavefunction constants
  -nu_M N/2 - c0 N(N-1)/2.  SCF `energy_tot` takes them back out.
* Any new wavefunction code (CAHF, CASCI, CC) uses `mf.get_jk` and
  `energy_nuc()` as they are.  It must not add Madelung terms again.
* Nuclear model: build RSDF with `nuc_zeta=spinor_scf.nuclear_zeta(cell)`
  and a Gaussian `cell.nucmod`.  `check_nuclear_model` raises on a
  mismatch.

## Open items, in order

1. **Active-space integrals without the global cderi** (4i).
   - (tu|vw) for the 14 4f spinors: SR from one shared aux domain around
     Eu, which is an ordinary variational fit in a subspace; LR exact from
     rho_tu(G).
   - Check against `_CDERIS` (expect ~1e-7 Eh and sub-cm-1 levels).
   - h1eff comes from the converged CAHF core Fock.
2. **SR-only storage that is local.**
   - Either a global SR fit with sparse algebra (sparse Cholesky of M_SR)
     or SR-PARI.
   - First measure how fast L_SR^-1 (P|mn)_SR decays with the aux-pair
     distance, and the SR-PARI K error against domain radius and omega,
     on Y2O3 and quartz (`pbc/pari.py`; quartz with first neighbours at
     omega 1.6 was 9.4e-5 off).
   - With range separation, local fitting has no long-range coupling left.
     What remains is the omega trade-off: small omega needs large domains;
     large omega gives more LR rows and an overlap-like metric.
3. **LR J/K on the fly instead of stored rows** (`lr='gspace'` stores
   ~V (2 omega)^3 rows).
   - J_LR via a smoothed density on a coarse grid
     (erf(wr)/r = 1/r convolved with a Gaussian).
   - K_LR with density-matrix sparsity.
4. W_NSM on the CASCI states (one-electron, active orbitals only).
5. Known loose end: the auxgen tau = 1e-8 aux set has metric condition
   number ~1e13.  Prune it if fits look noisy.

## Notes from the design discussion (K_SR options)

* Range separation removes what makes local fitting dangerous in molecules.
  There, the long-range Coulomb coupling of the fit residuals makes the
  optimal coefficients nonlocal: domain truncation gives first-order errors,
  and the robust form can lose positive definiteness (Merlot et al. 2013).
  With only the erfc kernel fitted, M_SR^-1 decays exponentially, so a
  local fit converges to the global SR fit with domain radius, and
  non-robust PARI (C V_SR C) stays PSD.  Left over: the omega trade-off.
* Pair-independent domains (one domain for all active pairs) are an
  ordinary variational fit: use them for active / fragment integrals.
* The candidates for K_SR:

  | scheme | fit error | storage |
  |---|---|---|
  | global SR fit with sparse algebra | 2nd order, variational | ~linear if M_SR^-1 decays fast |
  | SR-PARI | 1st order, PSD | linear |
  | LinK/ONX-type exact erfc 4c with density screening (as in CP2K HFX) | none | none |

  For LinK, the 4c prefactor for the deep x2c contractions is the question.
  A count of surviving shell quartets and primitive combinations on
  Y2O3/YSO decides between the three (no timing needed).
* occ-RI and Kramers pairing only lower the prefactor (nao/nocc, 2x).  The
  scaling changes only through locality: localized occupied orbitals make
  X^P_mi sparse (the Polly/Werner local-fit exchange), or density screening
  at the 4c level (LinK).

## Tests

No central runner.  Run each script with `PYTHONPATH` set as above:

```bash
for t in pbc/test/test_rsdf.py pbc/test/test_rsdf_gspace.py pbc/test/test_spinor_casci.py \
         pbc/test/test_periodic_cahf.py pbc/test/test_periodic_ks.py \
         pbc/test/test_periodic_x2camf.py pbc/test/test_pari.py \
         scf/test/test_kramers_dfk.py scf/test/test_projected_occ.py; do python $t; done
# slower: pbc/test/test_coulomb_reference.py (~1.5 min)
```

All of these passed at `a4f7683` (4 cores, minutes in total).
