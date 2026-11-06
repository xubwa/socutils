# Handoff: periodic W_NSM results (Eu compounds, Gamma point, X2CAMF)

Results of the periodic SCF -> CAHF -> CASCI(6e, 14 spinors) -> W_NSM runs done so far, collected
2026-11-02 from the run logs.  Driver: `examples/14-periodic_dopant_casci.py`; general pipeline notes in
`pbc/HANDOFF.md`.

## Definition and conventions

* W_NSM: the atomic CFOUR `SOSFMO2C` operator of Eu (directory `aimp/basis_test/cfour_eu/ETB0T10`)
  placed in the AO block of the **active** Eu atom and traced with the full (core + active) density of
  CASCI root 0.  The log line splits it into core and active parts.
* Components in the **Cartesian frame of the cell** as built from the CIF (not a crystal-field frame);
  units as printed by the driver (those of the CFOUR operator).
* One active Eu per cell.  The other Eu atoms of stoichiometric cells are frozen with a fractional 4f
  occupation (`mcscf/frozen_frac.py`: 6/14 per spinor, `FrozenFracSCF`).
* Relativity: rs1e X2CAMF (`mf.x2camf()`), Gaussian nuclei.  The 4f open shell starts from the atomic
  reference `ref4f_ETB0T10.npz` (`--open-ref`).
* J/K: RSDF (omega 0.5, `lr` as noted) unless marked THC.
* `d00` lines: the driver's root 0 -> root 49 (~20.5e3 cm-1) transition quantity for several Eu-centred
  radii rc (0, 2.6, 2.9, 3.5 A); |d| quoted for rc = 0 and 2.9 A.

Run directory root: `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic` (logs next to their launch scripts;
per-run working directories hold `scf_*.npz`, `cahf_*.npz`, `casci_*.npz`, `wnsm_*.npz`, `d00_*.npz`).

## Results

### EuOCl (COD 1542170), convergence with the cell

| run (log) | cell | atoms | basis | E_SCF | E_CAHF | E(CASCI root 0) | 7F1 (cm-1) | W_NSM (x, y, z) | W core z / active z | \|d00\| rc 0 / 2.9 A |
|---|---|---|---|---|---|---|---|---|---|---|
| `euocl/prim.log` | primitive | 6 | basis_rules | -22715.9715727 | -22719.4142167 | -22719.7656319 | 282.6, 372.3, 372.3 | (-0.256, 0.444, **-3507.71**) | -3509.21 / +1.50 | 8.29e-4 / 1.05e-3 |
| `euocl/prim_mixed.log` | primitive | 6 | basis_mixed (Eu1 active) | -22709.2555045 | -22712.7024450 | -22713.0538593 | 284.6, 371.2, 371.2 | (-0.262, 0.452, **-3482.82**) | -3484.48 / +1.66 | 7.93e-4 / 1.03e-3 |
| `euocl/r2r2.log` | sqrt2 x sqrt2 x 1 (`--supercell 1,1,0,-1,1,0,0,0,1`) | 12 | basis_rules | -45431.6806582 | -45438.4415588 | -45438.7933078 | 188.3, 438.6, 438.6 | (-0.264, 0.459, **-8520.20**) | -8507.06 / -13.15 | 6.83e-4 / 9.98e-4 |
| `euocl/mix_221b.log` | 2x2x1 | 24 | basis_mixed (Eu1 active) | -90816.0909606 | -90829.5381904 | -90829.8901417 | 221.2, 412.4, 412.4 | (-0.263, 0.456, **-8408.50**) | -8385.83 / -22.67 | 6.71e-4 / 9.37e-4 |
| `euocl/mix_222.log` | 2x2x2 | 48 | basis_mixed (Eu1 active) | -181625.157050 | -181651.919458 | -181652.271455 | 226.1, 408.1, 408.1 | (-0.285, 0.447, **-8631.52**) | -8606.40 / -25.12 | 7.01e-4 / 9.59e-4 |

### EuOCl primitive cell, J/K and numerical checks (same basis_rules setup as `prim`)

| run (log) | J/K and options | E_CAHF | E(CASCI root 0) | W_z | vs `prim` |
|---|---|---|---|---|---|
| `euocl/prim.log` | RSDF, global cderi | -22719.4142167 | -22719.7656319 | -3507.71 | reference |
| `euocl/prim_3ix.log` | RSDF `lr gspace`, three-index active ERIs | -22719.4142168 | -22719.7656321 | -3508.61 | +0.026% |
| `euocl/prim_lds.log` | RSDF, `--lindep-scaled` | -22719.4142170 | -22719.7656322 | -3508.33 | +0.018% |
| `euocl/prim_thc.log` | THC, Bloch CX fit (alpha_c 4, N_mu 16 n_smooth) | -22719.4144886 | -22719.7659036 | -3510.09 | +0.07% |
| `euocl/prim_thcloc.log` | THC route 1, local CX fit r_cut 6 A | -22719.4144728 | -22719.7658880 | -3510.02 | +0.07% |

The two THC rows predate the fixes of 2026-10-31/11-02 (far-field CX fit, omega 0.35, ISDF cut 1e-10).
Their local fit at r_cut 6 A without the far field is not converged for Eu (32 mEh at a fixed density) and
the 1e-13 ISDF cut gave ~7e-5 Eh thread-count noise; redo them with the current defaults before quoting a
THC W.  The THC W also has x, y components of 0.5 / 7.0 where RSDF gives 0.25 / 0.44: a symmetry-breaking
artefact of that THC build, to recheck.

### Other Eu compounds (primitive cells, basis_rules, RSDF, r2SCAN-relaxed structures from CP2K)

| system (log) | atoms | E_CAHF | E(CASCI root 0) | 7F1 (cm-1) | W_NSM (x, y, z) | \|W\| | \|d00\| rc 0 / 2.9 A |
|---|---|---|---|---|---|---|---|
| EuCl3, COD 1573333 (`eucl3/prim.log`) | 44 | -25325.5023176 | -25325.8548749 | 268.1, 338.3, 423.4 | (-0.275, **1770.44**, -0.410) | 1770.4 | 2.62e-4 / 2.89e-4 |
| EuBr3, COD 2218477 (`eubr3/prim.log`) | 44 | -38171.7771327 | -38172.1297644 | 268.4, 328.0, 437.0 | (-0.266, **2065.12**, -0.402) | 2065.1 | 3.47e-4 / 4.58e-4 |
| Eu(NO3)3, COD 2200124 (`euno3/prim.log`) | 62 | -24235.4768159 | -24235.8297344 | 283.3, 316.3, 432.3 | (2521.77, 1419.71, -761.60) | 2992.5 | 2.66e-4 / 3.13e-4 |

### Eu:Y2SiO5 (32-atom cell, one Eu on a Y site)

| run (log) | structure | E_SCF | E_CAHF | E(CASCI root 0) | 7F1 (cm-1) | W_NSM (x, y, z) | \|W\| | core / active \|W\| |
|---|---|---|---|---|---|---|---|---|
| `eu_yso_T10.log` | `eu_yso` (bench cell), basis_T10 | -37135.1347808 | -37136.8910946 | -37137.2433330 | 186.6, 403.0, 478.0 | (-844.76, 1944.28, -3514.77) | 4104.6 | core (-827.90, 1952.74, -3519.19), active (-16.86, -8.45, 4.42) |
| `yso_scan/prim.log` | `eu_yso_scan_prim.cif`, basis_rules | -37135.3633663 | -37137.1169315 | -37137.4686473 | 206.2, 378.4, 466.2 | (699.93, -2645.56, 3466.53) | 4416.5 | core (707.07, -2627.37, 3476.90), active (-7.14, -18.19, -10.37) |

`eu_yso_T10/old_x2camf_1c/` holds the superseded one-centre X2CAMF run (\|W\| 14631, 7F1 186.5 / 403.3 /
478.1); do not use it.  The two YSO cells differ in structure and frame, so their W vectors are not
comparable component by component.

## Transition dipole d00 (7F0 -> 5D0)

Operator and approximations (unlike W_NSM, not taken from CFOUR):
* transition density of CASCI root 0 -> root 49 (the first state above the 49 7F_J states, ~20.5e3 cm-1),
  active space only: T = C_cas gamma C_cas^H;
* dipole `int1e_r_spinor` (r - R_Eu, no picture change) contracted with T over the AO blocks of the
  atoms (with images) within rc of the active Eu: rc = 0 is the Eu block only (one-centre, like W_NSM),
  rc = 2.6 / 2.9 / 3.5 A add the ligands.  r is not defined on Gamma-point AOs; this works because the
  4f-4f transition density is local.  `q_trans` = the charge of T in the same region (an origin shift
  Delta changes d by -q Delta); the global phase is removed as in the cluster `props_bypass.py`.

| system | root 49 (cm-1) | \|d\| rc 0 | rc 2.6 A | rc 2.9 A | rc 3.5 A | \|q_trans\| rc 3.5 A |
|---|---|---|---|---|---|---|
| Eu:YSO `eu_yso_T10` (from `eu_yso_T10/d00_hf.npz`, computed by `d00_post.py`) | 20536.1 | 2.69e-4 | 4.04e-4 | 4.00e-4 | 3.97e-4 | -- |
| Eu:YSO `yso_scan/prim` | 20495.7 | 3.02e-4 | 4.34e-4 | 4.11e-4 | 4.28e-4 | 6.6e-7 |
| EuOCl primitive (`prim`) | 20431.4 | 8.29e-4 | 1.05e-3 | 1.05e-3 | 1.21e-3 | 7.2e-5 |
| EuOCl sqrt2 x sqrt2 (`r2r2`) | 20500.8 | 6.83e-4 | 9.98e-4 | 9.98e-4 | 1.03e-3 | 1.9e-5 |
| EuOCl 2x2x1 (`mix_221b`) | 20496.7 | 6.71e-4 | 9.37e-4 | 9.37e-4 | 9.33e-4 | 5.9e-6 |
| EuOCl 2x2x2 (`mix_222`) | 20499.6 | 7.01e-4 | 9.59e-4 | 9.59e-4 | 9.24e-4 | 4.9e-6 |
| EuCl3 | 20529.4 | 2.62e-4 | 3.82e-4 | 2.89e-4 | 2.88e-4 | 1.3e-6 |
| EuBr3 | 20533.8 | 3.47e-4 | 4.58e-4 | 4.58e-4 | 3.15e-4 | 8.8e-8 |
| Eu(NO3)3 | 20549.0 | 2.66e-4 | 3.23e-4 | 3.13e-4 | 3.20e-4 | 3.9e-7 |

`eu_yso_T10` d (rc 2.9 A, cell frame) = (5.17e-5, -1.25e-4, 3.77e-4).  The ligand shells matter (rc 0 ->
2.6 A: +30-50%).  Past the first shell d is stable within 2-4% for YSO and Eu(NO3)3.  In the EuOCl
primitive cell it is still moving; EuBr3 jumps at 3.5 A.  Candidates for a cleaner d: larger rc, an X2C
picture-change dipole, or a velocity-gauge (periodic) transition moment.

## Code and file locations

| what | where |
|---|---|
| W_NSM operator: read CFOUR, embed in the Eu block, trace | `pbc/atomic_prop.py`: `load_cfour_atom` (l. 19), `atomic_block_operator` (l. 45), `expectation` (l. 56) |
| local dipole over an image cluster | `pbc/atomic_prop.py`: `image_cluster` (l. 61), `local_dipole` (l. 81), `real_axis` (l. 104) |
| driver stages | `examples/14-periodic_dopant_casci.py`: W_NSM l. 397, d00 l. 413 |
| CFOUR atomic integrals | `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/basis_test/cfour_eu/ETB0T10/SOSFMO2C` (with `config.py`, `GENBAS`) |
| per-run results | `<workdir>/wnsm_hf.npz` (w, w_core, dm), `<workdir>/d00_hf.npz` (root, gamma, d, rc_A, e_rel) |
| d00 for a finished workdir | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/d00_post.py WORKDIR SYSTEM [--basis-py ...]` |

## Full paths (all checked to exist on 2026-11-02)

Root `P = /scratch4/lcheng24/xwang405/zmcscf/Eu/periodic`.  Every working directory holds `scf_hf.npz`,
`cahf_hf.npz`, `cahf.log`, `casci_hf.npz` (with ci), `wnsm_hf.npz`, `d00_hf.npz`; some also hold
`cahf_hf_partial.npz` or `cderi.h5`.

| run | log | working directory | launch script |
|---|---|---|---|
| Eu:YSO bench cell | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eu_yso_T10.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eu_yso_T10/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/run_T10.sh` |
| Eu:YSO scan prim | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/yso_scan/prim.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/yso_scan/prim/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/yso_scan/run.sbatch` |
| EuOCl prim | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run.sbatch` |
| EuOCl prim, mixed basis | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_mixed.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_mixed/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_prim_mixed.sbatch` |
| EuOCl prim, lr gspace + 3-index | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_3ix.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_3ix/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_3ix.sbatch` |
| EuOCl prim, lindep-scaled | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_lds.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_lds/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_lds.sbatch` |
| EuOCl prim, THC (Bloch) | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_thc.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_thc/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_thc.sbatch` |
| EuOCl prim, THC local fit | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_thcloc.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/prim_thcloc/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_thcloc.sbatch` |
| EuOCl sqrt2 x sqrt2 x 1 | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/r2r2.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/r2r2/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_r2r2.sbatch` |
| EuOCl 2x2x1, mixed basis | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/mix_221b.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/mix_221b/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_mix_221b.sbatch` |
| EuOCl 2x2x2, mixed basis | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/mix_222.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/mix_222/` (guess from `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/mix_221/`) | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/run_mix_222.sbatch` |
| EuCl3 | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eucl3/prim.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eucl3/prim/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eucl3/run.sbatch` |
| Eu(NO3)3 | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euno3/prim.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euno3/prim/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euno3/run.sbatch` |
| EuBr3 | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eubr3/prim.log` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eubr3/prim/` | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/eubr3/run.sbatch` |

Inputs:

| input | full path |
|---|---|
| EuOCl CIF | `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/struct/oxyhalides/COD_1542170.cif` |
| EuCl3 CIF + relaxed xyz | `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/refs/COD_1573333.cif`, `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/periodic_cp2k/eucl3_hopt_r2scan/eucl3_r2scan_final.xyz` |
| Eu(NO3)3 CIF + relaxed xyz | `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/refs/COD_2200124.cif`, `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/periodic_cp2k/euno3_hopt_r2scan/final.xyz` |
| EuBr3 CIF + relaxed xyz | `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/refs/COD_2218477.cif`, `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/periodic_cp2k/eubr3_hopt_r2scan/final.xyz` |
| Eu:YSO scan cell | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/yso_scan/eu_yso_scan_prim.cif` |
| Eu:YSO bench cell | built in code: `SYSTEMS['eu_yso']` in `pbc/bench/cells.py` (l. 219) |
| basis files | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/basis_T10.py`, `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/basis_rules.py`, `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/basis_mixed.py` |
| atomic 4f reference (`--open-ref`) | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/ref4f_ETB0T10.npz` |
| CFOUR NSM operator (`--nsm-dir`) | `/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/basis_test/cfour_eu/ETB0T10/SOSFMO2C` (dir also has `config.py`, `GENBAS`) |
| d00 post-processing | `/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/d00_post.py` |
| driver | `/home/xwang405/apps/socutils-git-pbc-env/socutils/examples/14-periodic_dopant_casci.py` |
| operator code | `/home/xwang405/apps/socutils-git-pbc-env/socutils/pbc/atomic_prop.py` |

## Observations

* **W is dominated by the core.** The active (4f) part is +1.5 to -25 against \|W\| ~ 3500-8600 in EuOCl.
* **EuOCl cell convergence is not reached at the primitive cell.**
  - W_z goes from -3508 (primitive) to -8520 (sqrt2 x sqrt2), -8409 (2x2x1, mixed basis) and -8632
    (2x2x2, mixed basis), so the primitive cell is far off.
  - 2x2x1 -> 2x2x2 still changes W_z by 2.7%.
  - In the primitive cell the frozen Eu partners are close images of the active Eu, which is the likely
    reason.
  - The 7F1 levels move the same way: 283/372/372 -> 226/408/408 cm-1.
* **Basis:** mixed vs rules basis changes the primitive W_z by 0.7% (-3483 vs -3508).
* **Integrals:** RSDF variants agree to 0.03%. The old THC builds agree to 0.07% in W_z; see the caveat above.
* **Lattice symmetry:** EuCl3 / EuBr3 have W along y only and EuOCl along z only (lattice symmetry);
  Eu(NO3)3 and YSO have no such constraint.

## THC + MPI pipeline (2026-11)

`examples/14-periodic_dopant_casci.py --jk thc` runs SCF -> CAHF -> CASCI -> W_NSM / d00 with the THC
route-1 J/K (`pbc/thc.py` `THCJK`, `pbc/thc_gapw.py`, `pbc/thc_local.py`, `lib/thc` -> `lib/libsocthc.so`)
and no cderi.  Under `mpirun` it is SPMD: every rank runs the script, the THC build and every J/K
(SCF, CAHF, CASCI core) are split over the ranks, rank 0 broadcasts orbitals/densities and is the
only one that prints and writes.  The active (tu|vw) come from the RSDF three-index route on rank 0.

    export OMP_NUM_THREADS=24 OMP_PLACES=cores OMP_PROC_BIND=close I_MPI_PIN=1 I_MPI_PIN_DOMAIN=socket
    mpirun -np 2 python -u -m mpi4py examples/14-periodic_dopant_casci.py CIF --basis-py B.py \
        --workdir W --jk thc --thc-local-rcut 6 --thc-alpha-c 4 --omega 0.35 --max-memory 80000 \
        --lr gspace --active-eri three-index --nroots 60 --open-ref REF.npz --nsm-dir CFOUR_DIR

(2 ranks per node, one per socket; `--max-memory` is per rank; `python -m mpi4py` aborts all ranks
on an exception.)  Defaults now: streamed build, phi distributed by grid rows under MPI, far-field CX
fit, ISDF cut 1e-10, uniform-grid cutoff precision 1e-12, space-group-symmetric grid and orbit pivots
whenever the cell has symmetry (spglib; P1 cells unchanged).

EuOCl primitive, symmetric THC (N_mu 16) vs RSDF `prim_3ix` (run `euocl/prim_thcsym`, log
`euocl/prim_thcsym.log`): 7F1 282.5 / 372.4 / 372.4 vs 282.6 / 372.3 / 372.3 cm-1, W (-0.262, 0.454,
-3510.4) vs (-0.255, 0.445, -3508.6), d00 1.052383e-3 vs 1.052370e-3.  Without the symmetric grid
the THC broke the C4 site symmetry (`euocl/prim_thcmpi`: W_y 122, 7F1 E split 3.3 cm-1; from the
ISDF grid / greedy points, not MPI, omega or the far fit, `euocl/wabl_A..D`).  A P1 cell keeps that
error: on the finer default grid the forbidden W_y of EuOCl is still -35 (`euocl/wabl_I`), so take
~1% of |W| as the THC error of W at a site without symmetry (Eu:YSO).

Caveats: BLAS thread count alone moves E_K by ~1e-6 Eh (ISDF cut 1e-10), so compare runs at equal
thread counts.  The uniform-grid cutoff decides ~1e-4 Eh in Eu cells (smooth h shells; EuOCl +4.6e-4 at
precision 1e-10, mesh 85x85x143); a mesh scan (`euocl/wabl_M88..M93`) was running at the hand-off.
Validation scripts: `pbc/bench/thc_tests/` (`quartz_scf_mpi.py`, `fixed_density_mpi.py`,
`w_nsm_scf_density.py`, `symm_pivots_check.py`, `gram_conditioning.py`, `pivot_residual_curve.py`).

## Open items

1. Eu:YSO 32 atoms with THC + MPI vs RSDF `eu_yso_T10` (queued: `run_yso_thcmpi.sbatch`, 4 nodes x 2
   ranks), then the 64-atom supercell; EuOCl 2x2x1 / 2x2x2 with THC.
2. EuOCl cell convergence beyond 2x2x2 (3x3x2 or larger, now feasible with THC), and the same
   cell-size check for EuCl3 / EuBr3 / Eu(NO3)3 (only primitive cells so far).
3. Express W in a common frame (crystal-field axes of the Eu site) to compare compounds.
