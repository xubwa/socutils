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

## Open items

1. Redo the EuOCl primitive THC check with the current THC defaults (far-field fit, omega 0.35,
   ISDF cut 1e-10), then a THC 2x2x1 / 2x2x2 run (MPI driver `pbc/bench/thc_tests/thc_mpi_scf.py`).
2. EuOCl cell convergence beyond 2x2x2 (3x3x2 or larger, now feasible with THC), and the same
   cell-size check for EuCl3 / EuBr3 / Eu(NO3)3 (only primitive cells so far).
3. Express W in a common frame (crystal-field axes of the Eu site) to compare compounds.
