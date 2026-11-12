# Periodic J/K accuracy and cost benchmarks (pyscf 2.14, Gamma point)

Scripts used to decide how socutils should build periodic J and K for
all-electron heavy-element crystals (target: Eu:Y2SiO5).  The validated,
numerically exact references live in `../coulomb_reference.py` and are
unit-tested in `../test/`; these benchmarks are measurements, not tests.

| script | question |
|---|---|
| `cells.py` | benchmark crystals (alpha-quartz, A-type Y2O3, 32-atom Y2SiO5) and basis trimming |
| `rsjk_precision.py` | how accurate is pyscf RSJK J vs `cell.precision`, and what does it cost? |
| `gdf_aux.py etb\|auxgen\|j2c` | GDF J and K error vs auxiliary basis, and the metric handling |
| `gdf_zero_dim.py` | fitting error vs periodic-integral error (YH3 in a box) |
| `../test/check_zero_dim_pyscf.py` | ZnH2 in boxes: pyscf RSJK J passes, GDF J has a fitting offset |

Basis: x2c-SVPall with primitives below 0.1 removed (`cells.trim_basis`;
`cell.exp_to_discard` crashes when it empties a shell).  Density: minao guess.
Machine: 4 cores.  Errors are E - E_ref in Hartree; "K dE" is the exchange
energy -1/4 tr(DK).

## RSJK J vs precision (`rsjk_precision.py`)

| system | nao | prec 1e-6 | 1e-8 (default) | 1e-10 | omega spread at 1e-12 | time at 1e-10 |
|---|---|---|---|---|---|---|
| alpha-quartz (max exp 7e3) | 135 | 7e-8 | 1.3e-8 | 1e-10 | 2e-10 | 109 s |
| A-Y2O3 (max exp 7e5) | 128 | **-0.11** | **-0.085** | 1.3e-9 | 7.5e-10 / 6.2e-9 | 585 s |

With Y the default precision is silently wrong by 85 mEh; it snaps to 1e-9 only
at 1e-10.  Small omega makes RSJK far slower (omega 0.5: 3686 s on Y2O3) and on
ZnO (omega 0.1) the supercell shell-pair index needed 65 GB.  The 32-atom YSO
cell (692 AO) did not finish one J at prec 1e-6 in 9 min.

## GDF J and K vs auxiliary basis (`gdf_aux.py`)

| system | aux | naux | J dE | K dE | max dK | build |
|---|---|---|---|---|---|---|
| quartz | ETB 2.0 / 1.7 / 1.5 / 1.3 | 1209-2715 | -3.0e-4 / -5.2e-5 / -6.5e-6 / -1.1e-6 | 4.6e-4 / 1.7e-4 / 9.2e-5 / 4.4e-5 | 3e-4 to 7e-5 | 33-109 s |
| quartz | AutoAux | 1122 | -5.6e-5 | 7.4e-4 | 1.2e-3 | 28 s |
| quartz | auxgen uncontracted 1e-7 / 1e-9 | 1530 / 1761 | -1.5e-5 / -2.2e-6 | 2.4e-4 / 2.0e-4 | 3e-4 | 59-67 s |
| Y2O3 | ETB 2.0 / 1.7 / 1.5 | 1221-2057 | -2.6e-4 / -4.8e-5 / -1.3e-5 | 6.3e-4 / 1.5e-4 / 5.5e-5 | 5e-4 to 1.6e-4 | 80-137 s |
| Y2O3 | ETB 1.3 | 2785 | **+1.3e-5** | **-1.7e-4** | 7.9e-4 | 212 s |
| Y2O3 | auxgen large / verylarge (SVD, up to 31 prims/shell) | 766 / 873 | -7.6e-5 / -5.8e-5 | 2.6e-4 / 1.5e-4 | 4e-4 | 135 s |
| Y2O3 | auxgen uncontracted 1e-7 / 1e-8 / 1e-9 | 1735-2003 | **+0.57 / +6.7 / +5e-4** | **-2.5e-2 / -0.29 / -5.8e-4** | up to 8e-2 | 120-148 s |

* K converges far more slowly than J with the auxiliary basis.
* A positive J error (or negative K error) violates the Coulomb-metric
  variational bound: it is numerical failure, not fitting error.
* `size` presets of auxgen force SVD contraction (shells of 22-31 primitives):
  half the functions of ETB, same build time.

## Is it the metric? (`gdf_aux.py y2o3A j2c`)

Forcing eigendecomposition of j2c drops the same 373 directions at any
threshold (1e-10..1e-6), i.e. the periodic j2c has negative eigenvalues; the
remaining J error is still positive (+2.2e-3).  The periodic j2c/j3c are not
accurate or mutually consistent for compact auxiliary functions in a dense
lattice; thresholds cannot repair it.

## Is it the auxiliary basis? (`gdf_zero_dim.py`)

| aux | naux | molecular DF dE_J | periodic GDF in box dE_J |
|---|---|---|---|
| ETB 2.0 | 567 | -2.8e-5 | -2.5e-5 |
| ETB 1.3 | 1028 | -2.7e-6 | +1.1e-6 / -3.1e-6 |
| auxgen uncontracted 1e-8 | 807 | -4.1e-6 | -1.1e-6 / +5.9e-5 |

In vacuum the same sets behave.  The failure needs a dense lattice: the
lattice-sum truncation for compact functions (estimators) is what breaks, the
same mechanism as the RSJK precision failure above.

## Consequences for socutils

1. The hard part of periodic K for all-electron heavy atoms is rigorous
   lattice-sum error control for compact functions, not the auxiliary basis.
2. Validation needs a dense-lattice reference for compact functions; the zero-
   dimensional test does not exercise lattice sums.  Today the only such
   reference is RSJK at prec 1e-12 with an omega check (~1e-8, slow).
3. auxgen uncontracted sets fit well (molecular DF) and are single-primitive
   shells, which suit rigorous screening.
