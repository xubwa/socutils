# Relativistic multireference correlation methods on O2, NdO+ and UO+: an assessment

Technical report, state of 2026-10-07.  Sources: `docs/calculations.md`
(sections 1 and 6) and `~/scr4/zmr_runs/dsrg/STATUS.md`; every number below
is taken from those records or from the logs they name.  Energies in cm-1
unless stated.

## 1. Summary

Spinor-based (two-component, spin-orbit at the orbital level) multireference
methods were applied to three problems of very different size: the
zero-field splitting of O2 (4 cm-1), and the Omega levels of the f^3 ground
multiplet of NdO+ and UO+ (500 - 2000 cm-1).  The methods are PC-NEVPT2 and
internally contracted MRCI / MR-AQCC of this package, and DSRG-MRPT2,
DSRG-MRPT3 and MR-LDSRG(2) of forte2 on the same integrals.

1. **O2.**  The basis-set limit of the spin-orbit part is 2.46 (CASSCF,
   minimal active space) and needs tight p primitives, nothing else; the
   spin-spin part is +1.55 in every basis.  Their sum, 4.01, is within 0.05
   of experiment (3.96) by cancellation: dynamical correlation lowers both
   parts, and the best correlated values are 3.74 - 3.78.  Second-order
   perturbation theory lowers the spin-orbit part four to seven times too
   much.  The non-perturbative methods (MRCI, AQCC, MR-LDSRG(2)) agree
   within 0.1 and are insensitive to the state average of the orbitals.
2. **NdO+ and UO+.**  All methods give the order 4.5 < 3.5 < 2.5 < (1.5,
   0.5) and the onset of the J = 5.5 group.  None determines the positions
   to better than about 100, and the sign of the Omega 0.5 - 1.5 gap of UO+
   (experiment +40) is not determined: -82 to +101 over the methods and
   settings.  For NdO+ every method except AQCC puts Omega 3.5 220 - 470
   below the quoted experimental level.
3. **Why.**  Each method fails in its own, identifiable way once the metal
   semicore is correlated: MRCI is not size-extensive (reference weight
   0.88); its internal contraction depends on the number of reference
   states; ACPF / CEPA(0) are undefined because excited configurations lie
   below the reference energy; DSRG damps exactly the low-denominator
   excitations (5f -> 6d, 7s) that carry the differential correlation, and
   the result keeps a dependence on the flow parameter of 20 - 60 even at
   the non-perturbative level.
4. **Consequence.**  Sensitivity to reasonable choices (state average, core,
   flow parameter, functional, number of roots) is the error bar.  On this
   measure the O2 result is a result (3.76 +- 0.02 from correlation
   treatment, 0.2 below experiment, cause not located); the UO+ gap and the
   NdO+ positions are not.

## 2. Methods and settings

| method | code | remarks |
|---|---|---|
| SA-CASSCF | socutils (`zmcscf-qn` branch: SO-SCI + L-BFGS) | spinor orbitals, Kramers-restricted start |
| PC-NEVPT2 | zmr | union contraction over the model states, state-specific E0 |
| ic-MRCI, +Q, MR-AQCC | zmr (`zmr/mrci`) | product basis, fully internally contracted, relaxed reference; several roots together or one root per Omega with root following |
| DSRG-MRPT2 / MRPT3 | forte2 (branch `mrpt3`), through an adapter | one transformation for the state-averaged ensemble, reference relaxed once |
| MR-LDSRG(2) | forte2, local branch `zmr-interface` (`rel_ldsrg2.py`, not pushed) | two-component, NIVO approximation, written in the DSRG session |

Hamiltonians: X2CMP with Coulomb, Gaunt and Breit two-electron picture
change for O2; X2CAMF (Gaunt, Breit) for NdO+ and UO+; Cholesky vectors with
a threshold of 1e-7 throughout.  forte2 reads the same Cholesky vectors,
one-electron operator and spinor coefficients: its CASCI reproduces the zmr
CASCI to 2e-12 Eh (O2) and 3e-11 Eh (NdO+, UO+, ten states).

Status of the evidence.  The zmr MRCI is validated against determinant
MRCISD and against ORCA (N2, 3e-7 Eh).  The MR-LDSRG(2) solver is validated
on small molecules only (kernels equal forte2's to 1e-15, N2 correlation
energy equals forte's to 6e-10); its NdO+ / UO+ numbers are single runs and
are preliminary.  NIVO changes excitation energies of OH / O2 by 8 - 11.
The experimental levels of NdO+ and UO+ are those supplied for this work;
their Omega assignment may itself rest on calculations.

## 3. O2: zero-field splitting X(+-1) - X(0+)

r = 1.20752 A, minimal active space (2 electrons in the four pi* spinors),
SA-3 orbitals (the three components of the ground state), O 1s frozen,
virtual spinors up to about 100 Eh, uncontracted basis sets.  Experiment
3.96.

### 3.1 Basis-set limit

| basis | largest p exponent | CASSCF | PC-NEVPT2 | spin-spin (CASCI) |
|---|---|---|---|---|
| aug-cc-pVTZ | 34 | 2.307 | 1.569 | +1.545 |
| aug-cc-pVQZ | 63 | 2.397 | 1.661 | +1.549 |
| aug-cc-pV5Z | 196 | 2.452 | 1.716 | +1.550 |
| aug-cc-pV6Z | 526 | 2.462 | | |
| TZ + 1 / 2 / 3 tight p (geometric) | 153 / 682 / 3031 | 2.436 / 2.457 / 2.458 | 1.656 / 1.671 / 1.672 | +1.548 |
| TZ + 1 / 2 tight p (optimised) | 226 / 1223 | 2.449 / 2.458 | 1.665 / 1.672 | +1.547 / +1.548 |
| QZ + 1 tight p (optimised) | 418 | 2.459 | 1.704 | +1.549 |
| s, p of 7Z + d, f of TZ | 1349 | 2.465 | 1.679 | +1.550 |
| s, p of 7Z + d, f, g of QZ | 1349 | 2.464 | 1.710 | +1.550 |
| dyall av3z / av4z | 194 / 557 | 2.453 / 2.462 | 1.668 / 1.708 | +1.549 / +1.550 |

* The basis-set dependence is in the spin-orbit matrix element, i.e. in the
  O 2p spinors near the nucleus.  The limit is 2.46; three independent
  routes agree (6Z, 7Z s and p functions, Dyall's quadruple zeta).  Higher
  angular momenta do not matter for this part (2.465 against 2.464).
* The spin-spin term changes by 0.005 from TZ to 5Z.
* Energy-optimised correlation-consistent sets stop at p exponents that
  describe the valence shell.  One optimised tight p function brings QZ to
  2.459; two bring TZ to 2.458.  The exponents come from the atomic
  four-component 2p splitting (X2CAMF as engine, basis-optimisation session,
  preliminary); the atomic ratio squared predicts the molecular value to
  0.002 - 0.005.
* A geometric extrapolation of TZ / QZ / 5Z gave 2.54 and was wrong: the
  convergence is not that of a correlation energy.
* The full aug-cc-pV7Z calculation did not run: the Cholesky decomposition
  stalled at a residual of 1.0e-5 (threshold 1e-7).  Not repeated.

### 3.2 Correlation

| method | TZ, spin-orbit only | TZ + 2 tight p, spin-orbit only | TZ + 2 tight p, with spin-spin | QZ, with spin-spin |
|---|---|---|---|---|
| CASSCF (+ CASCI spin-spin) | 2.307 | 2.457 | 4.005 | 3.946 |
| PC-NEVPT2 | 1.569 | 1.671 | | |
| MRCI | 2.140 | 2.279 | 3.738 | 3.707 |
| MRCI+Q (Davidson) | 2.187 | 2.334 | 3.761 | 3.724 |
| MRCI+Q (renormalised) | 2.198 | 2.348 | 3.769 | 3.730 |
| MR-AQCC | 2.206 | 2.350 | 3.777 | 3.743 |
| DSRG-MRPT2, s = 0.5 (0.1 - 2.0) | 0.99 (1.83 - 0.63) | | | |
| DSRG-MRPT3, s = 0.5 (range) | 2.46 (2.28 - 2.92) | | | |
| MR-LDSRG(2) | 2.12 - 2.21 | | | |

"With spin-spin": the four-component Coulomb and Breit integrals of the
active space replace the two-component ones in the all-active part of H, so
the term is evaluated with the correlated wave function.

* Correlation lowers the spin-orbit part by 0.11 - 0.18 (MRCI to AQCC) and
  the spin-spin part by 0.09 - 0.12 (1.459 for MRCI, 1.427 for AQCC, against
  1.548 at the CASCI level).  The uncorrelated sum 4.01 therefore agrees
  with experiment by cancellation of missing terms.
* The best values, 3.74 - 3.78, are 0.2 (5 %) below experiment.  Not
  examined: correlation of O 1s, the virtual cutoff, the convergence of the
  correlation part with the angular momentum of the basis (TZ to QZ changes
  it by 0.03), vibrational averaging, and the first-order treatment of the
  spin-spin term.  Which of these holds the 0.2 is not known.
* PC-NEVPT2 lowers the spin-orbit part by 0.74 - 0.79 in every basis, four
  to seven times the MRCI / AQCC lowering; DSRG-MRPT2 has no plateau in the
  flow parameter; DSRG-MRPT3 raises the value.  Perturbation theory in the
  minimal space does not give the sign and size of the correlation effect.
* With SA-6 orbitals (CASCI 1.83 instead of 2.31) MR-LDSRG(2) gives 2.06 -
  2.21: the non-perturbative methods remove the dependence on the state
  average, second order does not (NEVPT2 1.83 / 1.68).
* Full valence space (8 electrons in 12 spinors, TZ): CASSCF 2.817, NEVPT2
  2.038, MRCI 2.069, MRCI+Q 1.805, AQCC 1.821; the spin-spin correction is
  +0.90 at the CASCI level.  Here the size-consistency correction moves the
  splitting by 0.25, and MRCI is not converged with respect to it.

## 4. NdO+ and UO+: the f^3 ground multiplet

Three electrons in the 14 4f / 5f spinors; contracted bases (Nd, U:
DYPVTZ-SO; O: PVTZ-X2C); NdO+ at 1.742 A, 382 spinors, SA-52 orbitals; UO+
at 1.803 A, 450 spinors, SA-10 orbitals.  "26 core": Nd 4d 5s 5p or U 5d 6s
6p, and O 2s 2p; "O 2p": six core spinors.  Levels above Omega = 4.5,
vertical.

### 4.1 UO+

| method | core | 3.5 | 2.5 | 1.5 | 0.5 | 0.5 - 1.5 |
|---|---|---|---|---|---|---|
| experiment | | 764.9 | 1132.4 | 1284.5 | 1324.9 | +40 |
| CASCI | | 567 | 1132 | 1334 | 1321 | -14 |
| PC-NEVPT2 | none | 561 | 1021 | 1197 | 1207 | +10 |
| PC-NEVPT2 | O 2p | 559 | 924 | 1078 | 1106 | +28 |
| PC-NEVPT2 | 26 | 700 | 1072 | 1180 | 1178 | -2 |
| MRCI, 5 roots | none | 706 | 1119 | 1310 | 1366 | +56 |
| MRCI, 5 roots | O 2p | 585 | 964 | 1143 | 1211 | +68 |
| MRCI, 5 roots | 26 | 655 | 1165 | 1338 | 1346 | +8 |
| MRCI+Q, 5 roots | 26 | 662 | 1133 | 1316 | 1364 | +48 |
| MRCI, one root per Omega | 26 | 635 | 1167 | 1327 | 1318 | -9 |
| MR-AQCC, one root per Omega | 26 | 753 | 1267 | 1454 | 1526 | +71 |
| DSRG-MRPT2, s = 0.5 / 1.0 | 26 | 644 / 584 | 1041 / 899 | 1193 / 1059 | 1210 / 1112 | +17 / +54 |
| DSRG-MRPT3, s = 0.5 / 1.0 | 26 | 624 / 597 | 1188 / 1160 | 1338 / 1307 | 1291 / 1256 | -47 / -51 |
| MR-LDSRG(2), s = 0.5 / 1.0 | 26 | 626 / 601 | 1130 / 1072 | 1285 / 1227 | 1263 / 1216 | -22 / -10 |
| DSRG-MRPT3, s = 0.5 / 1.0 | O 2p | 508 / 484 | 1039 / 991 | 1217 / 1168 | 1200 / 1160 | -17 / -8 |
| MR-LDSRG(2), s = 0.5 / 1.0 | O 2p | 515 / 497 | 1043 / 999 | 1221 / 1174 | 1204 / 1165 | -17 / -9 |
| MR-LDSRG(2), s = 2.0 / 4.0 | O 2p | 509 / 560 | 983 / 1007 | 1155 / 1174 | 1153 / 1175 | -1.5 / +1.0 |

Higher roots (MRCI without core, 11 roots together): (1)5.5 4058, (2)4.5
4591, second 3.5 4932, (2)2.5 5128; experiment 4177.8, 4758.5, 5162.0,
5219.4.  The observed (2)3.5 at 4982.4 is not a 5f^3 state and is outside
this reference space.

### 4.2 NdO+

| method | core | 3.5 | 2.5 | 1.5 | 0.5 |
|---|---|---|---|---|---|
| experiment | | 1280 | 1538 | 1624 | 1719 |
| CASCI | | 890 | 1165 | 1191 | 1174 |
| PC-NEVPT2 | 26 | 940 | 1214 | 1245 | 1237 |
| MRCI, 5 roots | 26 | 961 | 1256 | 1305 | 1318 |
| MRCI+Q, 5 roots | 26 | 1006 | 1348 | 1420 | 1450 |
| MRCI / +Q, 5 roots | 16 (no 4d) | 984 / 1059 | 1306 / 1426 | 1361 / 1509 | 1375 / 1551 |
| MR-AQCC, one root per Omega | 26 | 1385 | 1864 | 1952 | 2013 |
| DSRG-MRPT2, s = 0.5 / 1.0 | 26 | 870 / 810 | 1176 / 1066 | 1293 / 1218 | 1305 / 1260 |
| DSRG-MRPT3, s = 0.5 / 1.0 | 26 | 993 / 964 | 1335 / 1342 | 1401 / 1425 | 1406 / 1428 |
| MR-LDSRG(2), s = 0.5 / 1.0 | 26 | 948 / 932 | 1290 / 1273 | 1384 / 1391 | 1392 / 1415 |
| DSRG-MRPT3, s = 0.5 / 1.0 | O 2p | 840 / 780 | 1236 / 1150 | 1312 / 1253 | 1275 / 1229 |
| MR-LDSRG(2), s = 0.5 / 1.0 | O 2p | 838 / 792 | 1233 / 1179 | 1313 / 1280 | 1277 / 1256 |

### 4.3 Sensitivity to the choices

| choice varied (UO+ unless noted) | effect |
|---|---|
| orbitals SA-10 / SA-52 / SA-80, PC-NEVPT2, 26 core | Omega 3.5: 700 / 716 / 761; 2.5: 1072 / 1092 / 1178 |
| orbitals SA-10 / SA-52, MRCI without core | 3.5: 706 / 742; 2.5: 1119 / 1162 |
| bond length | 3.5 moves by 35 - 40 per 0.03 A |
| number of reference states 1 / 5 / 10, MRCI, O 2p core | 0.5: 1192 / 1211 / 1162; gap +40 / +68 / +37 |
| 5 roots together / one root per Omega, MRCI, 26 core | gap +8 / -9 |
| MRCI / MRCI+Q / AQCC, 26 core | 3.5: 633 - 753; 0.5: 1316 - 1526 |
| flow parameter 0.5 -> 1.0, MRPT2, 26 core | up to -142 |
| flow parameter 0.5 -> 1.0, MRPT3, 26 core | -27 to -35; NdO+ -29 to +24 |
| flow parameter 0.5 -> 1.0, MR-LDSRG(2), 26 core | -25 to -58; NdO+ -17 to +23 |
| 7s pair in the active space, SA-80, MRPT2 / MRPT3, 26 core | gap +33 / +101 and -76 / -82; lowest 7s state 5070 - 12243 |

Stable in all calculations: the order of the levels and the onset of the
J = 5.5 group near 4100 - 4300.  Not determined: positions to better than
about +-100, and the sign of the UO+ gap.

## 5. Analysis of the methods

### 5.1 Second-order perturbation theory

PC-NEVPT2 with O 2p or no core gives a small positive gap for UO+ (+28,
+10) and loses it when the 26 core spinors are correlated (-2); DSRG-MRPT2
moves by up to 140 with the flow parameter.  In O2 the second-order lowering of
the spin-orbit part is several times too large.  In both cases the quantity
of interest is a small difference of large, nearly equal correlation
energies of states of the same configuration, and second order is not
accurate to that difference.

### 5.2 MRCI and its functionals

* **Size-extensivity.**  With 26 core spinors the reference weight is 0.88
  and the Davidson correction is 0.12 Eh, 12 % of the correlation energy.
  MRCI, MRCI+Q and AQCC then differ by 100 - 700 in the levels (NdO+ Omega
  3.5: 961, 1006, 1385).  The level differences inherit an unlinked
  contribution that is different for each Omega.
* **Dependence of the internal contraction on the reference states.**  In
  the classes with core indices the contracted functions are built from the
  reference states (ccvv uses 5 of the 364 active functions, cavv 53 of 91,
  ccav 70 of 1001); reference states of other Omega contribute because the
  external operators carry Omega.  With the O 2p core, 1 / 5 / 10 reference
  states give 1192 / 1211 / 1162 for Omega 0.5.  Without core indices the
  contracted space is complete and the three agree exactly.
* **ACPF and CEPA(0) are not defined.**  The lowest eigenvalue of H outside
  the reference space (5f -> virtual singles, mostly 6d and f type) is
  +0.036 Eh above the reference energy without core correlation and -0.013
  Eh with the O 2p core.  H - E_ref is then not positive definite and
  functionals with a small weight of the correlation norm collapse.  AQCC
  survives but overshoots (UO+ gap +71, NdO+ levels 100 - 300 above
  experiment).
* **Root following.**  Several roots of one Omega per calculation did not
  converge for four of six Omega; all roots together with following of the
  initial vectors works (11 roots).

### 5.3 DSRG

The DSRG source multiplies an amplitude of denominator D by 1 - exp(-s
D^2).  For s = 0.5 Eh^-2 this is 0.12 at D = 0.5 Eh, 0.39 at 1 Eh and 0.86
at 2 Eh; for s = 4 it is still 0.15 at D = 0.2 Eh.  The excitations that
are damped are those with small denominators: here 5f -> 6d and 7s, the
ones that distinguish the Omega components.  The damped couplings are not
folded into the effective Hamiltonian; they are left out.  Carrying the
amplitudes to infinite order (MR-LDSRG(2)) does not change the window,
which is set by s.

Evidence in the data:

* The UO+ gap rises with s in every setting: -22 -> -10 (26 core), -17 ->
  -9 -> -1.5 -> +1.0 (O 2p, s = 0.5 to 4.0); NdO+ +8 -> +25 and -36 -> -25.
* The amplitudes that grow with s are the active -> virtual singles (largest
  0.054 -> 0.083 from s = 2 to 4) and the semi-internal doubles aa -> av
  (0.079 -> 0.112).  No amplitude exceeds 0.12 and all runs converge: no
  intruder appears up to s = 4.
* The gap saturates near zero, 40 from experiment and 35 - 70 from the MRCI
  with the same core (+37 to +68), and the levels are not monotonic in s
  (Omega 3.5: 515, 497, 509, 560).  The scan does not distinguish between
  the excitations still damped at s = 4 and the truncation of the
  commutators as the origin of the remainder.
* MR-LDSRG(2) and MRPT3 agree to 12 with the O 2p core and differ by up to
  88 with 26 core spinors: the difference between the orders comes from the
  metal semicore, as does the dependence on s.

### 5.4 Enlarging the active space

A 7s pair added to the active space is lost in state averages over 5f^3
states (it turns into an f-type correlating orbital); a state average that
contains 5f^2 7s states changes the 5f orbitals and the levels (section
4.3).  The natural orbitals of the virtual space of the MRCI without core
hold 0.028 electrons in a long tail of 6d and f-type spinors with no 7s:
the missing correlation is dynamical and is not concentrated in a few
orbitals.  Adjusting the active space or the state average until the levels
match experiment is not a result.

## 6. Cost

| calculation | functions / spinors | resources | wall time | peak memory |
|---|---|---|---|---|
| O2 full valence MRCI, 3 states | 7.4e6 | 32 cores | 393 s | 57 GB |
| UO+ MRCI, no core, 5 roots | 8.4e5 | bigmem | 10 min | 89 GB |
| UO+ MRCI, 26 core, one root per Omega | 4e7 | 8 cores | 10 iterations of 170 s | 240 GB |
| UO+ MRCI, 26 core, 5 roots | 1.85e8 | 16 cores | 8.2 h | 499 GB |
| UO+ DSRG-MRPT2 / MRPT3, 26 core | 450 spinors | 8 / 16 cores | 102 s / 56 min | 6 / 34 GB |
| UO+ MR-LDSRG(2), O 2p | 450 spinors | 32 cores | 2.2 - 3.9 h | 75 - 77 GB |
| UO+ MR-LDSRG(2), 26 core | 450 spinors | 10 cores | 17.7 - 21.2 h | 137 GB |
| NdO+ MR-LDSRG(2), 26 core | 382 spinors | 32 cores | 8.3 - 11.3 h | 109 GB |

The memory of the MRCI is set by the (vv|vv) matrix and the three-virtual
integral blocks, not by the dimension; the Davidson vectors are kept in
scratch files above 40 GB.

## 7. What is established and what is open

Established:

* The basis-set limit of the O2 spin-orbit splitting and the recipe to reach
  it (tight p, one optimised function at QZ).
* For O2, non-perturbative methods agree with each other (0.1) and are
  insensitive to the orbitals; the correlated value is 3.74 - 3.78.
* For NdO+ and UO+, the level order; and the specific reason why each
  method is unreliable for the positions once the semicore is correlated.

Open:

* The 0.2 between the correlated O2 value and experiment (section 3.2 lists
  what has not been examined).
* The UO+ Omega 0.5 - 1.5 gap and the NdO+ positions.  A method that is
  size-extensive, does not damp the low-denominator excitations, and does
  not depend on the number of reference states is needed; none of the
  methods available here has all three properties.  CASPT2 was considered
  and set aside (same order as NEVPT2, with additional multi-state variants
  and shifts).
* The basis sets of the chalcogen series for O and S are not converged in
  the spin-orbit part (O2 is 6 % low with uncontracted aug-cc-pVTZ); not
  rerun.  Optimised tight exponents for S and Se are in progress in the
  basis-optimisation session.
* Technical: per-Omega MRCI with several roots does not converge for all
  Omega; MR-LDSRG(2) for NdO+ / UO+ has no independent check.

## 8. Where things are

| what | where |
|---|---|
| numbers and settings of every run | `docs/calculations.md` sections 1 and 6 |
| O2 basis-set runs | `~/scr4/zmr_runs/o2_cbs` |
| NdO+, UO+ (zmr) | `~/scr4/zmr_runs/ndo_tz`, `~/scr4/zmr_runs/uop` |
| DSRG runs and their record | `~/scr4/zmr_runs/dsrg` (`STATUS.md`) |
| basis optimisation | `~/scr4/zmr_runs/basis_opt`, `~/apps/x2camf-basis` |
| MRCI equations and solver | `docs/derivations/mrci.md`, `zmr/mrci` |
| forte2 with MR-LDSRG(2) | `~/apps/forte2`, branch `zmr-interface` (local commits) |
