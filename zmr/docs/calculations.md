# Calculations done with zmr (state of 2026-10-02)

Where every production calculation lives, how it was set up, and its
numbers.  Energies in cm^-1 unless stated.  NEVPT2 is union-contracted
PC-NEVPT2 with state-specific E0 unless stated.  Experimental values of
atoms and ions were entered from memory and have not been checked against a
database; those of the diatomics are from arXiv:2602.24236, Table 2.

Run directories are under `~/scr4/zmr_runs/`; each has `src/` (code
snapshot), the driver script, job scripts and logs.

## 1. Chalcogen diatomics, zero-field splitting X(+-1) - X(0+)

`chalcogen/` with `examples/05-chalcogen_zfs.py`; work files (Cholesky
vectors, X2CMP operator, SCF, CASSCF orbitals) in `chalcogen/work/<molecule>_x2cmp_dcb/`,
logs in `chalcogen/logs/`, `collect.py` tabulates the `RESULT` lines.

X2CMP(DCB), bond lengths and basis sets of arXiv:2602.24236 (uncontracted
aug-cc-pVTZ for O, S; dyall av3z for Se, Te), CAS(8e, 12 spinors), Cholesky
1e-7, virtual spinors above about 100 Eh removed.  SA-3 = the three
components of X 3Sigma-, SA-6 = with a 1Delta and b 1Sigma+.  KRHF of the
dication as the start, except SeO and SeS (neutral closed shell,
`--scf-charge 0`; their dication SCF oscillates).

| molecule | exp. | paper total | CASPT2-SO | IHFSCC | CASSCF SA-3 | CASSCF SA-6 | NEVPT2 ns | NEVPT2 +(n-1)d | NEVPT2 shell |
|----------|------|-------------|-----------|--------|-------------|-------------|-----------|----------------|--------------|
| O2 | 4.0 | 3.6 | - | - | 2.82 | 2.20 | 2.04 | - | 2.04 |
| SO | 10.5 | 11.6 | - | - | 11.39 | 8.18 | 8.62 | - | 8.74 |
| S2 | 23.5 | 25.4 | 19 | 25 | 25.31 | 20.65 | 20.86 | - | 21.18 |
| SeO | 166 | 192.7 | 138 | 137 | 192.7 | 129.4 | 149.8 | 150.1 | 151.6 |
| SeS | 205 | 208.8 | 165 | 192 | 208.5 | 171.8 | 177.5 | 177.3 | 179.8 |
| Se2 | 510 | 540.2 | 446 | 574 | 539.4 | 463.4 | 457.4 | 460.2 | 464.3 |
| TeO | 789 | 873.7 | 635 | 468 | 872.6 | 573.4 | 711.0 | 705.5 | 712.7 |
| TeS | 836 | 819.9 | 675 | 619 | 818.3 | 686.2 | 745.8 | 738.3 | 747.4 |
| TeSe | 1233 | 1275.5 | 1073 | 1238 | 1273.4 | 1123.1 | 1123.6 | 1123.4 | 1132.2 |
| Te2 | 1975 | 2053.4 | 1793 | 2178 | 2050.7 | 1869.2 | 1814.8 | 1812.9 | 1825.8 |

NEVPT2 columns: SA-3 orbitals, three model states; correlated besides the
active space: ns only / ns and (n-1)d of Se, Te / the whole (n-1) shell
(O 1s, S 2s2p, Se 3s3p3d, Te 4s4p4d).  With SA-6 orbitals or six model
states the NEVPT2 values change by at most 31 (TeO: 684 - 715 over the four
combinations), while CASSCF changes by 300.  The common E0 gives values 10 -
17 % larger (TeO about 800); it is not separable and not to be used.  The
SA-3 CASSCF values reproduce the `mpdcb` logs of `~/scr4/zmcscf`.

O2 with spin-spin coupling (`--corr`, X2Ccorr: four-component Coulomb and
Breit integrals in the active space, CASCI level shift added to every
method), SA-3 orbitals, experiment 3.96:

| active space | CASSCF | + corr | NEVPT2 3 st. | + corr | NEVPT2 6 st. | + corr |
|--------------|--------|--------|--------------|--------|--------------|--------|
| 2e, 4 spinors | 2.31 | 3.85 | 1.57 | 3.11 | 1.42 | 2.96 |
| 8e, 12 spinors | 2.82 | 3.72 | 2.04 | 2.94 | 2.08 | 2.98 |

The correction is +1.55 (minimal) and +0.90 (full valence; the paper's Table
1 has 0.9).  With SA-6 orbitals and the minimal space: CASSCF 1.83, NEVPT2
1.83 / 1.68, correction +1.51.

MRCI for the minimal space (2026-10-04; `--cas 2,4 --mrci ci,aqcc`, SA-3
orbitals, 1s frozen, virtuals up to 100 Eh, three model states, fully
internally contracted with relaxed reference, dimension 3480931; 32 cores,
581 s for MRCI and 1316 s for AQCC, 58 GB), without the spin-spin
correction:

| method | X(+-1) - X(0+) / cm^-1 | E(X 0+) / Eh | weight of the internal block |
|--------|------------------------|--------------|------------------------------|
| CASSCF | 2.307 | -149.742436 | |
| PC-NEVPT2 | 1.569 | -150.234025 | |
| MRCI | 2.140 | -150.181888 | 0.9106 |
| MRCI+Q (Davidson) | 2.187 | -150.221181 | |
| MRCI+Q (renormalized) | 2.198 | -150.225039 | |
| MR-AQCC | 2.206 | -150.213438 | 0.8915 |

With the CASCI-level spin-spin correction of the table above (+1.55, not
recomputed for these wave functions): MRCI 3.69, MRCI+Q 3.73 / 3.74, AQCC
3.75; experiment 3.96.  Log: `~/scr4/zmr_runs/mrci/o2.31600158.log`.

The same with the correction recomputed in the MRCI (`--corr --mrci
ci,aqcc`: the four-component active-space integrals replace `(tu|vw)` in
the all-active part of H, model states of the corrected CASCI): MRCI 3.597,
MRCI+Q 3.611 / 3.617, MR-AQCC 3.631.  The correction is +1.46 for MRCI
(+1.42 with +Q, +1.43 for AQCC), about 0.1 below the CASCI value, so adding
the CASCI shift overestimates.  Log: `~/scr4/zmr_runs/mrci/o2corr.31600524.log`.

Full valence space, 8 electrons in 12 spinors (`--mrci ci`, SA-3 orbitals,
1s frozen, 2s correlated, virtuals up to 100 Eh, three model states,
dimension 7379603, weight of the internal block 0.928; 32 cores, 2784 s,
peak 112.7 GB before the memory work of 2026-10-04), without the spin-spin
correction: CASSCF 2.817, PC-NEVPT2 2.038, MRCI 2.069, MRCI+Q 1.805
(Davidson) / 1.767 (renormalized).  The run did both frozen-core choices,
which are the same space for O2, and gave identical numbers twice.  AQCC
not done yet.  Log: `~/scr4/zmr_runs/mrci/o2fv.31600635.log`.

Repeated on 2026-10-04 with the final code (`--core valence --mrci ci,aqcc`):
the same MRCI numbers in 1062 s with a peak of 50 GB (507 s and 58 GB after the
later changes of the sigma code, `o2fv.31625304.log`; 393 s with the
residual criterion 5e-5, `o2fv.31648524.log`), and MR-AQCC 1.821
(weight of the internal block 0.915, E(X 0+) = -150.227089, 2179 s for the
three states).  AQCC agrees with MRCI+Q, so the lowering of the splitting
by the size-consistency correction in this space is not an artefact of the
a posteriori correction.  Log: `~/scr4/zmr_runs/mrci/o2fv.31603243.log`.

Observation on Table 1 of the paper, from the logs in `~/scr4/zmcscf`: the
columns Coulomb / Gaunt / gauge under "Delta X2CAMF" equal the differences
within the X2CMP series (mpdc, mpdcg, mpdcb), the column "Delta X2CMP" equals
X2CAMF(DCB) - X2CMP(DCB) (sign opposite to footnote b), and the columns do
not add up to the Total (they do without that column).  TeO is the exception
(-1.7 from the logs, -2.3 in the table).

### O2: basis-set limit of the zero-field splitting (2026-10-06 / 07)

`~/scr4/zmr_runs/o2_cbs` (`job.sh`, `job_qn.sh`; one directory per basis),
driver options `--basis`, `--tight p:2`, `--sp-from`, `--mcscf sosci`.
Minimal active space (2e, 4 spinors), SA-3 orbitals, X2CMP(DCB), O 1s
frozen, virtual spinors up to about 100 Eh; X(+-1) - X(0+) in cm-1.

| basis (uncontracted) | spinors | largest p exponent | CASSCF | PC-NEVPT2 | spin-spin (CASCI) |
|---|---|---|---|---|---|
| aug-cc-pVTZ | 232 | 34 | 2.307 | 1.569 | +1.545 |
| aug-cc-pVQZ | 372 | 63 | 2.397 | 1.661 | +1.549 |
| aug-cc-pV5Z | 576 | 196 | 2.452 | 1.716 | +1.550 |
| aug-cc-pV6Z | 840 | 526 | 2.462 | | |
| aug-cc-pVTZ + 1 / 2 / 3 tight p | 244 / 256 / 268 | 153 / 682 / 3031 | 2.436 / 2.457 / 2.458 | 1.656 / 1.671 / 1.672 | +1.548 |
| aug-cc-pVTZ + 1 / 2 optimised tight p | 244 / 256 | 226 / 1223 | 2.449 / 2.458 | 1.665 / 1.672 | +1.547 / +1.548 |
| aug-cc-pVQZ + 1 optimised tight p | 384 | 418 | 2.459 | 1.704 | +1.549 |
| s, p of aug-cc-pV7Z + TZ d, f | 348 | 1349 | 2.465 | 1.679 | +1.550 |
| s, p of aug-cc-pV7Z + QZ d, f, g | 468 | 1349 | 2.464 | 1.710 | +1.550 |
| dyall av3z / av4z | 284 / 444 | 194 / 557 | 2.453 / 2.462 | 1.668 / 1.708 | +1.549 / +1.550 |
| DYVTZ-SOG / ADYVTZ-SOG (contracted, 4s3p2d1f / 5s4p3d2f) | 120 / 184 | 194 | 2.421 / 2.439 | 1.600 / 1.585 | |
| DYVQZ-SOG / ADYVQZ-SOG (contracted, 5s4p3d2f1g / 6s5p4d3f2g) | 220 / 320 | 557 | 2.454 / 2.460 | 1.690 / 1.693 | - / +1.550 |
| ADYVTZ-SOG-P123 (one more p uncontracted, 5s5p3d2f) | 196 | 194 | 2.453 | 1.655 | |
| ADYVTZ-SOG-P13 / -P23 (other choices of the two uncontracted p, 5s4p3d2f) | 184 | 194 | 2.538 / 2.665 | 1.830 / 1.852 | |
| APVTZ+P1-SOG (aug-cc-pVTZ primitives + tight p 225.713, contracted 5s4p3d2f) | 184 | 226 | 2.453 | 1.656 | +1.548 |
| aug-cc-pVTZ / aug-cc-pVQZ, contracted as published | 184 / 320 | 34 / 63 | 2.263 / 2.399 | 1.528 / 1.660 | |

The optimised tight p exponents (TZ: 225.713; 1223.0, 182.873; QZ: 418.016)
come from the atomic four-component DHF 2p splitting (X2CAMF as engine,
`~/scr4/zmr_runs/basis_opt`); they are preliminary. Driver option
`--tight p:1223.0/182.873`.

The SOG sets are j-adapted contractions (one per kappa for p) of Dyall's
primitives made with the X2CAMF + Gaunt atomic program in the
basis-optimisation session (`~/scr4/zmr_runs/basis_opt/basis`; driver
option `--basis-file`, runs `sog_*`).  Contraction costs 0.014 (TZ) and
0.002 (QZ) in the CASSCF value against the uncontracted av3z / av4z.  The
spin-spin step (`--corr`) failed for these sets at first; the NEVPT2 values
are from runs without it.  Since 2026-10-09 the four-component active
integrals come from socutils (`tools.fcidump_rel.x2c_active_eri_4c`, commit
4430bc8 on `prop-cfour-interface`, 4decfb9 on `zmcscf-qn`), which expands
contracted spinors in the uncontracted basis of the X2C step before R and X
are applied: spin-spin +1.548 for APVTZ+P1-SOG against +1.547 for its
uncontracted parent, which is unchanged by the new routine, and +1.550 for
ADYVQZ-SOG as for the uncontracted av4z (70 min against 53 min).  Tests of the
routine: `o2_cbs/socutils_patch/test_contracted.py` (old against new
FCIDUMP 1e-15; a contraction spanning the primitive space 3e-15).  This step
is the time sink of the large-basis runs: 5.5 min for TZ, 53 min for dyall
av4z (the integrals are transformed in the uncontracted spinor basis), and
it is where the 5Z and 6Z jobs spent their hours.  Jobs started after
2026-10-09 10:40 run the CASSCF with the step control of zmcscf-qn 65e381a.
NEVPT2 - CASSCF is -0.74 for the cc sets, contracted or not, -0.78 / -0.75
for the uncontracted av3z / av4z, -0.86 for ADYVTZ-SOG and -0.80 with one
more p primitive uncontracted (P123), which also restores the CASSCF value
of the uncontracted set: the loss of the TZ contraction was in the p
contraction.  At 320 spinors the contracted aug-cc-pVQZ gives 2.399 and
ADYVQZ-SOG 2.460.
Uncontracting other pairs of p primitives at the same size (P13: 0.126 and
0.837; P23: 0.332 and 0.837) overshoots the basis-set limit by 0.07 and
0.20 with energies 1.2 - 1.9 mEh above ADYVTZ-SOG: not usable.  APVTZ+P1-SOG
(Dunning's primitives and correlating functions, one optimised tight p, cc-pVTZ
contraction pattern per kappa, point-nucleus contraction) reproduces its
uncontracted parent (2.449 / 1.665, 244 spinors) to +0.003 / -0.009 with
184 spinors, the size of the contracted aug-cc-pVTZ (2.263 / 1.528).

MRCI with the spin-spin term recomputed in the correlated wave function
(MRCI / +Q Davidson / +Q renormalized / AQCC): TZ 3.597 / 3.617 / 3.623 /
3.631; QZ 3.707 / 3.724 / 3.730 / 3.743; TZ + 2 tight p 3.738 / 3.761 /
3.769 / 3.777 (without spin-spin 2.279 / 2.334 / 2.348 / 2.350).
Experiment 3.96.

* The whole basis-set dependence is in the spin-orbit matrix element, i.e.
  in the O 2p spinors near the nucleus (r^-3 weight): the b - X gap changes
  by 7e-4 from TZ to QZ, the spin-spin term by 0.004, the correlation
  contribution (MRCI with spin-spin minus CASSCF: 1.290, 1.309, 1.281 for
  TZ, QZ, TZ + 2p) by 0.03.  Energy-optimized sets stop at p exponents that
  describe the valence shell; two tight p primitives continuing the series
  with its last ratio are enough.  Higher angular momenta do not matter
  (2.465 against 2.464).  The limit of the CASSCF value is 2.46.
* A geometric extrapolation of TZ / QZ / 5Z gave 2.54 and was wrong.
* The O atom is a proxy: 3P1 of O 2p^4 (CASSCF) 138.72 (TZ), 142.46 / 143.08
  / 143.11 (+1 / 2 / 3 tight p), 141.40 (QZ), 142.99 (5Z), 143.29 (6Z),
  143.36 (7Z s, p); the molecular splitting follows the square (0.9676^2 =
  0.936 = 2.307 / 2.465).
* Tight primitives must be inserted into their angular momentum in the
  basis: appended after the highest l they gave energies 0.3 - 12 Eh too
  high (socutils' X2C atomic part takes the number of shells from the last
  primitive); the directories `tz_p2`, `tz_p4`, `tz_p6`, `tz_p4d2`,
  `tz_s3p4`, `qz_p3` hold those invalid runs, the valid ones are `tzs_p*`.
* The new CASSCF optimiser (`--mcscf sosci`: SO-SCI + L-BFGS of the
  zmcscf-qn branch of socutils) gives the same CASSCF in 10 instead of 27
  iterations for TZ.
* The sets of the table in section 1 for O and S (uncontracted aug-cc-pVTZ,
  largest p exponent 34 and 574) are therefore not converged in the
  spin-orbit part; O2 is 6 % low.  Not rerun.

DSRG on the same input (forte2 through the adapter of the DSRG session,
`~/scr4/zmr_runs/dsrg/o2min`, TZ, single runs): CASCI agrees to 2e-12 Eh;
MRPT2 0.99 (s = 0.5; 1.83 - 0.63 for s = 0.1 - 2.0, no plateau), MRPT3 2.46
(2.28 - 2.92), MR-LDSRG(2) 2.12 - 2.21, the range of MRCI / MRCI+Q / AQCC
(2.14 / 2.19 / 2.21); with SA-6 orbitals (CASCI 1.83) MR-LDSRG(2) gives 2.06
- 2.21: the non-perturbative methods remove the dependence on the state
average, second order does not.

## 2. Atoms np^4 (O, S, Se, Te) and 6p^n (Tl, Pb, Bi)

Same driver and directory.  X2CMP(DCB), KRHF of the dianion (np^6), SA over
all 15 states of np^4, 15 model states, virtual cut at about 100 Eh.

CAS(4e, 6 spinors), ns correlated ("+d": also (n-1)d; "shell": the whole
(n-1) shell):

| atom | core | 3P1 | 3P0 | 1D2 | 1S0 |
|------|------|-----|-----|-----|-----|
| O | CASSCF | 139 | 207 | 17632 | 43975 |
| O | ns | 139 | 207 | 15878 | 40705 |
| O | shell | 140 | 208 | 15893 | 40709 |
| O | exp. | 158 | 227 | 15868 | 33793 |
| S | CASSCF | 394 | 576 | 11725 | 29009 |
| S | ns | 391 | 571 | 9537 | 27977 |
| S | shell | 394 | 575 | 9498 | 27870 |
| S | exp. | 396 | 574 | 9239 | 22180 |
| Se | CASSCF | 1942 | 2576 | 11875 | 28058 |
| Se | ns | 1924 | 2520 | 9920 | 26854 |
| Se | +d | 1949 | 2548 | 9949 | 26867 |
| Se | shell | 1949 | 2547 | 9915 | 26779 |
| Se | exp. | 1990 | 2534 | 9576 | 22446 |
| Te | CASSCF | 4590 | 5033 | 12674 | 27743 |
| Te | ns | 4533 | 4876 | 10907 | 26710 |
| Te | +d | 4571 | 4898 | 10964 | 26760 |
| Te | shell | 4586 | 4903 | 10944 | 26694 |
| Te | exp. | 4751 | 4707 | 10558 | 23198 |

1S0 is 3500 - 6900 too high.  With ns in the active space (`--cas 6,8`):
O CASSCF 138 / 206 / 17641 / 32051, NEVPT2 140 / 209 / 16544 / 35164;
S CASSCF 393 / 573 / 11734 / 21491, NEVPT2 393 / 574 / 10547 / 24936;
Se CASSCF 1941 / 2511 / 11881 / 21979, NEVPT2 1924 / 2505 / 10678 / 24573
(+3d: 1949 / 2535 / 10711 / 24800): 1S0 improves, 1D2 gets worse.  X2Ccorr
level corrections: O +5.5 / +0.4 / -4.6 / +10.3, S -0.3 / -1.6 / -1.6 /
+0.8, Se -1.2 / -1.4 / -2.1 / -1.2.  Te with `--cas 6,8` did not finish
(CASSCF 4586 / 4688 / 12677 / 22784 only).

O and S again with the basis sets found for the spin-orbit part (2026-10-09;
`o2_cbs/at_*`, same protocol, CAS(4e, 6 spinors), SA-15; NEVPT2 with the
whole (n-1) shell correlated).  The first row of each atom is the old basis
rerun with the present code: it reproduces the table above to all digits.

| atom, basis (uncontracted unless noted) | CASSCF 3P1 / 3P0 | NEVPT2 3P1 / 3P0 | NEVPT2 + X2Ccorr | NEVPT2 1D2 / 1S0 |
|---|---|---|---|---|
| O aug-cc-pVTZ | 138.5 / 206.6 | 139.7 / 208.1 | | 15893 / 40709 |
| O + tight p 225.713 | 142.7 / 212.8 | 143.9 / 214.3 | | 15898 / 40717 |
| O + tight p 1223.0, 182.873 | 142.9 / 213.2 | 144.2 / 214.7 | 157.9 / 227.2 | 15898 / 40717 |
| O APVTZ+P1-SOG (contracted) | 142.5 / 212.5 | 143.0 / 213.0 | | 15926 / 40834 |
| O dyall av3z | 142.8 / 213.0 | 144.0 / 214.5 | | 15904 / 40721 |
| O exp. | | | 158 / 227 | 15868 / 33793 |
| S aug-cc-pVTZ | 393.7 / 576.5 | 394.3 / 575.3 | | 9498 / 27870 |
| S + tight p 3853.75 | 395.4 / 578.9 | 396.0 / 577.7 | | 9499 / 27871 |
| S + tight p 21430.7, 3135.39 | 395.5 / 579.0 | 396.1 / 577.8 | 398.7 / 580.2 | 9499 / 27871 |
| S dyall av3z | 395.2 / 578.6 | 395.8 / 577.4 | | 9500 / 27873 |
| S exp. | | | 396 / 574 | 9239 / 22180 |

Tight p raise 3P1 of O by 4.4 (3 %) and of S by 1.8 (0.45 %).

**X2Ccorr: the scheme was applied incompletely until 2026-10-09.**  `--corr`
replaced the active two-electron integrals by four-component ones and kept
the X2C one-electron operator, whose mean-field two-electron spin-orbit term
is built from the atomic densities including the active electrons: the
active-active two-electron spin-orbit interaction was counted twice (found
by the basis-optimisation session on the F atom).  The complete scheme, as
`fourC_int` with `fourC_fcore` of socutils' `from_x2c`, takes the
four-component bare Hamiltonian with the mean field of the core spinors only
as one-electron part (`corr_core_hamiltonian` in the driver).  Level shifts,
integrals only -> complete (runs `*.cc.*.log`, TZ + tight p):

| system | integrals only | complete |
|---|---|---|
| F 2P1/2, X2CAMF(DCB) | -15.6 | -2.8 (contracted set: -2.8; X2CMP: -3.0) |
| F 2P1/2, X2CAMF(DC) | -25.9 | -4.8 |
| O 3P1 / 3P0 / 1D2 / 1S0 | +9.45 / +6.16 / -2.66 / +15.08 | +13.73 / +12.50 / -0.47 / +17.29 |
| S 3P1 / 3P0 / 1D2 / 1S0 | +1.73 / +1.19 / -0.54 / +2.90 | +2.58 / +2.40 / -0.08 / +3.38 |
| O2 X(+-1) - X(0+), CAS(2,4) | +1.547 | +1.538 |

With the complete correction O is at 157.9 / 227.2 (experiment 158 / 227)
and S at 398.7 / 580.2 (396 / 574).  O2 is unaffected (no first-order
spin-orbit contribution in the ground state): the spin-spin values +1.55 of
section 1 stand to 0.01.  All other X2Ccorr numbers of this document
computed before this date for states with first-order spin-orbit splitting
(the `--cas 6,8` atoms above, Pb, the earlier "+ss" of O and S) are of the
incomplete scheme and have not been recomputed.  The MRCI "+ X2Ccorr" runs
of O2 use the integrals with the X2C one-electron part; by the last row the
effect there is 0.01.

F atom, 2P1/2 above 2P3/2 (2026-10-09; `o2_cbs/f_*`, driver `F`: KRHF of
F-, SA-6 CASSCF(5e, 6 spinors), which for 2p^5 is the configuration-averaged
Hartree-Fock with its CASCI; basis files of the basis-optimisation session,
`basis_opt/basis/variants/f_*.py`; experiment 404.1; EOM-IP-CCSD from F- of
that session in the last column):

| basis | Hamiltonian | CASSCF | NEVPT2, 1s frozen / correlated | X2Ccorr | EOM-IP-CCSD |
|---|---|---|---|---|---|
| aug-cc-pVTZ uncontracted | X2CAMF(DCB) | 384.6 | 391.3 / 392.3 | | 404.6 |
| + tight p 286.19 | X2CAMF(DCB) | 395.7 | 402.7 / 403.8 | -2.8 | 416.4 |
| the same, contracted per kappa | X2CAMF(DCB) | 395.7 | 401.7 / 402.1 | -2.8 | 415.9 |
| + tight p | X2CAMF(DC) | 438.4 | 446.1 / 447.3 | -4.8 | 461.2 |
| + tight p | X2CMP(DCB) | 395.8 | 402.8 / 403.9 | -3.0 | |
| + tight p | X2CMP(DC) | 438.4 | 446.1 / 447.4 | | |

Tight p +11.1 (EOM +11.8); Gaunt and Breit -42.6 (EOM -44.8); contraction
-0.1 in CASSCF and -1 to -2 in NEVPT2; X2CAMF and X2CMP agree to 0.1.
With the complete X2Ccorr correction (see above; -2.8) NEVPT2 is at 401.0,
3 below experiment; EOM-IP-CCSD is 12.5 above NEVPT2.

Pb (6p^2, 15 states, KRHF of Pb2+), levels 3P1 / 3P2 / 1D2 / 1S0: CASSCF
6847 / 10628 / 21975 / 32821; NEVPT2 6s correlated 7001 / 10000 / 20598 /
30779; with 5d 6952 / 9934 / 20333 / 30423; X2Ccorr -0.3 / -0.8 / -0.6 /
-0.1; experiment about 7819 / 10650 / 21458 / 29467.  The X2Ccorr integral
transformation took more than an hour for this basis (286 functions).
Tl (SA-6 CASSCF not converged in 100 iterations) and Bi (abort in the
CASSCF step) failed and have not been looked into; for Tl the appropriate
method is single-reference (Delta-SCF + coupled cluster, results exist in
the literature).

## 3. Lanthanide ions

**Nd3+ 4f^3** (`nd3/`, `nd3.py` for the 52 states of 4I, `nd3_all.py
free|field 5s5p|4d|4s4p4d [cahf]` for all 364 states).  Orbitals and X2CMP(DCB)
operator are read from `~/scr4/zmcscf/nd/unc_tz_8pt_mpdcb` (dyall v3z,
SA-52 CASSCF or CAHF, eight point charges); Cholesky vectors recomputed;
virtual cut 100 Eh (330 of 548 kept).  Free ion, 364 model states, levels
assigned by degeneracy and order only:

| level | CASCI | 5s5p | +4d | +4s4p4d | +4s4p4d, CAHF | exp. (approx.) |
|-------|-------|------|-----|---------|---------------|----------------|
| 4I11/2 | 1785 | 1781 | 1843 | 1884 | 1870 | 1880 |
| 4I13/2 | 3729 | 3708 | 3825 | 3900 | 3869 | 3900 |
| 4I15/2 | 5799 | 5746 | 5910 | 6012 | 5965 | 5900 |
| 4F3/2 | 15860 | 13246 | 12862 | 11729 | 11686 | 11460 |
| 4F5/2 | 16830 | 14211 | 13884 | 12764 | 12712 | 12480 |
| 2H9/2 | - | 14030 | 13884 | 13322 | 13257 | 12590 |
| 4F7/2 | - | 15157 | 14845 | 13783 | 13722 | 13460 |
| 4S3/2 | - | 14752 | 14542 | 13877 | 13820 | 13560 |
| 4F9/2 | - | 16286 | 16058 | 15155 | 15079 | 14720 |

Correlating the whole n = 4 shell is valence correlation for 4f and is
required.  With the point charges and the 52 states of 4I (5s5p correlated)
the centres of gravity are CASSCF 61 / 1838 / 3781 / 5851, NEVPT2 69 / 1848 /
3778 / 5821.  One 364-state NEVPT2: 41 - 48 s on 16 cores.

**Pr3+ 4f^2** (`pr3_val/pr3.py <basis>`): X2CAMF(DCB), CAHF, exact
integrals, 91 states, x2c-SVPall:

| level | CASCI | 5s5p | +4d | +4s4p | +3d | exp. |
|-------|-------|------|-----|-------|-----|------|
| 3H5 | 2161 | 2118 | 2141 | 2150 | 2159 | 2152 |
| 3H6 | 4456 | 4368 | 4399 | 4417 | 4434 | 4389 |
| 3F2 | 6759 | 6108 | 5600 | 5093 | 5055 | 4997 |
| 3F3 | 8124 | 7465 | 6975 | 6497 | 6464 | 6415 |
| 3F4 | 8153 | 7773 | 7274 | 7032 | 7003 | 6855 |
| 1G4 | 11073 | 10678 | 10180 | 10120 | 10104 | 9921 |
| 1D2 | 23428 | 21356 | 20058 | 18451 | 18368 | 17334 |
| 3P0 | 29435 | 26312 | 24643 | 22868 | 22755 | 21390 |
| 3P1 | 29990 | 26866 | 25219 | 23455 | 23344 | 22007 |
| 1I6 | 28061 | 26104 | 25818 | 24887 | 24898 | 22212 |
| 3P2 | 31086 | 27984 | 26364 | 24629 | 24525 | 23161 |
| 1S0 | 64763 | 58434 | 53979 | 52381 | 52142 | 50090 |

The dyall v3z run of this script is not usable: the CAHF did not converge
and the job ran out of memory (exact AO integrals for 308 functions; use
Cholesky vectors).

## 4. Core holes: Ar 2p^-2

`ar_dch/ar_dch.py` (recipe after `~/scr4/zmcscf/co_cahf/ar_dch.py`).
Uncontracted cc-pVTZ, X2CAMF(DCB), CAHF with the six 2p spinors holding four
electrons, the 15 states of 2p^4, all electrons correlated, ground state with
the same partition.  `mc_diis` returns the orbitals in energy order: the 2p
spinors have to be sorted back into the active block.

| | double ionization energy / eV | 3P1 | 3P0 | 1D2 | 1S0 (eV above 3P2) |
|---|---|---|---|---|---|
| CASCI | 536.23 | 1.641 | 2.142 | 8.810 | 20.620 |
| NEVPT2, cvs | 536.58 | 1.630 | 2.110 | 8.952 | 19.217 |
| NEVPT2, no cvs | 535.96 | 1.626 | 2.129 | 8.901 | 19.856 |

Revised CVS (2026-10-03, by the number of holes: ccav keeps the pairs with
one inner and one outer core electron, all denominators positive, smallest
0.85 Eh): double ionization energy 536.135 eV, levels 0, 1.630, 2.109,
8.936, 19.175 eV.  The decay channels left out add +0.45 mEh (ccaa, caaa:
lower discrete states) and -6.85 mEh (ccav: Auger continuum, no term within
1 Eh of zero in this basis) to the lowest level, -0.174 eV in all; with them
the result is the one without CVS.  Run: `~/scr4/zmr_runs/ar_cvs/ar_cvs.py`.

No experimental value has been compared yet.

## 5. Timings

ORCA 6.0.1 FIC-NEVPT2 against `zmr.nr` PC-NEVPT2 (`orca_timing/`, 4 cores,
NEVPT2 step with integral transformation, seconds):

| case | zmr.nr | zmr.nr, density-matrix route (exact / RI) | ORCA exact | ORCA RI |
|------|--------|--------|------------|---------|
| N2 CAS(10,8) cc-pVTZ | 2.8 | 0.8 / 0.4 | 1.5 | 1.3 |
| N2 CAS(10,8) cc-pVQZ | 4.7 | 2.6 / 0.5 | 5.0 | 2.0 |
| N2 CAS(10,8) cc-pV5Z | 17.9 | not rerun | 29.9 | 7.3 |
| N2 CAS(10,10) cc-pVTZ | 120.1 | 4.9 / 4.5 | 11.0 | 10.7 |
| benzene CAS(6,6) cc-pVTZ | 49.1 | 49.1 / 1.5 | 79.0 | 20.3 |

The third column is the default of `zmr.nr.PCNEVPT2` since 2026-10-04
(`amplitudes='auto'`, here the density-matrix route in every class;
`derivations/performance.md` section 5, item 2; same script with both
integral sources, PySCF CASSCF, cc-pVXZ-RI vectors).  The second column
is the CI-vector route before the string-driven sigma kernel (52 s after it
for CAS(10,10)).

The last two rows compare different CASSCF solutions (E2 differs by 0.014
and 0.033 Eh).  Spinor against spin-restricted code on N2 (same directory,
`per_class.py`, `rel_x2c.py`): class sums 7 times slower for 16 spinors with
a spin-free Hamiltonian, 24 times with X2CAMF and no symmetry (82 times in
total wall time, dominated by the dense sector diagonalization).

Nd3+ 364 states, 4d5s5p correlated, 16 cores: 5046 s before and 41 s after
the interpolated contraction (commit e23abff).

## 6. NdO+ and UO+: levels of the f^3 ground multiplet (2026-10-05 / 06)

Run directories `~/scr4/zmr_runs/ndo_tz` (`ndo.py`) and `~/scr4/zmr_runs/uop`
(`uop.py`, `sa10/`, `sa10_omega/`, `cas16*/`, `sa80_as14/`); DSRG runs of the
separate session in `~/scr4/zmr_runs/dsrg` (`STATUS.md`).  X2CAMF with Gaunt
and Breit, Cholesky 1e-7, contracted bases (Nd / U DYPVTZ-SO, O PVTZ-X2C),
3 electrons in the 14 4f / 5f spinors, CAHF from fragment orbitals
(`socutils.tools.fragmo_guess`) or from the stored CAHF, then SA-CASSCF with
the SO-SCI + L-BFGS driver of the `zmcscf-qn` branch of socutils (the
super-CI of socutils-git stalled at a gradient of 2.5e-4 for UO+).  Levels
in cm-1 above the Omega = 4.5 ground state; Omega from the J_z-adapted
active spinors (`zmr.active.symmetry.adapt`); vertical, at one bond length.
Experimental numbers as supplied by the user (their assignment of Omega may
itself rest on calculations).

**UO+**, r = 1.803 A, 450 spinors, 26 correlated core spinors (U 5d 6s 6p,
O 2s 2p) unless noted, SA-10 orbitals (the five doublets of 4I9/2):

| method | 3.5 | 2.5 | 1.5 | 0.5 | 0.5 - 1.5 |
|---|---|---|---|---|---|
| experiment | 764.9 | 1132.4 | 1284.5 | 1324.9 | +40 |
| CASCI | 567 | 1132 | 1334 | 1321 | -14 |
| PC-NEVPT2, 10 model states | 700 | 1072 | 1180 | 1178 | -2 |
| PC-NEVPT2, O 2p only | 559 | 924 | 1078 | 1106 | +28 |
| PC-NEVPT2, no core | 561 | 1021 | 1197 | 1207 | +10 |
| MRCI, no core, 5 roots | 706 | 1119 | 1310 | 1366 | +56 |
| MRCI, O 2p only, 5 roots | 585 | 964 | 1143 | 1211 | +68 |
| MRCI, 26 core, 5 roots together | 655 | 1165 | 1338 | 1346 | +8 |
| MRCI+Q, the same | 662 | 1133 | 1316 | 1364 | +48 |
| MRCI, 26 core, one root per Omega | 635 | 1167 | 1327 | 1318 | -9 |
| MR-AQCC, one root per Omega | 753 | 1267 | 1454 | 1526 | +71 |
| DSRG-MRPT2 (forte2), s = 0.5 / 1.0 | 644 / 584 | 1041 / 899 | 1193 / 1059 | 1210 / 1112 | +17 / +54 |
| DSRG-MRPT3 (forte2), s = 0.5 / 1.0 | 624 / 597 | 1188 / 1160 | 1338 / 1307 | 1291 / 1256 | -47 / -51 |

Higher roots (MRCI without core, 11 roots together, root following):
(1)5.5 4058, (2)4.5 4591, second 3.5 4932, (2)2.5 5128 (experiment 4177.8,
4758.5, 5162.0, 5219.4; the observed (2)3.5 at 4982.4 is not a 5f^3 state
and is absent from this reference space).

**NdO+**, r = 1.742 A, 382 spinors, 26 correlated core spinors (Nd 4d 5s 5p,
O 2s 2p), SA-52 orbitals (4I):

| method | 3.5 | 2.5 | 1.5 | 0.5 |
|---|---|---|---|---|
| experiment | 1280 | 1538 | 1624 | 1719 |
| CASCI | 890 | 1165 | 1191 | 1174 |
| PC-NEVPT2, 10 model states | 940 | 1214 | 1245 | 1237 |
| MRCI, 5 roots together | 961 | 1256 | 1305 | 1318 |
| MRCI+Q, the same | 1006 | 1348 | 1420 | 1450 |
| MRCI / +Q, 16 core spinors (no 4d) | 984 / 1059 | 1306 / 1426 | 1361 / 1509 | 1375 / 1551 |
| MR-AQCC, one root per Omega | 1385 | 1864 | 1952 | 2013 |
| DSRG-MRPT3 (forte2), s = 0.5 | 993 | 1335 | 1401 | 1406 |

What these calculations established:

* **Sensitivity to the choices is as large as the differences between the
  methods.**  UO+ PC-NEVPT2 in the same 5f active space with SA-10 / SA-52 /
  SA-80 orbitals (the last from a state average that contains 5f^2 7s
  states): 700 / 716 / 761 for Omega 3.5 and 1072 / 1092 / 1178 for 2.5; MRCI
  without core with SA-10 / SA-52 orbitals: 706 / 742 and 1119 / 1162.  The
  bond length moves Omega 3.5 by about 35 - 40 cm-1 per 0.03 A.  Stable in
  all calculations: the order 4.5 < 3.5 < 2.5 < (1.5, 0.5) and the onset of
  the J = 5.5 group near 4100 - 4300.  Not determined: the positions to
  better than about +-100 and the sign of the 0.5 - 1.5 gap (-80 to +100).
  Tuning the active space or the state average towards experiment is not a
  result.
* **MRCI is not size-extensive at this number of correlated electrons.**
  With 26 core spinors the reference weight is 0.88, +Q is 0.12 Eh (12 % of
  the correlation energy), and MRCI, MRCI+Q and AQCC differ by 100 - 700 cm-1
  in the levels.  ACPF collapses.
* **States outside the reference space lie below the reference energy.**
  The lowest eigenvalue of H outside the internal block (5f -> virtual
  singles, mostly 6d and f type) is +0.036 Eh without core correlation and
  -0.013 Eh with the O 2p core (`uop.py --qmin`): H - E_ref is not positive
  definite there, so CEPA(0) / ACPF-type functionals are not defined with
  the uncorrelated reference energy.
* **Internal contraction depends on the number of reference states** in the
  classes with core indices (ccvv uses 5 of 364 active functions, cavv 53 of
  91, ccav 70 of 1001): UO+ with the O 2p core, MRCI with 1 / 5 / 10
  reference states: 586 / 585 / 581, 983 / 964 / 955, 1152 / 1143 / 1125,
  1192 / 1211 / 1162.  Without core indices the contracted space is complete
  and the three agree exactly.
* **Enlarging the active space is not the way.**  State averages over 5f^3
  states only lose a 7s pair added to the active space (it becomes an
  f-type correlating orbital); the natural orbitals of the virtual space in
  the MRCI without core (`--natvirt`) hold 0.028 electrons in a long tail of
  6d / f-type spinors with no 7s; DSRG-MRPT3 with 5f + 7s gives a gap of
  -76 / -82 and puts the 7s state anywhere between 5070 and 12243.
* Perturbation theory misses the gap once the core is correlated;
  DSRG-MRPT2 depends on the flow parameter by up to 140 cm-1, MRPT3 by 7 -
  35 but gives the opposite sign.
* **MR-LDSRG(2)** (two-component, NIVO, written in the DSRG session, local
  forte2 branch; single runs, preliminary), s = 0.5 / 1.0, 26 core: UO+ 626
  / 601, 1130 / 1072, 1285 / 1227, 1263 / 1216 (gap -22 / -10); NdO+ 948 /
  932, 1290 / 1273, 1384 / 1391, 1392 / 1415.  UO+ with the O 2p core, s =
  0.5 / 1.0 / 2.0 / 4.0: gap -17 / -9 / -1.5 / +1.0, all converged, largest
  amplitude 0.11.  The gap rises with s (less damping of the 5f -> 6d, 7s
  excitations) and saturates near zero.

The assessment of all methods on O2, NdO+ and UO+ is in
`report-2026-10-multireference-assessment.md`.

Sizes and cost of the MRCI (bigmem nodes): UO+ 26 core, 5 roots: 1.85e8
functions, 8.2 h on 16 cores, peak 499 GB with the Davidson vectors in
scratch files; one root per Omega: 4e7 functions, 10 iterations of about
170 s on 8 cores, 240 GB; no core: 8.4e5 functions, 10 min, 89 GB (the
`(vv|vv)` matrix and the three-virtual integral blocks set the memory, not
the dimension).  `contraction='wk'` exceeded 384 GB in the setup.

## 7. p-block atoms: fine-structure splitting in dyall v4z (2026-10-10 / 12)

Driver `examples/05-chalcogen_zfs.py --atom El` (table `PBLOCK`), X2CMP with
Coulomb, Gaunt and gauge terms, uncontracted dyall v4z, the np shell active
(n electrons in 6 spinors), orbitals averaged over all states of np^n (the
same as a CAHF of the np shell, checked), valence shell correlated
(`--core shell`; Tl: `--core d`, the shell choice finds no gap), emax 100 Eh.
Splitting in cm-1 in the convention of the experimental tables (upper minus
lower level of the ground term in the experimental order: 2P3/2 - 2P1/2 for
p^1, 3P1 - 3P0 for p^2, 2D3/2 - 2D5/2 for p^3, 3P1 - 3P2 for p^4,
2P1/2 - 2P3/2 for p^5).  "ss" is the X2Ccorr correction (complete scheme of
section 2) at the CASCI level, to be added to any column.  Fock H0: the PT2
of this code with the block-diagonal generalized Fock operator instead of the
Dyall Hamiltonian (`h0 = 'fock'`), model states rotated.  Run directory
`~/scr4/zmr_runs/pblock`, `collect.py`.

| atom | CASSCF | ss | PC-NEVPT2 | PT2, Fock H0 | experiment |
|---|---|---|---|---|---|
| B  | 14.96 | +0.21 | 15.32 | 15.32 | 15.29 |
| Al | 110.50 | +0.05 | 115.45 | 114.88 | 112.06 |
| Ga | 775.76 | +0.06 | 812.69 | 811.27 | 826.19 |
| In | 2099.23 | +0.04 | 2157.17 | 2154.79 | 2212.60 |
| Tl (`--core shell` of 6s only) | 7555.31 | +0.06 | 7388.30 | 7318.41 | 7792 |
| Tl (`--core d`) | 7555.31 | | 7372.38 | 7397.60 | 7792 |
| C  | 16.00 | +0.31 | 16.30 | 15.97 | 16.42 |
| Si | 77.20 | +0.07 | 79.26 | 78.08 | 77.11 |
| Ge | 512.59 | +0.05 | 535.88 | 541.66 | 557.13 |
| Sn | 1478.22 | +0.01 | 1565.31 | 1617.04 | 1691.81 |
| N  | -2.65 | +12.97 | -3.32 | -3.70 | 8.71 |
| P  | 14.45 | -2.62 | 14.19 | 17.60 | 15.61 |
| As | 246.90 | -1.90 | 250.51 | 298.50 | 322.10 |
| Sb | 1139.34 | -1.21 | 1106.43 | 1247.26 | 1341.89 |
| O  | 143.10 | +13.75 | 145.35 | 142.35 | 158.26 |
| S  | 395.59 | +2.58 | 401.01 | 391.56 | 396.06 |
| Se | 1942.83 | +1.56 | 1973.89 | 1916.41 | 1989.50 |
| Te | 4590.18 | +0.81 | 4663.89 | 4502.33 | 4706.49 |
| F  | 397.00 | -3.01 | 407.35 | 399.71 | 404.14 |
| Cl | 890.38 | -0.54 | 899.76 | 880.09 | 882.35 |
| Br | 3687.99 | -0.32 | 3680.95 | 3594.64 | 3685.24 |
| I  | 7666.82 | -0.19 | 7557.78 | 7355.22 | 7602.98 |

- Spin-spin matters for N (the 2D splitting is almost entirely spin-spin) and
  O (9 %), and is below 1 % from the third period on.
- Group 15 is poor with NEVPT2 from P on (As 250 against 322); the Fock H0
  is closer there, but it is worse for groups 16 and 17 of the heavier
  periods.  Neither is uniformly better; where one "wins" by a few cm-1
  against experiment the margin is within the sensitivity to the model space
  and is not evidence.
- Real CASPT2 (`zmr.mrci.caspt2.CASPT2`, H0 = PFP + SFS with the full Fock
  matrix, XMS-type rotation, all states of np^n): B 15.17, C 15.68, N -3.60,
  O 139.99, F 396.06, Al 114.52, Si 77.47, P 17.20, S 389.71, Cl 877.18.
  With the ground term only as model space (9 states): O 132.16, S 356.44
  (four-component CASPT2 of the literature for S: 355.94).  NEVPT2 changes by
  about 1 cm-1 between the two model spaces: the Fock-type theories are
  sensitive to the model space, NEVPT2 is not.
- The group's spreadsheet (`ZFS_error_analysis_1.xlsx`): its "CAS" sheet is a
  state-averaged CASSCF over the ground term with np active and is reproduced;
  its "CAHF" sheet averages ns and np; its MRPT rows are an uncontracted
  Fock-type theory, to which the Fock H0 column corresponds roughly (B and F
  to 0.01, Al and P off by 4 to 17 %).

## 8. O2: comparison with exact FCI, reference space, all-spinor X2Ccorr (2026-10-11 / 12)

Hamiltonians exported for the SHCI / zgasci sessions and all scripts are in
`~/scr4/zmr_runs/o2_cbs/shci` (`--dump` of the driver; `export_x2ccorr.py`,
`fno_ladder.py`, `big_ref.py`).  Basis DYVTZ-SOG (contracted), 1s frozen,
12 electrons in 116 spinors, SA-3 CASSCF(2,4) spinors; splitting
X(+-1) - X(0+) in cm-1.

**All-spinor X2Ccorr.**  The four-component Coulomb, Gaunt and gauge
integrals for all correlated spinors and, as one-electron part, the bare
four-component Hamiltonian plus the four-component mean field of the frozen
1s: the no-pair Dirac-Coulomb-Breit Hamiltonian in the X2CMP spinors.  The
scheme with four-component integrals for the active spinors only is not the
same at the correlated level: the "MRCI + X2Ccorr" values 3.74 to 3.78 quoted
before 2026-10-11 are of the active-only variant.

| 116 spinors | CASCI | PC-NEVPT2 | ic-MRCI | unc. MRCISD | +Q (unc.) | ic-AQCC |
|---|---|---|---|---|---|---|
| X2CMP | 2.4209 | 1.6000 | 2.1973 | 2.1964 | 2.2625 / 2.2782 | 2.2671 |
| X2Ccorr, active spinors only | | | 3.6594 | | | 3.6992 |
| X2Ccorr, all spinors | 3.9615 | 3.3892 | 3.8384 | 3.8257 | 3.8426 / 3.8473 | 3.8515 |

Experiment 3.96.  Large basis (uncontracted aug-cc-pVTZ + 2 tight p, 212
spinors, bigmem, 1 h 17 min, 177 GB; file `o2_tzp2_x2ccorr.npz`), all-spinor
X2Ccorr: CASCI 3.9948, PC-NEVPT2 3.4253, ic-MRCI 3.8940, uncontracted MRCISD
3.8797 (+Q 3.9004 / 3.9056), ic-AQCC 3.9066, uncontracted AQCC 3.9133; the
X2CMP uncontracted MRCISD of the same space is 2.2722.

Audit of the all-spinor Hamiltonian (2026-10-13, `audit_x2ccorr.py`; user's
requirement: the one-electron part must be made in the four-component
picture, which it is: bare Dirac Hamiltonian and the four-component J - K of
the frozen core between the back-transformed spinors R C, X R C; no
two-component operator enters):
- the prefactors of `fcidump_rel.x2c_active_eri_4c` are those of pyscf's DHF
  (`int2e_breit_*` is the full Breit operator, +1; Gaunt alone `int2e_*`
  times -1; small components times 1/2c);
- with the 1s spinors among the explicit integrals, folding them by
  sum_i [(pq|ii) - (pi|iq)] reproduces the DHF get_jk route: h1 to 2e-12 (O
  atom) / 9e-10 (O2, 116 spinors), constant to 3e-12, for Coulomb, + Gaunt,
  + Breit; (pq|rs) = (rs|pq) and Hermiticity to 2e-12;
- O atom (DYVTZ-SOG, 58 spinors): every J level degenerate to 1e-4 cm-1 in
  CASCI, NEVPT2 and uncontracted MRCISD; 3P1 / 3P0 (experiment 158.27 /
  226.98): X2CMP CASCI 142.25 / 212.15, Coulomb only 162.52 / 242.23,
  + Gaunt 156.00 / 224.65, + Breit 156.00 / 224.66 (MRCISD 153.73 / 220.18);
- O2, 116 spinors, splitting CASCI | NEVPT2 12 e / 8 e | unc. MRCISD 12 e /
  8 e (8 e: 2s pair frozen): X2CMP 2.4209 | 1.6000 / 1.7953 | 2.1964 / 2.2246;
  Coulomb 2.9479 | 2.1246 / 2.3477 | 2.7692 / 2.8127; + Gaunt 3.9614 | 3.3892
  / 3.4369 | 3.8256 / 3.6380; + Breit 3.9615 | 3.3892 / 3.4369 | 3.8257 /
  3.6380.  The gauge term does not matter.  The effect of correlating 2s
  differs with and without Gaunt in every measure (MRCISD, NEVPT2, valence
  FCI: CAS(12,16) - CAS(8,12) = +0.236 X2CMP, +0.314 X2Ccorr), but MRCISD is
  not size consistent and the measures disagree on the Gaunt-free part: what
  2s contributes is NOT established (user, 2026-10-13); only full FCI with
  12 and 8 electrons can settle it.

**Against exact FCI** (zgasci, the n lowest spinors in energy order, X2CMP):

| n | exact | PC-NEVPT2 | CASPT2 | ic-MRCI | unc. MRCISD | unc. +Q | ic-AQCC |
|---|---|---|---|---|---|---|---|
| 28 | 2.886 | 2.384 | 1.730 | 2.827 | 2.885 | 3.023 | 2.909 |
| 30 | 2.849 | 2.365 | 1.663 | 2.781 | 2.842 | 2.975 | 2.860 |
| 32 | 2.835 | 2.355 | 1.637 | 2.757 | 2.822 | 2.956 | 2.835 |

These are truncated spaces (the user: truncating the virtuals hides what one
wants to see; conclusions are to be drawn in complete spaces of a smaller
basis, below).  In them second order fails (errors of 0.5 to 1.2 cm-1); the
uncontracted MRCISD with the CAS(2,4) reference is within 0.01, although its
total energy is 7 to 10 mEh above FCI (49 mEh in the full space: not size
extensive); the Davidson correction moves the splitting the wrong way.

**Larger reference** (`big_ref.py`; the virtual spinors rotated within the
correlated space to the natural virtuals of the CAS(2,4) MRCI, the sigma* 2p
pair is the most occupied one, 0.033 against 0.003 for the next).  Reference
CAS(8,12) = sigma 2p, pi_u, pi*, sigma* 2p with the 2s pair as correlated
core:

| n | exact (E) | CAS(2,4) unc. (E - E_FCI) | CAS(8,12) ic | CAS(8,12) unc. (E - E_FCI) | unc. +Q |
|---|---|---|---|---|---|
| 28 | 2.886 (-149.8726103) | 2.8848 (7.0 mEh) | 2.8139 | 2.8868 (0.38 mEh) | 2.8765 |
| 30 | 2.849 (-149.8870689) | 2.8419 (8.9 mEh) | 2.7500 | 2.8492 (0.72 mEh) | 2.8336 |
| 32 | 2.835 (-149.8957245) | 2.8218 (9.8 mEh) | 2.5967 | | |

The uncontracted MRCISD with the larger reference is exact to 0.001 here.
**The internally contracted MRCI must not be used with the CAS(8,12)
reference for this property:** its energy is closer to FCI than that of the
small reference, but the splitting error grows with the space (ic CAS(8,12):
1.98 with 30 natural virtuals, 1.79 with 40, 1.74 in the full space, where the
uncontracted CAS(2,4) values are 2.18 to 2.20).  The contracted space is 6 %
of the uncontracted one (2.0 M of 35 M functions; 57 % for CAS(2,4)).  The
uncontracted CAS(8,12) needs dense operator matrices of the active space
(`Sigma._gamma_dense`) that exceed 15 GB already for n = 28; CAS(12,16) is
out of reach of this code (one array of 450 GB), these are left to zgasci
(GAS: at most two holes in 2s, at most two electrons in the virtuals).

**Complete double-zeta spaces** (52 correlated spinors, nothing truncated;
`methods_full.py`, `big_ref.py`): DYVDZ-SOG (made here with
`socutils.tools.basisopt.contract O --basis dyall-v2z --occupied --keep
"s:1;p:1" --kind sog --gau-nuc`, 10s6p1d -> 3s2p1d, in
`~/scr4/zmr_runs/o2_cbs/dz_basis`; CASSCF 2.3064 = 94 % of the limit) and
contracted cc-pVDZ.  12 electrons / 2s frozen:

| | DYVDZ X2CMP | DYVDZ X2Ccorr | cc-pVDZ X2CMP | cc-pVDZ X2Ccorr |
|---|---|---|---|---|
| CASCI | 2.3064 | 3.8594 | 1.9858 | 3.5137 |
| PC-NEVPT2 | 1.5594 / 1.7239 | 3.2931 / 3.3222 | 1.3786 / 1.5281 | 3.1306 / 3.1436 |
| CASPT2 (3 states), ic | 0.2636 / 0.6132 | 1.8250 / 2.0150 | 0.4108 / 0.7115 | 1.9759 / 2.1168 |
| ic-MRCI | 2.1397 / 2.1492 | 3.6841 / 3.5064 | 1.8085 / 1.8397 | 3.3855 / 3.2295 |
| unc. MRCISD | 2.1590 / 2.1728 | 3.6916 / 3.5117 | 1.8209 / 1.8554 | 3.3829 / 3.2238 |
| unc. +Q (Davidson) | 2.2453 / 2.2420 | 3.7121 / 3.4921 | 1.8913 / 1.9234 | 3.3913 / 3.2033 |
| unc. AQCC | 2.2674 / 2.2240 | 3.7236 / 3.5020 | 1.9150 / 1.9046 | 3.4008 / 3.2112 |
| CAS(8,12) unc. MRCISD, 12 e | 2.2235 | 3.6761 | 1.9051 | 3.3662 |
| CAS(8,12) unc. +Q, 12 e | 2.1062 | 3.6603 | 1.8167 | 3.3827 |
| CAS(8,12) unc. AQCC, 12 e | 2.1293 | | 1.8351 | |
| CAS(8,12) ic-MRCI, 12 e (not usable) | 1.9997 | | 1.3797 | |

In the complete space the larger reference raises the uncontracted X2CMP
MRCISD by 0.065 / 0.084 (X2Ccorr: -0.016 / -0.017), towards the +Q and AQCC values of the small reference (unlike
in the truncated spaces above); its own Davidson correction is negative and
not smaller (-0.12 / -0.09).  CASPT2 with three model states nearly removes
the splitting (6 states: 1.50).  Valence FCI in the natural virtuals, CAS(2,4)
/ CAS(8,12) / CAS(12,16): 212-spinor basis X2CMP 2.4570 / 2.9770 / 3.2178,
X2Ccorr 3.9948 / 3.9835 / 4.3018; DYVTZ-SOG 2.4209 / 2.9420 / 3.1779 and
3.9615 / 3.9494 / 4.2632; DYVDZ-SOG X2CMP 2.3064 / 2.8575 / 3.0836: the
valence correlation raises the spin-orbit part by 0.5 and leaves the X2Ccorr
total unchanged, and the X2Ccorr MRCI values are insensitive to the reference
and to +Q / AQCC (DYVDZ 3.66 to 3.72), the X2CMP ones are not (2.11 to 2.27).
The exact FCI of these spaces is being
computed by the zgasci session; no verdict on NEVPT2 against MRCI before.

**Basis-set convergence of the pieces, 1s correlation** (2026-10-13; logs
`o2_cbs/<dir>.ae.*.log`, `<dir>.cc.*.log`; uncontracted sets, no virtual
cutoff unless noted).  PC-NEVPT2, X2CMP, 1s frozen / all 16 electrons:

| basis (spinors) | CASSCF | 1s frozen | all-electron | 1s contribution |
|---|---|---|---|---|
| DYVTZ-SOG (120, contracted) | 2.4209 | 1.6000 | 1.5987 | -0.001 |
| DYVQZ-SOG (220, contracted) | 2.4544 | 1.6899 | 1.6973 | +0.007 |
| aug-cc-pVTZ + 2 tight p (256) | 2.4570 | 1.6708 | 1.6801 | +0.009 |
| aug-cc-pVTZ + optimised tight p (256) | 2.4584 | 1.6716 | 1.6809 | +0.009 |
| aug-cc-pCVTZ + the same tight p (308) | 2.4599 | 1.6798 | 1.7000 | +0.020 |
| dyall av3z (284) / acv3z (304) | 2.4527 | 1.6681 / 1.6693 | 1.6775 / 1.6848 | +0.009 / +0.016 |
| dyall av4z (444) / acv4z (512) | 2.4621 | 1.7083 / 1.7105 | 1.7214 / 1.7323 | +0.013 / +0.022 |
| aug-cc-pVQZ + tight p (384) / aug-cc-pCVQZ + tight p (500) | 2.4586 / 2.4600 | 1.7039 / 1.7096 | 1.7167 / 1.7331 | +0.013 / +0.024 |
| aug-cc-pV5Z (576) | 2.4515 | 1.7161 | 1.7331 | +0.017 |

- 1s correlation adds +0.02 with core-correlating functions, half of that
  without them, nothing in a contracted valence set.  The core-valence sets
  do not change the valence numbers (1s frozen: +0.001 to +0.008).
- emax = 100 Eh does not affect the frozen-1s values (<= 0.001); all-electron
  values lose up to 0.008 with it and must be run with `--emax 0`.
- NEVPT2 - CASSCF (1s frozen): -0.785 (TZ sets with tight p), -0.753 (QZ),
  -0.735 (5Z): the correlation contribution seen by NEVPT2 changes by 0.03
  from TZ to QZ and 0.02 more to 5Z.
- CASCI with the complete X2Ccorr (`--corr`): aug-cc-pVTZ + 2 tight p 3.995,
  with the optimised tight p 3.996, dyall av3z 3.992, aug-cc-pVQZ + tight p
  3.998, dyall av4z 4.002, aug-cc-pV5Z 3.992 (its CASSCF 2.4515 lacks the tight p); correction
  +1.538 to +1.540 in all of them.  The CASCI limit with
  X2Ccorr is 4.00 (experiment 3.96); what remains open is the dynamic
  correlation contribution (-0.57 NEVPT2, -0.12 to -0.17 uncontracted
  MRCISD(2,4)).

**Bond-length dependence and vibrational average** (2026-10-13; driver option
`--bond`, `~/scr4/zmr_runs/o2_cbs/scan`, `analyse.py`).  SA-3 CASSCF(2,4)
reoptimised at 16 bond lengths from 1.08 to 1.36 A, uncontracted aug-cc-pVTZ
+ optimised tight p (DYVTZ-SOG in brackets); vibrational functions of the
Morse potential with the experimental constants (r_e 1.20752 A, omega_e
1580.19, omega_e x_e 11.98 cm-1), <r>_0 - r_e = 0.0048 A:

| | D(r_e) | D' / cm-1 A-1 | <D>_0 - D(r_e) | <D>_1 - <D>_0 |
|---|---|---|---|---|
| CASSCF, X2CMP | 2.4584 | +1.83 (+1.70) | +0.0090 (+0.0086) | +0.0187 (+0.0178) |
| CASCI, complete X2Ccorr | 3.996 | +2.05 (+1.90) | +0.0093 (+0.0088) | +0.0191 (+0.0182) |
| PC-NEVPT2, X2CMP | 1.6716 | -0.73 (-0.73) | -0.0044 (-0.0039) | -0.0084 (-0.0074) |

The same scan with the uncontracted MRCISD(2,4) in DYVTZ-SOG (116 spinors,
all-spinor X2Ccorr recomputed at every bond length; `analyse_ci.py`):

| | D(r_e) | D' / cm-1 A-1 | <D>_0 - D(r_e) | <D>_1 - <D>_0 |
|---|---|---|---|---|
| X2CMP unc. MRCISD / AQCC | 2.1963 / 2.3001 | +0.93 / +1.49 | +0.0054 / +0.0105 | +0.0113 / +0.0219 |
| X2Ccorr CASCI | 3.9615 | +1.89 | +0.0087 | +0.0181 |
| X2Ccorr PC-NEVPT2 | 3.3893 | -0.62 | -0.0054 | -0.0104 |
| X2Ccorr unc. MRCISD | 3.8257 | +0.90 | +0.0038 | +0.0081 |
| X2Ccorr unc. MRCISD+Q | 3.8408 | +0.95 to +0.98 | +0.005 | +0.010 |
| X2Ccorr unc. AQCC | 3.8571 | +1.04 | +0.0059 | +0.0123 |

MRCISD halves the slope of the CASCI; its <D>_1 - <D>_0 (+0.008 to +0.012
with +Q / AQCC) is at the experimental +0.010 (still unverified), that of
NEVPT2 has the opposite sign.  The correlation contribution grows with r:
MRCISD - CASCI -0.011 (1.08 A), -0.136 (r_e), -0.291 (1.36 A); NEVPT2 -0.281,
-0.572, -0.981.  The vibrational correction of the v = 0 value at the
correlated level is +0.004 to +0.006.

The vibrational correction of the v = 0 value is +0.009 at the CASSCF level,
almost all of it D' (<r>_0 - r_e).  The experimental number quoted as 3.96 is
the v = 0 value, 2 lambda_0 = 3.9695 cm-1 (lambda_0 = 1.9847511).  lambda
increases with v (about +0.005 per quantum, i.e. +0.010 in D; from memory,
NOT verified against the original papers): CASSCF has the right sign and
twice the size, NEVPT2 the wrong sign.  The CASSCF(2,4) potential itself
(r_e 1.1525 A, G(1) - G(0) = 2021 cm-1) is not usable for the average.

**Correlation contribution along the SOG series and 1s correlation with
X2Ccorr** (2026-10-14).  All-spinor X2Ccorr, 1s frozen, no cutoff; method
minus CASCI (CASCI 3.8594 / 3.9615 / 3.9961 for DYVDZ / DYVTZ / DYVQZ-SOG;
the QZ run: bigmem, 2 h 09 min, 200 GB, `o2_dyvqz_sog_x2ccorr.npz`):

| | DZ | TZ | QZ | X2CMP: DZ / TZ / QZ |
|---|---|---|---|---|
| unc. MRCISD(2,4) | -0.168 | -0.136 | -0.107 | -0.147 / -0.224 / -0.161 |
| unc. MRCISD+Q | -0.147 | -0.119 | -0.085 | |
| unc. AQCC | -0.136 | -0.104 | -0.071 | -0.039 / -0.121 / -0.049 |
| PC-NEVPT2 | -0.566 | -0.572 | -0.555 | -0.747 / -0.821 / -0.764 |

With X2Ccorr the contribution shrinks by 0.03 per step, the same in MRCISD,
+Q and AQCC (QZ values: MRCISD 3.8888, +Q 3.9111, AQCC 3.9255); the
spin-orbit part alone is not monotonic.  NEVPT2 is no probe of this trend.
The sets with core functions of the basis-optimisation session (DYCVTZ-SOG,
172 spinors; its DYVDZ-SOG is identical to the one made here): PC-NEVPT2,
X2CMP, 1s frozen / all-electron 1.6348 / 1.6513 (+0.017; variants +0.013 to
+0.019); all-spinor X2Ccorr: NEVPT2 3.3972 / 3.4394 (+0.042), uncontracted
MRCISD 3.8312 / 3.8672 (+0.036), AQCC 3.8642 / 3.9138 (+0.050), CASCI 3.9624
in both.  The 1s contribution is +0.04 with the two-electron Gaunt / Breit
terms and +0.02 without.

Frozen natural virtuals (uncontracted MRCISD, CAS(2,4), X2CMP / all-spinor
X2Ccorr): 30 virtuals 2.1816 / 3.7879, 40: 2.1905 / 3.8221, 46: 2.1812 /
3.8213, 60: 2.1906 / 3.8306, all 102: 2.1964 / 3.8257.  Truncation in energy
order does not converge like this.

## Not yet recorded elsewhere

The numbers of sections 1 - 4 are not in `derivations/validation-log.md`
(only O2 with X2CAMF is), and the result page
https://claude.ai/artifact/7dAuLQuEQVawxcsnkd3T89 has the state before the
"shell" runs, the 6p atoms, Nd3+, Pr3+ and Ar.
