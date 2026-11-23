# Basis-set optimisation with the X2CAMF atomic program: status

Updated 2026-10-07.  Code: `~/apps/x2camf-basis` (branch `basis-opt`, worktree of
~/apps/x2camf, nothing pushed), package `basisopt/` (see its README.md).  Runs: this
directory (`env.sh`, `validate/`, `opt/`, `contract/`, `build_logs/`).

## Verified from TASK.md
- pybind interface, flag bits, the 13 matrices of `atm_integrals`: as described.
- Pitfall confirmed in the source (INT_SPH constructor, src/int_sph_basic.cpp): primitives
  are counted per l and the exponents then taken consecutively, so they must be grouped by
  l.  Second pitfall found: occupations are read from a file `input` in the current
  directory if it contains `%occAMFI_<EL>` (~/apps/x2camf/input has one for U).
- The working tree of ~/apps/x2camf has an uncommitted change of src/dhf_sph.cpp (OpenMP
  pragmas, timings); the installed .so there was built with it.  The worktree is plain
  master + my additions, built separately (`build_logs/build.sh`).
- ND:DYPVTZ-SO in ~/apps/cfour-x2cso/basis/GENBAS indeed has an empty spin-orbit line.

## Done
1. Engine (`basisopt/engine.py`, new pybind function `libx2camf.atom_scf`): primitives by l
   -> SCF energy, spinor energies, coefficients; 4c DC / DCG / DCB / spin-free, X2C-1e,
   X2C-1e + X2CAMF.  0.03 s (O, DC) to 0.4 s (O, DCB) per SCF.
2. Optimiser (`basisopt/optimize.py`), contraction and GENBAS I/O (`contract.py`,
   `genbas.py`).

## Validated
- 4c DC energy against PySCF DHF, same primitives, point nucleus (`validate/v1_pyscf.out`):
  Ne cc-pVTZ 1e-9 Eh, Kr cc-pVDZ 3e-7 Eh, spinor energies 1e-7.  (Ar: PySCF did not
  converge, no comparison.)
- O atom, eps(2p3/2) - eps(2p1/2) of the DCB atomic DHF against the basis reproduces the
  ratios of the CASSCF 3P fine structure of the zmr session to 5e-4 (`validate/v2`):
  TZ 0.9669 (zmr 0.9676), +1p 0.9932 (0.9937), +2p 0.9976 (0.9980), QZ 0.9861 (0.9863),
  5Z 0.9975 (0.9974), 6Z 0.9995 (0.9995); reference 7Z-sp + TZ.
- Contraction recipe: XE:PWCVQZ-SO reproduced to 5e-9 (X2C-1e HF + X2CAMF(DC) spin-orbit,
  point nucleus); BR:PVTZ-SO to 1.6e-4 (s) / 2.3e-3 (p) with the configuration-average SCF.
  The generated -SO entries are read correctly by socutils parse_genbas(so_basis=True).

## Preliminary (design not yet agreed with the user)
Objective used: (D/D_ref - 1)^2 for the valence shell, D = eps(np3/2) - eps(np1/2), 4c DCB,
Gaussian nucleus, fractional-occupation SCF; D_ref = same set with the tight region of the
optimised l saturated.  Ratios below are relative to that saturated limit.

| fixed set (uncontracted) | added | optimised exponents | optimised | geometric |
|---|---|---|---|---|
| O aug-cc-pVTZ | 1 p | 225.713 | 0.99799 | 0.99524 |
| O aug-cc-pVTZ | 2 p | 1223.0, 182.873 | 0.99986 | 0.99959 |
| O aug-cc-pVQZ | 1 p | 418.016 | 0.99923 | 0.99787 |
| O aug-cc-pVQZ | 2 p | 2288.84, 340.268 | 0.99998 | 0.99984 |
| O aug-cc-pV5Z | 1 p | 1314.14 | 0.99986 | 0.99955 |
| O aug-cc-pV5Z | 2 p | 7286.64, 1067.54 | 1.00000 | 0.99996 |
| S aug-cc-pVTZ | 1 p | 3853.75 | 0.99973 | 0.99919 |
| S aug-cc-pVTZ | 2 p | 21430.7, 3135.39 | 0.99998 | 0.99992 |

- Energy as objective gives similar exponents (O TZ: 196.7 instead of 225.7; 1103, 185.8
  instead of 1223, 182.9); DC instead of DCB gives the same (1220, 182.8).
- The saturated-tight-p limit of O aug-cc-pVTZ is 0.9978 of the 7Z-sp reference (QZ 0.9992,
  5Z 0.9999): the rest is the intermediate p region and cannot be recovered by tight
  functions.  Tight s have no effect (0.99782 vs 0.99783).
- S aug-cc-pVTZ unaugmented is already at 0.9956 for 3p (largest p exponent 574).
- S with 2p and 3p as targets gives the same exponents (21358, 3125.7); 2p and 3p errors of
  aug-cc-pVTZ are equal (0.9959 / 0.9956).  S aug-cc-pVQZ: 0.9989 unaugmented, 1 p
  (10527.4) -> 0.99993.
- Se aug-cc-pVTZ: 4p already 0.9985 unaugmented, 1 p (53845.7) -> 0.99985 (geometric
  0.99968).  The Se run with 2 p was killed (out of memory at 12 GB with DCB): rerun with
  more memory or DC.
- Pattern: the deficiency is large only where the valence shell is the FIRST shell of its
  l (O 2p: 3.3 %; S 3p: 0.44 %; Se 4p: 0.15 %), because core shells of the same l force
  tight exponents into energy-optimised sets.  By the same argument 3d (first d) and 4f
  (first f) shells should be checked.
- Molecular check on O2 by the zmr session (SA-3 CASSCF(2e,4), X2CMP(DCB), cm-1; logs
  ~/scr4/zmr_runs/o2_cbs/{tzo_p1,tzo_p2,qzo_p1}.pt.*.log): TZ+1p(225.713) 2.4491,
  TZ+2p(1223.0/182.873) 2.4584, QZ+1p(418.016) 2.4586; predicted from the squared atomic
  ratio 2.444 / 2.454 / 2.457; geometric TZ+1p 2.436, TZ+2p 2.457; plain TZ 2.307,
  QZ 2.397; limit 2.462-2.465.  So the atomic objective transfers to the molecule.
- 2026-10-07: the user asked to stop running optimisations until the design is agreed
  (separate exponents per zeta level vs one primitive set contracted to several levels;
  how many primitives per level; objective).  The Se runs (2 p; p+d) and the experiment
  optimising all p exponents of O were stopped unfinished.
- Demonstration contracted sets, O aug-cc-pVTZ + p(225.713): `contract/GENBAS`
  (O:APVTZ+P1-SF, -SO, -SOB; `contract/make_o_sets.py`).  Not yet tested in a molecule.

## Findings about the existing GENBAS entries
- O:PVTZ-X2C (and presumably the other light-element -X2C sets) were not made with the
  X2CAMF generator: 1s/2s agree to 1e-6 with a SFX2C-1e ROHF of 3P, and the 2p contraction
  is the average of the doubly and singly occupied ROHF 2p orbitals.  The AOC recipe gives
  the same space to 1e-7, coefficients to 6e-3.

## Open questions for the user
1. Objective: valence splitting only, or core shells too (S 2p, Se 3p/3d) with weights?
2. How many functions per l (O TZ: 1 p -> 0.998, 2 p -> 0.9999 of the tight limit)?
3. Names of the new sets; should the tight primitives be contracted into the np function
   only (as in the demonstration) or also left uncontracted?
4. How were the -X2C and the open-shell -SO sets generated (CFOUR ROHF? xx2cbasis version)?
5. Which Dyall set of Hg was wrong (PySCF dyall-v4z g exponents 3.1505 1.9906 0.78325
   0.3082 look suspicious)?

## Not done yet
- Heavy elements (tight p, d), Te and below (no cc sets in PySCF for Te: needs the user's
  GENBAS as source).
- Screening tool for existing sets (exponent-ratio smoothness, TZ/QZ extrapolation).
- Test of the contracted tight-augmented sets in a molecule.

## Effect of nuclear model and two-electron Hamiltonian on what the basis needs (2026-10-07)
`validate/v5_models.py`, output `validate/v5_models.out` (+ `v5_models_Rn_point.out`; the Rn
Gaussian-nucleus DC part was not run).  No exponent optimisation: tight s or p added to a
fixed uncontracted set (ratio 2.5), errors relative to the longest extension (14 s, 12 p),
which is itself NOT converged for the point nucleus in heavy atoms.  Errors of the
unextended set:

| atom / set | setting | no tight s: dE/Eh | no tight p: dE/Eh | no tight p: valence D(np) |
|---|---|---|---|---|
| O aug-cc-pVTZ | Gauss DCG | 6.4e-4 | 1.4e-3 | -3.0e-2 |
| | point DCG | 6.4e-4 | 1.4e-3 | -3.0e-2 |
| | Gauss DC | 6.4e-4 | 1.4e-3 | -2.8e-2 |
| Se cc-pVTZ | Gauss DCG | 7.4e-4 | 5.0e-3 | -1.5e-3 |
| | point DCG | 1.9e-3 | 5.0e-3 | -1.5e-3 |
| | Gauss DC | 7.5e-4 | 5.0e-3 | -1.5e-3 |
| Xe dyall-v2z | Gauss DCG | -6.5e-5 | 3.8e-3 | -1.5e-4 |
| | point DCG | 1.3e-2 | 4.1e-3 | -1.7e-4 |
| | Gauss DC | -6.5e-5 | 3.8e-3 | -1.5e-4 |
| Rn dyall-v2z | Gauss DCG | -2.3e-3 | 1.3e-3 | -6.2e-6 |
| | point DCG | 0.92 | 2.8e-2 | -1.3e-4 |

- DC vs DCG: same convergence everywhere -> one set for both; use DCG (twice as fast as
  DCB, valence splittings within 4e-5 of DCB, `validate/v4_ham.py`).
- Nuclear model: no effect for O; grows with Z and sits mainly in s (Rn point nucleus:
  0.92 Eh missing, still 6.5e-4 Eh after 10 more s) and to a lesser extent in p (Rn: 28 mEh
  and a 20 times larger splitting error than with the Gaussian nucleus).  Heavy-element
  sets must be made per nuclear model, or for the Gaussian nucleus only (proposed).
- Negative dE with the Gaussian nucleus (Xe, Rn): prolapse; "lower energy" is not a safe
  convergence criterion for heavy atoms.
- Not looked at: tight d, f; 7p and actinides.

## Other findings of the discussion (2026-10-07)
- cc vs Dyall for O: with the same number of p primitives the exponents are the same
  (cc-pVQZ 6p max 63.42 / dyall-v2z 6p 63.34; cc-pV5Z 8p 195.5 / dyall-v3z 8p 194.3;
  cc-pV6Z 10p 525.6 / dyall-v4z 10p 556.6).  The cc deficiency for spin-orbit is the NUMBER
  of p primitives per level, not the nonrelativistic optimisation.
- CAHF quantities as spin-orbit measure (`validate/v6_cahf.py`, O): the CAHF orbital-energy
  difference is the splitting plus about 1940 cm-1 of exchange, itself varying by 5 cm-1
  with the basis; total-energy differences of jj configurations are dominated by the
  multiplet splitting.  Use the CAHF energy as optimisation objective, the
  fractional-occupation splitting as spin-orbit measure.
- Dyall's procedure (Gomes, Dyall, Visscher, TCA 127, 369 (2010), read): DC, Gaussian
  nucleus, average energy of the configuration; l-optimisation cycled over l; different
  configurations for different l (d on 4f^n-1 5d^1 6s^2, 6p on 6s^2 -> 6p^2); set sizes by
  consistency along the row and the description of the outermost maxima.

## GRASP as numerical Dirac-Fock reference (2026-10-07)
- GRASP2018 is installed in ~/apps/grasp (bin on PATH; source commit 8f6c08db).  Nuclear
  models: point (A = 0) or two-parameter Fermi (a = t / (4 ln 3), t = 2.30 fm default, c
  from the rms radius; potential semi-analytic in src/lib/lib9290/nucpot.f90).  No Gaussian
  or uniform-sphere model.  Default c = 137.035999139.  Breit only perturbatively (rci).
- Driver `grasp/grasp_df.sh <dir> <Z> <A> <rms|-> <config> <active> [c] [RNT H N]`:
  rnucleus, rcsfgenerate, rangular, rwfnestimate (Thomas-Fermi), rmcdhf with explicit speed
  of light, radial grid and ACCY = 1e-12 (the non-default prompt sequences are in the
  script).  Grid: R(i) = RNT (exp((i-1) H) - 1), N <= 590 (compile-time); the default
  point-nucleus grid (N = 220, H = 1/16) gives only 5 digits in the spinor energies.
  Closed shells only so far (one CSF); open shells need the level/weight prompts.
- Ne, point nucleus, DC, c = 137.0359895: GRASP E = -128.6919695138, eps = -32.817471681,
  -1.9358460166, -0.85282949745, -0.84826681548 (stable to 1e-10 for three grids).
  X2CAMF in an even-tempered basis (ratio 2, s up to 1e9..1e13): E higher by 5.5e-7..3.3e-7
  Eh, eps(1s) by 1.5e-7, others < 8e-8.  The residual decreases very slowly with tighter
  functions (point-nucleus singularity); a finite-nucleus comparison needs the Fermi model
  in X2CAMF (not implemented yet).

## Numerically integrated nuclear models in X2CAMF; Xe against GRASP (2026-10-07)
- INT_SPH::set_nuclear_model / nuc_zeff / nuc_radial (src/int_sph_basic.cpp, used by get_h1e
  for "nuc_attra" and "s_p_nuc_s_p*"): Fermi (closed form of GRASP's NUCPOT), Gaussian,
  uniform sphere; the point-nucleus radial integrals minus a Gauss-Legendre quadrature of
  r^n exp(-a r^2) (1 - Z_eff/Z) over the nuclear region.  Python: engine.atom(...,
  nucleus=('fermi', c, a) | ('gaussian', xi) | ('sphere', R)), engine.fermi_from_isodata.
- Self-test (`validate/v7_nucmodel.py`): Gaussian nucleus by quadrature vs the analytic
  integrals: |dE| < 3e-9 Eh, |d eps| < 6e-11 (Ne, Xe, Rn).  Same rms radius, uniform sphere
  minus Gaussian in dyall-v2z: Xe 1.2 mEh, Rn 97 mEh -> the model must be identical when
  comparing programs.
- Xe, Fermi nucleus (A = 132, GRASP defaults: c = 5.6450 fm, a = 0.52339 fm), DC,
  c = 137.0359895.  GRASP: E = -7446.89852155 Eh (`grasp/xe_fermi`).  X2CAMF in
  even-tempered s/p/d sets (`grasp/compare_grasp.py`, `grasp/xe_fermi_compare.out`):

  | ratio | n(s/p/d) | E - GRASP / Eh | d eps(1s) | d eps(5p3/2) |
  |---|---|---|---|---|
  | 2.0 | 37/33/27 | 5.5e-4 | -9.0e-5 | 8.5e-7 |
  | 1.9 | 39/36/29 | 8.1e-5 | -1.2e-5 | -6.0e-8 |
  | 1.8 | 43/39/31 | 9.0e-6 | -1.6e-6 | 4.5e-8 |
  | 1.7 | 47/43/34 | 5.5e-7 | -1.4e-6 | 3.2e-8 |

  The basis-set result converges onto the numerical one (7e-11 relative in E), from above.
- The dense sets do not reach conv = 1e-10 in the density (noise about 1e-9); use 1e-8.
  atom_scf now returns NaN for the energy of an unconverged SCF (it was stale memory).

## Memory leak of X2CAMF fixed on the branch (2026-10-07)
- int2eJK arrays were never freed: every amfi / atm_integrals / pcc_K call kept all
  two-electron integrals of the atom until the process ended (present in master too).
- Fix: commit "Free the two-electron integral arrays" on basis-opt (include/general.h,
  include/dhf_sph.h, src/dhf_sph.cpp only; can be cherry-picked onto master).
- Test `validate/v8_leak.py` (Xe dyall-v2z, 8 call types incl. Gaunt/gauge, SD-Gaunt, PT,
  PCC, AOC): installed library grows by 0.18-0.68 GB per three repeated calls, the fixed
  one by 0.00; the 32 returned matrices are identical (max difference 2e-314).
- Not freed (only when allInt is false, which the pybind interface never uses): partially
  filled integral arrays.
- 2026-10-07, at the user's request: `leak_fix_for_working_tree.patch` (leak fix merged with
  the user's uncommitted OpenMP/timing change) applied to the working tree of ~/apps/x2camf
  (include/dhf_sph.h, include/general.h, src/dhf_sph.cpp; not committed).  Compiled out of
  tree and tested: no memory growth, 32 matrices identical to the installed library.  The
  installed x2camf/libx2camf*.so there was NOT replaced (running jobs load it); the tested
  build is `libx2camf_main_patched.cpython-311-x86_64-linux-gnu.so` in this directory.

## O: error against the number of s / p primitives, and the size rule (2026-10-07)
`curves/o_curves.py` (.out, .json): N exponents of one l fully optimised (CAHF energy, DC,
Gaussian nucleus), other l dense; evaluated with DCG against a dense even-tempered
reference (E_CA = -74.825135783, D(2p) = 274.8481 cm-1).  `curves/o_existing.py`: the s and
p sets of cc and Dyall sets in the same measure.  `curves/o_corr_gate.py`: valence CCSD
contribution of each polarisation shell of cc-pVnZ (nonrelativistic, ROHF 3P).

Energy error (mEh), best of own optimisation and existing sets (own BFGS runs for
N_s >= 10 and N_p >= 8 did not reach the optimum: the cc/Dyall sets are lower there):
  s: 6: 84.3  7: 24.2  8: 8.53  9: 3.90  10: 1.77  11: <=0.80  12: 0.272  14: 0.061
     16: 0.0128  18: 0.0029      (factor about 2.2 per function)
  p: 3: 86.3  4: 19.9  5: 4.81  6: 1.23  7: 0.33  8: 0.094  10: 0.0089
                                 (factor about 4 per function)
Relative error of D(2p) against N_p: 4: -11.7 %  5: -5.6 %  6: -2.7 %  7: -1.3 %
  8: -0.63 %  10: -0.16 %  (factor about 2 per function = square root of the energy
  factor; N_s has no influence, < 1e-4).
- Own optimisation reproduces the cc energies exactly where converged (9s: 3.90 / 3.90;
  4p 19.9 / 19.9; 5p 4.81 / 4.82; 6p 1.23 / 1.23 mEh): the cc s, p exponents are the
  energy optimum, also at the 4c level.  dyall-v2z = 10s6p, v3z = 14s8p, v4z = 18s10p give
  the same errors as the cc sets of the same size.
- Entry thresholds (smallest polarisation-shell contribution): DZ 61.8, TZ 15.0, QZ 3.7,
  5Z 1.1 mEh.
- Rule "add while the next primitive lowers the energy by more than the threshold": p gives
  4 / 5 / 5-6 / 6 (cc: 4 / 5 / 6 / 8), s gives 6-7 / 8 / 9 / 10 (cc: 9 / 10 / 12 / 14).
- What cc actually corresponds to: total s+p SCF error = 0.4 x threshold:
  9s4p 23.8 mEh (0.39), 10s5p 6.6 (0.44), 12s6p 1.50 (0.41), 14s8p 0.155 (0.14).
  Dyall is far stricter: 10s6p 3.0 mEh, 14s8p 0.155, 18s10p 0.012.
- Consequence for spin-orbit: with energy-chosen sizes D(2p) is off by 12 / 5.6 / 2.7 /
  0.66 % at DZ / TZ / QZ / 5Z; one optimised tight p brings TZ to 0.42 %.

## Dyall 1998 (TCA 99, 366; read) and Gaussian-nucleus numerical references (2026-10-07)
- Optimisation target: degeneracy-weighted average energy of all states of the
  nonrelativistic s2p^n configuration (so that no spin-orbit component is favoured) = the
  CAHF functional of X2CAMF.  l-optimisation (j-optimisation said to be better for DZ).
  Sizes from the rare gases: p space has the largest truncation error and is fixed first
  (largest DZ p set, description of the radial maxima); other l so that one more function
  gives about the same energy lowering in each l.  DZ truncation errors 20-54 mEh.
- Tables 1-3 give numerical DF energies with the Gaussian nucleus.  X2CAMF, built-in
  Gaussian nucleus, DC, c = 137.0359895, even-tempered s/p/d (`grasp/xe_gauss_dyall.py`):
    Kr (numerical -2788.860624): ratio 2.0 +7.9e-5, 1.9 +1.0e-5, 1.8 +1.4e-6 Eh
    Xe (numerical -7446.895440): ratio 2.0 +5.5e-4, 1.9 +8.2e-5, 1.8 +9.5e-6 Eh
  i.e. the same convergence as against GRASP with the Fermi nucleus: the Gaussian-nucleus
  dense-basis values are validated directly (closed shells), to the 6 decimals tabulated.

## Moved to socutils (2026-10-07, at the user's request)
- Worktree `~/apps/socutils-git-basis`, branch `basis-tools` from master (local, not pushed);
  import through `PYTHONPATH=$HOME/apps/socutils-basis-env` (symlink `socutils`).  The
  user's other socutils checkouts and their built libraries were not touched.
- Commit 3861b3e: `x2camf_atom_scf` and the Fermi / Gaussian / sphere nuclear models ported
  to the pure-C X2CAMF (lib/x2camf), ctypes binding `atom_scf`, test
  `x2camf_c/tests/test_atom_scf.py` (6 tests pass; test_smoke and test_against_cpp pass).
- Commit 97e24c1: the Python tools as `socutils.tools.basisopt` (engine, optimize, contract,
  genbas, README with the validation record).  The engine loads lib/libx2camf_c.so by path,
  whatever `x2camf` is on the path.
- C backend vs the C++ development build, atom_scf: O, 9 Hamiltonians x (fractional, AOC),
  |dE| < 3e-11 Eh; Xe DCG / Fermi / Gaussian / sphere / X2CAMF, |dE| < 2e-8 Eh.
  End to end on the C backend: optimiser O aug-cc-pVTZ + 1 p -> 225.713 (same as C++),
  XE:PWCVQZ-SO contraction reproduced to 5e-9.
- Built there with the same configuration as the existing socutils build (conda gcc,
  libmkl_rt); only the x2camf target was built.
- The C++ branch basis-opt of ~/apps/x2camf remains as the prototype; for upstream only the
  leak fix (43c01fc) and the import-order fix (69ae398) are meant to be offered.
- Not ported / not done: flake8 is not installed here (no lint run); the scripts under
  ~/scr4/zmr_runs/basis_opt (validate/, curves/, grasp/) still import the C++ `basisopt`.

## Working scripts on the socutils path; contracted sets from the command line (2026-10-07)
- `env.sh` now puts `~/apps/socutils-basis-env` on PYTHONPATH; all scripts under validate/,
  opt/, curves/, grasp/, contract/ import `socutils.tools.basisopt` (v2_o_series re-run:
  same numbers as with the C++ prototype).
- socutils commit a-la "tools/basisopt: command line for contracted sets, contraction-error
  check": `python -m socutils.tools.basisopt.contract EL --basis NAME | --genbas F --entry E
  [--add p:225.713] --kind sf|so|sog|sob --name NAME --out GENBAS [--check] [--compare]`.
- `contract/make_o_sets.sh` -> `contract/GENBAS`: O:APVTZ+P1-SF, -SO, -SOG (the earlier file
  from the C++ prototype is GENBAS.cpp_prototype).  Parsed by socutils basis_parser.
- Contraction error of the O 2p splitting (--check, X2CAMF-Gaunt): SF contraction 0.99944,
  SF with the tight p also uncontracted 0.99970, j-adapted 0.99991 of the uncontracted
  value (TZ without tight p: 0.99961 / 0.99991).  Not yet tested in a molecule.

## SO vs SOG vs SOB contractions, rare gases, dyall-v3z (2026-10-07)
`contract/rare_gas_so_kinds.py` (.out).  Gaussian nucleus; all occupied shells contracted,
2 outermost s, p (1 outermost d, f) primitives also uncontracted.  Non-self-consistent
estimate: uncontracted Fock of Hamiltonian H diagonalised in the contracted space of K.
- Uncontracted valence D(np): Gaunt changes it by -8.1 % (Ne), -3.9 (Ar), -1.8 (Kr),
  -1.3 (Xe), -1.0 (Rn), -1.05 (Og); the gauge term by +0.010 ... +0.001 %.
- SOG contraction used with the Breit Hamiltonian (and vice versa): D(np) error 1.5e-9 (Ne),
  1.3e-8, 8.5e-8, 1.8e-7, 3e-8, 1.2e-6 (Og); largest spinor-energy error 7e-9 ... 4.6e-4 Eh.
- SO (no Gaunt) contraction used with the Gaunt Hamiltonian: D(np) error 5e-5 (Ne), 8.7e-5,
  1.3e-4, 1.7e-4, 2.6e-4, 4.8e-4 (Og); largest spinor-energy error 1.5e-6, 1.6e-5, 2.6e-4,
  1.3e-3, 7.8e-3, 2.9e-2 Eh.  Matching contraction and Hamiltonian: 1e-9 (check).
- Conclusion: one Gaunt-type set (SOG) serves DCG and DCB; SO vs SOG matters for the
  spinor energies of heavy atoms (mEh), little for splittings.
- Defaults now Gaunt: contract kind 'sog', optimiser Hamiltonian 'dcg' (user: SOG is the right choice in theory and in the results).

## TZ -> QZ extrapolation with Dyall sets, O atom (2026-10-07)
`curves/o_extrap.py` (.out): nonrelativistic ROHF + valence UCCSD, E_corr in mEh.
  cc-pVnZ (contracted):  D -121.58  T -165.20  Q -178.81  5 -183.66;  X^-3: (T,Q) -188.745, (Q,5) -188.744
  unc-cc-pVnZ:           D -135.20  T -169.07  Q -179.92  5 -184.06;  X^-3: (T,Q) -187.84,  (Q,5) -188.41
  dyall-v2z/v3z/v4z:     2 -139.54  3 -170.48  4 -180.61;             X^-3: (3,4) -188.00
  E(HF): dyall-v3z -74.812034, v4z -74.812332 (cc-pV5Z -74.812231): no SCF extrapolation needed.
- Dyall (3,4) is 0.74 mEh (0.4 %) from the cc (Q,5) limit, like uncontracted cc (0.9 mEh);
  contracted cc hits it.  The uncontracted s, p primitives act as extra radial correlating
  functions, more at low X, which is outside the one-shell-per-level pattern.

## Audit of the Dyall sets; the Hg 4z f error (2026-10-07)
- User's decision: keep Dyall's numbers of functions; verify his sets, redo an element only
  where something is wrong.  Sources: DIRAC 23 basis library (reader
  `audit/dirac_basis.py`), PySCF, CFOUR GENBAS (DYVxZUNC, ...).
- Hg dyall.*4z: the first nine f exponents are a copy/paste error, corrected by K. Dyall on
  the Dirac Users list (27 Feb): 1681.92948 567.566140 241.660175 115.945443 59.3846121
  31.6007908 17.1340403 9.21325305 4.82670353 (exponents 2-10 in ae4z, aae4z).  All local
  copies still have the old values: DIRAC 23 dyall.v4z/av4z/cv4z/acv4z/4zp/ae4z/aae4z,
  PySCF dyall-v4z, GENBAS HG:DYVQZUNC, HG:DYCVQZUNC, HG:DYPVQZ (not modified by me).
  The nine wrong numbers occur nowhere else in the DIRAC library (origin not pursued).
  The g set of Hg v4z (ratios 1.58, 2.54, 2.54) is by design: Pt and Au have the same pattern.
- Calibration of the gradient test (`audit/hg_v4z_f*.py`, DC, Gaussian nucleus, SCF
  threshold 2e-8 because the 4z sets do not reach 1e-9): max |dE/d ln(alpha)| over the f
  exponents: erroneous Hg 0.22 Eh, corrected Hg 9e-4, Au 1e-3; E(corrected) - E(erroneous)
  = -8.55 mEh.  Ratio to the neighbour (Hg/Au, per exponent index): erroneous 1.11-1.22
  then 1.07; corrected 1.05-1.06 rising smoothly to 1.10.
- Planned criteria: gradient > 1e-2 Eh suspicious (normal < 1e-3); smoothness of the ratio to
  neighbouring elements; cross-comparison of the three sources.  The audit must use DC
  (Dyall's functional); `audit/xe_dc_vs_dcg.py` tests whether DC-optimal exponents are also
  stationary with the Gaunt term.

## Exponent gradient in the C program (2026-10-07)
- User's scheme: converge the atom in the C program, then only contract the density with
  integrals over ONE displaced primitive at a time.  First prototyped with PySCF spinor
  integrals (`audit/pyscf_exp_grad.py`; correct, but 8 min for Xe because every block
  carries all m components: n2c^3 complex numbers per primitive function).
- C implementation, socutils commit "x2camf: exponent gradient of the atomic Dirac-Coulomb
  SCF energy": `engine.exponent_gradient(el, prims, gau_nuc=...)`.  Six integral
  evaluations for all exponents.  Xe dyall.v2z: 47 gradients in 2.6 s, max deviation from
  the PySCF prototype 7e-8 Eh.  Hg dyall.v4z: 96 gradients in 37 s (was about 50 min).
- Correction of an earlier statement: the brute-force finite differences with a 2 % step
  had a truncation error of up to 1e-4 Eh; the true gradients of Dyall's SCF exponents are
  1e-6 ... 1e-3 Eh.
- Hg dyall.v4z, max |g| per l without the two outermost functions: erroneous f set: s 4.5e-6,
  p 2.9e-4, d 2.3e-4, f 0.22; corrected: s 4.5e-6, p 3.0e-4, d 4.0e-6, f 9.4e-4.  Au: s 3.7e-5,
  p 4.2e-4, d 4.5e-4, f 1.1e-3.
- Limits: DC only (no Gaunt), fractional-occupation functional (not the CA functional that
  Dyall optimised for open-shell atoms: open-shell atoms may show larger gradients for that
  reason alone; not yet looked at).

## Gradient audit of dyall.v2z / v3z / v4z, all 118 elements (2026-10-07)
Slurm job bo_audit (31737988), `audit/scan_gradients.py` -> `audit/grad_dyall.v?z.json`,
`audit/scan_dyall.v?z.out`; summary `audit/analysis_tight_half.txt`
(`audit/analyse_gradients.py`).  DIRAC 23 library, DC, Gaussian nucleus (X2CAMF masses),
fractional ground-state occupations.  All 354 calculations converged (266 min in total).
Judged: the TIGHT HALF of the exponents of each occupied l (core region; the user's point:
core exponents are not affected by the open-shell treatment -- confirmed, e.g. Gd s 1e-9..1e-5).
- Closed-shell atoms: 1e-10 ... 1e-6.  Open-shell atoms: mostly 1e-7 ... 1e-3.
- The only gross outlier: Hg v4z f, 6e-2 (neighbours Au 2e-5, Tl 2e-6) = the known error.
- Milder local outliers with the same alternating pattern, 20-100 times above the
  neighbours: Ge v3z d (6e-3; As 1e-4), Sm v4z p (4e-3; Pm 2e-4), Se v4z d (1e-3; Br 8e-6).
  Estimated energy cost (scaled from Hg) below 1e-5 Eh: incomplete convergence, harmless.
- Systematic, by design: p of the s-, d-, f-block elements (K-Zn, Rb-Cd, Cs-Hg, Fr-Cn):
  v2z 1e-2 ... 7e-2, v3z 1e-3 ... 6e-3, v4z < 5e-4.  The outer p exponents were reoptimised
  for the np^2 configuration (Gomes, Dyall, Visscher 2010), the inner ones kept.
- Tight s of Z >= 103 (4e-3 ... 2e-2): mass numbers.  X2CAMF's table equals the
  Visscher-Dyall masses of DIRAC (src/include/nucdata.h) for Z <= 102 and differs for Lr,
  Rf, Db, Sg, Bh, Hs, Mt, Rg, Nh, Mc, Lv, Ts.  The gradient vanishes at A = 292 for Ts and
  A = 280 for Rg (X2CAMF 294, 282).
- H v2z s 1e-2: X2CAMF treats the one-electron atom specially; not followed up.
- Not covered: the outer half of each l (valence, functional-dependent), correlating
  functions (g, h, ...; outer d, f), the cv / ae / aug variants, the CA functional.
- The Z-smoothness check (`audit/scan_smooth.py`) is too noisy to use alone (configuration
  changes, shell openings); it ranks the Hg error below many legitimate irregularities.

## Core-correlating functions in contracted (X2C) sets (2026-10-07)
- User's concern = the problem Dyall's prescriptions address.  Checked in the DIRAC library:
  dyall.cvNz adds NO s, p, d functions to dyall.vNz (Xe, Pb, Hg; 2z and 3z), only f, g, h:
  the core-correlating s, p, d functions are SCF primitives.  In a contracted set the
  question is therefore which primitives stay uncontracted.
- Explicit prescriptions read in Gomes, Dyall, Visscher 2010, section 4 (lanthanides; dz,
  tz, qz; outer valence 5d/6s/6p, inner valence 4f/5s/5p, core 4d), primitives counted from
  the smallest exponent.  Other blocks: in the respective papers (not read).
- Tool: `contract --occupied --keep "s:1,3;p:1,3;d:1,2;f:1,3"` (socutils commit
  "contract uncontracted sets by Dyall-style prescriptions").  Demonstrated on Yb dyall-v3z
  with the three published TZ levels: j-adapted sizes 8s/10p/6d/4f, 10s/12p/9d/6f,
  12s/14p/11d/8f (+ 3g 2h).  Not verified by a correlated calculation.
- Test of the prescription (`contract/yb_core_corr.py`, .out): Yb dyall-v3z, SF-X2C-1e RHF +
  MP2 in PySCF (Gaussian nucleus), SF contraction of all occupied orbitals + the published
  TZ lists; fraction of the uncontracted MP2 correlation energy (uncontracted: 6s -0.01921,
  4f5s5p6s -1.49771, 4d4f5s5p6s -2.24079 Eh; 308 functions):
    kept primitives   functions   6s      4f5s5p6s   4d4f5s5p6s
    none                 84       0.011   0.179      0.161
    outer valence       116       0.854   0.374      0.321
    inner valence       153       0.921   0.958      0.909
    core 4d             185       0.922   0.987      0.987
  E(HF) = -14054.732990 for all five sets (contraction exact at the SCF level; with a point
  nucleus in PySCF and the Gaussian-nucleus contraction it is 0.25 Eh too high:
  yb_core_corr_pointnuc.out).  Scalar-relativistic and MP2 only.
- Which primitives matters (`contract/yb_which_prims.py`, .out; same test, same numbers of
  uncontracted primitives).  Recovered MP2 fractions for 6s / 4f5s5p6s / 4d4f5s5p6s:
    inner-valence level: Dyall list (as I read it) 0.921 / 0.958 / 0.909;
      the outermost consecutive primitives 0.998 / 0.860 / 0.796; list shifted inwards by
      one 0.825 / 0.927 / 0.927
    core-4d level: Dyall list 0.922 / 0.987 / 0.987; consecutive 0.9998 / 0.995 / 0.983;
      shifted by one 0.827 / 0.944 / 0.955
  An off-by-one in the ordinal numbers costs 3-10 %.  The current library files do not mark
  which exponents are SCF primitives (Yb v3z has 11 f, the paper's SCF set 10 f, and the three
  outermost f are even-tempered with ratio 2.68), so "the n-th primitive" of the 2010 paper
  cannot be mapped onto the files with certainty (user's concern).  Any pattern should be
  checked with this kind of test rather than trusted from the ordinal description.
- Own selection instead of the ordinal lists (user: the tables are too vague, do it
  ourselves): `socutils.tools.basisopt.select.Selector` (greedy, recovered MP2 fraction,
  1 s per candidate).  Yb dyall-v3z, `contract/yb_select.py` (.out), target 99 % per level:
    outer valence (6s):      keep s:3,4; p:2,3,4; d:1,2,4; f:1,2          2s3p3d2f   0.992
    + inner valence:         s:3,4,7; p:2,3,4,5,7; d:1-5; f:1-6             3s5p5d6f   0.991
    + core 4d:               s:3,4,7; p:2,3,4,5,7; d:1-7; f:1-7             3s5p7d7f   0.992
  (list read from the paper: 2s2p2d2f 0.854, 4s4p5d4f 0.958, 6s6p7d6f 0.987.)
  Closed-shell atoms, spin-free MP2, greedy (not proven optimal).
- 2026-10-07, user: use non-canonical MP2 (coarse screen with UMP2-type MP2, verify with
  spinor methods), start with light elements (lanthanides first is risky), a pattern from a
  closed-shell atom should carry over to its neighbours.  The lanthanide UHF transfer test
  (contract/ln_transfer.py) was stopped unfinished.
- Selector rewritten (socutils "select: non-canonical MP2 on a fixed atomic reference"):
  reference determinant from the contracted atomic orbitals, no SCF per candidate, open
  shells included.  Ar: E2 identical to PySCF RMP2 (-0.20992 valence, -0.33175 with 2s2p).
- `contract/light_select.py` (.out), dyall-v3z, target 99 %:
    Ar: valence s:1,3;p:1,3,4 (0.991); + 2s2p s:1,2,3,6;p:1,3,4,5,6 (0.990)
    Ar lists on Cl / S / P / Si: valence 0.992 / 0.993 / 0.993 / 0.993;
      + 2s2p 0.989 / 0.988 / 0.986 / 0.985  (own selections: 0.993 / 0.992 / 0.991 / 0.994)
  -> transfer within a block works to better than 1 %.

## Uncontracted primitives: Dyall's own prescription (2026-10-07)

- Dyall's archive files (Zenodo 10.5281/zenodo.7606546) are in `dyall_archive/files/` (57 files,
  block x dz/tz/qz).  Each has a section "Basis set construction" that states, counted from the
  smallest exponent, which SCF primitives to add for valence, core-valence, deeper core.  This
  replaces reading the ordinal tables in the papers and replaces my own valence selection.
- `dyall_archive/crosscheck.py` -> `crosscheck.out`: parses sets 1 and 2 of every file and compares
  with the DYVnZ-SF / DYCVnZ-SF entries of the cfour GENBAS.  Identical for 2p, 3p (all three
  zetas), 4p dz/tz, 5p dz/tz/qz (except Sn), 6p valence.  The GENBAS has contracted Dyall entries
  only for the p blocks.  Deviations to examine: 4p qz (GENBAS uses Dyall's "alternative" p set),
  TL..RN:DYCVQZ-SF (p primitive #5 uncontracted twice = linear dependence; Dyall's valence and
  core lists overlap there), SN:DYCVTZ-SF (s block looks wrong: 7.7e7 exponent listed last and
  uncontracted), 6p core f counts (probably only a counting offset from the added correlating f).
- MP2 recovered fraction, Ar, dyall.cv3z: Dyall valence 97.3 %; Dyall core-valence (s 4,6; p 4,6)
  94.1 % for valence+2s2p; with s 2,6; p 5,6 instead 98.2 %.  PRELIMINARY (MP2, one atom; Dyall's
  criterion was MR-SDCI on the 2s2p pair excitations).
- The earlier "99 % target" selections (Ne/Ar/Kr blocks, Yb) are superseded: counts follow the
  cc convention (1/2/3 per layer for dz/tz/qz), valence fixed first, core added on top.

## Dunning/Peterson sets as the reference for which primitives to uncontract (2026-10-08)

User's direction: primitives stay Dyall's; the cc sets (Dunning, Peterson) only tell where the
correlating functions sit -> uncontract the Dyall primitive nearest (log scale) to each
uncontracted cc exponent.  Valence from cc-pVnZ(-DK/-DK3/-X2C), core from cc-pwCVnZ minus cc-pVnZ.
- Sources: `basis_set_exchange` (installed, 0.13 dev) and Grant Hill's ccRepo (fills Kr core,
  Ag/Cd/Hg TZ-DK, Au DK3).  `ccrepo/ccrepo.py` downloads (2 s between requests, cached in
  `ccrepo/cache/`) and parses; `ccrepo/map_all.py` -> `ccrepo/map_all.out` is the comparison with
  Dyall's archive rule for the closed-shell atom of each block, DZ/TZ/QZ.
- p blocks: TZ valence identical to Dyall's rule for Kr, Xe, Rn (s 1,3; p 1,3; d 1,2), one p
  differs for Ne, Ar.  QZ: Peterson keeps the outermost functions where Dyall's first choice
  does not (Rn p: 1,2,4 vs Dyall 3,4,5).  Core sets agree to within one neighbouring primitive.
- s blocks: Dyall's set 1 already contains the (n-1)s(n-1)p functions; union of Peterson's
  valence+core equals Dyall's set 1 for Ra TZ, differs by one neighbour for Sr, Ba.
- d and f blocks: Peterson's valence sets get their flexibility from extra contracted functions,
  not from uncontracted primitives -> the valence part is not comparable this way; the core
  additions are (Zn, Cd, Hg, Yb, No within one or two neighbours of Dyall's rule).
- Peterson's Cs-Lu, Fr-Lr primitives: inner exponents identical to Dyall's, outer reoptimised.
- Not done: MP2 spot checks of the "one neighbour" differences; superheavy elements have no cc
  reference; Mg core mapping (7,9 vs Dyall 4,6) not understood.

## Basis-set files and records started (2026-10-08)

- Converter: `socutils.tools.basisopt.genbas` (worktree `~/apps/socutils-git-basis`, branch
  `basis-tools`, commit daab947, local only): PySCF internal format <-> GENBAS, plain and
  j-adapted (kappa).  `to-genbas` / `to-pyscf` commands; round-trip tests pass; the kappa output
  equals that of the existing `socutils/tools/basis_parser.py`.
- `basis/GENBAS` and `basis/pyscf/dyv{d,t,q}z_sog.py`: H and He, Dyall v2z/v3z/v4z primitives,
  1s contracted (X2CAMF with Gaunt, configuration average, Gaussian nucleus), uncontracted s
  per Dyall's archive (DZ 1; TZ 1,2; QZ 1,2,4), Dyall's correlating p/d/f.  Names
  `<EL>:DYV{D,T,Q}Z-SOG` are PROVISIONAL (naming is the user's decision).
  `basis/check_formats.py`: both formats give identical energies (1e-14) and reproduce the record.
- `records/` (`record_1s.py`, `H.json`, `He.json`, `H.out`, `He.out`): per element and zeta the
  primitives, atomic DC / X2C-1e energies, exponent gradient, Dyall rule vs Dunning positions vs
  all same-size choices, exact correlation energies (CCSD; He atom, H2 at 1.4 a0) incl. cc-pVnZ.
  Findings: Dyall's rule is the best choice for DZ, TZ; for QZ s 1,2,3 is better than his 1,2,4
  (H2 99.7 vs 98.9 %, He 98.9 vs 96.2 %) -- entries still follow Dyall, change not yet agreed.
  The H exponent gradient in the record (5e-2) is an artefact (fractional-occupation functional
  for one electron), to be marked invalid.
- Peterson 5Z-DK: main group = nonrelativistic cc-pV5Z exponents recontracted; no exponents
  shared between QZ and 5Z.  A Dyall-style 5Z is feasible (not started; needs the user's go).
- zmr (`~/apps/zmr/docs/usage.md`): spinor NEVPT2 and ic-MRCI available for verification and
  for optimising correlating functions; first test proposed (Ne, TZ, contracted vs uncontracted),
  not run yet.

## Web page for the basis sets (2026-10-08)

- `site/build_site.py` reads `basis/GENBAS` and writes `site/index.html` (complete static page for
  any web host, with download buttons), `site/artifact.html` (same page as a claude.ai artifact
  fragment, copy only) and `site/data.json`.  `site/template.html` is the page; a new family of
  sets is described in `FAMILIES` in the script.  Rebuild after adding entries: `python build_site.py`.
- Preview (private to the user): https://claude.ai/artifact/31zeNoyo6nzLnk8R4tN14Z
- Content now: H, He, DYV{D,T,Q}Z-SOG (provisional).  Not hosted publicly; where to host is the
  user's decision.  The page script was not run through a JavaScript checker (no node here).

## Registration with PySCF (2026-10-08)

- PySCF output of the converter now follows PySCF's own basis modules (one variable per
  element, internal format with kappa), commit bfd16a3; the site shows the same.
- `socutils/gto/basis/__init__.py` (commit on `basis-tools`, local): `import socutils` registers
  basis modules by file name (`basis='dyvtz-sog'`), from `socutils/gto/basis/` (empty: names
  are provisional) and from `SOCUTILS_BASIS_PATH` (e.g. `basis/pyscf` here).  Scalar SCF objects
  given a j-adapted set emit `ScalarWithSpinorBasisWarning` (user's decision: warn); PySCF X2C,
  DHF and socutils SpinorSCF stay silent.  6 tests pass.  Not covered: programs that never
  build a PySCF SCF object.
- Decision (user, 2026-10-08): registration by name and the scalar warning are to be written
  into PySCF itself when the socutils code is cherry-picked into PySCF.  Until then
  `socutils.gto.basis.register()` is opt-in only (commit 217cce0); `import socutils` does not
  touch PySCF.  TODO at that time: basis modules + ALIAS entries in `pyscf/gto/basis`, the check
  in PySCF's SCF (warning, not error).

## H and He: cc sets kept (2026-10-08, user's decision)

- H and He have no spin-orbit splitting; the Dyall-contracted sets were no better than cc-pVnZ
  (H2: DZ 5.1 mEh, TZ 1.6 mEh worse in total energy).  `basis/GENBAS` and
  `basis/pyscf/pv{d,t,q,5}z_x2c.py` now hold `H:` and `HE:` `PV{D,T,Q,5}Z-X2C`: cc-pVnZ exponents
  and pattern, 1s recontracted (X2CAMF-G, configuration average, point nucleus).  The H entries
  agree with the existing `H:PVnZ-X2C` of the cfour GENBAS to 1e-5 in the normalised
  coefficients; the cfour GENBAS has no He X2C entries.  Both formats verified identical
  (`basis/check_formats.py`).  The Dyall H/He entries are in `basis/superseded/`; the comparison
  data stay in `records/`.
- Site rebuilt (4 sets, H and He), same preview link.

## Oxygen, Dyall primitives with the spin-orbit contraction (2026-10-08)

- `basis/GENBAS` + `basis/pyscf/*_sog.py`: `O:DYVTZ-SOG` (14s8p2d1f -> 4s3p2d1f), `O:ADYVTZ-SOG`
  (5s4p3d2f), `O:DYVQZ-SOG` (5s4p3d2f1g), `O:ADYVQZ-SOG` (6s5p4d3f2g).  X2CAMF-G, configuration
  average, Gaussian nucleus; uncontracted per Dyall's archive (TZ s 1,3; p 1,2 / QZ s 1,2,4;
  p 1,2,4; counted without the diffuse functions, which stay uncontracted).  Names provisional.
- Contraction check (non-self-consistent): 2p splitting contracted/uncontracted 1.00020,
  0.99978, 0.99986, 0.99990; spinor energies within 8e-6 Eh.
- Predicted O2 ZFS (squared atomic ratio): TZ 2.43-2.44, QZ ~2.46 cm-1 (limit 2.462-2.465).
  Calculation requested from the zmr session (message sent 2026-10-08); NOT yet confirmed.
- Site rebuilt with O (first j-adapted entries).

## O2 check of the oxygen SOG sets by the zmr session (2026-10-08)

SA-3 CASSCF(2e,4), X2CMP(DCB), X(+-1)-X(0+) in cm-1 (logs ~/scr4/zmr_runs/o2_cbs/sog_*):

| set | spinors | CASSCF | PC-NEVPT2 |
|---|---|---|---|
| DYVTZ-SOG | 120 | 2.4209 | 1.600 |
| ADYVTZ-SOG | 184 | 2.4394 | 1.585 |
| DYVQZ-SOG | 220 | 2.4544 | 1.690 |
| ADYVQZ-SOG | 320 | 2.4602 | 1.693 |
| Dyall av3z uncontracted | 284 | 2.4527 | 1.668 |
| Dyall av4z uncontracted | 444 | 2.4621 | 1.708 |

- CASSCF: prediction (TZ 2.43-2.44, QZ ~2.46; limit 2.462-2.465) holds for aTZ and both QZ;
  DYVTZ is 0.01 below.  Contraction costs 0.013 (aTZ), 0.002 (aQZ).
- NEVPT2: contraction costs 0.083 (aTZ, 5 %) and 0.015 (aQZ, 1 %): the correlation part of the
  ZFS is sensitive to the TZ contraction.  NOT yet understood; first suspect is the choice of
  uncontracted p (Dyall's rule: the two outermost).  Variants generated for a follow-up:
  `basis/variants/` ADYVTZ-SOG-P23 (p 2,3 besides the diffuse one) and -P123 (three p).
- zmr limitation reported by that session: the spin-spin step (X2Ccorr) does not support a
  kappa-contracted basis (dimension mismatch), so no spin-spin values with these sets.
- Follow-up from zmr (same protocol; CASSCF / PC-NEVPT2, cm-1): ADYVTZ-SOG-P123 (5s5p3d2f, 196
  spinors) 2.4534 / 1.6545 = uncontracted av3z to +0.001 / -0.013 -> the NEVPT2 loss was in the
  p contraction.  Contracted aug-cc-pVTZ 2.2626 / 1.5281 (184 spinors), contracted aug-cc-pVQZ
  2.3987 / 1.6601 (320); uncontracted 2.307 / 1.569 and 2.397 / 1.661.  NEVPT2 - CASSCF: cc
  -0.735 (contracted or not), Dyall uncontracted -0.785 / -0.754, ADYVTZ-SOG -0.855 (outlier),
  P123 -0.799, QZ SOG -0.766.  Same frozen core (O 1s) and 100 Eh virtual cut everywhere.
- Conclusion: Dyall's TZ rule for the 2p block (uncontract the two outermost p) is not enough in
  the j-adapted contraction; three p give the uncontracted result.  Same-size alternative under
  test: p 1st and 3rd (variants/ADYVTZ-SOG-P13).  Entries in basis/GENBAS not changed yet.

## O2: same-size variants and the cc-based set (zmr, 2026-10-08)

CASSCF / PC-NEVPT2 ZFS (cm-1), all 184 spinors (size of contracted aug-cc-pVTZ):

| set | uncontracted p (besides diffuse) | CASSCF | NEVPT2 | E(CASSCF) |
|---|---|---|---|---|
| aug-cc-pVTZ contracted | 0.716, 0.214 | 2.2626 | 1.5281 | |
| ADYVTZ-SOG (Dyall rule) | 0.126, 0.332 | 2.4394 | 1.585 | -149.7534140 |
| ADYVTZ-SOG-P13 | 0.126, 0.837 | 2.5381 | 1.8301 | -149.7515388 |
| ADYVTZ-SOG-P23 | 0.332, 0.837 | 2.6647 | 1.8517 | -149.7521964 |
| APVTZ+P1-SOG (cc prims + p 225.713) | 0.716, 0.214 | 2.4525 | 1.6557 | -149.7452698 |
| (196 spinors) ADYVTZ-SOG-P123 | 0.126, 0.332, 0.837 | 2.4534 | 1.6545 | -149.7537324 |
| (284) Dyall av3z uncontracted | all | 2.4527 | 1.668 | -149.7540450 |

- P13 and P23 overshoot the limit (2.462-2.465): NOT usable.  The atomic contraction check had
  flagged exactly these two (2p splitting 1 % and 3 % off); my withdrawal of that check was wrong.
  The check (configuration-average contraction vs fractional-occupation Fock matrix) is a valid
  screen: ratio within ~2e-4 for the good sets, off for the bad ones.
- APVTZ+P1-SOG reproduces its uncontracted parent (2.4491 / 1.6653) to +0.003 / -0.010 at the
  size of aug-cc-pVTZ: keeping Dunning's own uncontracted and correlating functions and only
  improving the inner p region works.  Its total energy is 8 mEh above the Dyall-based sets
  (fewer s, p primitives in the SCF part).
- Published baseline: Zhang, Peterson, Dyall, Cheng, JCP 161, 054105 (2024) (cc-pVXZ-SO and
  dy-VXZ-SO, p block, DC); its Table I already shows larger contraction errors for dy-VTZ than
  for cc-pVTZ (NO -1.8 vs +0.2 cm-1).  Names here should follow that paper; to be agreed.

## O: one added tight p vs re-optimised inner p with the correlating p fixed (2026-10-08)

`curves/o_fix_corr.py` (Slurm 31781446), aug-cc-pVnZ, DCG, Gaussian nucleus, s dense for the
evaluation.  dE = CAHF energy error of the p set; SO = 2p splitting / dense reference.

| | TZ dE (mEh) | TZ SO | QZ dE (mEh) | QZ SO |
|---|---|---|---|---|
| cc p set | 4.21 | 0.9667 | 1.10 | 0.9859 |
| A: +1 p, on the splitting (225.7 / 418.9) | 2.71 | 0.9958 | 0.74 | 0.9984 |
| A: +1 p, on the energy (196.5 / 367.3) | 2.68 | 0.9954 | 0.73 | 0.9983 |
| B: uncontracted p fixed, inner n+1 p re-optimised on the energy | 1.57 | 0.9912 | 0.45 | 0.9963 |

B inner p: TZ 89.32, 20.75, 6.335, 2.161; QZ 164.4, 38.45, 11.99, 4.219.
- A: objective hardly matters.  The added tight p lowers the energy by 1.5 mEh (TZ), 0.36 mEh
  (QZ) -- NOT negligible (an earlier statement of mine said microhartree; wrong).
- B gains another 1.1 / 0.29 mEh but doubles the spin-orbit error (0.88 % vs 0.42 %; 0.37 % vs
  0.16 %), because its tightest p ends lower (89 vs 226).
- Also `curves/so_deficit.out`: deficit of the valence splitting with cc primitives, by block:
  2p 2.8-3.1 % (TZ), 1.1-1.4 % (QZ), 0.23-0.28 % (5Z); 3p 0.33-0.42 / 0.10-0.12 / 0.05-0.06 %;
  4p 0.11-0.15 / 0.03-0.04 / 0.02 %.
- Without an added primitive (`curves/o_fix_corr_same.py`, Slurm 31781569; inner p re-optimised,
  uncontracted p fixed): on the energy the cc exponents come back (TZ 34.96, 7.848, 2.303 vs
  34.46, 7.749, 2.280; dE 4.195 vs 4.211 mEh; SO 0.9674 vs 0.9667), so the cc set is already the
  optimum of its size also in DCG.  On the splitting: TZ 79.85, 12.84, 2.937 -> SO 0.9844 but
  dE 17.5 mEh; QZ 139.3, 22.48, 5.277 -> SO 0.9934, dE 3.72 mEh.  Same number of p cannot give
  both; one added p gives SO 0.9958 / 0.9984 AND a lower energy (2.71 / 0.74 mEh).

## Tight p for the 2p block; SCF level of Dyall's sets (2026-10-08)

- `opt/tight_p_2p.py` (Slurm 31781759): one p added to cc-pVnZ, optimised on the 2p splitting
  (DCG, Gaussian nucleus).  Exponents (D / T / Q / 5):
  B 40.37 / 81.59 / 143.98 / 436.50;  C 63.24 / 124.04 / 227.02 / 685.91;
  N 90.20 / 174.68 / 326.85 / 992.03;  O 117.31 / 225.27 / 419.85 / 1317.60;
  F 149.49 / 286.19 / 531.80 / 1630.59;  Ne 186.44 / 356.62 / 658.87 / 2009.91.
  Always 6.5-6.8 times the largest cc p exponent.  Splitting relative to the tight-saturated set:
  DZ 0.905-0.929 -> 0.992-0.995; TZ 0.961-0.970 -> 0.998; QZ 0.982-0.987 -> 0.999; 5Z 0.997 -> 0.9998.
- SCF error (DC, Gaussian nucleus, vs numerical) and valence np splitting relative to dyall.v4z:
  Kr: dyall v2z 47.5 mEh / -2.0 %, v3z 0.80 / -0.12 %, v4z 0.03; cc-pVDZ 134 / -1.7 %, cc-pVTZ
  10.5 / -0.41 %, cc-pVQZ 2.3 / -0.06 %.  Xe: dyall v2z 19.0 / -1.3 %, v3z 0.40 / -0.07 %, v4z
  0.03; cc-pVTZ-DK3 1.21 / -0.54 %, cc-pVQZ-DK3 0.24 / -0.01 %.  O: dyall v2z 3.0 mEh, v3z 0.16;
  cc-pVTZ 6.6, cc-pVQZ 1.5, cc-pV5Z 0.16.
  -> Dyall TZ primitives are QZ level everywhere; Dyall DZ primitives are TZ level only for the
  2p block (for Kr, Xe they are between cc DZ and TZ).

## X2Ccorr (spin-spin) with contracted bases: done by the zmr session (2026-10-09)

- socutils `tools/fcidump_rel.py` (`x2c_4c_spinors`, `x2c_active_eri_4c`), commit 4430bc8 on
  `prop-cfour-interface` in the main tree `~/apps/socutils-git` and 4decfb9 on `zmcscf-qn`; zmr
  master c598ab2.  Active spinors are expanded back in the uncontracted spinor basis
  (pyscf `_decontract_spinor`), then X, R of the run; integrals on xmol.  (Not on my branch
  `basis-tools`; that worktree is untouched.)
- O2 spin-spin part of the ZFS (cm-1): APVTZ+P1-SOG +1.548 (uncontracted parent +1.547),
  ADYVQZ-SOG +1.550 (uncontracted av4z +1.550), plain TZ +1.545.  Independent of the tight p and
  of the contraction, as expected for an r12 operator.  Totals (CASCI): APVTZ+P1-SOG 4.001,
  uncontracted parent 3.997, ADYVQZ-SOG 4.011, av4z 4.012.
- Cost: the step runs in the uncontracted spinor basis (5.5 min TZ, 70 min aQZ, 16 cores).

## The tight p of ccJ-pVnZ is as good as the one optimised here (2026-10-09)

- ccJ-pVnZ (cc-pVnZ + 2 tight s + 1 tight p, for NMR spin-spin coupling) adds a p at exactly
  6.0 x the largest cc p; ours is 6.5-6.8 x.  `opt/ccj_vs_opt.py` -> `ccj_vs_opt.out`, B-Ne,
  DZ-5Z, 2p splitting relative to the tight-saturated set: with the ccJ p 0.9914-0.9941 (DZ),
  0.9976-0.9979 (TZ), 0.9989-0.9992 (QZ), 0.9998 (5Z); with the optimised p larger by at most
  0.0007 (DZ), 0.0003 (TZ), 0.0001 (QZ, 5Z).
- Consequence: no exponent optimisation needed for the 2p block; take the tight p of ccJ-pVnZ
  (the tight s are not needed for the splitting).  Precedent to cite; the idea of a tight p for
  r^-3 operators is not new.

## 3d and 4f: the predicted deficit is NOT there (2026-10-09)

`curves/so_deficit_df.py` (Slurm 31824875) -> `curves/so_deficit_{3d,4d,4f,5f}.out`.  Deficit of
the valence d / f splitting with the primitives as they are vs tight end saturated (DC, Gaussian
nucleus, fractional occupation):
- 3d (Sc..Zn): cc-pVTZ-DK (8d) 0.65 % (Sc), 0.36 % (Ti), 0.10-0.23 % (Cr-Zn); cc-pVQZ-DK (11d)
  <= 0.05 %; cc-pV5Z-DK <= 0.02 %; dyall.v3z (9d) 0.04-0.16 %.
- 4d (Y..Pd): <= 0.01 %.   4f (Ce..Lu): 0.01-0.02 % (TZ), 0.00 % (QZ), both cc-pVnZ-DK3 and Dyall.
  5f: 0.01 %.
- So my prediction "the first shell of each l lacks a tight function" is wrong for d and f.  The
  deficit is specific to p, i.e. to the 2p block (2.8-3.1 % at TZ).  Likely reason (not tested):
  near the nucleus R ~ r^l, so the r^-3 weight of the spin-orbit operator reaches the cusp region
  only for p; and the 3d / 4f sets already carry 8-11 primitives for one shell.
- Caveat: the absolute splittings of Sc, Ce, Nd differ between basis sets (e.g. Ce 2177 vs 4601
  cm-1 for cc-pVTZ-DK3 vs QZ): the fractional-occupation SCF lands on different configurations.
  The deficits are ratios within one set, but those rows should not be quoted as absolute numbers.
- Decision (user): the 2p block uses the tight p optimised here; ccJ-pVnZ is cited as precedent.

## Framing (user, 2026-10-09)

The concern is that Dyall's sets are not balanced; the goal is a new cc-pVXZ family for
relativistic, in particular spin-orbit, calculations: cc-pVXZ correlating functions as the
blueprint, spin-orbit contraction, primitives balanced per zeta level (2p block: cc + one tight
p, exponent optimised here).
- First-shell test (`contract/first_shell.py`, Slurm 31825060, dyall.v3z, only the valence l
  contracted, k outermost primitives uncontracted, recovered valence MP2 energy for k = 0..4):
  Ne 2p 0.42 0.65 0.91 0.982 0.998; Ar 3p 0.70 0.88 0.979 0.995 0.999; Kr 4p 0.73 0.90 0.984
  0.993 0.9995; Zn 3d 0.49 0.57 0.82 0.968 0.997; Cd 4d 0.75 0.82 0.94 0.990 0.998.  (Yb, No: see
  the .out files.)  The first shell of an l needs one more uncontracted function on Dyall's dense
  primitives.
- cc-pVTZ does not have the problem: Ne valence MP2 (mEh) cc-pVTZ contracted -264.8 (30 fns),
  uncontracted -272.1, Dyall TZ uncontracted -274.7, Dyall rule contracted -246.6 (30 fns), with
  three p -265.7 (33 fns); Ar -206.0 / -209.4 / -209.9 / -204.7.
- Acceptance test: contraction loss of the valence correlation energy below the contribution of
  the last correlating shell of that zeta (about 15 mEh at TZ, 3.7 mEh at QZ for O).

## F 2P splitting, EOM-IP-CCSD on F- (socutils ZCCSD, X2CAMF, 1s frozen) (2026-10-09)

`f_eomip/f_eomip.py` (Slurm 31825905), aug-cc-pVTZ-based sets; experiment 404.1 cm-1.

| F basis (TZ) | spinors | splitting (cm-1) | E(corr) (Eh) |
|---|---|---|---|
| cc uncontracted | 116 | 404.56 | -0.29404 |
| cc + tight p uncontracted | 122 | 416.41 | -0.29410 |
| cc as published | 92 | 402.99 | -0.28878 |
| cc, SF contraction | 92 | 404.17 | -0.28866 |
| cc, SO contraction | 92 | 404.04 | -0.28867 |
| cc + p, SF contraction | 92 | 415.98 | -0.28870 |
| cc + p, SO contraction | 92 | 415.87 | -0.28871 |
| Dyall av3z uncontracted | 142 | 417.77 | -0.29549 |
| Dyall, SO contraction, Dyall rule | 92 | 413.81 | -0.27829 |

- Tight p: +11.8 cm-1 (+2.9 %), the atomic deficit.  SF vs SO contraction: 0.1 cm-1.
  Contraction of cc + p: -0.5 cm-1.  Dyall rule: -4.0 cm-1 and 17 mEh of correlation energy.
- With the basis converged the method gives 416-418 cm-1, 3 % above experiment; plain cc-pVTZ
  hits 404 by cancellation of the basis deficit against the method error.
- Hamiltonian (cc + p uncontracted): DC only 461.2, DC + Gaunt 416.4, + gauge 416.4 cm-1.
- QZ results: `f_eomip/f_eomip_q.out` (running when this was written).

## F 2P splitting, multireference (zmr) vs EOM-IP-CCSD (2026-10-09)

cm-1, experiment 404.1; X2CAMF Gaunt+Breit unless noted; CASSCF = SA-6 CASSCF(5e,6) = CAHF + CASCI.

| F basis | CASSCF | NEVPT2 (1s frozen) | NEVPT2 (all) | EOM-IP-CCSD from F- |
|---|---|---|---|---|
| aug-cc-pVTZ uncontracted | 384.6 | 391.3 | 392.3 | 404.6 |
| + tight p, uncontracted | 395.7 | 402.7 | 403.8 | 416.4 |
| + tight p, SO contraction (TZ size) | 395.7 | 401.7 | 402.1 | 415.9 |
| + tight p, Dirac-Coulomb only | 438.4 | 446.1 | 447.3 | 461.2 |

- Tight p: +11.1 (CASSCF) / +11.8 (EOM).  Gaunt+Breit: -42.6 / -44.8.  Contraction: -0.1 CASSCF,
  -1 to -2 NEVPT2.  X2CMP agrees with X2CAMF to 0.1.
- With the tight p NEVPT2 is at experiment (402-404); EOM-IP-CCSD from the anion is 12.6 higher.
  QZ EOM: cc 413.5, cc + p 418.6 (uncontracted); contracted SF/SO 413.8/413.3 and 418.8/418.4.
- Delta-CCSD on two separately optimised hole determinants (`f_eomip/f_delta.py`) is NOT usable:
  aug-cc-pVDZ gives dHF 878, dCCSD 410, dCCSD(T) 602 cm-1, one SCF unconverged.
- Open (zmr): X2Ccorr (--corr) changes the F splitting by -15.6 (DCB) / -25.9 (DC), basis
  independent; not spin-spin for a doublet; possibly a double counting of the 2p-2p two-electron
  spin-orbit term already in the mean field.  Not added; the user has to decide.

## F: atomic mean field vs molecular mean field vs four-component, EOM-IP-CCSD from F- (2026-10-09)

Uncontracted cc-pVTZ (NOT augmented: with aug-cc-pVTZ the four-component SCF of F- does not
converge with scf.dhf.UDHF, E = -74.5; the user found the same and dropped the diffuse functions).
`f_eomip/noaug_t.out`, splitting in cm-1:

| basis | Hamiltonian | X2CAMF | X2Cmmf (F- mean field) | four-component |
|---|---|---|---|---|
| cc-pVTZ unc | DC | 431.75 | 431.17 | 429.30 |
| cc-pVTZ unc | DC + Gaunt | 388.89 | 389.42 | 387.31 |
| cc-pVTZ unc + tight p | DC | (invalid) | 442.82 | 440.95 |
| cc-pVTZ unc + tight p | DC + Gaunt | (invalid) | 400.77 | 398.67 |

- X2CAMF = X2Cmmf to 0.6 cm-1 and = four-component to 2 cm-1: for the IP from F- the atomic
  mean field is NOT off by the -15.6 cm-1 that X2Ccorr gives for the CAS treatment of neutral F.
  Unresolved tension; to be looked at by the user / zmr.
- Tight p: +11.4 cm-1 also at the four-component level.
- The two "invalid" X2CAMF rows (41 and 115 cm-1): my input appended the tight p shell after the
  d and f shells; socutils X2CAMF needs the shells grouped by l.  `input_f_minus_4c.py` fixed
  (sorts the shells).  The earlier X2CAMF tables used sorted primitives and are not affected.
- Non-augmented sets are poor for F- (388.9 vs 404.6 with aug, same Hamiltonian): these numbers
  compare Hamiltonians, not with experiment.
- Cause of the failed four-component SCF with diffuse functions: PySCF's DHF occupies
  mo[n2c:n2c+nelec]; with aug sets the linear-dependence removal leaves fewer than n2c
  negative-energy states (140 of 142 for dyall-av3z), so the 1s spinors stay empty (E = -74.5).
  `f_eomip/f_eomip_mmf.py` now occupies the lowest states above -c^2 and freezes by energy.
  With that: Dyall av3z uncontracted, DC + Gaunt, X2Cmmf EOM-IP-CCSD 418.27 cm-1 (X2CAMF with
  Gaunt+Breit in the same basis: 417.77).  More in `f_eomip/f_eomip_mmf_t.out` (Slurm 31827xxx).

## F 2P splitting with full triples (zccsdt session, 2026-10-09)

EOM-IP from F-, X2CAMF Gaunt+Breit, 1s frozen, basis variants/f_apvtz_p1_sog.py (92 spinors):
EOM-IP-CCSD 415.865 cm-1 (independent code; socutils gave 415.9), EOM-IP-CCSDT (1h+2h1p+3h2p on
CCSDT amplitudes) 393.445 cm-1: triples -22.4 cm-1.  Experiment 404.1.  Script
/home/xwang405/.claude/jobs/7ba4fcf7/tmp/fso/f_ip_ccsdt.py, zccsd master 8a27656.
- Implementation check: Li/6-31G EOM-IP-CCSDT = FCI differences to 1e-8.
- Its caveat: the EOM-IP series from the anion oscillates with the excitation level (absolute
  IP of F-, nonrelativistic aug-cc-pVDZ: EOM-IP-CCSD 3.175, EOM-IP-CCSDT 3.140, Delta-CCSDT
  3.230 eV), so the splitting is not converged to better than ~10 cm-1 at triples; CCSD +11.8
  and CCSDT -10.7 bracket experiment; consistent with NEVPT2 403.8.
- Summary of the F study: the basis effects are the same in every method (tight p +11 to +12
  cm-1, contraction < 2 cm-1, SF vs SO contraction ~0.1); the absolute value depends on the
  method at the +-3 % level (CASSCF 395.7, NEVPT2 403.8, EOM-IP-CCSD 416, EOM-IP-CCSDT 393).
- zccsdt follow-up (uncontracted sets), cm-1: aug-cc-pVTZ unc CCSD 404.563 / CCSDT 382.784;
  + tight p unc 416.407 / 393.984; + tight p contracted 415.865 / 393.445.  Triples shift -21.8
  to -22.4 in every basis; tight p +11.84 (CCSD) and +11.20 (CCSDT); contraction -0.54 at both
  levels.  My socutils CCSD numbers reproduced to 0.05 cm-1 by the independent code.

## X2Ccorr double counting found and fixed in the zmr driver (2026-10-09)

- Cause (found by reading the driver): `--corr` replaced the active (tu|vw) by four-component
  integrals but kept the X2C one-electron Hamiltonian, whose mean-field two-electron spin-orbit
  term contains the active electrons; socutils `from_x2c` has the missing half (`fourC_fcore`:
  four-component hcore + mean field of the core only).  Confirmed and fixed by the zmr session
  (zmr master 1f65954).
- F 2P splitting, shift (a) old -> (b) fixed, cm-1: X2CAMF(DCB) -15.59 -> -2.80; X2CAMF(DC)
  -25.86 -> -4.77; X2CMP(DCB) -15.61 -> -3.01.  With (b): CASSCF 392.9, PC-NEVPT2 401.0
  (experiment 404.1).
- O atom (TZ + 2 tight p) with (b): NEVPT2 157.9 / 227.2 vs experiment 158 / 227 (3P1, 3P0);
  S atom 398.7 / 580.2 vs 396 / 574; O2 spin-spin +1.547 -> +1.538 (unaffected).

## Spin-orbit information lines of the cfour GENBAS -SO entries (request of the zccsdt session, 2026-10-10)

`so_info/so_info.py` (read only) -> `so_info/so_info.out`, `so_info/proposed_lines.tsv`.  Lines
derived from the coefficients (columns k and k+n of an l are the two kappas of one shell);
reproduces 431 of the 434 existing lines.
- 52 entries with an empty line: actinides (Th, Pa, U, Np, Pu, all DY* variants) "3  1 6  2 4
  3 2"; Ir, Pt, Au, Hg, Yb, Nd "3  1 5  2 3  3 1"; Rh "2  1 4  2 2"; Cu PVTZ-SO / -SOF
  "2  1 2  2 1" (extra contracted p and d functions of the Peterson set are shared, not j pairs).
- The three lines that exist but are wrong: BR:PVTZ-SO1P / SO2P / SO3P carry "2  1 3  2 1";
  the coefficients give "1  1 1", "1  1 2", "1  1 3" (d is spin-free in all three).
- Side finding: in many heavy-element entries two contracted functions of the same kappa are
  nearly parallel (overlap 0.9-1.0: actinide d and f, lanthanide d, Hg/Ir QZ p, also BR:PVQZ-SO
  and I:PVQZ-SO), i.e. near linear dependence inside the entry.  Not investigated.
- Regeneration check (`so_info/regen.py`, `regen_check.py` -> `regen_check.out`; run in the
  tunnel allocation with `srun --jobid=<tunnel> --overlap`): for TH:DYVTZ-SO, YB:DYPVTZ-SO,
  HG:DYPVTZ-SO every library column equals my atomic X2CAMF spinor of that kappa (overlap 1.0000)
  once the rows of the uncontracted primitives are zeroed in mine.  So (a) the proposed lines are
  right, (b) where n exceeds the occupied shells the extra column is the lowest virtual spinor
  (Th 7p, 5f; Yb, Hg 6p), (c) the library zeroes the contracted coefficients on the uncontracted
  primitives; that explains the "nearly parallel" columns -- NOT a linear dependence, my earlier
  side finding is withdrawn.  To reproduce such entries, `contract.contract` needs
  zero_uncontracted=True and an option for n above the occupied shells (not implemented).
- Since 2026-10-10 the sessions run on login01: small calculations go through the tunnel job
  (`squeue -u xwang405 -n tunnel` for the id), larger ones through sbatch.

## 2026-10-13: near linear dependence in the cfour -SO entries; corrected entries; O core sets

- My statement of 10 Oct ("only a representation, no linear dependence") was too weak.  The
  zeroing is harmless, but the SPACE of many entries contains a nearly redundant direction: the
  outermost contracted shell lies almost entirely in the span of the uncontracted primitives
  (Th/U 6d: independent norm 0.001-0.004; smallest overlap eigenvalue 5e-5 to 4e-7).  zccsdt saw
  the consequence in ThF+ (forbidden <vv||vv> elements 5e-7).  `so_info/lindep.py` ->
  `lindep.out`: 116 of 1874 (entry, kappa) sets below 1e-4 for l > 0, 21 elements; exact
  dependence in Tl-Rn (A)DYCVQZ-SO (duplicated p primitive).
- User's decision (relayed by zccsdt): correct the entries.  `so_info/fix_lindep.py` ->
  `so_info/GENBAS.fixed` (76 entries, name = original + "-LD", information line filled in),
  `so_info/pyscf_fixed/*.py`, report `so_info/fix_report.txt`.  Rule: per l (s included) while
  the smallest eigenvalue is below 1e-4, remove a duplicated shared primitive, otherwise the
  outermost contracted shell in both kappas, at most one per l.  The cfour library is untouched.
- Atomic check of the four priority entries (`check_fixed_4.out`): cost 1e-5 to 7e-5 Eh in the
  6d spinor energy at TZ, 1e-6 at QZ; removing the 7s contraction at QZ costs 1e-4 to 2e-4 Eh
  for a modest gain (3e-5 -> 1e-4): d-only variant offered.  Remaining 72: Slurm 31941181 ->
  `check_fixed_all.out` (pending when written).
- New-family consequence: generation must test the overlap eigenvalues of (contracted + shared)
  per kappa and drop a contracted shell that is redundant.
- Oxygen (`o_core/make_o_core.py`, Slurm 31940911): `basis/pyscf/dyvdz_sog.py` (O:DYVDZ-SOG,
  3s2p1d), `basis/pyscf/dycvtz_sog.py` (O:DYCVTZ-SOG, 6s5p3d1f = DYVTZ-SOG + s 6.717, 15.18,
  p 14.38, 45.53 uncontracted + d 10.238 of cc-pwCVTZ), variants and the uncontracted
  aug-cc-pVTZ + tight p (+ core d) files in `basis/variants/`.  1s correlation recovered (UMP2,
  vs uncontracted cc-pwCV5Z, O2): DYVTZ-SOG 12 %, DYCVTZ-SOG 85 %, Dyall-rule variant 82 %,
  uncontracted aug-cc-pVTZ + p 81 %, + d 89 %, aug-cc-pwCVTZ 86 %, cc-pwCVQZ 97 %.  dyall.cv3z
  = v3z + one d (38.204) for O.  Sent to the dice and zmr sessions.
- 2026-10-13, later: `so_info/GENBAS.so_original` = the 486 -SO entries of the cfour library,
  verbatim.  `thuof/GENBAS.thuof` (+ `thuof/pyscf/`), draft: O, F PVnZ-TP-SOG / APVnZ-TP-SOG
  (cc primitives + one tight p, D/T/Q) and Th, U DYVnZ-SOG (Dyall primitives, occupied shells
  only, redundant 6d contraction dropped at TZ/QZ by the eigenvalue test, threshold 1e-5);
  report `thuof/make_thuof.out`.  Names and threshold provisional.  Generation of further
  corrected library entries stopped at the user's request (to be decided in this session).
- Peterson's actinide sets (Feng & Peterson 2017, read): Dyall inner primitives, outer ones
  re-optimised, ANO contractions for the s,p,d,f correlating functions (3 per l at TZ, the third
  added against BSSE), most diffuse primitive uncontracted.  His outer functions span about the
  6 outermost primitives (to 2.47 for Th d), Dyall's rule frees 4 (to 0.63).  User's idea under
  discussion: re-optimise the outer primitives to a sparser set (6 -> 5 for Th d) and leave
  them all uncontracted.  No optimisation started.
- SCF level of heavy-atom primitives (`thuof/scf_level.out`, DC, Gaussian nucleus, relative to
  dyall.v4z, mEh): Th dyall v2z 11.8, v3z 0.71; cc-pVDZ/TZ/QZ-DK3 12.0 / 0.68 / 0.00.  U 15.7,
  1.02; 13.8 / 0.78 / -0.03.  Rn dyall 21.8, 0.70; cc-pVTZ/QZ-DK3 1.59 / 1.16 (point-nucleus
  optimised, not a clean comparison).  Peterson's re-optimised outer exponents do not change
  the SCF energy for Th, U.

## 2026-10-13/14  Rn pilot: how many outer primitives (preliminary, scalar only)

Directory `rn_pilot/` (`baseline.py`, `scan.py`, `ref2.py`; outputs `baseline.out`, `scan_*.out`, `ref2.out`).
Rn, dyall.v3z (30s26p17d11f), sf-X2C-1e RHF + MP2 of 6s6p (39 frozen), Gaussian nucleus.
Uncontracted: E(HF) -23556.385954, E2(6s6p) -131.62 mEh.

Design under test: inner Dyall primitives, inner occupied shells contracted, the outer K0
primitives (s 6, p 6, d 5) all uncontracted, no contracted function for 6s / 6p / 5d;
f = 4f + most diffuse f.  Then the K0 outer exponents replaced by n re-optimised ones
(objective E(HF) + E2(6s6p), BFGS in log exponents, one l at a time).

Before re-optimisation (Dyall exponents): Dyall rule 68 functions, E2 -125.24 (95.1 %);
new construction for s, p, d together 90 functions, E(HF) +3.88 mEh, E2 -129.31 (98.2 %);
s only +0.004 mEh, p only +0.006 mEh, d only +3.87 mEh (5d needs more than the 5 outer d).

Re-optimised, change against "same construction, Dyall exponents" (mEh; HF / E2):
s6 +0.01/-0.02; s5 +0.03/+0.02; s4 +0.27/+0.34; s3 +4.4/+1.2;
p5 +1.2/+0.30; p4 +7.9/+1.7; p3 +99/+8.3 (p6 still running when written);
d5 -3.0/-0.09 (largest d 3.98 -> 4.31; leaves +0.87 mEh in total); d4 -0.9/+2.4; d3 +13.6/+30.
Exponents in `scan_*.out`.

Scale (`ref2.out`): E2(6s6p) cc-pVTZ-DK3 -126.71, cc-pVQZ-DK3 -146.21 (step 19.5 mEh);
ANO-R1/R2/R3 -79.9/-114.3/-132.3.  E(HF) of the uncontracted primitives above dyall.v3z:
cc-pVTZ-DK3 0.66 mEh, cc-pVQZ-DK3 0.48, ANO-R 11.7.  (Contracted cc-pVnZ-DK3 HF energies
are not comparable: DK3 contraction used with X2C.)

Reading: the SCF energy, not the correlation energy, limits the count.  s 4, p 5-6, d 5 keeps
each loss near or below 1 mEh; that is [9s9p7d2f] = 85 functions against 68 for Peterson's TZ
and Dyall's rule.  Open (user): SCF tolerance; then joint s,p,d optimisation and a spinor check.
Not done: j-adapted contraction of the inner shells, f and higher l, 5d correlation.

Side note: ANO-R (Zobel, Widmark, Veryazov, JCTC 16, 278 (2020)) has its own exponents made
for X2C, shared by R0-R3; Rn 25s22p14d9f6g4h, outer region sparser than dyall.v3z (s 5, p 6,
d 4 below ~4).  ANO-RCC exponents are family-type (d from s, f from p), none equal to Dyall's.

SCF tolerance (user, 2026-10-14): take the SCF error of Dunning's cc-pVnZ in the
non-relativistic setting.  Measured (`rn_pilot/cc_scf_error.out`, RHF above the numerical
HF limit, mEh, D/T/Q/5): He 6.5/0.53/0.17/0.06; Ne 58/15.2/3.6/0.33; Ar 17.6/4.4/0.73/0.17;
Kr 80/2.9/0.26/0.06 (uncontracted; contracted identical except Kr DZ 80.1, 5Z 0.20).
TZ level for heavier atoms: about 3-4 mEh.  Joint optimisations submitted: s4 p5 d5
(`scan_all455.out`) and s4 p4 d5 (`scan_all445.out`).

SCF tolerance, decided with the user 2026-10-14: the total-energy error of cc-pVnZ is
dominated by the core (and uneven: Ne TZ 15 mEh, Kr DZ 78 mEh from 5 d primitives for 3d), so
the ruler is the outer-region error (`rn_pilot/cc_scf_outer.out`: exponents >= 4 saturated):
D/T/Q/5 in mEh  Ne 22/4.8/1.2/0.12; Ar 4.6/1.44/0.24/0.13; Kr 9.2/1.03/0.11/(0.10, at the
noise of the saturation).  def2 (TURBOMOLE library page, not the paper): atomization energy
per atom ~3 kJ/mol (1.1 mEh) "quantitative", ~1 kJ/mol (0.4 mEh) "basis set limit".
User: calibrate to the same value and, this being a set for correlated calculations, a little
stricter is fine.  Working target: outer-region SCF error <= ~1 mEh at TZ, <= ~0.2 mEh at QZ
(atomic, during optimisation); validation later by atomization energies per atom (def2's
definition).  p6 re-optimised in place: HF +0.02, E2 -0.06 mEh (exponents barely move).
Nuclear model (`nuc_check.out`): cc-pVnZ-DK3 contraction error 2.5 Eh (point) / 3.2-3.6 Eh
(Gaussian) with sf-X2C, i.e. DKH3 coefficients do not fit X2C; ANO-R2 0.26 mEh with Gaussian
nucleus (2 Eh with point).  Joint runs: s4p5d5, s4p4d5, s4p6d5.

Tolerances fixed by the user (2026-10-14), outer-region SCF error of the atom:
DZ 3 mEh, TZ 1 mEh, QZ 0.3 mEh, 5Z 0.1 mEh.
Open: the user gave the numbers only; "outer-region error" is my reading and the user found
it awkward.  Quantity the tolerance refers to is not settled.

2026-10-14, rulers for a total-SCF tolerance (user now wants ALL exponents re-optimised and the
tolerance on the total atomic SCF error; numbers under discussion: 3/1/0.3/0.1 or 10/3/1).
`rn_pilot/def2_light.out` (non-relativistic, contracted, above numerical HF, mEh):
def2-SVP/TZVP/QZVP  He 6.5/1.8/0.06  Be 19/0.44/0.02  Ne 171/5.6/0.25  Mg 80/8.0/0.15
Ar 194/14.8/0.32  Ca 97/12.4/0.28  Zn 287/41/0.95  Kr 385/52/0.68;
cc-pVDZ/TZ/QZ/5Z  He 6.5/0.53/0.17/0.06  Be 0.69/0.15/0.06/0.01  Ne 58/15.2/3.6/0.33
Mg 6.3/1.3/0.40/0.03  Ar 17.6/4.4/0.73/0.17  Ca 8.4/0.25/0.04/0.01  Zn 1.5/1.5/0.09/-
Kr 80/2.9/0.26/0.20.
`rn_pilot/def2_scf.out` (sf-X2C, Gaussian nucleus, above dyall.v4z uncontracted, mEh):
dyall v2z/v3z  Kr 45.3/0.65  Xe 14.6/0.26  Rn 14.7/0.55;
x2c-SVPall/TZVPall/QZVPall contracted  Kr 587/95/1.1  Xe 902/102/4.0  Rn 1761/336/10.1
(segmented sets: only the contracted numbers mean anything; -2c versions within 10 %).
Reading: Karlsruhe sets are balanced in atomization energies, not in atomic total energies;
only their QZ level is held near 1 mEh for light atoms.  My suggestion to the user: DZ 10,
TZ 1-3, QZ 0.3, 5Z 0.1 (total error), TZ value after the Rn count-vs-error curve.  Not decided.
Joint Rn s4 p4 d5 (outer-only design): E(HF) +9.0 mEh above dyall.v3z uncontracted, E2 -127.46.

2026-10-14 (later).  Literature read with the user's PDFs (memory notes written):
Peterson 2003 (JCP 119, 11099) gives the criteria for matching HF sets to zeta levels: HF error
below correlation error; outer HF primitives usable as correlating functions; priority on fast
HF convergence; DZ has 2 exponents in the outermost region of the valence s, p orbitals, TZ 3,
..., the outermost exponent decreasing level by level.  Balabanov-Peterson 2005: achieved HF
errors TZ 0.3-1.8 mEh, QZ 0.03-0.1, 5Z 0.004-0.045; spd correlating functions as ANOs.
Consequence agreed in discussion: no per-level SCF tolerance is fixed (3/1/0.3/0.1 and 10/3/1
were discussed and dropped as design inputs); SCF error is reported as a result.  Dyall's TZ/QZ
SCF errors are at the level of Peterson's modern sets; the "too good" impression came from
comparing with Dunning's first-row sets.

`outer_count/outer_count.out`: primitives whose radial-density maximum lies beyond the outermost
node of the valence orbital (D/T/Q/5):
  cc-pVnZ   Ar s 2/3/3/4, p 3/4/4/5;  Kr s 2/3/3/4, p 3/3/4/5
  dyall     Kr s 2/3/4, p 3/4/5;  Xe s 2/3/4, p 2/3/5;  Rn s 2/3/4, p 2/3/5, d(5d) 4/5/7
  Rn cc-pVTZ-DK3 s 3, p 6, d 6; cc-pVQZ-DK3 s 3, p 4, d 6; ANO-R s 3, p 4, d 5
  Ne: 2p has no node (all 4/5/6/8 p primitives describe the one lobe) - the first-shell case.
Reading: Dyall's sets obey the "n exponents in the outermost valence region" rule; the six
"outer" primitives of the Rn scan included three that describe inner lobes / the n-1 shell.
The Rn scan above (outer all free, counts reduced) is therefore not the route; kept as data.

`marginal/`: value of the last HF primitive against the last correlating function in cc-pVnZ
(Ne, Ar done; Kr running): Ar p 0.22/0.29/0.09 of the d/f/g function (D/T/Q); Ne p 2.4/1.7/1.0
(2p under-described at every level); s always small.

Plan under discussion (not started): Peterson's four steps on Rn TZ - optimal valence and
core-valence (5s5p5d) correlating exponents on the uncontracted set, compare with Dyall's outer
primitives, pin them and relax the neighbouring HF exponents only where they do not match;
V and CV sets to share one primitive set.

ANO correlating functions, scalar test (`ano_test/`, preliminary; in-sample: the natural
orbitals come from the same atomic MP2 they are tested on; no molecule yet, not j-adapted).
dyall.v3z primitives, occupied AOs + n most occupied virtual MP2 natural orbitals per l.
Recovered fraction of the uncontracted MP2 energy at equal size:
  Ne 2s2p (30 functions): 2 primitives 0.893, ANO 2s2p 0.979, ANO 1s1p + diffuse 0.946;
     1 function each (26): 0.622 vs 0.914; 3 each (34): 0.975 vs 0.9965.
  Ar 3s3p (34 functions): 0.965 vs 0.990; 1 each 0.870 vs 0.953; 3 each 0.992 vs 0.998.
  Rn 6s6p (68 functions): Dyall rule 0.9515, ANO 2s2p2d1f 0.9690, ANO 1s1p1d + diffuse 0.954;
     84 functions: primitives 0.984, ANO 3s3p3d2f 0.995.
  Rn 6s6p5d (68 functions): Dyall rule 0.378, ANO 2s2p2d1f 0.8625; 84 functions 0.782 vs 0.964.
Smallest overlap eigenvalue: ANO sets 0.4-1.0; with a diffuse primitive added 1e-3..2e-2;
uncontracted-primitive sets down to 1e-5.  E(HF) unchanged in all (all occupied AOs kept).

Molecular flexibility test (`mol_flex/`, scalar, preliminary; ANOs from atomic UMP2 densities
averaged over m, not a proper state average).  X in dyall.v3z, occupied AOs contracted, two
correlating functions per occupied l; dE(HF) in mEh / recovered valence MP2, against X
uncontracted; last number = MP2 error of the reaction energy (mEh).
  F  (F2 | HF | F2+H2->2HF):   prim +0.91/0.919 | +0.11/0.924 | -1.83;  opt +0.93/0.955 | +0.46/0.954 | +1.36;
     ano +2.27/0.982 | +0.78/0.980 | +1.12;  ano-1+ +0.52/0.957 | +0.27/0.960 | -0.64;  ano+ (+4 fn/atom) +0.20/0.990 | +0.15/0.991 | -0.12
  N  (N2 | NH3 | N2+3H2->2NH3): prim +0.95/0.944 | +0.19/0.966 | -5.52;  opt +1.99/0.964 | +1.13/0.974 | -1.00;
     ano +3.07/0.985 | +0.99/0.987 | -0.66;  ano-1+ +2.05/0.967 | +0.34/0.977 | -3.03;  ano+ +1.83/0.992 | +0.23/0.994 | -1.54
  I  (I2 | HI | I2+H2->2HI):   prim +0.33/0.973 | +0.08/0.977 | -0.49;  opt +0.34/0.976 | +0.14/0.979 | -0.32;
     ano +0.85/0.985 | +0.21/0.987 | -0.76;  ano-1+ +0.29/0.977 | +0.07/0.980 | -0.37;  ano+ (+9 fn/atom) +0.05/0.997 | +0.02/0.997 | -0.01
Optimised correlating exponents (HF most diffuse in brackets): F s 0.239 0.547 (0.237 0.605),
p 0.262 0.876 (0.158 0.422); N s 0.133 0.301 (0.142 0.361), p 0.173 0.556 (0.098 0.252);
I s 0.088 0.190 (0.089 0.200), p 0.087 0.231 (0.072 0.184), d 0.191 0.472 (0.186 0.449).
Reading: pure ANOs lose SCF flexibility in molecules (2-3 mEh for 2p, 0.8 for I2) though they
recover most correlation; for the ordinary shell (I) all schemes agree within ~1 %; for 2p the
optimal correlating p is about twice as tight as the diffuse HF p.
User (2026-10-14): proper ANOs would come from atomic CASSCF + multi-state NEVPT2 densities
averaged over states; with spin-orbit these split into j = l-1/2 and l+1/2, nearly equal for
the halogens -> redundant; correlating functions are to be shared by both kappa.

User (2026-10-14): simply adding the optimised correlating exponents to the primitive set makes
the uncontracted set, in which the X2C transformation is done, nearly linearly dependent.
Checked (analytic primitive overlap, dyall.v3z + the added exponents): smallest neighbouring
ratio drops from 1.8-2.5 to 1.01-1.2; smallest overlap eigenvalue N s 1.7e-3 -> 8.9e-6,
F s 1.7e-3 -> 3.0e-7, I s 2.3e-5 -> 4.9e-8, I d 1.3e-3 -> 4.8e-6.  Not acceptable; my
contracted-basis eigenvalues had missed this.  Route now: Peterson's device (pins inserted
into the HF set, neighbouring HF exponents re-optimised, ratio >= 1.6): `mol_flex/pin_test.py`,
jobs pin_N, pin_F, pin_I.

Pin-and-reoptimise test (`mol_flex/pin_N.out`, `pin_F.out`; I running; scalar, preliminary).
N: s outer 0.1423 0.3605 0.9027 -> 0.1332 0.3006 (pins) 0.7649; p outer 0.0978 0.2524 0.629 1.551
   -> 0.0352 0.1732 0.5559 (pins) 1.554.  Atomic E(UHF) +0.64 mEh; ratios >= 2.2; primitive
   overlap eigenvalue s 1.9e-3, p 3.3e-2 (as dyall).  Molecules, against dyall uncontracted:
   new uncontracted N2 +1.02 mEh; new with pins free (60 fn) N2 +2.36 mEh / 0.965, NH3 +1.57 /
   0.975, reaction error HF +0.78, MP2 -0.29 mEh (Dyall, 2 most diffuse: +0.95 / 0.944,
   +0.19 / 0.966, -0.57, -5.52).
F: s 0.2367 0.6048 1.512 -> 0.2391 0.5465 (pins) 1.351; p 0.1581 0.4219 1.072 2.664 ->
   0.0587 0.2621 0.8764 (pins) 2.542.  Atomic E(UHF) +1.62 mEh.  F2 +4.12 mEh / 0.958, HF +1.89 /
   0.957, reaction error HF -0.33, MP2 +1.13 (Dyall rule +0.91 / 0.919, +0.11 / 0.924, -0.69, -1.83).
Note: the left-over most diffuse p runs out to 0.035 / 0.059 (the atomic SCF has no use for it);
it should be dropped or placed by another criterion (8p -> 7p).

Pin variants compared (`mol_flex/pin_*.out`, `pindrop_*`, `pinfix_*`, `pinfill_*`; N and F; I only
differs marginally and has no HF exponent outside the pins).
  count kept, diffuse p free: runs out to 0.035 / 0.059; atom +0.64 / +1.62 mEh; reaction errors
      (HF/MP2) N +0.78/-0.29, F -0.33/+1.13.
  diffuse p dropped (7p): reaction errors N +3.01/+2.41, F +2.52/+3.88 - NOT acceptable: the diffuse
      p is needed in NH3 and HF although the neutral atom has no use for it.
  diffuse p fixed at Dyall's: atom +0.81 / +2.27; reaction errors N +1.21/+0.41, F +1.45/+2.86;
      outer p of F 0.158 0.262 0.876 2.54 has gaps of a factor 3 (cc-pVTZ-like), hence the SCF cost.
  gaps filled (diffuse end as Dyall, pins fixed, free exponents wherever a ratio of 1.7 leaves room,
      re-optimised with ratio >= 1.6):  s 15 (Dyall 14), p 10 (8) primitives; atomic SCF -0.02 /
      -0.01 mEh against Dyall; smallest ratio 1.6; primitive overlap eigenvalue 4e-4..7e-4.
      N: s outer 0.1332* 0.3006* 0.5891 1.182; p 0.0978 0.1732* 0.2932 0.5559* 1.124 1.808
      F: s outer 0.2391* 0.5465* 1.042 1.957; p 0.1581 0.2621* 0.4193 0.8764* 1.68 3.027   (* pinned)
      TZ contracted, pins free: N2 +1.87 mEh / 0.965, NH3 +1.09 / 0.974, reaction +0.30/-0.91;
      F2 +0.85 / 0.955, HF +0.49 / 0.955, reaction +0.14/+1.52  (Dyall rule: N -0.57/-5.52, F -0.69/-1.83).
  I, count kept: atom +0.19 mEh; I2 +0.51 / 0.976, HI +0.26 / 0.980, reaction +0.02/-0.29
      (Dyall rule +0.33 / 0.973, +0.08 / 0.977, -0.18/-0.49).
Working recipe (scalar prototype): optimal correlating exponents pinned as members of the
primitive set; diffuse end kept; gaps filled and re-optimised under ratio >= 1.6; occupied AOs
contracted; pins uncontracted.  Validation must include polar molecules.
To do: move to the X2CAMF optimiser (analytic gradient, DCG), j-adapted inner contractions,
core-valence pins, d/f valence shells (Zn, Yb, Th).

2026-10-14, Th/U direction.  User: O, F = cc-pVnZ + tight p as already in `thuof/GENBAS.thuof`
(sent to the zccsdt session as a draft).  Th, U correlating functions: UHF/UMP2 rejected by the
user (no such state), and a fractional-occupation second-order energy on the CAHF orbitals
(`thuof2/corr_exp.py`, jobs ce_*) is at most a starting guess - several open shells need a
multireference treatment.
zmr session (reply, nothing run): NEVPT2 has no density; MRCI has root-averaged virtual
natural orbitals (`natural_virtuals`, --natvirt), virtual-virtual block only.  U as 5f3 6d1 7s2
with 5f+6d+7s active (26 spinors) is not feasible.  Feasible: one-shell active spaces - U3+ 5f3
(14 spinors) or Th 6d2 7s2 with 6d+7s (12 spinors); NEVPT2 energy uncontracted (~800 spinors)
hours; MRCI natural virtuals only in a contracted/truncated space of 200-300 spinors; in a
contracted set of ~140 scalar functions one CAHF + PC-NEVPT2 energy takes minutes to tens of
minutes.  d/f-block atoms need an entry in the zmr driver's atom table.
zmr O2 check of the O core sets (CASSCF | NEVPT2 fc | ae): DYVTZ-SOG 2.4209 | 1.6000 | 1.5987;
DYCVTZ-SOG 2.4236 | 1.6348 | 1.6513; unc aug-cc-pVTZ + tight p 2.4584 | 1.6716 | 1.6809;
DYVDZ-SOG file identical to zmr's (5e-9), CASSCF 2.3064.  Logs ~/scr4/zmr_runs/o2_cbs/sog_dycvtz*.
Cheap stand-in check (`thuof2/corr_*_3.out`, CAHF orbitals + fractional-occupation E2): optimal
correlating p of N 0.115 0.381, F 0.198 0.690, against 0.173 0.556 / 0.262 0.876 from UMP2 (and
cc-pVTZ 0.1725 0.555 / 0.267 0.913); I agrees with UMP2.  So the stand-in is off by a factor
~1.4 already for a single open 2p shell: not usable for Th/U, as the user said.
Rn valence (6s6p) optimised correlating primitives, `ano_test/opt_Rn_39_*`: 68 functions 0.9585
(HF primitives 0.9515, ANO 0.9690); 84 functions 1.010 (> 1: the added exponents extend the set).
Open: zmr Th basis test, request in requests/zmr_th_basis_test.md (zmr session not reachable on 2026-10-14).

Th TZ test by the zmr session (results: requests/zmr_th_basis_test_results.md; one CAHF averaged
over (6d 7s)^4 for both configurations, X2CMP DCG, 6s6p6d7s correlated, PC-NEVPT2):
draft TH:DYVTZ-SOG recovers 97.95 % (6d2 7s2) / 97.88 % (6d3 7s1) of the uncontracted dyall.v3z
correlation energy; excitation draft minus uncontracted +2 cm-1 (CASCI), +26 cm-1 (NEVPT2 configuration
average, 9592.5 vs 9566.5), +9 cm-1 (lowest levels).  cc-pVTZ-X2C: 101.4 % of the correlation
energy (its total energy and CASCI excitation not comparable: scalar contraction).  Spinors 342 /
734 / 322; smallest overlap eigenvalue 4.8e-4 / 3.2e-6 / 3.3e-3.
Reading: Dyall's primitives with Dyall's uncontraction (6d contraction dropped) are adequate for Th
valence correlation at TZ; no new correlating exponents needed there.  Open: Th4+ KRHF draft 20 mEh
above uncontracted (contraction from the neutral atom); U (5f), DZ/QZ.

2026-10-14/15, U for the electron-affinity work of the zccsdt session (user's request via zccsdt):
- thuof/GENBAS.thuof_aug: U:ADYV{D,T,Q}Z-SOG = DYVnZ-SOG + one even-tempered diffuse primitive per l
  (a1^2/a2), uncontracted, shared.  Overlap eigenvalue barely changes.  Answer to zccsdt in
  ~/.claude/jobs/7ba4fcf7/tmp/thbas/q_diffuse_answer.md (session unreachable).
- Core-valence (5s5p5d): CFOUR's U:DYPCVnZ-SO only add g (TZ) / f+h (QZ), no s, p, d.  Two versions:
  v1 GENBAS.thuof_cv (Dyall "valence + 5d"), v2 GENBAS.thuof_cv2 (Peterson cc-pwCVnZ-X2C positions on
  Dyall primitives; his s, p and DZ/QZ d ARE Dyall primitives; tight f -> nearest Dyall f; his extra
  g/h added).  U6+ scalar MP2 (5s5p5d6s6p) recovered: DZ 51 / 74 / 86 %, TZ 90.5 / 97.4 / 98.75 %
  (valence / v1 / v2); CFOUR-like (+g) TZ 93.4 %.  QZ v1 98.1 %; QZ v2 rerun with DF-MP2 (job u_cv2q).
- Atomic SCF check of the EA states, socutils SymmSpinorSCF(symmetry='sph') with the cross-block part
  of the two-electron potential masked (the (l,j,m_j) blocking is a constraint for an M_J = 6
  determinant; without the mask |g| stays at 0.23 and the SCF never "converges").  thuof/ea_scf/ea_sph.py,
  results ea_sph_{D,T}.res.  Contraction error of EA_SCF (contracted - uncontracted, same primitives):
  DZ  aug: 7p -64 meV, 6d -0.4 meV;  no aug: 7p -903 meV, 6d -8 meV
  TZ  aug: 7p -9.8 meV, 6d -0.1 meV; no aug: 7p -184 meV, 6d -1.2 meV
  U6+ contraction error DZ 18 mEh, TZ 4.3 mEh.  Reason for 7p: no contracted 7p function (7p empty in
  the neutral atom).  Fix option: contracted 7p (per kappa) from a 7p-occupied configuration, as
  Peterson does for his actinide sets.  Not done.
- Th TZ test (zmr NEVPT2): DYVTZ-SOG adequate (97.9 % evenly).

2026-10-15, U for the EA work (continued):
- -P1 (user's decision): uncontract the most diffuse p primitive that Dyall's valence rule leaves in
  the contraction (DZ 0.0240, TZ 0.0156, QZ 0.0127).  SCF-EA contraction error, aug sets, 7p state:
  TZ -9.8 -> +0.07 meV, QZ -1.8 -> -0.03 meV; 6d state TZ -0.1, QZ +0.3 meV (unaffected).
  Contracted 7p per kappa (rejected by the user) kept in thuof/rejected_7p/.  More/other diffuse p
  does not help (TZ: Lan's 0.00521 -23.8 meV, second diffuse p -11.6 meV).
- Core-valence v3 = union of v1 (Dyall) and v2 (Peterson positions): U6+ MP2 recovered DZ 91.6,
  TZ 98.8, QZ 99.55 % (v2 had an f hole between 0.67 and 5.7 at QZ).  GENBAS.thuof_cv3.
- Files for zccsdt: thuof/GENBAS.thuof_ea (-P1, valence + CV v3), GENBAS.thuof_ea6d (+ g/h at the
  6d radius, diagnostic of Dyall's design only).  Note: ~/.claude/jobs/7ba4fcf7/tmp/thbas/q_basis_package.md
  (zccsdt session unreachable).
- Atomic SCF method (thuof/ea_scf/): (l,j,m_j)-constrained, cross-block 2e potential masked;
  QZ U- 6d needs a start from the projected TZ solution + maximum overlap (from the neutral density
  it oscillates on a spurious solution ~15 mEh above TZ).  Unconstrained gradient of the constrained
  U solution: 79 % Delta l = 2 rotations of the outer core (5d->g, 5p->f, 6s->d, 6p->f), i.e.
  quadrupole polarisation by the 5f3 shell; j-mixing in the open shell only 3.5 %.
- Discussion (user, Lan): the 6d state converges slowly because a d electron sees the set at its
  nominal zeta level while the 7p electron sees a much higher one; adding an angular layer to one
  level is not acceptable (breaks the hierarchy); the 7p state needs no more work.
- Y:DYVDZ-SF (spin-free, dyall.v2z, Dyall 4d DZ rule): y_sf/GENBAS.y, y_sf/dyvdz_sf.py.
- User's plan (2026-10-15): stopgap first - zccsdt uses -P1-6D (TZ/QZ) and -P1 (DZ) for a few EA
  calculations; the proper redesign of the U (actinide) high-l correlating functions comes after.
  Evidence that Dyall's U design is unbalanced: no g/h at the 6d radius (TZ g 0.175 -> 0.824,
  h 0.265 -> 1.54; QZ h 0.193 -> 0.834) while Th has them; most diffuse p locked; QZ core f hole.
  Redesign idea (not started): cc layer counts, no separate polarizing functions (diffuse -> aug),
  exponents optimised on correlation of 5f (U3+ 5f3) and 6d (Th 6d2 7s2, same 6d radius) with zmr
  NEVPT2, then validated on the U EA by zccsdt.
- 2026-10-17: zccsdt reports the user's decision of 10-16 that Lan runs the U- EA calculations
  (zccsdt will not); it has recorded the package note for reference.  The -6D CCSD(T) test is
  therefore not scheduled anywhere.  Eu: eu/GENBAS.eu (EU:DYVDZ-SF, EU:DYVDZ-SOG), eu/*.py.
- 2026-10-19, request from o2 zfs run coordination (user: "ccpvxz-x2c ... 2p 上加 so 重要的 function"):
  basis/GENBAS.o_x2c + basis/pyscf/{pv,apv}{d,t,q,5}z_{x2c_tp,tp_sog}.py.  (aug-)cc-pVXZ primitives
  and pattern, one tight p (D 117.31, T 225.27, Q 419.85, 5 1317.60) inside the contracted 2p;
  -X2C-TP spin-free X2C-1e, -TP-SOG j-adapted X2CAMF+Gaunt; Gaussian nucleus; sizes = cc-pVXZ.
  Contraction errors <= 4e-5 Eh (DZ) .. 5e-8 (5Z); 2p splitting ratio 0.994 (DZ) .. 1.000 (Q, 5).
  Correction: the O/F -TP-SOG entries of 2026-10-13 also used the Gaussian nucleus (my note to
  zccsdt then said point nucleus - wrong).
- 2026-10-19 (later): user chose the ccJ-pVXZ tight p (D 106.2, T 206.76, Q 380.52, 5 1173.0) for the
  O sets of the O2 ZFS; basis/GENBAS.o_x2c and the pyscf modules regenerated in place; the version
  with our optimised exponents kept in basis/superseded/o_x2c_opt_tightp_20261019/.  o2 coordination told.
- 2026-10-19: user: for the O2 ZFS series use the scalar sets only (O:(A)PVXZ-X2C-TP); reason given:
  with a j-adapted contraction the outermost shell easily becomes nearly linearly dependent and
  correlating it gains little.  o2 coordination told.
- 2026-10-20, o2 coordination (user: "就是在 QZ 上做 core valence 呗"): basis/GENBAS.o_x2c_cv,
  pyscf/pwcv{t,q}z_x2c_tp.py.  PVXZ-X2C-TP contraction (cc-pVXZ + ccJ tight p) + Dunning's
  core-valence functions uncontracted; 43 / 84 functions per O; smallest overlap eigenvalue TZ
  7.1e-3, QZ 3.5e-5 (s 71.21/69.32 as in cc-pwCVQZ).  Full recontraction in the pwCVQZ primitives
  fails (atomic SCF does not converge).  socutils basis-tools 73a87f6: add_primitives checks only
  the added exponents for closeness.
- 2026-11-05: user: U is dropped ("U 我就不管了").  No further U work (redesign of the actinide
  high-l correlating functions, -6D test); the delivered U files stay as they are.

## 2026-11-05 socutils basis library extraction (staging, not installed)
- `socutils_lib/`: inventory.py, dupcheck.py, extract.py, validate.py. Source: ~/apps/cfour-x2cso GENBAS (read-only), families -SO/-SF/-X2C/_X2C/-DE4.
- 121 staging modules / 1496 entries in `socutils_lib/modules/`, written via genbas.to_pyscf (kappa format for -SO, scalar otherwise).
- validate.py: all 121 register() and every element builds a PySCF molecule (0 failures).
- report.txt: 44 empty -SO info lines filled from so_info/proposed_lines.tsv, 22 malformed entries skipped, 87 near-dependent entries (flagged, kept).
- Awaiting user OK before copying into ~/apps/socutils-git-basis/gto/basis/ and committing locally.

## 2026-11-07 Ca/N/H aug-cc-pVXZ pattern check (side task)
- `canh/pattern.py` (per-l exponents, ratios, overlap eigenvalues, from BSE), `canh/mol_ovlp.py` (linear CaNH overlap).
- Ca: no non-relativistic aug-cc-pVXZ anywhere; aug only as -X2C (Hill-Peterson), T/Q only; no 5Z of any X2C kind.
  Non-rel Ca TZ/QZ s have a ~4x hole (0.36->0.081, 0.33->0.082); 5Z and -X2C sets are regular (~2).
  BSE Ca cc-pCV5Z = cc-pV5Z primitives recontracted (no added core functions); cc-pCVTZ/QZ, cc-pwCVTZ-X2C add exponents at ratio 1.02-1.15 (min eig to 6e-9).
- CaNH molecular S min eig: TZ 2e-4, QZ 2e-5, 5Z 4e-6 (contracted and uncontracted similar). H/N aug X2C/DE4 exist in staging modules.

## 2026-11-07 Ca/N/H X2C sets, aug-cc-pV{T,Q,5}Z, sf + sog (canh_x2c/)
User decisions: Ca 5Z = cc-pVQZ-X2C s/p + Koput-Peterson cc-pV5Z d/f/g/h; Ca d fully uncontracted; CV wanted.
My choice (to report): Ca 4p contraction of the source (unoccupied) replaced by one more free p in the 4p region.
- GENBAS.canh: H/N APV{T,Q,5}Z-{SF,SOG} (Dunning prims, recontracted), CA APV{T,Q,5}Z-{SF,SOG}. Gaussian nucleus, AOC.
- Ca SF-X2C HF (uncontracted, vs QZ-X2C ~ limit): TZ +0.094 mEh, QZ/5Z 0. Contraction errors ~1e-10 (Ca), 1e-7 (N, AOC vs FON check).
- pyscf/apv{t,q,5}z_x2c_{sf,sog}.py. CaNH S min eig (spinor) 1.6e-4 / 1.8e-5 / 4.0e-6. sog sets only usable as spinor (kappa) basis.
- Atomic SCF of diffuse Ca sets needs max_iter up to 1000 (slow, not wrong).
- CV (build_ca_cv.py, pins = Hill-Peterson pwCV-X2C added exponents): TZ drop+window4 gives HF +0.155 mEh vs valence prims; testing keep-HF / wider window.
- CV result (DROP=1, WIN=10, ratio>=1.6 penalty, Nelder-Mead): GENBAS.canh_cv CA:APWCV{T,Q}Z-{SF,SOG}.
  TZ: HF +0.129 mEh vs valence prims (total ~0.22 vs limit) - acceptable for TZ. 9s8p7d3f.
  QZ: HF +0.211 mEh vs valence prims - too large for QZ (target 0.03-0.1); s min ratio 1.49, d 1.50; s 0.1127 moved to 0.195. NOT final.
  Keep-HF (DROP=0) variant was worse (penalty-driven, 0.40-0.48 mEh at TZ). CV PySCF modules not written yet. 5Z CV exponents need a correlated optimisation - to discuss.
- Node c498 interact job ended 2026-11-08 ~07:13.

## 2026-11-10 Y TZ request from Eu:YSO session (y_sf/)
- Their Y:DYVDZ-SF files are byte-identical to mine (y_sf/make_y.py, 2026-10-15).
- Plan sent (Dyall 4d_tz rule 1: s 1,3; p 2,3; d 1,2,3; options A 5p contraction, B f/g 3f2g vs 4f2g).
- polar.py (finite-field RHF sfx2c1e, Y+ and Y3+): current DZ Y+ alpha -11% contraction error (no free diffuse p);
  DZ with most diffuse p free (dz_p1.py, test only): +0.3%; Y3+ -2.7 / -3.7 %. Option A3 (free diffuse p) proposed. Waiting for reply.
- Done (A3/B1): GENBAS.y Y:DYVDZ-SF-P1 (6s5p4d2f, 55 fn), Y:DYVTZ-SF-P1 (7s6p5d3f2g, 89 fn); dyvdz_sf_p1.py, dyvtz_sf_p1.py; make_y_p1.py.
  alpha contraction error Y+ DZ +0.28 % TZ -0.26 %; Y3+ DZ -3.7 % TZ -1.8 %. RHF E contraction error Y+ 1.8/1.0 mEh, Y3+ 3.3/1.9 mEh.
- Eu:YSO cell: P1 diffuse functions linearly dependent; user (via Eu:YSO) chose cutting exponents < 0.05.
  Y:DYVTZ-SF-T05 (make_y_t05.py): 26s18p13d2f2g -> 7s5p5d2f2g, 79 fn. vs full unc. dyall.v3z: Y3+ RHF +2.27 mEh, alpha +1.6 %;
  Y+ alpha -84 % (5s cut), neutral Y AOC energy +56 mEh, all orbital energies up ~0.065 Eh (5s compressed). Fine only for Y3+.
- Y:DYVTZ-SF-T07 (make_y_t07.py, cut < 0.07): 26s18p12d2f1g -> 7s5p5d2f1g, 70 fn; Y3+ RHF +2.43 mEh, alpha +1.6 %; Y+ alpha -84 %. Sent to Eu:YSO 2026-11-11.
- 2026-11-11: read Hill-Peterson 2017 (memory hill-peterson-2017-s-block-sets): Ca 4p contraction from 3P state, d from 3D; wCV by CCSD on closed-shell Ca ground state.

## 2026-11-16 Ln3+ sets for Eu:YSO (lnbas/) and Y:DYVDZ-SOG
- make_ln.py: dyall.v3z, drop < 0.07, Ln3+ 4f^n configuration average contraction, Dyall 4f_tz combined uncontraction; KIND=sf|sog.
  GENBAS.ln: {CE,PR,ND,SM,EU,GD,TB,DY,HO,ER,TM,YB}:DYVTZ-{SOG,SF}-T07; <el>_dyvtz_{sog,sf}_t07.py; <el>_v3z_unc.py. SOG 9s12p9d6f3g2h (181), SF 9s8p7d5f3g2h (152).
  Pr/Yb: truncation effect on E, zeta_4f, <r-3>_4f, F2/4/6 below 1e-5 relative. check_ln.py per ion running -> check_<El>.out.
- y_sf/make_y_sog.py: Y:DYVDZ-SOG (6s7p6d2f), same recipe as DYVDZ-SF; contraction errors 1e-5 valence, SO splittings ratio 1.0000-1.0007.
- check_<El>.out done for all 12: truncation E diff <= 0.006 mEh, zeta/<r-3>/F2 ratios 1.00000+-2e-5, contraction errors (Ln3+ AOC Fock) <= 6e-10 Eh. Polarizability not computed (open shell).
