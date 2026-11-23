# Basis-set optimisation driver around the X2CAMF atomic program

You are an independent session started for the user (xwang405) by their `zmr` session
("zmr multi-reference perturbation"). Work autonomously; the user may attach and steer.
Chat with the user in Chinese; code, comments and documents in English.

## What the user asked for (2026-10-07, their words, translated)
"Basis optimisation can use the X2CAMF program, the atomic program ... its atomic
calculations are more efficient than ours, so make it a driver, and have a basis-optimisation
program drive it.  And once we know how to use this thing, it is also convenient for
generating the contracted basis sets I need."  So two deliverables:
1. An optimiser of basis-set exponents (to begin with: added tight functions) whose engine
   is the X2CAMF atomic four-component SCF.
2. A documented, scripted way to generate relativistic contracted basis sets (the
   `-X2C` / `-SO` recontractions in the user's GENBAS) with the same program.

## Why: what was found on O2 (zmr session, run directory ~/scr4/zmr_runs/o2_cbs)
Zero-field splitting of O2 X 3Sigma_g^- (X2CMP with Dirac-Coulomb-Breit, CAS(2e,4 spinors),
SA-3 CASSCF, cm-1, without spin-spin): uncontracted aug-cc-pVTZ 2.307, QZ 2.397, 5Z 2.452,
6Z 2.462; aug-cc-pVTZ with the s and p primitives of aug-cc-pV7Z 2.465; aug-cc-pVTZ plus
1 / 2 / 3 tight p primitives (exponents continuing the series with its last ratio 4.447:
153, 682, 3031) 2.436 / 2.457 / 2.458.  The spin-spin contribution (+1.55) and the
correlation contribution (MRCI - CASSCF) do not depend on the basis.  So the whole basis-set
effect is the spin-orbit matrix element, i.e. the O 2p spinors near the nucleus (r^-3
weight), which energy-optimised sets do not describe: the largest p exponent of O is 34
(aug-cc-pVTZ), 63 (QZ), 196 (5Z), 526 (6Z), 1349 (7Z); dyall v3z has 194, v4z 557.
The user does NOT want to switch to Dyall sets in general: they found errors in them (Hg:
TZ -> QZ could not be extrapolated; one set of exponents of the QZ set was wrong; with the
TZ exponents in its place it extrapolates).  In PySCF's dyallv4z for Hg the g exponents are
3.1505 1.9906 0.78325 0.3082 (ratios 1.58, 2.54, 2.54), which looks suspicious, but which
set the user means is not known: ask them.  Correlation-consistent sets are preferred
"because they are used more and their problems have been fixed"; what they lack are the
tight functions for the spin-orbit part.

## What was found in the code (read by the zmr session; verify before relying on it)
- ~/apps/x2camf (user's repository, master 42be969; other branches origin/dev, eamf,
  x2camf4cfour) and ~/apps/x2camf_newbuild.  Python module `x2camf` (pybind,
  x2camf/libx2camf.*.so, built for python 3.11): `libx2camf.amfi`, `atm_integrals`,
  `pcc_K`, each (input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
  speed_of_light[, spin_free]).  `input_string` is an integer of flags (see
  x2camf/x2camf.py: with_gaunt bit 0, with_gauge 1, gaussian_nuclear 2, aoc 3, pt 4, pcc 5,
  int4c 6, with_gaunt_sd 7).  According to a comment in socutils
  (~/apps/socutils-git/somf/x2cmp.py, around line 549) atm_integrals returns
  {0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 6 fock_2c_2e,
  7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c}.  No total energy or
  orbital energies are returned directly; they can be had from these matrices (fock and
  overlap -> spinor energies, e.g. eps(2p3/2) - eps(2p1/2); density, h1e and fock ->
  energy) or by exposing more through pybind.
- PITFALL: `shell` must list the primitives sorted by angular momentum (socutils computes
  nshell = shell[-1] + 1).  Tight primitives appended after the highest l gave total
  energies 0.3 - 12 Eh too HIGH in the molecular calculation; sorted, everything is fine.
- executables/basisGenerator.cpp (in ~/apps/x2camf) and
  ~/apps/cfour-x2cso/x2camf/basis_interface.cpp (built as ~/apps/cfour-x2cso/bin/xx2cbasis,
  reads ZMAT with %x2cbasis and %basisinfo): atomic 4c or spin-free SCF, then contraction
  coefficients for a given pattern, written in GENBAS format (spin-free, X2CAMF or AMFX2C
  variants; also MINAO).  `input` in ~/apps/x2camf is an example for U (atom, basis name in
  GENBAS, flags, pattern per l, %occAMFI_<EL> occupations).  This is presumably how
  PVTZ-X2C / DYPVTZ-SO in the GENBAS files were made; confirm with the user.
- The spin-orbit information of the `-SO` sets is the line after the title in GENBAS (e.g.
  `3 1 5 2 3 3 1` = for l = 1, 2, 3 the number of contracted functions specific to each
  kappa); parsed by socutils.tools.basis_parser.parse_genbas(..., so_basis=True).
  ~/apps/cfour-x2cso/basis/GENBAS has that line EMPTY for ND:DYPVTZ-SO, while
  ~/scr4/zmcscf/Eu/GENBAS has it; the GENBAS files differ between directories.
- There are entries TH:ETB1, TH:ETB2, FR:ETB0, EU:ETB0 in ~/apps/cfour-x2cso/basis/GENBAS and
  a backup GENBAS.bak_before_ETB0_20260925: someone worked on even-tempered sets recently;
  the script that made them was not found.
- No exponent optimisation code was found anywhere (X2CAMF branches, the CFOUR tree).

## Suggested plan (your call where the code tells you otherwise)
1. Engine: a Python function atom(element, primitives by l, hamiltonian flags) -> spinor
   energies (hence the spin-orbit splittings of the valence and core p, d, f shells) and
   the SCF energy, from the X2CAMF atomic program; validate against an independent
   four-component atomic calculation (PySCF DHF on the atom, or published Dirac-Fock
   numbers) and against saturated basis sets.
2. Target: the user's need is the spin-orbit part.  Candidate objectives: the error of
   eps(np3/2) - eps(np1/2) (and of <r^-3>-type quantities) relative to a saturated
   reference, with as few added functions as possible; energy alone is insensitive
   (the first tight p of O lowers the energy by 3 mEh and raises the O2 splitting by 5.6 %).
   Decide how many tight functions per l and their exponents (not necessarily on the
   geometric continuation), for O first, then S, Se; think about what the heavy elements
   need (tight p, d).
3. Check transferability against the molecular numbers above (the zmr driver
   ~/apps/zmr/examples/05-chalcogen_zfs.py has `--basis`, `--tight p:2`, `--sp-from`; an
   atomic series for O is in ~/scr4/zmr_runs/o2_cbs/atom_*: CASSCF fine structure of O 2p^4).
4. Contracted sets: reproduce one existing entry of the GENBAS (e.g. O:PVTZ-X2C or
   ND:DYPVTZ-SO) with basisGenerator / xx2cbasis to learn the recipe, document it, then
   generate the contraction for the tight-function-augmented set.
5. Optional screening tool for existing sets (exponent-ratio smoothness; TZ/QZ
   extrapolation of the atomic energy), motivated by the Hg case.

## Hard rules
- ~/apps/x2camf is the user's repository: do not edit, commit, stash or check out anything
  in its working tree.  Work in a NEW worktree: `git -C ~/apps/x2camf worktree add
  ~/apps/x2camf-basis -b basis-opt` (from master) and keep everything there; nothing
  pushed.  If the library must be rebuilt, build it in the worktree (compute node, see
  below), never over the installed one.
- Read only: ~/apps/socutils-git, ~/apps/zmr, ~/apps/zmr-dev, ~/apps/cfour-x2cso (including
  its GENBAS: write new basis sets to a GENBAS of your own under your run directory),
  everything under ~/scr4/zmcscf, ~/scr4/zmr_runs/o2_cbs.
- Do not install anything into the shared base conda environment.
  Environment: `source ~/.conda_activate`; socutils via
  PYTHONPATH=$HOME/apps/socutils-git-env; zmr via PYTHONPATH=$HOME/apps/zmr.
- Calculations through Slurm, right-sized; atomic SCF runs of seconds may run inline.
  Binaries in /cm/shared/apps/slurm/current/bin/.  express (-p express -A lcheng24, <= 4
  CPUs, 16 GB), shared (-A lcheng24, <= 32 CPUs, 4 GB per CPU), bigmem
  (-A lcheng24_bigmem) only when memory requires it.  The login node has no AVX-512,
  compile on a compute node.  Run directory: ~/scr4/zmr_runs/basis_opt/.
- Never cancel or touch jobs you did not submit; never `pkill -f` a pattern that could
  match your own shell; wait for jobs with background commands (match full job names with
  `squeue -o "%j"`, the default output truncates names to 8 characters); no scancel in
  wait loops.
- Discuss design choices that are the user's to make (objective function, how many
  functions, naming of new sets) before building a lot on them; mechanical work just do.

## Reporting
Keep ~/scr4/zmr_runs/basis_opt/STATUS.md current.  The zmr session can be reached with
cross-session messages (name "zmr multi-reference perturbation"); send it the optimised
exponents when you have them so that it can rerun the molecules.  Say clearly what is
validated and what is preliminary.
