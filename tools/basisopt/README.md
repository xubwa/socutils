# socutils.tools.basisopt: basis-set tools driven by the X2CAMF atomic SCF

Exponent optimisation, relativistic recontraction and GENBAS I/O on top of the bundled
pure-C X2CAMF atomic program (`lib/libx2camf_c.so`, entry point `x2camf_atom_scf`).

    from socutils.tools.basisopt import engine, optimize, contract, genbas

The engine loads the bundled C library directly, whatever `x2camf` package is on the path.

## engine

    prims = engine.prims_from_pyscf('O', 'aug-cc-pvtz')        # {l: [exponents]}
    r = engine.atom('O', prims, 'dcg', gau_nuc=True)            # 4c Dirac-Coulomb-Gaunt
    r.e_scf; r.so_splitting(2, 1); r.eps(2, 1, 3); r.r_power(2, 1, 3, -3); r.occupied()

* Hamiltonians: `dc`, `dcg`, `dcb` (4c), `sfdc`, `sfx2c1e`, `x2c1e`, and `x2camf`,
  `x2camf-g`, `x2camf-b` (X2C-1e HF with the X2CAMF spin-orbit integrals of a 4c DHF).
  DCG and DCB give the same basis-set requirements; DCG costs half of DCB.
* Open shells: default = fractional occupation of the l shell (meaningful spinor energies;
  the total energy contains the self-interaction of the fractional occupations).
  `aoc=True` = configuration average (degeneracy-weighted average over the states of the
  nonrelativistic configuration, as in Dyall's basis-set optimisations): proper energy,
  orbital energies not meaningful.  Use the default for spin-orbit splittings, `aoc=True`
  for energies and contraction coefficients.  `occ=[...]` sets the electrons per kappa.
* Nucleus: point (default), `gau_nuc=True` (Visscher-Dyall Gaussian, analytic integrals),
  or `nucleus=('fermi', c, a) | ('gaussian', xi) | ('sphere', R)` in bohr, integrated
  numerically; `engine.fermi_from_isodata(path)` reads the Fermi parameters of a GRASP run.
* Pitfalls handled here: the primitives must be grouped by l; the library reads
  occupations from a file (default `input` in the working directory in the molecular
  entry points) -- the engine always passes its own.
* Dense sets (even-tempered ratio <= 1.8) reach a density convergence of about 1e-9 only:
  use `conv=1e-8`.  An unconverged SCF returns `e_scf = nan`.

### Exponent gradients

    e, g = engine.exponent_gradient('Hg', prims, gau_nuc=True)     # {l: [dE/d ln(alpha)]}

Gradient of the 4c Dirac-Coulomb SCF energy (fractional occupations) with respect to the
logarithm of every exponent, from one SCF plus six integral evaluations in which only one
primitive position carries a displaced exponent (`x2camf_atom_exp_grad`; the energy is
stationary in the orbitals, so only the density and the energy-weighted density are
needed).  Xe dyall.v2z, 47 exponents: 2.6 s, equal to 7e-8 Eh to a PySCF-integral
implementation of the same formula; Hg dyall.v4z, 96 exponents: 37 s.  Not available with
the Gaunt term or for the configuration-average SCF (DC-optimal exponents are also optimal
with Gaunt to 0.1 % in the exponent, Xe-Og).

Use: a set optimised for the DC energy has |g| < 1e-3 Eh for its SCF exponents (the one or
two outermost functions of an l are often correlating functions and not at the SCF
optimum).  The erroneous f exponents of Hg dyall.*4z (corrected by K. Dyall, Dirac Users
list, 27 Feb) show up as |g| up to 0.22 Eh; after the correction 9e-4.

## optimize

    python -m socutils.tools.basisopt.optimize O aug-cc-pvtz --add p:1 [--ham dcg]
           [--objective so|energy] [--targets valence|all|2p,3p] [--weights 1,4]

adds primitives to the (uncontracted) named set and optimises their exponents.  Objective
`so`: sum_i w_i (D_i/D_i^ref - 1)^2 with D = eps(n l j=l+1/2) - eps(n l j=l-1/2) and the
reference the same set with the tight region of the optimised l saturated.

## contract, genbas

    ref = genbas.read(GENBAS, 'XE:PWCVQZ-SO')
    pattern = contract.pattern_from(ref)              # {l: (n contracted, [uncontracted])}
    prims = {l: list(e) for l, (e, c) in ref.shells.items()}
    new = contract.contract('Xe', prims, pattern, kind='so', name='PWCVQZ-SO')
    contract.compare(new, ref); genbas.append('GENBAS', new)

Command line (appends an entry to a GENBAS file):

    python -m socutils.tools.basisopt.contract O --basis aug-cc-pvtz --add p:225.713 \
           --kind sog --name APVTZ+P1-SOG --out GENBAS --check
    python -m socutils.tools.basisopt.contract Xe --genbas GENBAS --entry XE:PWCVQZ-SO \
           --kind so --name PWCVQZ-SO-NEW --out GENBAS.new --compare

The primitives and the contraction pattern come from a PySCF basis-set name (`--basis`) or
an existing GENBAS entry (`--genbas/--entry`); `--add` merges new primitives, which enter
the contracted functions only (`--uncontract-added` keeps them as primitives too).
`--check` prints the contraction error of the occupied spinor energies and of the
spin-orbit splittings (uncontracted X2CAMF-Gaunt Fock matrix diagonalised in the
contracted space).  For O aug-cc-pVTZ + p(225.713) the 2p splitting keeps 0.99944 (SF
contraction), 0.99970 (SF, tight p also uncontracted) and 0.99991 (j-adapted) of the
uncontracted value.  Only occupied atomic orbitals can be contracted: a pattern asking for
more contracted functions than occupied shells is refused.

Recipe: atomic X2C-1e HF in the primitives, occupied atomic orbitals = contracted functions,
chosen primitives appended uncontracted.  Default `kind='sog'` (use it for DCG and DCB:
rare gases Ne-Og with dyall-v3z, SOG vs SOB contractions differ by < 1.2e-6 in the valence
splitting, SO vs SOG by up to 5e-4 and by mEh in the spinor energies of heavy atoms).
`kind='sf'`: SF-X2C-1e; `'so'`, `'sog'`, `'sob'`:
X2C-1e + X2CAMF spin-orbit integrals of a 4c DC / DCG / DCB DHF, p1/2-type contractions
first, and the GENBAS comment line "nl  l n ..." read by
`socutils.tools.basis_parser.parse_genbas(..., so_basis=True)`.

## Validation (2026-10)

* C backend against the C++ implementation: all Hamiltonians, open-shell variants and
  nuclear models, |dE| < 3e-11 Eh (O), < 2e-8 Eh (Xe); matrices to 1e-8 relative.
* Ne, 4c DC, point nucleus, same primitives: PySCF DHF to 1e-9 Eh.
* Numerical Dirac-Fock, Xe, DC, even-tempered s/p/d sets of ratio 2.0 / 1.9 / 1.8 / 1.7:
  Fermi nucleus against GRASP2018 (-7446.89852155): +5.5e-4 / +8.1e-5 / +9.0e-6 / +5.5e-7;
  Gaussian nucleus against Dyall, TCA 99, 366 (1998) (-7446.895440): +5.5e-4 / +8.2e-5 /
  +9.5e-6 Eh; Kr (-2788.860624): +7.9e-5 / +1.0e-5 / +1.4e-6.  (Done with the C++
  implementation; the C backend reproduces it to the figures above.)
* O: the 2p spinor-energy splitting against the basis reproduces the ratios of the CASSCF
  3P fine structure to 5e-4 and predicts the O2 zero-field splitting to 0.2 %.
* Contraction: XE:PWCVQZ-SO of the CFOUR GENBAS reproduced to 5e-9.
