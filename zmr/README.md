# zmr

Relativistic multireference methods for spinor (and spin-orbital) Hamiltonians,
aimed at problems such as lanthanide 4f levels and the zero-field splitting of
O2.

Roadmap:

1. Fully uncontracted, multi-state NEVPT2 (UC-NEVPT2) with the Dyall
   Hamiltonian and a Laplace-transformed treatment of the external indices.
2. Partially contracted NEVPT2 (PC-NEVPT2) from a Wick-theorem engine.
3. Internally contracted MRCI.

The working equations are written for complex spin-orbitals / spinors
(`docs/derivations/conventions.md`).  A non-relativistic Hamiltonian can be
fed to this code through spin-orbital expanded integrals, but it has its own
implementation: `zmr.nr`, spin-free UC- and PC-NEVPT2 with real spatial
orbitals (`docs/derivations/nr-nevpt2.md`).

Start with `docs/usage.md` (how to run a calculation) and
`docs/calculations.md` (what has been computed, where, and the numbers).

## Layout

    zmr/reference/  brute-force reference (determinants, operator strings, RDMs by definition)
    zmr/active/     active-space sectors, operator strings, symmetry-adapted active orbitals
    zmr/uc/         UC-NEVPT2, perturber classes, contraction engines
    zmr/interface/  PySCF / socutils adapters (the only place importing them), AO -> MO transformation
    zmr/pc/         PC-NEVPT2 (contracted spaces from CI vectors or from RDMs)
    zmr/nr/         non-relativistic spin-free UC- and PC-NEVPT2 (real spatial orbitals; CI-vector and density-matrix routes)
    zmr/wick/       Wick engine and generated PC-NEVPT2 terms
    zmr/lib/        native code: OpenMP kernels (libzmr) and MKL wrappers, with numpy fallbacks
    zmr/tests/
    docs/derivations/

zmr lives in socutils as the subpackage `socutils.zmr` (this directory; paths
below are relative to it), imported as `from socutils.zmr...`.

## Building

    make            # here, or `make` at the socutils top level (builds it too)

builds `lib/libzmr.so` (C + OpenMP) and links it to the `libmkl_rt` of the
Python environment, the library PySCF uses; dense linear algebra goes through
the same MKL.  Without the build (or with `ZMR_NO_NATIVE=1`) everything runs
on numpy fallbacks, which are slower but identical in results.  Threads:
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` or `socutils.zmr.lib.set_num_threads(n)`.
See `docs/derivations/performance.md`.

## Running the tests

The reference implementation uses `socutils.fci.zfci` from the socutils-git
checkout:

    export PYTHONPATH=<directory holding a link socutils -> this checkout>:$PYTHONPATH
    cd zmr && python -m pytest tests

## Example

    from socutils.zmr.interface import pyscf_nr        # or socutils_x2c for X2C spinors
    from socutils.zmr.uc.nevpt2 import UCNEVPT2

    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    uc = UCNEVPT2(eris, nelec, nstate=3, e0='average',
                  method={'ccav': 'laplace', 'cavv': 'laplace'}, nquad=30)
    energies = uc.kernel()        # eigenvalues of the effective Hamiltonian
    uc.h2_by_class['cava']        # per-class second-order matrices

PC-NEVPT2 takes the same arguments plus the multi-state contraction:

    from socutils.zmr.pc.nevpt2 import PCNEVPT2

    pc = PCNEVPT2(eris, nelec, nstate=3, e0='average',
                  contraction='union',      # or 'state' (QD-NEVPT2 style)
                  amplitudes='spectral')    # or 'rdm' (Wick-generated RDM formulas)
    pc.kernel()

For atoms and linear molecules (or spin-free integrals) rotate the active
orbitals to symmetry-adapted ones first; the sector Hamiltonians then become
block diagonal and the active-space diagonalizations cost one to two orders
of magnitude less, with unchanged results:

    from socutils.zmr.active import symmetry

    eris, sym = symmetry.adapt(eris)   # sym is None if only the particle number is conserved
    pc = PCNEVPT2(eris, nelec, nstate=9, e0='average')

Non-relativistic calculations start from a PySCF CASCI / CASSCF object;
`nelec` is `(n_alpha, n_beta)` and the model states are the CASSCF CI vectors
(or `nstate` lowest roots of that sector):

    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2, UCNEVPT2

    eris, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=1)
    pt = PCNEVPT2(eris, nelec, civecs=pyscf_nr.nr_civecs(mc), e0='average')
    pt.kernel()

The contraction is spin-free, so the result does not depend on the spin
projection of the model states, and states of different spin can share a
model space (`examples/03-nr_o2_states.py`).

Derivations: `docs/derivations/` (`uc-nevpt2.md`, `pc-nevpt2.md`,
`nr-nevpt2.md`, `wick.md`, `performance.md`, generated formulas in
`generated/`).  Validation record:
`docs/derivations/validation-log.md`.  Examples: `01-o2_zfs.py` and
`04-o2_zfs_active_spaces.py` (zero-field splitting of O2 from X2CAMF SA-CASSCF
orbitals, full-valence and minimal active space), `02-pr3_levels.py` (4f^2
levels of Pr3+), `03-nr_o2_states.py` (non-relativistic multi-state NEVPT2).

For energy differences between model states use the default `e0='state'`;
`e0='average'` is invariant under arbitrary rotations of the model space but
not separable (`docs/derivations/conventions.md`, section 9).
