# Using zmr

How to run a calculation with the code as it is.  Derivations are in
`docs/derivations/`, the calculations done so far in `docs/calculations.md`.

## Environment

    source ~/.conda_activate
    export PYTHONPATH=$HOME/apps/socutils-git-env:$PYTHONPATH    # socutils, with socutils.zmr
    export OMP_NUM_THREADS=N MKL_NUM_THREADS=N

`socutils/zmr/lib/libzmr.so` is built with `-march=native` on an AVX-512 node (the
`sr*` nodes of the `express` and `shared` partitions).  On other CPUs set
`ZMR_NO_NATIVE=1` (numpy fallbacks, same results) or rebuild with `make`.
Rebuild (in socutils/zmr, or `make` at the socutils top level) after changing `lib/src/zmr.c`:

    srun -p shared -A lcheng24 -n1 -c16 --mem=32000M --time=1:00:00 make

Tests (about 3 minutes on 16 cores, 601 tests; the warnings about intruder
states come from random model Hamiltonians and are expected):

    python -m pytest zmr/tests -q

## The three steps of a calculation

1. **Orbitals** from socutils (spinor SCF, CAHF or SA-CASSCF) or PySCF.
2. **Hamiltonian** (`eris`): an interface function turns orbitals into an
   object that delivers integral blocks, the Fock matrix and the core energy.
3. **NEVPT2**: `PCNEVPT2` or `UCNEVPT2`; `kernel()` returns the eigenvalues of
   the effective Hamiltonian of the model space.

### Hamiltonian from a socutils spinor SCF

    from socutils.zmr.interface import socutils_x2c
    eris = socutils_x2c.from_scf(mf, ncore=NC, nact=NA, nfrozen=NF,
                                 mo_coeff=mo, mol=mol, cderi=cderi)

* Counts are spinors.  Orbital order in `mo`: `NF` frozen, `NC` correlated
  core, `NA` active, the rest virtual.  The active block may be any orbitals
  (a deep shell for core holes): reorder the columns of `mo`.
* `mf.get_hcore()` supplies the one-electron operator (X2CAMF, X2CMP, with
  point charges, ...).  `mol` may differ from `mf.mol` in charge and spin.
* `cderi`: Cholesky vectors of the real spherical AO integrals, packed
  (`naux, nao (nao + 1) / 2`) or square; the ones of the socutils CASSCF
  (`lib.chkfile.load('cderi.chk', 'j3c')`).  With them zmr reproduces the
  CASSCF energies to 1e-11 Eh and stores nothing of the size of the AO
  integrals.  Without `cderi` the exact AO integrals are kept in memory
  (`nao^4 / 4` numbers).

From PySCF: `pyscf_nr.from_casci(mc, nfrozen=, cderi=)` (spin-orbital form for
the spinor code) and `pyscf_nr.nr_from_casci(mc, nfrozen=, cderi=)` with
`pyscf_nr.nr_civecs(mc)` for the spin-restricted code `zmr.nr`.

### Choosing the correlated space

    from socutils.zmr import fock
    eris, kept, dropped = fock.truncate_virtuals(eris, gamma, 100.0)

removes virtual orbitals above about 100 Eh (the cut is placed in a gap of
the virtual spectrum); `truncate_virtuals(eris, gamma, nkeep=N)` keeps
exactly the N lowest.

Which core orbitals to freeze and which virtuals to drop is a choice to be
made by looking at the orbitals.  `fock.orbital_report(eris, gamma)`, called
for the Hamiltonian without frozen core and with all virtuals, prints the
quasi-canonical core and virtual shells with their energies and the gap
above each; the column "below" is the `nfrozen` (or `nkeep`) that cuts
there.  Canonicalize first, then freeze: `eps, u = fock.canonical_core(eris,
gamma)`, `mo[:, :ncore] = mo[:, :ncore] @ u`, then `from_scf(..., nfrozen=)`.
The orbitals of the SCF or CASSCF are defined only up to a rotation among
the core, and a frozen core cut out of them is not a defined space.  The
driver below has `--orbitals`, `--nfrozen` and `--nkeep` for this.  `gamma` is the averaged active 1-RDM,
`act.averaged_rdm1(civecs, nelec)`.  The frozen core is the first `nfrozen`
columns of `mo`; to freeze by shell, diagonalize the core block of
`eris.generalized_fock(gamma)` first (as `examples/05-chalcogen_zfs.py` does).

What to correlate (from the calculations so far): for 4f ions the whole
n = 4 shell and 5s 5p (freeze up to 3d); for p-block atoms and molecules
ns and (n-1)d, the rest of the (n-1) shell changes little.

### Model states

    from socutils.zmr.active import symmetry
    from socutils.zmr.active.sectors import ActiveSpace
    eris, sym = symmetry.adapt(eris)        # optional: atoms, linear molecules
    act = ActiveSpace.from_eris(eris)
    e_cas, civecs = act.solve(nelec, nstate)

`symmetry.adapt` rotates the active spinors to eigenfunctions of the one-body
symmetries it finds (Jz for linear molecules, J for atoms), which makes the
sector Hamiltonians block diagonal; results do not change.

### NEVPT2

    from socutils.zmr.pc.nevpt2 import PCNEVPT2
    pt = PCNEVPT2(eris, nelec, civecs=civecs, e0='state', contraction='union')
    pt.verbose = 1             # default; 0 silent, 2 per-state tables for many states
    energies = pt.kernel()

| option | values | meaning |
|--------|--------|---------|
| `civecs` / `nstate` | model states as columns, or the lowest `nstate` | model space of the effective Hamiltonian |
| `e0` | `'state'` (default), `'average'` | E0 in the denominators.  Use `'state'` for energy differences: the common E0 is not separable and enlarges every splitting by a fixed fraction |
| `contraction` | `'union'`, `'state'` | first-order space spanned by the functions of all model states, or of each ket state |
| `amplitudes` | `'spectral'`, `'rdm'`, `'auto'` | CI-vector route, or reduced quantities of 3-RDM cost plus intermediate states (no 4-RDM).  Same energies.  Spinor code: `'spectral'` is the default, `'rdm'` needs CASCI eigenstates.  `zmr.nr`: `'auto'` is the default and picks, class by class, `'rdm'` if the functions of all states are fewer than the determinants of the target sectors (few states, large active space) and `'spectral'` otherwise (many states) and for complex model states; `pt.routes` records the choice |
| `pt.cvs` | `False`, `True` | core-valence separation for core-hole model states, by the number of holes: ccaa takes both electrons, ccav at least one, caaa its electron from the inner core (core orbitals deeper than the active shell); the other labels (decay channels) are left out of the energies |
| `pt.cvs_core` | `None` or indices | inner core: indices among the quasi-canonical correlated core orbitals in ascending energy; `None`: those below the lowest active orbital energy |
| `pt.cvs_decay` | `True`, `False` | also evaluate the decay channels and print, per root, E(CVS), their contributions (Q discrete: ccaa, caaa from outer shells; Q continuum: ccav, one electron emitted) and E(CVS+Q); results in `pt.h2_decay_by_class`, `pt.e_tot_decay` |
| `pt.cvs_window` | Eh, default 0.05 | Q continuum: terms with \|denominator\| below this are left out on both sides (principal value); the number and weight of these terms are printed |
| `pt.skip_classes` | tuple of class names | leave classes out |

`kernel()` prints, in the manner of ORCA's NEVPT2 section (`zmr/output.py`):
the settings and dimensions, the reference energies, one line per step
(written when the step starts and completed when it ends, so a running job
shows where it is), the second-order energies by class, the eigenvalues of
the effective Hamiltonian in Eh / cm^-1 / eV, and the timings.  Every class
line gives its smallest energy denominator (also in `pt.min_denominators`);
below `zmr.output.INTRUDER_THRESHOLD` (0.01 Eh) the class is reported as
affected by an intruder state, in the output and as a Python warning, and
the calculation continues.  Inside a step
that takes long a heartbeat line is written every `pt.heartbeat` seconds
(default 30) with the position in the loop that is running (external block,
interpolation node, state) and the elapsed time.  `pt.stdout` redirects the
output to a file object.

Results: `pt.heff` (effective Hamiltonian), `pt.e_tot`, `pt.hmodel`,
`pt.h2`, `pt.h2_by_class`, `pt.timings`.  `UCNEVPT2` has the same interface
without `contraction`; `zmr.nr.PCNEVPT2` / `UCNEVPT2` take the Hamiltonian of
`nr_from_casci` and `(n_alpha, n_beta)`.

The sums over external labels are interpolated in the energy argument where
that is exact to round-off (theory in
`docs/derivations/resolvent-interpolation.md`, validation in
`docs/derivations/performance.md` section 6);
`ZMR_NO_INTERPOLATION=1` in the environment uses exact denominators
throughout, for checking.

### MRCI

    from socutils.zmr.mrci import MRCI
    ci = MRCI(eris, nelec, civecs=civecs)        # or nstate=
    energies = ci.kernel()                       # one root per model state

Internally contracted MRCI with singles and doubles for the same spinor
Hamiltonian objects (`docs/derivations/mrci.md`); meant for a few states.

| option | values | meaning |
|--------|--------|---------|
| `contraction` | `'fic'` (default), `'wk'`, `'none'` | all external blocks contracted; only the doubly external ones (Werner-Knowles type); uncontracted MRCISD |
| `internal` | `'cas'` (default), `'states'` | the reference coefficients relax (all CAS determinants), or the model states only |
| `functional` | `'ci'` (default), `'acpf'`, `'aqcc'` | MRCI, or the approximately size-extensive functionals (one eigenproblem per model state) |
| `active_eri` | array | `(tu|vw)` that replace those of `eris` in the all-active part of H (X2Ccorr) |

`kernel()` prints the settings, the table of blocks (external labels x
active functions), the setup steps, one line per Davidson iteration, the
results per root (energy, rotated reference, weight of the internal block
and of every other block, Davidson corrections) and the timings with the
largest sigma contributions.  `ci.verbose = 0` is silent.  The roots are
converged when their residual norms are below `ci.conv_tol` (5e-5 by
default; set it before `kernel()` for tighter energies).  With
`ci.follow = True` the roots are the states with the largest weight in the
model states instead of the lowest ones (for the lowest states of one
symmetry, e.g. one Omega of a linear molecule, when lower states of other
symmetries exist).  Davidson trial vectors beyond `ci.davidson_incore`
bytes (4e10) are kept in mapped scratch files in `ZMR_TMPDIR` (else PySCF's
`TMPDIR`): put it on a local disk.  Results:
`ci.e_tot`, `ci.e_davidson`, `ci.e_davidson_renormalized`, `ci.weights`,
`ci.block_weights`, `ci.ci`, `ci.converged`; `ci.sigma.memory_report()`
lists the arrays kept between iterations.  With Cholesky / RI vectors the
`(vv|vv)` terms use a matrix built from the three-index integrals up to
`Sigma.vvvv_dense_memory` bytes and the three-index integrals directly
beyond that.

## Ready-made drivers

`examples/05-chalcogen_zfs.py` does all of the above for the chalcogen
diatomics and for atoms and caches every stage in the working directory:

    python 05-chalcogen_zfs.py TeO --emax 100 --core valence,d,shell --uc
    python 05-chalcogen_zfs.py Se --nsa 15 --cas 6,8 --corr
    python 05-chalcogen_zfs.py SeO --scf-charge 0

`--model x2cmp|x2camf`, `--ham dcb|dcg|dc`, `--nsa` (state averages),
`--core` (frozen-core choices; the default runs `valence` and `d`, which
are the same space for light molecules: give one), `--emax` (virtual cutoffs, 0 = none), `--cas
nelec,nspinor`, `--corr` (X2Ccorr correction of the levels from
four-component active-space integrals), `--mrci ci,aqcc` (MRCI for the first
model space; with `--corr` the correction is recomputed in the MRCI),
`--scf-charge` (closed-shell species
of the SCF).  Result lines start with `ZFS`; one `RESULT {json}` line per
calculation.

Slurm: `~/scr4/zmr_runs/new_run.sh <name>` snapshots the working tree into
`~/scr4/zmr_runs/<name>/src`; job scripts there set `PYTHONPATH=$PWD/src`.
`express`: at most 4 CPUs and `--mem=16000M` per job (16G counts as 5 CPUs),
8 h.  `shared`: up to 32 CPUs, 4 GB per CPU, starts within a minute.

## What the code cannot do yet

* Large active spaces: sectors above `ActiveSpace.dense_limit` (20000)
  determinants are solved matrix free, but with a plain Lanczos solver that
  is slow for many states, and the perturber classes still use dense sector
  algebra.
* The RDM routes store all functions `O_X |Psi>` of a sector at once (and the
  spinor one the 3-RDM explicitly); no batching.  For `zmr.nr` that is
  `2 nstate (n + n^3) D` numbers, about 20 GB for one state in CAS(12,12).
* `zmr.nr` with Cholesky / RI vectors (`cderi=`) uses the same scheme as the
  spinor code (three-index integrals, task kernels, interpolation, `cvs`).
  Its RDM route (`docs/derivations/nr-nevpt2.md` section 6) brings one state
  in CAS(10,10) to the time of ORCA; with several states its cost grows with
  the square of their number, and the two spin sectors are computed
  separately unless the CI vectors are spin-flip symmetric to 1e-9
  (`pt.rdm_flip_tol`; converge the CI vectors tightly).
* Core holes in more than one shell of the active space (an occupation-
  restricted model space) are not possible; `cvs` assumes one active shell.
