# zshci

Selected configuration interaction for spinor (two- or four-component)
Hamiltonians, in complex or in real arithmetic.

Given one- and two-electron integrals over spinors, `zshci`

- selects a variational space of determinants with the heat-bath criterion
  (`"mode": "select"`), or takes a list of determinants as given
  (`"mode": "diag"`),
- diagonalizes the Hamiltonian in it (Davidson),
- optionally adds second-order Epstein-Nesbet perturbation theory:
  deterministic, stochastic or semistochastic,
- writes energies, one- and two-body reduced density matrices and transition
  one-body density matrices.

Additive orbital labels (for example twice the angular momentum projection of
a linear molecule, optionally modulo an integer) restrict the calculation to
one symmetry block, and bounds on the occupation of sets of spinors restrict
it to RAS-type spaces.

## Building

A C++17 compiler with OpenMP, Eigen 3 and HDF5 are needed; a serial build of
HDF5 is enough, also for the MPI program. HighFive and nlohmann/json are
included in `third_party/`.

Eigen and HDF5 are found automatically in an active conda environment
(`conda install eigen hdf5`) or under `/usr`. For other locations, write them
into `config.mk`:

    EIGEN_ROOT = /path/to/eigen      # the directory that contains Eigen/
    HDF5_ROOT  = /path/to/hdf5       # contains include/ and lib/

Then

    make                    # bin/zshci: threads only, no MPI needed
    make MPI=1 CXX=mpicxx   # bin/zshci_mpi: for runs on several nodes
    make test               # self-checks

Both programs do everything; `zshci_mpi` is only needed to use more than one
node. Any combination of ranks and threads gives the same results.

## Running

    bin/zshci input.json
    mpirun -np 4 bin/zshci_mpi input.json

The input is a JSON file; comments are allowed and unknown keys are errors.

```json
{
  "mode": "select",
  "integrals": "FCIDUMP",
  "nroots": 1,
  "prefix": ".",
  "dets": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
  "symmetry": [{"labels": "fcidump", "mod": 0}],
  "connect": {"method": "strings"},
  "select": {"schedule": [[0, 1e-3], [3, 1e-4]], "max_iter": 12},
  "davidson": {"tol": 1e-6},
  "pt": {"method": "semistochastic", "epsilon2": 1e-7, "epsilon2_large": 1e-5,
         "target_error": 1e-5},
  "rdm": {"one": true, "two": true, "transition": false}
}
```

`"symmetry"` lists the conserved quantities, each an integer label per spinor
and a modulus (0: none): `{"labels": [...], "mod": 4}` is the label of a
twofold axis, a second entry `{"labels": [0, 1, ...], "mod": 2}` adds
inversion. The entries are independent of each other. A single entry may be
written without the brackets; `"labels": "fcidump"` takes ORBSYM of the
integral file. The integrals must obey the labels: an integral that would
change one is an error if it exceeds `"symmetry_tolerance"` (default 1e-8) and
is dropped otherwise. The two-electron integrals are stored in the blocks the
labels define, so labels also save memory.

`src/input.h` documents every key. The integrals are read from a FCIDUMP
file (`src/integrals.h`) in the complex format, `(re,im) p q r s`, or the real
format, `value p q r s`. A file in the real format is treated with real
numbers throughout (integrals, Hamiltonian, CI vectors, density matrices),
which takes about two thirds of the memory; `"scalar": "real"` asks for the
same with a file in the complex format whose imaginary parts are all zero, and
`"scalar": "complex"` forces complex numbers. The integrals are still indexed
by spinors: real numbers do not imply spin symmetry. They are held
antisymmetrized, `<pq||rs> = (pr|qs) - (ps|qr)`, about `n^4/8` numbers for `n`
spinors (3.2 GB of complex numbers for 200 spinors). The program exits with a
nonzero status on any error.

Output, in the directory `prefix` (`src/output.h`):

| file | content |
|---|---|
| `shci.e` | total energies, including the perturbative correction if any |
| `wfn.h5` | determinants, CI vectors, energies, the resolved input |
| `rdm_i_i.h5` | `dm1` and `dm2` of state i, as real or complex arrays (the number type of the calculation) in the convention of PySCF |
| `rdm_0_j.h5` | `dm1`: transition density matrix between states 0 and j |

`dm1[p,q]` is `<p^+ q>` and `dm2[p,q,r,s]` is `<p^+ r^+ s q>`; a reader takes
the datasets as they are. `wfn.h5` can be given as `"detfile"` to restart from
a previous space.

## Layout

| file | content |
|---|---|
| `src/det.h` | determinants as fixed-width bit strings |
| `src/integrals.*`, `src/hamiltonian.h` | integrals, Slater-Condon rules |
| `src/heat_bath.*`, `src/excite.h` | excitation generation, symmetry and occupation filters |
| `src/connect.h`, `src/sparse_h.*` | connected pairs of determinants, the sparse Hamiltonian |
| `src/davidson.*`, `src/distributed_h.*` | eigensolver, alone and over MPI ranks |
| `src/select.h`, `src/diag.h`, `src/driver_common.h` | the two modes |
| `src/pt.h`, `src/rdm.h` | perturbation theory, density matrices |
| `src/comm.*`, `src/parallel.h` | MPI wrapper, node-shared arrays, threading helpers |
| `test/` | self-checks and a small test Hamiltonian |
| `bench/` | inputs and job scripts of the timing runs |

The code follows the Google C++ style guide.
