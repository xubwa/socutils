# zshci

Selected configuration interaction for spinor (two- or four-component,
complex) Hamiltonians.

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
  "symmetry": {"labels": "fcidump", "mod": 0},
  "select": {"schedule": [[0, 1e-3], [3, 1e-4]], "max_iter": 12},
  "davidson": {"tol": 1e-6},
  "pt": {"method": "semistochastic", "epsilon2": 1e-7, "epsilon2_large": 1e-5,
         "target_error": 1e-5},
  "rdm": {"one": true, "two": true, "transition": false}
}
```

`src/input.h` documents every key. The integrals are read from a complex
FCIDUMP file (`src/integrals.h`). The program exits with a nonzero status on
any error.

Output, in the directory `prefix` (`src/output.h`):

| file | content |
|---|---|
| `shci.e` | total energies, including the perturbative correction if any |
| `wfn.h5` | determinants, CI vectors, energies, the resolved input |
| `Dice_i_i.rdm1.h5`, `Dice_i_i.rdm2.h5` | density matrices of state i |
| `Dice_0_j.rdm1.h5` | transition density matrix between states 0 and j |

The energy and density-matrix files have the names and layout of the ZSHCI
program of Dice, so readers written for it keep working. `wfn.h5` can be given
as `"detfile"` to restart from a previous space.

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
