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
  "connect": {"method": "groups"},
  "select": {"schedule": [[0, 1e-3], [3, 1e-4]], "max_iter": 12},
  "davidson": {"tol": 1e-6},
  "pt": {"method": "semistochastic", "epsilon2": 1e-7, "epsilon2_large": 1e-5,
         "target_error": 1e-5},
  "rdm": {"one": true, "two": true, "transition": false}
}
```

`"orbitals": "spatial"` is for a non-relativistic calculation with a
restricted reference: the FCIDUMP file holds real integrals `(ij|kl)` over
spatial orbitals (the standard format, as written by PySCF or Molpro), which
are stored as such, `n^4/8` numbers for `n` orbitals. The determinants are over
the spin orbitals `2P` (alpha) and `2P+1` (beta) of spatial orbital `P`, the
spin projection is conserved automatically, and ORBSYM is read as the
irreducible representations of D2h or a subgroup in Molpro's numbering.
Connected determinants are then found from tables of alpha and beta strings
(`src/connect_groups.h`), as in the non-relativistic Dice program.

`"orbitals": "unrestricted"` is for an unrestricted reference, different
spatial orbitals for the two spins: the integrals come from an HDF5 file
(`h1_a`, `h1_b`, `eri_aa`, `eri_bb`, `eri_ab`; `src/integrals.h`) and are held
over the spin orbitals in spin blocks, the density matrices are over the spin
orbitals.

`"symmetry"` lists the conserved quantities, each an integer label per spinor
and a modulus (0: none): `{"labels": [...], "mod": 4}` is the label of a
twofold axis, a second entry `{"labels": [0, 1, ...], "mod": 2}` adds
inversion. The entries are independent of each other. A single entry may be
written without the brackets; `"labels": "fcidump"` takes ORBSYM of the
integral file. The integrals must obey the labels: an integral that would
change one is an error if it exceeds `"symmetry_tolerance"` (default 1e-8) and
is dropped otherwise. The two-electron integrals are stored in the blocks the
labels define, so labels also save memory.

`"select": {"method": ...}` chooses the kind of selected CI. The loop is the
same for all (`src/select.h`); a selector class says which determinants come
in:

| method | adds | controlled by |
|---|---|---|
| `"heat_bath"` (default) | D_a with \|H(a,i) c_i\| > epsilon1 for some D_i in the space | `"schedule"` |
| `"cipsi"` | the D_a of largest second-order energy \|V_a\|^2 / (E_a - E), V_a = sum_i H(a,i) c_i | `"target_dets"`, `"growth"` (factor per iteration, default 2) |
| `"asci"` | the D_a of largest first-order amplitude \|V_a\| / (E_a - E), V_a from the `"core_dets"` determinants of largest coefficient | `"target_dets"`, `"growth"`, `"core_dets"` |
| `"aci"` | all D_a except the smallest second-order energies that sum to `"sigma"` | `"sigma"` |

The last three score every determinant outside the space, which costs about
one deterministic perturbation-theory pass per iteration and root;
`"epsilon_generate"` restricts V_a to the terms above it. None of them
removes determinants from the space, unlike the published ASCI and ACI,
unless `"prune": true` is given. For `"asci"` the determinants of the space
are then ranked with the candidates, by the same amplitude from the
generating determinants, and the space is the `"target_dets"` highest of both
(while it grows: `"growth"` times its size). For `"aci"` the determinants of
largest coefficients that hold all but `"gamma"` (default 1) times sigma of
the squared norm generate the candidates and stay, and every other
determinant, of the space or not, is scored from them and kept or left out
by the sigma rule. With any method, `"prune_coefficient": c` takes out, after
each diagonalization, the determinants whose coefficients are below c
(root-sum-square over the roots) and diagonalizes again in what remains.
A determinant taken out stays in
the list of the program with coefficient zero, so that the stored Hamiltonian
is neither reordered nor rebuilt, and comes back at no cost if it is selected
again; the eigensolver works in the others. The list is cleared of such
determinants, and the Hamiltonian made afresh, at the end and, with
`"compact_above": f`, whenever more than the share f of the list is out of
the space. On N2
CAS(10e,12o) at equal size the cipsi and asci spaces are lower in energy than
the heat-bath one (0.81 and 0.83 against 1.30 mHartree above full CI at 1635
determinants), and the aci error follows sigma (0.93, 0.096 and 0.0096
mHartree for sigma = 1, 0.1 and 0.01 mHartree).

`"connect": {"method": ...}` chooses how connected determinants are found
(`src/connect.h`): `"groups"`, the default, divides the spinors into two
groups (`"group_a"`, default the even ones) and works from tables of the
strings of each group, electrons being allowed to move between the groups;
`"strings"` and `"segments"` are the earlier methods. With a conserved spin projection
that is the alpha/beta scheme and the default. Timings of the three on two
spinor cases, 16 threads, seconds spent building the Hamiltonian over a whole
selection run and the peak memory of the program:

| | strings | segments | groups |
|---|---|---|---|
| O2, 12 electrons in 32 spinors, 433k determinants | 6.2 s, 3.5 GB | 11.0 s, 2.9 GB | 2.7 s, 3.2 GB |
| Cl2, 34 electrons in 52 spinors, 124k determinants | 4.2 s, 2.5 GB | 1.8 s, 0.4 GB | 0.8 s, 0.7 GB |

`"hamiltonian": "direct"` does not store the matrix: every product with a
vector works the connections out again from the tables of the group method
and evaluates the elements on the way, and the density matrices are computed
the same way. On the O2 case above (433k determinants, 131 million elements,
16 threads) a product costs several times a stored one and the peak memory
is a fraction of a gigabyte instead of 3.1 GB; at 243k determinants a product
takes 1.0 s against 0.2 s. For when the matrix does not fit.

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
a previous space; a selection then also starts the eigensolver from the CI
vectors of that file, if it holds one per root.

## Layout

| file | content |
|---|---|
| `src/det.h` | determinants as fixed-width bit strings |
| `src/integrals.*`, `src/hamiltonian.h` | integrals, Slater-Condon rules |
| `src/heat_bath.*`, `src/excite.h` | excitation generation, symmetry and occupation filters |
| `src/connect.h`, `src/connect_groups.h`, `src/sparse_h.*`, `src/direct_h.h` | connected pairs of determinants, the sparse Hamiltonian, the Hamiltonian applied without storing it |
| `src/davidson.*`, `src/distributed_h.*` | eigensolver, alone and over MPI ranks |
| `src/select.h`, `src/diag.h`, `src/driver_common.h` | the two modes |
| `src/pt.h`, `src/rdm.h` | perturbation theory, density matrices |
| `src/comm.*`, `src/parallel.h` | MPI wrapper, node-shared arrays, threading helpers |
| `test/` | self-checks and a small test Hamiltonian |
| `bench/` | inputs and job scripts of the timing runs |

The code follows the Google C++ style guide.

## Benchmarks

`bench/n2/` is a non-relativistic comparison with Dice: `make_inputs.py`
writes the integrals and equivalent inputs for both programs, `run.sh` runs
zshci with threads, zshci_mpi and Dice on one node. Measured on 16 cores
(N2, cc-pVDZ, 10 electrons in 26 orbitals, 674709 determinants at
epsilon1 = 5e-5, deterministic PT with epsilon2 = 1e-6):

| | zshci, 16 threads | zshci_mpi, 16 ranks | Dice, 16 ranks |
|---|---|---|---|
| variational part | 8.0 s | 7.5 s | 9.4 s |
| perturbation theory | 4.2 s | 3.1 s | 23.1 s |
| whole run | 12.6 s | 12.8 s | 34.4 s |

The two programs select slightly different spaces (673981 determinants in
Dice) and agree to 4e-8 Eh in the energy with the perturbative correction.
