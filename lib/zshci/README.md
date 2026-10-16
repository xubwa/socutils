# zshci

Selected configuration interaction for spinor (two- or four-component)
Hamiltonians, in complex or in real arithmetic.

Given one- and two-electron integrals over spinors, `zshci`

- selects a variational space of determinants with the heat-bath criterion
  (`"mode": "select"`), or takes a list of determinants as given
  (`"mode": "diag"`), or does full or GAS CI without determinant lists
  (`"mode": "fci"`, below),
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
    MKL_LIB    = /path/to/mkl/lib     # contains libmkl_rt.so (fci mode)

Then

    make                    # bin/zshci: threads only, no MPI needed
    make MPI=1 CXX=mpicxx   # bin/zshci_mpi: for runs on several nodes
    make test               # self-checks
    make BLAS=none          # bin/zshci_noblas: the fci kernel without BLAS

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
`"strings"` and `"segments"` are the earlier methods. With a conserved spin
projection that is the alpha/beta scheme. Connected determinants have
strings that are related (one or two electrons out, one or two in), and
`"relations"` says how the strings related to a string are found:

- `"bytes"`, the default, stores no lists of them. The strings of a group
  are read byte by byte, eight spinors of the group at a time; one table
  made at the start says what each of the 256 values of a byte turns into
  when up to two electrons leave and enter it; and the distinct strings are
  kept as a tree over their bytes, in which the strings related to a string
  are found by a walk that gives up a family of candidates at the first
  byte no string has (`src/connect_bytes.h`). What is held, once per node
  of the machine, is the trees and the determinants ordered by string with
  their indices, 24 bytes per one-word determinant for both groups. Nothing
  is kept to extend the tables: they are made anew when determinants are
  added, which takes 2 s for 22 million determinants.
- `"subgroups"` and `"keys"` store, for every string, the lists of the
  strings related to it (`src/connect_groups.h`), found by dividing each
  group once more into two sub-groups or from keys, of which a string of n
  electrons has n (n + 1) / 2. For spinors without a conserved spin
  projection the lists have about 800 entries per string, more than a
  kilobyte per determinant.

Old (the program before the byte tables, with its defaults `"subgroups"`
for spinors and `"keys"` for spatial orbitals) against `"bytes"`, on
restarts that build everything for the saved determinants at once and do
one selection step; 24 threads on one socket of a Xeon Gold 6248R. Tables:
held per node, and the most held in addition while they are made. Search:
the tables and the connections of all rows. Wall time and peak memory are
of the whole run; the Cr2 cases have 24 electrons in 60 spinors (30 spatial
orbitals for RHF); M: millions of determinants.

| case, `"hamiltonian"` | tables (MB) | search (s) | products (s) | wall (s) | peak (GB) |
|---|---|---|---|---|---|
| Cr2 X2C, 2.05 M, `"stored"` | 1455 + 2064 → 56 + 63 | 13.2 → 4.3 | 10.4 → 9.5 | 34 → 26 | 8.6 → 5.9 |
| Cr2 X2C, 2.05 M, `"connections"` | 1455 + 2064 → 56 + 63 | 12.9 → 3.6 | 26.4 → 25.1 | 50 → 42 | 5.6 → 2.8 |
| Cr2 X2C, 2.05 M, `"direct"` | 1486 + 2064 → 56 + 63 | — | 76.0 → 59.9 | 95 → 74 | 4.9 → 2.3 |
| Cr2 X2C, 5.79 M, `"stored"` | 3786 + 5356 → 151 + 177 | 46.1 → 15.9 | 40.9 → 33.2 | 120 → 84 | 23.6 → 18.9 |
| Cr2 X2C, 5.79 M, `"connections"` | 3786 + 5356 → 151 + 177 | 45.2 → 14.2 | 113.2 → 104.9 | 191 → 154 | 13.4 → 8.3 |
| Cr2 X2C, 5.79 M, `"direct"` | 3874 + 5356 → 151 + 177 | — | 384.4 → 308.4 | 442 → 349 | 10.2 → 4.3 |
| 26 e, 60 spinors, 1.21 M, `"stored"` | 286 + 374 → 31 + 37 | 13.9 → 12.8 | 32.0 → 30.8 | 75 → 74 | 21.6 → 21.8 |
| 26 e, 60 spinors, 1.21 M, `"connections"` | 286 + 374 → 31 + 37 | 12.9 → 10.3 | 106.1 → 98.2 | 149 → 139 | 6.3 → 5.3 |
| 26 e, 60 spinors, 1.21 M, `"direct"` | 304 + 374 → 31 + 37 | — | 177.9 → 142.8 | 212 → 176 | 2.6 → 1.3 |
| Cr2 RHF, 6.01 M, `"stored"` | 260 + 196 → 145 + 183 | 21.0 → 12.3 | 23.6 → 15.6 | 63 → 47 | 12.2 → 11.3 |
| Cr2 RHF, 6.01 M, `"connections"` | 260 + 196 → 145 + 183 | 20.4 → 11.4 | 102.7 → 99.0 | 143 → 130 | 6.5 → 5.9 |
| Cr2 RHF, 6.01 M, `"direct"` | 352 + 196 → 145 + 183 | — | 413.2 → 220.9 | 444 → 245 | 3.2 → 2.4 |
| Cr2 X2C, 22.1 M, `"connections"` | 12250 + 17234 → 549 + 676 | 286.0 → 91.6 | 637.7 → 629.0 | 1054 → 860 | 46.7 → 34.3 |
| Cr2 X2C, 22.1 M, `"adapt"`, 80 GB | 12588 + 17234 → 549 + 676 | 294.7 → 93.2 | 399.0 → 352.0 | 862 → 583 | 73.6 → 73.7 |
| Cr2 X2C, 22.1 M, `"direct"` | 12588 + 17234 → 549 + 676 | — | 2885.5 → 1920.4 | 3136 → 2093 | 30.6 → 16.4 |

"direct" has no rows to find; its products do the search. In all 15 pairs
of runs the energies agree to 4e-12 Hartree or better. A whole selection
from the Hartree-Fock determinant to 2.04 million determinants of the first
system (11 iterations, 289 products, stored) takes 70.1 s and 8.0 GB
against 63.0 s and 5.9 GB, of which the tables and the rows took 14.7
against 4.6 s.

Where the time goes with `"bytes"`: the rows of a stored Hamiltonian are
made for the determinants of a string together, so that the determinants
of a related string are read once for all of them, and the loop that
compares determinants bit by bit handles four at a time (AVX2, one-word
determinants). `"blocks": false` with `"hamiltonian": "direct"`, which
goes determinant by determinant, is the one path that is slower than with
stored lists (all connections of the 2.04 million determinants in 7.4 s
with the lists, 10.5 s with bytes, 16 threads).

Timings of the three methods on two spinor cases, 16 threads, seconds spent
building the Hamiltonian over a whole selection run and the peak memory of
the program, measured with `"relations": "subgroups"`:

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

`"hamiltonian": "connections"` is in between: it stores which determinants are
connected, 4 bytes per element, but not the values, which are worked out again
in every product. Finding the connections is nine tenths of the cost of a
direct product and evaluating an element one tenth, so this keeps most of the
speed for a fifth of the memory of the matrix with complex numbers and a third
with real ones. Cr2 with the X2CAMF Hamiltonian (`bench/cr2`, 24 electrons in
60 spinors, 706655 determinants, 57 million elements, 16 threads): the matrix
takes 1092 MB stored and 218 MB as connections, the 231 products of a run 10.0
and 24.8 s, and the energies are the same; a direct product of that size
takes 2.1 s, against 0.25 s with the connections and 0.044 s stored.

For a Hamiltonian that is not stored the connections are found string by
string rather than determinant by determinant (`src/connect_groups.h`): the
determinants of two related strings lie together in memory and are compared
pair by pair, instead of each determinant going through the lists of several
hundred related strings by itself. The same Cr2 run takes 77 s in its 231
direct products that way and 221 s the other way
(`"connect": {"blocks": false}`); stored 10 s, connections 25 s.

`"hamiltonian": "adapt"` with `"memory": G` chooses between the three, block
by block (a block is the rows added in one iteration), so that the process -
each rank - stays within G gigabytes for the Hamiltonian together with the
string tables, the determinants and the vectors of the eigensolver. A byte
spent on connections saves about ten times the time of a byte spent on
values, hence the order: all connections first; with what is left, values
for as many blocks as fit; and when the connections of new rows do not all
fit, those of as many rows as fit are stored and the other rows are applied
directly. When new rows need room, values of
other blocks are dropped, which releases an array and rebuilds nothing. The
same Cr2 Hamiltonian restarted at 2.04 million determinants (205 million
elements, 16 threads; the tables, determinants and vectors are counted as 5.8
GB):

| | elements with values | peak memory | 34 products |
|---|---|---|---|
| `"stored"` | all | 8.5 GB | 13.5 s |
| `"adapt"`, 12 GB | all | 8.6 GB | 13.5 s |
| `"adapt"`, 8 GB | 44 % | 6.8 GB | 30.0 s |
| `"adapt"`, 6.5 GB | 0.5 % | 5.4 GB | 43.5 s |
| `"connections"` | none | 5.3 GB | 45.0 s |
| `"adapt"`, 5.5 GB | no row stored | 4.5 GB | 139 s |
| `"direct"` | no row stored | (not measured) | 118 s |

with the same energy in every case.

At 22.1 million determinants (3.5 billion elements, 48 threads), where the
tables, the determinants and the larger of extending the tables and the
vectors of the eigensolver are counted as 47 GB:

| | elements with values | peak memory | 36 products |
|---|---|---|---|
| `"stored"` | all | 108.9 GB | 150 s |
| `"adapt"`, 150 GB | all | 109.9 GB | 149 s |
| `"adapt"`, 80 GB | 36 % | 74.7 GB | 286 s |
| `"adapt"`, 60 GB | 0.7 % | 55.4 GB | 397 s |

A direct product of that size takes 54 s, against 4.2 s stored and about 10 s
with the connections.
With 45 GB the connections of 71 % of the rows fit (45.2 GB peak, 1679 s; no
row stored: 33 GB, 1978 s). The gain is small because the direct product
still goes through all pairs of related strings to find the connections of
the rows that are left; it pays to give the limit that holds all connections.

`src/input.h` documents every key. The integrals are read from an HDF5 file
that holds the two-electron integrals antisymmetrized
(`socutils.tools.fcidump_rel.from_integrals_h5`; layout in
`src/integrals.h`), or from a FCIDUMP file in the complex format,
`(re,im) p q r s`, or the real format, `value p q r s`. The binary file is
what to use beyond a few dozen spinors: for Cr2 in 100 spinors with 2 m_j
labels it takes 38 MB, 1.5 s to write and 0.06 s to read, the text file 771
MB, 54 s and 15 s. A file in the real format is treated with real
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

## Full and GAS CI: `"mode": "fci"`

Full CI, or CI in a space bounded by generalized-active-space constraints,
over all spinors of the integral file, without lists of determinants: the
sigma vector comes from the kernel of socutils' zgasci, copied unchanged into
`src/zgasci` (commit and re-sync in `src/zgasci/UPSTREAM`). The spinors are
reordered and cut into sub-groups as `socutils.hci.gasci` does it
(`src/gas_space.*`); the labels of `"symmetry"` select one block.

```json
{
  "mode": "fci",
  "integrals": "o2.h5",
  "nroots": 1,
  "prefix": "/scratch/run",
  "symmetry": [{"labels": "fcidump", "mod": 0}, {"labels": [0, 0, 1, 1], "mod": 2}],
  "symmetry_tolerance": 1e-5,
  "restrictions": [{"orbitals": [0, 1, 2, 3], "min": 4, "max": 4}],
  "fci": {"gas": [{"orbitals": [10, 11, 12], "max": 2, "weights": [1, 1, 2]}],
          "target": [0, 0], "max_space": 3, "checkpoint_every": 1},
  "davidson": {"tol": 1e-6, "max_iterations": 200}
}
```

| key in `"fci"` | meaning (default) |
|---|---|
| `nelec` | electrons (NELEC of the file) |
| `gas` | constraints `{"orbitals", "min", "max", "weights"}`: `min <= sum weight * n_p <= max`; the top-level `restrictions` are added to them |
| `target` | sum of the labels of the occupied spinors, one per label set (that of the lowest `nelec` spinors) |
| `max_subgroup`, `subgroups` | the cut into sub-groups (12; `[[groups of side A], [groups of side B]]`) |
| `kernel` | `"blas"`, `"block"` or `"reference"` (`"blas"`) |
| `max_space` | basis vectors of the Davidson, each with its product (3 per root) |
| `level_shift` | of the preconditioner `1 / (H_ii - E + shift)` (1e-3) |
| `checkpoint`, `checkpoint_every`, `restart` | file of the Ritz vectors (`<prefix>/fci.ckpt`, `"none"`: none), written every so many iterations (1); `restart` starts from it |
| `huge_pages` | `madvise` the vectors for transparent huge pages (true) |
| `distributed`, `balance`, `calibrate`, `slice_gb` | MPI, see below |
| `write_wfn` | `wfn.h5` is written up to this many determinants (1e7) |
| `dry_run` | only the plan and the memory needed |
| `benchmark_sigmas` | only time this many products with H |

`davidson.tol` (residual norm), `davidson.max_iterations` and
`davidson.dense_dim` apply. Imaginary parts must be exactly zero for real
arithmetic: write such integral files with real numbers. Integrals that
change a label are dropped below `symmetry_tolerance` (they cannot act within
one block); the O2 files of the zfs work have such elements up to 1.3e-6.

**Davidson** (`src/fci_davidson.*`). It keeps `max_space` basis vectors and
their products with H and nothing else of the full length besides the
diagonal; combinations are formed in place, element by element. On
collapse it keeps the Ritz vectors and the previous ones (as LOBPCG does),
so 3 vectors per root converge in about as many products as the 12-vector
Davidson of `pyscf.lib.davidson1` (O2 8 in 48: 19 against 23). Every
`checkpoint_every` iterations the Ritz vectors go to `checkpoint` (written
beside it and renamed, so a killed job leaves the previous one) with a
fingerprint of the space; `"restart": true` reads them back, also on
another number of ranks. Memory is printed and checked at the start: per
rank `2 max_space` vectors, the diagonal (8 bytes per determinant), two
vectors of kernel workspace, and the plan.

**Output**: `shci.e`; with one rank also `wfn.h5` (determinants over the
original spinors, coefficients in their phase) and, as requested by `"rdm"`,
`rdm_i_i.h5` and transition `rdm_0_j.h5` in the convention of the other
modes (from the kernel's pair density, a slow reference loop: meant for
moderate spaces).

**MPI** (`src/fci_dist.*`, `src/gas_dist*`): with several ranks every rank
owns a range of A buckets of the plan (a segment of each vector) and a range
of B buckets of the transposed vector. c is transposed into the second
distribution by pairwise exchanges; each half of the kernel then goes
through the source vector slice by slice, each rank reading the slices of
the others one-sidedly (`MPI_Get`, the next while the present one is worked
on) and applying a copy of the kernel's `blas_half` restricted to its own
target rows and the rows of the slice (`src/gas_dist_half.inc`, derived from
`zgasci_blas.inc`, changes marked `dist:`); the result of the transposed
half is transposed back. With `calibrate` (default) one product is timed
per bucket first and the ranges are cut by these times. `slice_gb` bounds a
slice (default: a rank's whole range); two slices are held at a time.
Results agree with one rank to rounding (energies to 1e-11 Eh in the tests).
One rank per node with all its threads is the arrangement to use: one rank
per socket gains 6 % on one node and loses across nodes.

**Measured** (O2 DYVDZ X2C, real, labels 2Ω and parity, block 2Ω = 0 g;
Xeon Gold 6248R, 48 cores, MKL; one sigma, seconds). Nodes of the same type
differ by up to 25 %: the same input took 182 s on some and 235 s on
others, so comparisons below are within one job.

| block | determinants | 1 node | 2 nodes | 4 nodes |
|---|---:|---:|---:|---:|
| 12 in 40 | 3.5e8 | 55.6 | 33.5 | |
| 12 in 44 | 1.16e9 | 182.6 | 99.9 | 70.6 |
| 10 in 52 | 9.4e8 | 298 (slower node) | | 84.9 |

Energies of the Davidson with 3 vectors (tol 1e-6) against zgasci
(`pyscf.lib.davidson1`, 12 vectors): 8 in 48 2Ω = 0 −149.953598673306 (19
products; zgasci −149.95359867330146, 23), 2Ω = 2 −149.953588854049, SDTQ
−150.086940751954, 30 spinors complex 2 roots −149.887068875422 and
−149.828174697142; all within 5e-12 Eh. Where the time of one sigma goes
(12 in 44, thread seconds): dot products of the dilute products 53 %, gathers
14 %, scatter 14 %, matrix products 6 %, axpy 6 %, enumeration of the
operators 5 %; transparent huge pages and interleaved memory change nothing
(235.5 and 238.4 s against 235.6 s), one socket alone takes 1.74 times as
long as two.

**Limits and the plan.** At most 32 sub-groups of at most 16 spinors, 4
label sets, 2^31 strings per side; a side may hold any number of spinors
(the kernels act on the sub-strings of a string; up to 2026-10-16 more than
64 on a side sent the run to the reference kernel). The electrons of a
sub-group are bounded by the constraints without negative weights (`k <=
max / weight`), and the distributions of electrons over the sub-groups of a
side are enumerated within these bounds, at most 1e8 (24 bytes each). The
plan itself takes per rank about 24 bytes per string of each side (plus 16
for the BLAS kernel) and 12 bytes per pair (A bucket, B class) and (B
bucket, A class); the matrices `W` of the nine kinds of terms hold about
9 (n/2)^4 numbers. O2 DYCVTZ SDTQ (fc: 12 electrons in 168 spinors, at most
4 in the 154 virtual ones, 2Ω = 0 g): 2.48e9 determinants, plan 11 s and
28 GB; all-electron (16 in 172): 1.05e10 determinants, plan 17 s and 48 GB.
The fc block on 4 nodes (1 rank x 48 threads each, `slice_gb` 4): 1107 s per
sigma, peak 84 GB on the largest rank with two vectors (calibration product
1481 s before the parts are cut). Such dilute spaces spend their time on
enumerating and sorting operators and on gathers, not in matrix products
(SDT, 3.7e7 determinants, one node: 57 s with `max_subgroup` 16, 68 s with
12).

## Layout

| file | content |
|---|---|
| `src/det.h` | determinants as fixed-width bit strings |
| `src/integrals.*`, `src/hamiltonian.h` | integrals, Slater-Condon rules |
| `src/heat_bath.*`, `src/excite.h` | excitation generation, symmetry and occupation filters |
| `src/connect.h`, `src/connect_groups.h`, `src/connect_bytes.h`, `src/sparse_h.*`, `src/direct_h.h` | connected pairs of determinants, the sparse Hamiltonian, the Hamiltonian applied without storing it |
| `src/davidson.*`, `src/distributed_h.*` | eigensolver, alone and over MPI ranks |
| `src/select.h`, `src/diag.h`, `src/driver_common.h` | the select and diag modes |
| `src/fci.*`, `src/gas_space.*`, `src/fci_davidson.*`, `src/big_array.h` | the fci mode: space, driver, Davidson of few vectors |
| `src/fci_dist.*`, `src/gas_dist.*`, `src/gas_dist_*.inc` | the fci mode over MPI ranks |
| `src/zgasci/` | copy of the zgasci kernel (C99, BLAS) |
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
