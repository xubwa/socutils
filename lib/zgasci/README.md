# zgasci

Full CI and generalized-active-space CI (GAS, with RAS and unrestricted FCI as
special cases) over spinors, as kernels in plain C99 + OpenMP + BLAS (GEMM
and AXPY only), driven from Python by `socutils.hci.gasci`. No other
dependency: no LAPACK, no HDF5, no input files. As in PySCF's FCI, C only
applies the Hamiltonian and builds density matrices; the CI vectors live in
numpy arrays and are diagonalized with `pyscf.lib.davidson1` (complex
Hermitian or real symmetric).

Build: `make` at the top of socutils (lib/CMakeLists.txt; BLAS as for the
other libraries, e.g. `make CMAKE_ARGS=-DBLAS_LIBRARIES=mkl_rt`), or directly

    gcc -std=c99 -O3 -funroll-loops -fcx-limited-range -fopenmp -fPIC -shared \
        -o lib/libzgasci.so lib/zgasci/zgasci.c -lmkl_rt -lm

(any BLAS with the Fortran interface and 32-bit integers).
`SOCUTILS_ZGASCI_LIBRARY` overrides the library path. BLAS is called from
inside the kernel's own parallel region and must run sequentially there: the
kernel asks MKL (`MKL_Set_Num_Threads_Local`) or OpenBLAS
(`openblas_set_num_threads_local`) for one thread in each of its threads and
restores the setting; measured CPU time over wall time is 15.8–15.9 on 16
threads, so there is no oversubscription.

## Algorithm

* **Sub-groups.** The spinors (in the order of the kernels) are cut into
  consecutive sub-groups of at most 16; the first ones form side A, the others
  side B. A determinant is an A string times a B string, a string a sub-string
  of each sub-group of its side. The Python driver reorders the orbitals so
  that every constrained set is a union of sub-groups, half of each set on
  each side, sub-groups of at most `max_subgroup` (default 12) spinors.
* **Classes, buckets, blocks.** The class of a string is its number of
  electrons in each sub-group; within a class strings are numbered by the
  mixed radix of their sub-string ranks. Strings of a class are split into
  buckets by their symmetry labels (several additive label sets, each with an
  optional modulus). A block is (A bucket, the one bucket of a B class with the
  matching label), a dense matrix (A string, B string); the CI vector is the
  blocks one after another. Which classes pair is decided by the electron
  number and the constraints "electrons in this union of sub-groups within
  [min, max]". Tables are per sub-group only (2^m entries); neither
  determinants nor H are stored.
* **Hamiltonian.** `H = sum_{P,R} v[P,R] a+_p a+_q a_s a_r` over pairs
  `P = (p<q)`, `R = (r<s)`, `v` the antisymmetrized integrals with the
  one-electron part folded in (divided by N - 1), built by Python.
* **Reference kernel.** For each determinant K of N - 2 electrons, gather the
  coefficients of the determinants K + R in the space (with their signs),
  `e = v d`, scatter `sign * e[P]` to K + P (atomic adds). Electrons passing
  between A and B need no special case. The pair density
  `<bra|A+_P A_R|ket> = sum_K conj(d_bra[P]) d_ket[R]` is the same loop; dm1
  follows by contraction. The determinants K come in tasks (pairs of N - 2
  electron tuples, larger ones cut) under dynamic scheduling.
* **Block kernel** (`zg_sigma_block_{d,z}`, `sigma_kernel='block'`). Each
  term `v[P,R] a+_p a+_q a_s a_r` falls in one of nine kinds by whether P and
  R lie within A, across, or within B; each kind is exactly
  `sign (-1)^(k_B N_A) <a'|O_A(x)|a> <b'|O_B(y)|b>` with `W[x][y] = ±v[P,R]`
  (one, two electrons passing between the sides are kinds of their own, as
  in Olsen's LUCIAREL). sigma is built by tiles (rows x columns of a target
  block), each written by the thread that owns it: no atomic additions, the
  unit of a later MPI distribution. Strings are 64-bit words on each side
  (each side at most 64 spinors); excitations are done on the words in the
  loop, with only the per-sub-group tables, and no string-level excitation
  table is stored. Tiles are weighted from the classes for the dynamic
  schedule. `ZGASCI_TERMS` (bit mask) restricts the kinds,
  `ZGASCI_STATS` prints their multiply-adds.
* **BLAS kernel** (`zg_sigma_blas_{d,z}`, `sigma_kernel='blas'`, the
  default; design note at the top of `zgasci_blas.inc`). The same nine kinds,
  each done as gather – matrix product – scatter in the way of Knowles and
  Handy / Olsen: for a target row string and a run of 256 source column
  strings, `D[x][c] = ±C[row(x)][c]` is gathered for the row operators `x`
  valid on the row, `F = W D` is one GEMM, and `F[y][c]` is scattered to the
  column `O(y) c`. Only the scatter is indirect, one addition per connected
  column string. Since electrons pass between the sides, the row side is
  chosen per kind: the side with more operators, and for `a+ a+ ⊗ a a` the
  side of the creation operators. Five kinds have A strings as rows; the
  other four are their mirror images and run by the same routine on the
  transposed vector (two vectors of workspace). (AA,AA) and (BB,BB) are
  AXPYs of whole rows. A piece of work is up to 256 rows of an A (B) bucket
  with all their blocks, written by its thread alone. Before any string is
  touched, the operators are placed in the sub-groups in every way that can
  lead to a block of the space (classes, i.e. the constraints), then
  enumerated placement by placement, the string numbers from the ranks of
  the sub-strings; row operators also pass a label filter. Column operators
  are found once per piece and shared by its rows. A product computes
  `F[y][c]` for all `y` that occur in the run, though each column takes only
  some; where that would be more than four times what is needed (dilute
  spaces, small blocks), the needed elements are worked out as dot products
  over `x` instead. `ZGASCI_STATS` prints multiply-adds and thread seconds
  per step; `ZGASCI_BLAS_COLS`, `ZGASCI_BLAS_DIRECT`, `ZGASCI_BLAS_ROWS` set
  the run length, the threshold and the rows of a piece.
* **Phase.** A determinant is `a+_p1 a+_p2 ... |0>`, `p1 < p2 < ...` in the
  order of the kernels; `GASCI.wavefunction` converts to the original order.

## Interface

C (`zgasci.h`): `zg_plan_new` / `zg_plan_free`, `zg_dim`, `zg_dets`,
`zg_info`, and `zg_hdiag_{d,z}`, `zg_sigma_{d,z}`, `zg_sigma_block_{d,z}`,
`zg_sigma_blas_{d,z}` (both with `zg_block_ok`), `zg_rdm2_{d,z}` (real and
complex from one source: `zgasci_kernel.inc`, `zgasci_block.inc` and
`zgasci_blas.inc` compiled twice).

Python (`socutils.hci.gasci`):

* `GASCI(mol)` — spinors; `kernel(h1e, eri, norb, nelec, ci0, nroots, ecore)`
  with `eri[p,q,r,s] = (pq|rs)`, `make_rdm1`, `make_rdm2`, `make_rdm12`,
  `trans_rdm1`, `trans_rdm12` with the conventions of `SHCI` and
  `zfci.FCISolver` (`dm1[p,q] = <p^+ q>`, `dm2[p,q,r,s] = <p^+ r^+ s q>`,
  `t[p,q] = <bra|p^+ q|ket>`); a fcisolver of `socutils.mcscf.zmcscf.CASSCF`.
  Attributes: `gas` and `restrictions` (lists of `dict(orbitals, min, max)`),
  `orbsym` / `orbsym_mod` (as SHCI), `target`, `max_subgroup`, `subgroups`
  (explicit `(groups_a, groups_b)`), Davidson settings (`conv_tol`,
  `conv_tol_residual`, `max_cycle`, `max_space`, `level_shift`), `direct_dim`
  (spaces up to this size are diagonalized as a full matrix), `sigma_kernel`
  (`'blas'`, the default, `'block'` or `'reference'`).
* `SpatialGASCI(mol)` — non-relativistic, real spatial orbitals, the
  interface of `pyscf.fci.direct_spin1` (`nelec` as `(na, nb)` or a number
  with `spin`; spin-traced and spin-resolved density matrices, `spin_square`,
  `orbsym` with D2h-subgroup irreps, `wfnsym`); a fcisolver of
  `pyscf.mcscf.CASSCF`. Side A holds the alpha, side B the beta spin orbitals,
  N_alpha is one more constraint, constraints name spatial orbitals.
  `trans_rdm1` follows pyscf: `t[p,q] = <bra|q^+ p|ket>`.
* `Plan` — the C plan: `dim`, `dets()`, `occslst()`, `hdiag(v)`,
  `sigma(v, c, kernel='blas')`, `rdm2_pairs(bra, ket)`; `antisymmetrized(h1, eri, n, nelec)`
  builds `v`.

## Limits

At most 16 sub-groups of at most 16 spinors (so 256 spinors), 4 label sets,
2^31 strings per side, 4·10^6 electron distributions per side. Vectors are in
memory; OpenMP only; time reversal is not used (plan below). The reference
kernel does `sum_K n_conn(K)^2` multiply-adds (`n_conn` up to
C(n - N + 2, 2)). The block and the BLAS kernel need each side to hold at
most 64 spinors (otherwise the reference kernel runs) and build the matrices
`W` of the nine kinds at every call: nine arrays of about (n/2)^4 numbers,
0.02 s and 33 MB for 52 spinors with real integrals (measured), but 0.8 GB
real or 1.6 GB complex for 116 spinors (from the sizes); for that many
spinors they should be kept between calls or stored by label (not done). The BLAS kernel needs two more vectors of
workspace while it runs, and per thread the operators of one piece of work
(megabytes).

## Tests and measured timings

`python hci/test/test_gasci.py` (26 tests; `ZGASCI_TEST_KERNEL=blas`, `block`
or `reference` chooses the kernel of the solvers, all 26 pass with each; the
exhaustive tests check the three kernels): kernels against H built from the
operators on bit strings (≤ 8 spinors; real and complex; FCI, GAS, labels with
and without modulus; several divisions), the BLAS kernel against the
reference kernel element by element to 1e-10 in spaces of 10–12 spinors
(real and complex; FCI; constraints, also across the sides and weighted;
labels with and without modulus, with and without integrals that conserve
them; several divisions; 2 to 9 electrons; with the matrix products forced,
the dot products forced, and the default choice), against zfci,
zfci.kernel_dets and SHCI (FCI, GAS by occslst and by restrictions in a
selection, labels), the same problem under five divisions to 1e-10, pyscf
`direct_spin1` (closed and open shell), `direct_spin1_symm` (C2v),
SpatialSHCI (GAS), pyscf CASSCF and zmcscf CASSCF (against SHCI as solver,
6e-11 Eh).

### One sigma: reference, block and BLAS kernel

All numbers measured (wall seconds, best of the repeats in one process) on
Xeon Gold 6448Y nodes shared with other jobs (repeated runs differ by up
to 10 %); MKL (`mkl_rt`), gcc 9.3 `-O3`. "Random": dense random integrals,
12 electrons, no labels. "O2": the integral files of the O2
zero-field-splitting work, 12 electrons, labels 2Ω (no modulus) and parity,
block 2Ω = 0 (parity g where the file has it): full CI in 30 and 32 spinors;
52 spinors with spinors 0–3 doubly occupied ("8 in 48", full CI of the rest)
and with at most k excitations out of the (2 electrons, 4 spinors) reference
("SD", "SDT", "SDTQ": many small blocks). The integrals of the O2 files are
real to 1e-10; "complex" is the complex kernel on the same integrals.
The last column is reference time over BLAS time. The largest element of
the difference between the kernels' results, relative to the largest
element, is at most 3e-15 in every row.

16 threads:

| case | determinants | scalar | reference | block | BLAS | ref / BLAS |
|---|---:|---|---:|---:|---:|---:|
| random, 20 spinors | 125 970 | complex | 0.053 | 0.047 | 0.026 | 2.0 |
| random, 20 spinors | 125 970 | real | 0.032 | 0.036 | 0.015 | 2.1 |
| random, 22 spinors | 646 646 | complex | 0.302 | 0.328 | 0.132 | 2.3 |
| random, 22 spinors | 646 646 | real | 0.177 | 0.200 | 0.063 | 2.8 |
| random, 24 spinors | 2 704 156 | complex | 1.51 | 2.28 | 0.671 | 2.2 |
| random, 24 spinors | 2 704 156 | real | 0.788 | 1.30 | 0.289 | 2.7 |
| O2, 30 spinors, full CI | 13 997 577 | complex | 8.19 | 27.33 | 3.17 | 2.6 |
| O2, 30 spinors, full CI | 13 997 577 | real | 5.96 | — | 1.56 | 3.8 |
| O2, 32 spinors, full CI | 36 649 008 | complex | 24.03 | — | 10.25 | 2.3 |
| O2, 32 spinors, full CI | 36 649 008 | real | 16.35 | — | 4.96 | 3.3 |
| O2, 8 in 48, full CI | 23 829 426 | complex | — | — | 10.84 | — |
| O2, 8 in 48, full CI | 23 829 426 | real | 84.71 | 139 | 7.79 | 10.9 |
| O2, 52 spinors, SDTQ | 9 883 313 | complex | 22.64 | — | 5.67 | 4.0 |
| O2, 52 spinors, SDTQ | 9 883 313 | real | 20.94 | 164 | 4.71 | 4.4 |
| O2, 52 spinors, SDT | 603 938 | real | 1.66 | 10.74 | 0.404 | 4.1 |
| O2, 52 spinors, SD | 18 942 | real | 0.073 | 0.328 | 0.034 | 2.1 |

1 thread:

| case | determinants | scalar | reference | block | BLAS | ref / BLAS |
|---|---:|---|---:|---:|---:|---:|
| random, 20 spinors | 125 970 | complex | 0.594 | 0.588 | 0.217 | 2.7 |
| random, 20 spinors | 125 970 | real | 0.333 | 0.381 | 0.106 | 3.1 |
| random, 22 spinors | 646 646 | complex | 4.09 | 4.97 | 1.52 | 2.7 |
| random, 22 spinors | 646 646 | real | 2.17 | 2.97 | 0.615 | 3.5 |
| random, 24 spinors | 2 704 156 | complex | 22.00 | 33.89 | 8.77 | 2.5 |
| random, 24 spinors | 2 704 156 | real | 11.43 | 19.07 | 3.42 | 3.3 |
| O2, 30 spinors, full CI | 13 997 577 | complex | 127 | — | 46.28 | 2.7 |
| O2, 30 spinors, full CI | 13 997 577 | real | 92.41 | — | 23.22 | 4.0 |
| O2, 32 spinors, full CI | 36 649 008 | complex | 374 | — | 151 | 2.5 |
| O2, 8 in 48, full CI | 23 829 426 | real | 1264 | — | 118 | 10.7 |
| O2, 52 spinors, SDTQ | 9 883 313 | real | 321 | — | 68.72 | 4.7 |

24 threads were not measured (the 48-core job for it did not start in
time). From 1 to 16 threads the BLAS kernel gains a factor 13.1 (random, 24
spinors, complex), 14.6 (O2, 30 spinors, complex; SDTQ) and 15.1 (8 in 48).

Where the time goes (`ZGASCI_STATS`, thread seconds): for the dense random
case the matrix products are about half (3.3e10 complex multiply-adds at
0.15 ns each, against 1.6e10 at 1.5 ns in the reference kernel: twice the
arithmetic, ten times the speed), the rest gathers, AXPYs and the scatter.
For 8 in 48, where a string pairs with few strings of the other side, the
products are a tenth or less; the enumeration of the operators (a third),
the gathers and the dot products (latency of the memory) take most of it.
The gain there (11×) comes less from BLAS than from finding the connections
per string and per class instead of per determinant. The block kernel is
the slowest of the three wherever blocks are many (it tries every operator
on every string and looks the result up).

### Energies with the BLAS kernel

Davidson (`pyscf.lib.davidson1`), 16 threads, same settings as the runs with
the reference kernel; energies in Eh.

| case | determinants | E (BLAS kernel) | E (reference kernel) | sigma calls | s per sigma: reference → BLAS | whole run / s: reference → BLAS |
|---|---:|---:|---:|---:|---:|---:|
| 8 in 48, 2Ω = 0, real | 23 829 426 | −149.95359867330146 | −149.95359867330146 | 23 | 89.5 → 8.0 | 2107 → 234 |
| 8 in 48, 2Ω = 2, real | 22 699 070 | −149.95358885404800 | −149.95358885404795 | 16 | 88.0 → 7.6 | 1443 → 158 |
| SDTQ, 2Ω = 0, real | 9 883 313 | −150.08694075195103 | −150.08694075195100 | 28 | 21.0 → 4.8 | 615 → 167 |
| SDTQ, 2Ω = 2, real | 9 353 001 | −150.08693088201204 | −150.08693088201207 | 19 | 20.7 → 4.6 | 411 → 106 |
| 30 spinors, 2Ω = 0, complex, 2 roots | 13 997 577 | −149.88706887542185 | −149.88706887542185 | 33 | — → 3.3 | 618 → 422 |
| 30 spinors, 2Ω = 2, complex, 2 roots | 12 931 149 | −149.88705589361200 | −149.88705589361203 | 35 | — → 3.0 | 1343 → 430 |
| 32 spinors, 2Ω = 0, complex, 2 roots | 36 649 008 | −149.89572450083006 | −149.89572450083003 | 34 | — → 10.7 | 1638 → 1228 |
| 32 spinors, 2Ω = 2, complex, 2 roots | 33 839 520 | −149.89571158147655 | −149.89571158147655 | 37 | — → 9.7 | 1706 → 1235 |

The splitting E(2Ω = 2) − E(2Ω = 0) of the 8-in-48 space is 2.1551 cm⁻¹ with
either kernel. With the BLAS kernel the sigma calls are no longer most of a
run with long complex vectors: in the 32-spinor run they take 363 s of
1228 s; the rest is the vector algebra of the Davidson routine in numpy
(two roots, up to 20 vectors of 0.6 GB).

## Plan: time-reversal (Kramers) symmetry

Not implemented. What follows is the plan.

**The symmetry.** Without a magnetic field H commutes with the time-reversal
operator K (antiunitary). Spinors come in Kramers pairs (p, p̄ = K p) with
K p̄ = −p, so `K a+_p K⁻¹ = a+_p̄` and `K a+_p̄ K⁻¹ = −a+_p`, and K² = (−1)^N on
N-electron states.

**Pairing of determinants, N even.** Put one spinor of every pair on side A
and its partner on side B, in the same order and with the same sub-groups
(the driver already puts half of every GAS set on each side; the halves
become "unbarred" and "barred"). An A string a and a B string b are then
both sets of pairs, and

    K |a, b> = (−1)^(k_B (k_A + 1)) |b, a>

(k_B signs from the barred spinors, k_A k_B from putting the unbarred ones
first again). For even N the sign is +1 whatever k_A: K exchanges the two
strings. Since K² = 1, every eigenvector can be chosen with K ψ = ψ, that
is `C[b, a] = conj(C[a, b])`: **the coefficient matrix over (A string, B
string) is Hermitian**, and so is sigma. The determinants with a = b have
real coefficients. The space must be mapped to itself: constraints
symmetric under the exchange of the sides, labels with Ω(p̄) = −Ω(p) and
equal parity, target 2Ω = 0. (K maps the block of Ω to the block of −Ω; for
Ω ≠ 0 it relates two blocks and halves nothing within one, the block of −Ω
is simply never computed, as now. For odd N, K² = −1, no vector is its own
image, and there is nothing to halve.)

**What it halves.**

1. *The sigma build.* K maps each kind of term to its mirror image:
   (AA,AA) ↔ (BB,BB), (AA,AB) ↔ (BB,AB), (AB,AA) ↔ (AB,BB), (AA,BB) ↔
   (BB,AA), and (AB,AB) to itself. These are exactly the two halves of the
   BLAS kernel. With S the result of the four kinds of half 0 other than
   (AB,AB) and T that of (AB,AB), `sigma = S + S† + T`, and T is Hermitian by
   itself, so only its blocks on one side of the diagonal are needed. Half 1,
   the transposed copy of C and both transpositions disappear (one
   conjugate-transpose-add of the result replaces them), and (AB,AB) is done
   for half of the target blocks.
2. *The vectors.* Only the blocks (A bucket, B bucket) on one side of the
   diagonal are kept, and of the diagonal blocks one triangle: dim real
   numbers instead of dim complex ones. In that packed form H is a real
   symmetric matrix of order dim (basis `(|a,b> + |b,a>)/√2` and
   `i(|a,b> − |b,a>)/√2`), so the Davidson vectors, the diagonal and the
   subspace matrices are real: half the memory and half the vector algebra.
3. *Real integrals.* When the real kernel applies (as for O2 in the
   symmetry-adapted spinors used here), K acts on real vectors as C → Cᵀ, a
   linear symmetry: the eigenvectors are symmetric or antisymmetric, the
   block 2Ω = 0 splits into two blocks of half the size each (for a linear
   molecule presumably the 0+ and the 0− states; to be checked), and each
   is found separately with vectors half as long.

**What the addressing needs.** A check in `zg_plan_new` (asked for by a
flag) that the plan is symmetric: the same sub-group sizes on both sides,
labels of side B the images of those of side A, constraints and targets
invariant. Then an A string and the B string with the same sub-strings have
the same class tuple and the same number within the class, so the mirror of
a block is found from a table bucket ↔ bucket and the transposition needs no
look-up of strings. For packed vectors: block offsets for the blocks on one
side of the diagonal and a rule for the diagonal blocks; a routine that
expands a packed vector into the rows a piece of work reads (or the gathers
read `conj(C[b, a])` for the missing blocks, which keeps the rows
contiguous only on one side: expanding is simpler and is the first step).

**What the kernel needs.** Stage 1 (time only, full vectors, the Python
interface unchanged but for a flag): skip half 1, restrict (AB,AB) to the
blocks with A bucket ≥ mirror of the B bucket, and add the conjugate
transpose; the rows are still written by the thread that owns them, the
last step block by block. Stage 2 (memory): packed vectors in the driver
and a real Davidson; the density matrices (reference kernel) from the
expanded vector. Stage 3: the split into symmetric and antisymmetric
vectors for real integrals.

**What the driver needs.** The pairing and the phases: the two spinors of a
pair must be exact images, `v` then satisfies `v[P̄, R̄] = ± conj(v[P, R])`
with the signs of the barred indices. Spinors from a Kramers-restricted
calculation have this; in general the driver must be given the pairs (or
the matrix of K in the spinor basis), fix the phases, and verify the
relation on `h1` and `eri` to 1e-10, refusing otherwise. Labels alone do not
determine the pairing.

**Expected gain** (estimates from the measured shares of the present
kernel, not measurements). Stage 1 leaves half 0 less half of (AB,AB) plus
one pass over the result. Measured now, 16 threads, whole sigma / half 1
with its transpositions / half 0 / (AB,AB) alone: O2 30 spinors complex
3.31 / 1.35 / 1.96 / 0.71 s, real 1.68 / 0.68 / 1.00 / 0.41 s; 8 in 48 real
8.60 / 2.94 / 5.64 / 1.53 s; random 24 spinors complex 0.72 / 0.30 / 0.42 /
0.18 s. So stage 1 should make a sigma of the block 2Ω = 0 about 2.1 times
faster for the full-CI cases with 30 spinors and for the dense case, and
about 1.8 times for 8 in 48. The vectors and the
vector algebra of the Davidson routine halve as well in stage 2, which
matters as much now (see the 32-spinor run above). For real integrals stage
3 halves the dimension on top of the real kernel. Nothing is gained for the
blocks with Ω ≠ 0 or for odd N.
