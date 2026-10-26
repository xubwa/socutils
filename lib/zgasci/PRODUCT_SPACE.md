# Product spaces in zgasci (design)

A *product space* is the set of determinants `|a, b>` for every A string `a`
of an explicit list `L_A` and every B string `b` of a list `L_B`, with
`n(a) + n(b) = N` and the labels of the occupied spinors adding up to the
target. This is the space of sample-based quantum diagonalization (SQD /
QSCI: the strings come from sampled bit strings) and of TBSCI. It is also
the space of `pyscf.fci.selected_ci`. Non-relativistic: A = alpha, B = beta
spin orbitals, as in `SpatialGASCI`, one electron count per side. Spinors:
A and B are the two halves of the spinors, a side may have strings of
several electron counts, labels 2Omega / parity as now.

## 1. What carries over, what changes

The plan's notions stay: **classes, buckets, blocks**. They become simpler.

* **Class** = the electron count of a string (no sub-groups needed for
  addressing). Non-relativistic: one class per side.
* **Bucket** = the strings of one class with one label tuple. Within a
  bucket, strings are kept in increasing order as integers (word by word,
  the highest word first). Without labels, the CI vector of a
  non-relativistic product space is the matrix `C[a][b]` over the sorted
  lists, exactly the layout of `pyscf.fci.selected_ci` (same phase
  convention: alpha creators first, each string in increasing order).
* **Block** = (A bucket, the B bucket of complementary count and label).
  It holds *all* present A strings of the bucket times *all* present B
  strings of the matching bucket. So blocks are dense, and the gather –
  GEMM – scatter of the BLAS kernel applies unchanged. Strings whose
  bucket pairs with nothing are dropped when the plan is made.
* **Sub-groups** are kept only as a formality: each side is cut into
  pseudo-sub-groups of at most 16 spinors, so that the per-spinor labels,
  the nine kinds of terms and the W matrices are set up by the existing
  code. They play no part in the numbering.

## 2. Numbering and look-up

The string number (`sid`) is the position in the sorted per-side list
(class, then bucket, then value). Each string is stored once as
`ceil(n_side / 64)` 64-bit words.

Strings are looked up only **when the plan is built**, never in a sigma.
Look-up uses a hash table keyed by the words. It is open addressing with
linear probing, `2^k >= 2 M` slots of 4 bytes, and expected O(1).

Alternatives that were not chosen:
* A dense map over the class (the present scheme: a mixed radix of
  sub-string ranks, `bp[sid]` for every string of the class). For SQD lists
  it holds about `C(n, k)` entries for `M ~ sqrt(dim)` present strings: 9e7
  entries for `[4Fe-4S]` (36 orbitals, 27 alpha) and 3e13 for 58 orbitals
  with 15 electrons. Impossible in general.
* Binary search in the sorted list. It costs O(log M), with a cache miss at
  each step, and is about 5 times slower than hashing in the build. Nothing
  else needs it.

## 3. Operators: string-level tables, not placements with a filter

The combinatorial kernels enumerate the operators of a string sub-group by
sub-group and compute the source string number arithmetically. Every result
exists, because all strings of a class are present. In a product space
almost all of them are absent. For example, with N2/cc-pVDZ (26 orbitals,
5 alpha) and 6300 strings out of 65780, about 10 % of the double
excitations of a top-weighted string are present, and fewer for random
subsets. Each absent one would need a look-up to reject it.

The column operators are also enumerated again for every piece of rows of
the BLAS kernel. That is `pieces x M x D` enumerations (about 1e10 per sigma
in the example).

**Choice (as in SBD and selected_ci): per-side excitation tables,
precomputed once when the plan is made.** For each side, each operator
*pattern* (the operators of one side in one of the nine kinds, e.g.
`a+ a+ a a` within the side, `a+ a`, `a+ a+ a`, `a`, ...; up to 8 per side)
and each target string `w`, the table lists the operators `x` with
`O(x)^dagger |w> = ± |source>` and the source present: `(x, source sid,
sign)`. This is exactly what `zg_asg_enum` returns, minus the absent
strings.

* Storage is CSR: `ptr[M + 1]`, then `src` (int32) and `x` (int32, with the
  sign as `~x`), 8 bytes per entry.
* Building enumerates `M x D` candidates per pattern and does one hash
  look-up each (parallel over strings, two passes: count, then fill).
  `D = C(k, 2) C(n - k + 2, 2)` for `a+ a+ a a`. Patterns that change the
  electron count to a count with no string skip the enumeration: in the
  non-relativistic case only `a+ a+ a a` and `a+ a` are built.
* The tables are bounded by `M x min(M, D)` entries per pattern. For
  M = 6300, N2 (5 alpha in 26) this is at most 2.8e7 entries, 0.2 GB per
  side. Typically a few times less.
* Limit: `n_side^k < 2^31` for the x of a pattern (n_side <= 215). The W
  matrices bound n_side far below that in practice.

The BLAS kernel (`zgasci_blas.inc`) is unchanged except at three points:
* `zg_asg_list`: in a product plan a "placement" is just the source class,
  `count(target) + add`.
* The row operators of a row come from the table of its pattern instead of
  `zg_asg_enum`.
* So do the column operators of a target column.

The rest stays the same:
* sorting by source bucket, runs of 256 source columns;
* the gather `D[x][c]`, the GEMM `F = W D` (or dot products where F would be
  mostly unused), and the scatter;
* pieces of at most 256 rows of a row bucket, each written by one thread
  (no atomics);
* the transposed vector for the four kinds with B strings as rows.

The column operators are still collected per piece. Now that is a copy
from the table, so its cost is `pieces x (table entries of the target
columns)`.

The label filter of the row operators is not used. Operators whose source
bucket pairs with no candidate column bucket fall out at the candidate step,
as they do now.

**Kinds without column operators** ((AA,AA), and (BB,BB) on the transposed
vector) are, in a product space, `sigma = H_AA C`: a sparse matrix between
the strings of the row side, with typically 4–15 % of its elements nonzero,
times the vector as a matrix. One AXPY of a whole row per operator, as the
combinatorial kernel does, is limited by memory bandwidth. For example,
6325 x 6325 N2 strings took 80 thread-s in AXPYs, about half of the sigma.

Instead (`blas_rows`):
* Tiles are 128 rows of a row bucket x 256 columns of one of their blocks,
  each written by one thread.
* For each source bucket, the operators of the tile's rows are summed into
  a dense slab `H(rows, source strings)`, and the GEMM
  `slab x C(source rows, tile columns)` follows. This is done when the slab
  has at least one entry in 16 of its elements (`ZGASCI_ROWS_DENSE`) and at
  most 2^21 elements.
* Otherwise it is one AXPY per *distinct* (row, source row). Operators
  that connect the same two strings are summed first: the diagonal ones,
  and the single excitations with each spectator.

Thresholds (measured):
* `ZGASCI_ROWS_DENSE`: 8–16 is best or near best everywhere.
* The GEMM / dot-product threshold of the other kinds is 16 for product
  plans (4 for the combinatorial ones). Between 4 and 16 it gains 16 % on
  the N2 "top" space, and nothing is lost elsewhere.

## 4. hdiag, density matrices

* **hdiag**: a determinant at a time, from the stored words.
* **Pair density** `g[P,R] = <bra|A+_P A_R|ket>` (`zg_rdm2_blas`, used for
  product plans; it also works for the combinatorial ones). This is the
  BLAS kernel in a second mode, the derivative of sigma with respect to W.
  The same loops accumulate
  `G[x][y] += conj(bra[target]) s_x s_y ket[source]` into a per-thread
  W-shaped array of each kind (allocated on first use), always by dot
  products.
  * Map back: `g[P,R] = sign_kind * G[x][y]`, the inverse of `term_matrix`.
  * The one-particle density is its partial trace, as now. Spin-resolved
    1- and 2-RDMs and `spin_square` follow in Python exactly as for
    `SpatialGASCI`.
  * Memory: per thread, the W arrays of the kinds met, `(n_side)^4` numbers
    at most each. For 26 orbitals that is 3.6 MB for each of the two
    non-relativistic kinds.

## 5. Memory

* Per string: `8 ceil(n_side/64)` (words) + 24 (bucket, place, bucket list,
  bucket-and-place) + 8 per table entry.
* Per determinant: the vectors only. Davidson keeps `max_space` + a few.
  While it runs, the BLAS kernel needs two more vectors for the transposed
  half.

## 6. Spin flip (M_s = 0, identical lists): implemented

`zg_sigma_flip`, `ProductSpaceCI.spin_flip = ±1`. The rest of this section
is the design as written before. Measured gain: 18 % (N2 "top", 4e7),
31 % (N2 random), 10–12 % (Fe–S-like).

If `L_A = L_B` (same strings, same labels per spatial orbital) and
`N_alpha = N_beta`, the spin flip maps `C[a][b]` to `C[b][a]`. Eigenvectors
satisfy `C^T = (-1)^S C`. Then the (BB,BB) part of sigma is `±(X)^T`, where
X is the (AA,AA) part. So half 1 of the kernel, both transpositions and the
transposed workspace are not needed:

    sigma = X ± X^T + (AB,AB)(C)

as `pyscf.fci.direct_spin0` does. The (AB,AB) kind itself is symmetric
under the flip and could be halved by computing only blocks on one side of
the diagonal.

This fits the plan directly. With identical lists the A and B buckets
coincide, and the transposed layout of a mirror plan is the native layout
of the transpose. It needs:
* a check that the lists and labels mirror each other;
* a flag that runs half 0 kinds {AA,AA} into X and {AB,AB} into sigma, then
  adds `±X^T`;
* the Davidson to stay in the symmetric subspace (symmetrize the guess and
  each new vector, as `direct_spin0` does).

As built:
* `zg_is_mirror` is set when the plan is made.
* `zg_sigma_flip` checks that only (AA,AA), (AB,AB) and (BB,BB) are
  nonzero and that W of (BB,BB) equals W of (AA,AA). Otherwise it returns 0
  and the driver uses the full sigma.
* `zg_flip_vector` is the parallel exchange of the sides (the transposition
  of the kernel).
* The driver symmetrizes the guess, the input of sigma, *and the output of
  the preconditioner*. Without the last one, the antisymmetric rounding
  noise, which the projected sigma does not see, grew by two orders of
  magnitude per iteration once the residual was small (measured), and the
  Davidson stalled: 18 instead of 12 iterations at 1e6.

Measured gain: see the top of this section.
This is the same structure as the Kramers plan in README.md (stage 1). For
spinors, time reversal pairs A and B halves in the same way when `L_B` is
the time-reversed image of `L_A`.

## 7. What stays for later

* **(AB,AB)** is the larger half of a sigma. Its GEMMs do about 9 times
  the needed multiply-adds: F is formed for every y of a run of 256 source
  columns. The column operators are collected again for every piece of
  rows, 12 % of the time, although in a product space they could be sorted
  once per target block and shared.
* **Davidson**: at 4e7 determinants the numpy vector algebra of
  `davidson1` takes 60 % of a run. A low-memory Davidson in C, such as
  zshci's, would remove most of it.

* **MPI / distributed vectors** (zshci sync): the unit of work (rows of a
  bucket) and the source reads per piece are unchanged. The string tables
  are per side and small, so they are replicated on each rank.
* **GPU**: the tables make every step a gather / GEMM / scatter over
  integer lists. There is no recursion, so the kernel ports well.
