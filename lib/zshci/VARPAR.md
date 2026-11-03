# Multi-node efficiency of the variational loop

Notes on where the time of selection, Hamiltonian build and Davidson goes
on several nodes, what was changed, and what would need a redesign.
Measured on Cr2 24e/30o def2-SVP, spin_flip, 5.8e7 determinants (2.9e7
functions) in the last iteration, one rank of 48 threads per node; stage
timings from `ZSHCI_PROFILE=1`.

## Findings

- **Products.** Every rank holds the rows of 1/p of the determinants but
  vectors of full length. `SparseH` used the one-sweep product only while
  its thread vectors (threads x n) took at most a quarter of the stored
  elements; on a rank that holds 1/4 of them this failed, and every rank
  read its matrix twice (two-sweep product). That, not communication, was
  most of the 1.3x speed-up of Davidson on 4 nodes. The merge of the
  thread vectors went one row range per thread, so that the first thread
  summed 48 vectors and the last one: 0.1 s per product on every rank.
- **Communication** is collective already (leader `MPI_Bcast` of x,
  `MPI_Reduce` of y). One `MPI_Reduce` of 2.9e7 doubles took 215 ms on 4
  nodes, the same in pieces of 2^19 doubles 72 ms (Intel MPI 2021.6;
  `Reduce_scatter` + `Gatherv` 106 ms, `Allreduce` 92 ms).
- **Selection**: the root sorted the space serially (`ShareSortedActive`),
  every rank its candidates and the gathered new determinants, and the
  root its spin-flip pairs (`AppendPairs`), all with serial `std::sort`.

## Changes

1. Stage timings (`src/profile.h`, `ZSHCI_PROFILE`), per-rank product
   breakdown (local, share x, sum y).
2. With a memory budget (`"hamiltonian": "adapt"`) the one-sweep product is
   used whenever its vectors take no more than the elements and fit the
   budget; balanced, blocked merge of the thread vectors (same summation
   order); `SumToRoot` in pieces of 2^19 doubles.
3. Thread-parallel sorts (`src/parallel_sort.h`) in the places above; the
   full-length sum of left-out row flags only when some rank left a row out.

Energies agree with the previous code to 5e-12 in every iteration, with the
same determinant counts, for 1, 2, 3 and 4 ranks.

## Last iteration, 5.8e7 determinants

| nodes | | t_sel | t_ham | t_dav | per product | peak RSS / rank |
|---|---|---|---|---|---|---|
| 1 | before | 14.2 | 42.7 | 87.4 | 2.82 s | 70.0 GB |
| 1 | after  |  9.1 | 45.0 | 85.2 | 2.75 s | 70.0 GB |
| 2 | before | 12.6 | 38.5 | 119.0 | 3.84 s | 36.6 GB |
| 2 | after  |  6.8 | 26.7 | 62.4 | 2.01 s | 40.5 GB |
| 4 | before |  8.6 | 28.3 | 71.9 | 2.32 s | 21.1 GB |
| 4 | after  |  4.5 | 18.4 | 44.7 | 1.44 s | 25.6 GB |

(1 node: both binaries on the same node, one after the other; node to node
the times vary by about 10%. The extra memory on 2 and 4 nodes is the
thread vectors of the one-sweep product.)

## What is left, and would need a redesign

- **Local products are still 1.4x the ideal 1/p** (4 nodes: 1.06 s per
  product against 0.70 s for a quarter of one node). Each rank's rows are
  every p-th chunk of strings, so its accesses to x and to its thread
  vectors spread over the whole vector with 1/p of the reuse, and the
  thread vectors are zeroed and summed over full length on every rank.
  Contiguous, element-balanced row ranges per rank would confine both
  (needs an estimate of the elements per row before the search, or a
  re-distribution after it).
- **The full-length sum of y on the root** (0.16 to 0.23 s per product on 4
  nodes, growing with n) and the eigensolver's vector work on the root only
  would go with Davidson vectors distributed over the ranks: products
  reduce-scattered to the owners and x all-gathered, the solver's dot
  products and updates done per segment (the few-vector solver already
  takes a `VectorLayout`).
- **Element values are worked out twice** in the build: once in the
  connection search (for the threshold) and again in `FillBlockValues`
  (2.9 s of 18 on 4 nodes, 8 s of 41 on 1 node). Keeping them from the
  search needs the memory budget to count them up front.
- **Root-side stages of the build** that do not shrink with nodes: new
  list and pairs, diagonal, string tables, together 4.6 s on 4 nodes.

# Round 2: memory first, toward 1e9 to 1e10 determinants

## Fe4S4 (54e,36o) measured, one node

Natural orbitals of the SQD run (`fcidump_run3`), broken-symmetry start,
schedule 1e-3 / 3e-4 / 1e-4 / 5e-5 / 2e-5, `"adapt"` with 160 GB, real,
W = 2, no spin_flip. 1.006e7 determinants, E = -327.215992, 799 s.

| | per determinant | at 1.006e7 |
|---|---|---|
| stored H (lower triangle, 4 + 8 bytes) | 1262 elements, 14.4 kB | 145.3 GB |
| string tables (bytes, per node) | 40 B | 0.40 GB |
| determinants, per copy (list, root copy, sorted copy) | 16 B | 0.15 GB |
| eigensolver, root (9 vectors) | 72 B | 0.7 GB |
| thread vectors of the one-sweep product (48) | 384 B | 3.9 GB |
| peak RSS | 14.5 kB | 146 GB |

Elements per determinant (lower triangle) grow as N^0.21: 676 at 5.4e5,
883 at 1.8e6, 1205 at 7.6e6, 1262 at 1.0e7 (Cr2: 91 per spin-flip function
at 2.9e7 functions; Fe4S4 has 14 times the connections of Cr2 at a sixth of
the size). Of the D = 84807 single and double excitations of a determinant,
2.97 % lie in the space at 1e7.

Products: stored 1.55 s, direct 81.8 s (one product over 1.27e10 pairs:
6.4 ns per pair per node). Of the direct product about 77 % is finding the
connections (the byte-tree search tests 248 candidates per connected pair)
and 23 % evaluating them (1.47 ns per element, from the value pass of the
build). The build of iteration 12 (5.5e6 new rows, 7.3e9 new elements):
68 s searching with evaluation, 10.7 s evaluating the values a second time.
Selection: 7 s at 7.6e6, 17 s at 1e7 (heat-bath candidates).

**Why PT2 ran out of memory at 1e7** (run3, `"stored"`, explicit
`epsilon2_large` 1e-6, 20000 samples): the deterministic part at 1e-6 is
2.39e9 terms (4.18e8 distinct outside the space) in one pass, 108 GB on top
of the node's baseline (measured in diag mode with a direct H), while the
stored Hamiltonian of the selection, 145 GB, was still held: it is kept
past perturbation theory for the density matrices. 108 + 145 GB exceed the
188 GB node. Two causes: the matrix is not released before PT2 (it is only
needed by `WriteResults` for the density matrices, which can use the
string tables), and an explicit epsilon2_large bypasses the memory-sized
choice of "auto" (`pt auto` sizes the deterministic part to 60 % of the
memory available, but would also not know of the 145 GB unless they are
freed first). Fix (own item, not done yet): release the stored matrix
before PT2 and compute density matrices from the tables, and size the
deterministic part in batches when epsilon2_large is given.

## Projection to 1e10 determinants (FeMoco)

Connections per determinant c = D f(N): D from the orbital and electron
counts, f the share of them inside the space, measured on Fe4S4 as
f = 0.0297 (N / 1e7)^0.21. How f carries over to a much larger orbital
space is the open question, so three cases: "dense", the Fe4S4 law at the
same N; "dilute", the same law at the same N / N_FCI (N_FCI = 8.9e15 for
Fe4S4, 3.5e30 and 3.6e35 below); "middle", their geometric mean. Direct
product at Fe4S4's 6.4 ns per pair per node. Real numbers, no spin_flip.

| | D | c (dense / middle / dilute) | stored H, TB (middle) | direct product, node-hours (middle) |
|---|---|---|---|---|
| Reiher (54e,54o), W = 2, Ms = 1 * | 777140 | 100561 / 2803 / 78 | 153 (920 nodes) | 25 |
| Li-Chan (113e,76o), W = 3, S = 3/2 | 1772778 | 229396 / 1869 / 15 | 102 (615 nodes) | 17 |

(*: an assumption of the projection: 54 electrons cannot have S = 3/2, so
Ms = 1 (28 alpha, 26 beta) is taken for the Reiher space; its D and FCI
size differ from Ms = 0 or an odd-electron variant by a few per cent.)

Arrays over the determinants at N = 1e10, today held whole on every node
or rank:

| item | today | Reiher (W=2) | Li-Chan (W=3) |
|---|---|---|---|
| determinant list, per copy | per node (shared), root copy, sorted copy | 149 GB | 224 GB |
| string tables (two groups) | per node | ~370 GB | ~520 GB |
| eigensolver vectors | root, 9 | 670 GB | 670 GB |
| x | per node | 75 GB | 75 GB |
| y | per rank | 75 GB | 75 GB |
| thread y of the one-sweep product | per rank, 48 | 3.5 TB | 3.5 TB |

None of these fits a node of 170 GB on its own; each must be divided over
the ranks. The matrix (100+ TB in the middle case) cannot be stored, so the
product is direct, at 17 to 25 node-hours each in the middle case: some
1000 node-hours per iteration of 40 products.

## Ranking

By what makes 1e10 feasible (memory), then by speed:

1. **(e) the determinant list and string tables divided over the ranks** -
   new; 0.5 to 0.75 TB per node today at 1e10, the largest replicated item.
   A rank owns determinants (by string or hash), the connection search runs
   over owned rows and looks up remote columns. Prerequisite for anything
   past ~2e9 (W = 2).
2. **(b) eigensolver vectors in segments** - removes the root's 9 full
   vectors (72 B per determinant, 670 GB at 1e10). Done (below).
3. **(a) rows owned in contiguous ranges equal to the vector segments, x
   streamed in blocks, y only for owned rows, no thread y** - removes the
   remaining full-length x per node and y per rank (16 B per determinant)
   and the thread vectors; the product becomes: own block first, then the
   blocks of x from the other ranks in turn (pairwise, overlapped with the
   work on the previous block), the transposed contributions sent back to
   the block's owner. An allgathered x per node (8 B per determinant) is
   affordable to ~1e10 / 2 nodes' worth (75 GB at 1e10 real) and is kept
   while it fits; the streamed form is the fallback.
4. **Fast direct product** - the only product at 1e10; 77 % of it is the
   search for connections (248 candidates per connected pair). Enumerating
   doubles from the determinant through the heat-bath-like tables of the
   space (string pairs) instead of testing tree candidates.
5. **(c) values from the search kept** - 10.7 of 79 s in the build of
   Fe4S4 at 7.6e6 (13 %), 21 of 156 s for Cr2 at 1.2e8 on one node. Done
   (below), speed only.
6. **(d) root-side serial stages** - partition of rows 2 x 14 s, new list,
   diagonal, tables 8 s at 1.2e8 on one node; speed only.
7. **PT2 memory** (own item) - release the stored H before PT2; batch the
   deterministic part when epsilon2_large is given.
## Design: Davidson vectors in segments (item b)

Layout. Rank r holds elements [n r/p, n (r+1)/p) of every vector of the
few-vector solver (`fci_davidson.cc`, `VectorLayout` with one segment): the
basis, its products, the diagonal and, with a mask, the mask. Only the small
projected matrices are replicated; every dot product is a local OpenMP sum
and one `MPI_Allreduce` per projection (all columns batched), as the solver
already did for the fci mode. The root's start vectors are scattered before
and the eigenvectors gathered after a solve, so that selection, PT2, the
checkpoints of the selection loop and `wfn.h5` see the same root vectors as
before (memory on the root: those, once, as before).

Product. x is gathered once per node into a shared array (leaders
`MPI_Allgatherv`), every rank applies its part of H into a full-length y of
its own, and the products are summed onto the owners of the segments
(`Comm::SumScatter`, chunked `MPI_Reduce` to each owner). The operator, the
stored or direct H, does not change.

Stability (checked against the SQD list): the basis is orthonormalized
with two-pass Gram-Schmidt and a relative dependence test (1e-10 of the
norm before), collapse keeps the Ritz vectors and the previous ones
orthogonalized (LOBPCG-like), a basis that cannot be extended stops the
solver. The preconditioner is 1/(H_ii - E + shift) with |.| clamped at 1e-8,
not 1/max(|H_ii - E|, 1e-3); unchanged here (it decides the iterates).

Restart files: the eigensolver's checkpoint is written by every rank at
its offsets (as in the fci mode); the fingerprint, computed on the root, is
broadcast. The selection checkpoints are written from the gathered vectors.

PT2 takes the gathered root vectors (ShareVectors), unchanged.

Used when: the few-vector solver, more than one rank, the ranks of each node
consecutive, and a space above `dense_dim` and 4096 per rank; otherwise the
root solver as before.

## Measured after items b, c, d (one node so far)

Cr2 1.2e8 point (k15 input, variational only), one node with 140 GB (the
dev node holds 35 GB besides; 170 GB ran out of memory there): both runs
on the same node, last two iterations.

| | iteration | dets | t_sel | t_ham | t_dav | per product | peak RSS |
|---|---|---|---|---|---|---|---|
| before | 14 | 5.8e7 | 6.8 | 49.5 | 88.4 | 2.85 s | |
| after | 14 | 5.8e7 | 6.1 | 45.6 | 90.3 | 2.91 s | |
| before | 15 | 1.2e8 | 13.5 | 147.3 | 1437.6 | 49.6 s | 132 GB |
| after | 15 | 1.2e8 | 13.1 | 130.9 | 1406.3 | 48.5 s | 132 GB |

At 1.2e8 on one node 24 % of the values do not fit 140 GB and are worked
out in every product: 49.6 s per product against 4.65 s on 4 nodes with
all values held (README) - memory decides the speed.

Fe4S4 1e7, one node, 160 GB: 799 s before, 804 s after (same node; item c
saves 10.7 s of value evaluation at 7.6e6 but the search that keeps the
values is 7.4 s slower).

## Measured after items b, c, d on 4 and 8 nodes

Cr2 k15 input, variational only, "adapt" 170 GB, one rank of 48 threads
per node; before and after on the same nodes, one after the other. Seconds;
per product: t_dav over the products of the iteration; RSS: the root's
peak (GB), node: memory in use above the node's baseline.

| nodes | dets | | t_sel | t_ham | t_dav | per product | RSS root | node |
|---|---|---|---|---|---|---|---|---|
| 1 (140 GB) | 5.8e7 | before | 6.8 | 49.5 | 88.4 | 2.85 | 69.9 | |
| 1 (140 GB) | 5.8e7 | after | 6.1 | 45.6 | 90.3 | 2.91 | 70.0 | |
| 4 | 5.8e7 | before | 3.1 | 16.4 | 41.2 | 1.33 | 25.7 | |
| 4 | 5.8e7 | after | 3.1 | 14.8 | 40.6 | 1.31 | 25.1 | |
| 8 | 5.8e7 | before | 2.3 | 16.5 | 42.0 | 1.36 | 12.8 | |
| 8 | 5.8e7 | after | 2.2 | 12.4 | 32.6 | 1.05 | 12.2 | |
| 1 (140 GB) | 1.2e8 | before | 13.5 | 147.3 | 1437.6 | 49.6 | 131.7 | 132 |
| 1 (140 GB) | 1.2e8 | after | 13.1 | 130.9 | 1406.3 | 48.5 | 131.8 | 132 |
| 4 | 1.2e8 | before | 6.4 | 39.5 | 96.1 | 3.31 | 56.3 | 52-56 |
| 4 | 1.2e8 | after | 6.4 | 35.5 | 95.8 | 3.30 | 55.0 | 53-57 |
| 8 | 1.2e8 | before | 4.9 | 37.9 | 95.5 | 3.29 | 28.0 | 24-28 |
| 8 | 1.2e8 | after | 4.7 | 28.3 | 75.0 | 2.58 | 26.7 | 24-29 |

Per-rank product breakdown at 1.2e8 (s over 29 products, local / sending
x / summing y with the wait for the slowest rank):
4 nodes before 74.5-82.8 / 3.4-8.2 / 4.9-13.3, after 74.7-81.0 / 1.1 /
10.4-16.7; 8 nodes before 56.7-74.4 / 2.5-12.8 / 9.4-27.0, after 57.2-63.9
/ 1.3 / 7.1-13.8.

What this says:
- **Item b** gains 21 % of t_dav on 8 nodes (95.5 -> 75.0 s at 1.2e8,
  42.0 -> 32.6 s at 5.8e7) and nothing on 4. The root no longer does the
  solver's vector work while the other ranks wait (their "send x" time,
  6-13 s, was that wait), and the bucketed partition (item d) evens out the
  local work (spread 57-64 s instead of 57-74 s). The root's peak shrinks
  by only 1.3 GB: other arrays dominate it at 1.2e8.
- **4 to 8 nodes** gains little: local work per product 2.8 -> 2.2 s,
  because at 8 nodes a rank holds too few elements for the one-sweep
  product (its 48 thread vectors of 6e7 numbers, 23 GB, exceed its 18.6 GB
  of elements) and reads its matrix twice. This is item a: no thread y.
- **Item d** takes 4-10 s off t_ham (the row partition of the two-sweep
  product); item c is not taken at 8 nodes either.

Energies agree with the code before to 1.2e-11 or better (the difference
grows with the space and stays far below the 5e-5 residual of the solver;
1, 4 and 8 ranks of the old code differ among themselves by up to 3e-12
at 1.2e8), with the same determinant counts in every iteration.

## Fe4S4 toward 1e8 on 8 nodes

From the 1e7 space, epsilon1 1e-5, 5e-6, 3e-6, "adapt" 160 GB per rank:

| iteration | dets | elements per det | stored, all ranks | values held | t_sel | t_ham | per product | root RSS |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.54e7 | 1528 | 434 GB | all | 4.4 | 49.8 | 2.45 s | 62.7 GB |
| 1 | 7.37e7 | 1881 | 1102 GB | 56 % | 13.2 | 262.3 | 42.5 s | 145.8 GB |
| 2 | (selected) | | | | 36.6 | out of memory in the build | | |

Elements per determinant keep growing as N^0.20 (1262 at 1e7). At 7.4e7
the matrix needs 22.6 kB per determinant (1.5 TB with values on 8 nodes);
44 % of the values are worked out in every product, which is then 17
times slower. The third iteration ran out of memory while building: with
160 GB per rank on nodes of 188 GB with 25 GB in use beforehand there is no
room for the transient memory of the build (the per-thread row arrays grow
by doubling, the string tables are rebuilt beside the old ones). A limit
of 150 GB, and a build that bounds its transient memory, are needed
there. The rows-left-out sum took 17.5 s at 7.4e7: one collective waited
for the slowest rank's search.
