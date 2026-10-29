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
