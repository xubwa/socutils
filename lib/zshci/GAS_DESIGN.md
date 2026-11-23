# A faster GAS sigma: occupation classes and an external space (design)

Status: design only, nothing here is implemented yet (2026-10-16). It covers
the `"fci"` mode (`src/zgasci`, `src/gas_space.*`, `src/fci*`,
`src/gas_dist*`). Numbers are tagged **[measured]** (Slurm runs, 1 node =
48 cores of Xeon Gold 6248R, OpenMP, MKL, real arithmetic, `ZGASCI_STATS=1`,
`benchmark_sigmas` 2, second sigma) or **[estimate]** (a model, to be checked
by the go/no-go measurements of section 5). Inputs, logs and the counting
script are in `/scratch4/lcheng24/xwang405/zshci/fci/gasdesign/`.

## 1. What the kernel does now, and where the time goes

**Now.** Each GAS space is cut in two halves (side A and side B), each half
into sub-groups of at most 16 spinors. A string's class is its number of
electrons in each sub-group, so a space of 102 spinors with at most 4
electrons is ten sub-groups and many small classes. For each of the nine
kinds of terms (which side p, q, r, s are on) and for each piece of at most
256 target rows, the BLAS kernel (`zgasci_blas.inc`) does this:

- it lists the placements of the row operators over the sub-groups (by
  class);
- it enumerates the row operators x of every row, placement by placement,
  and sorts them by source bucket;
- it enumerates the column operators y from the target columns of the piece
  and sorts them by run of source columns;
- for each run it either gathers D, does a GEMM `F = W D` and scatters, or,
  when F would be more than `ZG_BLAS_DIRECT` times what the scatter needs,
  computes the needed `F[y][c]` one by one as dot products (one indirect
  load per multiply-add).

The row side is fixed per kind: it is the side with more operators, and for
`a+a+ (x) aa` the side of the creation operators. When the column operators
are annihilations (or a single replacement `a+_q a_s`) on a dilute space,
each column takes only the few y that remove its own electrons. A run of
columns then needs the union of these, which is much larger, so the kernel
falls back to dot products. The W matrices are also rebuilt for every sigma.

**Measured** (one sigma, 48 threads; "thread-s" is summed over the threads,
for both halves):

| case | dets | s/sigma | sub-groups (A+B) | classes A | buckets A | blocks |
|---|---:|---:|---|---:|---:|---:|
| (a) TZ 12e SDTQ 2Ω=0 | 4.500e8 | 455.6 | 7+7 | 1890 | 51877 | 352674 |
| (b) DZ 8 in 48 FCI 2Ω=0 | 2.383e7 | 5.35 | 3+3 | 45 | 1035 | 2460 |
| (c) CVTZ fc SDT 2Ω=0 (`max_subgroup` 16) | 3.682e7 | 56.8 | 7+7 | 672 | 18954 | 63436 |

Share of the thread-seconds per step **[measured]**:

| step | (a) | (b) | (c) |
|---|---:|---:|---:|
| placements | 1 % | 0 % | 0 % |
| row operators: enumeration | 19 % | 13 % | 26 % |
| row operators: sorting | 8 % | 5 % | 10 % |
| candidates (source column buckets) | 8 % | 1 % | 6 % |
| column operators (enumeration and sorting) | 11 % | 17 % | 12 % |
| gather D | 15 % | 9 % | 11 % |
| gather W | 8 % | 6 % | 11 % |
| GEMM | **1 %** | 3 % | **1 %** |
| dot products | 22 % | 40 % | 17 % |
| axpy (kinds with no column operator) | 6 % | 4 % | 5 % |
| scatter | 1 % | 2 % | 0 % |
| total thread-s (wall x 48) | 21774 (21866) | 242 (257) | 2554 (2724) |

Work done in one sigma **[measured]**, and the work that is needed
**[computed exactly]**:

| | (a) | (b) | (c) |
|---|---:|---:|---:|
| multiply-adds in GEMM / dot / axpy | 1.17e12 / 1.05e12 / 2.8e11 | 3.6e10 / 2.8e10 / 7.6e9 | 7.9e10 / 1.08e11 / 2.5e10 |
| executed multiply-adds, total | 2.49e12 | 7.17e10 | 2.13e11 |
| needed: T = Σ_K P_K² | 2.02e12 | 5.52e10 | 1.90e11 |
| executed / needed | 1.23 | 1.30 | 1.12 |
| rate per thread: GEMM / dot / axpy (FMA/s) | 4.9e9 / 2.2e8 / 2.0e8 | 5.0e9 / 2.9e8 / 7.9e8 | 5.0e9 / 2.5e8 / 2.1e8 |
| rows x kinds; placements tried | 1.46e8; 1.21e10 | 4.0e6; 3.1e7 | 2.64e7; 1.71e9 |
| operators enumerated (row + column) | 1.02e11 + 3.7e10 | 1.37e9 + 7.9e8 | 1.98e10 + 5.6e9 |
| minimum: Σ_K R_K = C(N,2) N_det | 2.97e10 | 1.57e9 | 2.43e9 |

T is the number of multiply-adds of a resolution through the determinants K
of N−2 electrons: P_K is the number of pairs P such that K+P is a
determinant of the block, with the labels. It is counted exactly by a
dynamic program over the spinors (`gasdesign/count.py`). The program
reproduces the determinant counts of the kernel exactly, and agrees with
brute force on small random spaces.

What the tables show:

1. **The arithmetic is not wasted.** The kernel executes only 1.1 to 1.3
   times the needed multiply-adds. The time goes into doing them about 20
   times slower than GEMM: 85 to 95 % of the thread time is enumeration,
   sorting and indirect access, and about 1 % is GEMM.
2. **Operators are enumerated 5 to 10 times more often than needed.** (a)
   enumerates 1.4e11 operator entries against the 3.0e10 gathers of a K
   resolution, and (c) 2.5e10 against 2.4e9. The kernel tries 83 placements
   per row and kind in (a), 65 in (c), and enumerates the column operators
   again for every piece of rows. Nearly every enumerated operator lands in
   the space, because placements are filtered by class first. The cost is
   the enumeration and sorting itself, per string and per kind. The kernel
   has no counter for operators that are rejected, and counting them needs
   code.
3. **Fragmentation.** The 16-spinor sub-groups cut the external space into
   ten pieces. (a) then has 1890 classes per side and 352674 blocks of 1276
   determinants on average, so GEMMs are small and lists are short.
4. Even the FCI-like case (b) spends 40 % in dot products (53 % for 12 in
   44, README). The dot path, not the GEMM path, is what the kernel runs.

**No-code experiment [measured]: external space alone on side B**
(`"subgroups"`: side A = spinors 0-13 as groups [0-9], [10-13]; side B = the
external spinors in groups of at most 16):

| | (a) default | (a) int/ext | (c) default | (c) int/ext |
|---|---:|---:|---:|---:|
| s/sigma | 455.6 | **359.8** | 56.8 | 85.5 |
| classes A / B | 1890 / 1890 | 19 / 330 | 672 / 672 | 14 / 286 |
| blocks | 352674 | 17646 | 63436 | 12166 |
| W matrices per sigma (serial) | 0.6 s | 5.4 s | 2.9 s | **32.6 s** |
| main steps | dot 22 %, row ops 27 % | column ops 48 %, dot 25 %, axpy 15 % | row ops 36 %, dot 17 % | column ops 64 %, dot 17 % |

Coarse blocks alone help in (a), by 21 %. They do not help when the W
matrices of a 154-spinor side are rebuilt for every sigma, as in (c). In
both, the remaining time is enumerating column operators string by string
(about 100 ns each) and dot products. This is exactly what layer 2 removes.

**Lower bound [estimate].** T at the kernel's own GEMM rate (4.9e9 FMA/s per
thread, 2.4e11 per node), plus 2 Σ_K R_K gathers and scatters at 1e8 random
accesses/s per thread:

| case | now | lower bound | headroom |
|---|---:|---:|---:|
| (a) | 455.6 s/node | 8 s + 12 s | ~20x |
| (c) | 56.8 s/node | 0.8 s + 1 s | ~30x |
| DZ 12 in 52 FCI, 1.15e10 dets, T = 5.9e13 | 7800 node-s | ~560 node-s | ~14x |
| CVTZ fc SDTQ, 2.48e9 dets, T = 2.25e13 | 4300 node-s | ~160 node-s | ~27x |
| CVTZ ae SDTQ, 1.05e10 dets, T = 1.10e14 | 22000-29000 node-s | ~1000 node-s | ~25x |

The first column of the last three rows is production experience; the T
values are computed exactly.

## 2. Layer 1: a general GAS mode driven by occupation classes

The approach of LUCIA, and of its relativistic forms LUCIAREL and KRCI
(Olsen; Fleig, Olsen and Visscher; Knecht et al.), on the A/B strings that
zgasci already has. FCI is the special case of one space per side.

**2.1 Atoms and classes.** An *atom* is a set of spinors with equal weights
in all constraints (as `MakeGasLayout` groups them now; with disjoint
constraints, an atom is a GAS space). Each atom has a side assignment: by
default half of it on A and half on B, as now; layer 2 puts a whole atom on
one side. The *class* of a string is its number of electrons in each atom
on its side, within the per-atom bounds that follow from the constraints.
The class of a determinant is (A class, B class) such that every constraint
`min <= Σ w n <= max` holds. Blocks stay (A class, A label, B class), with
the B label implied, as now; only the classes are coarser.

- (a) and (c): A has atoms S1 (5 spinors, 0-5 electrons), act (2, 0-2) and
  ext (51 or 77, 0-4). That is at most 90 classes per side against 1890 or
  672 now, and blocks bigger by 1 to 2 orders of magnitude **[estimate]**.
- There is no 16-spinor limit and no `2^m` rank tables. A string is
  addressed by the lexical rank of its sub-string in each atom
  (combinatorial number system over a binomial table), in mixed radix over
  the atoms. Strings of a class are sorted by label into buckets, keeping
  `bucket`, `pos` and `bp` as now (24 + 16 bytes per string). Atoms of up
  to 512 spinors are held as occupation lists or as multi-word masks.

**2.2 Types and connections.** A *type* of a kind is the atom of each of its
operators, for example `a+_p(ext,A) a_r(S1,A) (x) a+_q(ext,B) a_s(act,B)`.
Class arithmetic gives the source class pair from the target class pair, the
kind and the type. The plan tabulates `conn[kind][type][target class pair]
-> source class pair or none` once. Constraints are therefore applied per
class pair, not per string, and the placement search and its 1e10 tries
disappear. Spinors of each atom are kept in lists sorted by label, so the
operators of a type come out grouped by the label change. The source bucket
follows from that label change, so no sorting is needed.

**2.3 The contractions.** Rows (targets) stay owned by one thread, as now.
For each (kind, type) the plan chooses one of three dense forms and the half
(row side) by a cost model. The half is no longer fixed per kind.

- **F0, row gather** (today's GEMM path, cleaned up). For each target row r'
  the operators x of the type give source rows `r_x` (sub-string rank
  updates, about 1 ns each). The kernel gathers `D[x][c] = ±C[r_x][c]`, a
  contiguous stretch of each source row, computes `F = W_type D` by GEMM,
  and scatters F to `c' = O(y) c`, the address computed by a rank update.
  It is used when the column operators are *particle-like* on a dilute atom
  (creations into it) or *hole-like* on a nearly full atom (annihilations
  from it). Then every column takes nearly all y of the label: the GEMM is
  dense, and needed / executed is close to 1.
- **F1, column resolved.** When the column operators remove electrons from a
  dilute atom, they are resolved through `Kc = a_S c` (N−|S| electrons on
  the column side): `D[(x,S)][Kc] = ±C[r_x][a+_S Kc]`, then
  `F[Q][Kc] = W'[Q][(x,S)] D` by GEMM, then `σ[r'][a+_Q Kc] += ±F`. All
  column indices become free-spinor indices, so the GEMM is dense, and the
  scatter stays inside the owned row r'. Here m = #Q of one label, which is
  only 9 to 15 for a single ext spinor (computed: 9.3 for TZ, 15.3 for
  CVTZ), so the GEMM is narrow. That is still about 10 multiply-adds per
  gathered element, against 1 on the dot path.
- **F2, pair resolved** (N−2 on both sides). The unit is one target bucket
  of the row side: `K = (Kr, Kc)`, `D[R][K]` gathered, `E[P][K] = V[P,R] D`,
  then scatter. Within a bucket all P of the label land in that bucket, so
  E is used in full and nothing is written outside the unit. Here m and k
  are pair counts of one label: 337 for ext pairs of TZ and 834 for CVTZ
  (computed), so this is GEMM at full speed. It is used for types where P
  and R both lie in big atoms and the row bucket is small enough to be one
  unit, which is typical of dilute GAS.

Example. Take `(AA,BB)` with P and R in ext. Today the rows are A (the
creation side), so the column operators annihilate pairs of B, which is
sparse, and the dot path runs. In layer 1 the rows are B (half 1) and the
column operators are creations of a pair on A, which is dense, so F0 does a
GEMM with m = #pairs of one label. The dot path remains only for tiny
pieces.

**2.4 Labels.** The operators x and y are listed per label, and a GEMM is
done per (source bucket, target bucket) pair, so W is cut to (type, label of
x, label of y) blocks. These blocks are built once per run and kept, not per
sigma (today 0.6 to 33 s per sigma, serial). 2Ω, parity and any `mod n`
change only the label arithmetic.

**2.5 Sides and sub-groups.** Side A/B and the two halves stay as they are
(the transposed vector for half 1, two vectors of workspace). The input
`subgroups` / `max_subgroup` becomes optional: if given, it splits an atom
into pseudo-atoms (finer classes); it is no longer needed for addressing.

**2.6 MPI and Davidson unchanged.** A rank still owns ranges of row buckets
of each half (as `gas_dist_half.inc`) and reads the source vector slice by
slice. F0 and F1 read only source rows `r_x`, and F2 only source rows
`a+_R Kr`. So the existing filter "source row bucket in this slice" applies
unchanged, as do calibration and the cost-based ranges. The low-memory
Davidson only sees a vector and a sigma. The plan order changes, and with it
`Fingerprint()`, so old checkpoints need a conversion through `PlanDets` or
a fresh start. `Diagonal`, `PlanDets` and `PairDensity` move to the new
addressing.

**2.7 Memory [estimate].** Strings: the same 40 bytes per string, with equal
or fewer strings. `conn`: (classes)² x types, small. W blocks: at most the
present `9 (n/2)^4` numbers (3.6 GB for n = 168), but cut by type and label,
typically some 100 MB. Thread workspace: D, E and F of at most `k x n`
(about 5000 x 256 x 8 B = 10 MB per thread). N−1 replacement tables per
type are not stored: rank updates compute them. As an option they can be
cached (8 B per string and electron, 1.6 GB for (a)).

**2.8 FCI must not get slower.** With one atom per side, the classes are
just electron counts on A, there is one block per label pair, and the types
are trivial. F0 serves creation-type column operators and F1 annihilation
or replacement types. This replaces the dot path (40 % in (b), 53 % for 12
in 44) and the per-string enumeration of column operators (17 %).
**Expected [estimate]:** 1.5-3x for FCI-like blocks. Gate: (b) and 12 in 44
must not be slower.

**2.9 Expected for dilute GAS [estimate].** The 47 to 54 % spent on
placements, enumeration, sorting and candidates mostly goes away. The dot
products become F0/F1/F2 GEMMs at 1-5e9 FMA/s per thread. The result is
about 4-8x for (a) and (c), from 455 to about 60-110 s and from 57 to about
8-15 s.

## 3. Layer 2: an external space on top of layer 1

**3.1 Definition.** A GAS atom is marked external (X), with a small maximum
`e_max <= 4` (min 0). Example input: `{"orbitals": [...], "max": 4,
"external": true}`. A determinant is an *internal determinant* I (any GAS
over the other spinors, handled by layer 1) times an *external tuple* t
(sorted, e = N − n_int <= e_max spinors of X). In layer-1 terms, X is one
atom put whole on one side, and the internal A/B pair is the other side.
The plan code and blocks are those of layer 1.

**3.2 Storage.** There is one block per (internal block, e, λ_X) with
`λ_X = target − λ(I)`. It is a dense matrix `C[i][t]`: i the internal
determinants of the block (an internal A x B block, flattened), t the rank of
the tuple among those with e electrons and label λ_X. The rank uses
label-restricted lexical addressing (weights `Wgt[spinor][k][λ]`, O(e) per
address, nothing stored per tuple). (a): internal determinants number about
5000, and a row with e = 4 has about 1.5e5 columns. Rows are owned whole.

**3.3 Terms.** These are grouped by the number of X operators among p, q, r,
s. The shares are those of T **[computed]**, (a) / (c):

| X operators | example | share of T | contraction |
|---|---|---:|---|
| 4 (P, R ⊂ X) | `<ab‖cd>` | 46.6 % / 53.3 % | row-local F2: for each internal row i, `D[R][(i,Kt)] = C[i][Kt+R]`, `E = V_X D`, `σ[i][Kt+P] += E` (Kt has e−2 electrons). Rows of one block batch into the n dimension; m = k = 337 / 834. No conflicts: the row is owned. |
| 2 (P = (i,a), R = (j,b)) | `<ia‖jb>` | 22 + 9 + 4 % / 23 + 8 + 4 % | F2 over the pair (int, ext): `Kint = I − j`, `Kt = t − b`, `E[(i,a)] = V[(i,a),(j,b)] D[(j,b)]`, m = k = 112 / 180. Unit = one internal target bucket. When internal buckets are large: F1 over X with m = #a of one label. |
| 3 (P ⊂ X, R = (j,c), or the reverse) | `<ab‖jc>` | 8.0 + 4.9 % / 4.6 + 3.5 % | F2 over X pairs on the side of the pair, with internal single creation or annihilation from the layer-1 tables (Δe = ±1). |
| 2 (P ⊂ X, R internal, or the reverse) | `<ab‖jk>` | < 1 % | F0: internal pair tables times X pair creation (Δe = ±2). |
| 1 | `<ia‖jk>` | ~2-3 % | F0 with internal two-operator tables times one X spinor. |
| 0 | internal only | ~2 % / ~1.5 % | layer-1 sigma applied to all columns t of a block at once (multi-vector GEMM, n = #t). |

**3.4 Integrals and memory [computed].** `V_X[ab,cd]` over X pairs takes
0.21 GB full or 14 MB cut by label (102 X spinors), and 1.11 GB full or
79 MB cut (154, i.e. C(154,2)² doubles = 1.1 GB). `<ia‖jb>`:
(n_int n_X)² = (14·154)² doubles = 37 MB. `<ab‖jc>`: C(154,2)·14·154
doubles = 0.2 GB, about 10 MB cut by label. Everything is small next to one
vector (20 GB for 2.48e9 determinants). The current W matrices of a whole
154-spinor side (`9 (n/2)^4`) are not needed.

**3.5 Labels.** λ(I) fixes λ_X. Pairs of X are grouped by pair label (28
values; 337 / 834 effective size), and single X spinors by label (16 values;
9 / 15). These groups are the GEMM blocks.

**3.6 Fit with layer 1, MPI and threads.** Rows are internal determinants,
owned whole by one thread and one rank. All X-only terms are row-local.
Mixed terms read source rows I from other ranks through the slices
(MPI_Get) as now. No term needs the transposed vector, which saves two
vectors of workspace and the two transpositions. Work is balanced by a per-row cost (e and the
internal class); (a) has about 5000 rows for 48 threads. A row that is too
big for one thread is split by internal pieces, never by columns, so no
atomic additions are needed. The Davidson, checkpoints (new fingerprint)
and `PlanDets` work as in layer 1.

**3.7 Expected [estimate].** T at 2.4e11 FMA/s per node, gathers and
scatters at 1e8 to 3e8/s per thread, and internal tables negligible, give:

| case | now | layer 2 |
|---|---:|---:|
| (a) | 455.6 s/sigma [measured] | ~25-45 s, 10-18x |
| (c) | 56.8 s [measured] | ~4-8 s |
| CVTZ fc SDTQ on 4 nodes | 1076 s | ~60-110 s |
| CVTZ ae SDTQ on 12 nodes | 30-40 min | ~2-4 min |

## 4. Start vector and preconditioner for the 2Ω = 0 blocks

Today the start vector is the single lowest-diagonal determinant. The X 0+
state of O2 is a ± combination of two near-degenerate references (π*²). In
the production TZ run (`tz_sdtq/w0`) the first two iterations hardly move
the residual (1.58, then 1.55), and convergence takes 25 iterations [measured, log].
In production, 2Ω = 0 blocks took 25-36 iterations against 19-20 for
2Ω = 2 (reported).

- **Start vector.** Diagonalize H exactly in a reference space of the block
  and start from its lowest roots, embedded in the vector. The reference
  space is either the internal determinants with no external electron (layer
  2: the e = 0 block, at most 91 determinants for (a) before labels), a user-named space
  (`"fci": {"reference": {"orbitals": [10,11,12,13], "nelec": 2}}`, a
  CAS(2,4): 2 determinants in the block), or by default the M
  lowest-diagonal determinants (M about 500). H_PP comes from the
  Slater-Condon routines of zshci (`src/hamiltonian.h`) over `PlanDets`.
  The start-vector argument being added to `FciDavidson` (uncommitted work in
  the tree at the time of writing) is the natural hook.
- **Preconditioner.** A P-space preconditioner (as in Knowles-Handy FCI):
  `(H_PP − θ)^-1` on the reference block and `1/(H_ii − θ)` elsewhere,
  using the same M <= 2000 determinants. The cost is one dense solve of size
  M per iteration.
- **Expected [estimate].** The 2Ω = 0 blocks come down from 25-36 to the
  ~19-21 iterations of the 2Ω = 2 blocks, which is 25-40 % fewer sigmas,
  independent of layers 1 and 2.

## 5. Implementation plan

Each step ends with tests and a measurement that decides whether to go on.

| step | content | tests | go / no-go |
|---|---|---|---|
| 0 | Keep W across sigmas (built once per run). Try the int/ext `subgroups` layout on TZ production. | Energies unchanged to 1e-12. | Saves 0.6-33 s per sigma. TZ int/ext: −21 % [measured]. Not for CVTZ (W rebuild, column operators). |
| 1 | Microbenchmarks (standalone C in `bench/`) on the real (a) shapes: (i) row-local F2 with V_X, (ii) F1 for `(AB,AB)` ext-ext, (iii) F0 with label-sorted x and no placements. | Results against `zg_sigma` (reference kernel) on (b)-size vectors. | Go if each reaches ≥ 1e9 FMA/s per thread including gather and scatter (5x today's dot rate). Otherwise rethink before the plan changes. |
| 2 | Layer-1 plan: atoms, classes, rank addressing without 16-spinor groups, `conn` tables, new `PlanDets` and `Diagonal`. | Same determinant counts as today (all tests, and (a)-(c) in `dry_run`). Same diagonal set. | Plan time and memory no worse than today. |
| 3 | Layer-1 sigma: F0 with types, then F1 and F2, then the half chosen per (kind, type). Old dot path as fallback. | Sigma against the reference kernel with random v (existing tests, sides wider than 64), 1e-12 relative. O2 8 in 48 −149.953598673306, SDTQ −150.086940751954. | (c) ≥ 3x faster; (b) and 12 in 44 not slower. |
| 4 | MPI: port into `gas_dist_half.inc` (slices, calibration). | Ranks against 1 rank to 1e-11 Eh. | 12 in 52 on 10 nodes, CVTZ fc on 4 nodes: faster than 780 s and 1076 s. |
| 5 | Layer 2: input flag, storage and addressing; the X-only terms first, then mixed with 2 X operators, then 3, 1 and 0. | Against layer 1 on (c) and on small random-v spaces. Energies as in 3. | (a) ≤ 60 s per sigma on 1 node; CVTZ fc ≤ 200 s on 4 nodes. |
| 6 | Reference start vector and P-space preconditioner. | Same energies. | 2Ω = 0 blocks of (b) and (a) within +2 iterations of 2Ω = 2. |

**Risks.**

- F1 GEMMs with a single X spinor label are narrow (m = 9 to 15), which may
  cap F1 near 1-2e9 FMA/s. F2 avoids this, at the cost of bucket-sized units.
- Coarse classes give fewer and bigger blocks, so load balance needs the row
  pieces (which exist) and, in layer 2, per-row costs.
- Addressing strings over large atoms must stay O(1) per rank update.
  Otherwise enumeration moves cost around instead of removing it.
- Checkpoints of the old plan cannot be read directly.
- The kernel is shared with socutils' zgasci: decide where it lives (open
  question 4).
- Time reversal (Kramers pairs) is not used. That is a further factor of up
  to 2, outside this design.

**Open questions.** See the report that came with this document.
