# Performance notes

How UC- and PC-NEVPT2 are organized for speed: the cost model, the choices
taken from the PC-NEVPT2 literature, the native kernels and the threading
model.  Measurements are in `validation-log.md` ("4f-scale timing").

Notation: `n_c`, `n_a`, `n_v` correlated core, active and virtual spinors;
`D` dimension of a target sector; `S` model states; `n_X` active strings of a
class; `M = S n_X` (union contraction) or `n_X` (state contraction) functions;
`m <= min(M, D)` dimension of their span; `n_e` external labels of a block.

## 1. Cost model

| class | sector | n_X | external labels (blocks x n_e) |
|-------|--------|-----|--------------------------------|
| ccvv | N   | 1 | C(n_c,2) x C(n_v,2) |
| ccav | N+1 | n_a | C(n_c,2) x n_v |
| cavv | N-1 | n_a | n_c x C(n_v,2) |
| ccaa | N+2 | C(n_a,2) | 1 x C(n_c,2) |
| aavv | N-2 | C(n_a,2) | 1 x C(n_v,2) |
| cava | N   | 1 + n_a^2 | n_c x n_v |
| caaa | N+1 | n_a + n_a C(n_a,2) | 1 x n_c |
| aaav | N-1 | n_a + n_a C(n_a,2) | 1 x n_v |

Per class:

| step | cost | implementation |
|------|------|----------------|
| functions `O_X Psi_I` | `S n_X D` | OpenMP kernel `zmr_apply_strings` |
| span | `min(M,D)^2 max(M,D)` + one eigenproblem of dimension `min(M,D)` | `zherk`, `zheevd` |
| Koopmans problem | `D^2 m` + one eigenproblem of dimension `m` | `zgemm`, `zheevd` |
| projection | `S n_e D (n_X + m)` (integrals first, `'R'`) or `S n_X D m` (functions first, `'T'`) | `zgemm`, order chosen by flop count |
| sum over `(k, e)` | rows: `S m n_e (n_X + S)`; external labels first: `m n_e n_X^2 / 2` per distinct E0 shift | OpenMP kernels `zmr_weighted_gram` / `zmr_contract_t`, `zmr_contract_w` |

plus, once, the integral transformation of the superblock
`(av ca|av ca)` (every block of the eight classes) and the generalized Fock
matrix.  UC-NEVPT2 is the same with `m = D` and the sector eigenstates in
place of the span.

For a 4f shell (`n_a = 14`, `D` up to 3432) and a few states the
three-operator classes have `n_X = 1288` and `M > D` in the union
contraction: the span is nearly the whole sector and the cost is that of
dense eigenproblems of dimension `D`.  For a 4f^2 or 4f^12 ion (`D <= 1001`)
the active-space algebra is negligible and the time goes to the integral
transformation and to the doubly external classes ccav / cavv.

## 2. Choices and their sources

**One eigenproblem per class, external labels only in denominators.**  This
is the structure of PC-NEVPT2 since the Koopmans-like matrices of Angeli,
Cimiraglia and Malrieu (J. Chem. Phys. 117, 9138 (2002)); Helmich-Paris and
Knecht (J. Chem. Phys. 146, 224101 (2017)) write every class as a generalized
eigenproblem `K c = M c eps` solved once, after which the external indices
enter through orbital energies only.  Here `K` is `Q^+ H_act Q` in an
orthonormal basis of the span (`pc-nevpt2.md` section 2).

**CI vectors instead of the 4-RDM.**  Kollmar, Sivalingam, Guo and Neese
(J. Chem. Phys. 155, 234104 (2021)) and Chatterjee and Sokolov (J. Chem.
Theory Comput. 16, 6343 (2020)) avoid the 4-RDM by inserting a resolution of
the identity of the CASCI space, i.e. by intermediates that are CI vectors.
With explicit determinant vectors this is the whole method: metric and
Koopmans matrices are Gram matrices of the vectors `O_X Psi_I` and
`H_act O_X Psi_I`.  It also bounds every matrix dimension by `D`, which is
the relevant bound for a small active space with several states (`M > D`),
where an RDM formulation would diagonalize matrices of dimension `M`.

**Three-operator classes: integrals first.**  Sokolov and Chan (J. Chem.
Phys. 144, 064102 (2016)) contract the three-index active strings with the
integrals before anything else, one vector per external orbital.  The engine
does the same when it is cheaper (`'R'` route): `r^J_e = sum_X v[e, X] O_X
Psi_J` (`n_e x D`), then one projection onto the first-order space.

**Exact denominators for ccav / cavv in the contracted space.**  With `m`
of the order of `S n_a` the explicit sum over `(k, e)` is an MP2-like sum with
one extra index; the Laplace engine (`uc-nevpt2.md` section 5) is needed only
when the first-order space is the whole sector (UC) and large.

**Two forms of the sum over `(k, e)`.**  With projected amplitudes
`t[J, k, X]` the matrix `a[I, J] = -sum_{k,e} conj(u^I_ke) u^J_ke / D_ke`,
`u = t v^T`, is evaluated either row-wise (`u` for one `k` at a time in
thread-private buffers, then the weighted Gram matrix of its rows) or with
the external labels first,

    W_k[X, Y] = sum_e conj(v[e, X]) v[e, Y] / (mu_k + delta_e - s),
    a[I, J] = - sum_k t[I, k, :]^+ W_k t[J, k, :],

whose cost does not depend on the number of states.  The second form is the
exact-denominator analogue of the Laplace intermediate `M_q` and wins for
many states and few active operators (ccav, cavv); `engine.gram_route` picks
by operation count.  The state-specific contraction needs only one column of
`a` per ket state, which the kernels compute alone.

**Non-redundant strings.**  Ordered pairs of equal operators (`t < u`) halve
`n_X` of the two- and three-operator classes, the Gram matrices and the
memory, and remove the trivial null vectors of the metric.

**Linear dependence.**  Thresholds on the metric eigenvalues in the
literature range from 1e-6 (ORCA FIC-NEVPT2, DALTON) over 1e-8 (BAGEL
CASPT2 / MRCI) to 1e-10 (Prism, double excitations); the default `lindep` is
1e-10.  The span is
taken from the eigenvectors of the Gram matrix of the smaller dimension and
re-orthonormalized by one Cholesky-QR step, which restores the orthonormality
that an eigenvalue of 1e-10 costs (1e-6) without an SVD of the functions
(the `zgesdd` of the function matrix of one state took 5.3 s for the caaa
class of a 4f shell, against 0.4 s for the Koopmans problem behind it).

**Integral transformation.**  For a small active space the literature agrees
that the transformation of the two-external integrals and the doubly external
classes dominate (ORCA manual; Freitag, Knecht, Angeli and Reiher, J. Chem.
Theory Comput. 13, 451 (2017); Park, J. Chem. Theory Comput. (2019),
doi:10.1021/acs.jctc.9b00762).  The interface keeps the real AO integrals with the `s4` pair
symmetry in memory and forms every MO block from a folded two-component
transition density and two matrix products (`zmr/interface/ao2mo.py`).
Density fitting will replace this source without touching the classes.

## 3. Symmetry blocks

For atoms and linear molecules `H_act` commutes with one-body operators
(`J_x, J_y, J_z`, or `J_z`; the spin for spin-free integrals), but the active
spinors delivered by an SCF are arbitrary mixtures within degenerate levels
and the sector Hamiltonians are dense.  `zmr.active.symmetry`:

1. solves the linear equations `[A, H_act] = 0` for one-body `A`
   (`[A, h] = 0` and `sum_p (A_tp g_pvuw + A_vp g_tpuw - g_tvpw A_pu -
   g_tvup A_pw) = 0`, `g = <tv||uw>`); the Hermitian solutions are the Lie
   algebra of the one-body symmetry group;
2. takes a generic element (it lies in one Cartan subalgebra) and
   diagonalizes it inside every degenerate level of `h`; its eigenvectors are
   the adapted orbitals and its eigenvalues their charges;
3. `symmetry.adapt(eris)` rotates the active orbitals; `ActiveSpace` then
   sets the active integrals that violate charge conservation to zero.  They
   vanish to the accuracy of the symmetry of the input integrals; the largest
   one is reported (`ActiveSymmetry.error`, `ActiveSpace.symmetry_error`) and
   the adaptation is refused if it exceeds `tol` (default 1e-8 Eh).

Every determinant now has a definite charge and the sector Hamiltonians are
block diagonal exactly.  The blocks are the connected components of the zero
pattern (`zmr.lib.blocks`), so nothing has to be known about the symmetry
group downstream: sectors are diagonalized block by block; the model states
come out symmetry pure; every function `O_X Psi_I` then lies in one block and
the span and the Koopmans problem of a class decouple.  Functions that
connect blocks (user-supplied model states that mix them) merge those blocks.

The results do not change: NEVPT2 with the Dyall Hamiltonian is invariant
under rotations of the active orbitals (the contraction `'state'` depends on
the individual model states and is therefore only comparable for the same
states).  For a 4f shell with 7 electrons the 3432 determinants split into
26 blocks of at most 327.

Only continuous (Lie) symmetries are found.  Point-group operations and time
reversal are not used.

## 4. Native code and threading

`make` builds `zmr/lib/libzmr.so` (C99 + OpenMP) and links it to the
`libmkl_rt` of the Python environment, the library PySCF uses.  Both are
loaded with ctypes (`zmr/lib/__init__.py`); every routine has a numpy / scipy
fallback, used when a library is missing or `ZMR_NO_NATIVE=1` is set, and the
fallbacks are the reference in the tests.

| kernel (OpenMP) | parallel over | used for |
|-----------------|---------------|----------|
| `zmr_apply_strings` | operator strings | functions `O_X Psi_I` |
| `zmr_apply_op` | (vector, orbital) | 1-RDMs, RDM reference |
| `zmr_build_hmat` | determinants (columns) | sector Hamiltonians |
| `zmr_sigma` | the strings with two electrons fewer (batches of vectors one after the other) | `H_act` applied to vectors without storing it (large spinor sectors); one `zgemm` per string, rows of the result updated under striped spin locks |
| `zmr_weighted_gram` | `k`, or the states when there are few `k` | `-sum conj(u_I) u_J / D` for stored `u` |
| `zmr_contract_t` | the same | the same sum with `u = t v^T` formed on the fly |
| `zmr_contract_w` | `k` | the same sum, external labels first |
| `zmr_fold_density` | orbital pairs | AO -> MO transformation |
| `zmr_block_support` | functions | assignment of functions to symmetry blocks |
| `zmr_components` | (sequential union-find) | blocks of a matrix from its zero pattern |
| `zmr_build_hmat_ab`, `zmr_apply_strings_ab` | determinants, operator strings | spin-restricted sectors (`zmr.nr`) |
| `zmr_sigma_ab` | batches of vectors, or the alpha strings when there are few vectors | `H_act` applied to vectors without storing it (large spin-restricted sectors); string-driven, one `dgemm` per block of determinants |

The three Gram kernels exist for complex and for real arrays (suffix `_r`);
they share one driver that keeps real and imaginary parts in separate arrays.
Every parallel region sizes its team from the operation count (at least 2e5
operations per thread), because a calculation makes thousands of small calls
(one external block, one symmetry block) for which starting all threads costs
more than the work.

Dense algebra goes through MKL (`zmr/lib/linalg.py`): `gemm`, `herk`,
`trsm`, `potrf`, `heevd`, `heevr` in their z- or d-variants according to the
dtype of the input, so real problems stay real.

numpy's own BLAS (OpenBLAS in conda / pip builds) has a separate thread pool.
The drivers limit it to one thread while they run
(`zmr.lib.numpy_blas_single_thread`, through `threadpoolctl`): the few small
numpy products that remain would otherwise wake a second full pool next to
the OpenMP threads, which cost about a second per calculation on 48 cores.  Kernels and MKL are never
nested: an OpenMP region contains no threaded BLAS call (the task kernels of
section 7, `zmr_sigma` and `zmr_sigma_ab` call BLAS inside their region, where MKL runs
sequentially, on blocks owned by one thread), so there is one level of
parallelism and `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (or
`zmr.lib.set_num_threads`) set it.

## 5. What limits the speed now

**Few states in a large active space.**  Profile of the spin-restricted
PC-NEVPT2 for N2, CAS(10,10), cc-pVTZ, one state, 4 cores (2026-10-03):
120 s in all, 91 s (76%) in H_act applied to the perturber functions one by
one in determinant space (`sigma_ab`, about 4900 vectors of dimension
30240 ... 63504), 13 s in matrix products, 6 s in forming the functions,
5 s in the orthonormalization; by class caaa 42 s, aaav 41 s, cava 17 s.
The Koopmans matrix `K = Q^+ H_act Q` is built by applying H_act to every
column of Q: `n_a + n_a^3` functions per class and spin pattern, cost
`n_a^3 D n_a^4`.  ORCA needs 11 s for the whole step.  Open items, in the
order they were proposed to the user (recorded at their request; items 1
and 2 are done and the numbers of this paragraph are those before them,
item 3 is not a to-do but the regime where the CI-vector route stays):

1. **Sigma kernel.**  Done (2026-10-04) for the spin-restricted code.
   `zmr_sigma_ab` was determinant-driven (Slater-Condon rules on the
   combined string, one `axpy` per connected determinant) and took 18.5 ms
   per vector against 6.4 ms of PySCF's `direct_spin1.contract_2e`
   (dimension 52920, 200 vectors, 4 CPUs of a Slurm allocation).  It is now
   string-driven (Knowles-Handy): with `E_pq = sum_sigma p^+_sigma q_sigma`

       H = sum_pq k_pq E_pq + 1/2 sum_pqrs (pq|rs) E_pq E_rs,
       k_pq = h_pq - 1/2 sum_r (pr|rq),

   and for a block of determinants `J` (one alpha string, up to 64 beta
   strings, 8 vectors) the single excitations of the alpha and of the beta
   string gather `D[rs, J] = (E_rs c)_J`, one `dgemm` forms
   `G[pq, J] = 1/2 sum_rs (pq|rs) D[rs, J] + k_pq c_J`, and the same links
   scatter `sigma_I += <I| E_pq |J> G[pq, J]`.  The identity holds for any
   real `h` and `(pq|rs)`; when `(pq|rs) = (qp|rs) = (pq|sr)` (detected, real
   orbitals) the pairs `p >= q` suffice and the product is 3.3 times smaller.
   The vectors of a batch are the innermost index, so one link moves eight
   numbers.  Threads take whole batches of vectors (private copies of the
   columns, nothing shared); with fewer than four vectors per thread they
   share the alpha strings of one batch instead and the part of the scatter
   that leaves the alpha string goes through a buffer per thread.

   | 4 CPUs, dimension 52920 | ms per vector |
   |---|---|
   | old kernel (`zmr_sigma_ab_det`, kept as reference), 200 vectors | 18.4 |
   | PySCF `direct_spin1.contract_2e`, one vector at a time | 5.4 ... 6.4 |
   | new kernel, 1 / 2 / 4 / 8 vectors | 5.1 / 4.4 / 4.6 / 4.3 |
   | new kernel, 200 / 1000 vectors | 4.3 / 4.1 |

   About 55% of the new time is the `dgemm` (55 x 56 per determinant), which
   runs at the rate MKL reaches in the same allocation for a large product
   with all four threads busy (36 GFlops per thread; 75 with one thread),
   about 20% the gather and 15% the scatter.  The N2 CAS(10,10) calculation above went
   from 120 s to 51 s (caaa 42 -> 20 s, aaav 41 -> 20 s, cava 17 -> 5.5 s;
   `sigma_ab` 91 -> 22 s), the rest being matrix products and the
   orthonormalization.  Without libzmr `zmr.nr` applies H_act through
   `pyscf.fci.direct_nosym` vector by vector (`kernels.sigma_ab_pyscf`).
   The spinor kernel `zmr_sigma` was rewritten afterwards; see below the
   list.
2. **Density-matrix route for `zmr.nr`.**  Done (2026-10-04,
   `zmr/nr/rdm.py`, derivation in `nr-nevpt2.md` section 6).  The metric and
   the Koopmans matrix of every class are formed from the vectors
   `W_w |J>` of an operator basis (at most `n + n^3` per sector) and from
   intermediate states: `K = W^T R + G M` with
   `R = y_1 (H y_2 |J>) +- y_2 ([H, y_1] |J>)`, the scheme of Chatterjee and
   Sokolov with which the 4-RDM terms cost one product `D n^6`.  H_act is
   applied to `1 + 4 n` vectors per state instead of to every function.
   What was done for speed after the first working version (15.4 s for the
   case below): the vectors of caaa and aaav are written straight into their
   place (`zmr_apply_strings_ab`, and the new scatter-add kernel
   `zmr_apply_strings_ab_add` for `E_uv` on the intermediate states) and
   their memory is reused by the next sector; `H E_uv |J>` from the
   one-operator intermediate states instead of `n^2` products with H_act;
   `G` by `dsyrk`, one triangle of `W^T R` by `dgemmt`; the matrices of the
   beta sectors from those of the alpha sectors for spin-flip symmetric
   states.

   N2, cc-pVTZ, all electrons correlated, 4 CPUs (express partition),
   exact integrals, NEVPT2 step (wall time including the integral
   transformation / sum over the classes), seconds:

   | case | CI-vector route | density-matrix route | ORCA 6 |
   |---|---|---|---|
   | CAS(10,8), 1 state | 2.3 / 1.8 | 0.8 / 0.3 | |
   | CAS(10,8), 3 states (state-averaged singlets) | 6.4 / 5.9 | 3.1 / 2.6 | |
   | CAS(10,10), 1 state | 52.0 / 51.5 | 4.8 / 4.4 | 11.0 / 4.1 |
   | CAS(10,10), 3 states | 238.1 / 237.6 | 53.9 / 53.4 | |

   By class, CAS(10,10), one state (CI-vector route -> density-matrix
   route): caaa 20.1 -> 1.8, aaav 20.1 -> 1.5, cava 5.6 -> 0.44,
   ccaa 2.05 -> 0.22, aavv 2.02 -> 0.22, ccav 0.85 -> 0.07,
   cavv 0.85 -> 0.07, ccvv 0.01.  The energies of the two routes agree to
   1e-14 Eh in every one of these runs (E2 = -0.2117672 Eh for CAS(10,10);
   between separate jobs E2 differs by a few 1e-8 Eh because the PySCF
   CASSCF is converged to 1e-9).  With RI vectors the times of the classes
   are the same (4.4 s; wall 4.5 s).  Profile of the 4.4 s: `dsyrk` 1.0 s
   and `dgemmt` 1.0 s (two sectors of 1010 vectors of dimension 52920, at
   the rate of MKL on 4 cores), operator strings 0.9 s, eigenproblems
   0.4 s, H_act 0.2 s.

   In the three-state CAS(10,10) run the CI vectors of the state-averaged
   PySCF CASSCF were spin-flip symmetric only to 1.6e-7 (1.3e-10 in another
   run of the same input), so both spin sectors were computed; the time is
   `dsyrk` and `dgemmt` 29 s (four sectors of 3030 vectors), eigenproblems
   of dimension 3030 11 s, operator strings 4 s.

   **Choice of the route.**  Model Hamiltonians (4 core, 30 virtual
   orbitals), 4 CPUs, union contraction, total time in seconds; `M` is the
   number of functions of all states in caaa / aaav, `D` the determinants
   of their stacked target sectors:

   | active space | states | M | D | CI vectors | density matrices | automatic |
   |---|---|---|---|---|---|---|
   | (3,3) in 6 | 1 | 222 | 600 | 0.13 | 0.11 | 0.10 |
   | | 3 | 666 | 600 | 0.24 | 0.32 | 0.25 |
   | | 10 | 2220 | 600 | 0.53 | 4.0 | 0.80 |
   | | 30 | 6660 | 600 | 1.5 | 74 | 1.4 |
   | (4,3) in 7 | 1 | 350 | 1470 / 2450 | 0.35 | 0.20 | 0.21 |
   | | 3 | 1050 | | 1.3 | 1.1 | 1.1 |
   | | 6 | 2100 | | 4.4 | 5.8 | 4.2 |
   | | 12 | 4200 | | 6.4 | 25 | 6.2 |
   | | 30 | 10500 | | 17 | 248 | 18 |

   (These model Hamiltonians have intruder states for the higher model
   states; the two routes then differ by up to 2e-6 Eh in single elements
   of the effective Hamiltonian, the sensitivity to the metric threshold
   `lindep` of terms with denominators near zero.)  The density-matrix route wins while `M < D` and loses quickly beyond,
   where its eigenproblems have the dimension `M` and the CI-vector route
   works in the whole sector.  `amplitudes='auto'`, the default of
   `zmr.nr.PCNEVPT2`, therefore takes the density-matrix route for a class
   if `nstate * n_X < D` and the CI-vector route otherwise.  State-specific
   contraction: the same criterion (measured: 30 states in (4,3) in 7,
   9.5 s / 24.6 s / 10.4 s).

   Not done: batches of determinants for the vectors (memory
   `2 nstate (n + n^3) D` numbers per sector, about 20 GB for one state in
   CAS(12,12)); a spin-adapted basis that would halve the work for states
   that are not spin-flip symmetric.
3. **Many states are a different regime.**  With `M = S n_X` functions
   exceeding the sector dimension (4f ions with hundreds of states) the
   span is the whole sector and the CI-vector route needs one dense
   eigenproblem; the density-matrix route has no advantage there.

**Spinor sigma kernel** (2026-10-04).  `zmr_sigma` was determinant-driven
as well: every ordered combination `(q, s, r, p)`, four times the connected
determinants, a string rank for each, no blocks of vectors, and it used the
Hermiticity of H.  The `E_pq E_rs` form of `zmr_sigma_ab` does not carry
over: without spin separation and pair symmetry the product would be
`n^2 x n^2` complex per determinant (`n` spinors), no faster than before.
The kernel now resolves through the strings `K` with two electrons fewer.
In a sector of `N >= 2` electrons `p^+ q = sum_r p^+ r^+ r q / (N - 1)`, so
the one-body part joins the integrals,
`g(pq|rs) = (pq|rs) + 2 h_pq delta_rs / (N - 1)`, and

    H = sum_{p<r} sum_{q<s} W[pr, qs] p^+ r^+ s q,
    W[pr, qs] = [g(pq|rs) + g(rs|pq) - g(ps|rq) - g(rq|ps)] / 2.

This is an identity for any complex `h` and `(pq|rs)`: the kernel relies on
no symmetry of the integrals (not on `(pq|rs) = (rs|pq)`, not on
Hermiticity); `N = 1` is the product with `h`.  For every `K` and the pairs
`q < s` of its empty orbitals: gather `D[qs] = <K| s q |J> c_J`, one `zgemm`
`G = W_K D` with the block of `W` of these pairs, scatter
`sigma_I += <I| p^+ r^+ |K> G[pr]`; gather and scatter share addresses
(from partial sums of binomials, no string is ranked) and signs.  The number
of multiplications per determinant, `C(N, 2) C(n - N + 2, 2)`, is about the
number of its connected determinants.  The block `W_K` is collected for
every `K`, which a batch of up to 128 vectors (the columns of `D`) pays for.
The threads share the strings `K` of one batch; different `K` meet in the
rows of sigma, which are updated under the spin lock of their stripe (4096
stripes), so no thread needs a copy of sigma.

| 4 CPUs, ms per vector | 1 vector | 8 vectors | 64 vectors |
|---|---|---|---|
| 10 in 20 spinors (184756), old / new | 14108 / 335 | 2646 / 86 | 1299 / 46 |
| 8 in 16 (12870), old / new | 377 / 12.4 | 58.7 / 2.9 | 27.2 / 1.41 |
| 7 in 14 (3432), old / new | 59.2 / 2.5 | 9.0 / 0.54 | 3.4 / 0.25 |
| 3 in 14 (364), old / new | 2.7 / 0.20 | 0.21 / 0.047 | 0.095 / 0.019 |

The old kernel is kept as `zmr_sigma_det` (`kernels.sigma_det`); the new one
is faster in every case measured, so there is no switch between them.  With
16 threads: 8.2 ms per vector for 10 in 20 (64 vectors), 0.28 ms for 8 in
16; with one thread 2.8 ms for 8 in 16.  For 64 vectors of 10 in 20 about
70% of the time is the `zgemm` (66 x 66 per `K`), 15% the scatter, 9% the
gather and 7% collecting `W_K`; for a single vector collecting `W_K` (40%)
and the scatter (30%, cache lines moving between the threads) dominate,
which is why one vector costs seven times more than one of a batch.
What it buys: the Lanczos solver for the model states (2 roots of 8 in 16,
192 products) takes 2.7 s instead of 73 s.  The RDM route of PC-NEVPT2
applies H_act only to the model states and their intermediate states, so
its run time does not depend on this kernel: for the model
Hamiltonian `model_eris(4, 16, 6, seed=1)`, 8 electrons, 2 states, all
sectors matrix free, 4 CPUs, 507 s with the new and 510 s with the old
kernel (200 products, 0.4 s against 6.3 s; caaa 264 s, aaav 223 s, ccvv
16 s in both), effective Hamiltonians equal to 3e-15.  The time of that
route is in the density matrices of the three-operator classes.

Other limits:

* No usable symmetry, `M > D`: two Hermitian eigenproblems of dimension `D`
  per three-operator class (the Gram matrix and the Koopmans matrix).  They
  are threaded by MKL but scale worse than the level-3 parts.
* State-specific contraction lists the doubly external blocks instead of
  using the task kernels.
* Single node; the tasks of the external sums are independent and would
  map onto MPI processes.

## 6. Interpolation of the resolvent in the energy argument

With projected amplitudes `t[J, k, X]` the contribution of a class is

    a[I, J] = - sum_k t[I, k, :]^+ W(lam_k - s_J) t[J, k, :],
    W(d)[X, Y] = sum_e conj(v[e, X]) v[e, Y] / (d + delta_e),

and the exact evaluation needs one external sum `W` for every `k` and every
distinct shift `s_J`: `m` times the number of levels of the model space.
`engine.chebyshev_weights` replaces this by interpolation in `d`: `W` is
evaluated at the `n` roots of the Chebyshev polynomial `T_n` on the interval
`[lo, hi]` of the arguments `lam_k - s_J` that occur, and every `W(d)` is the
barycentric Lagrange combination of these `n` matrices.  The external labels
are then summed `n` times, whatever the number of states, and the matrices
`W` of all external blocks of a class add up before the contraction with `t`.
With three-index integrals the kernels of ccvv, cavv and aavv deliver `W` at
the nodes directly (`ExternalBlocks.w_nodes`).

**Error.**  Each term is a simple pole.  For `g(d) = 1 / (d + delta)` and
interpolation at the roots of `T_n` on `[lo, hi]` (centre `mid`, half width
`half`) the error is known in closed form,

    (g - p_n)(d) / g(d) = T_n(x) / T_n(-z_delta),
    x = (d - mid) / half,   z_delta = (delta + mid) / half,

so the relative error of every term of the external sum is at most
`1 / T_n(z)` with `z = (min(delta) + mid) / half` of the nearest pole, and
smaller for all others.  `1 / T_n(z) < 2 rho^-n`, `rho = z + sqrt(z^2 - 1)`;
the number of nodes is

    n = ceil(INTERP_DIGITS / ln(rho)) + 1,   INTERP_DIGITS = 40,

which puts the bound at `2 exp(-40) = 8e-18`, below round-off.  The
barycentric formula for these points is numerically stable (Lebesgue constant
about `(2 / pi) ln n + 1`).  For a smallest denominator `gap = min(delta) + lo`
small against the width `hi - lo` of the interval, `n` is about
`20 sqrt(width / gap)`.

**When it is not used.**  `chebyshev_weights` returns `None`, and the
contraction falls back to the kernels with exact denominators, if

* the nearest pole is inside or at the edge of the interval
  (`min(delta) + lo <= 0`, an intruder, reported as such), or
* more than `INTERP_MAX_NODES = 96` nodes would be needed
  (`gap / width` below about 0.043), or
* the interpolated form would cost more operations than the exact kernels
  (few states, few `k`).

If the distinct arguments at which amplitudes are present are no more than
the nodes, the sums are evaluated at these arguments (`resolvent_nodes`):
ccvv with state-specific E0 has a single argument, whatever the number of
states.  The theory (error formula, node count, cost criterion) is in
`resolvent-interpolation.md`.

A single needed argument (one state) is a single node, i.e. the exact sum.
`engine.INTERPOLATE = False` or the environment variable
`ZMR_NO_INTERPOLATION=1` switches the interpolation off everywhere.

**Validation** (`zmr/tests/test_interpolation.py` and calculations with the
switch on and off; 32 cores for Nd3+, 16 for the molecules):

* `1 / (d + delta)` is reproduced with a relative error below 2e-13 for
  gaps from 0.01 to 30 Eh and widths from 1e-3 to 3 Eh; with fewer nodes
  (`INTERP_DIGITS` = 5 ... 30) the error equals the formula above; refusal
  for poles inside the interval and for too many nodes; arguments on nodes
  and degenerate intervals.
* Block contractions against both exact kernels, real and complex, single
  column; the single-operator class; PC-NEVPT2 (union, state, RDM route) and
  UC-NEVPT2 on a model Hamiltonian and on water with four- and three-index
  integrals: classes and effective Hamiltonian agree to 1e-11 Eh.

| system | states | max. difference of the energies | of the levels | NEVPT2 time on / off |
|--------|--------|------|------|------|
| Nd3+ 4f^3, 5s5p correlated | 364 | 1.1e-10 Eh | 2.8e-5 cm^-1 | 16 s / 121 s |
| Nd3+ 4f^3, 4s4p4d5s5p correlated | 364 | 3.6e-11 Eh | 1.0e-5 cm^-1 | 21 s / 395 s |
| Te2, (n-1) shell correlated, PC and UC, both E0 | 3 and 6 | < 1e-10 Eh | 2.4e-6 cm^-1 | 10 s / 34 s |
| TeO, the same | 3 and 6 | < 1e-10 Eh | 1.6e-6 cm^-1 | |

The "off" times are those of the block-wise kernels with exact
denominators, which also give up the task kernels of the doubly external
classes: they are an upper bound of what the interpolation saves.  For a
few states the saving proper is small (`resolvent-interpolation.md`
section 5).  The differences are at the level of the round-off of sums over 1e6 to 2e7
external labels evaluated in a different order.  The smallest denominators
of these calculations are between 0.47 and 3.3 Eh; calculations with
denominators of a few mEh fall back to the exact kernels by construction.
The scheme was designed for this code; no literature source has been
checked for it.

## 7. Spin-restricted code with three-index integrals

`zmr.nr` uses the same scheme when the Hamiltonian is built from Cholesky or
RI vectors: real kernels `zmr_wsf_ccvv`, `zmr_wsf_cavv`, `zmr_wsf_aavv` for the
labels and parameters of the spin-free classes (tasks over core pairs, core
orbitals and virtual segments), the labels with coinciding indices
(`(ai|at)`, `(at|au)`) from single contractions of the three-index
integrals, and no `(vc|vc)`, `(vc|va)`, `(va|va)` block.  Benzene, CAS(6,6)
on RHF/AVAS orbitals, RI vectors, 4 cores, NEVPT2 step, before / after
(energies identical to all printed digits):

| basis | functions | vectors | 1 state | 3 states | peak memory |
|-------|-----------|---------|---------|----------|-------------|
| cc-pVTZ / cc-pVTZ-RI | 264 | 666 | 1.9 s / 1.1 s | 3.4 s / 1.6 s | 1.9 GB / 1.1 GB |
| cc-pVQZ / cc-pVQZ-RI | 510 | 1122 | 8.8 s / 4.8 s | 15.0 s / 6.7 s | 7.3 GB / 4.9 GB |
