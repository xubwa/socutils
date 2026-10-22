# Internally contracted MRCI in a product basis

Conventions of `conventions.md`: complex spin-orbitals / spinors, core
orbitals `i, j`, active `t, u, v, w`, virtual `a, b`; `H = E_nuc + sum h_pq
p^+ q + 1/2 sum (pq|rs) p^+ r^+ s q` with only `(pq|rs) = (rs|pq)` and
Hermiticity assumed.  Code: `zmr/mrci/`, terms from `zmr/wick/mrci.py`,
references `zmr/reference/mrci_bruteforce.py` and `icmrci_bruteforce.py`.

## 1. The space

The MRCISD space of a CAS reference is the set of determinants with at most
two core holes and two virtual particles.  It is the direct sum of nine
blocks labelled by (holes, particles), the internal block `cas` and the
eight classes of NEVPT2:

| block | external string `E_e` | active sector |
|-------|-----------------------|---------------|
| cas  | 1 | N |
| caaa | `i` | N + 1 |
| ccaa | `j i` | N + 2 |
| aaav | `a^+` | N - 1 |
| aavv | `a^+ b^+` | N - 2 |
| cava | `a^+ i` | N |
| ccav | `a^+ j i` | N + 1 |
| cavv | `a^+ b^+ i` | N - 1 |
| ccvv | `a^+ b^+ j i` | N |

A basis function is `|B; e, k> = E_e |Phi_c b_k>`: the external string
acting on the state with the core filled, the virtuals empty and the active
part `b_k`, a vector of the sector of the block.  For orthonormal `b_k` the
functions are orthonormal (independent labels `a < b`, `i < j`), and blocks
are orthogonal to each other.

The active vectors of a block are chosen block by block:

* **full**: all determinants of the sector.  With every block full the space
  is the uncontracted MRCISD space.
* **pc**: an orthonormal basis of the span of `O_X |Psi_J>` over the active
  operators `O_X` of the class and the model states `J`.  The functions
  `E_e O_X |Psi_J>` are all single and double excitations `tau |Psi_J>`
  that lie in the block, so with every external block contracted the space
  is the fully internally contracted one, the union of the first-order
  interacting spaces of the model states.
* internal block: all CAS determinants (the reference coefficients relax),
  or the model states only.

`contraction='fic'` contracts all external blocks, `'wk'` only the doubly
external ones (aavv, cavv, ccvv; the Werner-Knowles choice), `'none'`
nothing.  The energies obey `E(none) <= E(wk) <= E(fic) <= E(CAS)`.  With
many model states the union spans whole sectors and the contraction
disappears: the method is meant for a few states.

## 2. Matrix elements

With coefficients `C^B[e, k]`, antisymmetric in equivalent external indices
and summed over their full ranges,

    sigma^{B'}[e', k'] = sum_B 1 / (n_h! n_p!) sum_{e, k}
                         <Phi_c b'_k'| E_e'^+ H E_e |Phi_c b_k> C^B[e, k].

The operator string `E_e'^+ H E_e` is separated into its core/virtual part
`X` and its active part `O` (sign of the permutation).  `X` has an even
number of operators whenever the element does not vanish, so it passes
through the active string without a sign and

    <Phi_c b'| X O |Phi_c b> = <X>_0 <b'| O |b>,

where `<X>_0` is the expectation value in the state with the core filled
and the virtuals empty: a sum of full contractions, i.e. Kronecker deltas
between the external indices of bra, ket and H.  `O` is normal ordered;
what remains are the matrices of active operator strings between the block
bases,

    Gamma[k', t.., u.., k] = <b'_k'| t^+ .. u .. |b_k>,

at most three operators long.  They play the role of the density matrices
of the NEVPT2 derivation, but they are matrix elements between vectors of
the active sectors, evaluated in the determinant basis: no density matrix
of the reference appears at all, whatever the contraction.

The part of H with active indices only commutes with every external
string.  It is kept as one operator, `H_aa = sum h_tu t^+ u + 1/2 sum
(tu|vw) t^+ v^+ w u` with the bare one-electron integrals: its contribution
is the overlap of the external strings times `<b'_k'| H_aa |b_k>`.  The
mean field of the core arises from the contractions of the core operators
of H among themselves.

All block pairs: 970 terms (`generated/mrci_sigma_terms.md`, written by
`python -m socutils.zmr.wick.mrci` from the terms the code evaluates).

## 3. Validation

`zmr/tests/test_mrci.py`, random complex model Hamiltonians, several
partitions and one or two model states:

* `contraction='none'`: dimension and the four lowest energies equal those
  of the determinant MRCISD (1e-10);
* `contraction='fic'`, internal block relaxed or fixed: dimension and
  energies equal those of the brute-force internally contracted MRCI, whose
  space is built from the definition `Q tau |Psi_J>` without the class
  operators (1e-10);
* `E(none) <= E(wk) <= E(fic)`; sigma keeps the antisymmetry of the
  coefficients; the Hamiltonian is Hermitian to 1e-10.
* Davidson and dense diagonalization agree; the energies do not change
  under unitary rotations among the core or among the virtual orbitals;
  ACPF / AQCC solutions are stationary points of their functional.

### Against ORCA FIC-MRCI

Non-relativistic integrals expanded in spin orbitals, exact (no density
fitting), cc-pVDZ, ORCA 6.0.1 `! FIC-MRCI` after its own CASSCF (same
CASSCF energy as PySCF to 1e-9 Eh); `~/scr4/zmr_runs/mrci/orca_n2`.  ORCA
keeps the reference as one contracted function (number of amplitudes =
number of contracted CSFs + 1), which is `internal='states'` here.

| case | ORCA | zmr `states`, one state | zmr `cas`, one state |
|---|---|---|---|
| N2 CAS(6,6), 1s frozen | -109.26760750 | -109.2676077975 | -109.2678608970 |
| N2 CAS(6,6), all electrons | -109.27130002 | -109.2713003220 | -109.2715497625 |
| O2 triplet CAS(2,2), 1s frozen | -149.94992650 | -149.9503740771 | -149.9503740771 |

Closed-shell singlet reference: agreement to 3e-7 Eh, the convergence of
the ORCA run (residual threshold 1e-5, last energy change 2.5e-7, metric
threshold 1e-6; with `lindep=1e-6` zmr gives -109.2676076549).  The singlet
part of the span of the spin-orbital excitations of a singlet is the span
of the spin-free ones, so the two spaces are the same.  Relaxing the
reference in the CAS lowers the energy by 0.25 mEh.

Triplet reference: zmr is lower by 0.45 mEh, and by 1.17 mEh
(-149.9510917357) when the three components are the model states.  The
spin-orbital excitations of an open-shell state span more than the
spin-free ones (spin-flip excitations recoupled to the same total spin),
and the union basis of three states is larger again.  Not an error; the
comparison digit by digit is possible for singlets only.

Timings on 16 cores (ORCA 16 MPI processes, zmr with RI integrals; zmr for
one state / for the three triplet components): water CAS(4,4) cc-pVTZ 212 s
vs 20 s, cc-pVQZ 293 s vs 80 s; O2 uncontracted cc-pVTZ CAS(2,2) 225 s vs
25 / 104 s, CAS(8,6) 275 s vs 107 / 458 s (before the last change of the
sigma code, which removed a third of the time per vector).  ORCA spends
18 to 28 s per iteration independent of the size in these cases.  The
spin-adapted space is 5.5 to 6.4 times smaller than the spin-orbital one
(N2: 23465 against 128349 functions, O2 cc-pVDZ: 10875 against 69816).

## 4. Solver, corrections

The core and virtual orbitals are made quasi-canonical (generalized Fock
matrix of the averaged reference density) and every block basis
diagonalizes H_act within its span (energies `mu_k`, as in PC-NEVPT2).  The
zeroth-order energies `E_core + mu_k + sum eps_a - sum eps_i` are the
diagonal preconditioner of the block Davidson iteration
(`zmr.mrci.davidson`), which starts from the model states.  A root is
converged when its residual norm `|H x - E x|` is below `MRCI.conv_tol`,
5e-5 by default (the default of Dice); the error of the energy is of
second order in the residual, about 1e-9 Eh at that value.  Full-valence
O2: 10 iterations instead of 13 (393 s instead of 507 s), E changed by
2e-10 Eh and the splitting by less than 1e-4 cm-1; the Davidson
corrections, which use the weight of the reference and are of first order
in the residual, changed the splitting by 0.0007 cm-1.  Spaces up to
`MRCI.dense_limit` functions are diagonalized densely.

With `c0^2` the weight of the internal block in a root and `E_rot` the
energy of that part (the rotated reference), the Davidson corrections are
`E + (1 - c0^2)(E - E_rot)` and the renormalized `E + (1 - c0^2) / c0^2
(E - E_rot)`.

`functional='acpf'` / `'aqcc'`: stationary points of

    E - E_ref = <Psi| H - E_ref |Psi> / (<Psi_int|Psi_int> + g <Psi_ext|Psi_ext>),

`g = 2 / n` (ACPF) or `1 - (n - 3)(n - 2) / (n (n - 1))` (AQCC), `n` the
number of correlated electrons, `E_ref` the CASCI energy of the model state:
the eigenproblem of `M^-1/2 (H - E_ref) M^-1/2` with `M = 1` on the internal
block and `g` on the external ones, one per model state.

## 5. Evaluation of sigma

Of the 970 terms, those that are equal after antisymmetrization (C is
antisymmetric in equivalent ket indices, sigma in equivalent bra indices)
are represented by one: 391 terms, whose sum is antisymmetrized in the bra
indices at the end.

In a term only the coefficients change from iteration to iteration.  The
other factors (integrals, deltas, Gamma) are contracted once into a dressed
integral `W`, terms with the same index structure are added, and `W` is
stored as a matrix (own indices) x (indices contracted with C); an
iteration transposes C, does one matrix product per structure (`zgemm`) and
transposes the result.  Terms without active operators are an integral
tensor with up to four external indices times the overlap of the block
bases `M[L, K]` (the unit matrix within a block): the integral tensor is
kept alone and `M` is applied to the sum of these contributions, so that
no array larger than the integrals is formed.

Two cases are not precontracted.

* `W` with more than `Sigma.w_limit` elements (integrals with three virtual
  indices times a Gamma between large block bases) is not formed, unless it
  is no larger than the intermediate it saves in every iteration.  The term
  is evaluated in two steps, in the order with the smaller intermediate:
  Gamma on the last axis of the coefficients and then the integrals, or the
  integrals first when they contract external indices of the coefficients
  (the doubly external blocks) and Gamma on the result.
* The `(vv|vv)` terms, one per doubly external block and all of the form
  `sum_pq (Ap|Bq) C[p, q, ..]`, are evaluated from the three-index
  integrals `B[L, A, p]` when the block would exceed `Sigma.vvvv_memory`:
  `R_x = sum_L B_L C_x B_L^T` for every remaining label `x`, two matrix
  products of cost `naux nv^3`.

Within one sigma every permutation of the coefficients of a block is
copied once and shared by all contractions, and results are accumulated per
bra block and permutation and transposed back once.  `Sigma.timings` holds
the time by block pair and kind of contraction.

Water, CAS(4,4), O 1s frozen, fully internally contracted, one state.
6-31G and cc-pVDZ with exact integrals on 4 cores, cc-pVTZ (16 cores) and
cc-pVQZ (32 cores) with RI vectors:

| basis | virtual spin-orbitals | dimension | first version | now | peak memory |
|-------|-----------------------|-----------|---------------|-----|-------------|
| 6-31G | 12 | 8994 | 54 s | 3.8 s | |
| cc-pVDZ | 34 | 49320 | - | 15.8 s | |
| cc-pVTZ | 102 | 375924 | out of memory (60 GB) | 19.5 s | 13.5 GB |
| cc-pVQZ | 216 | 1608150 | - | 72.8 s | 22.4 GB |

The energies did not change in any step (cc-pVQZ: correlation energy
-0.23073684 Eh, 13 sigma vectors).  In the cc-pVQZ run the `(vv|vv)`
products take 19 s, the one-time setup about 25 s and transposed additions
of intermediate results about 20 s.

## 6. Memory

`Sigma.memory_report()` lists the arrays kept between iterations.  For O2
with the full valence space (4 core, 12 active, 196 virtual spinors, three
states, dimension 7379603; 32 cores) the first version kept 61.7 GB and
peaked at 112.7 GB:

| | first | after |
|---|-------|-------|
| `(vv|vv)` matrix | 23.6 GB | 5.8 GB |
| integrals with three virtual indices (in 8 layouts) | 11.6 GB | 4.3 GB |
| all arrays kept | 61.7 GB | 27.9 GB |
| peak, setup | 64.7 GB | 43.5 GB |
| peak, one sigma with three vectors | 111.3 GB | 49.7 GB |
| one sigma, one vector | 51 s | 38 s |
| whole calculation | 2784 s | 1691 s (before the packing below) |

What was done:

* the vectors of a Davidson step go through sigma one at a time;
* two-step contractions without copies: the indices of C are ordered as
  (spectators, contracted), Gamma acts on the last axis, and the integrals
  are applied as a stack of matrix products, in slices that keep the
  intermediate below `Sigma.x_limit`;
* matrices Gamma above `Sigma.gamma_limit` elements are never stored: the
  dressed integrals are accumulated from slices over the ket basis;
* the Gamma cache and the integral blocks are released after the setup;
* independent labels: matrices above `Sigma.pack_limit` elements are packed
  over antisymmetric index pairs (a contracted ket pair is summed over
  `x < y` with `W[x, y] - W[y, x]`; a bra pair that the matrix carries alone
  is stored for `A < B` as `(W[A, B] - W[B, A]) / 2` and the product is
  scattered with both signs), and the `(vv|vv)` matrix is `G[(A < B), (p <
  q)] = (Ap|Bq) - (Aq|Bp)`.

Transposed additions and copies and the gathers and scatters over index
pairs are OpenMP kernels (`zmr_transpose_add`, `zmr_pair_gather`,
`zmr_pair_scatter`); with numpy doing them on one thread the packing made
the calculations slower (water cc-pVQZ 125 s, one sigma of the O2 case
38 s).  With the kernels, 32 cores:

| | before the memory work | now |
|---|------------------------|-----|
| O2 full valence, setup | 150 s | 101 s |
| O2 full valence, one sigma, one vector | 51 s | 17 s |
| water cc-pVQZ, total | 73 s | 79 s (setup 42 s, iterations 37 s) |
| water cc-pVQZ, peak memory | 22 GB | 24 GB |

In the water case the `(vv|vv)` matrix is now kept (8.7 GB packed) instead
of being contracted from the three-index integrals, which is why its
memory did not go down.

Transients of one sigma (full-valence O2, 2026-10-05): the coefficients
times a block overlap, `C'[.., L] = sum_K M[L, K] C[.., K]`, are used by one
pair of blocks only and are released after it (they were kept for the whole
sigma), and are formed in slices of `L` above `Sigma.x_limit` elements.
Peak of the complete run 79 GB -> 58 GB (45 GB at the end of the setup
against 30 GB of kept arrays: the cached integral blocks and matrices
Gamma).  The printed output gives the peak after every block of
the setup and after the iterations, and with `timing_lines > 6` the
contractions during which the peak grew.

Memory against time: with `Sigma.w_max = 2e9` (driver option
`--mrci-wmax`) the dressed integral of aavv <- aavv is kept, 24 GB.  One
product of 3.0e11 multiply-adds replaces the two steps with 4.3e11:
4.3 s -> 2.7 s for this term, 10.7 -> 9.0 s per sigma, setup + 20 s, peak
100 GB.  Both forms run at the speed of zgemm on this node (about
0.85 TFlop/s on 32 cores), so sigma is bound by the operation count, not by
memory traffic as assumed in section 7 before.  Not the default.

## 7. Where the time is now

O2 with the full valence space (dimension 7379603, three states, 32 cores),
final code: setup 104 s (60 s after the changes of the setup described
below), one sigma 11.5 s, 39 sigma vectors (13 Davidson
iterations for three roots, 38 s each), 595 s in that run (2784 s
originally); 507 s with the setup at 50 s (`o2fv.31625304.log`).  Of the 490 s of the iterations 449 s are sigma, 16 s
the orthogonalization, 7 s the Ritz vectors and residuals, 4 s the
preconditioner.

The contributions between different blocks that have no active operator
are a tensor of integrals times the overlap `M[L, K]` of the two block
bases.  The overlap is applied to the coefficients first (once per ket
block), so that no intermediate carries the ket basis index; doing it last
built a 4.9 GB array for ccvv <- cas and cost 7.5 s per sigma.

The largest single term of one sigma, 4.6 s, is
aavv <- aavv, `sum (Ap|tu) Gamma[L, tu, K] C[p, B, K]` with 198 functions in
the aavv block: 4.3e11 complex multiply-adds in products whose inner
dimension is about 200 (see the end of section 6 for the form with the
dressed integral).  Summing over the active pairs `(t, u)` one at a time
(`Sigma.loop_limit`) needs no large intermediate but is not faster (6.2 s).
All other terms take less than 0.6 s each.

Setup, 60 s (150 s, then 104 s before): the `(vv|vv)` matrix and the dressed
integrals.

* Contractions of two operands are one threaded matrix product between
  threaded transpositions (strided views of the integral blocks are read in
  place); the copies that put the dressed integrals into matrix form,
  their antisymmetrization over pairs and the accumulation of terms go
  through the same kernels instead of single-threaded numpy copies.
* `(vv|vv)` is built for `B > A` only (half of the products).
* A matrix `Gamma[L, X, K]` is built with the strings `X` acting on the block
  with fewer functions (on the bra functions as the adjoint strings) and in
  batches of functions (`Sigma.string_limit`): `cas <- caaa` with one
  reference function and 740 functions in caaa (N2, CAS(6,6)) needed
  740 x 1728 vectors at once, 18.9 GB, now 1728.
* A `Gamma` too large to store is contracted with the integrals slice by
  slice of the ket functions.  The integrals `G[w, X]` are applied to the
  vectors `O_X b_K` before these are projected on the `m` bra functions
  when the number `n_w` of values of the external indices of the dressed
  integral is below `m`: cost `n_K n_X dim (n_w + n_w m / n_X)` instead of
  `n_K n_X dim m`.

N2 CAS(6,6) / cc-pVDZ (dimension 128349, 16 cores): 41 s with a peak of
18.2 GB before these changes, 19 s and 9.5 GB after.

## 8. Not yet

* The integrals with three virtual indices are dense blocks.
* The symbolic terms are derived at every start (2 s).
* Fewer sigma vectors: a better preconditioner than the diagonal Dyall
  energies, or exploiting the degeneracy of symmetry partners.
