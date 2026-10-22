# Non-relativistic (spin-free) UC- and PC-NEVPT2

Implementation: `zmr/nr/`.  The Dyall Hamiltonian, the eight classes, the
contraction engines and the multi-state effective Hamiltonian are those of
`uc-nevpt2.md` and `pc-nevpt2.md`; this note covers what changes with real
spatial orbitals and a closed-shell core.

## 1. Setting

Spatial orbitals: core `i, j` (doubly occupied), active `t, u, v`, virtual
`a, b`; `E_pq = sum_s p^+_s q_s`.

    H = ecore + sum_pq h_pq E_pq + 1/2 sum_pqrs (pq|rs) (E_pq E_rs - delta_qr E_ps)

    f_pq  = h_pq + sum_i [2 (pq|ii) - (pi|iq)] + sum_tu D_tu [(pq|tu) - 1/2 (pu|tq)]
    E_core = ecore + 2 sum_i h_ii + sum_ij [2 (ii|jj) - (ij|ji)]

`D_tu = <E_tu>` is the spin-summed 1-RDM averaged over the model states; the
core and virtual orbitals are quasi-canonicalized with `f`, and `fc`
(`D = 0`) enters the one-body parts of the right-hand sides.  Because `f` is
spin-free, `H_D` commutes with the total spin.

The active space works with determinants `|I_alpha I_beta>` of a fixed
`(n_alpha, n_beta)`; a determinant is the bit string
`(alpha string << n) | beta string`, its address is
`rank(alpha) * dim(beta) + rank(beta)`.  Order and phases are those of
`pyscf.fci`, whose CI vectors can be used directly (`pyscf_nr.nr_civecs`).
All model states live in one sector `(n_alpha, n_beta)`; they may have
different total spin.

## 2. Spin-free form of the classes

For a spatial external label `e` the part of `V|Psi>` in a class is

    sum_P x[e, P] F_P |Phi_c Psi>

with spatial integrals `x[e, P]` and products `F_P` of spin-free excitation
operators:

| class | label | x[e, P] : F_P |
|-------|-------|---------------|
| ccvv | (a<b; i<j) etc. | (ai\|bj), (aj\|bi) : no active operator |
| ccav | (a; i<j) | (ai\|tj) : E_ai E_tj ;  (aj\|ti) : E_aj E_ti |
|      | (a; i=j) | (ai\|ti) : E_ai E_ti |
| cavv | (a<b; i) | (ai\|bt) : E_ai E_bt ;  (at\|bi) : E_at E_bi |
|      | (a=b; i) | (at\|ai) : E_at E_ai |
| ccaa | (i<j) | (ti\|uj) : E_ti E_uj, all (t, u) |
|      | (i=j) | (ti\|ui) : E_ti E_ui, t <= u |
| aavv | (a<b) | (at\|bu) : E_at E_bu, all (t, u) |
|      | (a=b) | (at\|au) : E_at E_au, t <= u |
| cava | (a; i) | fc_ai : E_ai ;  (ai\|tu) : E_ai E_tu ;  (au\|ti) : E_ti E_au |
| caaa | (i) | fc_ti : E_ti ;  (ti\|uv) : E_ti E_uv |
| aaav | (a) | fc_at : E_at ;  (at\|uv) : E_at E_uv |

The functions `F_P |Psi>` are the perturber functions of spin-free PC-NEVPT2
(Angeli, Cimiraglia and Malrieu, J. Chem. Phys. 117, 9138 (2002)).  Every
operator term of the Hamiltonian is its own parameter: `(ai|tu)` and
`(ai|ut)` belong to the different functions `E_ai E_tu |Psi>` and
`E_ai E_ut |Psi>` although the integrals are equal for real orbitals.  Labels
with coinciding external indices have fewer functions (`E_ti E_ui` is
symmetric in `t, u`) and are separate cases.

## 3. Spin patterns

Written in spin-orbitals, `F_P = sum_p E_(e,p) A_(P,p)`: a sum over the spin
patterns `p` of the external string.  `E_(e,p)` is a string of spin-orbital
core annihilators and virtual creators, and `A_(P,p)` an active operator
that leads to one sector `(n_alpha + da_p, n_beta + db_p)`.  The patterns
and operators follow from

    <p_s r_t || q_s' s_t'> = (pq|rs) d(s,s') d(t,t') - (ps|rq) d(s,t') d(t,s')

applied to the spin-orbital classes of `uc-nevpt2.md` (`s'`: the spin
opposite to `s`; both values of `s` occur):

| class | external pattern | sector change | A for the parameters, in order |
|-------|------------------|---------------|--------------------------------|
| ccav (a; i<j) | a_s <- i_s j_s | +s | t^+_s ; -t^+_s |
|  | a_s' <- i_s j_s' | +s | 0 ; -t^+_s |
|  | a_s' <- j_s i_s' | +s | -t^+_s ; 0 |
| ccav (a; i=j) | a_s' <- i_s i_s' | +s | -t^+_s |
| cavv (a<b; i) | a_s b_s <- i_s | -s | -t_s ; t_s |
|  | a_s b_s' <- i_s' | -s | 0 ; t_s |
|  | b_s a_s' <- i_s' | -s | t_s ; 0 |
| cavv (a=b; i) | a_s a_s' <- i_s' | -s | t_s |
| ccaa (i<j) | i_s j_s | +2s | t^+_s u^+_s |
|  | i_A j_B | +A +B | t^+_A u^+_B |
|  | j_A i_B | +A +B | u^+_A t^+_B |
| ccaa (i=j) | i_A i_B | +A +B | t^+_A u^+_B + [t<u] u^+_A t^+_B |
| aavv (a<b) | a_s b_s | -2s | u_s t_s |
|  | a_A b_B | -A -B | u_B t_A |
|  | b_A a_B | -A -B | t_B u_A |
| aavv (a=b) | a_A a_B | -A -B | u_B t_A + [t<u] t_B u_A |
| cava | a_s <- i_s | 0 | 1 ; E_tu ; -t^+_s u_s |
|  | a_s' <- i_s | +s -s' | 0 ; 0 ; -t^+_s u_s' |
| caaa | i_s | +s | -t^+_s ; -t^+_s E_uv |
| aaav | a_s | -s | t_s ; E_uv t_s |

(`A`, `B`: alpha, beta.)  External strings of different patterns are
orthonormal, and `H_D` does not change the external string, so with
`|k_p>` the eigenstates of `H_act` in the sector of pattern `p` the functions
`E_(e,p) |Phi_c k_p>` are eigenfunctions of `H_D` with eigenvalue
`E_core + lambda_k + delta_e`, exactly as in the spinor case.

## 4. UC and PC

Represent `F_P |Psi>` by the tuple of CI vectors `(A_(P,p) |Psi>)_p`.

* **UC**: every pattern is an independent problem in its whole sector,

      a_IJ = - sum_p sum_{e,k} conj(u^I_p[k,e]) u^J_p[k,e] / (lambda_k + delta_e - s_J),
      u^J_p[k,e] = <k_p| sum_P x[e,P] A_(P,p) |Psi_J>.

* **PC**: the tuples are stacked into one vector of the direct sum of the
  pattern sectors, with `H_act = (+)_p H_act(sector of p)`.  The span of the
  stacked functions (of all model states for `'union'`, of the ket for
  `'state'`) is orthonormalized and `H_act` diagonalized in it as in
  `pc-nevpt2.md` section 2; the result replaces the sector eigenstates.  One
  amplitude per function `F_P |Psi>` is shared by all its spin components:
  this is the spin-free first-order space.

Contracting the single spin-orbital strings of `uc-nevpt2.md` instead (what
the spinor code does when it is given a spin-free Hamiltonian) lets the
components of different patterns vary independently.  For a singlet model
state that makes no difference: the larger space is invariant under spin
rotations, `V|Psi>` is a singlet and `H_D` is spin-free, so the first-order
function is the singlet-coupled part, which is the spin-free space.  For
`S > 0` and a single spin component it is a different, less contracted
method whose result depends on the component.  The spin-free contraction
gives the same energies for every `M_S` (tested for `M_S = 1, 0, -1`) and is
the one of the literature and of ORCA's FIC-NEVPT2.

## 5. Arithmetic and cost

Integrals, CI vectors and all intermediates are real (`dgemm`, `dsyrk`,
`dsyevd`, real Gram kernels); complex model states are accepted and switch
the arithmetic.  The cost model of `performance.md` applies with the sector
dimensions `C(n, n_alpha) C(n, n_beta)`, the stacked dimension in PC, and
real integral blocks over spatial orbitals.

Sectors of up to `ActiveSpace.dense_limit` (8000) determinants keep a dense
Hamiltonian.  Larger sectors are matrix free: the span construction needs
`H_act` only applied to the orthonormalized functions (`H Q`, one sigma
vector per function, OpenMP kernel `zmr_sigma_ab`), so PC-NEVPT2 with given
model states (the CASSCF CI vectors) runs without any `D x D` matrix.  The
full spectrum of a sector (UC-NEVPT2) and the built-in solver for the model
states are available for dense sectors only.  The kernel uses the
Slater-Condon rules (every connected determinant once) and processes the
vectors in cache-sized blocks of columns.  N2 with CAS(10,10) (63504
determinants in the `M_S = 0` sector): 14 s on 48 threads of one node (23 s
on 16), 5.3 GB; `E_PC <= E_SC(PySCF)` in every class, ccvv equal to 4e-15.

## 6. Density-matrix route (`amplitudes='rdm'`, `zmr/nr/rdm.py`)

The contracted space of a class needs the metric and the Koopmans matrix of
its functions,

    S[I P, J Q] = <I| F_P^+ F_Q |J>,    K[I P, J Q] = <I| F_P^+ H_act F_Q |J>,

(`F_P^+ F_Q` with the core and virtual operators contracted) and then only
the projected amplitudes `t[I, k, P] = sum_{L,Q} c[(L,Q), k] S[L Q, I P]`,
with `c`, `mu_k` from `S` and `K` as in `pc-nevpt2.md` section 3.  The
CI-vector route gets `K` by applying `H_act` to every orthonormalized
function (`n + n^3` products of cost `D n^4` each for caaa and aaav).  The
explicit formula for `K` of these two classes contains the 4-RDM.  Following
Chatterjee and Sokolov (J. Chem. Theory Comput. 16, 6343 (2020), appendix),
the 4-RDM terms are transition quantities of 3-RDM cost between the model
states and intermediate states that carry the contraction with the
integrals.  (Kollmar et al., J. Chem. Phys. 155, issue 23 (2021), "An
efficient implementation of the NEVPT2 and CASPT2 methods avoiding
higher-order density matrices", is a related scheme; it was not consulted
for this implementation.)

**Operator bases.**  For every class and every sector reached by its
patterns, basis operators `W_w` are chosen such that the pattern operators
of section 3 are short combinations, `A_(P,p) = sum_w W_w C_p[w, P]`
(`rdm.coefficients`, read off the patterns of `zmr.nr.classes`):

| class | sector change | basis `W_w` |
|-------|---------------|-------------|
| ccvv | 0 | 1 |
| ccav | +s | t^+_s |
| cavv | -s | t_s |
| ccaa | +s +s' | t^+_s u^+_s', (s, s') = (A, A), (B, B), (A, B) |
| aavv | -s -s' | u_s' t_s, (s', s) = (A, A), (B, B), (B, A) |
| cava | 0 | 1, t^+_A u_A, t^+_B u_B |
|      | +s -s' | t^+_s u_s' |
| caaa | +s | t^+_s, t^+_s E_uv |
| aaav | -s | t_s, E_uv t_s |

    G[I w, J w'] = <I| W_w^+ W_w' |J>,     Kw[I w, J w'] = <I| W_w^+ H_act W_w' |J>
    S = sum_p C_p^T G C_p,                 K = sum_p C_p^T Kw C_p

`G` is the Gram matrix of the vectors `W_w |J>`: a spin-summed contraction
of the transition RDMs of rank up to three (e.g. caaa:
`<I| E_vu t_s t'^+_s E_u'v' |J>`), obtained in one product of cost `D n^6`
without forming the RDM.

**Intermediate states.**  Write `W = y_1 y_2` (`y_1` the first factor of the
table, `y_2` the rest).  Then

    H y_1 y_2 |J> = y_1 (H y_2 |J>) + [H, y_1] y_2 |J>
                  = y_1 (H y_2 |J>) +- y_2 ([H, y_1] |J>) + [[H, y_1], y_2} |J>

(`-` and the anticommutator `{` if `y_2` is a single fermion operator, `+`
and the commutator if it is `E_uv`).  The intermediate states are

    H |J>,   H t^+_s |J>,   H t_s |J>           (1 + 4 n products with H_act per state)
    [H, t^+_s] |J> = H t^+_s |J> - t^+_s H |J>,   [H, t_s] |J> likewise
    H E_uv |J>

and `[H, y_1]`, which is where the integrals are contracted with three
operators, acts on `|J>` only:

    [H, t^+_s] = sum_p h_pt p^+_s + sum_pqr (pt|rq) p^+_s E_rq
    [H, t_s]   = - sum_p h_tp p_s - sum_pqr (tp|qr) E_qr p_s

The double (anti)commutators have at most as many operators as `W` and are
combinations of the basis functions, `[[H, y_1], y_2} |J> = sum_w W_w |J> M[w, (y_1 y_2)]`:

    ccaa   {[H, t^+_s], u^+_s'} =   sum_pr (pt|ru) p^+_s r^+_s'
    aavv   {[H, u_s'], t_s}     =   sum_pr (up|tr) p_s' r_s
    cava   {[H, t^+_s], u_s'}   =   d(s,s') [h_ut + sum_rq (ut|rq) E_rq] - sum_pq (pt|uq) p^+_s q_s'
    caaa   [[H, t^+_s], E_uv]   = - h_vt u^+_s + sum_pr (pt|ru) p^+_s E_rv
                                  - sum_pq (pt|vq) p^+_s E_uq - sum_rq (vt|rq) u^+_s E_rq
    aaav   [[H, t_s], E_uv]     = - h_tu v_s - sum_qr (tu|qr) E_qr v_s
                                  - sum_pq (tp|qu) E_qv p_s + sum_pr (tp|vr) E_ur p_s

(`h = heff`, `(pq|rs)` the active integrals, `H_act` as in section 1; they
follow from `[E_pq, t^+_s] = d(q,t) p^+_s`, `[E_pq, t_s] = - d(p,t) q_s` and
`[E_rq, E_uv] = d(q,u) E_rv - d(r,v) E_uq`.)  For aaav the basis operator is
`E_uv t_s = t_s E_uv - d(u,t) v_s`, so its vectors are those of `t_s E_uv`
minus `d(u,t)` times those of `v_s`.  Hence

    Kw = W^T R + G M,
    R  = y_1 (H y_2 |J>) +- y_2 ([H, y_1] |J>)

and for the one-operator classes simply `R = H y |J>`.  The 4-RDM of the
explicit formula is in `W^T [y_2 ([H, y_1] |J>)]` for caaa and aaav
(`<I| E_v'u' t'_s E_uv (pt|rq) p^+_s E_rq |J>`, eight operators): it is
evaluated as one matrix product between the `n + n^3` vectors `W |I>` and the
`n^3` vectors `E_uv` applied to the `n` intermediate states `[H, t^+_s] |J>`,
cost `D n^6`.  `Kw` is symmetric; only one triangle of `W^T R` is computed
(`dgemmt`) and mirrored.  `H E_uv |J>` needs no further products with
`H_act` (the cava formula summed over the spin):

    H E_uv |J> = sum_s [u^+_s (H v_s |J>) - v_s ([H, u^+_s] |J>)]
                 + 2 h_vu |J> + sum_pq [2 (vu|pq) - (pu|vq)] E_pq |J>

No invariance of the model space under `H_act` is assumed (`H |J>` enters as
a vector), so the route is valid for any real model vectors, like the
CI-vector route.  For `contraction='state'` only the blocks `Kw[J, J]` are
formed.

These formulas were derived by hand from the three commutators above, not
with the Wick engine of `zmr/wick` (which is spin-orbital and would give the
RDM form `K = sum (integrals) x (RDMs)`, with spin-resolved 3-RDMs of
`(2 n)^6` elements per spin block; the vector form above needs none).  They
are checked against the brute-force matrices `zmr.reference.nr_koopmans`
(`H_act` applied to every function) for every class, for model vectors that
are not eigenstates, and at run time by the symmetry of `Kw`
(`pt.rdm_check = True` forms all of `Kw` and stores its asymmetry in
`pt.rdm_info`).

**Spin-flip symmetry.**  For `n_alpha = n_beta`, if every model state goes
into `p_J = +-1` times itself when its alpha and beta strings are exchanged
(`Ms = 0` states of definite spin), the matrices of the sector with alpha
and beta exchanged are `p_I p_J` times those already computed and are not
formed again (caaa, aaav, ccav, cavv: half the work).  The symmetry is used
if `max |C^T -+ C| <= pt.rdm_flip_tol` (1e-9); the matrices taken over are
then off by that order.  CI vectors from an iterative solver often have it
only to the convergence of the solver (1e-7 was seen for a state-averaged
PySCF CASSCF), in which case both sectors are computed.

**Cost** (per state pair, caaa and aaav): the vectors `W |J>` and `R |J>`,
`2 (n + n^3) D` numbers; one `dsyrk` and one `dgemmt` of dimension
`n + n^3` over `D` determinants per sector; two eigenproblems of dimension
`nstate (n + n^3)`.  With `S` states the products grow as `S^2` and the
eigenproblems as `S^3`; when the functions of all states outnumber the
determinants (`S n_X >= D`) the CI-vector route, which then works in the
whole sector, is the cheaper one.  `amplitudes='auto'` (the default of
`zmr.nr.PCNEVPT2`) decides class by class with this criterion.  Numbers:
`performance.md` section 5.

The vectors of one sector are held in memory at once (two arrays
`S (n + n^3) D`: 0.9 GB for CAS(10,10) with one state, about 20 GB for
CAS(12,12)); they are not processed in batches of determinants.

## 7. Validation

`test_nr_active`, `test_nr_nevpt2`, `test_nr_rdm` (both routes against each
other and against the brute-force matrices); numbers in `validation-log.md`:
agreement with the spinor code on the spin-orbital form of the same
Hamiltonian (UC always, PC for singlets), independence of `M_S`, PySCF
(strongly contracted bound), ORCA FIC-NEVPT2 for a closed-shell molecule and
two triplets.
