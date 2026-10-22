# Conventions

Every formula in the code and in the other documents follows the conventions
below.  Anything that is still to be pinned down numerically is marked
**(verified by test ...)** and must not be relied upon before that test exists.

## 1. Orbitals and index letters

All orbitals are **spin-orbitals or spinors**.  A non-relativistic calculation
is the special case in which real spatial integrals are expanded into
spin-orbitals (index `p = 2*mo + sigma`, `sigma = 0` for alpha, `1` for beta).
Nothing assumes spin symmetry, time-reversal symmetry or real orbitals.

The correlated orbitals are ordered as

| space   | letters        | count  |
|---------|----------------|--------|
| core    | i, j, k, l     | `nc`   |
| active  | t, u, v, w, x, y, z | `na` |
| virtual | a, b, c, d     | `nv`   |
| general | p, q, r, s     | `nmo = nc + na + nv` |

Frozen core orbitals do not appear at all: their contribution is folded into
the scalar `ecore` and the one-electron matrix `h` by the interface layer
(section 3).  "Core" always means the *correlated* core.

## 2. Hamiltonian and integrals

    H = ecore + sum_pq h_pq p^+ q + 1/2 sum_pqrs (pq|rs) p^+ r^+ s q

with chemists' notation

    (pq|rs) = int phi_p^*(1) phi_q(1) r12^-1 phi_r^*(2) phi_s(2)

Only the following relations are used anywhere:

    h_pq    = h_qp^*
    (pq|rs) = (rs|pq)
    (pq|rs)^* = (qp|sr)

The 8-fold permutational symmetry of real integrals is **never** assumed.
The antisymmetrized integral is `<pr||qs> = (pq|rs) - (ps|rq)`.

Two-electron integrals are accessed only as blocks of the `eris` object,
`eris.cvcv` (or `eris.eri_block('cvcv')`), named by orbital-space letters as
in PySCF mcscf (`p` = all correlated orbitals).  In stage 1 that block is sliced
from a stored four-index array.  Density fitting will replace the
source of the blocks without touching any caller.

## 3. Frozen core

With frozen orbitals `f`, the interface layer returns

    ecore = E_nuc + sum_f h0_ff + 1/2 sum_fg [(ff|gg) - (fg|gf)]
    h_pq  = h0_pq + sum_f [(pq|ff) - (pf|fq)]

where `h0` is the bare one-electron operator.  The frozen orbitals are then
dropped.

## 4. Determinants and phases

A determinant with occupied orbitals `q_1 < q_2 < ... < q_n` is

    |q_1 q_2 ... q_n> = a^+_{q_n} ... a^+_{q_2} a^+_{q_1} |vac>

(the lowest index acts first).  Consequently

    a^+_p |I> = (-1)^{n_>(p)} |I + p>,   a_p |I> = (-1)^{n_>(p)} |I - p>

where `n_>(p)` is the number of occupied orbitals in `|I>` with index greater
than `p`.  This is the phase convention of `pyscf.fci.cistring` (`cre_sign`,
`des_sign`) and therefore of `socutils.fci.zfci`.  CI vectors from those
modules can be used directly.  **(verified by `test_dets`)**

Determinants are stored as integer bit strings (bit `p` set means orbital `p`
is occupied).  CI vectors are complex.

## 5. Reduced density matrices

    gamma_pq   = <Psi| p^+ q |Psi>
    Gamma_pqrs = <Psi| p^+ r^+ s q |Psi>

so that `E = ecore + sum h_pq gamma_pq + 1/2 sum (pq|rs) Gamma_pqrs`.
Transition quantities have the bra state on the left, `<I| ... |J>`, and the bra
CI vector enters complex-conjugated.  `zfci.make_rdm12_dets` and `fci_dhf_slow.make_rdm12`
(with the default `reorder=True`) use exactly the same index order; no
transposition is needed.  **(verified by `test_rdm_bydef`)**

## 6. Model space

The model space is spanned by `nstate` CASCI states `|Psi_I>` (core fully
occupied, virtuals empty, complete active space).
State weights `w_I` (default: equal) define the averaged active 1-RDM

    gammabar_tu = sum_I w_I <Psi_I| t^+ u |Psi_I>

## 7. Generalized Fock matrix and quasi-canonical orbitals

    f_pq = h_pq + sum_i [(pq|ii) - (pi|iq)] + sum_tu [(pq|tu) - (pu|tq)] gammabar_tu

The core-core and virtual-virtual blocks of `f` are diagonalized (eigenvalues
`eps_i`, `eps_a`), and all integrals are rotated to these quasi-canonical
orbitals.  The active block is not rotated: the Dyall Hamiltonian treats the
active space exactly, so the results do not depend on active orbital rotations.
The off-diagonal blocks `f_ct`, `f_tv`, `f_cv` are generally non-zero (they
vanish only for a converged state-averaged CASSCF) and are part of `V`.

## 8. Dyall zeroth-order Hamiltonian

    H_D = sum_i eps_i i^+ i + sum_a eps_a a^+ a + H_act + C

    H_act = sum_tu h^eff_tu t^+ u + 1/2 sum_tuvw (tu|vw) t^+ v^+ w u
    h^eff_tu = h_tu + sum_i [(tu|ii) - (ti|iu)]
    C = E_core - sum_i eps_i
    E_core = ecore + sum_i h_ii + 1/2 sum_ij [(ii|jj) - (ij|ji)]

On the CAS space (core full, virtuals empty), `H_D` coincides with `H`, so
`H_D |Psi_I> = E_I |Psi_I>` with `E_I` the CASCI energy, and `V = H - H_D` has
no CAS-CAS block.  `H_D` conserves the number of electrons in each of the three
spaces separately, and its eigenvalue on a perturber with core holes `h`,
virtual particles `p` and active eigenstate `|k>` of the corresponding
electron-number sector is

    E_core + sum_{a in p} eps_a - sum_{i in h} eps_i + lambda_k

where `lambda_k` is the eigenvalue of `H_act` (without `E_core`).

## 9. Multi-state second-order effective Hamiltonian

With `Q` the projector onto the complement of the CAS space and
`R(E) = Q (E - H_D)^{-1} Q`,

    H_eff_IJ = E_I delta_IJ + H2_IJ

Two choices of `E0` are supported:

* `e0='state'` (default):
  `H2_IJ = 1/2 [<I|V R(E_J) V|J> + <I|V R(E_I) V|J>]`, which is Hermitian and
  reduces to state-specific NEVPT2 for a single state.
* `e0='average'`: a single `Ebar = sum_I w_I E_I` is used,
  `H2_IJ = <I|V R(Ebar) V|J>`.

The final energies are the eigenvalues of `H_eff`.

**Invariance.**  Under a unitary rotation of the model states, `H_eff`
transforms as `U^+ H_eff U` (eigenvalues unchanged) only if both `gammabar` and
`E0` are invariant.  That holds for equal weights with `e0='average'`, where
`E_I delta_IJ` is replaced by the model-space matrix `<I|H|J>`.  `e0='state'`
needs the model states to be CASCI eigenstates (their energies enter the
resolvents), and unequal weights change `gammabar`; neither is rotation
invariant.

**Separability.**  The two choices differ in how splittings inside the model
space are treated.  The diagonal element of a class is
`a_JJ(s) = - sum_{e,k} |u^J_ek|^2 / (lambda_k + delta_e - s)` with
`d a_JJ / d s = -N_J`, `N_J = <psi1_J|psi1_J>` the norm of the first-order
function, so to first order in the spread of the model energies

    E_J(average) = E_J(state) + N_J (E_J^CAS - Ebar).

With `e0='state'` an excitation that does not involve the active electrons
(ccvv on a CASCI eigenstate, or the correlation of a distant closed-shell
fragment) has the denominator `delta_e` for every model state and cancels in
energy differences: the splittings are separable.  With `e0='average'` the same
excitation has the denominator `delta_e + E_J^CAS - Ebar`, and every splitting
of the model space gains `N` times its CASSCF value; `N` grows with the
number of correlated electrons.  This is the price of the common `E0`; it is
exact only for degenerate model states.  Example (O2 X 3Sigma_g-, CAS(2e, 4
spinors), `validation-log.md`): the ccvv class contributes exactly zero to the
zero-field splitting with `e0='state'` and a spurious positive amount with
`e0='average'`, and the two splittings differ by `N` times the CASSCF
splitting (`N = 0.13`).

For level splittings use `e0='state'` with CASCI eigenstates as model states.
Degenerate eigenstates share their `E0`, so that choice is still invariant
under rotations within degenerate levels; it is not invariant under rotations
that mix levels.
