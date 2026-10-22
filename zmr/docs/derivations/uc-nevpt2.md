# Fully uncontracted multi-state NEVPT2

Notation and conventions: `conventions.md`.  Implementation: `zmr/uc/`.
Validation record: `validation-log.md`.

## 1. First-order equation

With the Dyall Hamiltonian `H_D` (conventions.md section 8) and `V = H - H_D`,
the first-order wave function of model state `J` is

    (E0_J - H_D) psi1_J = Q V |Psi_J>

and the second-order effective Hamiltonian is assembled from
`a_IJ = <I|V R(E0_J) V|J>`, `R(E) = Q (E - H_D)^{-1} Q`, as described in
conventions.md section 9.

`H_D |Psi_J>` stays in the CAS space (it equals `H` there), so

    Q V |Psi_J> = Q H |Phi_c Psi_J>,

where `|Phi_c Psi_J>` denotes the determinant expansion with the core filled
and the virtuals empty.  Only `H` has to be analysed.

## 2. Normal ordering with respect to the filled core

Normal-ordering `H` with respect to `|Phi_c>` (the core-filled state; active
and virtual orbitals are treated as particles with respect to the physical
vacuum) gives

    H = E_core + sum_pq fc_pq {p^+ q} + 1/2 sum_pqrs (pq|rs) {p^+ r^+ s q}
    fc_pq = h_pq + sum_i <pi||qi>

In a normal-ordered string, core creation operators stand to the right and
annihilate `|Phi_c>`; virtual annihilation operators stand to the right and
annihilate every CAS function.  Hence, acting on `|Phi_c Psi>`, **only strings
in which every core index is an annihilator and every virtual index is a
creator survive**, and active operators act on `Psi` directly.  For strings
without repeated core indices `{...}` is the plain product up to reordering,
so the contractions over the core are exactly the ones collected in `fc` and
`E_core`.

Writing the two-electron part as `1/4 sum <pr||qs> p^+ r^+ s q`
(`<pr||qs> = (pq|rs) - (ps|rq)`) and sorting the surviving strings by the
number of core annihilators (holes `h`) and virtual creators (particles `p`)
gives eight classes, `(h, p) != (0, 0)`.

## 3. Right-hand sides

Each class is written as `sum_e E_e sum_X v[e, X] O_X |Phi_c Psi>` with a
canonical external string `E_e` placed to the left of the active string `O_X`.
Moving `E_e` to the left produces the signs below (every move is past
operators on different orbitals).

| class | (h, p) | dN_act | E_e | O_X | v[e, X] |
|-------|--------|--------|-----|-----|---------|
| ccvv | (2, 2) | 0  | a^+ b^+ j i  (i<j, a<b) | 1 | <ab‖ij> |
| ccav | (2, 1) | +1 | a^+ j i  (i<j) | t^+ | <at‖ij> |
| cavv | (1, 2) | -1 | a^+ b^+ i  (a<b) | t | -<ab‖it> |
| ccaa | (2, 0) | +2 | j i  (i<j) | t^+ u^+  (t<u) | <tu‖ij> |
| aavv | (0, 2) | -2 | a^+ b^+  (a<b) | u t  (t<u) | <ab‖tu> |
| cava | (1, 1) | 0  | a^+ i | 1 ; t^+ u | fc_ai ; <at‖iu> |
| caaa | (1, 0) | +1 | i | t^+ ; t^+ u^+ v  (t<u) | -fc_ti ; -<tu‖iv> |
| aaav | (0, 1) | -1 | a^+ | t ; u^+ v t  (t<v) | fc_at ; <au‖tv> |

Examples of the bookkeeping:

* ccav: the two orderings of the creators give
  `1/2 sum_{atij} <at||ij> a^+ t^+ j i`; restricting to `i < j` removes the 1/2,
  and `a^+ t^+ j i = a^+ j i t^+` (t^+ moves past two operators).
* cavv: `1/2 sum <ab||it> a^+ b^+ t i`, `a < b` removes the 1/2, and
  `t i = -i t`.
* caaa: `t^+ u^+ v i = -i t^+ u^+ v` (three moves), and the one-electron term
  `fc_ti t^+ i = -fc_ti i t^+`.
* Two active creators (or two annihilators) carry an ordered pair of indices,
  `t < u` in ccaa, aavv and caaa and `t < v` in aaav, which removes the factor
  1/2 of the unrestricted sum just as for the external pairs.  The strings
  `O_X` of a class are therefore non-redundant as operators: none is a multiple
  of another one.  (As functions `O_X |Psi_J>` they can still be linearly
  dependent; that is the business of the PC metric, pc-nevpt2.md.)

The functions `E_e |Phi_c k>`, with `k` running over the eigenstates of `H_act`
in the sector `N + dN_act`, are orthonormal, and `H_act` (an even operator on
the active orbitals) commutes with `E_e`, so

    H_D E_e |Phi_c k> = (E_core + lambda_k + delta_e) E_e |Phi_c k>
    delta_e = sum(eps of the virtual creators) - sum(eps of the core annihilators)

## 4. Class energies

With `T^J[k, X] = <k|O_X|Psi_J>` and `u^J[e, k] = sum_X v[e, X] T^J[k, X]`,

    a_IJ(class) = - sum_{e,k} conj(u^I[e,k]) u^J[e,k] / (lambda_k + delta_e - s_J),
    s_J = E0_J - E_core.

The sum over `k` runs over the full spectrum of the sector, i.e. this is the
exact resolvent of `H_D`; no approximation is made in the `exact` engine.

For ccvv, `O = 1` and `T^J[k] = <k|Psi_J>`; for CASCI eigenstates
`a_IJ = delta_IJ * (-sum |<ab||ij>|^2 / (eps_a + eps_b - eps_i - eps_j + E_J - E0_J))`,
which is the MP2-like expression of every NEVPT2 variant.

## 5. Laplace separation

For a positive denominator range `D in [D_min, D_max]` (computed exactly from
the sector spectrum, the orbital energies and the shifts),

    1/D ~= sum_q w_q exp(-t_q D)

and with `c = min lambda_k`, `c' = min s_J`,

    a_IJ ~= - sum_q w_q exp(t_q (s_J - c')) sum_XY M_q[X, Y] G_q^{IJ}[X, Y]
    M_q[X, Y]      = sum_e conj(v[e, X]) v[e, Y] exp(-t_q (delta_e + c - c'))
    G_q^{IJ}[X, Y] = sum_k conj(T^I[k, X]) T^J[k, Y] exp(-t_q (lambda_k - c))

`M_q` contains the sum over external labels (built like a Laplace-MP2
intermediate); `G_q` contains only active-space quantities.  The shifts `c`,
`c'` keep every exponential at most one.  This is needed for ccav and cavv,
whose three external indices make the explicit sum over `(e, k)` too
expensive; the other classes can use either engine.  If `D_min <= 0` (an
intruder state), the transform does not exist and the code raises an error.

The quadrature is the trapezoidal rule after the double-exponential
substitution `t = exp(u - exp(-u))`, with the truncation interval chosen per
`(D_max / D_min, n)`.  Its relative error decreases exponentially with `n`;
minimax exponential sums would need fewer points and can be added behind the
same interface.

## 6. Multi-state effective Hamiltonian

`H_eff = <I|H|J> + H2_IJ` with `H2 = 1/2 (a + a^+)` for `e0='state'` and
`H2 = a` for `e0='average'` (conventions.md section 9).  Both are Hermitian.
Rotational invariance of the model space requires `e0='average'` and equal
weights.
