# Wick engine

Implementation: `zmr/wick/algebra.py` (algebra), `zmr/wick/numeric.py`
(numerical evaluation), `zmr/wick/pc_nevpt2.py` (PC-NEVPT2 derivation and
code generation).  Conventions: `conventions.md`.

## 1. Scope

The engine evaluates matrix elements

    <Phi_c Psi_I| o_1 o_2 ... o_m |Phi_c Psi_J>

of arbitrary strings of creation/annihilation operators on core (`c`), active
(`a`) and virtual (`v`) spin-orbitals or spinors, where `|Phi_c Psi>` has the
core filled, the virtuals empty and an arbitrary active part.  The result is a
sum of terms

    coefficient x Kronecker deltas x tensors (integrals) x <I| active string |J>

with the active matrix element in normal order, i.e. an element of a
transition RDM.  Indices are either free (kept) or dummies (summed).

## 2. Factorization

Orbitals are ordered core < active < virtual, and a determinant is
`a^+_{q_n} ... a^+_{q_1} |vac>` with `q_1 < ... < q_n`, so the phase of a
single operator counts the occupied orbitals above it.

1. Operators on different orbitals anticommute.  The string is reordered into
   `E A` (all core/virtual operators `E` to the left of all active operators
   `A`, relative orders preserved) with the sign of the permutation.
2. `A` acts on `|Phi_c Psi_J>` with exactly the phases it has in the active
   space alone: the orbitals above an active orbital are active or virtual,
   and the virtuals are empty.
3. In `E`, every core operator counts the active electrons in its phase, but
   a non-vanishing `<Phi_c| E |Phi_c>` contains an even number of core
   operators (each core annihilator is matched by a creator), so these
   factors cancel.  Virtual operators count only virtual orbitals.

Hence `<Phi_c Psi_I| E A |Phi_c Psi_J> = <Phi_c|E|Phi_c> <Psi_I|A|Psi_J>`.
The core and virtual parts are separated the same way (with the sign of the
permutation) and evaluated by full contraction with

    core (filled):    <i^+ j> = delta_ij,  <i j^+> = 0
    virtual (empty):  <a b^+> = delta_ab,  <a^+ b> = 0

summing over all complete pairings with the sign of each pairing.

## 3. Active normal order and RDMs

The active string is brought to normal order by repeatedly replacing an
adjacent pair `a_q a_p^+` by `delta_qp - a_p^+ a_q`; creators and annihilators
keep their relative order.  A normal-ordered string
`c_1^+ ... c_k^+ a_1 ... a_k` (equal numbers, since bra and ket have the same
electron number) is the RDM element

    dm_k[I, J, c_1, a_k, c_2, a_(k-1), ..., c_k, a_1]

in the layout of `zmr.active.rdm` (`dm_k[p1, q1, ..., pk, qk] =
<p1^+ ... pk^+ qk ... q1>`, the PySCF layout).

## 4. Commutators

For an even string `E` (one- or two-body) and any operator string,
`[E, o_1 ... o_m] = sum_i o_1 ... [E, o_i] ... o_m`, with

    [p^+ q, t^+]          =  delta_qt p^+
    [p^+ q, t]            = -delta_pt q
    [p^+ r^+ s q, t^+]    =  delta_qt p^+ r^+ s - delta_st p^+ r^+ q
    [p^+ r^+ s q, t]      = -delta_pt r^+ s q  + delta_rt p^+ s q

so commutators are generated directly, without products whose highest-rank
parts would have to cancel.  This keeps `<O_X^+ [H, O_Y]>` at RDM rank
`(|O_X| + |O_Y| + 2) / 2`.

## 5. Simplification

* Deltas involving a dummy are removed by substituting the dummy; deltas
  between free indices are kept; deltas between different spaces vanish.
* RDM antisymmetry: creators and annihilators are sorted separately with the
  sign of each permutation; repeated indices give zero.
* `(pq|rs) = (rs|pq)` is used to order integral indices.  No relation
  involving complex conjugation is used (conventions.md section 2).
* Dummies are renamed to the lexicographically smallest assignment over all
  permutations, after which terms with the same structure are summed.

## 6. Code generation

`python -m socutils.zmr.wick.pc_nevpt2` writes, from the same term objects,

* `zmr/pc/generated_terms.py`: for every class and pair of operator blocks a
  list of `(coefficient, einsum subscripts, operand names)`; operands are
  `dm<k>`, `eye`, `heff`, `eri`.  Evaluated by `zmr/pc/rdm_terms.py`.
* `docs/derivations/generated/pc_nevpt2_terms.md`: the same terms in readable
  form.

`test_generated_files_up_to_date` fails if either file differs from a fresh
generation.

## 7. Validation

* Random operator strings of up to eight core/active/virtual operators:
  symbolic evaluation vs matrix elements between explicit determinant
  expansions in the full orbital space, < 1e-12 (`test_wick_algebra`).
* Term collection does not change values; the commutator identity
  `<X [E, O]> = <X E O> - <X O E>` holds for one- and two-body `E` with summed
  indices, and the commutator form needs a lower RDM rank.
* Generated PC-NEVPT2 `S` and `K` equal the spectral `T^+ T` and
  `T^+ lambda T` for all classes, complex model Hamiltonians, transition
  elements between three states (`test_pc_rdm`).
