# Multi-state NEVPT4(SD)

Code: `zmr/pt4/nevpt4.py`; reference `zmr/reference/nevpt4_bruteforce.py`;
tests `zmr/tests/test_nevpt4.py`.

## 1. Partitioning

`H = H0 + V` with the Dyall Hamiltonian outside the model space and one
zeroth-order energy per model state inside it:

    H0 = sum_J E0_J |J><J| + Q H_D Q,      Q = 1 - sum_J |J><J|

    e0='state'    E0_J = <J|H_D|J>
    e0='average'  E0_J = sum_K w_K <K|H_D|K>   (the same for all J)

`H_D |J> = E0_J |J>` for CAS eigenstates and the Dyall `H_D` (conventions.md
section 8), so with `e0='state'` the first-order matrix
`H1_KJ = <K|V|J> = <K|H|J> - E0_J delta_KJ` vanishes; with `e0='average'` it
is the splitting of the model states, `E_J - E0`.

## 2. Bloch expansion with one resolvent per column

Multi-partitioning perturbation theory (Zaitsevskii and Malrieu): column J
of the wave operator uses `R_J = Q (E0_J - H_D)^-1 Q`,

    Omega1 |J> = x_J = R_J V |J>
    Omega2 |J> = y_J = R_J [V x_J - sum_K x_K H1_KJ]
    Omega3 |J> = z_J = R_J [V y_J - sum_K y_K H1_KJ - sum_K x_K H2_KJ]
    H(n)_IJ   = <I| V Omega(n-1) |J>

and the effective Hamiltonian `H_PP + H2 + H3 + H4` is Hermitized,
`1/2 (H_eff + H_eff^+)`; at second order this is QD-NEVPT2.

For one state: `E2 = <V x>`, `E3 = <x|V - E1|x>`,
`E4 = <w|R|w> - E2 <x|x>` with `w = Q (V - E1) x`: Rayleigh-Schroedinger
perturbation theory.

## 3. The SD truncation

All functions are kept in the singles and doubles of the CAS (the first-order
interacting space), uncontracted or internally contracted.  The triples and
quadruples that `V x` reaches are left out of the second-order wave
function, as in NEVPT4(SD) (Kempfer, Sivalingam, Neese, J. Chem. Theory
Comput. 2025).  With them left out the renormalization term
`- sum_K x_K H2_KJ` (for one state `-E2 <x|x>`) must be left out as well for
size consistency (Grimme 2001); it is omitted by default and available as
`renormalization=True` for the contractions with one space for all states.

## 4. Spaces

The product basis of the MRCI (`zmr.mrci.basis`) in quasi-canonical core and
virtual orbitals of the averaged density.  Its block vectors diagonalize
`H_act` within their span, so `H_D` (projected on the span for the
contracted blocks) is diagonal there: `MRCI.hdiag`, and `R_J` is a division.

    contraction='none'   all determinants of every sector (uncontracted)
    contraction='union'  PC functions of all model states (rotation invariant
                         with e0='average' and equal weights)
    contraction='state'  column J in the PC functions of |J> and of the
                         reference functions degenerate with it (QD-NEVPT2);
                         needs CAS eigenstates as reference functions

With 'state', a degenerate set of reference functions (a Kramers pair, the
M_J components of a level) is arbitrary up to a unitary rotation; built from
one function of the set alone, the first-order spaces of its members are not
related by the symmetry and the degeneracy is lost at fourth order (O2 X,
PVDZ-X2C-TP: D = 22 cm-1 instead of 3.7).  The functions of the whole set are
therefore used for each of its members.

Even so 'state' is unbalanced at fourth order: the second-order wave function
of each column is truncated to its own first-order space, and the truncation
differs between reference functions of different structure.  O2 X,
PVDZ-X2C-TP (Eh, fourth-order terms H4_JJ): 0+ alone -0.019264, the +-1 pair
-0.019920; with 'union' -0.020576 for all three.  The difference, 6.6e-4 Eh
= 140 cm-1, swamps the zero-field splitting (3.7 cm-1).  At second order the
two contractions agree here (PC = UC for two electrons in four spinors).
States whose energy differences are wanted must share one space: 'union'
(the default) or 'none'.

`H x` is the MRCI sigma; a state costs four sigma evaluations (`H|J>`,
`H x`, `H y`, `H z`).

## 5. Validation

`zmr/tests/test_nevpt4.py`, random complex model Hamiltonians:

* second, third and fourth order against the determinant-basis reference
  for every combination of e0, contraction and the renormalization option
  (1e-9);
* second order equals PC-NEVPT2 (union, state) and UC-NEVPT2 (none);
* one state: 'state' and 'union' coincide;
* 'average' with 'union' is invariant under unitary rotations of the model
  states.
