# Internally contracted MRCC (ic-MRCC(SD)): design

Status: design; the reference code of section 5 (`zmr/reference/icmrcc_bruteforce.py`,
`tests/test_icmrcc_bruteforce.py`) answers section 3 for one state.  Conventions of `conventions.md`; the
space and the product basis of `mrci.md`; integrals through the block
interface of `zmr/integrals.py`.

## 0. Why

The ensemble-GNO MRCC of socutils `mrcc/ensgno` (one T for the three
components of O2 X, residuals = components of H_bar on the GNO excitation
operators) is not exact at second order.  With `H(l) = H_D + l (H - H_D)`
(O2, cc-pVDZ-X2C-TP; `o2zfs_coord/ensgno/lam/`), the coefficients of `l^n`
in E(0+) are, in mEh:

| order | exact (SDTQ) | ensemble GNO (2,2) | GNO (2,2) of 0+ alone |
|-------|--------------|--------------------|-----------------------|
| 2 | -379.16 | -412.41 | -412.40 |
| 3 | +22.34 | +77.7 | +77.5 |
| 4 | -28.12 | -50.1 | -50.7 |

and the second-order part of D is five times the exact one, cancelled by the
third order.  The state-specific run has the same second order, so the
cause is the operator-space residual, not the shared T.  Projecting the
residual on the states E_mu |Psi_0> instead makes the linearized equations
those of PC-NEVPT2 (section 3), which is the point of this method.

## 1. Ansatz (one state first)

    |Psi> = e^T |Psi_0>,   T = sum_B sum_{e, X} t^B[e, X] E_e O_X

`|Psi_0>`: a CASCI state; `B`: the eight external classes of `mrci.md`
section 1 (caaa ... ccvv); `E_e`: external string of the class (core
annihilators, virtual creators); `O_X`: the active string of the class, as in
`zmr/uc/classes.py` (non-redundant strings).  Every `E_e O_X` contains at
least one core annihilator or virtual creator, so `<Psi_0| E_e O_X = 0`:
`T` raises the external excitation level by `|e| >= 1`.  Purely internal
excitations are left out (the reference is a CASCI eigenstate; relaxation of
its coefficients comes back with the model space of section 6).

The operators of `T` do not commute (their active parts), so
`e^{-T} H e^{T}` does not terminate; it is truncated after the commutator
of order `m`, `H_bar_m = sum_{k <= m} [..[H, T], .. T] / k!` (Hanauer, Koehn,
J. Chem. Phys. 131, 124118 (2009), and later papers, use `m = 2` with
selected higher terms; here `m` is a parameter and its convergence is
checked against the untruncated exponential in the reference code).

## 2. Equations

    R_mu = <Phi_mu| H_bar_m |Psi_0> = 0,     E = <Psi_0| H_bar_m |Psi_0>

for `Phi_mu` the orthonormal internally contracted functions
`|B; e, k> = E_e |Phi_c b_k>`, `b_k` an orthonormal basis of the span of
`O_X |Psi_0>` (`zmr/pc/spectral.py`): the FIC space of `mrci.md`.  The same
vectors define the amplitudes: with `b_k = sum_X O_X |Psi_0> Y[X, k]`,

    T |Psi_0> = sum_{e, k} C[e, k] |B; e, k>   <->   t[e, X] = sum_k Y[X, k] C[e, k]

so the unknowns are the coefficients `C` of the MRCI space and the redundant
directions of the metric never enter.  The energy and residual are invariant
under rotations within the core, active and virtual spaces.

## 3. Perturbation structure (what has to be checked)

With `H = H_D + V`, `H_D |Psi_0> = E_0 |Psi_0>` and `<Psi_0| T = 0`:

* first order: `<Phi_mu| [H_D, T1] |Psi_0> = <Phi_mu| (H_D - E_0) T1 |Psi_0>`, so
  `<Phi_mu| (H_D - E_0) T1 + V |Psi_0> = 0` with `T1 |Psi_0>` in the FIC space:
  the first-order equation of PC-NEVPT2, and `E2 = <Psi_0| V T1 |Psi_0>` is
  the PC-NEVPT2 energy.  This holds for any `m >= 1`.
* `<Psi_0| [[H_D, T1], T1] |Psi_0> = 0` (because `<Psi_0| T = 0`).
* third order: `E3 = <Psi_0| V T2 |Psi_0> + 1/2 <Psi_0| V T1 T1 |Psi_0>`.  In
  single-reference CC this is MP3; here the non-commuting active parts leave
  terms such as `<psi1| T1 Q V |Psi_0>` and `<Psi_0| V T1 T1 |Psi_0>` without
  their single-reference cancellation.  Whether `E3` equals PC-NEVPT3
  (`<psi1| V - E_1 |psi1>` in the FIC space; `zmr.pt4.NEVPT4SD` with one state,
  `e3`) is the first open question; the answer decides how close the method
  stays to the exact series at third order (exact: `l^3` of SDTQ).

Answer (reference code, model Hamiltonians `model_eris`, one state): `E2` is
PC-NEVPT2 to 1e-15 in every case.  `E3` equals PC-NEVPT3 to 1e-17 when the FIC
space is the whole singles-and-doubles space ((1, 4, 3) with 1 and 2 active
electrons) and, for a reason not yet understood, also for (2, 4, 3) with 2
electrons, where the FIC space is smaller (99 of 114 functions); otherwise the
two differ by 1e-7 to 5e-5 Eh, of either sign (E3 itself about 1e-2 Eh), for
1 to 4 active electrons.  The difference is a third-order effect of the
contraction.  Commutator truncation: `m = 4` reproduces the untruncated
series to 2e-8 Eh, `m = 2` differs by 2e-5 to 1.4e-4 Eh.

## 4. Evaluation

The linear terms are what zmr has:

* `<Phi_mu| H T |Psi_0>`: the MRCI sigma (`zmr/mrci/sigma.py`) on `C`, with
  the (vv|vv) term from the integral source (Cholesky vectors or rows of a
  file).
* `<Phi_mu| T H |Psi_0> = E_CAS C + <Phi_mu| T Q H |Psi_0>`: `T` applied to the
  first-order-like vector `Q H |Psi_0>`.  Since `T` raises the external level,
  only the parts of `Q H |Psi_0>` with one external index survive the
  projection on the singles and doubles.

The nonlinear terms cannot be formed in the space: `T T |Psi_0>` has up to
four external indices on each side (ccccvvvv).  They are evaluated as
connected tensor contractions, as in CCSD: every term of the commutator
expansion is a product of integrals and amplitudes `t[e, X]` contracted over
external indices, times an active factor `<b'_k'| O_Z1 .. O_Zn |Psi_0>`.  The
terms are generated by extending `zmr/wick` (which now generates the MRCI
terms with active strings as RDM labels) to nested commutators with
amplitude tensors that carry active strings.  The active factors are not
formed as RDMs: the external contractions come first and leave a tensor over
active indices, which is applied to CI vectors (`O_X b_k`, `zmr_apply_strings`),
the CI-vector intermediates of PC-NEVPT2 (`performance.md` section 2) with
longer strings.  The order of contraction (externals first, actives first)
is chosen per term by cost, as in `sigma.py` now.

Cost: the doubly external terms scale as CCSD (`n_c^2 n_v^4`), the others
carry the active factors; for 30 correlated electrons, 300 spinors and an
active space of 10^4 determinants the CI-vector part is the one to measure.

## 5. Reference implementation and validation

`zmr/reference/icmrcc_bruteforce.py`: operators `E_e O_X` as matrices in the
determinant space of a small system (all determinants of the particle-number
sector), `T` as a matrix, `H_bar` by the truncated series and by `expm`, the
residual projected on the FIC basis, solved by Newton's method.  Checks:

1. no active orbitals: CCSD (determinant-space CCSD of `test_bruteforce`);
2. `m` convergence: `m = 2, 4, 6` against the untruncated exponential;
3. `l`-series on `H(l)`: `E2` equals PC-NEVPT2 (`zmr.pc.PCNEVPT2`); `E3`
   against PC-NEVPT3 (section 3); `E4` against the exact series;
4. size consistency: a CAS fragment plus a closed-shell fragment at large
   distance (product of the fragment energies);
5. O2 / STO-3G spinors: D against FCI.

The production code (section 4) is checked against this reference term by
term (residual and energy for random amplitudes), then on the same systems.

## 6. Several states (after the single-state method is validated)

The O2 problem needs 0+ and +-1 in one balanced treatment.  Options:

* state-specific: one `T` per component in its own FIC space; the +-1 pair
  shares it through time reversal.  `NEVPT4(SD)` with this contraction was
  unbalanced at fourth order, but that came from truncating the second-order
  wave function to the first-order space; whether the coupled-cluster
  equations, which use no such truncation, are balanced is to be measured
  (order by order on `H(l)`, against the exact series of each component).
* multi-state ic-MRCC with an effective Hamiltonian over the model space
  (Aoto, Koehn), one `T` per column.

The criterion for both is the one that exposed the GNO method: the `l^2` and
`l^3` coefficients of D against the exact ones (SDTQ on `H(l)`).

## 7. Stages

1. Reference code, checks 1-4 of section 5 on model Hamiltonians
   (`tests/_helpers.model_eris`); answer the question of section 3.
2. O2 / STO-3G and a small basis with the reference code: D, `l`-series.
3. Term generation (`zmr/wick`) and the production residual, checked term by
   term against stage 1.
4. Multi-state variant (section 6), O2 DZ / TZ against SDTQ.
5. Triples: perturbative, on the projected equations.
