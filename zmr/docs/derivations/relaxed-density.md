# Relaxed one-particle densities of PC-NEVPT2

Implementation: the derivatives of E2 in `zmr/uc/density.py` (unrelaxed density),
`zmr/uc/gradient.py` (orbital and CI gradients in one pass, `derivatives`), the Python
references in `zmr/reference/` (`orbital_gradient.gradient_general`, `ci_gradient`); the
CASSCF response in `zmr/reference/relaxed_fd.py` + `zmr/nr/relaxed.py` (scalar, PySCF
CASSCF) and `socutils/mcscf/zmc_response.py` + `zmr/spinor_relaxed.py` (spinors, socutils
CASSCF).  Tests: `tests/test_density.py`, `test_orbital_gradient.py`,
`test_relaxed_nevpt2.py`, `test_spinor_relaxed.py`, `mcscf/test/test_response_*.py`.

## 1. Lagrangian

A target `T` (the energy of a state, a level of the multi-state NEVPT2, or a matrix
element of `heff` within a degenerate level) depends on the orbitals and CI vectors of a
state-averaged CASSCF, which are fixed by its stationarity `G_SA(x) = 0`.  With
`L = T + Re<z, G_SA>` stationary in the CASSCF parameters `x`,

    J z = -dT/dx,        dT/dlambda = tr(O D_T) + tr(O D_z),

`J` the exact Hessian of the averaged energy, `D_T` the density at fixed `x` (CASSCF part
plus the unrelaxed E2 density), `D_z` the derivative of the averaged density along `z`.
One solution `z` per target serves every one-electron operator.

Parameters: orbitals `C exp(K)`, `K` anti-Hermitian (PySCF `pack_uniq_var` for real
orbitals, `zmcscf.CASSCF.pack_uniq_var` for spinors), CI vectors `(c + x)/|c + x|`.  For
spinors the complex vector `z` is treated through its real and imaginary parts:
`dE = Re<G, dz>`, `d^2E = Re<dz, J dz>` (`zmc_response`).

## 2. Derivatives of E2

Each class contributes, for amplitudes `c^J = u^J / D^J` of the model states and the
weights `rho_IJ` of the target `T = Re sum rho_IJ a_IJ` (`density.pair_amplitudes`):

* the unrelaxed density: denominators (core-core and virtual-virtual blocks of the
  generalized Fock matrix), `H_act` in the first-order space and the zeroth-order
  energies (active block), the Fock elements in `v` (`cava`, `caaa`, `aaav`);
* the derivatives with respect to the three-index integrals, kept as holomorphic blocks
  `G_h[P, x, y]` (E real, written with `v` and `conj(v)`; half of the real-orbital
  derivative); their conjugate blocks are implied;
* the CI gradient from the metric and `H_act` matrices of the functions `O_X |Psi_J>`
  (`ci_gradient.QAccumulator`, Hermitian parts `(Q + Q^+)/2`) and the Fock term through
  the averaged active density.

The orbital gradient is `dE2 = sum M K` with

    M = X - X^+ + F^T D - D F^T + (G Y - Y G)^T   (generalized and core Fock chains),
    X = sum_P conj(B_P) G_P,  Y = W[D^T],  G = gamma^T,

valid for real and complex orbitals (`pack(M - M^T)` is the PySCF gradient).

A hole index transforms with the transpose of the Fock matrix (`[F, i] = -sum_j F_ij j`):
`dE2/dF_ii' = -sum c_i conj(l_i')`.  For real orbitals this is invisible; for spinors the
transpose matters for complex operators.

## 3. Multi-state targets

A level of `heff`: `rho = conj(c) c^T` of its eigenvector, averaged over the multiplet
when it is degenerate (a Kramers pair).  ccvv then enters through `H_act` and the CI
vectors as well (its first-order space is the model space), with its energy times
`tr(rho)`.  All levels in one pass over the classes (`derivatives(rho=...)` with
`rho (L, J, J)`).

The zero modes of `J` are the norm and phase of every CI vector and, for degenerate
averaged states of equal weight, their rotations and symmetric mixings.  The rotations
among non-degenerate states are not zero modes: the CI gradients `w_I (H - E_I) c_I` couple
them with the symmetric mixings, and together they fix the model states as eigenvectors.
Projecting them out (as an earlier version did) makes `z` diverge whenever the target
couples states of one symmetry.

Time-odd properties split a Kramers pair; the observable is the matrix
`W_mn = d<c_m|heff|c_n>/dlambda` within the level, whose eigenvalues are the first-order
splittings (`relaxed_level_matrix`; real and imaginary parts of `<c_m|heff|c_n>` as two
targets with Hermitian `rho`, trace 0 off the diagonal).  With `e0='state'` the states of a
multiplet share their E0 (`NEVPT2Base.degeneracy_tol`); otherwise `heff` is not covariant
within a pair once a field splits it, and the finite-field splitting is not linear in the
operator.  The derivatives of a shared E0 are spread over its group
(`density.share_traces`).

## 4. Frozen core

The frozen orbitals are the lowest eigenvectors of the core block of the averaged
generalized Fock matrix (`spinor_relaxed.frozen_partition`, Kramers pairs kept).  The
orbital gradient of E2 includes the rotations of the frozen orbitals
(`derivatives(frozen_coeff=...)`: three-index rows `B[P, f, x]`, Fock chains over all
orbitals); the frozen-correlated core rotations are not CASSCF parameters, and the
condition `F_fc = 0` with multipliers `zeta_fc = -G_fc / (e_f - e_c)` adds its orbital, CI
(through `gamma_SA`) and density terms (`spinor_relaxed._e2_terms`).

## 5. Validation (finite fields with the CASSCF and the NEVPT2 redone)

| system | target | relaxed | finite field |
|---|---|---|---|
| water 6-31G, CAS(4,4) | state-specific, SA states, QD levels (dipole) | | to 1e-8 |
| water Cs, SA-3 QD levels | two states of one irrep | -0.07783520 | -0.07783521 |
| CH+ SA-3 | levels with a degenerate Pi pair | | to 1e-6 |
| water spinor 6-31G, SA-4 | ground state, complex dipole | 0.38371044 | 0.38371038 |
| OH spinor X2CAMF, X 2Pi | levels (Kramers pairs), dipole | 1.02509186 | 1.02509186 |
| OH, spin operator | eig W of 2Pi_3/2 | +-1.01847212 | 1.01847237 |
| OH, Zeeman (X2C) | g_parallel of 2Pi_3/2 | 4.0758 | to 1e-5 |
| OH, proton hyperfine | | | to 1e-5 relative |
| OH, contact density at O | | 284.23294009 | 284.23293998 |
| water spinor, O 1s frozen | ground state | | to 1e-6 |

E2 can be strongly nonlinear in the CI vectors when a first-order space is nearly
linearly dependent (metric eigenvalues 1e-9 .. 1e-7, just above `lindep`): finite fields
then need steps of 1e-5 or less and a CASSCF converged to |G| ~ 1e-12
(`relaxed_fd.polish`, `zmc_response.polish`); the analytic derivatives are not affected.

## 6. Cost

The E2 derivatives are one pass over the classes (cavv and ccav in segments, ccvv by the
native kernel with the Kramers shortcut); the Z-vector is MINRES with a diagonal
preconditioner (water cc-pVDZ SA-4: 276 -> 42 iterations).  Pyridine cc-pVDZ SA-3: three
levels in 90 s (the QD energy 0.3 s).  ccvv dominates large systems as in the energy.
