# X2CAMF with increasingly diffuse basis sets — summary

Branch `claude/x2camf-diffuse-stability`.  Scripts and outputs in `examples/qdhf/`:

| script | what |
|---|---|
| `36-x2camf_diffuse_stability.py` | k scan: X2C-1e, x2camf (block / projected 1e-6, 1e-5 / projector form), x2cmp, amfX2C (block / projector form); energies, lowest h_core eigenvalue, orbital energies, max\|eig\| of the AMF correction in the canonically orthonormalized basis |
| `37-amf_long_range_coulomb.py` | amfX2C with the Coulomb part as a true molecular operator (exact J of the atomic densities, or SAP); first-order breakdown of the x2camf AMF into DC-SO / Gaunt / gauge pieces |
| `38-onecenter_model_potential.py` | x2cmp rebuilt from one-center quantities: J from the spherical atomic density (Coulomb fit), one-center K (block or projector form) |
| `40-analytic_onecenter_j.py`, `ocj/` | exact analytic J of the atomic densities (existing libcint integrals), screened neutral-atom C kernel; cost against one scalar NR JK build |

## Setup

* Basis: fully uncontracted `unc-cc-pVDZ` (HCl, R = 1.275 Å) and `unc-cc-pVDZ-DK` (HBr, R = 1.414 Å),
  plus k even-tempered s and p functions (ratio 2.5, below the smallest exponent of each l) on
  **every** atom, k = 0 … 10.  Min eigenvalue of S (2c, decontracted) goes from 6.8e-3 (HCl) /
  5.0e-3 (HBr) at k = 0 to 3.2e-7 / 2.2e-7 at k = 10; the near-linear dependence is cross-atom
  (H and Cl/Br diffuse functions become the same function).
* Point nucleus, no pcc.  Atomic AMF either Dirac-Coulomb (DC) or Dirac-Coulomb-Breit (Gaunt +
  gauge, the `.x2camf()` default).
* **No DHF reference.**  pyscf 4c DHF does not converge for HCl in this setup (stalls at
  -357 Eh instead of -461 Eh, with every initial guess tried), so all energies are relative to
  X2C-1e (36, 37) or to x2cmp (38).  The stability question does not need E(DHF): a quantity
  that must not depend on k (e.g. E(x2camf) − E(X2C-1e)) is followed instead.
* **HI not run** (time spent on the Gaunt finding and the long-range / one-center schemes below).
* Step 0: the x2camf C-backend stack overflow (Br, I) was fixed by **porting the heap-allocation
  patch** from `ccr-17f4d4cb-lw45cf` (`lib/x2camf/csrc/x2c_int_sph.c`, `x2c_int_sph_gauge.c`),
  not by `ulimit -s`.
* pyscf 2.14 + numpy 2.5.  pyscf's DIIS catches `numpy.linalg.linalg.LinAlgError`, which numpy 2
  no longer has; with near-singular bases DIIS hits it, so the scripts alias
  `numpy.linalg.linalg = numpy.linalg`.  At k = 9–10 some runs still stop on a genuinely singular
  matrix (noted in the outputs).
* A consistency check runs at every k: the hand-built block x2camf h_core equals
  `x2cmp(flavor='x2camf')` exactly (0.0).

## 1. x2camf, Dirac-Coulomb AMF: stable

E(x2camf) − E(X2C-1e) / Eh, and max\|eig\| of the AMF correction in the orthonormal basis
(`36-…hcl_unc.out`, `36-…hbr_unc.out`):

| k | Smin HCl | HCl block | HCl projector | \|dF\|orth | Smin HBr | HBr block | HBr projector | \|dF\|orth |
|---|---|---|---|---|---|---|---|---|
| 0 | 6.8e-3 | 2.1013e-4 | 2.0988e-4 | 0.1477 | 5.0e-3 | 1.0567e-2 | 1.0565e-2 | 1.2275 |
| 4 | 6.2e-5 | 2.1081e-4 | 2.1060e-4 | 0.1477 | 7.0e-5 | 1.0573e-2 | 1.0571e-2 | 1.2275 |
| 7 | 1.3e-6 | 2.1080e-4 | 2.1060e-4 | 0.1477 | 1.1e-6 | 1.0573e-2 | 1.0571e-2 | 1.2275 |
| 10 | 3.2e-7 | 2.1079e-4 | 2.1060e-4 | 0.1477 | 2.2e-7 | 1.0573e-2 | 1.0571e-2 | 1.2275 |

Lowest h_core eigenvalue and lowest orbital energy equal the X2C-1e ones up to the SO shift at
every k; no intruder.  Projection (1e-6, 1e-5) changes nothing (≤ 2e-7 Eh).  The pure spin-orbit
AMF acts only on l ≥ 1 and is localized at the nucleus, so diffuse functions carry ~0 of it.

## 2. x2camf, Gaunt + Breit AMF (the default): block form drifts, projector form does not

Same quantity (`36-…hcl_unc_gb.out`, `36-…hbr_unc_gb.out`):

| k | Smin | HCl block | HCl proj 1e-5 | HCl projector | HCl \|dF\|orth block / projf | HBr block | HBr proj 1e-5 | HBr projector |
|---|---|---|---|---|---|---|---|---|
| 0 | 6.8e-3 | 0.21673 | 0.21673 | 0.21676 | 0.377 / 0.377 | 2.596537 | 2.596537 | 2.596562 |
| 4 | 6.2e-5 | 0.21672 | 0.21672 | 0.21675 | 0.377 / 0.377 | 2.596500 | 2.596500 | 2.596554 |
| 5 | 1.2e-5 | 0.21669 | 0.21669 | 0.21675 | 0.377 / 0.377 | 2.596446 | 2.596446 | 2.596554 |
| 7 | 1.3e-6 | 0.21664 | 0.21672 | 0.21675 | 0.510 / 0.377 | 2.596272 | 2.596493 | 2.596554 |
| 10 | 3.2e-7 | **0.21660** | 0.21671 | 0.21675 | **0.756** / 0.377 | **2.596130** | 2.596500 | 2.596554 |

* Block x2camf: the AMF contribution drifts monotonically once Smin < ~1e-5:
  −1.3e-4 Eh (HCl) and −4.1e-4 Eh (HBr) by k = 10, still going.  Lowest h_core eigenvalue
  follows (HCl −145.392580 → −145.392620).  Not a collapse at Smin = 2e-7, but a systematic
  error amplified by the near-linear dependence, larger for the heavier atom.
* Projector form A_K^mol = S_.K S_K⁻¹ A_K S_K⁻¹ S_K. (script 23 construction): flat to 1e-6 at
  every k, and \|dF\|orth stays at its k = 0 value.
* Canonical projection at thr = 1e-5 removes most but not all of it (residual ~1e-5 / 5e-5 Eh)
  and is threshold-dependent; 1e-6 removes about half.
* Orbital energies (HCl, `36-…hcl_unc_gb_orb.k0-6.out`, `…k7-8.out`): the drift sits in the core
  (Cl 1s −105.161583 at k = 4 → −105.161628 at k = 10 for block; projector form −105.161569
  constant); valence / SO splittings move by ≤ 3e-6.
* Which piece (`37-…hcl_unc.out`, first order Tr(D_X2C-1e dF), HCl): DC-SO 2.2e-4 Eh with
  \|dF\|orth 0.148 at every k (stable); **Gaunt** 0.234 Eh with \|dF\|orth 0.339 → 1.05 and
  **gauge** −0.018 Eh with 0.083 → 0.34 from k = 5 on.  The Gaunt/gauge AMF contains large
  scalar parts that are not confined to l ≥ 1 near the nucleus; in block form they are what the
  near-linearly-dependent cross-atom combinations amplify.

## 3. amfX2C: the collapse is the long-range Coulomb; fixing it helps, the block 2c correction remains

E − E(X2C-1e) / Eh, HCl, DC (`37-…hcl_unc.out`):

| k | Smin | block | projector | LR (exact J_mol) | LR + projector dF | SAP + K | SAP-X2C |
|---|---|---|---|---|---|---|---|
| 0 | 6.8e-3 | −0.082960 | −0.082953 | −0.082959 | −0.082953 | −0.082958 | −0.050167 |
| 4 | 6.2e-5 | −0.082978 | −0.082958 | −0.082967 | −0.082959 | −0.082966 | −0.050167 |
| 5 | 1.2e-5 | **−7.6e4** * | −0.082958 | −0.082968 | −0.082959 | −0.082967 | −0.050167 |
| 8 | 6.8e-7 | 1.2e6 * | −0.082958 | −0.083013 | −0.082969 | −0.083012 | −0.050167 |
| 9 | 4.4e-7 | −2.5e5 * | −0.082958 | **−0.298** * | −0.082981 | **−0.294** * | −0.050167 |

(* = SCF not converged; HBr behaves the same, `36-…hbr_unc.out`.)

* Block amfX2C collapses at k = 5 (lowest h_core eigenvalue −3.8e4, −2e9 later), as in script 21.
  Mechanism: −Z_K/r is a true molecular operator but the atomic Coulomb J[ρ_K] lives only in the
  K–K block, so diffuse functions of other atoms see an unscreened −Z_K/r tail, amplified by
  1/w_min.
* LR: G4c = J_mol[Σ_K D_K] (true molecular operator) + blockdiag(G4c_K − J[D_K]_KK).  Removes
  the k = 5 collapse; h_core stays at −145.508 to k = 9.  SAP (Lehtola Gaussian SAP for the
  Coulomb part, cf. Surjuse–Valeev SAP-X2C, `qdhf/x2c.py` sap1e) agrees with exact J to 1e-6 Eh
  and behaves identically.
* What remains: at k = 9 the converged Fock has a spurious state (lowest orbital −105.66 vs
  −105.26) with LR and SAP+K, from the **block-diagonal 2c correction** dF_K = to_2c(G4c_K) −
  G2c_K; putting dF in projector form (LR + projector dF) is stable to k = 9 (drift 2.8e-5).
* Pure SAP-X2C (no atomic 2e correction) is perfectly flat but recovers only ~60 % of the 2e
  picture change.

## 4. One-center model potential reproducing x2cmp (script 38)

x2cmp decouples with F4 = h4 + (J − K)[Σ_K D4_K] from molecular 2e integrals and subtracts the 2c
counterpart; this is why it was stable in every scan.  Rebuilt from one-center quantities:

* J: the atomic densities are spherical.  Coulomb fit (charge-conserving, s Gaussians,
  ratio 1.2) → Σ q erf(√ζ r)/r, SAP form from our own density, 1e integrals only.  (The
  exact analytic J, no fit, is section 5.)
* K: one-center (atom-only 2e integrals), block or projector form.

E − E(x2cmp) / Eh, HCl, DC (`38-…hcl_unc.out`):

| k | Smin | J fit + K 1c | J exact + K 1c | J fit + K 1c projector | J fit + K exact |
|---|---|---|---|---|---|
| 0 | 6.8e-3 | −1.09e-5 | −1.09e-5 | −2.74e-6 | 3e-8 |
| 5 | 1.2e-5 | −2.41e-5 | −2.41e-5 | −2.68e-6 | 3e-10 |
| 7 | 1.3e-6 | −8.40e-5 | −8.40e-5 | −2.68e-6 | 8e-10 |
| 10 | 3.2e-7 | **−1.93e-4** | −1.92e-4 | −2.68e-6 | −4e-10 |

Max \|Δε_occ\| vs x2cmp: block K 1.6e-6 → 4.3e-5; projector K ≤ 2.6e-7 at every k (1.3e-7 from k = 2 on).  The fitted
J is exact for this purpose at every k; the residual drift is entirely the
block-diagonal one-center K.  **J from the spherical atomic density + one-center K in projector
form reproduces x2cmp to 3e-6 Eh / 1e-7 Eh in orbital energies and is stable to Smin = 3e-7**,
with no multi-center 2e integrals.

HBr (`38-…hbr_unc.out`), same columns: k = 0 −1.20e-5 / −4.30e-6; k = 5 −5.11e-5 / −4.09e-6;
k = 7 −1.20e-4 / −4.08e-6; k = 10 **−2.47e-4** / −4.09e-6 (block K / projector K); max \|Δε_occ\|
block 2.6e-6 → 5.9e-5, projector ≤ 4e-7.  The block one-center K drifts faster for the heavier
atom (its exchange reaches a larger valence region); projector form stays flat.

Native x2camf against the same reference, E(x2camf) − E(x2cmp) / Eh (from the 36 outputs):

| k | Smin (HCl) | HCl, DC AMF | HCl, Gaunt+Breit AMF | HBr, DC AMF | HBr, Gaunt+Breit AMF |
|---|---|---|---|---|---|
| 0 | 6.8e-3 | 0.0831791 | 0.08312 | 0.885327 | 0.8848 |
| 4 | 6.2e-5 | 0.0831798 | 0.08311 | 0.885333 | 0.8848 |
| 7 | 1.3e-6 | 0.0831798 | 0.08303 | 0.885333 | 0.8846 |
| 10 | 3.2e-7 | 0.0831798 | 0.08299 | 0.885333 | 0.8844 |

The large constant is the scalar 2e picture change that x2camf (X2C-1e decoupling) does not
have and x2cmp does; it is k-independent, so not a diffuse-function effect (in the orbital
energies it sits in the core: Cl 1s 0.031 Eh, valence ~1e-6).  With the DC AMF native x2camf
is as stable as the projector-form one-center MP (constant to 1e-8 for k ≥ 1); with the
default Gaunt + Breit AMF it drifts like the block one-center K.

## 5. Exact analytic J of the atomic densities, and a C kernel (script 40, `ocj/`)

The J of section 4 without fit and without grid:

* the spherically averaged atomic density is exactly Σ c t_{n,ζ}, t = r^2n e^{-ζ r²}, ζ the
  sums of primitive exponents (angular averages of cartesian monomials; SS through the exact
  l±1 expansion of ∂φ); 1e-14 against the radial density;
* V[t_n] by the Laplacian recursion V[t_{n+1}] = ((8n+6)ζ V[t_n] − 2n(2n+1) V[t_{n−1}] −
  4π t_n)/(4ζ²): erf(√ζ r)/r plus r^2m e^{-ζ r²};
* integrals: int3c2e (erf part, one contracted s shell per atom), int3c1e and
  int3c1e_r{2,4,6}_origk (local part); SS from the gradient integrals.  1e-12 against the exact
  molecular J.
* with the nuclear attraction folded in, the neutral-atom potential −Σ E erfc(√ζ r)/r + local is
  short-ranged and screenable; `ocj/onecenter_j.c` contracts the k index on the fly, OpenMP
  over shell pairs, exponents grouped per decade, screening by an amplitude-aware
  Gaussian-overlap estimate (eps absolute).  SS gradients from int3c1e_ipvip1_r{2,4,6}_origk
  (libcint patch in `ocj/libcint-patch/`) or, with pyscf's own libcint, assembled in the
  kernel from cartesian l±1 shells (std mode, same result).

Cost against one scalar direct NR JK build in the same uncontracted basis (4 cores):

| system | n2c | NR JK | X2C decoupling | J, Python (40) | J, kernel | max err |
|---|---|---|---|---|---|---|
| PhCl / C10H7Br | 464 / 772 | 1.3 / 6.5 s | 2.0 / 5.0 s | 0.89x / 0.60x (integrals 0.62x / 0.55x) | | 4e-12 |
| C60, unc-cc-pVDZ | 3120 | 575 s | 238 s | 164 s (0.28x) | 49 s (0.09x), eps 1e-12 | 5e-10 |
| Au8 (cube), unc-dyall-v2z | 3264 | 293 s | 292 s | | 40 s (0.14x), decay screening | 1.8e-8 |

## Conclusion

* X2CAMF with a **Dirac-Coulomb** atomic AMF is stable with diffuse functions down to
  Smin ≈ 2e-7 (HCl, HBr): no drift, no intruder, projection irrelevant.
* With the default **Gaunt + Breit** AMF, the atom-block-diagonal form is **not** stable: the
  AMF energy contribution drifts by 1e-4 (HCl) to 4e-4 Eh (HBr) once Smin < 1e-5, in the core
  levels, from the Gaunt and gauge pieces.  No variational collapse down to Smin = 2e-7.
* The **projector form** fixes it completely (flat to 1e-6 Eh); canonical projection of the AMF
  matrix only partly, and threshold-dependently.
* Common rule from all variants (x2camf-Gaunt, amfX2C, one-center K): an atomic operator that is
  not confined to the nucleus breaks when placed atom-block-diagonally in a near-linearly
  dependent basis.  The long-range Coulomb must be a true molecular operator (exact J, SAP, or J
  from the spherical atomic density); the short-range rest (exchange, 2e picture-change
  correction, Gaunt/gauge AMF) needs projector form.
* Not done: HI (dyall-v2z), a DHF reference, Gaunt+Breit in the amfX2C / one-center variants.
