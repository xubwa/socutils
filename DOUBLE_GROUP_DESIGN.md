# Double-group symmetry for spinor (2c) and Dirac (4c) SCF — design note

Status: draft for discussion (2026-10-16). Nothing implemented yet. Target: prototype in socutils,
written so it can move into PySCF mainline (`pyscf/symm`, `pyscf/scf`) with little change.

## 0. Decisions already made

- No quaternion algebra. Everything is complex, in PySCF's j-adapted spinor GTO basis.
- Symmetry lives in the AO layer, shared by 2c spinor SCF and 4c DHF.
- DHF and spinor SCF do not share a base class. They share one symmetry mixin, backed by module-level functions.
- New spinor-specific attributes/functions; the single-group `mol.symm_orb` / `mol.irrep_id` are untouched.
- Double groups get their own direct product (`prod`, `conj`); no XOR encoding.
- Phase 1 groups: Abelian double subgroups C1*, Ci*, C2*, Cs*, C2h*, plus linear C∞v*, D∞h* (m_j labels).
- Orientation (user's decision 2026-10-16, option B): the user puts the molecule in the symmetry frame explicitly
  (`mol = to_symm_frame(mol)`); everything is computed and reported in that frame, nothing is rotated back
  automatically. The mixin checks the frame and refuses otherwise. No D^j. This check is meant to be removed
  once lab-frame SALCs (D^j) are implemented. See §2a.
- Class names: Kramers-restricted `RHF`, unrestricted `UHF` (old `KRHF`, `SpinorSCF` stay as aliases).

## 1. Group layer: labels and products

Group chosen from PySCF's single-group result (`mol.groupname`, from `symm.as_subgroup`):

| `mol.groupname` | double group used | generators (symmetry frame) |
|---|---|---|
| C1 | C1* | — |
| Ci | Ci* | i |
| C2 | C2* | C2z |
| Cs | Cs* | σz (PySCF puts the Cs mirror ⊥ z) |
| C2h | C2h* | C2z, i |
| C2v, D2 | C2* | C2z |
| D2h | C2h* | C2z, i |
| Coov | C∞v* (Abelian part: rotations about z) | C(φ) about z |
| Dooh | C∞h*-type (rotations about z + i) | C(φ), i |
| SO3 (atom) | as Dooh, axis z (κ blocking: later) | C(φ), i |

Label of a symmetry-adapted function = exponents of its eigenvalues under the generators:

- C2z: eigenvalue e^{+iπ m_j}  →  n = 2m_j mod 4 ∈ {1, 3} (fermion), {0, 2} (boson)
- σz = i·C2z: eigenvalue (−1)^l e^{+iπ m_j}  →  n = 2(m_j − l) mod 4
- i: eigenvalue (−1)^l  →  parity bit p
- C(φ): e^{+iφ m_j}  →  n = 2m_j (unbounded integer)

Integer encoding of `orbsym`: `id = 2*n + p` (`p = 0` for groups without inversion); for linear groups
`n = 2m_j` may be negative. Following PySCF's `symm` API, `mol` only stores the group name
(`mol.doublegroupname`) and the group operations are module functions taking that name; tables are internal:

```python
irrep_id2name(doublegroupname, ir)        # cf. symm.irrep_id2name
irrep_name2id(doublegroupname, name)      # cf. symm.irrep_name2id
direct_prod(a, b, doublegroupname)        # cf. symm.direct_prod: n_a + n_b (mod N), p_a ^ p_b
conj(a, doublegroupname)                  # new: -n_a (mod N), p_a  (complex-conjugate = Kramers-partner irrep)
is_fermion(a)                             # n odd
# N = 2 (C1, Ci), 4 (C2, Cs, C2h), None (linear: n = 2m_j unbounded)
```

Selection rule for integrals uses `conj` explicitly, e.g. ⟨pq|rs⟩ ≠ 0 iff
`direct_prod` of `conj(p)`, `conj(q)`, `r`, `s` is totally symmetric. Export for zccsdt: `n mod 4` (+ parity).

## 2. AO layer (spinor SALCs)

```python
def symm_adapted_basis_spinor(mol, doublegroupname, orig, axes):
    '''-> so: list of (n2c, n_ir) complex arrays, irrep_id: list of int'''
```

Construction:

1. Atoms in the symmetry frame (`orig = mol._symm_orig`, `axes = mol._symm_axes`), equivalent atoms from
   `symm.geom.symm_identical_atoms` (reuse).
2. In the symmetry frame every generator acts on a spinor AO |atom A, κ, m_j⟩ as
   atom permutation × phase (phases above). Verified numerically 2026-10-16: the operator matrix preserves
   S, T+V and σ·p V σ·p to < 1e-12 (H2O, CH4, Br2, HF, C2H4).
   Linear groups: only C2z/i are needed to separate atoms; m_j itself is the AO label.
3. Projection: P_Γ = |G*|^{-1} Σ_g χ_Γ(g)* O(g) per orbit of equivalent atoms. All groups are Abelian, so each
   AO gives at most one nonzero SALC per irrep (as in PySCF `symm_adapted_basis`).
4. Requirement instead of a rotation: the symmetry z axis must be parallel to lab z (`mol._symm_axes[2] = ±e_z`).
   x/y and the origin do not matter (all phase-1 operations only involve z; atom permutations are computed in the
   symmetry frame). Otherwise the mixin refuses (§2a). z → −z only relabels C2z irreps
   (n → −n), which is harmless but should be normalised so labels are reproducible.
5. Kramers pairing inside `so`: columns of the partner irrep `conj(Γ)` are generated as T·so[Γ] using
   `mol.time_reversal_map()` (sign-aware), not projected independently. For self-conjugate irreps (C1*, Ci*)
   the columns are laid out as Kramers pairs so the block can go to zquatev.

## 2a. Symmetry frame (option B)

PySCF does not move the molecule. Re-detecting symmetry on an already reoriented molecule gives `_symm_axes`
equal to the identity only up to signs / x-y swaps (checked 2026-10-16 on H2O, CH4, C2H4, HBr, NH3, trans-N2H2,
starting from randomly rotated inputs; z always came back as ±e_z), hence the z-only requirement in step 4.

```python
mol = to_symm_frame(mol)                 # new Mole, r_s = R (r - orig), then symmetrized; R proper (det = +1)
                                         # gives exactly the coordinates printed by the error message below
                                         # mol.symm_frame = (orig, R) kept for reference
...                                      # SCF / CC / properties all on this mol, results in this frame
to_lab_vector(v, mol.symm_frame)         # optional helpers, never called automatically:
to_lab_tensor(T, mol.symm_frame)         #   R^T v, R^T T R
```

Check in `DoubleGroupSymm.build()`: symmetry z ∥ lab z. If it fails, raise with a message that

1. explains that double-group symmetry currently needs the molecule in its symmetry frame and why
   (m_j labels are defined along lab z; lab-frame SALCs not implemented yet);
2. gives the one-line fix `mol = to_symm_frame(mol)`;
3. prints the symmetrized atom coordinates in the symmetry frame, ready to paste into an input: "use these
   coordinates and the calculation runs". Symmetrized = each atom averaged over its symmetry images (changes are
   below `symm.geom.TOLERANCE`, 1e-5 Bohr; the maximum displacement is reported).

On success the SCF output states the frame: point group / double group used, origin and axes of the symmetry
frame relative to the input, and that vectors/tensors and m_j labels refer to this frame.

- R must be a proper rotation: with det R = −1 the calculation would be done on the mirror image and axial
  quantities (spin, B field) change sign. PySCF's axes can be improper (seen for HBr), so flip x if needed.
- Inputs given in another frame (external field, point charges, nuclear-spin directions) are the user's
  responsibility; `mol.symm_frame` gives the rotation.
- Energies are invariant. Orbitals / density matrices are only needed in the lab frame for interop (reading a
  lab-frame chkfile as initial guess, cube files); that needs D^j and is deferred.

Stored on `mol` lazily, without touching single-group attributes:

```python
mol.doublegroupname        # str, e.g. 'C2h' for a D2h molecule, 'Coov', 'Dooh' (cf. mol.groupname)
mol.symm_orb_spinor        # cf. mol.symm_orb  (prototype: spinor_basis.get_symm_orb(mol), weak cache, because
                           #  mol.dumps() only serializes JSON-like attributes; mainline: add to dumps exclude_keys)
mol.irrep_id_spinor        # cf. mol.irrep_id
mol.irrep_name_spinor      # cf. mol.irrep_name
```

4c: `so_4c = block_diag(so, so)` with the SAME irrep for the S block. (First guess was "S irrep = L irrep ⊗
pseudoscalar" because σ·p is odd under i; wrong — inversion acts on 4-spinors as βP, whose extra sign on the small
component cancels that of σ·p. Checked 2026-10-16: with the ⊗Γ_ps mapping the DHF L–S hcore block couples
different irreps by ~1e2, with the same-irrep mapping < 1e-14.) Provided by a helper `dirac_symm_orb(so, irrep_id, doublegroupname)`.

## 3. SCF layer: module functions + mixin

Module functions (no SCF object needed; usable by MCSCF/CC/property code):

```python
eig(h, s, so, irrep_id, doublegroupname, eigh, kramers=False, tr_map=None)  # block solve, returns e, c tagged with orbsym
label_orb_symm(mo, s, so, irrep_id, thresh)                        # orbsym for given MOs (purity check)
get_occ(mo_energy, orbsym, nelec, irrep_nelec, doublegroupname, mask=None)  # mask: DHF positive-energy states
symmetrize_space(...)                                              # optional, later
```

`eig` with `kramers=True`: for each pair (Γ, conj Γ) with Γ ≠ conj Γ, diagonalize only Γ with `scipy.linalg.eigh`
and build the partner block by time reversal; for self-conjugate Γ call the host's Kramers `_eigh` (zquatev).
`kramers=False`: plain `eigh` on every block.

Mixin:

```python
class DoubleGroupSymm:
    _keys = {'irrep_nelec'}
    def build(self, mol=None)              # check irrep_nelec names / counts
    def eig(self, h, s, *args, **kwargs)   # -> module eig with self._dg_symm_orb(), self._eigh
    def get_grad(self, mo_coeff, mo_occ, fock=None)   # zero symmetry-forbidden occ-vir pairs
    def get_occ(self, mo_energy=None, mo_coeff=None)
    def _finalize(self)                    # sort by energy, keep orbsym
    def get_orbsym(self, mo_coeff=None, s=None)
    def get_irrep_nelec(self, ...)
    def analyze(self, ...)                 # per-irrep occupation table

    # hooks the host class provides (defaults in the mixin cover the 2c case)
    def _dg_symm_orb(self)            # 2c: mol.symm_orb_spinor; DHF: dirac_symm_orb(...)
    def _occ_mask(self, mo_energy)    # 2c: all True; DHF: positive-energy states
    _kramers_restricted = False       # True for RHF / RDHF
```

Classes (prototype names):

```python
class SymAdaptedUHF(DoubleGroupSymm, spinor_hf.UHF)
class SymAdaptedRHF(DoubleGroupSymm, spinor_hf.RHF)
class SymAdaptedDHF(DoubleGroupSymm, dhf.DHF)
class SymAdaptedRDHF(DoubleGroupSymm, dhf.RDHF)
```

Irrep names (user's decision 2026-10-16):
- finite groups, Altmann–Herzig / DIRAC short names: fermion `1E`, `2E` (C2, Cs), `1Eg 2Eg 1Eu 2Eu` (C2h);
  `1E` has character +i under C2 (Cs: σh) in the passive convention R(φ) = exp(+iφJ_z) of Altmann–Herzig and
  DIRAC; 1E contains m_j = 1/2, −3/2, 5/2, … = DIRAC's unbarred spinors (checked 2026-10-16: DIRAC LINSYM output
  for HBr lists MJ = 1/2, −3/2, 5/2; moltra/traone.F assigns unbarred spinors to 1E). AO-layer phases use the
  same convention (C2z: e^{+iπ m_j}). C1/Ci fermion irreps are `A1/2`, `A1/2g`, `A1/2u` (DIRAC's `A`,
  `AG`, `AU` would clash with PySCF's boson names `A`, `Ag`, `Au`). Boson irreps keep PySCF's single-group names.
- linear groups: m_j names, fermion `1/2`, `-3/2`, (`1/2g`, `-3/2u` for Dooh); boson m_l: `0`, `1`, `-2` (+ g/u).
- `irrep_nelec` keys are these names; m_j names exist only for linear molecules.
With Kramers
restriction a pair (Γ, conj Γ) must get equal counts; the mixin checks this.

Existing `SymmSpinorSCF` (`symmetry='linear'|'sph'`) and `linear_dhf.SymmDHF` become thin wrappers over the
above once validated; their `occup` dict format is translated to `irrep_nelec`.

## 4. Validation plan

1. Group layer unit tests: product tables, `conj`, closure; C2h*/C2* tables compared with DIRAC
   (`moltra/traone.F` `GMULTA`).
2. AO layer: for each test molecule and group, `so^† S so` and `so^† (T+V) so` block-diagonal (< 1e-12),
   SALCs orthonormal and complete, partner blocks equal T·so. Molecules: H2O, CH4 (D2→C2*), C2H4 (D2h→C2h*),
   HBr / Br2 (linear), an atom. Each also from a randomly rotated input through `to_symm_frame`: energies equal,
   lab-frame dipole / EFG equal to the C1 lab-frame run.
3. SCF: energies equal to the C1 run to 1e-10 for UHF/RHF (x2camf, x2cmp) and DHF/RDHF; `irrep_nelec`
   reproduces the C1 ground state; labels compared with CFOUR `dgcosym` on one case.
4. Runs on the interactive node or Slurm, not the login node.

## 5. Later (not phase 1)

- Spinor DFT (user 2026-10-16: current state is enough for now): align with pyscf when moving to
  pyscf/x2c/dft.py — UKS(dks.KohnShamDFT, spinor_hf.UHF), RKS(dks.KohnShamDFT, spinor_hf.RHF),
  SymAdaptedUKS/RKS(dks.KohnShamDFT, SymAdaptedUHF/RHF) so that SymAdaptedUHF.to_ks() keeps the symmetry
  (today it returns the non-symmetric KS class). Non-collinear: pyscf numint2c has 'ncol' only for LDA vxc
  (no GGA/MGGA, no fxc); GGA needs 'mcol' (mcfun). A traditional non-collinear GGA is a separate numint
  project (options to be researched from the literature first).

- D^j(R) = C^† [D^l_real(R) ⊗ U_SU(2)(R)] C per shell (C = `sph2spinor`, D^l from `pyscf.symm.Dmatrix`), for
  rotating orbitals / density matrices between frames, and if mainline review asks for lab-frame SALCs.
- Realification (phase 2, user's decision 2026-10-16): after every diagonalization, for groups D2/C2v/D2h (labels
  from C2*/C2h*) and linear molecules, fix the phase of each MO so that it is invariant under the antiunitary
  K = C2x·T (C2v: σx·T; linear: σv·T). Derivation: C2x C2z C2x^{-1} = C2z^{-1}, so C2x and T both map 1E ↔ 2E and
  K maps 1E → 1E; K² = (C2x)² T² = (−1)(−1) = +1; KT = TK. With Kφ = φ for the 1E MOs and φ̄ = Tφ, all
  one-electron and two-electron MO integrals (Coulomb, Gaunt, Breit, X2CAMF/X2CMP SO terms) are real.
  Per MO: Kφ = e^{iθ}φ (no degeneracy inside a 1E block) → φ ← e^{iθ/2}φ; degenerate subspaces: diagonalize K.
  On spinor AOs K = atom permutation × phase × complex conjugation (m_j preserved).
  Valid only when F commutes with T: Kramers-restricted RHF/RDHF (closed shell, average of configuration);
  not for Kramers-unrestricted UHF or magnetic fields. C2*/Cs*/C2h* (no extra C2x), Ci*, C1* stay complex.
  Downstream orbital rotations (CASSCF, localization) must be real orthogonal to keep this.
  Check to run then: C2H4 / HBr, max |Im| of MO integrals ~1e-14.
- κ blocking for atoms (current `symmetry='sph'` mode).
- Double-group direct product consumers: CASCI/FCI/CC (zccsdt export), ao2mo with symmetry.
- PySCF issue: `GHF(mol).x2c()` with `mol.symmetry=True` (warn + fall back to C1).

## 6. Open questions

- Whether linear groups should also report |m_j|-based names for printing (`Ω = 1/2, 3/2, ...`).
- Where the double-group functions live in PySCF (`symm/` new module) and how `mol.build` triggers them
  (lazy property vs explicit `mol.build(..., double_group=True)`).
