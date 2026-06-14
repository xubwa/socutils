"""
Two-gate validation harness for spinor post-HF methods in socutils.

Strategy
--------
Run the spinor machinery (spinor_hf, MP2, EE-ADC(2), G0W0) on an ORDINARY
NON-RELATIVISTIC Hamiltonian. The spinor solution is then just a complex
two-component representation of a problem whose physics PySCF already solves
in a real (restricted) basis. That gives two independent self-checks:

  Gate 1  (physical correctness, needs external gold standard)
      spinor-method(non-rel H)  ==  PySCF non-rel reference
      e.g. spinor MP2 corr E  == pyscf GMP2/RMP2 corr E
           spinor EE-ADC(2)    == pyscf EE-RADC ADC(2)
           spinor G0W0@HF      == pyscf gw (CD)

  Gate 2  (complex-implementation correctness, NO external gold standard)
      method(C)  ==  method(C @ U)   for a complex unitary U
      All physical observables are invariant; only the intermediates
      (mo_coeff, t2, Sigma, ISR vectors) become complex. This is what
      catches the #1 spinor bug class: ".T where it should be .conj().T".

Gate 2 is the workhorse for directions where PySCF has no reference
(CVS-EE, PCE-corrected GW): no gold standard, but still a closed-loop test.

IMPORTANT — which U is legal for Gate 2
---------------------------------------
Canonical MP2/ADC/GW assume diagonal Fock. An arbitrary occupied-occupied
rotation changes orbital energies and breaks the canonical formulas, giving
FALSE failures. Only two rotation types preserve canonicality while still
forcing coefficients complex:
  (a) per-orbital complex phase   diag(e^{i theta_p})        -- always legal
  (b) complex unitary WITHIN an exactly-degenerate energy block
      (Kramers pairs are exactly such blocks)               -- legal
We build U from (a)+(b) only. Degenerate blocks are detected from mo_energy,
so the Kramers structure need not be specified explicitly.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ADAPTER LAYER  --  wired to the real socutils API.
#
# build_spinor_mf  -> socutils.scf.spinor_hf.SpinorSCF   (non-rel spinor HF)
# run_spinor_mp2   -> socutils.mp.SpinorMP2              (spin-orbital MP2)
#
# The spinor mean field carries mo_coeff of shape (n2c, nmo) and real
# mo_energy of shape (nmo,), with n2c = mol.nao_2c() (every column is a full
# complex two-component spin-orbital).  SpinorMP2 reads mo_coeff/mo_energy/
# mo_occ off the mf at construction time, so Gate 2 (inject C @ U) works by a
# plain mf.copy() + overwrite, with no re-diagonalisation.
# ---------------------------------------------------------------------------

from pyscf import gto, scf, mp

from socutils.scf import spinor_hf
from socutils.mp import SpinorMP2


def build_mol(atom, basis="cc-pvdz", spin=0, charge=0):
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    return mol


def build_spinor_mf(mol):
    """Spinor mean field on the NON-RELATIVISTIC Hamiltonian.

    SpinorSCF builds its hcore from the non-relativistic spinor kinetic +
    nuclear integrals (int1e_kin_spinor / int1e_nuc_spinor), i.e. no X2C and
    no spin-orbit coupling.  The solution is a complex two-component image of
    the ordinary restricted problem and its total energy matches RHF.
    """
    mf = spinor_hf.SpinorSCF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf


def mo_coeff(mf):
    return np.asarray(mf.mo_coeff)          # shape (n2c, nmo), complex


def mo_energy(mf):
    return np.asarray(mf.mo_energy)


def set_mo(mf, C, E=None):
    """Return an mf-like object carrying rotated MOs, for Gate 2.

    Energies are UNCHANGED by a legal U (phases / degenerate-block mixing),
    so we keep E unless explicitly given.  SpinorMP2 consumes mo_coeff and the
    canonical mo_energy straight off this object -- there is no cached Fock or
    eri bound to mf, and nothing is re-diagonalised.
    """
    mf_rot = mf.copy()
    mf_rot.mo_coeff = C
    if E is not None:
        mf_rot.mo_energy = E
    return mf_rot


# --- method runners: each returns a dict of INVARIANT scalars/arrays ---------

def run_spinor_mp2(mf):
    """socutils spinor MP2.  Returns invariant scalars only: the correlation
    energy and the (unitary-invariant) Frobenius norm of the t2 amplitudes."""
    pt = SpinorMP2(mf).run()
    return {"e_corr": pt.e_corr, "t2_norm": np.linalg.norm(pt.t2)}


def run_spinor_eeadc2(mf, nroots=6):
    """TODO: return socutils spinor EE-ADC(2): excitation energies + osc str.

    Default is a placeholder so the harness imports/runs; replace with the
    real call. Gate-1 reference is pyscf EE-RADC (see reference runners).
    """
    raise NotImplementedError("wire run_spinor_eeadc2 to socutils EE-ADC(2)")


def run_spinor_g0w0(mf, norb=None):
    """TODO: return socutils spinor G0W0 QP energies (CD).

    Default placeholder; Gate-1 reference is pyscf gw (CD).
    """
    raise NotImplementedError("wire run_spinor_g0w0 to socutils G0W0-CD")


# --- Gate-1 external references (PySCF, non-relativistic) --------------------

def ref_mp2(mol):
    mf = scf.RHF(mol).run()
    pt = mp.MP2(mf).run()
    return {"e_corr": pt.e_corr}


def ref_eeadc2(mol, nroots=6):
    from pyscf import adc
    mf = scf.RHF(mol).run()
    myadc = adc.ADC(mf)
    myadc.method = "adc(2)"
    e, v, p, x = myadc.kernel(nroots=nroots)   # EE by default
    return {"exc": np.asarray(e)}


def ref_g0w0(mol, nocc_act=None, nvir_act=None):
    from pyscf import gw
    mf = scf.RHF(mol).run()
    g = gw.GW(mf, freq_int="cd")               # contour deformation
    g.kernel()
    return {"qp": np.asarray(g.mo_energy)}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def degenerate_blocks(energies, tol=1e-7):
    """Group MO indices into exactly-degenerate blocks (sorted-energy based)."""
    order = np.argsort(energies)
    blocks, cur = [], [order[0]]
    for i in order[1:]:
        if abs(energies[i] - energies[cur[-1]]) <= tol:
            cur.append(i)
        else:
            blocks.append(cur)
            cur = [i]
    blocks.append(cur)
    return blocks


def random_complex_unitary(n, rng):
    """Haar-ish complex unitary via QR of a complex Gaussian matrix."""
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    # fix phase so result is genuinely Haar-distributed
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def legal_rotation(energies, seed=0, phase=True, mix_degenerate=True):
    """Build a LEGAL Gate-2 unitary U (nmo x nmo): per-MO phases + complex
    rotation within each exactly-degenerate energy block. Preserves canonical
    (diagonal-Fock) structure, so MP2/ADC/GW formulas stay valid."""
    rng = np.random.default_rng(seed)
    nmo = len(energies)
    U = np.eye(nmo, dtype=complex)
    if phase:
        theta = rng.uniform(0, 2 * np.pi, nmo)
        U = U * np.exp(1j * theta)            # diag phase
    if mix_degenerate:
        for blk in degenerate_blocks(energies):
            if len(blk) > 1:
                u = random_complex_unitary(len(blk), rng)
                idx = np.ix_(blk, blk)
                U[idx] = U[idx] @ u
    return U


def assert_set_close(a, b, atol, rtol=0, label=""):
    """Degeneracy-safe comparison of energy/strength SETS (sort then compare).
    Use this for excitation energies and QP-energy sets, never element-by-
    element on eigenvectors."""
    a = np.sort(np.real_if_close(np.asarray(a)))
    b = np.sort(np.real_if_close(np.asarray(b)))
    assert a.shape == b.shape, f"{label}: length mismatch {a.shape} vs {b.shape}"
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=label)


# ---------------------------------------------------------------------------
# FIXTURES  --  small, fast, span no-SOC light atoms (gold standard valid)
# ---------------------------------------------------------------------------

MOLS = {
    "Ne":  "Ne 0 0 0",
    "HF":  "H 0 0 0; F 0 0 0.917",
    "H2O": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
}


@pytest.fixture(params=list(MOLS), ids=list(MOLS))
def mol(request):
    return build_mol(MOLS[request.param])


@pytest.fixture
def mf(mol):
    return build_spinor_mf(mol)


# ===========================================================================
# GATE 1 : physical correctness vs PySCF non-relativistic reference
# ===========================================================================

def test_gate1_mp2(mol, mf):
    got = run_spinor_mp2(mf)["e_corr"]
    ref = ref_mp2(mol)["e_corr"]
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=0,
                               err_msg="spinor MP2 corr E != PySCF MP2")


@pytest.mark.xfail(reason="enable once run_spinor_eeadc2 is wired",
                   raises=NotImplementedError)
def test_gate1_eeadc2(mol, mf):
    got = run_spinor_eeadc2(mf, nroots=6)["exc"]
    ref = ref_eeadc2(mol, nroots=6)["exc"]
    # spinor reproduces restricted roots up to Kramers multiplicity; compare
    # the unique set. Tighten/loosen tol to your solver convergence.
    uniq_got = np.unique(np.round(got, 6))
    assert_set_close(uniq_got[:len(ref)], ref, atol=1e-5,
                     label="EE-ADC(2) exc energies")


@pytest.mark.xfail(reason="enable once run_spinor_g0w0 is wired",
                   raises=NotImplementedError)
def test_gate1_g0w0(mol, mf):
    got = run_spinor_g0w0(mf)["qp"]
    ref = ref_g0w0(mol)["qp"]
    assert_set_close(got, ref, atol=1e-4, label="G0W0 QP energies")


# ===========================================================================
# GATE 2 : invariance under a legal complex orbital rotation (no gold standard)
# ===========================================================================

@pytest.mark.parametrize("method,key,atol", [
    (run_spinor_mp2, "e_corr", 1e-9),
])
def test_gate2_invariance_scalar(mf, method, key, atol):
    """Scalar invariants (corr E) must be identical for C and C@U."""
    E = mo_energy(mf)
    C = mo_coeff(mf)
    U = legal_rotation(E, seed=1)
    base = method(mf)[key]
    rot = method(set_mo(mf, C @ U, E))[key]
    np.testing.assert_allclose(rot, base, atol=atol, rtol=0,
                               err_msg=f"{key} changed under complex rotation"
                                       " -> conjugation/Hermiticity bug")


def test_gate2_intermediate_norm(mf):
    """Diagnostic: covariant tensors have UNITARY-INVARIANT norms.

    ||t2|| is invariant under a legal U; if it moves, the bug is localized to
    the amplitude contraction rather than a downstream energy expression.
    Extend with Sigma trace, ISR-block norms, etc. as those land.
    """
    E = mo_energy(mf)
    C = mo_coeff(mf)
    U = legal_rotation(E, seed=2)
    n_base = run_spinor_mp2(mf)["t2_norm"]
    n_rot = run_spinor_mp2(set_mo(mf, C @ U, E))["t2_norm"]
    np.testing.assert_allclose(n_rot, n_base, atol=1e-9, rtol=0,
                               err_msg="||t2|| not rotation-invariant"
                                       " -> covariant contraction has a bug")


@pytest.mark.xfail(reason="enable once run_spinor_eeadc2 is wired",
                   raises=NotImplementedError)
def test_gate2_eeadc2_invariance(mf):
    E, C = mo_energy(mf), mo_coeff(mf)
    U = legal_rotation(E, seed=3)
    base = run_spinor_eeadc2(mf)["exc"]
    rot = run_spinor_eeadc2(set_mo(mf, C @ U, E))["exc"]
    assert_set_close(rot, base, atol=1e-8, label="EE-ADC(2) under rotation")


@pytest.mark.xfail(reason="enable once run_spinor_g0w0 is wired",
                   raises=NotImplementedError)
def test_gate2_g0w0_invariance(mf):
    E, C = mo_energy(mf), mo_coeff(mf)
    U = legal_rotation(E, seed=4)
    base = run_spinor_g0w0(mf)["qp"]
    rot = run_spinor_g0w0(set_mo(mf, C @ U, E))["qp"]
    assert_set_close(rot, base, atol=1e-7, label="G0W0 QP under rotation")


# ---------------------------------------------------------------------------
# Sanity: the rotation we hand out is actually unitary and actually complex.
# ---------------------------------------------------------------------------

def test_rotation_is_unitary_and_complex(mf):
    E = mo_energy(mf)
    U = legal_rotation(E, seed=5)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(len(E)), atol=1e-12)
    assert np.max(np.abs(U.imag)) > 1e-3, "U is effectively real; would not" \
                                          " exercise the complex code path"
