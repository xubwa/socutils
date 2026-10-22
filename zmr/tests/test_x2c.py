'''X2C spinor Hamiltonians (validation items 3 and 4): O2 X 3Sigma_g- with
CAS(2e, 4 pi_g* spinors), model space = all six CASCI states.'''
import numpy
import pytest
from scipy.stats import unitary_group
from socutils.fci import zfci
from socutils.mcscf import zcasci

from socutils.zmr.active import symmetry
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import socutils_x2c
from socutils.zmr.reference.nevpt2_bruteforce import BruteForceNEVPT2
from socutils.zmr.tests.pipeline import Setup
from socutils.zmr.tests.systems_x2c import o2_dianion_orbitals
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc.nevpt2 import UCNEVPT2

HARTREE2WAVENUMBER = 219474.6313632
NFROZEN, NCORE, NACT, NELEC = 4, 10, 4, 2   # spinors


def _o2(soc, basis='cc-pvdz'):
    mf, mol = o2_dianion_orbitals(basis, soc)
    eris = socutils_x2c.from_scf(mf, NCORE, NACT, nfrozen=NFROZEN, mol=mol)
    return mf, mol, eris


@pytest.mark.parametrize('soc', [True, False])
def test_casci_matches_socutils(soc):
    mf, mol, eris = _o2(soc)
    assert eris.symmetry_error() < 1e-12
    e, _ = ActiveSpace.from_eris(eris).solve(NELEC, 6)
    mc = zcasci.CASCI(mf, NACT, NELEC)
    mc.ncore = NFROZEN + NCORE
    h1, ecore = mc.get_h1eff()
    ref, _ = zfci.kernel(h1, mc.ao2mo().reshape((NACT,) * 4), NACT, NELEC,
                         ecore=ecore.real, nroots=6)
    assert abs(e - ref).max() < 1e-10


@pytest.mark.parametrize('soc', [True, False])
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_bruteforce_x2c_sto3g(soc, e0):
    '''Validation item 4: spinor integrals, class by class vs brute force.'''
    _, _, eris = _o2(soc, basis='sto-3g')
    uc = UCNEVPT2(eris, NELEC, nstate=6, e0=e0)
    uc.kernel()
    s = Setup(eris, NELEC, civecs=uc.civecs)
    bf = BruteForceNEVPT2(s.eris, s.eps_core, s.eps_virt, NELEC, uc.civecs, e0=e0)
    bf.kernel()
    for name in bf.h2_by_class:
        assert abs(uc.h2_by_class[name] - bf.h2_by_class[name]).max() < 1e-10, name
    assert abs(uc.energies - bf.energies).max() < 1e-10


@pytest.mark.parametrize('e0', ['state', 'average'])
def test_triplet_degenerate_without_soc(e0):
    _, _, eris = _o2(soc=False)
    uc = UCNEVPT2(eris, NELEC, nstate=6, e0=e0)
    e = uc.kernel()
    assert abs(e[:3] - e[0]).max() < 1e-10      # X 3Sigma_g-
    assert abs(e[3:5] - e[3]).max() < 1e-10     # a 1Delta_g
    assert e[3] - e[2] > 1e-2
    assert abs(uc.h2).max() > 1e-2


def _omega_basis(uc, mol, mf):
    '''Model states rotated to Jz eigenstates, and their Omega values.'''
    jz_act = socutils_x2c.jz_mo(mol, mf.mo_coeff)[NFROZEN + NCORE:NFROZEN + NCORE + NACT,
                                                  NFROZEN + NCORE:NFROZEN + NCORE + NACT]
    act = ActiveSpace.from_eris(uc.eris0)
    c = uc.civecs
    n = c.shape[1]
    jz = numpy.array([[numpy.sum(jz_act * act.trans_rdm1(c[:, i], c[:, j], NELEC))
                       for j in range(n)] for i in range(n)])
    assert abs(jz - jz.conj().T).max() < 1e-10
    omega, w = numpy.linalg.eigh(jz)
    assert abs(omega - numpy.round(omega)).max() < 1e-8
    return c @ w, numpy.round(omega).astype(int)


def test_effective_hamiltonian_is_block_diagonal_in_omega():
    mf, mol, eris = _o2(soc=True)
    uc = UCNEVPT2(eris, NELEC, nstate=6, e0='average')
    uc.kernel()
    civecs, omega = _omega_basis(uc, mol, mf)
    assert sorted(omega.tolist()) == [-2, -1, 0, 0, 1, 2]
    rot = UCNEVPT2(eris, NELEC, civecs=civecs, e0='average')
    rot.kernel()
    differ = omega[:, None] != omega[None, :]
    assert abs(rot.heff[differ]).max() < 1e-10
    # same-Omega coupling (X 3Sigma_g-(0+) with b 1Sigma_g+) is present
    assert abs(rot.heff[~differ & ~numpy.eye(6, dtype=bool)]).max() > 1e-6
    assert abs(rot.energies - uc.energies).max() < 1e-10


def test_zero_field_splitting():
    '''X 3Sigma_g- splits into Omega = 0 and a degenerate Omega = +-1 pair.'''
    _, _, eris = _o2(soc=True)
    uc = UCNEVPT2(eris, NELEC, nstate=6, e0='average')
    e = uc.kernel()
    e_cas = numpy.linalg.eigvalsh(uc.hmodel)
    for x in (e_cas, e):
        assert abs(x[1] - x[2]) < 1e-10
        zfs = (x[1] - x[0]) * HARTREE2WAVENUMBER
        assert 0.5 < zfs < 10


def test_model_space_rotation_invariance_x2c():
    _, _, eris = _o2(soc=True)
    ref = UCNEVPT2(eris, NELEC, nstate=6, e0='average')
    ref.kernel()
    u = unitary_group.rvs(6, random_state=7)
    rot = UCNEVPT2(eris, NELEC, civecs=ref.civecs @ u, e0='average')
    rot.kernel()
    assert abs(rot.energies - ref.energies).max() < 1e-10


@pytest.mark.parametrize('soc', [True, False])
def test_symmetry_adapted_active_spinors(soc):
    '''The pi_g* spinors come out of the SCF as arbitrary mixtures within
    their degenerate pairs.  The adaptation finds the rotation about the axis
    (J_z; without SOC also the spin and L_z), makes every sector block
    diagonal and leaves all energies unchanged.'''
    _, _, eris = _o2(soc)
    adapted, sym = symmetry.adapt(eris)
    assert sym.ngen >= (2 if soc else 5)
    assert sym.error < 1e-10
    act = ActiveSpace.from_eris(adapted)
    assert ActiveSpace.from_eris(eris).blocks(NELEC)[1] == 1
    assert act.blocks(NELEC)[1] > 1
    for cls, kw in ((UCNEVPT2, {}), (PCNEVPT2, {}), (PCNEVPT2, {'contraction': 'state'})):
        ref = cls(eris, NELEC, nstate=6, e0='average', **kw)
        new = cls(adapted, NELEC, nstate=6, e0='average', **kw)
        assert abs(new.kernel() - ref.kernel()).max() < 1e-10
