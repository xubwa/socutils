'''Non-relativistic (spin-free) UC- and PC-NEVPT2:

* agreement with the spinor code on the same Hamiltonian in spin-orbital
  form: UC always, PC for singlets (where the spin-free functions span the
  same space as the single spin-orbital strings);
* independence of the spin projection of the model states;
* PySCF (strongly contracted bound) and ORCA FIC-NEVPT2 for closed- and
  open-shell states.'''
import json
import os

import numpy
import pytest
from pyscf import gto, scf
from pyscf.tools import molden

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.nr import PCNEVPT2, UCNEVPT2, classes
from socutils.zmr.nr.active import ActiveSpace
from socutils.zmr.nr.spinor import embed, spinor_eris
from socutils.zmr.pc.nevpt2 import PCNEVPT2 as SpinorPC
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import nr_model_eris
from socutils.zmr.uc.nevpt2 import UCNEVPT2 as SpinorUC

DATA = os.path.join(os.path.dirname(__file__), 'data', 'orca')
REF = json.load(open(os.path.join(DATA, 'reference.json')))
NACT, NELEC = 3, (2, 2)


@pytest.fixture(scope='module')
def model():
    '''Model Hamiltonian, the Ms = 0 spectrum and the indices of its lowest
    singlets and triplets.'''
    eris = nr_model_eris(2, NACT, 2, seed=1)
    act = ActiveSpace.from_eris(eris)
    sec = act.sector(NELEC)
    high = act.sector((3, 1)).lam                      # Ms = 1: the triplets
    triplet = [int(numpy.argmin(abs(sec.lam - x))) for x in high]
    singlet = [k for k in range(sec.dim) if k not in triplet]
    return eris, act, sec.vecs, singlet, triplet


def _run(cls, eris, nelec, c, **kw):
    obj = cls(eris, nelec, civecs=c, **kw)
    obj.kernel()
    return obj


def _same(a, b, tol=1e-10):
    assert abs(a.hmodel - b.hmodel).max() < tol
    for name in classes.CLASS_NAMES:
        assert abs(a.h2_by_class[name] - b.h2_by_class[name]).max() < tol, name
    assert abs(a.energies - b.energies).max() < tol


@pytest.mark.parametrize('states', ['singlets', 'triplet', 'mixed'])
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_uc_equals_spinor_code(model, states, e0):
    eris, act, vecs, singlet, triplet = model
    k = {'singlets': singlet[:3], 'triplet': triplet[:1], 'mixed': [singlet[0], triplet[0]]}[states]
    c = vecs[:, k]
    a = _run(UCNEVPT2, eris, NELEC, c, e0=e0)
    b = _run(SpinorUC, spinor_eris(eris), sum(NELEC), embed(c, NACT, NELEC), e0=e0)
    _same(a, b)
    assert all(abs(a.h2_by_class[name]).max() > 1e-4 for name in classes.CLASS_NAMES)
    # real arithmetic throughout
    assert a.civecs.dtype == a.heff.dtype == numpy.float64
    assert all(x.dtype == numpy.float64 for x in a.h2_by_class.values())


@pytest.mark.parametrize('contraction', ['union', 'state'])
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_pc_of_singlets_equals_spinor_code(model, contraction, e0):
    eris, act, vecs, singlet, _ = model
    c = vecs[:, singlet[:3]]
    kw = dict(e0=e0, contraction=contraction)
    a = _run(PCNEVPT2, eris, NELEC, c, **kw)
    b = _run(SpinorPC, spinor_eris(eris), sum(NELEC), embed(c, NACT, NELEC), **kw)
    _same(a, b)


def test_pc_of_a_triplet_is_more_contracted_than_spin_orbital_pc(model):
    '''For S > 0 the single spin-orbital strings span more than the spin-free
    functions: the spinor code (given one spin component) lies lower.'''
    eris, act, vecs, _, triplet = model
    c = vecs[:, triplet[:1]]
    a = _run(PCNEVPT2, eris, NELEC, c)
    b = _run(SpinorPC, spinor_eris(eris), sum(NELEC), embed(c, NACT, NELEC))
    diff = a.h2[0, 0].real - b.h2[0, 0].real
    assert diff > 1e-6
    uc = _run(UCNEVPT2, eris, NELEC, c)
    assert uc.h2[0, 0].real < b.h2[0, 0].real + 1e-12


@pytest.mark.parametrize('cls, kw', [(UCNEVPT2, {}), (PCNEVPT2, {}),
                                     (PCNEVPT2, {'contraction': 'state'}),
                                     (PCNEVPT2, {'method': {'ccav': 'laplace', 'cavv': 'laplace'},
                                                 'nquad': 48})])
def test_spin_projection_does_not_matter(model, cls, kw):
    '''Two triplets as Ms = 1, 0, -1 components.'''
    eris, act, vecs, _, triplet = model
    ref = _run(cls, eris, (3, 1), act.sector((3, 1)).vecs[:, :2], e0='average', **kw)
    for nelec, c in (((2, 2), vecs[:, triplet[:2]]), ((1, 3), act.sector((1, 3)).vecs[:, :2])):
        other = _run(cls, eris, nelec, c, e0='average', **kw)
        assert abs(other.energies - ref.energies).max() < 1e-10
        for name in classes.CLASS_NAMES:
            assert abs(numpy.linalg.eigvalsh(other.h2_by_class[name])
                       - numpy.linalg.eigvalsh(ref.h2_by_class[name])).max() < 1e-10, name


def test_laplace_and_exact_agree(model):
    eris, act, vecs, singlet, triplet = model
    c = vecs[:, [singlet[0], triplet[0], singlet[1]]]
    for cls in (UCNEVPT2, PCNEVPT2):
        ex = _run(cls, eris, NELEC, c, e0='average')
        lp = _run(cls, eris, NELEC, c, e0='average', method='laplace', nquad=48)
        assert abs(lp.heff - ex.heff).max() < 1e-10


def test_union_pc_is_invariant_under_model_space_rotations(model):
    from scipy.stats import ortho_group
    eris, act, vecs, singlet, triplet = model
    c = vecs[:, [singlet[0], triplet[0], singlet[1]]]
    ref = _run(PCNEVPT2, eris, NELEC, c, e0='average')
    u = ortho_group.rvs(3, random_state=3)
    rot = _run(PCNEVPT2, eris, NELEC, c @ u, e0='average')
    assert abs(rot.energies - ref.energies).max() < 1e-10
    assert abs(rot.h2 - u.T @ ref.h2 @ u).max() < 1e-10


def test_complex_model_states(model):
    '''Complex combinations of the model states (a unitary rotation) are
    accepted; the union contraction does not change.'''
    from scipy.stats import unitary_group
    eris, act, vecs, singlet, triplet = model
    c = vecs[:, [singlet[0], triplet[0], singlet[1]]]
    ref = _run(PCNEVPT2, eris, NELEC, c, e0='average')
    u = unitary_group.rvs(3, random_state=4)
    rot = _run(PCNEVPT2, eris, NELEC, c @ u, e0='average')
    assert ref.heff.dtype == numpy.float64 and rot.heff.dtype == numpy.complex128
    assert abs(rot.heff - u.conj().T @ ref.heff @ u).max() < 1e-10


@pytest.mark.parametrize('kw', [{}, {'contraction': 'state'},
                                {'method': {'ccav': 'laplace', 'cavv': 'laplace'}, 'nquad': 48}])
def test_matrix_free_sectors(model, kw):
    '''Above dense_limit the sector Hamiltonians are only applied to vectors;
    PC-NEVPT2 with given model states does not notice.'''
    from socutils.zmr import lib
    if lib.zmr() is None:
        pytest.skip('matrix-free sectors need libzmr')
    eris, act, vecs, singlet, triplet = model
    c = vecs[:, [singlet[0], triplet[0], singlet[1]]]
    ref = _run(PCNEVPT2, eris, NELEC, c, e0='average', **kw)

    class MatrixFree(PCNEVPT2):
        def make_active_space(self, eris):
            space = ActiveSpace.from_eris(eris)
            space.dense_limit = 0
            return space

    free = _run(MatrixFree, eris, NELEC, c, e0='average', **kw)
    assert not free.active_space._hmat                 # no dense Hamiltonian was built
    assert abs(free.heff - ref.heff).max() < 1e-10
    # what needs a spectrum says so
    space = ActiveSpace.from_eris(eris)
    space.dense_limit = 0
    for call in (lambda: space.sector(NELEC), lambda: space.solve(NELEC, 2, full=False)):
        with pytest.raises(NotImplementedError):
            call()
    q = numpy.random.default_rng(0).normal(size=(act.dim(NELEC), 3))
    assert abs(space.apply_h(NELEC, q) - act.hmat(NELEC) @ q).max() < 1e-12
    assert abs(space.apply_h(NELEC, q * 1j) - 1j * (act.hmat(NELEC) @ q)).max() < 1e-12


def test_default_model_states_and_bounds(model):
    eris = model[0]
    uc = UCNEVPT2(eris, NELEC, nstate=2)
    pc = PCNEVPT2(eris, NELEC, nstate=2)
    assert abs(uc.kernel() - numpy.sort(uc.kernel())).max() == 0
    pc.kernel()
    for name in classes.CLASS_NAMES:
        d_uc = uc.h2_by_class[name].diagonal().real
        d_pc = pc.h2_by_class[name].diagonal().real
        assert numpy.all(d_uc <= d_pc + 1e-12), name
    with pytest.raises(ValueError):
        PCNEVPT2(eris, NELEC, amplitudes='4rdm')


# ----------------------------------------------------------------------
# molecules

def test_h2o_vs_spinor_route_and_pyscf():
    '''PySCF CASCI object -> spin-restricted integrals and CI vector; the
    GHF-expanded route through the spinor code gives the same class energies,
    PySCF's strongly contracted ones lie above.'''
    from socutils.zmr.tests.test_pyscf_compare import _sc_nevpt2_by_class
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, nelec = pyscf_nr.nr_from_casci(mc, nfrozen=1)
    assert nelec == (2, 2) and eris.h1.dtype == numpy.float64
    c = pyscf_nr.nr_civecs(mc)
    pc = _run(PCNEVPT2, eris, nelec, c)
    uc = _run(UCNEVPT2, eris, nelec, c)
    assert abs(pc.hmodel[0, 0].real - mc.e_tot) < 1e-9
    so_eris, n = pyscf_nr.from_casci(mc, nfrozen=1)
    so_pc = SpinorPC(so_eris, n)
    so_uc = SpinorUC(so_eris, n)
    so_pc.kernel()
    so_uc.kernel()
    for name in classes.CLASS_NAMES:
        assert abs(pc.h2_by_class[name][0, 0] - so_pc.h2_by_class[name][0, 0]) < 1e-9, name
        assert abs(uc.h2_by_class[name][0, 0] - so_uc.h2_by_class[name][0, 0]) < 1e-9, name
    # no frozen core: bounded by SC-NEVPT2 class by class
    sc, _ = _sc_nevpt2_by_class(mc)
    eris0, _ = pyscf_nr.nr_from_casci(mc)
    pc0 = _run(PCNEVPT2, eris0, nelec, c)
    uc0 = _run(UCNEVPT2, eris0, nelec, c)
    for name in classes.CLASS_NAMES:
        e_uc, e_pc = uc0.h2_by_class[name][0, 0].real, pc0.h2_by_class[name][0, 0].real
        assert e_uc <= e_pc + 1e-10 <= sc[name] + 2e-10, name
    assert abs(pc0.h2_by_class['ccvv'][0, 0].real - sc['ccvv']) < 1e-9


@pytest.mark.parametrize('nfrozen', [0, 1])
def test_empty_active_space_is_mp2(nfrozen):
    from pyscf import mp
    mf = systems.h2o_rhf('6-31g')
    nocc = mf.mol.nelectron // 2
    eris = pyscf_nr.nr_eris(mf.mol, mf.get_hcore(), mf.mo_coeff, nfrozen, nocc - nfrozen, 0)
    e_mp2 = mp.MP2(mf, frozen=nfrozen or None).run(verbose=0).e_corr
    for cls in (UCNEVPT2, PCNEVPT2):
        pt = cls(eris, (0, 0))
        pt.kernel()
        assert abs(pt.hmodel[0, 0] - mf.e_tot) < 1e-10
        assert abs(pt.h2[0, 0] - e_mp2) < 1e-10
        for name, a in pt.h2_by_class.items():
            if name != 'ccvv':
                assert abs(a).max() < 1e-14, name


def test_high_spin_determinant():
    '''ROHF triplet O2 with two active alpha electrons: one determinant, so
    PC = UC; its Ms = 0 component (two determinants) gives the same energy.'''
    from pyscf import mcscf
    mol = gto.M(atom='O 0 0 0; O 0 0 1.2075', basis='6-31g', spin=2, verbose=0)
    rohf = scf.ROHF(mol).run(conv_tol=1e-11)
    mc = mcscf.CASCI(rohf, 2, (2, 0))
    mc.verbose = 0
    mc.kernel()
    eris, nelec = pyscf_nr.nr_from_casci(mc)
    assert nelec == (2, 0) and abs(mc.e_tot - rohf.e_tot) < 1e-10
    c = pyscf_nr.nr_civecs(mc)
    assert c.shape == (1, 1)
    pc = _run(PCNEVPT2, eris, nelec, c)
    uc = _run(UCNEVPT2, eris, nelec, c)
    assert abs(pc.h2[0, 0] - uc.h2[0, 0]) < 1e-10
    assert pc.h2[0, 0] < -0.1
    act = ActiveSpace.from_eris(eris)
    sec = act.sector((1, 1))
    k = int(numpy.argmin(abs(sec.lam + act.e_core - mc.e_tot)))
    assert abs(sec.lam[k] + act.e_core - mc.e_tot) < 1e-10
    low = _run(PCNEVPT2, eris, (1, 1), sec.vecs[:, k])
    for name in classes.CLASS_NAMES:
        assert abs(low.h2_by_class[name][0, 0] - pc.h2_by_class[name][0, 0]) < 1e-10, name


def _orca_case(name):
    ref = REF[name]
    mol = gto.M(atom=ref['atom'], basis=ref['basis'], spin=ref.get('spin', 0), verbose=0)
    _, _, mo, _, _, _ = molden.load(os.path.join(DATA, name + '.molden'))
    s = mol.intor('int1e_ovlp')
    assert abs(mo.T @ s @ mo - numpy.eye(mo.shape[1])).max() < 1e-6
    w, v = numpy.linalg.eigh(mo.T @ s @ mo)
    mo = mo @ (v / numpy.sqrt(w)) @ v.T
    eris = pyscf_nr.nr_eris(mol, scf.hf.get_hcore(mol), mo, 0, ref['ncore'], ref['ncas'])
    return eris, ref


def _state(eris, nelec, e_cas):
    act = ActiveSpace.from_eris(eris)
    sec = act.sector(nelec)
    k = int(numpy.argmin(abs(sec.lam + act.e_core - e_cas)))
    assert abs(sec.lam[k] + act.e_core - e_cas) < 2e-8
    return sec.vecs[:, k]


# classes in which ORCA's fixed 1e-6 metric threshold removes physical functions
ORCA_TRUNCATED = {'h2o_casscf44': ('aavv', 'cava'), 'n2_casscf66': ('caaa', 'aaav'),
                  'o2_t_casscf86': ('aaav',), 'ch2_t_casscf66': ('aaav',)}


@pytest.mark.parametrize('name', ['h2o_casscf44', 'n2_casscf66', 'o2_t_casscf86',
                                  'ch2_t_casscf66'])
def test_orca_fic_nevpt2(name):
    '''ORCA 6.0.1 FIC-NEVPT2 with ORCA's CASSCF orbitals: two closed-shell
    molecules and two triplets, the latter from both the Ms = 1 and the
    Ms = 0 component.'''
    eris, ref = _orca_case(name)
    n, spin = ref['nelecas'], ref.get('spin', 0)
    sectors = {((n + spin) // 2, (n - spin) // 2), (n // 2, n // 2)}
    results = []
    for nelec in sectors:
        pc = _run(PCNEVPT2, eris, nelec, _state(eris, nelec, ref['casscf_energy']))
        results.append(pc)
        for cls in classes.CLASS_NAMES:
            e, e_orca = pc.h2_by_class[cls][0, 0].real, ref['fic_nevpt2'][cls]
            if cls in ORCA_TRUNCATED[name]:
                assert e_orca - 2e-6 < e < e_orca, cls
            else:
                assert abs(e - e_orca) < 1e-9, (cls, e - e_orca)
    if len(results) == 2:
        assert abs(results[0].h2 - results[1].h2).max() < 1e-10


def test_orca_state_averaged_roots():
    '''ORCA's per-root FIC-NEVPT2 after SA-CASSCF (three singlets of H2O)
    uses each root's own Fock matrix: single-state runs on the SA orbitals
    reproduce it, except where ORCA's 1e-6 metric threshold acts (root 0).'''
    eris, ref = _orca_case('h2o_sa3_casscf44')
    nelec = (ref['nelecas'] // 2,) * 2
    for root, (e_cas, orca) in enumerate(zip(ref['casscf_states'], ref['fic_nevpt2_roots'])):
        pc = _run(PCNEVPT2, eris, nelec, _state(eris, nelec, e_cas))
        for cls in classes.CLASS_NAMES:
            e = pc.h2_by_class[cls][0, 0]
            if root == 0 and cls in ('ccaa', 'aavv', 'cava'):
                assert e < orca[cls], (root, cls)
            else:
                assert abs(e - orca[cls]) < 1e-9, (root, cls, e - orca[cls])
