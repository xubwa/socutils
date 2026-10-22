'''Spectral PC-NEVPT2 (union and state-specific contraction) vs the
brute-force internally contracted reference, and the ordering
E_UC <= E_PC <= E_SC per class.'''
import numpy
import pytest
from scipy.stats import unitary_group

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.pc import spectral
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.reference.pc_bruteforce import BruteForcePCNEVPT2
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.tests.pipeline import Setup
from socutils.zmr.tests.test_pyscf_compare import _sc_nevpt2_by_class
from socutils.zmr.uc import engine
from socutils.zmr.uc.nevpt2 import UCNEVPT2

TOL = 1e-10


def _compare(eris, nelec, nstate, contraction, e0):
    pc = PCNEVPT2(eris, nelec, nstate=nstate, e0=e0, contraction=contraction)
    pc.kernel()
    s = Setup(eris, nelec, civecs=pc.civecs)
    bf = BruteForcePCNEVPT2(s.eris, s.eps_core, s.eps_virt, nelec, pc.civecs,
                            e0=e0, contraction=contraction)
    bf.kernel()
    assert bf.rhs_outside < 1e-10
    for name in bf.h2_by_class:
        assert abs(pc.h2_by_class[name] - bf.h2_by_class[name]).max() < TOL, name
    return pc


@pytest.mark.parametrize('nstate, contraction, e0', [
    (1, 'union', 'state'), (3, 'union', 'state'), (3, 'union', 'average'),
    (3, 'state', 'state'), (3, 'state', 'average')])
def test_matches_bruteforce_model(nstate, contraction, e0):
    eris = model_eris(2, 6, 2, seed=3)
    pc = _compare(eris, 3, nstate, contraction, e0)
    assert abs(pc.h2 - pc.h2.conj().T).max() < 1e-12


def test_matches_bruteforce_h2o():
    mc = systems.h2o_casci(3, 4)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    _compare(eris, nelec, 1, 'union', 'state')


def test_contraction_matters():
    '''With 6 active spinors and 3 electrons the PC space of ccav, cavv is a
    proper subspace of the N+-1 sectors: PC lies strictly above UC.'''
    eris = model_eris(2, 6, 2, seed=3)
    uc = UCNEVPT2(eris, 3)
    uc.kernel()
    pc = PCNEVPT2(eris, 3)
    pc.kernel()
    for name in ('ccav', 'cavv'):
        assert pc.h2_by_class[name][0, 0].real > uc.h2_by_class[name][0, 0].real + 1e-8
    for name in uc.h2_by_class:
        assert pc.h2_by_class[name][0, 0].real >= uc.h2_by_class[name][0, 0].real - 1e-12


@pytest.fixture(scope='module')
def h2o_631g_all():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    sc, _ = _sc_nevpt2_by_class(mc)
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=0)
    uc = UCNEVPT2(eris, nelec)
    uc.kernel()
    pc = PCNEVPT2(eris, nelec)
    pc.kernel()
    return sc, uc, pc


def test_uc_pc_sc_ordering(h2o_631g_all):
    '''SC space within PC space within UC space, same H0, classes decoupled:
    E_UC <= E_PC <= E_SC for every class; all equal for ccvv.'''
    sc, uc, pc = h2o_631g_all
    for name, e_sc in sc.items():
        e_uc = uc.h2_by_class[name][0, 0].real
        e_pc = pc.h2_by_class[name][0, 0].real
        assert e_uc <= e_pc + 1e-10, name
        assert e_pc <= e_sc + 1e-10, name
    assert abs(pc.h2_by_class['ccvv'][0, 0].real - sc['ccvv']) < 1e-10
    assert pc.h2_by_class['cava'][0, 0].real > uc.h2_by_class['cava'][0, 0].real + 1e-7


def test_single_state_contractions_coincide():
    eris = model_eris(2, 6, 2, seed=4)
    a = PCNEVPT2(eris, 3, contraction='union')
    b = PCNEVPT2(eris, 3, contraction='state')
    a.kernel()
    b.kernel()
    assert abs(a.heff - b.heff).max() < 1e-12


def test_union_is_rotation_invariant_state_is_not():
    eris = model_eris(2, 6, 2, seed=5)
    for contraction, invariant in [('union', True), ('state', False)]:
        ref = PCNEVPT2(eris, 3, nstate=3, e0='average', contraction=contraction)
        ref.kernel()
        u = unitary_group.rvs(3, random_state=2)
        rot = PCNEVPT2(eris, 3, civecs=ref.civecs @ u, e0='average', contraction=contraction)
        rot.kernel()
        diff = abs(rot.e_tot - ref.e_tot).max()
        assert (diff < 1e-10) == invariant, (contraction, diff)


def test_union_space_contains_state_space():
    '''With a common E0 the union first-order space of each ket contains the
    state-specific one, so the diagonal second-order energies are lower.'''
    eris = model_eris(2, 6, 2, seed=6)
    u = PCNEVPT2(eris, 3, nstate=3, e0='average', contraction='union')
    s = PCNEVPT2(eris, 3, nstate=3, e0='average', contraction='state')
    u.kernel()
    s.kernel()
    for name in u.h2_by_class:
        du = u.h2_by_class[name].diagonal().real
        ds = s.h2_by_class[name].diagonal().real
        assert numpy.all(du <= ds + 1e-12), name


@pytest.mark.parametrize('contraction, e0', [('union', 'average'), ('state', 'average'),
                                             ('state', 'state')])
def test_laplace_converges_to_exact_pc(contraction, e0):
    eris = model_eris(2, 6, 2, seed=7)
    kw = dict(nstate=3, e0=e0, contraction=contraction)
    ex = PCNEVPT2(eris, 3, **kw)
    ex.kernel()
    lp = PCNEVPT2(eris, 3, method='laplace', nquad=48, **kw)
    lp.kernel()
    assert abs(lp.heff - ex.heff).max() < 1e-11


def test_lindep_stability():
    '''The metric of the contracted functions has round-off eigenvalues
    (<= 1e-14) and physical ones (>= 2e-9 here: nearly empty active orbitals
    with occupation 2.5e-4).  Any threshold inside the gap gives the same
    energy; a threshold above the physical eigenvalues removes functions.'''
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    e = {}
    for lindep in (1e-13, 1e-11, 1e-10, 1e-6):
        pc = PCNEVPT2(eris, nelec, lindep=lindep)
        pc.kernel()
        e[lindep] = pc.h2_by_class['cava'][0, 0].real
    assert abs(e[1e-13] - e[1e-11]) < 1e-12
    assert abs(e[1e-13] - e[1e-10]) < 1e-12
    assert e[1e-6] > e[1e-10] + 1e-7


def test_metric_matrices():
    '''S = T^+ T and K = T^+ lam T are the metric and H_act matrix of the
    contracted functions; K is Hermitian and S positive semidefinite.'''
    eris = model_eris(2, 6, 2, seed=8)
    from socutils.zmr.active.sectors import ActiveSpace
    from socutils.zmr.uc import classes
    act = ActiveSpace.from_eris(eris)
    _, c = act.solve(3, 2)
    for pclass in classes.ALL_CLASSES:
        t, sector = engine.transition_amplitudes(pclass, act, c, 3)
        if t is None:
            continue
        s, k = spectral.metric_matrices(t, sector.lam)
        nst, nx = s.shape[0], s.shape[2]
        smat = s.transpose(0, 2, 1, 3).reshape(nst * nx, nst * nx)
        kmat = k.transpose(0, 2, 1, 3).reshape(nst * nx, nst * nx)
        assert abs(smat - smat.conj().T).max() < 1e-12
        assert abs(kmat - kmat.conj().T).max() < 1e-12
        assert numpy.linalg.eigvalsh(smat).min() > -1e-12


@pytest.mark.parametrize('e0', ['state', 'average'])
def test_gram_routes_agree(e0):
    '''The two forms of the sum over the first-order space and the external
    labels (rows of u, or the external labels first) give the same matrices.'''
    from socutils.zmr.uc import classes
    eris = model_eris(2, 6, 2, seed=9)
    pc = PCNEVPT2(eris, 3, nstate=3, e0=e0)
    pc.kernel()
    ctx = classes.Context(pc.quasi_canonical.eris, pc.quasi_canonical.eps_core,
                          pc.quasi_canonical.eps_virt,
                          pc.quasi_canonical.eris.generalized_fock(numpy.zeros((6, 6))))
    act = pc.active_space
    for name in ('ccav', 'cavv', 'cava'):
        pclass = classes.CLASS_BY_NAME[name]
        t, sector = engine.transition_amplitudes(pclass, act, pc.civecs, 3)
        for delta, v in engine.class_blocks(pclass, ctx)[:2]:
            rows = engine.exact_block_t(t, sector.lam, delta, v, pc.shifts, gram='rows')
            w = engine.exact_block_t(t, sector.lam, delta, v, pc.shifts, gram='w')
            assert abs(rows - w).max() < 1e-12 * max(1.0, abs(rows).max()), name
            one = engine.exact_block_t(t, sector.lam, delta, v, pc.shifts, col=1, gram='w')
            assert abs(one[:, 1] - rows[:, 1]).max() < 1e-12 * max(1.0, abs(rows).max())
    # many states and few operators: the external labels first
    assert engine.gram_route(91, 102, 14, numpy.zeros(91)) == 'w'
    assert engine.gram_route(3, 102, 197, numpy.zeros(3)) == 'rows'
