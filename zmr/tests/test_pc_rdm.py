'''Wick-generated RDM formulas for the PC-NEVPT2 metric and zeroth-order
matrices vs the exact spectral quantities S = T^+ T, K = T^+ lam T, and the
RDM-based PC-NEVPT2 vs the spectral one.'''
import numpy
import pytest

from socutils.zmr.active import rdm as active_rdm
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.pc import rdm_terms, spectral
from socutils.zmr.reference import pc_rdm4
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.tests import systems
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.uc import classes, engine
from socutils.zmr.wick import pc_nevpt2


def test_generated_files_up_to_date():
    py_src, md_src = pc_nevpt2.generate()
    py_path, md_path, ref_path = pc_nevpt2.paths()
    assert open(py_path).read() == py_src
    assert open(md_path).read() == md_src
    assert open(ref_path).read() == pc_nevpt2.generate(explicit=True)[0]


def test_operator_blocks_match_classes():
    assert set(pc_nevpt2.OPERATORS) == set(classes.CLASS_BY_NAME)
    assert tuple(c.name for c in classes.ALL_CLASSES) == pc_nevpt2.CLASS_ORDER


@pytest.mark.parametrize('name', pc_nevpt2.CLASS_ORDER)
def test_class_strings_in_the_generated_layout(name):
    '''The strings of a class are the strings of OPERATORS (unrestricted
    indices) at the positions subset(nact); the strings left out are
    redundant.'''
    norb, nelec = 5, 2
    h, eri = model_eris(0, norb, 0, seed=1).active_hamiltonian()[:2]
    act = ActiveSpace(h, eri)
    rng = numpy.random.default_rng(2)
    c = rng.normal(size=act.dim(nelec)) + 1j * rng.normal(size=act.dim(nelec))
    pclass = classes.CLASS_BY_NAME[name]
    blocks = []
    for ops, layout in pc_nevpt2.OPERATORS[name]:
        index = classes._all(norb, len(layout))
        cols = [layout.index(n) for n, _ in ops]
        blocks.append((tuple(d for _, d in ops), index[:, cols]))
    full = act.apply_strings(c, nelec, blocks)
    subset = pclass.subset(norb)
    if subset is None:
        subset = numpy.arange(len(full))
    own = pclass.apply(act, c, nelec)
    assert abs(own - full[subset]).max() == 0
    assert len(numpy.unique(subset)) == len(own)
    if own.shape[1]:
        tol = 1e-10
        assert numpy.linalg.matrix_rank(full, tol) == numpy.linalg.matrix_rank(own, tol)


@pytest.mark.parametrize('nelec', [2, 3, 4])
def test_metric_and_k_match_spectral(nelec):
    eris = model_eris(2, 6, 2, seed=10 + nelec)
    act = ActiveSpace.from_eris(eris)
    _, c = act.solve(nelec, 3)
    c = c * numpy.exp(1j * numpy.array([0.2, 1.0, -0.7]))
    dms = {k: active_rdm.make_rdm(act, c, nelec, k) for k in range(rdm_terms.max_rank() + 1)}
    hmodel = c.conj().T @ act.make_hmat(nelec) @ c
    for pclass in classes.ALL_CLASSES:
        t, sector = engine.transition_amplitudes(pclass, act, c, nelec)
        s, kc = rdm_terms.class_matrices(
            pclass.name, dms, act.heff, act.aaaa, pclass.subset(act.norb),
            lambda name, items: rdm_terms.intermediate_terms(name, items, act, c, nelec))
        k = rdm_terms.full_k(s, kc, hmodel)
        if t is None:
            assert abs(s).max(initial=0) < 1e-12 and abs(k).max(initial=0) < 1e-12
            continue
        s_ref, k_ref = spectral.metric_matrices(t, sector.lam)
        assert abs(s - s_ref).max() < 1e-12, pclass.name
        assert abs(k - k_ref).max() < 1e-11, pclass.name


@pytest.mark.parametrize('contraction', ['union', 'state'])
@pytest.mark.parametrize('e0', ['state', 'average'])
def test_rdm_path_equals_spectral(contraction, e0):
    eris = model_eris(2, 6, 2, seed=3)
    a = PCNEVPT2(eris, 3, nstate=3, e0=e0, contraction=contraction)
    b = PCNEVPT2(eris, 3, nstate=3, e0=e0, contraction=contraction, amplitudes='rdm')
    a.kernel()
    b.kernel()
    assert abs(a.heff - b.heff).max() < 1e-12
    for name in a.h2_by_class:
        assert abs(a.h2_by_class[name] - b.h2_by_class[name]).max() < 1e-12


def test_rdm_path_h2o():
    mc = systems.h2o_casci(4, 4, basis='6-31g')
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1)
    a = PCNEVPT2(eris, nelec)
    b = PCNEVPT2(eris, nelec, amplitudes='rdm')
    assert abs(a.kernel() - b.kernel()).max() < 1e-10


def test_rdm_path_requires_invariant_model_space():
    eris = model_eris(2, 6, 2, seed=3)
    rng = numpy.random.default_rng(0)
    act = ActiveSpace.from_eris(eris)
    dim = act.dim(3)
    c = numpy.linalg.qr(rng.normal(size=(dim, 2)) + 1j * rng.normal(size=(dim, 2)))[0]
    pc = PCNEVPT2(eris, 3, civecs=c, amplitudes='rdm')
    with pytest.raises(ValueError):
        pc.kernel()


def test_no_4rdm_in_the_production_terms():
    assert rdm_terms.max_rank() == 3
    assert pc_rdm4.rdm_terms.max_rank(pc_rdm4.TERMS) == 4
    blocks = [key for key, d in rdm_terms.TERMS.items() if d['I']]
    assert sorted(blocks) == [('aaav', 1, 1), ('caaa', 1, 1)]


@pytest.mark.parametrize('nelec', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('name', ['caaa', 'aaav'])
def test_intermediate_states_equal_explicit_4rdm(name, nelec):
    '''K of the semi-internal classes through the intermediate states
    [H, y] |J> against the formulas with the explicit 4-RDM: unrestricted
    indices, several complex states, every element.'''
    eris = model_eris(2, 6, 2, seed=30 + nelec)
    act = ActiveSpace.from_eris(eris)
    _, c = act.solve(nelec, 3)
    c = c * numpy.exp(1j * numpy.array([0.3, -1.1, 0.6]))
    dms = {k: active_rdm.make_rdm(act, c, nelec, k) for k in range(4)}
    s, kc = rdm_terms.class_matrices(
        name, dms, act.heff, act.aaaa,
        intermediate=lambda cls, items: rdm_terms.intermediate_terms(cls, items, act, c, nelec))
    s_ref, kc_ref = pc_rdm4.class_matrices(name, act, c, nelec)
    assert abs(kc_ref).max() > 1e-3
    assert abs(s - s_ref).max() < 1e-12
    assert abs(kc - kc_ref).max() < 1e-11


def test_intermediate_states_do_not_need_eigenstates():
    '''The commutator matrix is an operator identity: arbitrary vectors.'''
    eris = model_eris(2, 6, 2, seed=41)
    act = ActiveSpace.from_eris(eris)
    rng = numpy.random.default_rng(3)
    c = rng.normal(size=(act.dim(3), 2)) + 1j * rng.normal(size=(act.dim(3), 2))
    c = numpy.linalg.qr(c)[0]
    dms = {k: active_rdm.make_rdm(act, c, 3, k) for k in range(4)}
    for name in ('caaa', 'aaav'):
        _, kc = rdm_terms.class_matrices(
            name, dms, act.heff, act.aaaa,
            intermediate=lambda cls, items: rdm_terms.intermediate_terms(cls, items, act, c, 3))
        _, kc_ref = pc_rdm4.class_matrices(name, act, c, 3)
        assert abs(kc - kc_ref).max() < 1e-11


# ----------------------------------------------------------------------
# matrix-free sectors

def _matrix_free(act):
    act.dense_limit = 0
    return act


@pytest.mark.parametrize('nelec', [1, 3, 5])
def test_matrix_free_hamiltonian(nelec):
    eris = model_eris(2, 6, 2, seed=50 + nelec)
    dense = ActiveSpace.from_eris(eris)
    free = _matrix_free(ActiveSpace.from_eris(eris))
    rng = numpy.random.default_rng(nelec)
    q = rng.normal(size=(dense.dim(nelec), 4)) + 1j * rng.normal(size=(dense.dim(nelec), 4))
    assert abs(free.apply_h(nelec, q) - dense.apply_h(nelec, q)).max() < 1e-12
    with pytest.raises(NotImplementedError):
        free.hmat(nelec)
    if dense.dim(nelec) > 4:
        e_ref, _ = dense.solve(nelec, 2)
        e, c = free.solve(nelec, 2)
        assert abs(e - e_ref).max() < 1e-10
        assert abs(free.apply_h(nelec, c) - c * (e - free.e_core)).max() < 1e-8


@pytest.mark.parametrize('contraction', ['union', 'state'])
def test_rdm_route_runs_matrix_free(contraction):
    '''No sector Hamiltonian is built: model states from the Lanczos solver,
    intermediate states from H applied to vectors.'''
    from socutils.zmr.pc import nevpt2 as pc_module
    eris = model_eris(2, 6, 2, seed=61)
    _, c = ActiveSpace.from_eris(eris).solve(3, 2)
    ref = PCNEVPT2(eris, 3, civecs=c, contraction=contraction)
    ref.kernel()

    class MatrixFree(PCNEVPT2):
        def make_active_space(self, eris):
            return _matrix_free(ActiveSpace.from_eris(eris))

    pt = MatrixFree(eris, 3, civecs=c, contraction=contraction, amplitudes='rdm')
    pt.kernel()
    assert not pt.active_space._hmat
    assert abs(pt.heff - ref.heff).max() < 1e-9
