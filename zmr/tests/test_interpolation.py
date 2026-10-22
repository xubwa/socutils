'''Chebyshev interpolation of the resolvent in the energy argument
(zmr.uc.engine.chebyshev_weights and the contractions that use it) against
the same contractions with exact denominators.

For g(d) = 1 / (d + delta) and the n roots of T_n on the interval of the
needed arguments (centre mid, half width half) the interpolation error is
known in closed form,

    (g - p_n)(d) / g(d) = T_n(x) / T_n(-z_delta),   x = (d - mid) / half,
    z_delta = (delta + mid) / half > 1,

so the relative error of every term of an external sum is at most
1 / T_n(z) with z of the nearest pole.  The tests check this formula, the
choice of n, the cases in which the interpolation must be refused, and that
whole calculations do not depend on the switch engine.INTERPOLATE.'''
import numpy
import pytest
from pyscf import gto, mcscf, scf

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.lib import kernels
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.tests._helpers import model_eris
from socutils.zmr.tests.test_cholesky import factorize
from socutils.zmr.uc import engine
from socutils.zmr.uc.nevpt2 import UCNEVPT2


def _interpolate(d, delta, **settings):
    '''Interpolated 1 / (d + delta) for arguments d (nk, nj) and all delta.'''
    old = {k: getattr(engine, k) for k in settings}
    for k, x in settings.items():
        setattr(engine, k, x)
    try:
        cheb = engine.chebyshev_weights(d, delta.min())
    finally:
        for k, x in old.items():
            setattr(engine, k, x)
    if cheb is None:
        return None, 0
    nodes, weights = cheb
    g = 1.0 / (nodes[:, None] + delta[None, :])                    # (node, e)
    return numpy.einsum('mkj,me->kje', weights, g), len(nodes)


def _arguments(seed, width, shape=(40, 7)):
    rng = numpy.random.default_rng(seed)
    d = rng.uniform(0.0, width, shape)
    d[0, 0], d[-1, -1] = 0.0, width
    return d


@pytest.mark.parametrize('gap', [1e-2, 3e-2, 0.1, 0.5, 2.0, 30.0])
@pytest.mark.parametrize('width', [1e-3, 0.3, 3.0])
def test_resolvent_is_reproduced_to_round_off(gap, width):
    '''Poles from next to the interval (smallest denominator = gap) to far
    away, intervals from a multiplet to several Eh.'''
    d = _arguments(1, width)
    rng = numpy.random.default_rng(2)
    delta = numpy.concatenate([[gap], gap + rng.uniform(0.0, 100.0, 300)])
    p, n = _interpolate(d, delta)
    # about 20 sqrt(width / gap) nodes are needed: refused below gap / width ~ 0.043
    if gap / width < 0.04:
        assert p is None
        return
    assert p is not None and n <= engine.INTERP_MAX_NODES
    exact = 1.0 / (d[:, :, None] + delta[None, None, :])
    assert abs(p / exact - 1).max() < 2e-13


@pytest.mark.parametrize('digits', [5.0, 10.0, 20.0, 30.0])
def test_error_follows_the_chebyshev_formula(digits):
    '''Fewer nodes than the default: the error is the predicted one, largest
    for the nearest pole and decaying geometrically with the node count.'''
    gap, width = 0.05, 1.0
    d = _arguments(3, width)
    delta = gap + numpy.linspace(0.0, 5.0, 60)
    p, n = _interpolate(d, delta, INTERP_DIGITS=digits)
    exact = 1.0 / (d[:, :, None] + delta[None, None, :])
    err = abs(p / exact - 1)
    mid, half = 0.5 * width, 0.5 * width
    z = (delta + mid) / half
    tn_pole = numpy.cosh(n * numpy.arccosh(z))
    x = (d - mid) / half
    tn = numpy.cos(n * numpy.arccos(numpy.clip(x, -1, 1)))
    predicted = abs(tn)[:, :, None] / tn_pole[None, None, :]
    assert abs(err - predicted).max() < 1e-12 + 1e-6 * predicted.max()
    # the bound used to choose n
    assert err.max() <= 2.0 * numpy.exp(-digits)
    assert err[:, :, 0].max() == err.max()


def test_node_count_grows_as_the_pole_approaches():
    d = _arguments(4, 1.0)
    counts = []
    for gap in (10.0, 1.0, 0.1, 0.05):
        counts.append(_interpolate(d, numpy.array([gap, gap + 1.0]))[1])
    assert counts == sorted(counts) and counts[0] < 20 < counts[-1]


def test_refused_when_not_safe():
    d = _arguments(5, 1.0)
    # a pole inside or at the edge of the interval (intruder)
    assert engine.chebyshev_weights(d, -0.3) is None
    assert engine.chebyshev_weights(d, 0.0) is None
    # a pole so close that more than INTERP_MAX_NODES nodes would be needed
    assert engine.chebyshev_weights(d, 1e-4) is None
    assert engine.chebyshev_weights(d, 0.1) is not None


def test_degenerate_interval_and_arguments_on_nodes():
    d = numpy.full((5, 3), 0.7)
    nodes, weights = engine.chebyshev_weights(d, 0.1)
    assert len(nodes) == 1 and abs(nodes[0] - 0.7) < 1e-15 and (weights == 1).all()
    # arguments that coincide with nodes
    d = _arguments(6, 2.0)
    nodes, _ = engine.chebyshev_weights(d, 0.5)
    d[3, 2], d[10, 1] = nodes[0], nodes[len(nodes) // 2]
    nodes2, weights = engine.chebyshev_weights(d, 0.5)
    assert numpy.array_equal(nodes, nodes2)
    assert numpy.isfinite(weights).all()
    g = numpy.einsum('mkj,m->kj', weights, 1.0 / (nodes + 0.5))
    assert abs(g * (d + 0.5) - 1).max() < 1e-13


def _block(seed, nstate=5, nk=9, nx=6, ne=200, gap=0.2, real=False):
    rng = numpy.random.default_rng(seed)

    def rand(*shape):
        x = rng.normal(size=shape)
        return x if real else x + 1j * rng.normal(size=shape)
    t, v = rand(nstate, nk, nx), rand(ne, nx)
    lam = numpy.sort(rng.uniform(0.0, 1.5, nk))
    shifts = numpy.sort(rng.uniform(-0.3, 0.0, nstate))
    delta = gap + rng.uniform(0.0, 40.0, ne)
    return t, lam, delta, v, shifts


@pytest.mark.parametrize('real', [False, True])
@pytest.mark.parametrize('gap', [0.05, 0.2, 5.0])
@pytest.mark.parametrize('col', [None, 2])
def test_block_contraction(real, gap, col):
    t, lam, delta, v, shifts = _block(7, gap=gap, real=real)
    a = engine.interpolated_block_t(t, lam, delta, v, shifts, col)
    assert a is not None
    for gram in ('rows', 'w'):
        ref = engine.exact_block_t(t, lam, delta, v, shifts, col, gram=gram)
        assert abs(a - ref).max() < 1e-12 * abs(ref).max()


def test_block_contraction_refused_for_an_intruder():
    t, lam, delta, v, shifts = _block(8)
    delta[3] = -0.5 * (lam[0] + lam[-1])
    assert engine.interpolated_block_t(t, lam, delta, v, shifts) is None


def test_single_operator_class(monkeypatch):
    t, lam, delta, v, shifts = _block(9, nx=1, ne=500, gap=0.3)
    blocks = [(delta[:200], v[:200]), (delta[200:], v[200:])]
    a = engine.single_operator_contract(blocks, t, lam, shifts)
    monkeypatch.setattr(engine, 'INTERPOLATE', False)
    ref = engine.single_operator_contract(blocks, t, lam, shifts)
    f = numpy.zeros((len(lam), len(shifts)))
    kernels.shifted_sums(abs(v[:, 0])**2, delta, lam, shifts,
                         numpy.ones(f.shape, dtype=bool), f)
    assert abs(a - ref).max() < 1e-12 * abs(ref).max()
    tk = t[:, :, 0]
    direct = -numpy.einsum('ik,jk,kj->ij', tk.conj(), tk, f)
    assert abs(ref - direct).max() < 1e-12 * abs(ref).max()


def _on_off(monkeypatch, make):
    monkeypatch.setattr(engine, 'INTERPOLATE', True)
    used = []
    weights = engine.chebyshev_weights
    monkeypatch.setattr(engine, 'chebyshev_weights',
                        lambda d, dmin: used.append(weights(d, dmin)) or used[-1])
    on = make()
    on.kernel()
    assert sum(x is not None and len(x[0]) > 1 for x in used) >= 2
    monkeypatch.setattr(engine, 'INTERPOLATE', False)
    off = make()
    off.kernel()
    for name, a in off.h2_by_class.items():
        assert abs(on.h2_by_class[name] - a).max() < 1e-11, name
    assert abs(on.heff - off.heff).max() < 1e-11
    return on


@pytest.mark.parametrize('method, options', [
    (PCNEVPT2, dict(contraction='union')), (PCNEVPT2, dict(contraction='state')),
    (PCNEVPT2, dict(amplitudes='rdm')), (UCNEVPT2, {})])
def test_model_hamiltonian_on_and_off(monkeypatch, method, options):
    eris = model_eris(3, 6, 5, seed=11)
    civecs = ActiveSpace.from_eris(eris).solve(3, 4)[1]
    _on_off(monkeypatch, lambda: _quiet(method(eris, 3, civecs=civecs, **options)))


def _quiet(pt):
    pt.verbose = 0
    return pt


@pytest.fixture(scope='module')
def water_casscf():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.conv_tol = 1e-11
    mc.kernel()
    return mc, factorize(mol)


@pytest.mark.parametrize('factorized', [False, True])
@pytest.mark.parametrize('method', [PCNEVPT2, UCNEVPT2])
def test_molecule_on_and_off(monkeypatch, water_casscf, factorized, method):
    '''Four model states spread over 0.4 Eh; with three-index integrals the
    external sums come from the task kernels at the interpolation nodes.'''
    mc, cderi = water_casscf
    eris, nelec = pyscf_nr.from_casci(mc, nfrozen=1, cderi=cderi if factorized else None)
    civecs = ActiveSpace.from_eris(eris).solve(nelec, 4)[1]
    _on_off(monkeypatch, lambda: _quiet(method(eris, nelec, civecs=civecs)))


def test_few_distinct_arguments_are_used_directly():
    '''Fewer distinct arguments than interpolation nodes (ccvv with a few
    states): the sums are evaluated at the arguments themselves.'''
    lam = numpy.array([0.0, 0.0, 0.01, 0.4, 0.9])
    shifts = numpy.array([0.0, 0.0, 0.01])
    d = lam[:, None] - shifts[None, :]
    used = numpy.zeros(d.shape, dtype=bool)
    used[[0, 1, 0, 1, 2], [0, 0, 1, 1, 2]] = True          # k in the level of J
    nodes, weights = engine.resolvent_nodes(d, 1.0, used)
    assert len(nodes) == 1 and abs(nodes[0]) < 1e-15
    assert numpy.array_equal(weights[0] != 0, used)
    # all arguments in use: 9 distinct values, fewer than the 14 nodes
    nodes, weights = engine.resolvent_nodes(d, 1.0)
    assert len(nodes) == len(numpy.unique(numpy.round(d, 12)))
    g = numpy.einsum('mkj,m->kj', weights, 1.0 / (nodes + 1.0))
    assert abs(g * (d + 1.0) - 1).max() < 1e-15
    # many distinct arguments: interpolation
    d = _arguments(12, 1.0)
    nodes, _ = engine.resolvent_nodes(d, 1.0)
    assert len(nodes) < d.size and len(nodes) == len(engine.chebyshev_weights(d, 1.0)[0])
