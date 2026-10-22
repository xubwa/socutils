'''Wick algebra vs brute-force matrix elements in the full determinant
space: random operator strings over core, active and virtual orbitals,
symbolic evaluation (external full contraction x active normal ordering ->
RDMs) compared with <Phi_c Psi_I| string |Phi_c Psi_J> computed directly.'''
import numpy
import pytest

from socutils.zmr.active import rdm as active_rdm
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.reference import dets
from socutils.zmr.tests._helpers import random_hamiltonian
from socutils.zmr.wick import algebra as w
from socutils.zmr.wick import numeric

NC, NA, NV = 3, 6, 3
NELEC = 3
OFFSET = {'c': 0, 'a': NC, 'v': NC + NA}
SIZE = {'c': NC, 'a': NA, 'v': NV}


@pytest.fixture(scope='module')
def model():
    h, eri = random_hamiltonian(NA, seed=51)
    act = ActiveSpace(h, eri)
    _, c = act.solve(NELEC, 3)
    c = c * numpy.exp(1j * numpy.array([0.3, -1.1, 2.0]))
    dms = {k: active_rdm.make_rdm(act, c, NELEC, k) for k in range(5)}
    strs = dets.cas_strings(NC, NA, NELEC)
    vecs = [dets.SparseVector(strs, c[:, i]) for i in range(c.shape[1])]
    return dms, vecs


def _random_string(rng, npair):
    '''A shuffled string of npair creator/annihilator pairs, each pair in
    one orbital space (so that the string can conserve the electron number of
    every space); core and virtual indices are drawn from two orbitals so that contractions occur
    often, active ones from the whole active space.'''
    ops, binding = [], {}
    for m in range(npair):
        space = 'cav'[rng.integers(3)]
        for dagger in (True, False):
            name = 'x%d%s' % (m, 'c' if dagger else 'a')
            binding[name] = int(rng.integers(min(2, SIZE[space]) if space != 'a' else SIZE[space]))
            ops.append(w.Op(w.Index(name, space), dagger))
    order = rng.permutation(len(ops))
    return [ops[i] for i in order], binding


def _brute(ops, binding, vecs):
    daggers = tuple(o.dagger for o in ops)
    orbs = [[OFFSET[o.index.space] + binding[o.index.name] for o in ops]]
    n = len(vecs)
    out = numpy.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            out[i, j] = dets.opstring_elements(daggers, orbs, vecs[i], vecs[j])[0]
    return out


@pytest.mark.parametrize('npair', [1, 2, 3, 4])
def test_random_strings(model, npair):
    dms, vecs = model
    rng = numpy.random.default_rng(100 + npair)
    nonzero = 0
    # external pairs vanish unless correctly ordered and equal, so longer
    # strings need more samples to hit non-zero elements
    for _ in range(150 if npair < 4 else 400):
        ops, binding = _random_string(rng, npair)
        free = set(binding)
        terms = w.evaluate(w.Term(ops=ops), free)
        # all indices are free here; active binding is relative to the active space
        sym = numeric.value(terms, binding, dms)
        ref = _brute(ops, binding, vecs)
        assert abs(sym - ref).max() < 1e-12, (ops, binding)
        nonzero += abs(ref).max() > 1e-8
    assert nonzero > 10


def test_collect_preserves_value(model):
    '''Canonicalization and collection of terms do not change the value.'''
    dms, vecs = model
    rng = numpy.random.default_rng(7)
    for _ in range(60):
        ops, binding = _random_string(rng, 3)
        free = set(binding)
        terms = w.evaluate(w.Term(ops=ops), free)
        if not terms:
            continue
        a = numeric.value(terms, binding, dms)
        b = numeric.value(w.collect(terms, free), binding, dms)
        assert abs(a - b).max() < 1e-12


def test_commutator_identity(model):
    '''<I| X [E, O] |J> = <I| X E O |J> - <I| X O E |J> for one- and two-body
    active strings E with summed (dummy) indices.'''
    dms, vecs = model
    rng = numpy.random.default_rng(3)
    a = lambda n: w.Index(n, 'a')
    h1 = rng.normal(size=(NA, NA)) + 1j * rng.normal(size=(NA, NA))
    g = rng.normal(size=(NA,) * 4) + 1j * rng.normal(size=(NA,) * 4)
    one = w.Term(tensors=(('h', (a('p'), a('q'))),), ops=(w.Op(a('p'), 1), w.Op(a('q'), 0)))
    two = w.Term(tensors=(('g', (a('p'), a('q'), a('r'), a('s'))),),
                 ops=(w.Op(a('p'), 1), w.Op(a('r'), 1), w.Op(a('s'), 0), w.Op(a('q'), 0)))
    x_ops = (w.Op(a('x3'), 1), w.Op(a('x2'), 0), w.Op(a('x1'), 0))
    o_ops = (w.Op(a('y1'), 1), w.Op(a('y2'), 1), w.Op(a('y3'), 0))
    free = {'x1', 'x2', 'x3', 'y1', 'y2', 'y3'}
    out = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3']
    tensors = {'h': h1, 'g': g}
    for e in (one, two):
        lhs = []
        for c in w.commutator(e, o_ops):
            lhs += w.evaluate(w.product(w.Term(ops=x_ops), c), free)
        rhs = w.evaluate(w.product(w.Term(ops=x_ops), e, w.Term(ops=o_ops)), free)
        rhs += [t.copy(coef=-t.coef) for t in
                w.evaluate(w.product(w.Term(ops=x_ops), w.Term(ops=o_ops), e), free)]
        lv = numeric.tensor(w.collect(lhs, free), out, dms, tensors, NA)
        rv = numeric.tensor(w.collect(rhs, free), out, dms, tensors, NA)
        assert abs(lv - rv).max() < 1e-11
        # the commutator lowers the RDM rank needed
        assert max(len(t.rdm[0]) for t in lhs) < max(len(t.rdm[0]) for t in rhs)
