'''zmr.blockstore: the same operations on dense blocks and on blocks in the DPD layout
(C2h*, Coov*) give the same operators.'''
import os
import sys

import numpy
import pytest

from socutils.zmr import blockstore as bs, dpd
from socutils.zmr.dsrg import commutator as cm

sys.path.insert(0, os.path.dirname(__file__))
from test_dpd import IRREPS, allowed  # noqa: E402


def make_dims(group, rng):
    irs = IRREPS[group]
    sym = {}
    for x in 'cav':
        k = rng.integers(1, len(irs) + 1)
        chosen = sorted(rng.choice(irs, size=k, replace=False))
        sym[x] = numpy.repeat(chosen, rng.integers(1, 4, size=k))
    return cm.Dims({x: len(v) for x, v in sym.items()}, sym=(group, sym))


def legs_of(dims, lab):
    return [dims.legs[x] for x in lab]


@pytest.mark.parametrize('group', ['C2h', 'Coov'])
def test_layouts_agree(group):
    rng = numpy.random.default_rng(21)
    dims = make_dims(group, rng)
    sym, dense = bs.Layout(dims), bs.Layout(dims, use_dpd=False)
    assert sym.use_dpd and not dense.use_dpd
    lab = 'cavv'
    shape = dense.shape(lab)
    mask = allowed(legs_of(dims, lab), bs.TWO_BODY, group)
    a = numpy.where(mask, rng.normal(size=shape) + 1j * rng.normal(size=shape), 0)
    b = numpy.where(mask, rng.normal(size=shape) + 1j * rng.normal(size=shape), 0)
    ts, td = sym.block(lab, a, alpha=2.0), dense.block(lab, a, alpha=2.0)
    assert isinstance(ts, dpd.Tensor) and isinstance(td, numpy.ndarray)
    assert abs(cm.materialize(ts) - td).max() < 1e-14
    # add in all forms of get: array, (array, sign, conj), a View of a Tensor
    for x in (b, (b, -1.0), (b, 0.5, True), (dpd.View(sym.block(lab, b)), 1.0, True)):
        bs.add_block(ts, x, alpha=0.3)
        bs.add_block(td, x if not isinstance(x[0] if isinstance(x, tuple) else x, dpd.View)
                     else (cm.materialize(x[0]), x[1], x[2]), alpha=0.3)
    assert abs(cm.materialize(ts) - td).max() < 1e-13
    assert abs(bs.vdot(ts, ts) - bs.vdot(td, td)) < 1e-10
    # a transposed placement of the DPD block into a dense block
    out = numpy.zeros(dense.shape('cavv'), dtype=complex)
    bs.add_block(out, (dpd.View(ts, (0, 1, 3, 2)), -1.0))
    assert abs(out + cm.materialize(ts).transpose(0, 1, 3, 2)).max() < 1e-14
    # denominators, also with the sign of a de-excitation, and elementwise products
    eps = {x: numpy.sort(rng.normal(size=dims[x])) for x in 'cav'}
    eps['c'] -= 3.0
    eps['v'] += 3.0
    for flip in (False, True):
        rs, es = sym.denominators(lab, eps, 0.5, flip=flip)
        rd, ed = dense.denominators(lab, eps, 0.5, flip=flip)
        assert abs(cm.materialize(rs) - numpy.where(mask, rd, 0)).max() < 1e-14
        assert abs(cm.materialize(es) - numpy.where(mask, ed, 0)).max() < 1e-14
    rd, ed = dense.denominators(lab, eps, 0.5)
    rs = sym.denominators(lab, eps, 0.5)[0]
    d = (eps['c'][:, None, None, None] + eps['a'][None, :, None, None]
         - eps['v'][None, None, :, None] - eps['v'][None, None, None, :])
    assert abs(rd - (1 - numpy.exp(-0.5 * d * d)) / d).max() < 1e-12
    assert abs(dense.denominators(lab, eps, 0.5, flip=True)[0] + rd).max() < 1e-14
    assert sym.denominators(lab, eps, 0.5)[0] is sym.denominators(lab, eps, 0.5)[0]
    ps = sym.block(lab, ts)
    bs.data(ps)[...] *= bs.data(rs)
    pd = td * rd
    assert abs(cm.materialize(ps) - pd).max() < 1e-13


def test_dense_when_no_sectors():
    dims = cm.Dims({'c': 2, 'a': 2, 'v': 3})
    lay = bs.Layout(dims)
    assert not lay.use_dpd
    assert isinstance(lay.zeros('ccvv'), numpy.ndarray)
