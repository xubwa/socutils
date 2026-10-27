'''Brute-force ic-MRCC(SD) (zmr.reference.icmrcc_bruteforce): perturbation
structure on H = H_D + V (docs/derivations/ic-mrcc.md section 3) and the
truncation of the commutator series.'''
import numpy
import pytest

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.pt4 import NEVPT4SD
from socutils.zmr.reference import icmrcc_bruteforce as icb
from socutils.zmr.tests._helpers import model_eris

CASES = [((2, 4, 3), 2), ((3, 4, 2), 3), ((2, 3, 3), 1)]


def _setup(sizes, nelec, seed=11):
    eris = model_eris(*sizes, seed=seed)
    c = ActiveSpace.from_eris(eris).solve(nelec, 1)[1][:, 0]
    er, hd_op = icb.dyall_split(eris, nelec, c)
    ic = icb.ICMRCC(er, nelec, c)
    hd = icb._matrix(hd_op, ic.strs)
    return er, c, ic, hd


@pytest.mark.parametrize('sizes, nelec', CASES)
def test_second_order_is_pc_nevpt2(sizes, nelec):
    er, c, ic, hd = _setup(sizes, nelec)
    o = ic.orders(hd, ic.h - hd)
    pc = PCNEVPT2(er, nelec, civecs=c[:, None], e0='state')
    pc.verbose = 0
    e_pc = numpy.atleast_1d(pc.kernel())[0]
    assert abs(o['e1']) < 1e-12
    assert abs(o['e0'] + o['e2'] - e_pc) < 1e-12


def test_third_order_without_contraction():
    '''If the FIC space is the whole singles-and-doubles space, E3 is
    PC-NEVPT3 (= the NEVPT3 of zmr.pt4); otherwise they differ slightly.'''
    er, c, ic, hd = _setup((1, 4, 3), 2)
    o = ic.orders(hd, ic.h - hd)
    pt = NEVPT4SD(er, 2, civecs=c[:, None], contraction='union')
    pt.verbose = 0
    pt.kernel()
    assert abs(o['e3_pc'] - (pt.e3 - pt.e2)[0]) < 1e-12
    assert abs(o['e3'] - o['e3_pc']) < 1e-12
    er, c, ic, hd = _setup((3, 4, 2), 3)
    o = ic.orders(hd, ic.h - hd)
    assert abs(o['e3'] - o['e3_pc']) > 1e-7


def test_orders_against_lambda_fit():
    er, c, ic, hd = _setup((2, 3, 3), 1)
    v = ic.h - hd
    o = ic.orders(hd, v)
    ls = numpy.array([-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2])
    es = [ic.kernel(hd + lam * v)[0] for lam in ls]
    co = numpy.polynomial.polynomial.polyfit(numpy.concatenate([[0.0], ls]),
                                             numpy.concatenate([[o['e0']], es]), 7)
    assert abs(co[2] - o['e2']) < 1e-8
    assert abs(co[3] - o['e3']) < 1e-7


@pytest.mark.parametrize('sizes, nelec', CASES)
def test_commutator_truncation(sizes, nelec):
    '''m = 4 reproduces the untruncated series; m = 2 does not.'''
    _, _, ic, _ = _setup(sizes, nelec)
    e_full = ic.kernel(m=None)[0]
    assert abs(ic.kernel(m=4)[0] - e_full) < 1e-7
    assert abs(ic.kernel(m=2)[0] - e_full) > 1e-6
