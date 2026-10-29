'''Block pairs applied as the adjoint of their reverse pair (Sigma.share_adjoint)
against the compiled pairs: the same sigma for every contraction route.'''
import numpy
import pytest

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.mrci import MRCI
from socutils.zmr.mrci import sigma as sg
from socutils.zmr.tests._helpers import model_eris

CASES = [((3, 4, 4), 2, 2), ((2, 3, 3), 1, 1), ((3, 4, 2), 3, 2)]
# (pack_limit, w_limit / w_max, x_limit): defaults; everything packed; every large term in two
# steps; two steps with sliced intermediates
ROUTES = [(5e6, (2e7, 5e8), 2e8), (0, (2e7, 5e8), 2e8), (0, (0, 0), 2e8), (0, (0, 0), 40)]


def _sigma(eris, nelec, c, contraction, internal, share, route, x):
    saved = (sg.Sigma.share_adjoint, sg.Sigma.pack_limit, sg.Sigma.w_limit, sg.Sigma.w_max,
             sg.Sigma.x_limit)
    pack, (wl, wm), xl = route
    sg.Sigma.share_adjoint, sg.Sigma.pack_limit = share, pack
    sg.Sigma.w_limit, sg.Sigma.w_max, sg.Sigma.x_limit = wl, wm, xl
    try:
        ci = MRCI(eris, nelec, civecs=c, contraction=contraction, internal=internal)
        ci.verbose = 0
        ci.build()
        x = x if x is not None else (
            numpy.random.default_rng(2).normal(size=(2, ci.basis.dim))
            + 1j * numpy.random.default_rng(3).normal(size=(2, ci.basis.dim)))
        return x, ci.sigma.matvec(x), len(ci.sigma._adjoint), len(ci.sigma._compiled)
    finally:
        (sg.Sigma.share_adjoint, sg.Sigma.pack_limit, sg.Sigma.w_limit, sg.Sigma.w_max,
         sg.Sigma.x_limit) = saved


@pytest.mark.parametrize('sizes, nelec, nstate', CASES)
@pytest.mark.parametrize('contraction, internal', [('fic', 'cas'), ('fic', 'states'), ('none', 'cas')])
@pytest.mark.parametrize('route', ROUTES)
def test_adjoint_pairs(sizes, nelec, nstate, contraction, internal, route):
    eris = model_eris(*sizes, seed=7)
    c = ActiveSpace.from_eris(eris).solve(nelec, nstate)[1]
    x, ref, nadj0, _ = _sigma(eris, nelec, c, contraction, internal, False, route, None)
    _, out, nadj, ncomp = _sigma(eris, nelec, c, contraction, internal, True, route, x)
    assert nadj0 == 0 and nadj > 0
    assert abs(out - ref).max() < 1e-11 * max(1.0, abs(ref).max())
