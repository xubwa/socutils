#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# CCSD(T) Lambda equations for the vvvv-free two-component CCSD.
#
# Ported from pyscf.cc.gccsd_t_lambda: the (T) correction adds constant terms
# l1_t, l2_t (built once from the connected/disconnected triples t3c, t3d) to
# the CCSD Lambda residual.  Like (T) itself, this needs only ovvv/ooov/oovv
# (no vvvv), so it sits directly on top of socutils.cc.lambda_zccsd.
#
# Note: the full triples t3c/t3d (O(o^3 v^3)) are formed in core, as in pyscf's
# gccsd_t_lambda.
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_lambda

from socutils.cc import lambda_zccsd

einsum = lib.einsum


def make_intermediates(mycc, t1, t2, eris):
    imds = lambda_zccsd.make_intermediates(mycc, t1, t2, eris)  # sets eris.ovvv

    nocc, nvir = t1.shape
    ovvv = np.asarray(eris.ovvv)
    ooov = np.asarray(eris.ooov)
    oovv = np.asarray(eris.oovv)
    bcei = ovvv.conj().transpose(3, 2, 1, 0)
    majk = ooov.conj().transpose(2, 3, 0, 1)
    bcjk = oovv.conj().transpose(2, 3, 0, 1)

    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - mo_e[nocc:]
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    t3c = (einsum('jkae,bcei->ijkabc', t2, bcei) -
           einsum('imbc,majk->ijkabc', t2, majk))
    t3c = t3c - t3c.transpose(0, 1, 2, 4, 3, 5) - t3c.transpose(0, 1, 2, 5, 4, 3)
    t3c = t3c - t3c.transpose(1, 0, 2, 3, 4, 5) - t3c.transpose(2, 1, 0, 3, 4, 5)
    t3c /= d3

    t3d = einsum('ia,bcjk->ijkabc', t1, bcjk)
    t3d += einsum('ai,jkbc->ijkabc', eris.fock[nocc:, :nocc], t2)
    t3d = t3d - t3d.transpose(0, 1, 2, 4, 3, 5) - t3d.transpose(0, 1, 2, 5, 4, 3)
    t3d = t3d - t3d.transpose(1, 0, 2, 3, 4, 5) - t3d.transpose(2, 1, 0, 3, 4, 5)
    t3d /= d3

    l1_t = einsum('ijkabc,jkbc->ia', t3c.conj(), oovv) / eia
    imds.l1_t = l1_t * .25

    m3 = t3c * 2 + t3d
    tmp = einsum('ijkaef,kbfe->ijab', m3.conj(), ovvv) * .5
    l2_t = tmp - tmp.transpose(0, 1, 3, 2)
    tmp = einsum('imnabc,mnjc->ijab', m3.conj(), ooov) * .5
    l2_t -= tmp - tmp.transpose(1, 0, 2, 3)
    l2_t += einsum('kc,ijkabc->ijab', eris.fock[:nocc, nocc:], t3c.conj())
    imds.l2_t = l2_t / lib.direct_sum('ia+jb->ijab', eia, eia)

    return imds


def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None:
        eris = mycc.ao2mo()
    if imds is None:
        imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = lambda_zccsd.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 = l1 + imds.l1_t
    l2 = l2 + imds.l2_t
    return l1, l2


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None:
        eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)
