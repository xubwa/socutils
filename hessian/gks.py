#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Analytic nuclear Hessian of GKS for a Kramers closed shell, with an X2C
spin-orbit hcore.

Nuclear displacements are time-even, so a Kramers-symmetric reference stays
Kramers symmetric to first order: the magnetization vanishes in the reference
and in its response, and the collinear functional of (Re D_aa, Re D_bb) is
exact here (col = ncol = mcol).  The exchange-correlation pieces are those of
pyscf.hessian.uks evaluated on that density -- Re D_aa and Re D_bb are real,
symmetric and positive semidefinite and are passed as pseudo-orbitals
V sqrt(n) of their eigendecompositions.  The Coulomb / exact-exchange pieces
are those of socutils.hessian.ghf.

Like pyscf's UKS Hessian, the response of the integration grid is not
included; its translational part is restored by rebuilding the diagonal
blocks from the sum rule sum_B H_AB = 0 (Hessian.translation_sum_rule).
'''

import types
import numpy
from pyscf import lib, dft
from pyscf.lib import logger
from pyscf.hessian import uks as uks_hess
from socutils.hessian import ghf as ghf_hess
import socutils.grad.gks_grad  # noqa: F401  (GKS.Gradients)


def _pseudo_mo(dm, thresh=1e-14):
    '''dm = C C^T for real symmetric PSD dm.'''
    w, v = numpy.linalg.eigh(dm)
    keep = w > thresh
    c = v[:, keep] * numpy.sqrt(w[keep])
    return c, numpy.ones(c.shape[1])


def _uks_proxy(hessobj, dm0):
    '''A pyscf UKS stand-in carrying the collinear density of the GKS dm0, for
    the pyscf.hessian.uks exchange-correlation helpers.'''
    mf = hessobj.base
    mol = hessobj.mol
    nao = mol.nao_nr()
    ca, occa = _pseudo_mo(dm0[:nao, :nao].real)
    cb, occb = _pseudo_mo(dm0[nao:, nao:].real)
    uks = dft.UKS(mol, xc=mf.xc)
    uks.grids = mf.grids
    uks._numint = mf._numint._to_numint1c()
    proxy = types.SimpleNamespace(mol=mol, base=uks, grids=hessobj.grids)
    mo_coeff = (ca, cb)
    mo_occ = (occa, occb)
    return proxy, mo_coeff, mo_occ, uks._numint



def partial_hess_veff(hessobj, dm0, atmlst):
    mf = hessobj.base
    mol = hessobj.mol
    ni = mf._numint
    if mf.do_nlc():
        raise NotImplementedError('GKS Hessian for NLC functional')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    de2 = ghf_hess.partial_hess_jk(mol, dm0, atmlst, with_k=hybrid, k_scale=hyb)
    if hybrid and omega != 0:
        with mol.with_range_coulomb(omega):
            de2 += ghf_hess.partial_hess_jk(mol, dm0, atmlst, with_k=True,
                                            k_scale=alpha - hyb) \
                 - ghf_hess.partial_hess_jk(mol, dm0, atmlst, with_k=False)

    proxy, mo_coeff, mo_occ, _ = _uks_proxy(hessobj, dm0)
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    nao = mol.nao_nr()
    dm0a = dm0[:nao, :nao].real
    dm0b = dm0[nao:, nao:].real
    veffa_diag, veffb_diag = uks_hess._get_vxc_diag(proxy, mo_coeff, mo_occ, max_memory)
    vxca, vxcb = uks_hess._get_vxc_deriv2(proxy, mo_coeff, mo_occ, max_memory)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        de2[i0, i0] += numpy.einsum('xypq,pq->xy', veffa_diag[:, :, p0:p1], dm0a[p0:p1]) * 2
        de2[i0, i0] += numpy.einsum('xypq,pq->xy', veffb_diag[:, :, p0:p1], dm0b[p0:p1]) * 2
        for j0, ja in enumerate(atmlst):
            q0, q1 = aoslices[ja][2:]
            de2[i0, j0] += numpy.einsum('xypq,pq->xy', vxca[ia][:, :, q0:q1], dm0a[q0:q1]) * 2
            de2[i0, j0] += numpy.einsum('xypq,pq->xy', vxcb[ia][:, :, q0:q1], dm0b[q0:q1]) * 2
    return de2


def veff_deriv1(hessobj, dm0, atmlst):
    mf = hessobj.base
    mol = hessobj.mol
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    v1 = ghf_hess.veff_deriv1(mol, dm0, atmlst, with_k=hybrid, k_scale=hyb)
    if hybrid and omega != 0:
        with mol.with_range_coulomb(omega):
            vlr = ghf_hess.veff_deriv1(mol, dm0, atmlst, with_k=True, k_scale=alpha - hyb)
            vj = ghf_hess.veff_deriv1(mol, dm0, atmlst, with_k=False)
        for ia in atmlst:
            v1[ia] += vlr[ia] - vj[ia]

    proxy, mo_coeff, mo_occ, _ = _uks_proxy(hessobj, dm0)
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    vxca, vxcb = uks_hess._get_vxc_deriv1(proxy, mo_coeff, mo_occ, max_memory)
    nao = mol.nao_nr()
    for ia in atmlst:
        v1[ia][:, :nao, :nao] += vxca[ia]
        v1[ia][:, nao:, nao:] += vxcb[ia]
    return v1


def enforce_translation_sum_rule(hess):
    '''H_AA = -sum_{B != A} H_AB.

    Without the response of the integration grid, the one-centre (diagonal)
    blocks of the exchange-correlation Hessian are wrong for atoms with steep
    core densities -- the AOs move with their nucleus, the grid points do not.
    For HBr/cc-pVDZ (PBE) the diagonal Br block is off by 1.4 Eh/Bohr^2 (the
    sum rule is violated by as much), while the off-diagonal blocks are within
    ~1e-6 of finite differences.  Rebuilding the diagonal from the sum rule is
    the translational part of the grid response.'''
    hess = hess.copy()
    natm = hess.shape[0]
    for a in range(natm):
        hess[a, a] = -sum(hess[a, b] for b in range(natm) if b != a)
    return hess


class Hessian(ghf_hess.Hessian):
    '''GKS Hessian (Kramers closed shell; no grid response, see
    translation_sum_rule)'''

    # rebuild the diagonal blocks from the translational sum rule (needs all atoms)
    translation_sum_rule = True

    def __init__(self, mf):
        ghf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self._keys = self._keys.union(['grids', 'translation_sum_rule'])

    partial_hess_veff = partial_hess_veff
    veff_deriv1 = veff_deriv1

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        de = ghf_hess.Hessian.kernel(self, mo_energy, mo_coeff, mo_occ, atmlst)
        if self.translation_sum_rule:
            if atmlst is not None and len(atmlst) != self.mol.natm:
                logger.warn(self, 'translation_sum_rule needs all atoms; not applied')
            else:
                logger.info(self, 'translational sum rule: max |sum_B H_AB| = %.3g '
                            'before rebuilding the diagonal blocks',
                            abs(de.sum(axis=1)).max())
                de = self.de = enforce_translation_sum_rule(de)
        return de


from pyscf.dft import gks as _pgks  # noqa: E402
_pgks.GKS.Hessian = lib.class_as_method(Hessian)
