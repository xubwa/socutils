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

The response of the integration grid is included by default
(Hessian.grid_response, analytic: socutils.hessian.xc_grid_response).  Without
it (as in pyscf's UKS Hessian) the one-centre blocks are badly wrong for steep
core densities; the translational part is then restored by rebuilding the
diagonal blocks from the sum rule sum_B H_AB = 0 (translation_sum_rule).
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

    if hessobj.grid_response:
        hxc = hessobj.xc_fd(dm0)[0]
        return de2 + hxc[numpy.ix_(atmlst, atmlst)]

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

    if hessobj.grid_response:
        vxc1 = hessobj.xc_fd(dm0, need_hess=False)[1]
        for ia in atmlst:
            v1[ia] += vxc1[ia]
        return v1

    proxy, mo_coeff, mo_occ, _ = _uks_proxy(hessobj, dm0)
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    vxca, vxcb = uks_hess._get_vxc_deriv1(proxy, mo_coeff, mo_occ, max_memory)
    nao = mol.nao_nr()
    for ia in atmlst:
        v1[ia][:, :nao, :nao] += vxca[ia]
        v1[ia][:, nao:, nao:] += vxcb[ia]
    return v1


def _xc_grad_fixed_dm(mf, mol, grids, dm):
    '''Exchange-correlation energy gradient (natm, 3) at a fixed AO density
    matrix, with the full response of the grid (points and Becke weights).'''
    from socutils.grad import gks_grad
    nao = mol.nao_nr()
    # Kramers closed shell: the collinear functional is exact (m = 0)
    with lib.temporary_env(mf._numint, collinear='col'):
        excsum, vxc = gks_grad.get_vxc_full_response(mf._numint, mol, grids, mf.xc, dm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.array(excsum, dtype=float)
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        for q0 in (0, nao):
            de[ia] += 2 * numpy.einsum('xij,ji->x', vxc[:, q0 + p0:q0 + p1],
                                       dm[:, q0 + p0:q0 + p1]).real
    return de


def _displaced_grids(grids, mol):
    import copy
    g = copy.copy(grids)
    g.reset(mol)
    g.build(with_non0tab=True)
    return g


def xc_geometric_derivs_fd(hessobj, dm0, step=1e-3):
    '''Semi-numerical exchange-correlation derivatives with the grid response.

    At fixed AO density dm0, the atoms (and their grids) are displaced by
    +-step, +-2 step; a five-point difference of (a) the analytic XC gradient
    with full grid response and (b) the Vxc matrix gives the XC partial
    Hessian (natm, natm, 3, 3) and the XC part of the first-order Fock
    matrices {atom: (3, 2n, 2n)}.'''
    mf = hessobj.base
    mol = hessobj.mol
    natm = mol.natm
    coords = mol.atom_coords()
    ni = mf._numint
    hxc = numpy.zeros((natm, natm, 3, 3))
    vxc1 = {ia: numpy.zeros((3,) + dm0.shape, dtype=numpy.complex128) for ia in range(natm)}
    coef = ((2, -1 / 12.), (1, 8 / 12.), (-1, -8 / 12.), (-2, 1 / 12.))
    for jb in range(natm):
        for y in range(3):
            for n, c in coef:
                xyz = coords.copy()
                xyz[jb, y] += n * step
                mold = mol.set_geom_(xyz, unit='Bohr', inplace=False)
                gridd = _displaced_grids(mf.grids, mold)
                g = _xc_grad_fixed_dm(mf, mold, gridd, dm0)
                hxc[:, jb, :, y] += c * g / step
                with lib.temporary_env(ni, collinear='col'):
                    v = ni.get_vxc(mold, gridd, mf.xc, dm0, hermi=1)[2]
                vxc1[jb][y] += c * v / step
    return hxc, vxc1


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
    '''GKS Hessian (Kramers closed shell)'''

    # rebuild the diagonal blocks from the translational sum rule (needs all
    # atoms); used when grid_response is off
    translation_sum_rule = True
    # include the response of the integration grid in the exchange-correlation
    # terms: 'analytic' (socutils.hessian.xc_grid_response: Becke weight
    # derivatives and the motion of the points) or 'fd' (finite differences of
    # the analytic grid-response gradient and of Vxc at fixed density)
    grid_response = True
    grid_response_method = 'analytic'
    grid_fd_step = 1e-3

    def __init__(self, mf):
        ghf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self._xc_fd_cache = None
        self._keys = self._keys.union(['grids', 'translation_sum_rule', 'grid_response',
                                       'grid_response_method', 'grid_fd_step'])

    partial_hess_veff = partial_hess_veff
    veff_deriv1 = veff_deriv1

    def xc_fd(self, dm0, need_hess=True):
        '''(XC partial Hessian, XC first-order Fock matrices) with the grid
        response, computed once per density.  need_hess=False (e.g. for the
        first-order Fock matrices of an excited-state gradient) skips the
        second derivatives of the analytic route.'''
        cache = self._xc_fd_cache
        if (cache is None or cache[0].shape != dm0.shape or abs(cache[0] - dm0).max() > 1e-12
                or (need_hess and cache[1] is None)):
            if self.grid_response_method == 'fd':
                hxc, vxc1 = xc_geometric_derivs_fd(self, dm0, self.grid_fd_step)
            elif self.grid_response_method == 'analytic':
                from socutils.hessian import xc_grid_response
                hxc, vxc1 = xc_grid_response.xc_derivs_grid_response(self.base, dm0,
                                                                      with_hess=need_hess)
                if not need_hess:
                    hxc = None
            else:
                raise ValueError('grid_response_method %s' % self.grid_response_method)
            self._xc_fd_cache = cache = (dm0.copy(), hxc, vxc1)
        return cache[1], cache[2]

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        de = ghf_hess.Hessian.kernel(self, mo_energy, mo_coeff, mo_occ, atmlst)
        if self.translation_sum_rule and not self.grid_response:
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
