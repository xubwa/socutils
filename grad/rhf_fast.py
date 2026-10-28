#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
pyscf RHF / RKS nuclear gradients with the exact two-electron part and the XC
part as contracted energy derivatives instead of (3, nao, nao) derivative
matrices contracted with the density atom by atom:

* exact J: the McMurchie-Davidson J engine (lib/vhf/md_j.c,
  socutils.hessian.rhf_fast.md_j_grad): decontracted Cartesian primitives,
  the density in Hermite form once per primitive pair, no Cartesian integrals;
* exact K: the unique-quartet Rys kernel of the Hessian at derivative order 1
  (lib/vhf/hess2e_cart.c, socutils.hessian.rhf_fast.grad2e_jk_cart): one
  pass, density contracted on the fly, the fourth centre by translational
  invariance, K-only density screening;
* XC (fixed grid, LDA / GGA / MGGA): per grid block U = D aow, V = D phi
  (two GEMMs, pyscf: six) and one C pass (lib/dft/xc_hess.c
  SOCDFTrks_xc_grad) for sum_{mu in A} d_A phi_mu ... of every atom.

Drop-in classes (same results as pyscf's):

    from socutils.grad import rhf_fast
    g = rhf_fast.Gradients(mf).kernel()     # mf: scf.RHF / dft.RKS, also
                                            # density_fit() (RI-J; RI-J + exact K
                                            # with only_dfj=True)

Range-separated functionals, NLC, grid response and fitted K (RI-JK) keep
pyscf's code.
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from socutils.hessian import rhf_fast as hess_fast


def xc_grad(ks_grad, mol, dm, max_memory=2000):
    '''XC contribution to the RKS gradient on the fixed grid (pyscf's
    get_vxc contracted as in grad_elec), (natm, 3); None if not applicable.'''
    mf = ks_grad.base
    ni = mf._numint
    libdft = hess_fast._load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTrks_xc_grad') or mol.cart:
        return None
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return None
    grids, _ = rks_grad._initialize_grids(ks_grad)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    D = numpy.ascontiguousarray(dm)
    ao_deriv = 1 if xctype == 'LDA' else 2
    nw = 1 if xctype == 'LDA' else 4
    itype = {'LDA': 0, 'GGA': 1, 'MGGA': 2}[xctype]
    out = numpy.zeros((3, nao))
    null = lib.c_null_ptr()
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ng = weight.size
        if xctype == 'LDA':
            rho = make_rho(0, ao[0], mask, xctype)
        elif xctype == 'GGA':
            rho = make_rho(0, ao[:4], mask, xctype)
        else:
            rho = make_rho(0, ao[:10], mask, xctype)
        vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
        wv = numpy.ascontiguousarray(vxc.reshape(-1, ng) * weight)
        if xctype != 'LDA':
            wv[0] *= .5
        if xctype == 'MGGA':
            wv[4] *= .5
        aoT, cs = hess_fast._ao_stack(ao)                      # (ncomp, nao, g)
        aow = numpy.empty((nao, ng))
        libdft.SOCDFTxc_grad_aow(aow.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p),
                                 wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nw), ctypes.c_int(ng),
                                 ctypes.c_int(nao), ctypes.c_long(cs))
        U = lib.dot(D, aow)
        V = lib.dot(D, aoT[0]) if xctype != 'LDA' else None
        T = None
        if xctype == 'MGGA':
            T = numpy.empty((3, nao, ng))
            for k in range(3):
                lib.dot(D, aoT[1 + k] * wv[4], c=T[k])
        libdft.SOCDFTrks_xc_grad(out.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p),
                                 U.ctypes.data_as(ctypes.c_void_p),
                                 null if V is None else V.ctypes.data_as(ctypes.c_void_p),
                                 wv.ctypes.data_as(ctypes.c_void_p),
                                 null if T is None else T.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(itype), ctypes.c_int(ng), ctypes.c_int(nao), ctypes.c_long(cs))
    de = numpy.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = -2 * out[:, p0:p1].sum(axis=1)
    return de


def _fast_terms(grad, mol, dm):
    '''(natm, 3): the exact 2e (J unless density-fitted, K for hybrids) and XC
    parts of the gradient; None if this case keeps pyscf's code.'''
    mf = grad.base
    dfm = hess_fast._df_mode(mf)
    is_ks = isinstance(mf, hf.KohnShamDFT)
    if is_ks:
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        if omega != 0 or mf.do_nlc() or grad.grid_response:
            return None
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    else:
        hyb, hybrid = 1., True
    if dfm == 'jk' and hybrid:
        return None
    t0 = (logger.process_clock(), logger.perf_counter())
    jf = 1. if dfm is None else 0.
    kf = -hyb if hybrid else 0.
    de = numpy.zeros((mol.natm, 3))
    if jf != 0:
        # exact J from the McMurchie-Davidson J engine
        r = hess_fast.md_j_grad(mol, dm)
        if r is not None:
            de += r[0]
            jf = 0.
            t0 = logger.timer(grad, 'exact J gradient (MD J engine)', *t0)
    if jf != 0 or kf != 0:
        g = hess_fast.grad2e_jk_cart(mol, dm, [(jf, kf)])
        if g is None:
            return None
        de += g[0]
        t0 = logger.timer(grad, 'exact 2e gradient (unique quartets)', *t0)
    if is_ks:
        max_memory = max(2000, grad.max_memory * .9 - lib.current_memory()[0])
        dx = xc_grad(grad, mol, dm, max_memory)
        if dx is None:
            return None
        de += dx
        logger.timer(grad, 'XC gradient (contracted)', *t0)
    return de


class _FastGradMixin:
    '''get_veff returns only what pyscf's grad_elec still needs as a matrix (the
    density-fitted J with its auxiliary-basis response, else zero); the exact
    2e and XC parts enter through extra_force as energy derivatives.'''

    def get_veff(self, mol=None, dm=None):
        mf = self.base
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = mf.make_rdm1()
        de = _fast_terms(self, mol, dm)
        self._de_fast = de
        if de is None:
            return super().get_veff(mol, dm)
        if hess_fast._df_mode(mf) is None:
            nao = dm.shape[-1]
            return lib.tag_array(numpy.zeros((3, nao, nao)), exc1_grid=None)
        vj = self.get_j(mol, dm)
        if self.auxbasis_response:
            return lib.tag_array(numpy.asarray(vj), exc1_grid=None, aux=vj.aux.sum((0, 1)))
        return lib.tag_array(numpy.asarray(vj), exc1_grid=None)

    def extra_force(self, atom_id, envs):
        f = super().extra_force(atom_id, envs)
        de = getattr(self, '_de_fast', None)
        if de is not None:
            f = f + de[atom_id]
        return f


class RHFGradients(_FastGradMixin, rhf_grad.Gradients):
    pass


class RKSGradients(_FastGradMixin, rks_grad.Gradients):
    pass


def Gradients(mf):
    '''The fast gradient object for an RHF / RKS mean field (density-fitted
    ones included).'''
    dft = isinstance(mf, hf.KohnShamDFT)
    if hess_fast._df_mode(mf) is not None:
        from pyscf.df.grad import rhf as df_rhf_grad, rks as df_rks_grad
        base = df_rks_grad.Gradients if dft else df_rhf_grad.Gradients
        return type('DF' + ('RKS' if dft else 'RHF') + 'Gradients', (_FastGradMixin, base), {})(mf)
    return (RKSGradients if dft else RHFGradients)(mf)
