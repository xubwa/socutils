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

# quartet cutoff of the exact 2e gradient (Schwarz x density bounds); the Hessians keep
# socutils.hessian.ghf.HESS2E_TOL (1e-13).  1e-11: max |g - g(1e-13)| 1.6e-9 (C16H34) /
# 7.1e-9 (cholesterol) Eh/Bohr, the 2e part 17 / 22 % faster (HF / cc-pVDZ)
GRAD2E_TOL = 1e-11
GRAD_SUBBLOCK = 112     # grid points per sub-block of the fused XC gradient (10 AO components per AO)


USE_FUSED_XC = True     # LDA / GGA XC gradient by the fused one-pass kernel (_xc_grad_fused)


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
    if xctype != 'MGGA' and USE_FUSED_XC and hasattr(libdft, 'SOCDFTxc_grad_fused'):
        de = _xc_grad_fused(libdft, ni, mol, grids, mf.xc, dm, xctype)
        if de is not None:
            return de
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


def _xc_grad_fused(libdft, ni, mol, grids, xc_code, dm, xctype):
    '''LDA / GGA XC gradient in one pass over the grid (lib/dft/vxc_rks.c
    SOCDFTxc_grad_fused: the AOs to the second derivatives, rho, the functional
    and the contraction per sub-block in cache); None if not applicable.'''
    from pyscf.gto import moleintor
    from socutils.dft import vxc_fast
    if grids.non0tab is None:
        grids.build(with_non0tab=True)
    if not vxc_fast._fusable(ni, xc_code, grids):
        return None
    xc = ni.libxc._get_xc(xc_code, 0)
    if xc.nfunc == 0:
        return numpy.zeros((mol.natm, 3))
    nao = mol.nao
    nthreads = lib.num_threads()
    ngrid = grids.weights.size
    coords = numpy.ascontiguousarray(grids.coords.T)
    weights = numpy.ascontiguousarray(grids.weights)
    non0tab = numpy.ascontiguousarray(grids.non0tab, dtype=numpy.uint8)
    lcg = moleintor.libcgto
    feval = ctypes.cast(lcg.GTOval_sph_deriv1 if xctype == 'LDA' else lcg.GTOval_sph_deriv2, ctypes.c_void_p)
    fxc = ctypes.cast(ni.libxc._itrf.LIBXC_eval_xc, ctypes.c_void_p)
    facs = (ctypes.c_double * xc.nfunc)(*xc.facs)
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    D = numpy.ascontiguousarray(dm)
    out = numpy.zeros((nthreads, 3, nao))
    libdft.SOCDFTxc_grad_fused(
        out.ctypes.data_as(ctypes.c_void_p), D.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p), weights.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrid), ctypes.c_int(GRAD_SUBBLOCK),
        ctypes.c_double(vxc_fast.AO_CUTOFF), ctypes.c_int(0 if xctype == 'LDA' else 1), feval,
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
        fxc, ctypes.c_int(xc.nfunc), ctypes.cast(xc.xc_arr, ctypes.c_void_p), facs, *vxc_fast._blas_ptrs())
    out = out.sum(axis=0)
    de = numpy.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = -2 * out[:, p0:p1].sum(axis=1)
    return de


def _gen_grad_usable(mol):
    '''the generated first-derivative kernels cover this basis (l <= 2, <= 4 contractions)'''
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    return (hess_fast.GRAD2E_GEN and bool(libv) and hasattr(libv, 'SOCVHFgen_grad2e')
            and mol._bas[:, 1].max() <= 2 and mol._bas[:, 3].max() <= 4)


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
    if jf != 0 and kf != 0 and _gen_grad_usable(mol):
        # hybrids: J and K in one pass of the generated kernels (J costs little more
        # there than the separate MD J pass)
        g = hess_fast.grad2e_jk_cart(mol, dm, [(jf, kf)], tol=GRAD2E_TOL)
        if g is not None:
            de += g[0]
            jf = kf = 0.
            t0 = logger.timer(grad, 'exact 2e gradient (generated kernels, J + K)', *t0)
    if jf != 0:
        # exact J from the McMurchie-Davidson J engine
        r = hess_fast.md_j_grad(mol, dm)
        if r is not None:
            de += r[0]
            jf = 0.
            t0 = logger.timer(grad, 'exact J gradient (MD J engine)', *t0)
    if jf != 0 or kf != 0:
        g = hess_fast.grad2e_jk_cart(mol, dm, [(jf, kf)], tol=GRAD2E_TOL)
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
