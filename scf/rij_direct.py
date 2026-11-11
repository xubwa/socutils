#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Direct RI-J: the density-fitted Coulomb matrix without the three-centre
tensor (lib/vhf/rij_direct.c), for molecules whose L^-1 (P|ij) does not fit
in memory.  Per build

    rho_P = sum_ij (P|ij) D_ij       (pass 1, McMurchie-Davidson on primitive Hermite densities)
    d     = M^-1 rho                 (Cholesky of the metric once)
    J_ij  = sum_P (ij|P) d_P         (pass 2)

on the orbital basis decontracted to Cartesian primitives and the auxiliary
primitives as pure Hermite Gaussians, the far-field interactions of the
auxiliary primitives of one centre summed into Hermite multipoles (exact to
FAR_EPS).

    eng = rij_direct.RIJDirect(mol, auxmol)
    vj = eng.get_j(dm)                    # dm (nao, nao) or (n, nao, nao), symmetric
'''

import ctypes
import time
import numpy
import scipy.linalg
import scipy.sparse
from pyscf import gto, lib
from pyscf.lib import logger

TOL = 1e-13             # screening of the bra pairs (sqrt((ab|ab)) x the other side's bound)
FAR_EPS = 1e-14         # relative accuracy of the asymptotic Boys functions in the far field
LMAX_BRA = 3
LMAX_AUX = 6


def _load():
    import os
    path = os.environ.get('SOCUTILS_RIJ_LIBRARY')     # a separately built rij_direct.c (development)
    if path:
        return numpy.ctypeslib.load_library(path, '.')
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFrij_direct_rho'):
        return None
    return libv


def available(mol, auxmol):
    return (_load() is not None and not mol.cart and not auxmol.cart
            and mol._bas[:, gto.ANG_OF].max() <= LMAX_BRA and auxmol._bas[:, gto.ANG_OF].max() <= LMAX_AUX)


def _top(l):
    '''the degree-l Hermite indices in the kernels' order (t ascending, then u ascending)'''
    return [(t, u, l - t - u) for t in range(l + 1) for u in range(l - t + 1)]


def _cart_order(l):
    '''pyscf's Cartesian order (x power descending, then y descending)'''
    return [(ix, iy, l - ix - iy) for ix in range(l, -1, -1) for iy in range(l - ix, -1, -1)]


def aux_primitives(auxmol):
    '''the auxiliary primitives (unique (atom, l, exponent)) sorted by atom and exponent descending, and
    the sparse map A (nh, naux) from the contracted spherical functions to their degree-l Hermite
    coefficients: h = A d; rho_aux = A^T X.
    Returns (aexp, al, hoff, cen_off, cxyz, A).'''
    prims = {}
    for ib in range(auxmol.nbas):
        ia = auxmol.bas_atom(ib)
        l = auxmol.bas_angular(ib)
        for e in auxmol.bas_exp(ib):
            prims[(ia, l, float(e))] = None
    keys = sorted(prims, key=lambda k: (k[0], -k[2], k[1]))
    index = {k: i for i, k in enumerate(keys)}
    aexp = numpy.array([k[2] for k in keys])
    al = numpy.array([k[1] for k in keys], dtype=numpy.int32)
    nt = (al + 1) * (al + 2) // 2
    hoff = numpy.zeros(len(keys) + 1, dtype=numpy.int32)
    hoff[1:] = numpy.cumsum(nt)
    natm = auxmol.natm
    cen = numpy.array([k[0] for k in keys])
    cen_off = numpy.searchsorted(cen, numpy.arange(natm + 1)).astype(numpy.int32)
    cxyz = numpy.ascontiguousarray(auxmol.atom_coords())
    rows, cols, vals = [], [], []
    ao_loc = auxmol.ao_loc_nr()
    for ib in range(auxmol.nbas):
        ia = auxmol.bas_atom(ib)
        l = auxmol.bas_angular(ib)
        exps = auxmol.bas_exp(ib)
        cs = auxmol._libcint_ctr_coeff(ib)              # (nprim, nctr)
        c2s = gto.cart2sph(l)                            # (ncart, nsph)
        cart = _cart_order(l)
        slot = {h: i for i, h in enumerate(_top(l))}
        perm = [slot[c] for c in cart]
        nsph = 2 * l + 1
        f = (-1) ** l           # libcint_ctr_coeff carries the s, p factors
        for ip, e in enumerate(exps):
            k = index[(ia, l, float(e))]
            for ic in range(cs.shape[1]):
                p0 = ao_loc[ib] + ic * nsph
                w = f * cs[ip, ic] / (2 * e) ** l
                for a, s in enumerate(perm):
                    for m in range(nsph):
                        if c2s[a, m] != 0:
                            rows.append(hoff[k] + s)
                            cols.append(p0 + m)
                            vals.append(w * c2s[a, m])
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(hoff[-1], auxmol.nao))
    return aexp, al, hoff, cen_off, cxyz, A


class RIJDirect:
    '''direct RI-J of mol with the auxiliary basis auxmol (spherical both)'''

    def __init__(self, mol, auxmol, tol=TOL, far_eps=FAR_EPS, verbose=None):
        from socutils.scf import jk_multi
        self.mol, self.auxmol = mol, auxmol
        self.tol, self.far_eps = tol, far_eps
        self.libv = _load()
        self.log = logger.new_logger(mol, verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        pmol, c = mol.decontract_basis(to_cart=True, aggregate=True)
        self.pmol = pmol
        self.c = scipy.sparse.csr_matrix(c)                       # (nprim, nao), block-sparse by shell
        self.cT = self.c.T.tocsr()
        self.qcond = numpy.ascontiguousarray(jk_multi._q_cond(pmol))
        self.aexp, self.al, self.hoff, self.cen_off, self.cxyz, self.A = aux_primitives(auxmol)
        self.AT = self.A.T.tocsr()
        j2c = auxmol.intor('int2c2e', hermi=1)
        self.qaux = numpy.sqrt(abs(numpy.diag(j2c)))
        self.low = scipy.linalg.cho_factor(j2c, lower=True)
        self.ao_loc = pmol.ao_loc_nr().astype(numpy.int32)
        self.log.timer('rij_direct setup', *t0)

    def _args(self):
        p = self.pmol
        return (ctypes.c_int(len(self.aexp)), ctypes.c_int(len(self.cen_off) - 1),
                self.aexp.ctypes.data_as(ctypes.c_void_p), self.al.ctypes.data_as(ctypes.c_void_p),
                self.hoff.ctypes.data_as(ctypes.c_void_p), self.cen_off.ctypes.data_as(ctypes.c_void_p),
                self.cxyz.ctypes.data_as(ctypes.c_void_p), self.ao_loc.ctypes.data_as(ctypes.c_void_p),
                p._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(p.natm),
                p._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(p.nbas),
                p._env.ctypes.data_as(ctypes.c_void_p))

    def get_rho(self, dm):
        '''rho_P = sum_ij (P|ij) D_ij of a symmetric dm (nao, nao)'''
        dmp = numpy.ascontiguousarray(self.c.dot(self.c.dot(dm).T))      # c D c^T, D symmetric
        X = numpy.empty(self.hoff[-1])
        self.npair_rho = self.libv.SOCVHFrij_direct_rho(
            X.ctypes.data_as(ctypes.c_void_p), dmp.ctypes.data_as(ctypes.c_void_p),
            self.qcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(self.tol),
            ctypes.c_double(self.qaux.max()), ctypes.c_double(self.far_eps), *self._args())
        return self.AT.dot(X)

    def get_vj_fit(self, dfit):
        '''J_ij = sum_P (ij|P) dfit_P'''
        h = numpy.ascontiguousarray(self.A.dot(dfit))
        npao = self.c.shape[0]
        vjp = numpy.empty((npao, npao))
        hmax = float(abs(self.qaux * dfit).max()) if dfit.size else 0.
        self.npair_vj = self.libv.SOCVHFrij_direct_vj(
            vjp.ctypes.data_as(ctypes.c_void_p), h.ctypes.data_as(ctypes.c_void_p),
            self.qcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(self.tol), ctypes.c_double(hmax),
            ctypes.c_double(self.far_eps), *self._args())
        return self.cT.dot(self.cT.dot(vjp).T)                    # c^T J c, J symmetric

    def get_vj_fit_multi(self, dfits, occ=None):
        '''J_ij of several fitted vectors dfits (nset, naux) in one pass (shared R tensors): (nset, nao, nao),
        or J C_occ (nset, nao, nocc) with occ'''
        dfits = numpy.asarray(dfits, dtype=numpy.double).reshape(-1, self.auxmol.nao)
        nset = dfits.shape[0]
        h = numpy.ascontiguousarray(numpy.asarray(self.A.dot(dfits.T)).T)          # (nset, nh)
        npao = self.c.shape[0]
        vjp = numpy.empty((nset, npao, npao))
        t0 = time.perf_counter()
        hmax = float(abs(self.qaux * dfits).max()) if dfits.size else 0.
        self.libv.SOCVHFrij_direct_vj_multi(
            vjp.ctypes.data_as(ctypes.c_void_p), h.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nset),
            self.qcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(self.tol), ctypes.c_double(hmax),
            ctypes.c_double(self.far_eps), *self._args())
        self.t_multi_c = time.perf_counter() - t0
        if occ is not None:
            # c^T Jp (c C_occ): one GEMM over the sets, then the block-sparse c^T
            cocc = numpy.asarray(self.c.dot(occ))                                     # (npao, nocc)
            nocc = cocc.shape[1]
            v = lib.dot(vjp.reshape(nset * npao, npao), cocc).reshape(nset, npao, nocc)
            return numpy.asarray(self.cT.dot(v.transpose(1, 0, 2).reshape(npao, -1))).reshape(
                -1, nset, nocc).transpose(1, 0, 2)
        return numpy.array([self.cT.dot(numpy.asarray(self.cT.dot(v)).T) for v in vjp])

    def fit(self, rho):
        return scipy.linalg.cho_solve(self.low, rho)

    def grad_3c(self, dm, dfit):
        '''the three-centre part of the RIJ gradient, sum_P dfit_P sum_ij D_ij d(ij|P)/dR, of a symmetric
        dm: (bra atoms (natm, 3), auxiliary centres (auxmol.natm, 3))'''
        dmp = numpy.ascontiguousarray(self.c.dot(self.c.dot(dm).T))
        h = numpy.ascontiguousarray(self.A.dot(dfit))
        gbra = numpy.zeros((self.pmol.natm, 3))
        gaux = numpy.zeros((self.auxmol.natm, 3))
        bound = float(abs(dmp).max() * abs(self.qaux * dfit).max())
        self.npair_grad = self.libv.SOCVHFrij_direct_grad(
            gbra.ctypes.data_as(ctypes.c_void_p), gaux.ctypes.data_as(ctypes.c_void_p),
            dmp.ctypes.data_as(ctypes.c_void_p), h.ctypes.data_as(ctypes.c_void_p),
            self.qcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(self.tol), ctypes.c_double(bound),
            ctypes.c_double(self.far_eps), *self._args())
        return gbra, gaux

    def deriv_y(self, dm):
        '''Y^xi_P = sum_ij D_ij d(ij|P)/dR_xi (natm * 3, naux) of a symmetric dm, the bra atoms by the
        derivative densities, the auxiliary centres by translational invariance; auxmol's atoms those
        of mol'''
        dmp = numpy.ascontiguousarray(self.c.dot(self.c.dot(dm).T))
        natm = self.pmol.natm
        nx = self.hoff[-1]
        Xh = numpy.empty((natm * 3, nx))
        self.libv.SOCVHFrij_direct_deriv_y(
            Xh.ctypes.data_as(ctypes.c_void_p), dmp.ctypes.data_as(ctypes.c_void_p),
            self.qcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(self.tol),
            ctypes.c_double(self.qaux.max()), ctypes.c_double(self.far_eps), *self._args())
        Y = numpy.asarray(self.AT.dot(Xh.T).T).reshape(natm, 3, -1)
        tot = Y.sum(axis=0)                                   # (3, naux): -1 x the auxiliary centres' part
        for ia, (p0, p1) in enumerate(self.auxmol.aoslice_by_atom()[:, 2:]):
            Y[ia, :, p0:p1] -= tot[:, p0:p1]
        return Y.reshape(natm * 3, -1)

    def deriv_w(self, dfit, c0=0, c1=0, occ=None):
        '''(Wrows (3, nao, nao) = sum_P (d_x i j|P) d_P (derivative on the first function), Waux (c1 - c0, 3,
        nao, nao) with Waux + Waux^T = sum_{P in C} d(ij|P)/dC_x d_P for the auxiliary centres c0 <= C < c1,
        or None if c1 <= c0).  occ: Waux as (Waux + Waux^T) C_occ (c1 - c0, 3, nao, nocc) instead.'''
        h = numpy.ascontiguousarray(self.A.dot(dfit))
        npao = self.c.shape[0]
        nc = max(0, c1 - c0)
        Wp = numpy.zeros((3, npao, npao))
        Wa = numpy.zeros((nc, 3, npao, npao)) if nc else None
        hmax = float(abs(self.qaux * dfit).max())
        self.libv.SOCVHFrij_direct_deriv_w(
            Wp.ctypes.data_as(ctypes.c_void_p), Wa.ctypes.data_as(ctypes.c_void_p) if nc else None,
            ctypes.c_int(c0), ctypes.c_int(c1), h.ctypes.data_as(ctypes.c_void_p),
            self.qcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(self.tol), ctypes.c_double(hmax),
            ctypes.c_double(self.far_eps), *self._args())

        def to_ao(w):
            return numpy.asarray(self.cT.dot(numpy.asarray(self.cT.dot(w.T)).T))   # c^T w c
        Wrows = numpy.array([to_ao(w) for w in Wp])
        Waux = None
        if nc and occ is not None:
            # c^T (Wp + Wp^T) (c C_occ): GEMMs over the centres, then the block-sparse c^T
            cocc = numpy.asarray(self.c.dot(occ))
            nocc = cocc.shape[1]
            Wa = Wa.reshape(nc * 3, npao, npao)
            v = lib.dot(Wa.reshape(-1, npao), cocc).reshape(nc * 3, npao, nocc)
            for k in range(nc * 3):
                v[k] += lib.dot(Wa[k].T, cocc)
            Wa = None
            Waux = numpy.asarray(self.cT.dot(v.transpose(1, 0, 2).reshape(npao, -1))).reshape(
                -1, nc, 3, nocc).transpose(1, 2, 0, 3)
        elif nc:
            Waux = numpy.array([[to_ao(w) for w in wc] for wc in Wa])
        return Wrows, Waux

    def grad(self, dm, dfit=None):
        '''the RIJ Coulomb gradient d/dR 1/2 rho^T M^-1 rho (natm, 3), auxiliary-basis response included
        (pyscf's auxbasis_response): g = d . rho^xi - 1/2 d M^xi d, d = M^-1 rho; auxmol's atoms
        those of mol.  dfit: d if known (else one rho pass).'''
        dm = numpy.asarray(dm, dtype=numpy.double)
        dm = (dm + dm.T) * .5
        tol, self.tol = self.tol, min(self.tol, TOL)       # (an SCF's engine may hold a loose cutoff)
        try:
            if dfit is None:
                dfit = self.fit(self.get_rho(dm))
            gbra, gaux = self.grad_3c(dm, dfit)
        finally:
            self.tol = tol
        auxmol = self.auxmol
        v = numpy.einsum('xpq,q->xp', auxmol.intor('int2c2e_ip1', aosym='s1'), dfit)
        for ia, (p0, p1) in enumerate(auxmol.aoslice_by_atom()[:, 2:]):
            gaux[ia] += v[:, p0:p1].dot(dfit[p0:p1])
        return gbra + gaux

    def get_j_multi(self, dm, hermi=1, occ=None, max_memory=4000):
        '''J of many densities ((nset,) nao, nao) without the three-centre tensor: rho of each density
        (pass 1, one density at a time), one fit of all of them, then the J of all fitted vectors in one
        pass (get_vj_fit_multi) in batches whose primitive-basis J (npao^2 per density) fit max_memory (MB).
        (nset, nao, nao), or J C_occ (nset, nao, nocc) with occ.'''
        dm = numpy.asarray(dm, dtype=numpy.double)
        nao = dm.shape[-1]
        dms = dm.reshape(-1, nao, nao)
        if hermi != 1:
            dms = (dms + dms.transpose(0, 2, 1)) * .5
        nset = dms.shape[0]
        t0 = (logger.process_clock(), logger.perf_counter())
        rho = numpy.empty((nset, self.auxmol.nao))
        for i, d in enumerate(dms):
            rho[i] = self.get_rho(d)
        t0 = self.log.timer_debug1('rij_direct rho of %d densities' % nset, *t0)
        dfit = numpy.ascontiguousarray(self.fit(rho.T).T)
        npao = self.c.shape[0]
        nb = max(1, int(max_memory * 1e6 / (8. * npao * npao * 1.5)))
        out = [self.get_vj_fit_multi(dfit[i:i + nb], occ) for i in range(0, nset, nb)]
        self.log.timer_debug1('rij_direct vj of %d fitted vectors' % nset, *t0)
        vj = out[0] if len(out) == 1 else numpy.concatenate(out)
        return vj if occ is not None else vj.reshape(dm.shape)

    def get_j(self, dm, hermi=1):
        dm = numpy.asarray(dm, dtype=numpy.double)
        shape = dm.shape
        nao = shape[-1]
        dms = dm.reshape(-1, nao, nao)
        if hermi != 1:
            dms = (dms + dms.transpose(0, 2, 1)) * .5
        vj = numpy.empty_like(dms)
        for i, d in enumerate(dms):
            t0 = (logger.process_clock(), logger.perf_counter())
            rho = self.get_rho(d)
            t0 = self.log.timer_debug1('rij_direct rho', *t0)
            vj[i] = self.get_vj_fit(self.fit(rho))
            self.log.timer_debug1('rij_direct vj', *t0)
        return vj.reshape(shape)
