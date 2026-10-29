#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
The SCF XC potential of RKS, pyscf.dft.numint.nr_rks (same conventions and
results).  LDA / GGA: cache-blocked C passes (lib/dft/vxc_rks.c) over
sub-blocks of SUBBLOCK grid points, each on the AOs with max |phi_k| >
AO_CUTOFF on it (cc-pVDZ (H2O)_16: ~160 of 384 per box), one read of the AO
block per pass; pass 1 rho, libxc, pass 2 V.  Meta-GGA (and without the
library) the whole-block path below:

    rho       from the occupied orbitals when the density carries them
              (c_k = C^T d_k phi, nocc instead of nao: rho = sum_i c_0 c_0,
              d_k rho = 2 sum_i c_0 c_k, tau = 1/2 sum_k sum_i c_k c_k),
              else from c = D phi;
    aow       = sum_k w_k d_k phi           (lib/dft/xc_hess.c SOCDFTxc_grad_aow)
    V        += phi_0 aow^T                  (GEMM; GGA / MGGA: V + V^T at the end)
    V1       += sum_k d_k phi (w_tau d_k phi)^T   (MGGA)


    vxc_fast.apply(mf)      # mf._numint.nr_rks from here (RKS, LDA / GGA)

On large AO blocks every whole-block pass is bound by memory bandwidth; the
sub-blocks keep a pass's data in cache.
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.dft import numint

USE_MGGA = False


def _orbitals(dm):
    '''sqrt(occ) C of the occupied orbitals if dm carries them (pyscf tags), else None'''
    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm, 'mo_occ', None)
    if mo_coeff is None or mo_occ is None:
        return None
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)
    if mo_coeff.ndim != 2 or mo_occ.ndim != 1 or numpy.iscomplexobj(mo_coeff):
        return None
    occ = mo_occ > 0
    return numpy.ascontiguousarray(mo_coeff[:, occ] * numpy.sqrt(mo_occ[occ]))


def _rho(aoT, xctype, dm, orb):
    '''rho (1, 4 or 5 rows) on the block from the AO stack (ncomp, nao, g)'''
    ng = aoT.shape[2]
    nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    ncomp = 1 if xctype == 'LDA' else 4
    rho = numpy.empty((nvar, ng))
    if orb is not None:
        c = [lib.dot(orb.T, aoT[k]) for k in range(ncomp)]                 # (nocc, g)
        rho[0] = numint._contract_rho(c[0].T, c[0].T)
        for k in range(1, ncomp):
            rho[k] = 2 * numint._contract_rho(c[0].T, c[k].T)
        if xctype == 'MGGA':
            rho[4] = .5 * sum(numint._contract_rho(c[k].T, c[k].T) for k in range(1, 4))
        return rho
    c = lib.dot(dm, aoT[0])                                                # (nao, g)
    rho[0] = numint._contract_rho(aoT[0].T, c.T)
    for k in range(1, ncomp):
        rho[k] = 2 * numint._contract_rho(aoT[k].T, c.T)
    if xctype == 'MGGA':
        rho[4] = 0
        for k in range(1, 4):
            rho[4] += .5 * numint._contract_rho(aoT[k].T, lib.dot(dm, aoT[k]).T)
    return rho


def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
    from socutils.hessian.rhf_fast import _load_socdft, _ao_stack
    libdft = _load_socdft()
    xctype = ni._xc_type(xc_code)
    # meta-GGA keeps pyscf's (the tau products are slower here, 2.5 vs 2.1 s for
    # (H2O)_16 TPSS); the MGGA branches below are kept for completeness
    if (libdft is None or not hasattr(libdft, 'SOCDFTxc_grad_aow') or hermi != 1 or mol.cart
            or xctype not in ('LDA', 'GGA', 'MGGA') or (xctype == 'MGGA' and not USE_MGGA)
            or (xctype == 'MGGA' and any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))):
        return numint.nr_rks(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
    dms_a = numpy.asarray(dms)
    if numpy.iscomplexobj(dms_a):
        return numint.nr_rks(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
    nao = dms_a.shape[-1]
    d = dms_a.reshape(-1, nao, nao)
    nset = d.shape[0]
    orb = _orbitals(dms) if nset == 1 else None
    ncomp = 1 if xctype == 'LDA' else 4
    ao_deriv = 0 if xctype == 'LDA' else 1
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset, nao, nao))
    v1 = numpy.zeros((nset, nao, nao)) if xctype == 'MGGA' else None
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if xctype != 'MGGA' and hasattr(libdft, 'SOCDFTvxc_mat'):
        return _nr_rks_blocked(libdft, ni, mol, grids, xc_code, dms_a, d, xctype, max_memory)
    aow = None
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        ng = weight.size
        aoT, cs = _ao_stack(ao)
        if aow is None or aow.shape[1] != ng:
            aow = numpy.empty((nao, ng))
        for i in range(nset):
            dm = numpy.ascontiguousarray(d[i])
            rho = _rho(aoT, xctype, dm, orb)
            exc, vxc = ni.eval_xc_eff(xc_code, rho[0] if xctype == 'LDA' else rho, deriv=1, xctype=xctype,
                                      spin=0)[:2]
            den = rho[0] * weight
            nelec[i] += den.sum()
            excsum[i] += numpy.dot(den, exc)
            wv = numpy.ascontiguousarray(numpy.asarray(vxc).reshape(-1, ng)[:ncomp] * weight)
            if xctype != 'LDA':
                wv[0] *= .5             # *.5 for V + V^T
            # by AO rows, each a contiguous stream (lib/dft/xc_hess.c; 2.3x the
            # grid-chunked SOCDFTaow_multi on large blocks, which are bandwidth bound)
            libdft.SOCDFTxc_grad_aow(aow.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p),
                                     wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncomp), ctypes.c_int(ng),
                                     ctypes.c_int(nao), ctypes.c_long(cs))
            lib.dot(aoT[0], aow.T, beta=1, c=vmat[i])
            if xctype == 'MGGA':
                wt = numpy.asarray(vxc)[4] * weight * .5        # *.5 for 1/2 in tau
                for k in range(1, 4):
                    lib.dot(aoT[k] * wt, aoT[k].T, beta=1, c=v1[i])
    if xctype != 'LDA':
        vmat = vmat + vmat.transpose(0, 2, 1)
    if xctype == 'MGGA':
        vmat += v1
    if dms_a.ndim == 2:
        return nelec[0], excsum[0], vmat[0]
    return nelec, excsum, vmat.reshape(dms_a.shape)


SUBBLOCK = 128          # grid points per cache-blocked sub-block
AO_CUTOFF = 1e-15       # max_g |phi_k| below which an AO is dropped from a sub-block
_blas = None


def _blas_ptrs():
    '''dgemm_ and (MKL) MKL_Set_Num_Threads_Local of the BLAS pyscf has loaded'''
    global _blas
    if _blas is None:
        from pyscf.lib import numpy_helper
        h = numpy_helper._np_helper
        dgemm = ctypes.cast(h.dgemm_, ctypes.c_void_p)
        try:
            # the C entry (by value); mkl_set_num_threads_local is the Fortran one
            local = ctypes.cast(h.MKL_Set_Num_Threads_Local, ctypes.c_void_p)
        except AttributeError:
            local = ctypes.c_void_p(None)
        _blas = (dgemm, local)
    return _blas


def _nr_rks_blocked(libdft, ni, mol, grids, xc_code, dms_a, d, xctype, max_memory):
    '''LDA / GGA through the cache-blocked passes of lib/dft/vxc_rks.c: sub-blocks
    of SUBBLOCK points, each on its AOs with max |phi_k| > AO_CUTOFF, one read
    of the AO block per pass'''
    from socutils.hessian.rhf_fast import _ao_stack
    nao = d.shape[-1]
    nset = d.shape[0]
    ncomp = 1 if xctype == 'LDA' else 4
    itype = 0 if xctype == 'LDA' else 1
    nthreads = lib.num_threads()
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset, nao, nao))
    vbuf = numpy.empty((nthreads, nao, nao))
    blas = _blas_ptrs()
    for i in range(nset):
        dm = numpy.ascontiguousarray(d[i])
        vbuf[:] = 0
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 0 if xctype == 'LDA' else 1,
                                                      max_memory=max_memory):
            ng = weight.size
            aoT, cs = _ao_stack(ao)
            nsub = (ng + SUBBLOCK - 1) // SUBBLOCK
            idx = numpy.empty((nsub, nao), dtype=numpy.int32)
            nl = numpy.empty(nsub, dtype=numpy.int32)
            rho = numpy.empty((ncomp, ng))
            libdft.SOCDFTvxc_rho(
                rho.ctypes.data_as(ctypes.c_void_p), idx.ctypes.data_as(ctypes.c_void_p),
                nl.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p), ctypes.c_long(cs),
                dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nao), ctypes.c_int(ng), ctypes.c_int(SUBBLOCK),
                ctypes.c_double(AO_CUTOFF), ctypes.c_int(itype), *blas)
            exc, vxc = ni.eval_xc_eff(xc_code, rho[0] if xctype == 'LDA' else rho, deriv=1, xctype=xctype,
                                      spin=0)[:2]
            den = rho[0] * weight
            nelec[i] += den.sum()
            excsum[i] += numpy.dot(den, exc)
            wv = numpy.ascontiguousarray(numpy.asarray(vxc).reshape(-1, ng)[:ncomp] * weight)
            if xctype != 'LDA':
                wv[0] *= .5             # *.5 for V + V^T
            libdft.SOCDFTvxc_mat(
                vbuf.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p), ctypes.c_long(cs),
                wv.ctypes.data_as(ctypes.c_void_p), idx.ctypes.data_as(ctypes.c_void_p),
                nl.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nao), ctypes.c_int(ng), ctypes.c_int(SUBBLOCK),
                ctypes.c_int(itype), *blas)
        v = vbuf.sum(axis=0)
        vmat[i] = v + v.T if xctype != 'LDA' else v
    if dms_a.ndim == 2:
        return nelec[0], excsum[0], vmat[0]
    return nelec, excsum, vmat.reshape(dms_a.shape)


def apply(mf):
    '''mf._numint.nr_rks from vxc_fast (RKS objects).  Modifies and returns mf.'''
    ni = mf._numint
    if isinstance(ni, numint.NumInt):
        ni.nr_rks = lambda *a, **k: nr_rks(ni, *a, **k)
    return mf
