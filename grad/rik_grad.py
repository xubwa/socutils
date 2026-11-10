#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Analytic nuclear gradient of the density-fitted (RI-JK) exchange energy of a
closed-shell RHF / RKS, auxiliary-basis response included (the exact
derivative of the fitted energy, pyscf's auxbasis_response):

    E_K = -1/4 sum_PQ (M^{-1})_PQ sum_mn A^P_mn A^Q_mn,   A^P = c^T (P|..) c,

c = C_occ sqrt(n_occ) (D = c c^T), M = (P|Q).  With d^P = sum_Q (M^{-1})_PQ A^Q,
Gamma^P = c d^P c^T and gamma_PQ = sum_mn d^P_mn d^Q_mn,

    dE_K/dR = -1/4 [2 sum_P sum_ij Gamma^P_ij d(ij|P)/dR - sum_PQ gamma_PQ dM_PQ/dR].

A^P from the SCF's factor B = L^{-1} (P|ij) (one pass of with_df.loop(), in
core or HDF5, d = L^{-T} (c^T B c)) when it is the Cholesky factor of this
auxiliary basis, else from the three-centre integrals per auxiliary block.
The three-centre derivative part per block of auxiliary shells: Gamma^P of
the block by two GEMMs, then one pass over the shell triplets
(lib/vhf/ri_k_grad.c SOCVHFri_k_grad: int3c2e_ip1 contracted on the fly,
Schwarz x max |Gamma| screening), so the memory is bounded by the block.

The same pass returns the fitted Coulomb coefficients d_J = M^{-1} rho, which
the RI-J gradient (socutils.hessian.rhf_fast.ri_j_grad) takes instead of
recomputing them.
'''

import ctypes
import numpy
import scipy.linalg
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.gto import moleintor

TOL = 1e-11          # triplet cutoff: Schwarz bound x max |Gamma| (Eh / Bohr scale)
BLK_MEMORY = .3      # fraction of the free memory for Gamma of one auxiliary block


def _load():
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFri_k_grad'):
        return None
    return libv


def _metric_low(auxmol, naux_factor):
    '''the lower Cholesky factor of (P|Q) as pyscf's cholesky_eri takes it, or None'''
    if naux_factor != auxmol.nao:
        return None
    try:
        return scipy.linalg.cholesky(auxmol.intor('int2c2e', hermi=1), lower=True)
    except scipy.linalg.LinAlgError:
        return None


def _check_cols(mol):
    '''packed pair indices of the first AO shell's pairs (the spot check of the factor)'''
    di = mol.ao_loc[1]
    return numpy.array([i * (i + 1) // 2 + j for i in range(di) for j in range(i + 1)])


def _factor_is_cholesky(mol, auxmol, low, got):
    '''with_df's factor is L^{-1} (P|ij) of auxmol: L B against (P|ij) of the pairs of the first
    AO shell (got: those columns of the factor, (naux, ncol))'''
    from pyscf import df
    ref = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', shls_slice=(0, 1, 0, 1, 0, auxmol.nbas))
    ref = ref.reshape(-1, auxmol.nao)                       # (npair of shell 0, naux)
    return abs(low.dot(got) - ref.T).max() < 1e-8 * max(1., abs(ref).max())


def _occ_factor(mo_coeff, mo_occ):
    sel = mo_occ > 0
    return numpy.asarray(mo_coeff[:, sel] * numpy.sqrt(mo_occ[sel]), order='F')


def fitted_coefficients(mol, auxmol, dm, c, with_df=None, max_memory=4000, verbose=None):
    '''(dJ (naux,), dK (naux, nocc, nocc)): dJ = M^{-1} rho, rho_P = (P|ij) D_ij, and
    dK^P = sum_Q (M^{-1})_PQ c^T (Q|..) c, from with_df's factor (when it is the
    Cholesky factor of auxmol) or the three-centre integrals.'''
    from pyscf import df
    from pyscf.df.grad.rhf import _gen_metric_solver
    from pyscf.ao2mo.outcore import balance_partition
    from socutils.scf import rik
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    nao = mol.nao
    nocc = c.shape[1]
    naux = auxmol.nao
    npair = nao * (nao + 1) // 2
    d = dm + dm.T
    d[numpy.diag_indices(nao)] *= .5
    dmtril = lib.pack_tril(d)
    low = None
    if with_df is not None:
        if getattr(with_df, '_cderi', None) is None:
            with_df.build()
        low = _metric_low(auxmol, with_df.get_naoaux())
    rho = numpy.empty(naux)
    A = numpy.empty((naux, nocc, nocc))
    mem_free = max(400., max_memory - lib.current_memory()[0])
    blk = int(max(8, min(naux, mem_free * .3e6 / 8 / (2 * npair + nocc * nao))))
    buf = numpy.empty(blk * nocc * nao)
    if low is not None:
        p0 = 0
        cols = _check_cols(mol)
        got = numpy.empty((naux, cols.size))
        for eri1 in with_df.loop(blk):
            nb = eri1.shape[0]
            got[p0:p0 + nb] = eri1[:, cols]
            rho[p0:p0 + nb] = lib.dot(eri1, dmtril[:, None])[:, 0]
            x = rik._half_trans(eri1, c, buf)                   # (nb, nocc, nao)
            lib.dot(x.reshape(-1, nao), c, c=A[p0:p0 + nb].reshape(-1, nocc))
            p0 += nb
        t0 = log.timer_debug1('RI-K grad: c^T B c from the factor', *t0)
        if not _factor_is_cholesky(mol, auxmol, low, got):
            log.info('RI-K grad: the factor is not L^-1 (P|ij) of the auxiliary basis; '
                     'the three-centre integrals instead')
            low = None
    if low is not None:
        dJ = scipy.linalg.solve_triangular(low, rho, lower=True, trans='T')
        A = A.reshape(naux, -1)
        A = scipy.linalg.solve_triangular(low, A, lower=True, trans='T', overwrite_b=True)
        t0 = log.timer_debug1('RI-K grad: L^-T', *t0)
        return dJ, A.reshape(naux, nocc, nocc)
    nbas = mol.nbas
    for k0, k1, _ in balance_partition(auxmol.ao_loc, blk):
        p0, p1 = auxmol.ao_loc[k0], auxmol.ao_loc[k1]
        int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', shls_slice=(0, nbas, 0, nbas, k0, k1))
        int3c = numpy.asarray(int3c.T, order='C')               # (nb, npair)
        rho[p0:p1] = lib.dot(int3c, dmtril[:, None])[:, 0]
        x = rik._half_trans(int3c, c, buf)
        lib.dot(x.reshape(-1, nao), c, c=A[p0:p1].reshape(-1, nocc))
    int3c = None
    t0 = log.timer_debug1('RI-K grad: c^T (P|ij) c from the integrals', *t0)
    solve = _gen_metric_solver(auxmol.intor('int2c2e', aosym='s1'))
    dJ = solve(rho.copy())
    A = solve(A.reshape(naux, -1)).reshape(naux, nocc, nocc)
    return dJ, A


def k_grad(mol, auxmol, c, dK, tol=TOL, max_memory=4000, verbose=None):
    '''dE_K/dR (natm, 3) of E_K = -1/4 sum (M^{-1})_PQ A^P.A^Q (see the module doc) from the
    fitted coefficients dK (naux, nocc, nocc); None if the library is unavailable.'''
    from socutils.hessian import ghf as ghf_hess
    libv = _load()
    if libv is None or mol.cart:
        return None
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    natm, nao, nbas = mol.natm, mol.nao, mol.nbas
    naux = auxmol.nao
    nocc = c.shape[1]
    c = numpy.ascontiguousarray(c)
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = moleintor.make_loc(bas, 'int3c2e_sph').astype(numpy.int32)
    opt = moleintor.make_cintopt(atm, bas, env, 'int3c2e_ip1_sph')
    q = numpy.max([ghf_hess._q_cond(mol, 'int2e_ip1')[k] for k in (0, 1)], axis=0)
    q = numpy.ascontiguousarray(numpy.maximum(q, q.T))
    aux_loc = auxmol.ao_loc_nr()
    m2 = auxmol.intor('int2c2e', aosym='s1')
    qP = numpy.ascontiguousarray(numpy.sqrt(numpy.maximum.reduceat(abs(numpy.diag(m2)), aux_loc[:-1])))
    gout = numpy.zeros((natm + auxmol.natm, 3))
    mem_free = max(400., max_memory - lib.current_memory()[0])
    blk = int(max(16, min(naux, mem_free * BLK_MEMORY * 1e6 / 8 / (nao * nao + nocc * nao + nbas * nbas / 3))))
    from pyscf.ao2mo.outcore import balance_partition
    for k0, k1, _ in balance_partition(auxmol.ao_loc, blk):
        p0, p1 = aux_loc[k0], aux_loc[k1]
        nb = p1 - p0
        # Gamma^P = c d^P c^T laid out (i, P, j): U = d^P c^T (P, m, j), Gamma = c U
        U = lib.dot(dK[p0:p1].reshape(-1, nocc), c.T)                 # (nb nocc, nao)
        U = numpy.ascontiguousarray(U.reshape(nb, nocc, nao).transpose(1, 0, 2)).reshape(nocc, -1)
        gam = lib.dot(c, U)                                            # (nao, nb nao)
        U = None
        gcond = numpy.empty((k1 - k0, nbas, nbas))
        sp, si = nao, nb * nao
        libv.SOCVHFri_k_gcond(gcond.ctypes.data_as(ctypes.c_void_p), gam.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_size_t(sp), ctypes.c_size_t(si), ctypes.c_int(nbas + k0),
                              ctypes.c_int(nbas + k1), ao_loc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas))
        libv.SOCVHFri_k_grad(
            gout.ctypes.data_as(ctypes.c_void_p), gam.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(sp), ctypes.c_size_t(si), gcond.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas + k0), ctypes.c_int(nbas + k1), q.ctypes.data_as(ctypes.c_void_p),
            qP.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol), opt,
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            ctypes.c_int(auxmol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            ctypes.c_int(auxmol.nbas), env.ctypes.data_as(ctypes.c_void_p))
        gam = None
    t0 = log.timer_debug1('RI-K grad: three-centre pass', *t0)
    # S = sum Gamma (ij|P)^x (AO atoms + auxiliary atoms); the auxiliary atoms are mol's atoms
    # when auxmol comes from make_auxmol, else by coordinates
    S = gout[:natm].copy()
    if auxmol.natm == natm and numpy.allclose(auxmol.atom_coords(), mol.atom_coords()):
        S += gout[natm:]
    else:
        return None
    # metric: sum_PQ gamma_PQ dM_PQ/dR_A = -2 sum_{P in A} sum_Q gamma_PQ (nabla P|Q)
    dKf = dK.reshape(naux, -1)
    gamma = lib.dot(dKf, dKf.T)
    ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')                      # (3, naux, naux)
    v = numpy.einsum('xpq,pq->xp', ip1, gamma)
    ip1 = gamma = None
    Mx = numpy.zeros((natm, 3))
    for ia, (a0, a1) in enumerate(auxmol.aoslice_by_atom()[:, 2:]):
        Mx[ia] = -2 * v[:, a0:a1].sum(axis=1)
    log.timer_debug1('RI-K grad: metric', *t0)
    return -.25 * (2 * S - Mx)


def jk_grad_terms(mf, mol, dm, hyb, max_memory=None, verbose=None):
    '''(dJ, dE_K/dR (natm, 3) scaled by hyb) for a closed-shell density-fitted mean field mf (its
    mo_coeff / mo_occ, its with_df factor); None if not applicable.'''
    from socutils.hessian.rhf_fast import _df_auxmol
    if max_memory is None:
        max_memory = mf.max_memory
    mo_occ = numpy.asarray(mf.mo_occ)
    mo_coeff = numpy.asarray(mf.mo_coeff)
    if mo_occ.ndim != 1 or _load() is None or mol.cart:
        return None
    c = _occ_factor(mo_coeff, mo_occ)
    if abs(c.dot(c.T) - dm).max() > 1e-8:
        return None
    auxmol = _df_auxmol(mf)
    dJ, dK = fitted_coefficients(mol, auxmol, dm, c, with_df=mf.with_df, max_memory=max_memory, verbose=verbose)
    g = k_grad(mol, auxmol, c, dK, max_memory=max_memory, verbose=verbose)
    if g is None:
        return None
    return dJ, hyb * g
