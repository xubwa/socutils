#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
RI-JK (density-fitted or Cholesky-decomposed J and K) from a three-index
factor B^P_ij, (ij|kl) ~ sum_P B^P_ij B^P_kl, read block by block through
``with_df.loop(blksize)`` -- the in-core array or the HDF5 file of a pyscf
``df.DF`` object, or the Cholesky vectors of ``socutils.cd.cd.CD``, alike:

    J_ij = sum_P B^P_ij (sum_kl B^P_kl D_kl)
    K_ij = sum_P sum_m X^P_im X^P_jm,   X^P = B^P C_o sqrt(n_o)

K from the occupied orbitals when the density carries them (pyscf's
``make_rdm1`` tags ``mo_coeff`` / ``mo_occ``), else from the eigenvectors of
the symmetric density (its rank is nocc for an SCF density; positive and
negative parts separately), else (non-symmetric or nearly full-rank) from
the density itself, K = sum_P B^P D B^P.  The half transformation
B^P C (pyscf's ``AO2MOnr_e2_drv`` on the packed lower triangle) and the
rank-k update K += X^T X (BLAS dsyrk, half the flops of the GEMM) per block.

    from socutils.scf import rik
    mf = rik.apply(scf.RKS(mol).density_fit(auxbasis='cc-pvdz-jkfit'))
    mf = rik.apply(socutils.cd.cd.cholesky(scf.RHF(mol)))
'''

import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

BLK_MEMORY = .4       # fraction of the free memory for the block buffers
RANK_TOL = 1e-12      # eigenvalues of a symmetric density below RANK_TOL * max |lambda| dropped
MAX_RANK_FRAC = .5    # factorised K only while the density's rank <= this * nao


def _dsyrk_acc(a, c, alpha=1.):
    '''c (F-ordered, upper triangle) += alpha a^T a for a C-ordered (m, n) array a'''
    if a.shape[0] == 0:
        return c
    # a^T as an (n, m) F-ordered matrix: c += alpha a^T (a^T)^T
    return scipy.linalg.blas.dsyrk(alpha, a.T, beta=1., c=c, trans=0, lower=0, overwrite_c=1)


def _orbital_factors(dm, hermi):
    '''list per density of (X+, X-) with D = X+ X+^T - X- X-^T, or None for densities that
    keep the density route.  dm (nset, nao, nao); mo_coeff / mo_occ tags first.'''
    nset, nao = dm.shape[:2]
    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm, 'mo_occ', None)
    if mo_coeff is not None and mo_occ is not None:
        mo_coeff = numpy.asarray(mo_coeff).reshape(-1, nao, numpy.shape(mo_occ)[-1])
        mo_occ = numpy.asarray(mo_occ).reshape(-1, mo_coeff.shape[-1])
        if mo_coeff.shape[0] == nset and (mo_occ >= 0).all():
            out = []
            for k in range(nset):
                sel = mo_occ[k] > 0
                c = numpy.asarray(mo_coeff[k][:, sel] * numpy.sqrt(mo_occ[k][sel]), order='F')
                out.append((c, None))
            return out
    if hermi != 1:
        return [None] * nset
    out = []
    for k in range(nset):
        d = numpy.asarray(dm[k])
        if abs(d - d.T).max() > 1e-12 * max(1., abs(d).max()):
            out.append(None)
            continue
        w, v = scipy.linalg.eigh(d)
        cut = RANK_TOL * max(abs(w).max(), 1e-300)
        pos, neg = w > cut, w < -cut
        if pos.sum() + neg.sum() > MAX_RANK_FRAC * nao:
            out.append(None)
            continue
        xp = numpy.asarray(v[:, pos] * numpy.sqrt(w[pos]), order='F')
        xn = numpy.asarray(v[:, neg] * numpy.sqrt(-w[neg]), order='F') if neg.any() else None
        out.append((xp, xn))
    return out


def half_transform(eri1, c, out=None):
    '''(nb, ncol, nao) X^P_{m i} = sum_j c_jm B^P_ij from the packed lower triangle eri1
    (nb, npair) of any three-index factor block (DF, CD, a sub-molecule's); c (nao, ncol) real
    (taken F-ordered); out: an optional buffer of >= nb ncol nao doubles'''
    eri1 = numpy.asarray(eri1, order='C')
    c = numpy.asarray(c, dtype=numpy.double, order='F')
    nb = eri1.shape[0]
    nao = c.shape[0]
    ncol = c.shape[1]
    assert eri1.shape[1] == nao * (nao + 1) // 2
    buf = numpy.ndarray((nb, ncol, nao), buffer=out)
    _ao2mo.libao2mo.AO2MOnr_e2_drv(
        _ao2mo.libao2mo.AO2MOtranse2_nr_s2, _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2,
        buf.ctypes.data_as(ctypes.c_void_p), eri1.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nb), ctypes.c_int(nao),
        (ctypes.c_int * 4)(0, ncol, 0, nao), lib.c_null_ptr(), ctypes.c_int(0))
    return buf


_half_trans = half_transform


def mo_transform(eri1, c1, c2=None, out=None):
    '''(nb, n1, n2) (P|pq) = sum_ij c1_ip B^P_ij c2_jq of a packed factor block (nb, npair)'''
    if c2 is None:
        c2 = c1
    nb = eri1.shape[0]
    x = half_transform(eri1, c1)                                  # (nb, n1, nao)
    r = lib.dot(x.reshape(-1, x.shape[2]), numpy.asarray(c2, order='C'))
    if out is not None:
        out = numpy.ndarray((nb, c1.shape[1], c2.shape[1]), buffer=out)
        out[:] = r.reshape(out.shape)
        return out
    return r.reshape(nb, c1.shape[1], c2.shape[1])


PAIR_TOL = 1e-13      # cholesky_eri: shell pairs with sqrt((ij|ij)) max sqrt((P|P)) below this are zero


def _load_socvhf():
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    return libv if libv and hasattr(libv, 'SOCVHFint3c2e_s2_pairs') else None


def cholesky_eri(mol, auxmol, pair_tol=PAIR_TOL, verbose=None):
    '''B = L^{-1} (P|ij) (naux, npair), M = (P|Q) = L L^T, as pyscf's df.incore.cholesky_eri
    (aosym s2ij) for any molecule / auxiliary basis (a sub-molecule's shells included), the
    three-centre integrals of the Schwarz-significant shell pairs only
    (lib/vhf/int3c_s2.c) and the triangular solve over those pair columns.  None if the
    library is unavailable or M is not positive definite (pyscf's eigh route then).'''
    from pyscf import gto
    from pyscf.gto import moleintor
    from socutils.hessian import ghf as ghf_hess
    libv = _load_socvhf()
    if libv is None or mol.cart or auxmol.cart:
        return None
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    j2c = auxmol.intor('int2c2e', hermi=1)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
    except scipy.linalg.LinAlgError:
        return None
    nao, nbas = mol.nao, mol.nbas
    naux = auxmol.nao
    npair = nao * (nao + 1) // 2
    q = ghf_hess._q_cond(mol, 'int2e_ip1')[1]                    # sqrt((ij|ij)) per shell pair
    qP = numpy.sqrt(abs(numpy.diag(j2c)).max())
    j2c = None
    ao_loc = mol.ao_loc_nr()
    ii, jj = numpy.tril_indices(nbas)
    sig = q[ii, jj] * qP >= pair_tol
    ii, jj = ii[sig], jj[sig]
    di = ao_loc[ii + 1] - ao_loc[ii]
    dj = ao_loc[jj + 1] - ao_loc[jj]
    nf = numpy.where(ii == jj, di * (di + 1) // 2, di * dj)
    pair_row = numpy.zeros(len(ii), dtype=numpy.int64)
    pair_row[1:] = numpy.cumsum(nf)[:-1]
    nrow = int(nf.sum())
    # pack_tril column of every row
    cols = numpy.empty(nrow, dtype=numpy.int64)
    for k in range(len(ii)):
        i0, j0 = ao_loc[ii[k]], ao_loc[jj[k]]
        if ii[k] == jj[k]:
            a, b = numpy.tril_indices(di[k])
        else:
            a, b = numpy.divmod(numpy.arange(di[k] * dj[k]), dj[k])
        i = i0 + a
        cols[pair_row[k]:pair_row[k] + nf[k]] = i * (i + 1) // 2 + j0 + b
    pairs = numpy.ascontiguousarray(numpy.stack([ii, jj], axis=1), dtype=numpy.int32)
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
    loc = moleintor.make_loc(bas, 'int3c2e_sph').astype(numpy.int32)
    opt = moleintor.make_cintopt(atm, bas, env, 'int3c2e_sph')
    X = numpy.empty((nrow, naux))
    libv.SOCVHFint3c2e_s2_pairs(
        X.ctypes.data_as(ctypes.c_void_p), pairs.ctypes.data_as(ctypes.c_void_p),
        pair_row.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(ii)), opt,
        loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm + auxmol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        ctypes.c_int(auxmol.nbas), env.ctypes.data_as(ctypes.c_void_p))
    t0 = log.timer_debug1('rik.cholesky_eri: (ij|P) of %d / %d pairs' % (nrow, npair), *t0)
    Y = scipy.linalg.solve_triangular(low, X.T, lower=True, overwrite_b=True, check_finite=False)
    X = None
    t0 = log.timer_debug1('rik.cholesky_eri: L^-1', *t0)
    if nrow == npair and (numpy.diff(cols) == 1).all():
        B = numpy.ascontiguousarray(Y)
    else:
        B = numpy.zeros((naux, npair))
        B[:, cols] = Y
    log.timer_debug1('rik.cholesky_eri: scatter', *t0)
    return B


def get_jk(with_df, dm, hermi=1, with_j=True, with_k=True, max_memory=None, verbose=None):
    '''J and K of one or several real densities from the three-index factor of with_df
    (any object with pyscf's DF loop(blksize) / get_naoaux(), packed lower-triangle pairs).
    One pass over the factor for J and K together.'''
    assert with_j or with_k
    log = logger.new_logger(with_df, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    dms = numpy.asarray(dm)
    shape = dms.shape
    nao = shape[-1]
    if getattr(with_df, '_cderi', None) is None:
        with_df.build()
    tags = {k: getattr(dm, k) for k in ('mo_coeff', 'mo_occ') if getattr(dm, k, None) is not None}
    dms = dms.reshape(-1, nao, nao)
    if tags:
        dms = lib.tag_array(dms, **tags)
    nset = dms.shape[0]
    naux = with_df.get_naoaux()
    if max_memory is None:
        max_memory = getattr(with_df, 'max_memory', 4000)
    mem_free = max(400., max_memory - lib.current_memory()[0])

    facs = _orbital_factors(dms, hermi) if with_k else [None] * nset
    ncol_max = 0
    for f in facs:
        if f is None:
            ncol_max = max(ncol_max, nao)
        else:
            ncol_max = max(ncol_max, f[0].shape[1], 0 if f[1] is None else f[1].shape[1])
    npair = nao * (nao + 1) // 2
    # per auxiliary function: the factor block (an h5 read and its prefetch: 2 npair), the
    # half-transformed buffer (ncol nao), the unpacked block for the density route (nao^2)
    per_p = 8 * (2 * npair + (ncol_max * nao if with_k else 0)
                 + (nao * nao if with_k and any(f is None for f in facs) else 0)) / 1e6
    blksize = int(max(8, min(naux, mem_free * BLK_MEMORY / max(per_p, 1e-9))))
    if with_k:
        # the dsyrk / GEMM efficiency wants >= a few hundred columns per block, but not huge
        blksize = min(blksize, max(64, int(2e8 / max(1, ncol_max * nao))))
    log.debug1('rik.get_jk: naux %d, blksize %d, factor columns %s', naux, blksize,
               [None if f is None else (f[0].shape[1], 0 if f[1] is None else f[1].shape[1]) for f in facs])

    if with_j:
        d = dms + dms.transpose(0, 2, 1)
        idx = numpy.arange(nao)
        d[:, idx, idx] *= .5
        # sum_kl B_kl D_kl over the packed pairs k >= l: D + D^T, the diagonal once
        dmtril = lib.pack_tril(d)
        vj = numpy.zeros((nset, npair))
    vk = numpy.zeros((nset, nao, nao), order='C') if with_k else None
    vkF = [numpy.zeros((nao, nao), order='F') for _ in range(nset)] if with_k else None
    buf = numpy.empty(blksize * ncol_max * nao) if with_k else None
    buf2 = numpy.empty((blksize, nao, nao)) if with_k and any(f is None for f in facs) else None
    dmsF = [None if f is not None else numpy.asarray(dms[k], order='F') for k, f in enumerate(facs)] \
        if with_k else None
    for eri1 in with_df.loop(blksize):
        nb = eri1.shape[0]
        assert eri1.shape[1] == npair
        if with_j:
            rho = lib.dot(dmtril, eri1.T)                       # (nset, nb)
            vj += lib.dot(rho, eri1)
        if not with_k:
            continue
        for k in range(nset):
            f = facs[k]
            if f is None:
                # density route: (B^P D)^T B^P summed over P as one GEMM
                bd = _half_trans(eri1, dmsF[k], buf)            # (nb, nao, nao): (D^T B^P)_{m i}
                b2 = lib.unpack_tril(eri1, out=buf2[:nb])
                lib.dot(bd.reshape(-1, nao).T, b2.reshape(-1, nao), 1., vk[k], 1.)
                continue
            for sgn, c in ((1., f[0]), (-1., f[1])):
                if c is None or c.shape[1] == 0:
                    continue
                x = _half_trans(eri1, c, buf)                     # (nb, ncol, nao)
                vkF[k] = _dsyrk_acc(x.reshape(-1, nao), vkF[k], sgn)
        t0 = log.timer_debug1('rik.get_jk block', *t0)

    if with_j:
        vj = lib.unpack_tril(vj, 1).reshape(shape)
    if with_k:
        for k in range(nset):
            if facs[k] is not None:
                u = numpy.triu(vkF[k])
                vk[k] = u + numpy.triu(u, 1).T
        vk = vk.reshape(shape)
    return (vj if with_j else None), vk


FAST_BUILD = True     # pyscf df.DF factors built by rik.cholesky_eri when they fit in memory


def _prebuild(mf):
    '''a plain pyscf df.DF without its factor: B by cholesky_eri (screened pairs) in memory when
    B and the integrals fit, else left to pyscf's build (in-core or HDF5)'''
    from pyscf import df
    with_df = mf.with_df
    if not FAST_BUILD or type(with_df) is not df.DF or with_df._cderi is not None:
        return
    if isinstance(with_df._cderi_to_save, str):
        return              # the user asked for the HDF5 file: pyscf's build writes it
    mol = with_df.mol
    if with_df.auxmol is None:
        with_df.auxmol = df.addons.make_auxmol(mol, with_df.auxbasis)
    auxmol = with_df.auxmol
    nao = mol.nao
    npair = nao * (nao + 1) // 2
    need = 2 * auxmol.nao * npair * 8 / 1e6
    if need > .8 * (mf.max_memory - lib.current_memory()[0]):
        return
    t0 = (logger.process_clock(), logger.perf_counter())
    B = cholesky_eri(mol, auxmol, verbose=mf.verbose)
    if B is not None:
        with_df._cderi = B
        logger.timer(mf, 'rik: L^-1 (P|ij) (screened pairs)', *t0)


def apply(mf):
    '''mf (a density-fitted / Cholesky-decomposed pyscf or socutils RHF / RKS / UHF / UKS with
    the J and K both from the factor: density_fit() without only_dfj, or socutils.cd.cholesky)
    with get_jk by rik.get_jk; range-separated calls and complex densities keep the original.
    Modifies and returns mf.'''
    from pyscf.df import df_jk
    if not isinstance(mf, df_jk._DFHF) or mf.only_dfj:
        return mf
    orig = mf.get_jk
    mol0 = mf.mol

    def mf_get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if mol is None:
            mol = mol0
        if dm is None:
            dm = mf.make_rdm1()
        if (omega is not None or mol is not mol0 or numpy.iscomplexobj(dm) or not mf.with_df
                or not (with_j or with_k)):
            return orig(mol, dm, hermi, with_j, with_k, omega)
        t0 = (logger.process_clock(), logger.perf_counter())
        _prebuild(mf)
        vj, vk = get_jk(mf.with_df, dm, hermi, with_j, with_k, max_memory=mf.max_memory)
        logger.timer_debug1(mf, 'vj, vk (rik)', *t0)
        return vj, vk

    mf.get_jk = mf_get_jk
    return mf
