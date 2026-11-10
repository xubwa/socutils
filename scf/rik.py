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
    (taken F-ordered) or complex (real and imaginary parts separately: the real symmetric B^P);
    out: an optional buffer of >= nb ncol nao doubles (real c only)'''
    if numpy.iscomplexobj(c):
        return half_transform(eri1, c.real) + 1j * half_transform(eri1, c.imag)
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
    '''(nb, n1, n2) (P|pq) = sum_ij c1_ip^* B^P_ij c2_jq of a packed factor block (nb, npair); c1, c2
    real or complex (spinor coefficients of one spin component at a time)'''
    if c2 is None:
        c2 = c1
    nb = eri1.shape[0]
    x = half_transform(eri1, numpy.conj(c1))                      # (nb, n1, nao)
    if numpy.iscomplexobj(x) or numpy.iscomplexobj(c2):
        r = numpy.dot(x.reshape(-1, x.shape[2]), c2)
        if out is not None:
            raise ValueError('out: real coefficients only')
        return r.reshape(nb, c1.shape[1], c2.shape[1])
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
    return libv if libv and hasattr(libv, 'SOCVHFscatter_cols') else None


class _Int3c:
    '''(ij|P) of the Schwarz-significant shell pairs of mol (ish >= jsh) against all shells of
    auxmol (lib/vhf/int3c_s2.c), by ranges of the i shell (contiguous pack_tril columns)'''

    def __init__(self, mol, auxmol, pair_tol=PAIR_TOL, j2c=None):
        from pyscf import gto
        from pyscf.gto import moleintor
        from socutils.hessian import ghf as ghf_hess
        self.libv = _load_socvhf()
        self.mol, self.auxmol = mol, auxmol
        nbas = mol.nbas
        if j2c is None:
            j2c = auxmol.intor('int2c2e', hermi=1)
        q = ghf_hess._q_cond(mol, 'int2e_ip1')[1]                # sqrt((ij|ij)) per shell pair
        qP = numpy.sqrt(abs(numpy.diag(j2c)).max())
        ao_loc = self.ao_loc = mol.ao_loc_nr()
        ii, jj = numpy.tril_indices(nbas)
        sig = q[ii, jj] * qP >= pair_tol
        self.ii, self.jj = ii[sig], jj[sig]
        self.atm, self.bas, self.env = gto.conc_env(mol._atm, mol._bas, mol._env,
                                                    auxmol._atm, auxmol._bas, auxmol._env)
        self.loc = moleintor.make_loc(self.bas, 'int3c2e_sph').astype(numpy.int32)
        self.opt = moleintor.make_cintopt(self.atm, self.bas, self.env, 'int3c2e_sph')
        self.nrow_sig = 0

    def ish_ranges(self, max_cols):
        '''[(s0, s1)] i-shell ranges with at most max_cols pack_tril columns each (>= one shell)'''
        ao_loc = self.ao_loc
        out = []
        s0 = 0
        nbas = self.mol.nbas
        while s0 < nbas:
            s1 = s0 + 1
            c0 = ao_loc[s0] * (ao_loc[s0] + 1) // 2
            while s1 < nbas and ao_loc[s1 + 1] * (ao_loc[s1 + 1] + 1) // 2 - c0 <= max_cols:
                s1 += 1
            out.append((s0, s1))
            s0 = s1
        return out

    def rows(self, s0, s1):
        '''(X (nrow, naux) of the significant pairs with s0 <= ish < s1, their pack_tril columns)'''
        ao_loc = self.ao_loc
        sel = (self.ii >= s0) & (self.ii < s1)
        ii, jj = self.ii[sel], self.jj[sel]
        di = ao_loc[ii + 1] - ao_loc[ii]
        dj = ao_loc[jj + 1] - ao_loc[jj]
        nf = numpy.where(ii == jj, di * (di + 1) // 2, di * dj)
        pair_row = numpy.zeros(len(ii), dtype=numpy.int64)
        pair_row[1:] = numpy.cumsum(nf)[:-1]
        nrow = int(nf.sum())
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
        X = numpy.empty((nrow, self.auxmol.nao))
        if nrow:
            self.libv.SOCVHFint3c2e_s2_pairs(
                X.ctypes.data_as(ctypes.c_void_p), pairs.ctypes.data_as(ctypes.c_void_p),
                pair_row.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(ii)), self.opt,
                self.loc.ctypes.data_as(ctypes.c_void_p), self.atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.mol.natm + self.auxmol.natm), self.bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.mol.nbas), ctypes.c_int(self.auxmol.nbas),
                self.env.ctypes.data_as(ctypes.c_void_p))
        self.nrow_sig += nrow
        return X, cols


def _scatter_cols(B, Y, cols):
    '''B[:, cols] = Y for B (naux, ncol) C-ordered and Y (naux, nrow) F-ordered (the solve's output)'''
    Yc = Y.T
    assert Yc.flags.c_contiguous and B.flags.c_contiguous
    cols = numpy.ascontiguousarray(cols, dtype=numpy.int64)
    _load_socvhf().SOCVHFscatter_cols(
        B.ctypes.data_as(ctypes.c_void_p), Yc.ctypes.data_as(ctypes.c_void_p),
        cols.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(B.shape[0]), ctypes.c_long(Yc.shape[0]),
        ctypes.c_long(B.shape[1]))
    return B


def _metric_cholesky(auxmol):
    j2c = auxmol.intor('int2c2e', hermi=1)
    try:
        return j2c, scipy.linalg.cholesky(j2c, lower=True)
    except scipy.linalg.LinAlgError:
        return j2c, None


def cholesky_eri(mol, auxmol, pair_tol=PAIR_TOL, verbose=None):
    '''B = L^{-1} (P|ij) (naux, npair), M = (P|Q) = L L^T, as pyscf's df.incore.cholesky_eri
    (aosym s2ij) for any molecule / auxiliary basis (a sub-molecule's shells included), the
    three-centre integrals of the Schwarz-significant shell pairs only
    (lib/vhf/int3c_s2.c) and the triangular solve over those pair columns.  None if the
    library is unavailable or M is not positive definite (pyscf's eigh route then).'''
    if _load_socvhf() is None or mol.cart or auxmol.cart:
        return None
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    j2c, low = _metric_cholesky(auxmol)
    if low is None:
        return None
    eng = _Int3c(mol, auxmol, pair_tol, j2c)
    j2c = None
    nao = mol.nao
    npair = nao * (nao + 1) // 2
    X, cols = eng.rows(0, mol.nbas)
    nrow = X.shape[0]
    t0 = log.timer_debug1('rik.cholesky_eri: (ij|P) of %d / %d pairs' % (nrow, npair), *t0)
    Y = scipy.linalg.solve_triangular(low, X.T, lower=True, overwrite_b=True, check_finite=False)
    X = None
    t0 = log.timer_debug1('rik.cholesky_eri: L^-1', *t0)
    B = numpy.zeros((auxmol.nao, npair))
    _scatter_cols(B, Y, cols)
    log.timer_debug1('rik.cholesky_eri: scatter', *t0)
    return B


def cholesky_eri_h5(mol, auxmol, filename, dataname='j3c', pair_tol=PAIR_TOL, max_memory=4000,
                    verbose=None, max_cols=None, resident=None):
    '''cholesky_eri written to the HDF5 file filename as one contiguous (naux, npair) dataset
    (pyscf's layout, read by df.DF.loop; rik.iter_factor reads it directly), one i-shell range of
    pack_tril columns in memory at a time.  resident: an optional (nres, npair) array that receives
    the first nres rows too (kept in memory by the caller: iter_factor reads only the rest).
    False if not applicable.'''
    import h5py
    if _load_socvhf() is None or mol.cart or auxmol.cart:
        return False
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    j2c, low = _metric_cholesky(auxmol)
    if low is None:
        return False
    eng = _Int3c(mol, auxmol, pair_tol, j2c)
    j2c = None
    naux = auxmol.nao
    mem = max(500., max_memory - lib.current_memory()[0])
    # per column: X (and its solve in place) and the dense block written
    if max_cols is None:
        max_cols = max(1, int(mem * .4e6 / 8 / naux / 2))
    ao_loc = mol.ao_loc_nr()
    nao = mol.nao
    npair = nao * (nao + 1) // 2
    with h5py.File(filename, 'a') as f:
        if dataname in f:
            del f[dataname]
        # one contiguous (naux, npair) dataset: a block of auxiliary rows is one contiguous read
        # (iter_factor reads it with os.preadv, outside the GIL, prefetched)
        dset = f.create_dataset(dataname, (naux, npair), 'f8')
        for k, (s0, s1) in enumerate(eng.ish_ranges(max_cols)):
            c0 = ao_loc[s0] * (ao_loc[s0] + 1) // 2
            c1 = ao_loc[s1] * (ao_loc[s1] + 1) // 2
            X, cols = eng.rows(s0, s1)
            Y = scipy.linalg.solve_triangular(low, X.T, lower=True, overwrite_b=True, check_finite=False)
            X = None
            blk = numpy.zeros((naux, c1 - c0))
            _scatter_cols(blk, Y, cols - c0)
            Y = None
            dset[:, c0:c1] = blk
            if resident is not None and len(resident):
                resident[:, c0:c1] = blk[:len(resident)]
            blk = None
            t0 = log.timer_debug1('rik.cholesky_eri_h5: block %d (shells %d:%d)' % (k, s0, s1), *t0)
    return True


def _h5_contiguous(with_df):
    '''(path, byte offset, naux, npair) of with_df's factor when it is a contiguous, unfiltered
    float64 HDF5 dataset (raw-readable), else None'''
    import h5py
    cderi = getattr(with_df, '_cderi', None)
    if not isinstance(cderi, str):
        return None
    try:
        with h5py.File(cderi, 'r') as f:
            d = f[getattr(with_df, '_dataname', 'j3c')]
            if (not isinstance(d, h5py.Dataset) or d.chunks is not None or d.compression is not None
                    or d.dtype != numpy.float64 or d.ndim != 2):
                return None
            off = d.id.get_offset()
            if off is None:
                return None
            return cderi, off, d.shape[0], d.shape[1]
    except (OSError, KeyError):
        return None


def iter_factor(with_df, blksize):
    '''blocks (nb, npair) of the factor, as with_df.loop(blksize); a contiguous HDF5 dataset is read
    with os.preadv (READ_THREADS parallel requests) in a background thread (the GIL released
    during the read, so the next block is read while this one is used; h5py holds it),
    double-buffered.  The rows kept in memory by rik's builder (with_df._rik_resident) come first,
    read meanwhile the first disk block.'''
    src = _h5_contiguous(with_df) if RAW_READ else None
    if src is None:
        for blk in with_df.loop(blksize):
            yield blk
        return
    import os
    import threading
    path, off, naux, npair = src
    res_rows = getattr(with_df, '_rik_resident', None)
    if res_rows is None or getattr(with_df, '_rik_resident_src', None) != path or res_rows.shape[1] != npair:
        res_rows = numpy.empty((0, npair))
    nres = len(res_rows)
    ranges = [(p0, min(naux, p0 + blksize)) for p0 in range(nres, naux, blksize)]
    bufs = [numpy.empty((min(blksize, naux - nres), npair)) for _ in range(min(2, len(ranges)))]
    fd = os.open(path, os.O_RDONLY)

    def read_part(view, pos):
        n = 0
        while n < view.nbytes:
            r = os.preadv(fd, [view[n:]], pos + n)
            if r <= 0:
                raise IOError('short read of %s' % path)
            n += r

    def read(k):
        p0, p1 = ranges[k]
        b = bufs[k % 2][:p1 - p0]
        view = memoryview(b.reshape(-1)).cast('B')
        pos = off + p0 * npair * 8
        nt = max(1, READ_THREADS)
        step = -(-view.nbytes // nt)
        ths = [threading.Thread(target=read_part, args=(view[i:i + step], pos + i))
               for i in range(0, view.nbytes, step)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        return b
    th = None
    try:
        res = {}

        def job(k):
            res[k] = read(k)
        if ranges:
            th = threading.Thread(target=job, args=(0,))
            th.start()
        for p0 in range(0, nres, blksize):
            yield res_rows[p0:p0 + blksize]
        for k in range(len(ranges)):
            th.join()
            blk = res.pop(k)
            if k + 1 < len(ranges):
                th = threading.Thread(target=job, args=(k + 1,))
                th.start()
            yield blk
    finally:
        if th is not None:
            th.join()
        os.close(fd)


READ_THREADS = 4      # parallel os.preadv requests per factor block (NVMe queue depth)
RAW_READ = True       # iter_factor: contiguous HDF5 factors by os.preadv (else with_df.loop)


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
    tw = [0.] * 4          # wall: waiting for the factor, J, half transformation, rank-k update
    tl = logger.perf_counter()
    for eri1 in iter_factor(with_df, blksize):
        t1 = logger.perf_counter()
        tw[0] += t1 - tl
        nb = eri1.shape[0]
        assert eri1.shape[1] == npair
        if with_j:
            rho = lib.dot(dmtril, eri1.T)                       # (nset, nb)
            vj += lib.dot(rho, eri1)
        t2 = logger.perf_counter()
        tw[1] += t2 - t1
        if with_k:
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
                    t3 = logger.perf_counter()
                    x = _half_trans(eri1, c, buf)                     # (nb, ncol, nao)
                    t4 = logger.perf_counter()
                    vkF[k] = _dsyrk_acc(x.reshape(-1, nao), vkF[k], sgn)
                    tw[2] += t4 - t3
                    tw[3] += logger.perf_counter() - t4
        eri1 = None
        tl = logger.perf_counter()
    log.debug1('rik.get_jk wall: factor read %.2f s, J %.2f s, half transformation %.2f s, '
               'rank-k update %.2f s', *tw)

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
FAST_BUILD_H5 = True  # ... and by rik.cholesky_eri_h5 into a temporary HDF5 file when they do not
RESIDENT_FRAC = .4    # ... with the first rows of the out-of-core factor kept in memory up to this * max_memory


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
    t0 = (logger.process_clock(), logger.perf_counter())
    if need < .8 * (mf.max_memory - lib.current_memory()[0]):
        B = cholesky_eri(mol, auxmol, verbose=mf.verbose)
        if B is not None:
            with_df._cderi = B
            logger.timer(mf, 'rik: L^-1 (P|ij) (screened pairs)', *t0)
        return
    if not FAST_BUILD_H5:
        return
    # out of core: an HDF5 temporary file in pyscf's TMPDIR, as pyscf's own DF.build
    tmp = lib.NamedTemporaryFile(dir=lib.param.TMPDIR)
    # the first rows stay in memory too (RESIDENT_FRAC of max_memory): fewer bytes read per pass
    nres = int(max(0, min(auxmol.nao, (RESIDENT_FRAC * mf.max_memory - lib.current_memory()[0]) * 1e6 / 8 / npair)))
    resident = numpy.empty((nres, npair))
    if cholesky_eri_h5(mol, auxmol, tmp.name, with_df._dataname, max_memory=mf.max_memory - nres * npair * 8e-6,
                       verbose=mf.verbose, resident=resident):
        with_df._cderi_to_save = tmp        # keeps the temporary file alive with the DF object
        with_df._cderi = tmp.name
        with_df._rik_resident = resident
        with_df._rik_resident_src = tmp.name
        logger.info(mf, 'rik: %d of %d factor rows resident (%.1f GB), the rest read from %s', nres, auxmol.nao,
                    resident.nbytes / 1e9, tmp.name)
        logger.timer(mf, 'rik: L^-1 (P|ij) (screened pairs) to %s' % tmp.name, *t0)


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
