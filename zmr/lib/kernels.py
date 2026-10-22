'''
OpenMP kernels of libzmr with numpy fallbacks (the fallbacks are also the
reference implementations the kernels are tested against).

    apply_op        batched single creation / annihilation operator
    apply_strings   batched strings of creation / annihilation operators
    weighted_gram   a[I,J] = -sum_{k,e} conj(u[I,k,e]) u[J,k,e] / (lam_k + delta_e - s_J)
    contract_t      the same for u = t v^T, without storing u
    contract_w      the same, summing the external labels first
    build_hmat      dense sector Hamiltonian from (h1, eri)
    fold_density    folded two-component transition densities (ao2mo)
    build_hmat_ab, apply_strings_ab   the same for spin-restricted sectors
    sigma_ab                          H_act on vectors of a spin-restricted sector
    components      connected components of the non-zero pattern of a matrix
    block_support   which components the rows of a matrix are supported on
'''

import ctypes

import numpy

from socutils.zmr import lib as _lib
from socutils.zmr.active import strings as _strings

_c16 = numpy.complex128


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def native():
    return _lib.zmr() is not None


def apply_op(target, sign, vecs, ndst):
    '''out[..., p, target[p, I]] = sign[p, I] * vecs[..., I] where
    target[p, I] >= 0; target, sign: (norb, nsrc) int64.'''
    vecs = numpy.asarray(vecs)
    norb, nsrc = target.shape
    lead = vecs.shape[:-1]
    z = _lib.zmr()
    if z is None or vecs.size == 0 or ndst == 0:
        out = numpy.zeros(lead + (norb, ndst), dtype=_c16)
        for p in range(norb):
            ok = target[p] >= 0
            out[..., p, target[p, ok]] = sign[p, ok] * vecs[..., ok]
        return out
    v = numpy.ascontiguousarray(vecs, dtype=_c16).reshape(-1, nsrc)
    target = numpy.ascontiguousarray(target, dtype=numpy.int64)
    sign = numpy.ascontiguousarray(sign, dtype=numpy.int64)
    out = numpy.empty((v.shape[0], norb, ndst), dtype=_c16)
    z.zmr_apply_op(_p(target), _p(sign), ctypes.c_int(norb), ctypes.c_int64(nsrc),
                   ctypes.c_int64(ndst), ctypes.c_int64(v.shape[0]), _p(v), _p(out))
    return out.reshape(lead + (norb, ndst))


def apply_strings(norb, nelec, daggers, ops, vecs, out=None, x0=0):
    '''out[v, x0 + x, :] = O_x vecs[v, :] for the operator strings
    O_x = c_1 ... c_n (applied right to left), c_k = a^+_{ops[x, k]} if
    daggers[k] else a_{ops[x, k]}.

    vecs: (nvec, dim(nelec)); ops: (nx, len(daggers)) orbital indices.
    out: (nvec, nx_total, dim of the target sector), allocated with
    nx_total = nx if not given; only its rows x0 .. x0 + nx - 1 are written.'''
    vecs = numpy.ascontiguousarray(vecs, dtype=_c16)
    ops = numpy.ascontiguousarray(ops, dtype=numpy.int64).reshape(len(ops), len(daggers))
    nvec = vecs.shape[0]
    nx = ops.shape[0]
    ntarget = nelec + sum(1 if d else -1 for d in daggers)
    dst = _strings.gen_strings(norb, ntarget)
    if out is None:
        out = numpy.empty((nvec, nx, len(dst)), dtype=_c16)
    if out.size == 0 or nx == 0:
        return out
    z = _lib.zmr()
    if z is None:
        src = _strings.gen_strings(norb, nelec)
        one = numpy.uint64(1)
        for x in range(nx):
            s = src.copy()
            sign = numpy.ones(len(src))
            ok = numpy.ones(len(src), dtype=bool)
            for k in reversed(range(len(daggers))):
                p = numpy.uint64(ops[x, k])
                occupied = (s >> p) & one != 0
                ok &= ~occupied if daggers[k] else occupied
                sign *= 1 - 2.0 * (numpy.bitwise_count(s >> (p + one)) & 1)
                s ^= one << p
            row = out[:, x0 + x]
            row[:] = 0
            row[:, numpy.searchsorted(dst, s[ok])] = sign[ok] * vecs[:, ok]
        return out
    dag = numpy.ascontiguousarray([1 if d else 0 for d in daggers], dtype=numpy.intc)
    z.zmr_apply_strings(ctypes.c_int(norb), ctypes.c_int(nelec), ctypes.c_int(len(daggers)),
                        _p(dag), _p(ops), ctypes.c_int64(nx), ctypes.c_int64(out.shape[1]),
                        ctypes.c_int64(x0), ctypes.c_int64(nvec), _p(vecs), _p(out))
    return out


def _real(*arrays):
    return not any(numpy.iscomplexobj(a) for a in arrays)


def _energies(lam, delta, shifts):
    return [numpy.ascontiguousarray(x, dtype=float) for x in (lam, delta, shifts)]


def _w_args(ev, ec, d):
    return [numpy.ascontiguousarray(x, dtype=float) for x in (ev, ec, d)]


def _rows(b):
    '''(array, leading dimension in elements) of three-index integrals
    b[L, x, y] for the kernels: a slice b = B[:, x0:x1, :] of a contiguous
    array is passed as it is (rows L are contiguous, with a gap between
    them); anything else is copied.'''
    n1, n2 = b.shape[1:]
    if not (b.dtype == _c16 and b.strides[1:] == (16 * n2, 16) and b.strides[0] % 16 == 0
            and b.strides[0] >= 16 * n1 * n2):
        b = numpy.ascontiguousarray(b, dtype=_c16)
    return b, b.strides[0] // 16


def w_ccvv(bvc, ev, ec, d, mode=-1):
    '''W[m] = sum_{i<j, a<b} |(ai|bj) - (aj|bi)|^2 / (d[m] + e_a + e_b - e_i - e_j)
    from bvc[L, a, i]; None without libzmr.  mode: 1 one thread per pair
    (i, j), 0 threads inside a pair, -1 chosen from the number of pairs.'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_w_ccvv'):
        return None
    naux, nv, nc = bvc.shape
    bvc, ld = _rows(bvc)
    ev, ec, d = _w_args(ev, ec, d)
    w = numpy.empty(len(d))
    z.zmr_w_ccvv(_p(bvc), ctypes.c_int64(ld), ctypes.c_int64(naux), ctypes.c_int64(nv),
                 ctypes.c_int64(nc), _p(ev), _p(ec), _p(d), ctypes.c_int64(len(d)),
                 ctypes.c_int(mode), _p(w))
    return w


def w_cavv(bvc, bva, ev, ec, d, mode=-1):
    '''W[m, t, u] = sum_{i, a<b} conj(v[ab, t]) v[ab, u] / (d[m] + e_a + e_b - e_i),
    v[ab, t] = -((ai|bt) - (at|bi)), from bvc[L, a, i] and bva[L, b, t].'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_w_cavv'):
        return None
    naux, nv, nc = bvc.shape
    na = bva.shape[2]
    bvc, ldvc = _rows(bvc)
    bva, ldva = _rows(bva)
    ev, ec, d = _w_args(ev, ec, d)
    w = numpy.empty((len(d), na, na), dtype=_c16)
    z.zmr_w_cavv(_p(bvc), ctypes.c_int64(ldvc), _p(bva), ctypes.c_int64(ldva),
                 ctypes.c_int64(naux), ctypes.c_int64(nv), ctypes.c_int64(na),
                 ctypes.c_int64(nc), _p(ev), _p(ec), _p(d), ctypes.c_int64(len(d)),
                 ctypes.c_int(mode), _p(w))
    return w


def w_aavv(bva, ev, d):
    '''W[m, tu, t'u'] = sum_{a<b} conj(v[ab, tu]) v[ab, t'u'] / (d[m] + e_a + e_b),
    v[ab, tu] = (at|bu) - (au|bt), t < u, from bva[L, b, t]; tasks are
    segments of a, the sum over b is a rank-k update.'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_w_aavv'):
        return None
    naux, nv, na = bva.shape
    bva, ld = _rows(bva)
    ev = numpy.ascontiguousarray(ev, dtype=float)
    d = numpy.ascontiguousarray(d, dtype=float)
    npair = na * (na - 1) // 2
    w = numpy.empty((len(d), npair, npair), dtype=_c16)
    z.zmr_w_aavv(_p(bva), ctypes.c_int64(ld), ctypes.c_int64(naux), ctypes.c_int64(nv),
                 ctypes.c_int64(na), _p(ev), _p(d), ctypes.c_int64(len(d)), _p(w))
    return w


def _rows_r(b):
    '''As _rows for real three-index integrals.'''
    n1, n2 = b.shape[1:]
    if not (b.dtype == numpy.float64 and b.strides[1:] == (8 * n2, 8) and b.strides[0] % 8 == 0
            and b.strides[0] >= 8 * n1 * n2):
        b = numpy.ascontiguousarray(b, dtype=float)
    return b, b.strides[0] // 8


def wsf_ccvv(bvc, ev, ec, d, mode=-1):
    '''Spin-free ccvv (zmr.nr.classes.CCVV): W[m] = sum over its labels of
    v^2 / (d[m] + e_a + e_b - e_i - e_j) from real bvc[L, a, i].'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_wsf_ccvv'):
        return None
    naux, nv, nc = bvc.shape
    bvc, ld = _rows_r(bvc)
    ev, ec, d = _w_args(ev, ec, d)
    w = numpy.empty(len(d))
    z.zmr_wsf_ccvv(_p(bvc), ctypes.c_int64(ld), ctypes.c_int64(naux), ctypes.c_int64(nv),
                   ctypes.c_int64(nc), _p(ev), _p(ec), _p(d), ctypes.c_int64(len(d)),
                   ctypes.c_int(mode), _p(w))
    return w


def wsf_cavv(bvc, bva, ev, ec, d, mode=-1):
    '''Spin-free cavv, a < b: W[m, X, Y] for the parameters
    X = [(ai|bt), (bi|at)] (2 nact) from real bvc[L, a, i], bva[L, b, t].'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_wsf_cavv'):
        return None
    naux, nv, nc = bvc.shape
    na = bva.shape[2]
    bvc, ldvc = _rows_r(bvc)
    bva, ldva = _rows_r(bva)
    ev, ec, d = _w_args(ev, ec, d)
    w = numpy.empty((len(d), 2 * na, 2 * na))
    z.zmr_wsf_cavv(_p(bvc), ctypes.c_int64(ldvc), _p(bva), ctypes.c_int64(ldva),
                   ctypes.c_int64(naux), ctypes.c_int64(nv), ctypes.c_int64(na),
                   ctypes.c_int64(nc), _p(ev), _p(ec), _p(d), ctypes.c_int64(len(d)),
                   ctypes.c_int(mode), _p(w))
    return w


def wsf_aavv(bva, ev, d):
    '''Spin-free aavv, a < b: W[m, tu, t'u'] for the parameters (at|bu),
    all (t, u), from real bva[L, b, t].'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_wsf_aavv'):
        return None
    naux, nv, na = bva.shape
    bva, ld = _rows_r(bva)
    ev = numpy.ascontiguousarray(ev, dtype=float)
    d = numpy.ascontiguousarray(d, dtype=float)
    w = numpy.empty((len(d), na * na, na * na))
    z.zmr_wsf_aavv(_p(bva), ctypes.c_int64(ld), ctypes.c_int64(naux), ctypes.c_int64(nv),
                   ctypes.c_int64(na), _p(ev), _p(d), ctypes.c_int64(len(d)), _p(w))
    return w


def shifted_sums(w, delta, lam, shifts, mask, out):
    '''out[k, j] += sum_e w[e] / (lam[k] + delta[e] - shifts[j]) where mask[k, j].'''
    lam, delta, shifts = _energies(lam, delta, shifts)
    w = numpy.ascontiguousarray(w, dtype=float)
    mask = numpy.ascontiguousarray(mask, dtype=numpy.uint8)
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_shifted_sums'):
        for k, j in zip(*numpy.nonzero(mask)):
            out[k, j] += (w / (lam[k] + delta - shifts[j])).sum()
        return out
    z.zmr_shifted_sums(_p(w), _p(delta), ctypes.c_int64(len(delta)), _p(lam),
                       ctypes.c_int64(len(lam)), _p(shifts), ctypes.c_int64(len(shifts)),
                       _p(mask), _p(out))
    return out


def weighted_gram(u, lam, delta, shifts, col=None):
    '''a[I, J] = - sum_{k, e} conj(u[I, k, e]) u[J, k, e] / (lam[k] + delta[e] - shifts[J])
    for u (ns, nk, ne), real or complex.  col: compute only that column J
    (the others are zero).'''
    ns, nk, ne = u.shape
    lam, delta, shifts = _energies(lam, delta, shifts)
    real = _real(u)
    dtype = numpy.float64 if real else _c16
    z = _lib.zmr()
    if z is None:
        a = numpy.zeros((ns, ns), dtype=dtype)
        if nk == 0 or ne == 0:
            return a
        u2 = u.reshape(ns, nk * ne)
        for s in numpy.unique(shifts):
            js = numpy.nonzero(shifts == s)[0]
            if col is not None:
                js = js[js == col]
            if len(js) == 0:
                continue
            denom = (lam[:, None] + delta[None, :] - s).ravel()
            a[:, js] = -(u2.conj() @ (u2[js] / denom[None, :]).T)
        return a
    u = numpy.ascontiguousarray(u, dtype=dtype)
    a = numpy.empty((ns, ns), dtype=dtype)
    fn = z.zmr_weighted_gram_r if real else z.zmr_weighted_gram
    fn(_p(u), ctypes.c_int64(ns), ctypes.c_int64(nk), ctypes.c_int64(ne),
       _p(lam), _p(delta), _p(shifts), ctypes.c_int64(-1 if col is None else col), _p(a))
    return a


def _contract(native, t, v, lam, delta, shifts, col):
    ns, nk, nx = t.shape
    dtype = numpy.float64 if _real(t, v) else _c16
    t = numpy.ascontiguousarray(t, dtype=dtype)
    v = numpy.ascontiguousarray(v, dtype=dtype)
    a = numpy.empty((ns, ns), dtype=dtype)
    native(_p(t), ctypes.c_int64(ns), ctypes.c_int64(nk), ctypes.c_int64(nx),
           _p(v), ctypes.c_int64(v.shape[0]), _p(lam), _p(delta), _p(shifts),
           ctypes.c_int64(-1 if col is None else col), _p(a))
    return a


def contract_t(t, v, lam, delta, shifts, col=None):
    '''weighted_gram(u, lam, delta, shifts, col) for u[J, k, e] = sum_X t[J, k, X] v[e, X],
    t (ns, nk, nX), v (ne, nX).  The native kernel never stores u.'''
    ns, nk, nx = t.shape
    lam, delta, shifts = _energies(lam, delta, shifts)
    z = _lib.zmr()
    if z is None:
        u = (t.reshape(ns * nk, nx) @ v.T).reshape(ns, nk, v.shape[0])
        return weighted_gram(u, lam, delta, shifts, col)
    native = z.zmr_contract_t_r if _real(t, v) else z.zmr_contract_t
    return _contract(native, t, v, lam, delta, shifts, col)


def contract_w(t, v, lam, delta, shifts, col=None):
    '''The same matrix as contract_t with the external labels summed first:
    W_k(s)[X, Y] = sum_e conj(v[e, X]) v[e, Y] / (lam[k] + delta[e] - s),
    a[I, J] = - sum_k t[I, k, :]^+ W_k(s_J) t[J, k, :].  Its cost does not grow
    with the square of the number of states.'''
    ns, nk, nx = t.shape
    lam, delta, shifts = _energies(lam, delta, shifts)
    z = _lib.zmr()
    if z is None:
        a = numpy.zeros((ns, ns), dtype=numpy.result_type(t, v))
        for s in numpy.unique(shifts):
            js = numpy.nonzero(shifts == s)[0]
            if col is not None:
                js = js[js == col]
            if len(js) == 0:
                continue
            for k in range(nk):
                w = (v.conj() / (lam[k] + delta[:, None] - s)).T @ v      # [X, Y]
                a[:, js] -= t[:, k].conj() @ (w @ t[js, k].T)
        return a
    native = z.zmr_contract_w_r if _real(t, v) else z.zmr_contract_w
    return _contract(native, t, v, lam, delta, shifts, col)


def build_hmat(norb, nelec, dim, h1, eri):
    '''Dense matrix of sum h_pq p^+ q + 1/2 sum (pq|rs) p^+ r^+ s q over the
    nelec-electron strings of norb orbitals, or None without libzmr.'''
    z = _lib.zmr()
    if z is None or norb > 62:
        return None
    h1 = numpy.ascontiguousarray(h1, dtype=_c16)
    eri = numpy.ascontiguousarray(eri, dtype=_c16)
    hmat = numpy.empty((dim, dim), dtype=_c16)
    z.zmr_build_hmat(ctypes.c_int(norb), ctypes.c_int(nelec), ctypes.c_int64(dim),
                     _p(h1), _p(eri), _p(hmat))
    return hmat


def sigma(norb, nelec, dim, h1, eri, q, kernel='zmr_sigma'):
    '''H q for the columns of q (dim, nvec) without building H, or None
    without libzmr.  The operator is that of build_hmat for any complex h1
    and eri (no symmetry of the integrals is used).'''
    z = _lib.zmr()
    if z is None or norb > 62 or not hasattr(z, kernel):
        return None
    h1 = numpy.ascontiguousarray(h1, dtype=_c16)
    eri = numpy.ascontiguousarray(eri, dtype=_c16)
    q = numpy.ascontiguousarray(q, dtype=_c16)
    out = numpy.empty_like(q)
    if q.size:
        getattr(z, kernel)(ctypes.c_int(norb), ctypes.c_int(nelec), ctypes.c_int64(dim),
                           _p(h1), _p(eri), ctypes.c_int64(q.shape[1]), _p(q), _p(out))
    return out


def sigma_det(norb, nelec, dim, h1, eri, q):
    '''sigma by the determinant-driven kernel, the reference of the
    string-driven one; needs a Hermitian H.'''
    return sigma(norb, nelec, dim, h1, eri, q, kernel='zmr_sigma_det')


def fold_density(c1a, c1b, c2a, c2b):
    '''out[(i, j), (m >= n)] = sum_spin conj(c1[m, i]) c2[n, j]
    + [m != n] conj(c1[n, i]) c2[m, j] for alpha / beta components c*a, c*b
    of shape (nao, n*); the AO pair index is the packed lower triangle.'''
    nao, n1 = c1a.shape
    n2 = c2a.shape[1]
    npair = nao * (nao + 1) // 2
    z = _lib.zmr()
    if z is None or n1 * n2 == 0:
        d = (numpy.einsum('mi,nj->ijmn', c1a.conj(), c2a)
             + numpy.einsum('mi,nj->ijmn', c1b.conj(), c2b)).reshape(n1 * n2, nao, nao)
        m = d + d.transpose(0, 2, 1)
        ix = numpy.arange(nao)
        m[:, ix, ix] = d[:, ix, ix]
        il = numpy.tril_indices(nao)
        return numpy.ascontiguousarray(m[:, il[0], il[1]])
    args = [numpy.ascontiguousarray(x, dtype=_c16) for x in (c1a, c1b, c2a, c2b)]
    out = numpy.empty((n1 * n2, npair), dtype=_c16)
    z.zmr_fold_density(ctypes.c_int64(nao), ctypes.c_int64(n1), ctypes.c_int64(n2),
                       _p(args[0]), _p(args[1]), _p(args[2]), _p(args[3]), _p(out))
    return out


def components(a):
    '''Connected components of the graph of the non-zero elements of the
    square matrix a (either triangle may be empty).

    Returns (labels, ncomp): labels (n,) numbered in order of first
    appearance.'''
    n = a.shape[0]
    labels = numpy.full(n, -1, dtype=numpy.int64)
    z = _lib.zmr()
    if z is not None and n > 0:
        a = numpy.ascontiguousarray(a, dtype=_c16)
        z.zmr_components.restype = ctypes.c_int64
        ncomp = z.zmr_components(_p(a), ctypes.c_int64(n), _p(labels))
        return labels, int(ncomp)
    mask = a != 0
    mask |= mask.T
    ncomp = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        labels[start] = ncomp
        frontier = numpy.array([start])
        while len(frontier):
            reach = mask[frontier].any(axis=0)
            reach &= labels < 0
            frontier = numpy.nonzero(reach)[0]
            labels[frontier] = ncomp
        ncomp += 1
    return labels, ncomp


def block_support(rows, labels, ncomp):
    '''touch[x, c]: does row x of rows (m, d) have a non-zero element at an
    index with label c?  Returns a boolean array (m, ncomp).'''
    m, d = rows.shape
    labels = numpy.ascontiguousarray(labels, dtype=numpy.int64)
    z = _lib.zmr()
    if z is None or m * d == 0:
        touch = numpy.zeros((m, ncomp), dtype=bool)
        nz = rows != 0
        for c in range(ncomp):
            touch[:, c] = nz[:, labels == c].any(axis=1)
        return touch
    rows = numpy.ascontiguousarray(rows, dtype=_c16)
    touch = numpy.empty((m, ncomp), dtype=numpy.uint8)
    z.zmr_block_support(_p(rows), ctypes.c_int64(m), ctypes.c_int64(d), _p(labels),
                        ctypes.c_int64(ncomp), _p(touch))
    return touch.view(bool)


def build_hmat_ab(norb, nelec, h1, eri):
    '''Dense real matrix of the spin-free Hamiltonian (h1, eri over spatial
    orbitals) in the (n_alpha, n_beta) sector (zmr.nr.active), or None
    without libzmr.'''
    z = _lib.zmr()
    if z is None or 2 * norb > 62:
        return None
    na, nb = nelec
    import math
    dim = math.comb(norb, na) * math.comb(norb, nb)
    h1 = numpy.ascontiguousarray(h1, dtype=float)
    eri = numpy.ascontiguousarray(eri, dtype=float)
    hmat = numpy.empty((dim, dim))
    z.zmr_build_hmat_ab(ctypes.c_int(norb), ctypes.c_int(na), ctypes.c_int(nb),
                        _p(h1), _p(eri), _p(hmat))
    return hmat


def sigma_ab(norb, nelec, h1, eri, q, kernel='zmr_sigma_ab'):
    '''H q for real q (dim, nvec) of the sector nelec = (n_alpha, n_beta)
    without building H, or None without libzmr.  The operator is that of
    build_hmat_ab for any real h1 and eri (no symmetry is assumed; the pair
    symmetry (pq|rs) = (qp|rs) = (pq|sr) is detected and used).'''
    z = _lib.zmr()
    if z is None or 2 * norb > 62 or not hasattr(z, kernel):
        return None
    q = numpy.ascontiguousarray(q, dtype=float)
    h1 = numpy.ascontiguousarray(h1, dtype=float)
    eri = numpy.ascontiguousarray(eri, dtype=float)
    out = numpy.empty_like(q)
    if q.size:
        getattr(z, kernel)(ctypes.c_int(norb), ctypes.c_int(nelec[0]), ctypes.c_int(nelec[1]),
                           _p(h1), _p(eri), ctypes.c_int64(q.shape[1]), _p(q), _p(out))
    return out


def sigma_ab_det(norb, nelec, h1, eri, q):
    '''sigma_ab by the determinant-driven kernel (Slater-Condon rules), the
    reference of the string-driven one; needs (pq|rs) = (rs|pq) and a
    symmetric H.'''
    return sigma_ab(norb, nelec, h1, eri, q, kernel='zmr_sigma_ab_det')


def sigma_ab_pyscf(norb, nelec, h1, eri, q):
    '''sigma_ab through pyscf.fci.direct_nosym (one vector at a time), the
    path without libzmr.  Same determinant order and phases; any real h1 and
    eri.'''
    from pyscf.fci import cistring, direct_nosym
    q = numpy.asarray(q, dtype=float)
    out = numpy.empty(q.shape)
    if q.size == 0:
        return out
    if sum(nelec) == 0:
        out[:] = 0
        return out
    # H = sum k_pq E_pq + 1/2 sum (pq|rs) E_pq E_rs, k_pq = h_pq - 1/2 sum_r (pr|rq)
    h1 = numpy.asarray(h1, dtype=float)
    eri = numpy.asarray(eri, dtype=float).reshape((norb,) * 4)
    k1 = numpy.ascontiguousarray(h1 - 0.5 * numpy.einsum('prrq->pq', eri))
    # direct_nosym applies the first pair of its integrals first
    half = numpy.ascontiguousarray(0.5 * eri.transpose(2, 3, 0, 1))
    link = tuple(cistring.gen_linkstr_index(range(norb), n) for n in nelec)
    na, nb = (cistring.num_strings(norb, n) for n in nelec)
    for k in range(q.shape[1]):
        c = numpy.ascontiguousarray(q[:, k]).reshape(na, nb)
        out[:, k] = (direct_nosym.contract_1e(k1, c, norb, nelec, link)
                     + direct_nosym.contract_2e(half, c, norb, nelec, link)).ravel()
    return out


def apply_strings_ab(norb, nelec, ntarget, daggers, ops, vecs, out, x0=0):
    '''out[v, x0 + x, :] = O_x vecs[v, :] for real vectors of the sector
    nelec = (n_alpha, n_beta) and operator strings in spin-orbital indices
    leading to the sector ntarget.  Returns False without libzmr (nothing is
    written).'''
    z = _lib.zmr()
    if z is None:
        return False
    vecs = numpy.ascontiguousarray(vecs, dtype=float)
    ops = numpy.ascontiguousarray(ops, dtype=numpy.int64).reshape(len(ops), len(daggers))
    dag = numpy.ascontiguousarray([1 if d else 0 for d in daggers], dtype=numpy.intc)
    assert out.dtype == numpy.float64 and out.flags.c_contiguous
    z.zmr_apply_strings_ab(ctypes.c_int(norb), ctypes.c_int(nelec[0]), ctypes.c_int(nelec[1]),
                           ctypes.c_int(ntarget[0]), ctypes.c_int(ntarget[1]),
                           ctypes.c_int(len(daggers)), _p(dag), _p(ops),
                           ctypes.c_int64(len(ops)), ctypes.c_int64(out.shape[1]),
                           ctypes.c_int64(x0), ctypes.c_int64(vecs.shape[0]), _p(vecs), _p(out))
    return True


def apply_strings_ab_add(norb, nelec, ntarget, daggers, ops, vecs, dest, out):
    '''out[dest[v, x], :] += O_x vecs[v, :] for real vectors of the sector
    nelec and operator strings in spin-orbital indices leading to the sector
    ntarget; out (nrow, dim of ntarget).  The destinations of different
    strings must be different rows (the strings are threaded).  Returns False
    without libzmr (nothing is written).'''
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_apply_strings_ab_add'):
        return False
    vecs = numpy.ascontiguousarray(vecs, dtype=float)
    ops = numpy.ascontiguousarray(ops, dtype=numpy.int64).reshape(len(ops), len(daggers))
    dag = numpy.ascontiguousarray([1 if d else 0 for d in daggers], dtype=numpy.intc)
    dest = numpy.ascontiguousarray(dest, dtype=numpy.int64)
    assert dest.shape == (vecs.shape[0], len(ops)) and dest.max(initial=-1) < out.shape[0]
    assert out.dtype == numpy.float64 and out.flags.c_contiguous and out.ndim == 2
    z.zmr_apply_strings_ab_add(ctypes.c_int(norb), ctypes.c_int(nelec[0]), ctypes.c_int(nelec[1]),
                               ctypes.c_int(ntarget[0]), ctypes.c_int(ntarget[1]),
                               ctypes.c_int(len(daggers)), _p(dag), _p(ops),
                               ctypes.c_int64(len(ops)), ctypes.c_int64(vecs.shape[0]),
                               _p(vecs), _p(dest), _p(out))
    return True


# ----------------------------------------------------------------------
# array bookkeeping of the MRCI sigma

def _plain(a):
    return a.dtype == _c16 and a.flags.c_contiguous


def transpose_add(dst, src, perm, alpha=1.0, assign=False):
    '''dst += alpha * src.transpose(perm) (or dst[...] = with assign) for
    complex C-contiguous arrays; numpy otherwise.'''
    z = _lib.zmr()
    perm = [int(p) for p in perm]
    # the source may be any view with positive strides (e.g. a transposed
    # integral block): no copy is made of it
    strided = (src.dtype == _c16 and src.ndim == len(perm)
               and all(st > 0 and st % 16 == 0 for st in src.strides))
    if (z is None or not hasattr(z, 'zmr_transpose_add') or not _plain(dst)
            or not strided or dst.size < 4096):
        if assign:
            dst[...] = alpha * src.transpose(perm)
        else:
            dst += alpha * src.transpose(perm)
        return dst
    strides = [s // 16 for s in src.strides]
    dshape = (ctypes.c_int64 * len(perm))(*dst.shape)
    sstride = (ctypes.c_int64 * len(perm))(*[strides[p] for p in perm])
    if tuple(dst.shape) != tuple(src.shape[p] for p in perm):
        raise ValueError('transpose_add: shapes %s, %s, perm %s' % (dst.shape, src.shape, perm))
    a = complex(alpha)
    z.zmr_transpose_add(ctypes.c_int(len(perm)), dshape, sstride, _p(src), _p(dst),
                        ctypes.c_int(int(assign)), (ctypes.c_double * 2)(a.real, a.imag))
    return dst


def transposed(a, perm):
    '''Contiguous copy of a.transpose(perm).'''
    a = numpy.asarray(a)
    if a.dtype != _c16:
        a = a.astype(_c16)
    out = numpy.empty([a.shape[p] for p in perm], dtype=_c16)
    return transpose_add(out, a, perm, assign=True)


def pair_gather(a, first):
    '''a with the axes (first, first + 1), both of dimension n, replaced by
    one axis over the pairs i < j.'''
    n = a.shape[first]
    z = _lib.zmr()
    npair = n * (n - 1) // 2
    shape = a.shape[:first] + (npair,) + a.shape[first + 2:]
    if z is None or not hasattr(z, 'zmr_pair_gather') or not _plain(a):
        i, j = numpy.triu_indices(n, k=1)
        flat = a.reshape(a.shape[:first] + (n * n,) + a.shape[first + 2:])
        return numpy.take(flat, i * n + j, axis=first)
    out = numpy.empty(shape, dtype=_c16)
    nl = int(numpy.prod(a.shape[:first]))
    nt = int(numpy.prod(a.shape[first + 2:]))
    z.zmr_pair_gather(ctypes.c_int64(nl), ctypes.c_int64(n), ctypes.c_int64(nt), _p(a), _p(out))
    return out


def pair_scatter(a, axis, n):
    '''The antisymmetric array whose pairs i < j along `axis` are a: the
    axis becomes two axes of dimension n.'''
    shape = a.shape[:axis] + (n, n) + a.shape[axis + 1:]
    z = _lib.zmr()
    if z is None or not hasattr(z, 'zmr_pair_scatter') or not _plain(a):
        i, j = numpy.triu_indices(n, k=1)
        src = numpy.moveaxis(a, axis, 0)
        out = numpy.zeros((n * n,) + src.shape[1:], dtype=a.dtype)
        out[i * n + j] = src
        out[j * n + i] = -src
        return numpy.ascontiguousarray(
            numpy.moveaxis(out.reshape((n, n) + src.shape[1:]), (0, 1), (axis, axis + 1)))
    out = numpy.empty(shape, dtype=_c16)
    nl = int(numpy.prod(a.shape[:axis]))
    nt = int(numpy.prod(a.shape[axis + 1:]))
    z.zmr_pair_scatter(ctypes.c_int64(nl), ctypes.c_int64(n), ctypes.c_int64(nt), _p(a), _p(out))
    return out
