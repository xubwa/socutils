'''The kernels of the spinor CCSD amplitude equations (csrc/zccsd_kernels.c,
OpenMP) with their numpy equivalents.

Pairs are i > j and a > b in numpy.tril_indices(n, -1) order; the "ring
layout" of a four-index array is the matrix [(i a), (j b)].  All arrays are
C-contiguous complex128 unless said otherwise; the output arguments are
written in place.

libzc_zccsd.so is looked for in socutils/lib ($SOCUTILS_ZCCSD_LIBRARY overrides).
Without it, or with USE_C = False, numpy does the same on one thread (the
reference of the tests).
'''

import ctypes
import functools
import os

import numpy as np

_P, _I, _D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
_SIGNATURES = {
    'zc_copy': [_P, _I, ctypes.c_int, _P],
    'zc_dotu': [_P, _P, _I, _P],
    'zc_diff_norm2': [_P, _P, _I],
    'zc_split3': [_P, _I, _I, _I, _P, _P, _P],
    'zc_combine3': [_P, _P, _P, _I, _D, ctypes.c_int, _P],
    'zc_combine3_ld': [_P, _P, _P, _I, _I, _I, _D, ctypes.c_int, _P, _I],
    'zc_ring_cols3': [_P, _I, _I, _I, _I, _P, _P, _P],
    'zc_outer_pp': [_P, _P, _I, _I, _D, _P],
    'zc_ring_cols': [_P, _I, _I, _I, _I, _P],
    'zc_asym_pack_cols': [_P, _I, _I, _I, _I, _I, _D, _P],
    'zc_kfill_ring': [_P, _I, _I, _I, _I, _I, _I],
    'zc_kfill_packed': [_P, _I, _I, _P, _P, _P, _P, _P],
    'zc_add_rows_k': [_P, _I, _P, _P, _P, _P, _I, _P, _P],
    'zc_asym_pack2': [_P, _I, _I, _I, _I, _D, _P],
    'zc_pack_tril_add': [_P, _I, _I, _I, _I, _D, _P],
    'zc_asym_pack2_set': [_P, _I, _I, _I, _I, _P],
    'zc_tril_matvec': [_P, _I, _I, _P, _I, _I, _I, _P],
    'zc_unpack_tril': [_P, _I, _I, _I, _I, _P],
    'zc_pack_t2': [_P, _I, _I, _P],
    'zc_unpack_t2': [_P, _I, _I, _P],
    'zc_div_packed': [_P, _P, _P, _I, _I],
    'zc_t2p_matvec': [_P, _P, _I, _I, _P],
    'zc_add_rows': [_P, _I, _P, _P, _P, _I, ctypes.c_int],
    'zc_ovvv_t1': [_P, _P, _I, _I, _I, _P, _P, _P],
    'zc_ladder_pack': [_P, _I, _I, _I, _I, _P, _P],
    'zc_ladder_unpack': [_P, _P, _I, _I, _I, _I, _P],
    'zc_ladder_rows': [_P, ctypes.c_int, _I, _I, _I, _I, _I, _P, _I, _P, _I],
    'zc_ksym_packed': [_P, _I, _I, _P, _P, _P, _P],
}


def _load():
    from socutils.lib import library_dir
    path = os.environ.get('SOCUTILS_ZCCSD_LIBRARY') or os.path.join(library_dir(), 'libzc_zccsd.so')
    try:
        lib = ctypes.CDLL(path)
        for name, args in _SIGNATURES.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, None
        lib.zc_diff_norm2.restype = _D
    except (OSError, AttributeError):       # not built, or built from an older source
        return None
    return lib


_lib = _load()
USE_C = _lib is not None


def _c(a, dtype=np.complex128):
    '''The address of an array the C side reads or writes in place.'''
    assert a.dtype == dtype and a.flags.c_contiguous, (a.dtype, a.flags.c_contiguous)
    return a.ctypes.data


def _rows(a, itemsize):
    '''Is ``a`` a block of rows of unit stride in a row-major array of that
    item size?'''
    return (a.ndim == 2 and a.itemsize == itemsize and a.strides[1] == itemsize
            and a.strides[0] >= a.shape[1] * itemsize and a.strides[0] % itemsize == 0)


@functools.lru_cache(maxsize=None)
def tril(n, k=-1):
    '''numpy.tril_indices(n, k), cached.'''
    i, j = np.tril_indices(n, k)
    i.setflags(write=False)
    j.setflags(write=False)
    return i, j


def npair(n):
    return n * (n - 1) // 2


# --------------------------------------------------------------------- vectors

def copy(src, dst=None, conj=False):
    '''dst = src (its complex conjugate with ``conj``) on all threads.'''
    if dst is None:
        dst = np.empty(src.shape, dtype=np.complex128)
    if USE_C:
        _lib.zc_copy(_c(src), src.size, int(conj), _c(dst))
        return dst
    np.conjugate(src, out=dst) if conj else np.copyto(dst, src)
    return dst


def dotu(x, y):
    '''sum x y, not conjugated.'''
    if USE_C:
        out = np.zeros(1, dtype=np.complex128)
        _lib.zc_dotu(_c(x), _c(y), x.size, _c(out))
        return out[0]
    return np.dot(x.ravel(), y.ravel())


def diff_norm2(x, y):
    '''||x - y||^2.'''
    if USE_C:
        return _lib.zc_diff_norm2(_c(x), _c(y), x.size)
    d = x - y
    return float(np.vdot(d, d).real)


def split3(x):
    '''(Re x, Im x, Re x + Im x) of a complex matrix (C-contiguous or a
    row-major submatrix), contiguous real arrays: the operands of a complex
    product by three real ones (combine3).'''
    m, n = x.shape
    out = tuple(np.empty((m, n)) for _ in range(3))
    if USE_C and x.dtype == np.complex128 and x.strides[1] == 16 and x.strides[0] % 16 == 0:
        _lib.zc_split3(x.ctypes.data, m, n, x.strides[0] // 16,
                       *(_c(a, np.float64) for a in out))
        return out
    out[0][:], out[1][:] = x.real, x.imag
    np.add(out[0], out[1], out=out[2])
    return out


def combine3(p1, p2, p3, alpha=1.0, C=None, acc=False):
    '''C (+)= alpha A B from p1 = Ar Br, p2 = Ai Bi, p3 = (Ar + Ai)(Br + Bi).
    The four may be blocks of columns of larger row-major matrices (the
    three products with the same leading dimension).'''
    if C is None:
        C, acc = np.empty(p1.shape, dtype=np.complex128), False
    if USE_C and all(x.flags.c_contiguous for x in (p1, p2, p3, C)):
        _lib.zc_combine3(_c(p1, np.float64), _c(p2, np.float64), _c(p3, np.float64), p1.size,
                         alpha, int(acc), _c(C))
        return C
    if USE_C and C.ndim == 2 and _rows(C, 16) and all(
            x.dtype == np.float64 and x.shape == C.shape and _rows(x, 8)
            and x.strides[0] == p1.strides[0] for x in (p1, p2, p3)):
        _lib.zc_combine3_ld(p1.ctypes.data, p2.ctypes.data, p3.ctypes.data, C.shape[0],
                            C.shape[1], p1.strides[0] // 8, alpha, int(acc), C.ctypes.data,
                            C.strides[0] // 16)
        return C
    val = alpha * ((p1 - p2) + 1j * (p3 - p1 - p2))
    if acc:
        C += val
    else:
        C[:] = val
    return C


# ------------------------------------------------------------------ amplitudes

def outer_pp(x, y, fac, R):
    '''R[(i>j),(a>b)] += fac (x[i,a] y[j,b] - x[i,b] y[j,a] - x[j,a] y[i,b] + x[j,b] y[i,a]).'''
    o, v = x.shape
    if USE_C:
        _lib.zc_outer_pp(_c(x), _c(y), o, v, fac, _c(R))
        return R
    (io, jo), (av, bv) = tril(o), tril(v)
    xi, xj, yi, yj = x[io], x[jo], y[io], y[jo]
    R += fac * (xi[:, av] * yj[:, bv] - xi[:, bv] * yj[:, av]
                - xj[:, av] * yi[:, bv] + xj[:, bv] * yi[:, av])
    return R


def pack_t2(t2, out=None):
    '''t2p[(i>j),(a>b)] = t2[i,j,a,b].'''
    o, v = t2.shape[0], t2.shape[2]
    if out is None:
        out = np.empty((npair(o), npair(v)), dtype=np.complex128)
    if USE_C:
        _lib.zc_pack_t2(_c(t2), o, v, _c(out))
        return out
    (io, jo), (av, bv) = tril(o), tril(v)
    out[:] = t2[io[:, None], jo[:, None], av, bv]
    return out


def unpack_t2(t2p, o, v, out=None):
    '''t2[i,j,a,b], antisymmetric in both pairs, from t2p[(i>j),(a>b)].'''
    if out is None:
        out = np.empty((o, o, v, v), dtype=np.complex128)
    if USE_C:
        _lib.zc_unpack_t2(_c(t2p), o, v, _c(out))
        return out
    (io, jo), (av, bv) = tril(o), tril(v)
    i, j = io[:, None], jo[:, None]
    out[:] = 0
    out[i, j, av, bv] = t2p
    out[j, i, av, bv] = -t2p
    out[i, j, bv, av] = -t2p
    out[j, i, bv, av] = t2p
    return out


def ring_cols(t2p, o, v, j0=0, j1=None, out=None):
    '''The columns j0 <= j < j1 of t2 in the ring layout: out[(n f),(j b)] =
    t2[n,j,f,b], an (o v) x ((j1 - j0) v) matrix.  The full matrix is
    symmetric.'''
    j1 = o if j1 is None else j1
    if out is None:
        out = np.empty((o * v, (j1 - j0) * v), dtype=np.complex128)
    if USE_C:
        _lib.zc_ring_cols(_c(t2p), o, v, j0, j1, _c(out))
        return out
    t2 = unpack_t2(t2p, o, v)
    out[:] = t2[:, j0:j1].transpose(0, 2, 1, 3).reshape(o * v, (j1 - j0) * v)
    return out


def ring_cols3(t2p, o, v, j0=0, j1=None):
    '''ring_cols as the three real arrays of split3.'''
    j1 = o if j1 is None else j1
    if USE_C:
        out = tuple(np.empty((o * v, (j1 - j0) * v)) for _ in range(3))
        _lib.zc_ring_cols3(_c(t2p), o, v, j0, j1, *(_c(a, np.float64) for a in out))
        return out
    return split3(ring_cols(t2p, o, v, j0, j1))


def asym_pack_cols(x, o, v, j0, j1, alpha, R, offset=0):
    '''R[(i>j),(a>b)] += alpha (x[i,a,j,b] - x[j,a,i,b] - x[i,b,j,a] + x[j,b,i,a])
    from the columns j0 <= j < j1 of x in the ring layout - the terms whose
    third index lies in the range.  x has o v rows; its columns from
    ``offset`` on are (j - j0, b).'''
    nc = (j1 - j0) * v
    if USE_C:
        _lib.zc_asym_pack_cols(_c(x) + 16 * int(offset), x.shape[1], o, v, j0, j1, alpha, _c(R))
        return R
    (io, jo), (av, bv) = tril(o), tril(v)
    full = np.zeros((o, v, o, v), dtype=np.complex128)
    full[:, :, j0:j1] = x[:, offset:offset + nc].reshape(o, v, j1 - j0, v)
    i, j = io[:, None], jo[:, None]
    R += alpha * (full[i, av, j, bv] - full[j, av, i, bv] - full[i, bv, j, av] + full[j, bv, i, av])
    return R


def _strides(n, v, order):
    '''(sp, sa) of the element (p, a, b) for the layouts 'pab' and 'apb'.'''
    return (v * v, v) if order == 'pab' else (v, n * v)


def asym_pack2(x, alpha, R, order='pab'):
    '''R[p,(a>b)] += alpha (x[p,a,b] - x[p,b,a]) for x[p,a,b] ('pab') or
    x[a,p,b] ('apb').'''
    n, v = R.shape[0], x.shape[2]
    if USE_C:
        _lib.zc_asym_pack2(_c(x), n, v, *_strides(n, v, order), alpha, _c(R))
        return R
    av, bv = tril(v)
    R += alpha * ((x[:, av, bv] - x[:, bv, av]) if order == 'pab'
                  else (x[av, :, bv] - x[bv, :, av]).T)
    return R


def asym_pack_view(x, R):
    '''R[p,(a>b)] = x[a,p,b] - x[b,p,a] for a block x[a,p,b] of a larger
    C-ordered array (v x n x v; any strides along a and p, unit stride along
    b): the antisymmetric part of the planes of the block, packed, without a
    copy of the block.'''
    v, n = x.shape[0], x.shape[1]
    sa, sp, sb = x.strides
    if (USE_C and x.dtype == np.complex128 and sb == 16 and sa > 0 and sp > 0
            and sa % 16 == 0 and sp % 16 == 0):
        _lib.zc_asym_pack2_set(x.ctypes.data, n, v, sp // 16, sa // 16, _c(R))
        return R
    av, bv = tril(v)
    R[:] = (x[av, :, bv] - x[bv, :, av]).T
    return R


def tril_matvec(G, q, out):
    '''out[f, a] += sum_r sum_c M_r[a,c] q[f,r,c], M_r the antisymmetric
    matrix of the packed row G[r, (a>c)]; q[f, r, c] may be a block of a
    larger C-ordered array (unit stride along c).  The packed rows are read
    as they are.'''
    nr, (nf, v) = G.shape[0], out.shape
    sf, sr, sc = q.strides
    if (USE_C and q.dtype == np.complex128 and sc == 16 and sf >= 0 and sr >= 0
            and sf % 16 == 0 and sr % 16 == 0):
        _lib.zc_tril_matvec(_c(G), nr, v, q.ctypes.data, nf, sf // 16, sr // 16, _c(out))
        return out
    out += np.einsum('rac,frc->fa', unpack_tril(np.ascontiguousarray(G), v, 'pab'), q)
    return out


def unpack_tril(xp, v, order='pab', out=None):
    '''The antisymmetric matrices of the rows of xp[p,(a>b)]: out[p,a,b]
    ('pab') or out[a,p,b] ('apb').'''
    n = xp.shape[0]
    if out is None:
        out = np.empty((n, v, v) if order == 'pab' else (v, n, v), dtype=np.complex128)
    if USE_C:
        _lib.zc_unpack_tril(_c(xp), n, v, *_strides(n, v, order), _c(out))
        return out
    av, bv = tril(v)
    x = np.zeros((n, v, v), dtype=np.complex128)
    x[:, av, bv] = xp
    x[:, bv, av] = -xp
    out[:] = x if order == 'pab' else x.transpose(1, 0, 2)
    return out


def pack_tril_add(x, alpha, R, order='pab'):
    '''R[p,(a>b)] += alpha x[p,a,b] for x[p,a,b] ('pab') or x[a,p,b] ('apb').'''
    n, v = R.shape[0], x.shape[2]
    if USE_C:
        _lib.zc_pack_tril_add(_c(x), n, v, *_strides(n, v, order), alpha, _c(R))
        return R
    av, bv = tril(v)
    R += alpha * (x[:, av, bv] if order == 'pab' else x[av, :, bv].T)
    return R


def div_packed(R, eo, ev):
    '''R[(i>j),(a>b)] /= eo[i] + eo[j] - ev[a] - ev[b].'''
    o, v = eo.size, ev.size
    eo, ev = np.ascontiguousarray(eo, dtype=np.float64), np.ascontiguousarray(ev, dtype=np.float64)
    if USE_C:
        _lib.zc_div_packed(_c(R), _c(eo, np.float64), _c(ev, np.float64), o, v)
        return R
    (io, jo), (av, bv) = tril(o), tril(v)
    R /= (eo[io] + eo[jo])[:, None] - (ev[av] + ev[bv])[None, :]
    return R


def mul_packed(R, eo, ev):
    '''R[(i>j),(a>b)] *= eo[i] + eo[j] - ev[a] - ev[b]: div_packed undone,
    row by row.'''
    (io, jo), (av, bv) = tril(eo.size), tril(ev.size)
    dv = np.asarray(ev, dtype=np.float64)[av] + np.asarray(ev, dtype=np.float64)[bv]
    for p in range(R.shape[0]):
        R[p] *= (eo[io[p]] + eo[jo[p]]) - dv
    return R


def t2p_matvec(t2p, F, out):
    '''out[i,a] += sum_me t2[i,m,a,e] F[m,e].'''
    o, v = F.shape
    if USE_C:
        _lib.zc_t2p_matvec(_c(t2p), _c(F), o, v, _c(out))
        return out
    out += np.einsum('imae,me->ia', unpack_t2(t2p, o, v), F)
    return out


def add_rows(R, rows, signs, Y, conj=False):
    '''R[rows[k]] += signs[k] Y[k] (conj(Y[k]) with ``conj``); rows < 0 are
    skipped, the others distinct.'''
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    signs = np.ascontiguousarray(signs, dtype=np.float64)
    if USE_C:
        _lib.zc_add_rows(_c(R), R.shape[1], _c(rows, np.int64), _c(signs, np.float64), _c(Y),
                         rows.size, int(conj))
        return R
    k = rows >= 0
    R[rows[k]] += signs[k, None] * (Y[k].conj() if conj else Y[k])
    return R


# -------------------------------------------------------------------- <ma||ef>

def ovvv_t1(Om, t1, m, Wm, fv, x1):
    '''The <mb||ef> t1 terms of the occupied m from Om[b,(e>f)] = <mb||ef>:
    with A[b,e,j] = sum_f <mb||ef> t1[j,f], Wm[e,(j b)] += A[b,e,j] (the rows
    (m e) of W in the ring layout), fv[b,e] = A[b,e,m] and x1[j,b] =
    sum_e t1[m,e] A[b,e,j].'''
    o, v = t1.shape
    if USE_C:
        _lib.zc_ovvv_t1(_c(Om), _c(t1), v, o, m, _c(Wm), _c(fv), _c(x1))
        return
    av, bv = tril(v)
    U = np.zeros((v, v, v), dtype=np.complex128)
    U[:, av, bv] = Om
    U[:, bv, av] = -Om
    A = (U.reshape(v * v, v) @ t1.T).reshape(v, v, o)
    Wm.reshape(v, o, v)[:] += A.transpose(1, 2, 0)
    fv[:] = A[:, :, m]
    x1[:] = np.einsum('e,bej->jb', t1[m], A)


# ------------------------------------------------------------------- AO ladder

def ladder_pack(X, n, nao, order='xpy'):
    '''The spin blocks of X[p] (n matrices over (spin, AO), X[x,p,y] for
    'xpy', X[p,x,y] for 'pxy') as real rows: AR (6 n, nao (nao - 1) / 2) with
    aa, bb, (ab - ab^T) / 2, real parts then imaginary parts, and SR
    (2 n, nao (nao + 1) / 2) with (ab + ab^T) / 2.'''
    n2 = 2 * nao
    AR = np.empty((6 * n, nao * (nao - 1) // 2))
    SR = np.empty((2 * n, nao * (nao + 1) // 2))
    if USE_C:
        sp, sx = (n2 * n2, n2) if order == 'pxy' else (n2, n * n2)
        _lib.zc_ladder_pack(_c(X), n, nao, sp, sx, _c(AR, np.float64), _c(SR, np.float64))
        return AR, SR
    X = X.reshape(n, n2, n2) if order == 'pxy' else X.reshape(n2, n, n2).transpose(1, 0, 2)
    (pi, pj), (si, sj) = tril(nao), tril(nao, 0)
    ab = X[:, :nao, nao:]
    anti = np.concatenate((X[:, :nao, :nao][:, pi, pj], X[:, nao:, nao:][:, pi, pj],
                           0.5 * (ab[:, pi, pj] - ab[:, pj, pi])))
    sym = 0.5 * (ab[:, si, sj] + ab[:, sj, si])
    AR[:3 * n], AR[3 * n:] = anti.real, anti.imag
    SR[:n], SR[n:] = sym.real, sym.imag
    return AR, SR


def ladder_unpack(GA, GS, n, nao, order='xpy'):
    '''The inverse layout for GA = AR W_same and GS = SR W_plus: G[p] with aa
    and bb antisymmetric, ab = GS + GA_ab and ba = -ab^T, as G[x,p,y]
    ('xpy') or G[p,x,y] ('pxy').'''
    n2 = 2 * nao
    G = np.empty((n2, n, n2) if order == 'xpy' else (n, n2, n2), dtype=np.complex128)
    if USE_C:
        sp, sx = (n2 * n2, n2) if order == 'pxy' else (n2, n * n2)
        _lib.zc_ladder_unpack(_c(GA, np.float64), _c(GS, np.float64), n, nao, sp, sx, _c(G))
        return G
    (pi, pj), (si, sj) = tril(nao), tril(nao, 0)
    Ga = GA[:3 * n] + 1j * GA[3 * n:]
    Gs = GS[:n] + 1j * GS[n:]
    g = np.zeros((n, n2, n2), dtype=np.complex128)
    aa, bb, ab, ba = g[:, :nao, :nao], g[:, nao:, nao:], g[:, :nao, nao:], g[:, nao:, :nao]
    aa[:, pi, pj] = Ga[:n]
    aa[:, pj, pi] = -Ga[:n]
    bb[:, pi, pj] = Ga[n:2 * n]
    bb[:, pj, pi] = -Ga[n:2 * n]
    ab[:, si, sj] = Gs
    ab[:, sj, si] = Gs
    ab[:, pi, pj] += Ga[2 * n:]
    ab[:, pj, pi] -= Ga[2 * n:]
    ba[:] = -ab.transpose(0, 2, 1)
    G[:] = g if order == 'pxy' else g.transpose(1, 0, 2)
    return G


def ladder_row_counts(nf, f0, nt, t0):
    '''(rows of W_same, rows of W_plus) that a block of first indices f0 ..
    f0 + nf - 1 and third indices t0 .. t0 + nt - 1 holds: the pairs with
    third < first, third <= first.'''
    f = np.arange(f0, f0 + nf)
    return (int(np.clip(f - t0, 0, nt).sum()), int(np.clip(f + 1 - t0, 0, nt).sum()))


def ladder_rows(E, f0, t0, same, plus):
    '''Rows of the AO operators of the ladder from the AO integrals
    E[f, x, t, y] = (f0 + f, x | t0 + t, y), x and y over the first m AO
    functions: with M[x, y] = E[f, x, t, y],

        same: M[x, y] - M[y, x] over x > y     for the pairs (f0 + f > t0 + t)
        plus: M[x, y] + M[y, x] over x >= y    for the pairs (f0 + f >= t0 + t)

    in the first m (m - 1) / 2 and m (m + 1) / 2 elements of the rows of
    ``same`` and ``plus`` (row-major arrays or blocks of columns), the pairs
    in the order (f, t).  A three-index E is E[f, x, (t >= y)], the integrals
    with their symmetry in the last two indices for all t < m (t0 = 0).
    Returns the numbers of rows written.'''
    packed = E.ndim == 3
    nf, m = E.shape[:2]
    nt = m if packed else E.shape[2]
    assert not packed or (t0 == 0 and E.shape[2] == m * (m + 1) // 2)
    ns, npl = ladder_row_counts(nf, f0, nt, t0)
    assert same.shape[0] >= ns and plus.shape[0] >= npl
    assert same.shape[1] >= m * (m - 1) // 2 and plus.shape[1] >= m * (m + 1) // 2
    if USE_C:
        assert plus.strides[1] == 8 and (not ns or same.strides[1] == 8)
        _lib.zc_ladder_rows(_c(E, np.float64), int(packed), nf, f0, nt, t0, m,
                            same.ctypes.data, same.strides[0] // 8 if ns else 0,
                            plus.ctypes.data, plus.strides[0] // 8)
        return ns, npl
    (pi, pj), (si, sj) = tril(m), tril(m, 0)
    if packed:
        idx = np.empty((m, m), dtype=np.int64)
        idx[si, sj] = idx[sj, si] = np.arange(si.size)
        E = E[:, :, idx]                                        # [f, x, t, y]
    M = E.transpose(0, 2, 1, 3)                                 # [f, t, x, y]
    f, t = np.arange(f0, f0 + nf)[:, None], np.arange(t0, t0 + nt)[None, :]
    same[:ns, :pi.size] = (M[:, :, pi, pj] - M[:, :, pj, pi])[t < f]
    plus[:npl, :si.size] = (M[:, :, si, sj] + M[:, :, sj, si])[t <= f]
    return ns, npl


# --------------------------------------------------------------- Kramers pairs

def _half_maps(n):
    '''(partner, sign) of the time reversal on n spinors [unbarred, barred]:
    K p = sign[p] partner[p].'''
    h = n // 2
    return np.r_[np.arange(h, n), np.arange(h)], np.r_[np.ones(h), -np.ones(h)]


def kfill_ring(X, o, v, i0, i1, nj):
    '''The rows (i, a) with barred i of X in the ring layout from those with
    unbarred i, i0 <= i < i1: X[(i + o/2, Ka), (K jj, Kb)] = s_a s_jj s_b
    conj(X[(i, a), (jj, b)]); the columns are (jj, b) for nj unbarred occupied
    jj followed by their nj partners.'''
    if USE_C:
        _lib.zc_kfill_ring(_c(X), o, v, i0, i1, nj, X.shape[1])
        return X
    ho = o // 2
    (kv, sv), (kj, sj) = _half_maps(v), _half_maps(2 * nj)
    X4 = X.reshape(o, v, 2 * nj, v)
    src = X4[i0:i1] * (sv[:, None, None] * sj[:, None] * sv)
    dst = np.empty_like(src)
    dst[:, kv[:, None, None], kj[:, None], kv] = src.conj()
    X4[i0 + ho:i1 + ho] = dst
    return X


def kfill_packed(L, drow, srow, kpv, sgv, R):
    '''R[drow[p], q] += srow[p] sgv[q] conj(L[p, kpv[q]]); drow < 0 skipped.'''
    drow = np.ascontiguousarray(drow, dtype=np.int64)
    srow = np.ascontiguousarray(srow, dtype=np.float64)
    if USE_C:
        _lib.zc_kfill_packed(_c(L), L.shape[0], L.shape[1], _c(drow, np.int64),
                             _c(srow, np.float64), _c(kpv, np.int64), _c(sgv, np.float64), _c(R))
        return R
    k = drow >= 0
    R[drow[k]] += srow[k, None] * sgv * L[k][:, kpv].conj()
    return R


def ksym_packed(X, kpo, sgo, kpv, sgv):
    '''X += its time-reversed image, in place: X[P, Q] += sgo[P] sgv[Q]
    conj(X[kpo[P], kpv[Q]]) with the values before the call, for a packed
    array X[(i>j), (a>b)] (kramers.pair_maps of the occupied and of the
    virtual spinors).  It completes a symmetric quantity of which the part
    summed over the unbarred half of an index is given.'''
    kpo, kpv = (np.ascontiguousarray(x, dtype=np.int64) for x in (kpo, kpv))
    sgo, sgv = (np.ascontiguousarray(x, dtype=np.float64) for x in (sgo, sgv))
    if USE_C:
        _lib.zc_ksym_packed(_c(X), X.shape[0], X.shape[1], _c(kpo, np.int64),
                            _c(sgo, np.float64), _c(kpv, np.int64), _c(sgv, np.float64))
        return X
    X += sgo[:, None] * sgv * X[kpo][:, kpv].conj()
    return X


def add_rows_k(R, rows, signs, Z, zrow, kpv, sgv):
    '''R[rows[k], q] += signs[k] sgv[q] Z[zrow[k], kpv[q]]; rows < 0 skipped.'''
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    signs = np.ascontiguousarray(signs, dtype=np.float64)
    zrow = np.ascontiguousarray(zrow, dtype=np.int64)
    if USE_C:
        _lib.zc_add_rows_k(_c(R), R.shape[1], _c(rows, np.int64), _c(signs, np.float64), _c(Z),
                           _c(zrow, np.int64), rows.size, _c(kpv, np.int64), _c(sgv, np.float64))
        return R
    k = rows >= 0
    R[rows[k]] += signs[k, None] * sgv * Z[zrow[k]][:, kpv]
    return R
