'''
Dense linear algebra through Intel MKL (cblas / LAPACKE of libmkl_rt, LP64
interface), with numpy / scipy fallbacks.

Every routine works on C-ordered arrays and keeps real input real: float64
arrays go to the d-routines (dgemm, dsyrk, dsyevd, ...), anything complex to
the z-routines.

    gemm(a, b, opa, opb)      op(a) @ op(b), op in 'N', 'T', 'C'
    herk(a, trans)            a @ a^+ ('N') or a^+ @ a ('C')
    gemmt_lower(a, b)         lower triangle of a @ b^T, real (dgemmt)
    eigh(a)                   all eigenvalues and eigenvectors (heevd / syevd)
    eigh_lowest(a, k)         the k lowest eigenpairs (heevr / syevr)
    cholesky(a)               lower Cholesky factor (potrf)
    is_positive_definite(a)   Cholesky test (potrf)
    orthonormalize(q)         Cholesky QR of nearly orthonormal columns
'''

import ctypes

import numpy
import scipy.linalg

from socutils.zmr import lib as _lib

_ROW_MAJOR = 101
_TRANS = {'N': 111, 'T': 112, 'C': 113}
_UPPER, _LOWER = 121, 122
_NON_UNIT = 131
_RIGHT = 142
_c_complex_p = ctypes.c_void_p


def _c16(a):
    return numpy.ascontiguousarray(a, dtype=numpy.complex128)


def _dtype(*arrays):
    '''float64 if every array is real, else complex128.'''
    if any(numpy.iscomplexobj(a) for a in arrays):
        return numpy.complex128
    return numpy.float64


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def gemm(a, b, opa='N', opb='N', out=None, beta=0.0):
    '''op(a) @ op(b) for 2-D arrays.  out: C-contiguous array of the result
    shape and dtype that receives beta * out + op(a) @ op(b) (no allocation;
    beta = 1 accumulates).'''
    mkl = _lib.mkl()
    dtype = _dtype(a, b)
    a = numpy.ascontiguousarray(a, dtype=dtype)
    b = numpy.ascontiguousarray(b, dtype=dtype)
    if mkl is None:
        fa = {'N': a, 'T': a.T, 'C': a.conj().T}[opa]
        fb = {'N': b, 'T': b.T, 'C': b.conj().T}[opb]
        if out is None:
            return fa @ fb
        out *= beta
        out += fa @ fb
        return out
    m, k = a.shape if opa == 'N' else a.shape[::-1]
    k2, n = b.shape if opb == 'N' else b.shape[::-1]
    if k != k2:
        raise ValueError('gemm: shapes %s and %s do not match' % (a.shape, b.shape))
    if out is None:
        c = numpy.empty((m, n), dtype=dtype)
        beta = 0.0
    else:
        c = out
        if c.shape != (m, n) or c.dtype != dtype or not c.flags.c_contiguous:
            raise ValueError('gemm: out must be a C-contiguous %s array of shape %s'
                             % (dtype, (m, n)))
    if min(m, n, k) == 0:
        if beta == 0.0:
            c[:] = 0
        else:
            c *= beta
        return c
    if dtype == numpy.float64:
        ta, tb = (_TRANS['N' if op == 'N' else 'T'] for op in (opa, opb))
        mkl.cblas_dgemm(ctypes.c_int(_ROW_MAJOR), ctypes.c_int(ta), ctypes.c_int(tb),
                        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_double(1.0),
                        _ptr(a), ctypes.c_int(a.shape[1]), _ptr(b), ctypes.c_int(b.shape[1]),
                        ctypes.c_double(float(beta)), _ptr(c), ctypes.c_int(n))
        return c
    one = (ctypes.c_double * 2)(1.0, 0.0)
    scale = (ctypes.c_double * 2)(float(beta), 0.0)
    mkl.cblas_zgemm(ctypes.c_int(_ROW_MAJOR), ctypes.c_int(_TRANS[opa]), ctypes.c_int(_TRANS[opb]),
                    ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
                    one, _ptr(a), ctypes.c_int(a.shape[1]), _ptr(b), ctypes.c_int(b.shape[1]),
                    scale, _ptr(c), ctypes.c_int(n))
    return c


def gemmt_lower(a, b):
    '''Lower triangle of a @ b^T for real a, b (n, k), at half the cost of
    the full product; the upper triangle of the result is zero.'''
    mkl = _lib.mkl()
    a = numpy.ascontiguousarray(a, dtype=numpy.float64)
    b = numpy.ascontiguousarray(b, dtype=numpy.float64)
    if a.shape != b.shape:
        raise ValueError('gemmt_lower: shapes %s and %s do not match' % (a.shape, b.shape))
    if mkl is None or not hasattr(mkl, 'cblas_dgemmt'):
        return numpy.tril(a @ b.T)
    n, k = a.shape
    c = numpy.zeros((n, n))
    if n == 0 or k == 0:
        return c
    mkl.cblas_dgemmt(ctypes.c_int(_ROW_MAJOR), ctypes.c_int(_LOWER), ctypes.c_int(_TRANS['N']),
                     ctypes.c_int(_TRANS['T']), ctypes.c_int(n), ctypes.c_int(k),
                     ctypes.c_double(1.0), _ptr(a), ctypes.c_int(k), _ptr(b), ctypes.c_int(k),
                     ctypes.c_double(0.0), _ptr(c), ctypes.c_int(n))
    return c


def herk(a, trans='N', fill=True):
    '''Hermitian rank-k product: a @ a^+ for trans='N', a^+ @ a for 'C'.
    With fill=False only the lower triangle is computed (the upper one is
    zero), which is all eigh, cholesky and is_positive_definite read.'''
    mkl = _lib.mkl()
    dtype = _dtype(a)
    a = numpy.ascontiguousarray(a, dtype=dtype)
    if mkl is None:
        c = a @ a.conj().T if trans == 'N' else a.conj().T @ a
        return c if fill else numpy.tril(c)
    n, k = a.shape if trans == 'N' else a.shape[::-1]
    c = numpy.zeros((n, n), dtype=dtype)
    if n == 0 or k == 0:
        return c
    if dtype == numpy.float64:
        fn, tr = mkl.cblas_dsyrk, _TRANS['N' if trans == 'N' else 'T']
    else:
        fn, tr = mkl.cblas_zherk, _TRANS[trans]
    fn(ctypes.c_int(_ROW_MAJOR), ctypes.c_int(_LOWER), ctypes.c_int(tr),
       ctypes.c_int(n), ctypes.c_int(k), ctypes.c_double(1.0), _ptr(a),
       ctypes.c_int(a.shape[1]), ctypes.c_double(0.0), _ptr(c), ctypes.c_int(n))
    if fill:
        c += numpy.tril(c, -1).conj().T
    return c


def eigh(a):
    '''Eigenvalues (ascending) and eigenvectors (columns) of a Hermitian
    (real symmetric) matrix; only the lower triangle is read.'''
    mkl = _lib.mkl()
    n = a.shape[0]
    dtype = _dtype(a)
    if mkl is None or n == 0:
        if n == 0:
            return numpy.zeros(0), numpy.zeros((0, 0), dtype=dtype)
        return scipy.linalg.eigh(a, driver='evr')
    v = numpy.array(a, dtype=dtype, order='C')
    w = numpy.empty(n)
    # row-major lower triangle == column-major upper triangle of the transpose;
    # LAPACKE handles the layout, eigenvectors come back as columns
    fn = mkl.LAPACKE_dsyevd if dtype == numpy.float64 else mkl.LAPACKE_zheevd
    info = fn(ctypes.c_int(_ROW_MAJOR), ctypes.c_char(b'V'), ctypes.c_char(b'L'),
              ctypes.c_int(n), _ptr(v), ctypes.c_int(n), _ptr(w))
    if info != 0:
        raise numpy.linalg.LinAlgError('eigh (heevd/syevd) failed with info = %d' % info)
    return w, v


def eigh_lowest(a, k):
    '''The k lowest eigenvalues and eigenvectors of a Hermitian matrix.'''
    mkl = _lib.mkl()
    n = a.shape[0]
    k = min(k, n)
    dtype = _dtype(a)
    if mkl is None or n == 0 or k == 0:
        if n == 0 or k == 0:
            return numpy.zeros(0), numpy.zeros((n, 0), dtype=dtype)
        return scipy.linalg.eigh(a, subset_by_index=[0, k - 1])
    work = numpy.array(a, dtype=dtype, order='C')
    w = numpy.empty(n)
    z = numpy.empty((n, k), dtype=dtype)
    isuppz = numpy.empty(2 * max(1, k), dtype=numpy.int32)
    m = ctypes.c_int(0)
    fn = mkl.LAPACKE_dsyevr if dtype == numpy.float64 else mkl.LAPACKE_zheevr
    info = fn(ctypes.c_int(_ROW_MAJOR), ctypes.c_char(b'V'), ctypes.c_char(b'I'),
              ctypes.c_char(b'L'), ctypes.c_int(n), _ptr(work), ctypes.c_int(n),
              ctypes.c_double(0.0), ctypes.c_double(0.0),
              ctypes.c_int(1), ctypes.c_int(k), ctypes.c_double(0.0),
              ctypes.byref(m), _ptr(w), _ptr(z), ctypes.c_int(k),
              isuppz.ctypes.data_as(ctypes.c_void_p))
    if info != 0:
        raise numpy.linalg.LinAlgError('eigh_lowest (heevr/syevr) failed with info = %d' % info)
    return w[:k].copy(), z


def _potrf(a):
    '''potrf on a copy of a (lower triangle); returns (factor, info).'''
    mkl = _lib.mkl()
    dtype = _dtype(a)
    work = numpy.array(a, dtype=dtype, order='C')
    fn = mkl.LAPACKE_dpotrf if dtype == numpy.float64 else mkl.LAPACKE_zpotrf
    info = fn(ctypes.c_int(_ROW_MAJOR), ctypes.c_char(b'L'),
              ctypes.c_int(work.shape[0]), _ptr(work), ctypes.c_int(work.shape[0]))
    if info < 0:
        raise numpy.linalg.LinAlgError('potrf failed with info = %d' % info)
    return work, info


def _hermitian_from_lower(a):
    return numpy.tril(a) + numpy.tril(a, -1).conj().T


def cholesky(a):
    '''Lower triangular L with a = L L^+ for a Hermitian positive definite
    matrix (lower triangle read), or None if a is not positive definite.'''
    n = a.shape[0]
    if n == 0:
        return numpy.zeros((0, 0), dtype=_dtype(a))
    if _lib.mkl() is None:
        try:
            return numpy.linalg.cholesky(_hermitian_from_lower(a))
        except numpy.linalg.LinAlgError:
            return None
    work, info = _potrf(a)
    return numpy.tril(work) if info == 0 else None


def is_positive_definite(a):
    '''True if the Hermitian matrix a (lower triangle read) has a Cholesky
    factorization.'''
    n = a.shape[0]
    if n == 0:
        return True
    if _lib.mkl() is None:
        return cholesky(a) is not None
    return _potrf(a)[1] == 0


def orthonormalize(q):
    '''Orthonormalize the columns of q (n, m) by Cholesky QR: q L^{-+} with
    L L^+ = q^+ q.  Meant for columns that are already close to orthonormal
    (the Gram matrix has to be numerically positive definite).'''
    dtype = _dtype(q)
    q = numpy.array(q, dtype=dtype, order='C')
    n, m = q.shape
    if m == 0:
        return q
    low = cholesky(herk(q, 'C', fill=False))
    if low is None:
        raise numpy.linalg.LinAlgError('orthonormalize: columns are linearly dependent')
    mkl = _lib.mkl()
    if mkl is None:
        # x L^+ = q  <=>  L x^+ = q^+
        return numpy.ascontiguousarray(
            scipy.linalg.solve_triangular(low, q.conj().T, lower=True).conj().T)
    args = (ctypes.c_int(_ROW_MAJOR), ctypes.c_int(_RIGHT), ctypes.c_int(_LOWER))
    tail = (ctypes.c_int(_NON_UNIT), ctypes.c_int(n), ctypes.c_int(m))
    if dtype == numpy.float64:
        mkl.cblas_dtrsm(*args, ctypes.c_int(_TRANS['T']), *tail, ctypes.c_double(1.0),
                        _ptr(low), ctypes.c_int(m), _ptr(q), ctypes.c_int(m))
    else:
        one = (ctypes.c_double * 2)(1.0, 0.0)
        mkl.cblas_ztrsm(*args, ctypes.c_int(_TRANS['C']), *tail, one,
                        _ptr(low), ctypes.c_int(m), _ptr(q), ctypes.c_int(m))
    return q


def rgemm(a, b):
    '''a @ b for real 2-D arrays (dgemm).'''
    return gemm(numpy.asarray(a, dtype=numpy.float64), numpy.asarray(b, dtype=numpy.float64))
