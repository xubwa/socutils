'''GEMM on C-ordered arrays with accumulation (beta).

Uses zc_gemm from the zccsdt C library when available, so that Python-side
contractions run on the same BLAS (MKL with the GNU OpenMP layer) and thread
pool as the C kernels; otherwise SciPy BLAS with transposed (F-order) views.
numpy.matmul cannot accumulate into an existing array, which costs a
temporary of the output size plus an extra pass over memory.
'''

import ctypes
import os

import numpy as np
from scipy.linalg import blas as _blas

_GEMM = {np.dtype(np.float64): _blas.dgemm, np.dtype(np.complex128): _blas.zgemm}
_SFX = {np.dtype(np.float64): 'd', np.dtype(np.complex128): 'z'}


def _clib():
    # zccsdt's zc_gemm (in its T3 kernel library) is not part of socutils:
    # numpy/BLAS is used
    return None


def openmp_runtimes():
    '''File names of the OpenMP runtime libraries loaded in this process
    (from /proc/self/maps; [] where that does not exist).  One runtime means
    one pool of threads for the OpenMP kernels and an OpenMP BLAS - also when
    it is LLVM's libomp serving the GNU interface, as in conda environments
    where libgomp.so.1 is a link to libomp.so.  With two runtimes the kernels
    and the BLAS may each keep a pool busy.'''
    try:
        with open('/proc/self/maps') as f:
            paths = {line.split()[-1] for line in f if '/' in line}
    except OSError:
        return []
    stems = ('libgomp', 'libomp', 'libiomp5')
    return sorted({os.path.basename(os.path.realpath(p)) for p in paths
                   if os.path.basename(p).split('.so')[0] in stems})


def backend():
    '''(kind, description) of the BLAS behind gemm: 'mkl' (the MKL build of
    the kernel library), 'other' (the BLAS another build of the kernel
    library was linked with) or 'scipy' (no kernel library: SciPy's BLAS).'''
    runtimes = ', '.join(openmp_runtimes())
    omp = '; OpenMP runtime: %s' % runtimes if runtimes else ''
    lib = _clib()
    if lib is None:
        return 'scipy', "SciPy's BLAS (the kernel library libzc_ccsdt_t3 is not built%s)" % omp
    name = os.path.basename(lib._name)
    try:
        buf = ctypes.create_string_buffer(256)
        lib.MKL_Get_Version_String(buf, ctypes.c_int(256))
        version = buf.value.decode().strip()
        version = version[version.find('Version'):].split(' for ')[0] or 'MKL'
        nthreads = lib.MKL_Get_Max_Threads()
        return 'mkl', 'MKL %s, %d threads (%s%s)' % (version, nthreads, name, omp)
    except AttributeError:
        return 'other', 'the BLAS of %s%s' % (name, omp and ' (%s)' % omp[2:])


def _ld(a):
    '''(transposed flag, leading dimension) for a 2-D view that BLAS can read
    in place: C- or F-contiguous, or a row- / column-major submatrix (unit
    stride along one axis, leading stride >= that axis' length).'''
    if a.flags.c_contiguous:
        return 0, a.shape[1]
    if a.flags.f_contiguous:
        return 1, a.shape[0]
    isz = a.itemsize
    s0, s1 = a.strides
    m, n = a.shape
    if s1 == isz and s0 > 0 and s0 % isz == 0 and s0 // isz >= n:
        return 0, s0 // isz
    if s0 == isz and s1 > 0 and s1 % isz == 0 and s1 // isz >= m:
        return 1, s1 // isz
    return None


def gemm(A, B, C=None, alpha=1.0, beta=0.0):
    '''C = alpha * A @ B + beta * C in place.  A and B may be C- or
    F-contiguous 2-D views or submatrices of such (transposes are free); C
    must be C-contiguous or a block of columns of a C-contiguous array.'''
    dtype = np.result_type(A, B) if C is None else C.dtype
    if A.dtype != dtype or _ld(A) is None:
        A = np.ascontiguousarray(A, dtype=dtype)
    if B.dtype != dtype or _ld(B) is None:
        B = np.ascontiguousarray(B, dtype=dtype)
    if C is None:
        C = np.empty((A.shape[0], B.shape[1]), dtype=dtype)
        beta = 0.0
    # C must be the caller's own buffer, row-major (C-contiguous or a block
    # of the columns of such an array): a reshape of anything else would be
    # a silent copy and the result lost
    ldc = (0, C.shape[1]) if C.flags.c_contiguous else _ld(C)
    assert ldc is not None and ldc[0] == 0 and C.shape == (A.shape[0], B.shape[1])
    m, k = A.shape
    n = B.shape[1]
    if k == 0 or m == 0 or n == 0:
        C *= beta
        return C
    lib = _clib()
    if lib is not None:
        ta, lda = _ld(A)
        tb, ldb = _ld(B)
        al = np.array(alpha, dtype=dtype)
        be = np.array(beta, dtype=dtype)
        # (the address: data_as() leaves a reference cycle for the garbage
        # collector with every call)
        p = lambda x: ctypes.c_void_p(x.ctypes.data)          # noqa: E731
        i = ctypes.c_int64
        getattr(lib, 'zc_gemm_' + _SFX[np.dtype(dtype)])(
            ctypes.c_int(ta), ctypes.c_int(tb), i(m), i(n), i(k), p(al), p(A),
            i(lda), p(B), i(ldb), p(be), p(C), i(ldc[1]))
        return C
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    out = _GEMM[np.dtype(dtype)](alpha, B.T, A.T, beta=beta, c=C.T, overwrite_c=1)
    if not np.shares_memory(out, C):
        C[...] = out.T
    return C


def gemm_tasks(tasks, A, B, C, alpha=1.0, beta=0.0, cols=96):
    '''A list of products in one parallel region,
    C_t = alpha A_t B_t + beta C_t, with tasks[t] = (m, n, k, offset of A_t,
    lda, offset of B_t, ldb, offset of C_t, ldc): the matrices are row-major
    parts of the C-contiguous arrays A, B, C (same dtype; offsets in
    elements).  Each product is split into pieces of about ``cols`` columns,
    one sequential GEMM each, which the threads share: for many products too
    small for a threaded BLAS.  The C_t must not overlap.'''
    tasks = np.ascontiguousarray(tasks, dtype=np.int64).reshape(-1, 9)
    dtype = C.dtype
    assert A.dtype == dtype and B.dtype == dtype
    assert A.flags.c_contiguous and B.flags.c_contiguous and C.flags.c_contiguous
    lib = _clib()
    if lib is not None and hasattr(lib, 'zc_gemm_tasks_d') and dtype in _SFX:
        al, be = np.array(alpha, dtype=dtype), np.array(beta, dtype=dtype)
        p = lambda x: ctypes.c_void_p(x.ctypes.data)          # noqa: E731
        getattr(lib, 'zc_gemm_tasks_' + _SFX[np.dtype(dtype)])(
            ctypes.c_int64(len(tasks)), p(tasks), p(A), p(B), p(C), p(al), p(be),
            ctypes.c_int64(cols))
        return C
    a, b, c = A.reshape(-1), B.reshape(-1), C.reshape(-1)
    for m, n, k, ao, lda, bo, ldb, co, ldc in tasks.tolist():
        if m == 0 or n == 0:
            continue
        At = np.lib.stride_tricks.as_strided(a[ao:], (m, k), (lda * a.itemsize, a.itemsize))
        Bt = np.lib.stride_tricks.as_strided(b[bo:], (k, n), (ldb * b.itemsize, b.itemsize))
        Ct = np.lib.stride_tricks.as_strided(c[co:], (m, n), (ldc * c.itemsize, c.itemsize))
        Ct[...] = alpha * (At @ Bt) + beta * Ct
    return C


def gemm_slices(A, B, C, alpha=1.0, beta=0.0):
    '''C[i] = alpha * A @ B[:, i, :] + beta * C[i] for all i, in place: A
    (m, k), B (k, nb, n) and C (nb, m, n) C-contiguous float64.  The slices
    run on separate threads (one sequential GEMM each), which suits many
    small products whose result must land transposed in the first two axes
    of a (k-independent) array.'''
    k, nb, n = B.shape
    m = A.shape[0]
    assert A.shape == (m, k) and C.shape == (nb, m, n)
    assert B.flags.c_contiguous and C.flags.c_contiguous and B.dtype == C.dtype == np.float64
    A = np.ascontiguousarray(A, dtype=np.float64)
    lib = _clib()
    if lib is not None and hasattr(lib, 'zc_gemm_slices_d') and nb and m and n and k:
        al, be = np.array(alpha, dtype=np.float64), np.array(beta, dtype=np.float64)
        p = lambda x: ctypes.c_void_p(x.ctypes.data)          # noqa: E731
        i = ctypes.c_int64
        lib.zc_gemm_slices_d(ctypes.c_int(0), i(nb), i(m), i(n), i(k), p(al), p(A), i(k),
                             p(B), i(n), i(nb * n), p(be), p(C), i(m * n), i(n))
        return C
    for j in range(nb):
        gemm(A, B[:, j, :], C[j], alpha=alpha, beta=beta)
    return C


def gemm_each(A, B, C):
    '''C[i] = A[i] @ B[i] for all i, in place: A (nb, m, k), B (nb, k, n) and
    C (nb, m, n), C-contiguous and of one type.  With the kernel library the
    products run on separate threads, one sequential GEMM each - for many
    products of a few milliseconds; otherwise (or for fewer products than
    two per thread) one after the other on the threaded BLAS.'''
    nb, m, k = A.shape
    n = B.shape[2]
    assert B.shape == (nb, k, n) and C.shape == (nb, m, n) and A.dtype == B.dtype == C.dtype
    assert A.flags.c_contiguous and B.flags.c_contiguous and C.flags.c_contiguous
    lib = _clib()
    name = 'zc_gemm_each_' + _SFX[C.dtype]
    if lib is not None and hasattr(lib, name) and nb >= 2 * _threads() and m and n and k:
        p = lambda x: ctypes.c_void_p(x.ctypes.data)          # noqa: E731
        i = ctypes.c_int64
        getattr(lib, name)(i(nb), i(m), i(n), i(k), p(A), i(m * k), i(k), p(B), i(k * n), i(n),
                           p(C), i(m * n), i(n))
        return C
    for j in range(nb):
        gemm(A[j], B[j], C[j])
    return C


def _threads():
    try:
        return int(os.environ.get('OMP_NUM_THREADS') or len(os.sched_getaffinity(0)))
    except (ValueError, AttributeError):
        return 8
