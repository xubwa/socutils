#!/usr/bin/env python
# Copyright 2026 The socutils developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Minimal quadruple precision (IEEE binary128) array type.

The numbers are stored bit-exactly in numpy arrays of dtype ``V16``.  Numpy
is used only to move the data around (reshape, transpose, slicing); all
arithmetic is carried out by the __float128 kernels in libqdhf
(socutils/lib/qdhf).  Complex
arrays carry two real arrays for the real and imaginary parts.

Scalars are exchanged with Python as :class:`mpmath.mpf` numbers of the
context :data:`qmp` (113-bit mantissa, identical to binary128).
'''

import ctypes
import numbers
import numpy
import mpmath
from socutils.lib import load_library

libqd = load_library('libqdhf', 'SOCUTILS_QDHF_LIBRARY')

QDTYPE = numpy.dtype('V16')

# mpmath context with the binary128 mantissa
qmp = mpmath.MPContext()
qmp.prec = 113

# mpmath context with a few guard bits, used to set up the input data
hmp = mpmath.MPContext()
hmp.prec = 160

_SLOT = 64


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _raw_from_double(x):
    x = numpy.ascontiguousarray(x, dtype=numpy.double)
    out = numpy.empty(x.shape, dtype=QDTYPE)
    libqd.QDd2q(_ptr(out), _ptr(x), ctypes.c_long(x.size))
    return out


def _raw_from_strings(strs, shape):
    out = numpy.empty(shape, dtype=QDTYPE)
    if out.size == 0:
        return out
    buf = ' '.join(strs).encode()
    ierr = libqd.QDparse(_ptr(out), ctypes.c_char_p(buf), ctypes.c_long(out.size))
    if ierr != 0:
        raise ValueError('Failed to parse quad precision numbers')
    return out


def _mpf_to_str(x):
    return mpmath.libmp.to_str(x._mpf_, 42)


def _raw_from_mpf(x):
    x = numpy.asarray(x, dtype=object)
    strs = [_mpf_to_str(hmp.mpf(v)) for v in x.ravel()]
    return _raw_from_strings(strs, x.shape)


def _raw_to_strings(raw):
    raw = numpy.ascontiguousarray(raw)
    n = raw.size
    buf = ctypes.create_string_buffer(n * _SLOT)
    libqd.QDformat(buf, _ptr(raw), ctypes.c_long(n))
    b = buf.raw
    return [b[i*_SLOT:(i+1)*_SLOT].split(b'\0', 1)[0].decode() for i in range(n)]


def _scalar_raw(x):
    '''Convert a Python/mpmath scalar to a 0-d raw quad array'''
    if isinstance(x, QArray):
        assert x.im is None and x.size == 1
        return numpy.ascontiguousarray(x.re.reshape(()))
    if isinstance(x, numbers.Integral):
        if abs(x) < 2**53:
            return _raw_from_double(float(x))
        return _raw_from_strings([str(x)], ())
    if isinstance(x, float):
        return _raw_from_double(x)
    if type(x).__name__ == 'mpf':
        return _raw_from_strings([_mpf_to_str(hmp.mpf(x))], ())
    if isinstance(x, str):
        return _raw_from_strings([x], ())
    raise TypeError('Unsupported scalar type %s' % type(x))


_ONE = None
def _one():
    global _ONE
    if _ONE is None:
        _ONE = _raw_from_double(1.)
    return _ONE


def _axpby(x, y, a, b):
    '''a * x + b * y on raw arrays.  y can be None'''
    x = numpy.ascontiguousarray(x)
    out = numpy.empty(x.shape, dtype=QDTYPE)
    if y is None:
        libqd.QDaxpby(_ptr(out), _ptr(x), None, _ptr(a), None,
                      ctypes.c_long(x.size))
    else:
        y = numpy.ascontiguousarray(y)
        assert x.shape == y.shape
        libqd.QDaxpby(_ptr(out), _ptr(x), _ptr(y), _ptr(a), _ptr(b),
                      ctypes.c_long(x.size))
    return out


def _neg(x):
    x = numpy.ascontiguousarray(x)
    out = numpy.empty(x.shape, dtype=QDTYPE)
    libqd.QDunary(_ptr(out), _ptr(x), ctypes.c_long(x.size), ctypes.c_int(3))
    return out


def _unary(x, op):
    x = numpy.ascontiguousarray(x)
    out = numpy.empty(x.shape, dtype=QDTYPE)
    libqd.QDunary(_ptr(out), _ptr(x), ctypes.c_long(x.size), ctypes.c_int(op))
    return out


def _mul(x, y):
    x = numpy.ascontiguousarray(x)
    y = numpy.ascontiguousarray(y)
    assert x.shape == y.shape
    out = numpy.empty(x.shape, dtype=QDTYPE)
    libqd.QDmul(_ptr(out), _ptr(x), _ptr(y), ctypes.c_long(x.size))
    return out


def _dot_raw(x, y):
    x = numpy.ascontiguousarray(x)
    y = numpy.ascontiguousarray(y)
    assert x.size == y.size
    out = numpy.empty((), dtype=QDTYPE)
    libqd.QDdot(_ptr(out), _ptr(x), _ptr(y), ctypes.c_long(x.size))
    return out


def _gemm(a, b, c=None, alpha=None, beta=None, trans_a=False, trans_b=False):
    '''c = alpha * op(a) @ op(b) + beta * c on raw 2D arrays'''
    a = numpy.ascontiguousarray(a)
    b = numpy.ascontiguousarray(b)
    if trans_a:
        k, m = a.shape
    else:
        m, k = a.shape
    if trans_b:
        n, k1 = b.shape
    else:
        k1, n = b.shape
    assert k == k1
    if alpha is None:
        alpha = _one()
    if c is None:
        c = numpy.empty((m, n), dtype=QDTYPE)
        beta = None
    else:
        assert c.flags.c_contiguous and c.shape == (m, n)
    if k == 0:
        if beta is None:
            c[:] = numpy.zeros((), dtype=QDTYPE)
        return c
    libqd.QDgemm(ctypes.c_int(trans_a), ctypes.c_int(trans_b),
                 ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
                 _ptr(alpha), _ptr(a), _ptr(b),
                 None if beta is None else _ptr(beta), _ptr(c))
    return c


class QArray:
    '''Quad precision array.  im is None for real arrays.'''

    __array_priority__ = 100

    def __init__(self, re, im=None):
        self.re = numpy.ascontiguousarray(re)
        if im is not None:
            im = numpy.ascontiguousarray(im)
            assert im.shape == self.re.shape
        self.im = im
        assert self.re.dtype == QDTYPE

    # ---------------------------------------------------------- creation
    @classmethod
    def zeros(cls, shape, iscomplex=False):
        re = numpy.zeros(shape, dtype=QDTYPE)
        im = numpy.zeros(shape, dtype=QDTYPE) if iscomplex else None
        return cls(re, im)

    @classmethod
    def eye(cls, n):
        return cls.from_numpy(numpy.eye(n))

    @classmethod
    def from_numpy(cls, a):
        '''Exact conversion of a double precision real/complex array'''
        a = numpy.asarray(a)
        if numpy.iscomplexobj(a):
            return cls(_raw_from_double(a.real), _raw_from_double(a.imag))
        return cls(_raw_from_double(a))

    @classmethod
    def from_mpf(cls, a):
        '''Conversion of an array of mpmath real numbers'''
        return cls(_raw_from_mpf(a))

    @classmethod
    def from_mpc(cls, a):
        a = numpy.asarray(a, dtype=object)
        re = numpy.vectorize(lambda x: hmp.mpf(hmp.re(x)), otypes=[object])(a)
        im = numpy.vectorize(lambda x: hmp.mpf(hmp.im(x)), otypes=[object])(a)
        return cls(_raw_from_mpf(re), _raw_from_mpf(im))

    @classmethod
    def scalar(cls, x):
        return cls(_scalar_raw(x))

    # ---------------------------------------------------------- conversion
    def to_numpy(self):
        re = numpy.empty(self.shape)
        libqd.QDq2d(_ptr(re), _ptr(self.re), ctypes.c_long(self.size))
        if self.im is None:
            return re
        im = numpy.empty(self.shape)
        libqd.QDq2d(_ptr(im), _ptr(self.im), ctypes.c_long(self.size))
        return re + 1j * im

    def to_mpmath(self):
        '''Object array of mpmath numbers (qmp context)'''
        re = numpy.array([qmp.mpf(s) for s in _raw_to_strings(self.re)],
                         dtype=object).reshape(self.shape)
        if self.im is None:
            return re
        im = numpy.array([qmp.mpf(s) for s in _raw_to_strings(self.im)],
                         dtype=object).reshape(self.shape)
        return re + im * qmp.mpc(0, 1)

    def item(self):
        assert self.size == 1
        v = self.to_mpmath().ravel()[0]
        return v

    def __float__(self):
        assert self.im is None
        return float(self.to_numpy())

    # ---------------------------------------------------------- properties
    @property
    def shape(self):
        return self.re.shape

    @property
    def ndim(self):
        return self.re.ndim

    @property
    def size(self):
        return self.re.size

    @property
    def iscomplex(self):
        return self.im is not None

    @property
    def T(self):
        return self.transpose()

    @property
    def real(self):
        return QArray(self.re)

    @property
    def imag(self):
        if self.im is None:
            return QArray.zeros(self.shape)
        return QArray(self.im)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return 'QArray(shape=%s, complex=%s)' % (self.shape, self.iscomplex)

    # ---------------------------------------------------------- data movement
    def copy(self):
        return QArray(self.re.copy(), None if self.im is None else self.im.copy())

    def reshape(self, *shape):
        im = None if self.im is None else self.im.reshape(*shape)
        return QArray(self.re.reshape(*shape), im)

    def ravel(self):
        return self.reshape(-1)

    def transpose(self, *axes):
        im = None if self.im is None else self.im.transpose(*axes)
        return QArray(self.re.transpose(*axes), im)

    def conj(self):
        if self.im is None:
            return self
        return QArray(self.re, _neg(self.im))

    def __getitem__(self, idx):
        im = None if self.im is None else self.im[idx]
        return QArray(self.re[idx], im)

    def __setitem__(self, idx, val):
        if not isinstance(val, QArray):
            val = QArray.from_numpy(val)
        self.re[idx] = val.re
        if val.im is not None:
            if self.im is None:
                self.im = numpy.zeros(self.shape, dtype=QDTYPE)
            self.im[idx] = val.im
        elif self.im is not None:
            self.im[idx] = numpy.zeros((), dtype=QDTYPE)

    def as_complex(self):
        if self.im is not None:
            return self
        return QArray(self.re, numpy.zeros(self.shape, dtype=QDTYPE))

    # ---------------------------------------------------------- arithmetic
    def _binary(self, other, sa, sb):
        if not isinstance(other, QArray):
            other = QArray.from_numpy(other)
        a = _scalar_raw(sa)
        b = _scalar_raw(sb)
        re = _axpby(self.re, other.re, a, b)
        if self.im is None and other.im is None:
            return QArray(re)
        im1 = self.im if self.im is not None else numpy.zeros(self.shape, dtype=QDTYPE)
        im2 = other.im if other.im is not None else numpy.zeros(self.shape, dtype=QDTYPE)
        return QArray(re, _axpby(im1, im2, a, b))

    def __add__(self, other):
        return self._binary(other, 1, 1)

    def __radd__(self, other):
        return self._binary(other, 1, 1)

    def __sub__(self, other):
        return self._binary(other, 1, -1)

    def __rsub__(self, other):
        return self._binary(other, -1, 1)

    def __neg__(self):
        im = None if self.im is None else _neg(self.im)
        return QArray(_neg(self.re), im)

    def __mul__(self, other):
        if isinstance(other, QArray) and other.size != 1 or \
           isinstance(other, numpy.ndarray):
            if not isinstance(other, QArray):
                other = QArray.from_numpy(other)
            assert other.shape == self.shape
            if self.im is None and other.im is None:
                return QArray(_mul(self.re, other.re))
            a = self.as_complex()
            b = other.as_complex()
            one = _one()
            mone = _scalar_raw(-1)
            re = _axpby(_mul(a.re, b.re), _mul(a.im, b.im), one, mone)
            im = _axpby(_mul(a.re, b.im), _mul(a.im, b.re), one, one)
            return QArray(re, im)
        if isinstance(other, complex) or type(other).__name__ == 'mpc':
            xr = other.real
            xi = other.imag
            a = self.as_complex()
            re = _axpby(a.re, a.im, _scalar_raw(xr), _scalar_raw(-xi))
            im = _axpby(a.re, a.im, _scalar_raw(xi), _scalar_raw(xr))
            return QArray(re, im)
        s = _scalar_raw(other)
        im = None if self.im is None else _axpby(self.im, None, s, None)
        return QArray(_axpby(self.re, None, s, None), im)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, QArray):
            assert other.size == 1 and other.im is None
            inv = _unary(other.re.reshape(()), 1)
            return self * QArray(inv)
        return self * (hmp.mpf(1) / hmp.mpf(other))

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(QArray.from_numpy(other), self)

    # ---------------------------------------------------------- reductions
    def sum(self):
        re = numpy.ascontiguousarray(self.re)
        out = numpy.empty((), dtype=QDTYPE)
        libqd.QDsum(_ptr(out), _ptr(re), ctypes.c_long(self.size))
        if self.im is None:
            return QArray(out)
        out1 = numpy.empty((), dtype=QDTYPE)
        libqd.QDsum(_ptr(out1), _ptr(self.im), ctypes.c_long(self.size))
        return QArray(out, out1)

    def max_abs(self):
        '''max(|re|, |im|) as an mpf number'''
        out = numpy.empty((), dtype=QDTYPE)
        libqd.QDmaxabs(_ptr(out), _ptr(self.re), ctypes.c_long(self.size))
        v = QArray(out).item()
        if self.im is not None:
            libqd.QDmaxabs(_ptr(out), _ptr(self.im), ctypes.c_long(self.size))
            v = max(v, QArray(out).item())
        return v

    def norm(self):
        '''Frobenius norm as an mpf number'''
        s = _dot_raw(self.re, self.re)
        v = QArray(s).item()
        if self.im is not None:
            v += QArray(_dot_raw(self.im, self.im)).item()
        return qmp.sqrt(v)

    def sqrt(self):
        assert self.im is None
        return QArray(_unary(self.re, 0))

    def reciprocal(self):
        assert self.im is None
        return QArray(_unary(self.re, 1))


def matmul(a, b, trans_a=False, trans_b=False):
    '''Matrix product of two 2D QArrays'''
    assert a.ndim == 2 and b.ndim == 2
    if a.im is None and b.im is None:
        return QArray(_gemm(a.re, b.re, trans_a=trans_a, trans_b=trans_b))
    one = _one()
    mone = _scalar_raw(-1)
    if a.im is None:
        re = _gemm(a.re, b.re, trans_a=trans_a, trans_b=trans_b)
        im = _gemm(a.re, b.im, trans_a=trans_a, trans_b=trans_b)
    elif b.im is None:
        re = _gemm(a.re, b.re, trans_a=trans_a, trans_b=trans_b)
        im = _gemm(a.im, b.re, trans_a=trans_a, trans_b=trans_b)
    else:
        re = _gemm(a.re, b.re, trans_a=trans_a, trans_b=trans_b)
        _gemm(a.im, b.im, re, mone, one, trans_a=trans_a, trans_b=trans_b)
        im = _gemm(a.re, b.im, trans_a=trans_a, trans_b=trans_b)
        _gemm(a.im, b.re, im, one, one, trans_a=trans_a, trans_b=trans_b)
    return QArray(re, im)


def vdot(a, b):
    '''sum(conj(a) * b) as an mpmath number'''
    re = QArray(_dot_raw(a.re, b.re)).item()
    if a.im is None and b.im is None:
        return re
    zero = qmp.mpf(0)
    rr = re
    ii = QArray(_dot_raw(a.im, b.im)).item() if a.im is not None and b.im is not None else zero
    ri = QArray(_dot_raw(a.re, b.im)).item() if b.im is not None else zero
    ir = QArray(_dot_raw(a.im, b.re)).item() if a.im is not None else zero
    return qmp.mpc(rr + ii, ri - ir)


def trace_dot(a, b):
    '''sum_ij a_ij b_ji = tr(a b) as an mpmath number'''
    return vdot(a.conj(), b.T)


def eigh(a):
    '''Eigenvalues and eigenvectors of a Hermitian matrix (Jacobi method)'''
    n = a.shape[0]
    a = a.as_complex()
    are = a.re.copy()
    aim = a.im.copy()
    w = numpy.empty(n, dtype=QDTYPE)
    vre = numpy.empty((n, n), dtype=QDTYPE)
    vim = numpy.empty((n, n), dtype=QDTYPE)
    nsweep = libqd.QDzheev(ctypes.c_int(n), _ptr(are), _ptr(aim), _ptr(w),
                           _ptr(vre), _ptr(vim))
    if nsweep < 0:
        raise RuntimeError('Quad precision Jacobi diagonalization not converged')
    return QArray(w), QArray(vre, vim)


def concatenate(arrs, axis=0):
    iscomplex = any(x.im is not None for x in arrs)
    if iscomplex:
        arrs = [x.as_complex() for x in arrs]
        return QArray(numpy.concatenate([x.re for x in arrs], axis=axis),
                      numpy.concatenate([x.im for x in arrs], axis=axis))
    return QArray(numpy.concatenate([x.re for x in arrs], axis=axis))


def solve(a, b):
    '''Solve the real linear equations a x = b (Gaussian elimination with
    partial pivoting in quad precision).  b can be 1D or 2D.'''
    assert a.im is None and b.im is None
    n = a.shape[0]
    are = a.re.copy()
    x = numpy.ascontiguousarray(b.re.reshape(n, -1)).copy()
    ierr = libqd.QDgesv(ctypes.c_int(n), ctypes.c_int(x.shape[1]), _ptr(are), _ptr(x))
    if ierr != 0:
        raise numpy.linalg.LinAlgError('Singular matrix')
    return QArray(x.reshape(b.shape))


def diag(w):
    '''Diagonal matrix from a 1D array'''
    n = w.shape[0]
    d = QArray.zeros((n, n), iscomplex=w.im is not None)
    for i in range(n):
        d[i, i] = w[i]
    return d
