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
Two interchangeable dense-array backends with the same small interface:

* ``Double``: numpy float64 + scipy.linalg.eigh
* ``Quad``:   IEEE binary128.  Data are numpy arrays of dtype V16 (bit
  copies of __float128); numpy only moves data (reshape, broadcast,
  index), every arithmetic operation is a libqdhf / librf12 kernel.

The radial / pair code in socutils.rf12 is written once against this
interface, so the double run is an independent cross-check of the quad run.
'''

import ctypes
import numpy
import scipy.linalg
from socutils.lib import load_library
from socutils.qdhf import qarray as _qa


class Double:
    name = 'double'

    def from_mp(self, a):
        return numpy.array(a, dtype=object).astype(float)

    def from_float(self, a):
        return numpy.asarray(a, dtype=float)

    def to_mp(self, a, ctx):
        return [ctx.mpf(float(x)) for x in numpy.asarray(a).ravel()]

    def to_float(self, a):
        return numpy.asarray(a, dtype=float)

    def scalar(self, x):
        return numpy.asarray(float(x))

    def mul(self, a, b):
        return a * b

    def add(self, a, b):
        return a + b

    def axpby(self, a, x, b, y):
        return float(a) * x + float(b) * y

    def scale(self, x, a):
        return float(a) * x

    def recip(self, a):
        return 1. / a

    def matmul(self, a, b, trans_a=False, trans_b=False):
        if trans_a:
            a = a.T
        if trans_b:
            b = b.T
        return a @ b

    def sum(self, a):
        return a.sum()

    def sygv(self, h, s):
        return scipy.linalg.eigh(h, s)

    def zeros(self, shape):
        return numpy.zeros(shape)


class Quad:
    name = 'quad'

    def __init__(self):
        self.lib = load_library('librf12', 'SOCUTILS_RF12_LIBRARY')
        self.libqd = _qa.libqd

    def from_mp(self, a):
        return _qa._raw_from_mpf(numpy.array(a, dtype=object))

    def from_float(self, a):
        return _qa._raw_from_double(a)

    def to_mp(self, a, ctx):
        return [ctx.mpf(s) for s in _qa._raw_to_strings(a)]

    def to_float(self, a):
        a = numpy.ascontiguousarray(a)
        out = numpy.empty(a.shape)
        self.libqd.QDq2d(_qa._ptr(out), _qa._ptr(a), ctypes.c_long(a.size))
        return out

    def scalar(self, x):
        return _qa._scalar_raw(x)

    def _bc(self, a, b):
        shape = numpy.broadcast_shapes(a.shape, b.shape)
        return (numpy.ascontiguousarray(numpy.broadcast_to(a, shape)),
                numpy.ascontiguousarray(numpy.broadcast_to(b, shape)))

    def mul(self, a, b):
        a, b = self._bc(a, b)
        return _qa._mul(a, b)

    def add(self, a, b):
        a, b = self._bc(a, b)
        one = _qa._one()
        return _qa._axpby(a, b, one, one)

    def axpby(self, a, x, b, y):
        x, y = self._bc(x, y)
        return _qa._axpby(x, y, _qa._scalar_raw(a), _qa._scalar_raw(b))

    def scale(self, x, a):
        return _qa._axpby(x, None, _qa._scalar_raw(a), None)

    def recip(self, a):
        return _qa._unary(a, 1)

    def matmul(self, a, b, trans_a=False, trans_b=False):
        return _qa._gemm(a, b, trans_a=trans_a, trans_b=trans_b)

    def sum(self, a):
        out = numpy.empty((), dtype=_qa.QDTYPE)
        a = numpy.ascontiguousarray(a)
        self.libqd.QDsum(_qa._ptr(out), _qa._ptr(a), ctypes.c_long(a.size))
        return out

    def sygv(self, h, s):
        n = h.shape[0]
        h = numpy.ascontiguousarray(h).copy()
        s = numpy.ascontiguousarray(s).copy()
        w = numpy.empty(n, dtype=_qa.QDTYPE)
        ierr = self.lib.QRsygv(ctypes.c_int(n), _qa._ptr(h), _qa._ptr(s), _qa._ptr(w))
        if ierr != 0:
            raise RuntimeError('QRsygv failed, info = %d' % ierr)
        return w, h

    def zeros(self, shape):
        return _qa._raw_from_double(numpy.zeros(shape))
