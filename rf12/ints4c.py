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
Two-electron integrals with derivative components (libcint, incl. the F12
classes of lib/rf12/f12ints.c) and the on-the-fly bra contraction driver
RF12_contract2e.

All four-component RKB integrals are built from real spherical-AO integrals
with up to one derivative per basis function:
    small component of a 2-spinor basis function = sigma.p (phi_nu chi_s)
so every spinor integral is a contraction of
    (d_ci i  d_cj j | K | d_ck k  d_cl l),  d in {1} or {x, y, z}
with complex coefficients carrying spin and Pauli matrices.

Classes (nci, ncj, nck, ncl):
    plain          (1,1,1,1)   int2e_sph, int2e_stg_sph, int2e_yp_sph
    ipvip1         (3,3,1,1)   (grad i grad j | K | k l)
    ipvip1ipvip2   (3,3,3,3)   (grad i grad j | K | grad k grad l)
    ipip1          9 comps on i (grad grad i j | ...)
    ipip1ipip2     9 x 9 comps
'''

import ctypes
import os
import numpy
import pyscf.lib
from pyscf import gto
from socutils.lib import load_library

_depdir = os.path.join(os.path.dirname(pyscf.lib.__file__), 'deps', 'lib')
libcint = ctypes.CDLL(os.path.join(_depdir, 'libcint.so'))
librf = load_library('librf12_f12', 'SOCUTILS_RF12_F12_LIBRARY')

KERNELS = {'G': '', 'F': 'stg_', 'Y': 'yp_'}     # 1/r12, exp(-z r12), exp(-z r12)/r12
CLASSES = {'plain': ('', (1, 1, 1, 1)),
           'ipvip1': ('ipvip1', (3, 3, 1, 1)),
           'ipvip1ipvip2': ('ipvip1ipvip2', (3, 3, 3, 3)),
           'ip1': ('ip1', (3, 1, 1, 1)),
           'ip1ip2': ('ip1ip2', (3, 1, 3, 1)),
           'ipip1': ('ipip1', (9, 1, 1, 1)),
           'ipip1ipip2': ('ipip1ipip2', (9, 1, 9, 1))}


def _fn(kernel, cls):
    suffix, nc = CLASSES[cls]
    name = 'int2e_' + KERNELS[kernel] + suffix
    name = name.rstrip('_')
    for lib in (librf, libcint):
        try:
            f = getattr(lib, name + '_sph')
            o = getattr(lib, name + '_optimizer')
            return f, o, nc, name
        except AttributeError:
            continue
    raise KeyError(name)


def _c(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def contract2e(mol, kernel, cls, A, B, zeta=None, mode=0):
    '''T[a, cy1, y1, b, cy2, y2] = sum A[a, cx1, x1] B[b, cx2, x2] (ci i cj j|K|ck k cl l)

    mode bit 0: electron 1 contracts x1 = i (0) or j (1); bit 1: electron 2
    contracts x2 = k (0) or l (1); the other function of each pair is open.
    mode 0: contract (i, k); 3: (j, l); 1: (j, k); 2: (i, l).
    A: (na, ncx1, nao) complex, B: (nb, ncx2, nao) complex.'''
    f, o, nc, name = _fn(kernel, cls)
    nci, ncj, nck, ncl = nc
    nao = mol.nao
    ncx1, ncy1 = (ncj, nci) if mode & 1 else (nci, ncj)
    ncx2, ncy2 = (ncl, nck) if mode & 2 else (nck, ncl)
    A = numpy.asarray(A, dtype=complex).reshape(-1, ncx1, nao)
    B = numpy.asarray(B, dtype=complex).reshape(-1, ncx2, nao)
    na, nb = A.shape[0], B.shape[0]
    env = mol._env.copy()
    if zeta is not None:
        env[gto.PTR_F12_ZETA] = zeta
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int32)
    opt = ctypes.c_void_p()
    o(ctypes.byref(opt), _c(atm), ctypes.c_int(mol.natm), _c(bas), ctypes.c_int(mol.nbas), _c(env))
    Are = numpy.ascontiguousarray(A.real)
    Aim = numpy.ascontiguousarray(A.imag)
    Bre = numpy.ascontiguousarray(B.real)
    Bim = numpy.ascontiguousarray(B.imag)
    Tre = numpy.zeros((na, ncy1, nao, nb, ncy2, nao))
    Tim = numpy.zeros_like(Tre)
    librf.RF12_contract2e(ctypes.c_void_p(ctypes.cast(f, ctypes.c_void_p).value), opt, ctypes.c_int(mode),
                          ctypes.c_int(nci), ctypes.c_int(ncj), ctypes.c_int(nck), ctypes.c_int(ncl),
                          _c(Are), _c(Aim), ctypes.c_int(na), _c(Bre), _c(Bim), ctypes.c_int(nb),
                          _c(Tre), _c(Tim), _c(ao_loc), _c(atm), ctypes.c_int(mol.natm),
                          _c(bas), ctypes.c_int(mol.nbas), _c(env))
    libcint.CINTdel_optimizer(ctypes.byref(opt))
    return Tre + 1j * Tim


def full(mol, kernel, cls, zeta=None):
    '''All AO integrals of a class (small molecules only):
    I[ci, cj, ck, cl, i, j, k, l]'''
    f, o, nc, name = _fn(kernel, cls)
    nci, ncj, nck, ncl = nc
    nao = mol.nao
    eye = numpy.zeros((nci * nao, nci, nao))
    for c in range(nci):
        eye[c * nao:(c + 1) * nao, c, :] = numpy.eye(nao)
    eyek = numpy.zeros((nck * nao, nck, nao))
    for c in range(nck):
        eyek[c * nao:(c + 1) * nao, c, :] = numpy.eye(nao)
    T = contract2e(mol, kernel, cls, eye, eyek, zeta)
    T = T.real.reshape(nci, nao, ncj, nao, nck, nao, ncl, nao)
    return T.transpose(0, 2, 4, 6, 1, 3, 5, 7)
