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
Two-electron integrals over 2-spinor functions with one-particle operators
applied to the functions:

    <op1 b1, op2 b2 | K | op3 k1, op4 k2>,   op in {'L': 1, 'lap': -nabla^2, 'sp': sigma.p}

A 2-spinor set is a coefficient array C[n, s, nu] (s = alpha, beta; nu = sph AO).
At most one function per electron may carry an operator (enough for ansatz
B, rf12/notes/stage3_derivation.md Sec. 4).  The operator-carrying function
is placed at the libcint position that the derivative class differentiates
(i for electron 1, k for electron 2; electrons are swapped when only
electron 2 carries one) and the contraction mode of RF12_contract2e is chosen
so that the requested functions are contracted.

    sigma.p (a phi alpha + b phi beta) = -i { alpha [a dz + b (dx - i dy)]
                                            + beta  [a (dx + i dy) - b dz] } phi
'''

import numpy
from socutils.rf12 import ints4c

CLASSES = {(None, None): 'plain', ('lap', None): 'ipip1', ('sp', None): 'ip1',
           ('lap', 'lap'): 'ipip1ipip2', ('sp', 'sp'): 'ip1ip2'}


def _sp(C):
    '''sigma.p applied to 2-spinors C[n,s,nu] -> D[n,s,d,nu] (derivative comps x,y,z)'''
    a, b = C[:, 0], C[:, 1]
    D = numpy.zeros(C.shape[:2] + (3,) + C.shape[2:], dtype=complex)
    D[:, 0, 0] = -1j * b
    D[:, 0, 1] = -b
    D[:, 0, 2] = -1j * a
    D[:, 1, 0] = -1j * a
    D[:, 1, 1] = a
    D[:, 1, 2] = 1j * b
    return D


def _expand(C, op):
    '''coefficients per derivative component: [n, s, comp, nu]'''
    C = numpy.asarray(C, dtype=complex)
    if op == 'L':
        return C[:, :, None, :]
    if op == 'sp':
        return _sp(C)
    if op == 'lap':
        out = numpy.zeros(C.shape[:2] + (9,) + C.shape[2:], dtype=complex)
        for d in (0, 4, 8):
            out[:, :, d] = -C
        return out
    raise KeyError(op)


def eri2(mol, kernel, b1, b2, k1, k2, zeta=None, contract=('b', 'b')):
    '''<b1 b2|K|k1 k2> for (C, op) pairs; contract[e] in {'b', 'k'} selects
    which function of electron e is contracted on the fly (the smaller set).'''
    sets = {'b1': b1, 'k1': k1, 'b2': b2, 'k2': k2}
    t1 = [o for o in ('b1', 'k1') if sets[o][1] != 'L']
    t2 = [o for o in ('b2', 'k2') if sets[o][1] != 'L']
    assert len(t1) <= 1 and len(t2) <= 1, 'one operator per electron'
    swap = (not t1) and bool(t2)
    if swap:
        # <b1 b2|K|k1 k2> = <b2 b1|K|k2 k1>
        out = eri2(mol, kernel, b2, b1, k2, k1, zeta, contract=contract[::-1])
        return out.transpose(1, 0, 3, 2)
    op1 = sets[t1[0]][1] if t1 else None
    op2 = sets[t2[0]][1] if t2 else None
    cls = CLASSES[(op1, op2)]
    # electron 1: the operator function (or b1) sits at i, the other at j
    e1_i = t1[0] if t1 else 'b1'
    e1_j = 'k1' if e1_i == 'b1' else 'b1'
    e2_k = t2[0] if t2 else 'b2'
    e2_l = 'k2' if e2_k == 'b2' else 'b2'
    c1 = contract[0] + '1'
    c2 = contract[1] + '2'
    bit1 = 0 if c1 == e1_i else 1
    bit2 = 0 if c2 == e2_k else 1
    mode = bit1 | (bit2 << 1)

    def coef(name):
        C, op = sets[name]
        E = _expand(C, op)
        return E.conj() if name[0] == 'b' else E

    X1, Y1 = coef(c1), coef(e1_j if c1 == e1_i else e1_i)
    X2, Y2 = coef(c2), coef(e2_l if c2 == e2_k else e2_k)
    n1, n2 = X1.shape[0], X2.shape[0]
    nao = mol.nao
    A = X1.reshape(n1 * 2, X1.shape[2], nao)
    B = X2.reshape(n2 * 2, X2.shape[2], nao)
    T = ints4c.contract2e(mol, kernel, cls, A, B, zeta, mode=mode)
    T = T.reshape(n1, 2, T.shape[1], nao, n2, 2, T.shape[4], nao)
    res = numpy.einsum('xsajytbm,usaj,vtbm->xyuv', T, Y1, Y2, optimize=True)
    # res[x1, x2, y1, y2] -> order <b1 b2|k1 k2>
    o1 = (c1, 'k1' if c1 == 'b1' else 'b1')
    o2 = (c2, 'k2' if c2 == 'b2' else 'b2')
    labels = [o1[0], o2[0], o1[1], o2[1]]
    perm = [labels.index(x) for x in ('b1', 'b2', 'k1', 'k2')]
    return res.transpose(perm)
