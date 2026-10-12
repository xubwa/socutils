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
Four-component RKB orbitals in terms of real spherical AOs and their
gradients, and 4c MO two-electron integrals for any kernel
(1/r12, Slater, Yukawa) assembled from the classes of socutils.rf12.ints4c.

pyscf DHF conventions: mo_coeff = [C^L; C^S], large basis = 2-spinors
chi_m = sum_nu phi_nu (ca[nu,m] alpha + cb[nu,m] beta), small basis =
sigma.p chi_m / (2c), p = -i grad.

An orbital psi is stored as
    L[s, nu]      : large component  sum_nu L[s,nu] phi_nu        (s = alpha, beta)
    D[s, d, nu]   : small component  sum_{d,nu} D[s,d,nu] d_d phi_nu
with, for a 2-spinor  a phi alpha + b phi beta  (a, b = C^S coefficients / 2c),
    sigma.p (a phi alpha + b phi beta) = -i { alpha [a dz + b (dx - i dy)]
                                            + beta  [a (dx + i dy) - b dz] } phi
'''

import contextlib
import numpy
from pyscf import lib
from socutils.rf12 import ints4c


@contextlib.contextmanager
def light_speed(c):
    '''Set the speed of light for pyscf DHF.  Also switches off pyscf's
    removal of small overlap eigenvalues: the RKB small-component metric is
    T/(4c^2), which falls below its 1e-6 threshold for large c (or diffuse
    functions) although it is only a diagonal rescaling of the basis.'''
    from pyscf.scf import hf
    old = lib.param.LIGHT_SPEED, hf.remove_overlap_zero_eigenvalue
    lib.param.LIGHT_SPEED = c
    hf.remove_overlap_zero_eigenvalue = False
    try:
        yield
    finally:
        lib.param.LIGHT_SPEED, hf.remove_overlap_zero_eigenvalue = old


class Orbitals:
    '''RKB orbital set in the sph/derivative representation.
    Xi: small-component generator, psi^S = sigma.p Xi (Xi includes 1/2c)'''
    def __init__(self, L, D, Xi=None):
        self.L = L          # (n, 2, nao)
        self.D = D          # (n, 2, 3, nao)
        self.Xi = Xi        # (n, 2, nao)

    def __len__(self):
        return self.L.shape[0]

    def __getitem__(self, idx):
        return Orbitals(self.L[idx], self.D[idx], None if self.Xi is None else self.Xi[idx])

    @staticmethod
    def concatenate(sets):
        return Orbitals(numpy.concatenate([s.L for s in sets]), numpy.concatenate([s.D for s in sets]),
                        numpy.concatenate([s.Xi for s in sets]))


def orbitals(mol, mo, c):
    '''mo: (2 n2c, n) pyscf DHF coefficients (large; small)'''
    ca, cb = mol.sph2spinor_coeff()
    n2c = ca.shape[1]
    CL, CS = mo[:n2c], mo[n2c:]
    L = numpy.stack([numpy.dot(ca, CL).T, numpy.dot(cb, CL).T], axis=1)
    a = numpy.dot(ca, CS).T / (2 * c)
    b = numpy.dot(cb, CS).T / (2 * c)
    n, nao = a.shape
    D = numpy.zeros((n, 2, 3, nao), dtype=complex)
    D[:, 0, 0] = -1j * b
    D[:, 0, 1] = -b
    D[:, 0, 2] = -1j * a
    D[:, 1, 0] = -1j * a
    D[:, 1, 1] = a
    D[:, 1, 2] = 1j * b
    return Orbitals(L, D, numpy.stack([a, b], axis=1))


def _bra(o, comp):
    '''conjugated coefficients, rows (orbital, spin), for the bra position'''
    if comp == 'L':
        return o.L.conj().reshape(-1, 1, o.L.shape[-1])
    return o.D.conj().reshape(-1, 3, o.D.shape[-1])


def _ket(o, comp):
    if comp == 'L':
        return o.L[:, :, None, :]        # (n, 2, 1, nao)
    return o.D                           # (n, 2, 3, nao)


def mo_eri(mol, kernel, k, l, x, y, zeta=None, blocks=('LL', 'SL', 'LS', 'SS')):
    '''<k l | K | x y> (physicist) for RKB orbital sets k, l, x, y.

    Electron 1: k -> x, electron 2: l -> y.  Blocks label the components of
    electron 1 then electron 2 (S = small).  Spin is conserved per electron.'''
    nk, nl, nx, ny = len(k), len(l), len(x), len(y)
    out = numpy.zeros((nk, nl, nx, ny), dtype=complex)
    for blk in blocks:
        c1, c2 = blk
        if c1 == 'L' and c2 == 'L':
            T = ints4c.contract2e(mol, kernel, 'plain', _bra(k, 'L'), _bra(l, 'L'), zeta)
        elif c1 == 'S' and c2 == 'L':
            T = ints4c.contract2e(mol, kernel, 'ipvip1', _bra(k, 'S'), _bra(l, 'L'), zeta)
        elif c1 == 'L' and c2 == 'S':
            # swap the electrons: (l^S y^S | k^L x^L)
            T = ints4c.contract2e(mol, kernel, 'ipvip1', _bra(l, 'S'), _bra(k, 'L'), zeta)
            T = T.transpose(3, 4, 5, 0, 1, 2)
        else:
            T = ints4c.contract2e(mol, kernel, 'ipvip1ipvip2', _bra(k, 'S'), _bra(l, 'S'), zeta)
        # T[(k,s), cj, j, (l,s'), cl, l]
        nao = T.shape[2]
        T = T.reshape(nk, 2, T.shape[1], nao, nl, 2, T.shape[4], nao)
        X = _ket(x, c1)
        Y = _ket(y, c2)
        out += numpy.einsum('ksajlbcm,xsaj,ybcm->klxy', T, X, Y, optimize=True)
    return out


def no_pair_spaces(mf):
    '''occupied and positive-energy virtual columns of a pyscf DHF object'''
    n2c = mf.mol.nao_2c()
    nocc = mf.mol.nelectron
    occ = numpy.arange(n2c, n2c + nocc)
    vir = numpy.arange(n2c + nocc, 2 * n2c)
    return occ, vir


def mp2(mf, c):
    '''Dirac-Coulomb no-pair MP2 (positive-energy virtuals only) of a closed-
    shell DHF reference; spin-orbital form E = 1/4 sum |<ij||ab>|^2 / D.'''
    mol = mf.mol
    occ, vir = no_pair_spaces(mf)
    o = orbitals(mol, mf.mo_coeff[:, occ], c)
    v = orbitals(mol, mf.mo_coeff[:, vir], c)
    g = mo_eri(mol, 'G', o, o, v, v)
    g = g - g.transpose(0, 1, 3, 2)
    e = mf.mo_energy
    eo, ev = e[occ], e[vir]
    D = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    return 0.25 * numpy.sum(abs(g) ** 2 / D).real
