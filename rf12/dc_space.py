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
RKB complete space "1" for Dirac-Coulomb no-pair F12 (stage 3).

"1" = RKB span of OBS + CABS basis functions (large: 2-spinors, small:
sigma.p chi / 2c), orthonormal, in the order
    O   occupied DHF spinors (positive energy)
    V+  OBS positive-energy virtuals
    C+  Lambda+("1") minus (O + V+): Lambda+ = span of the positive-energy
        eigenvectors of the DHF Fock operator in the whole of "1"
        (Furry picture in the complete space), canonicalized
    N   the rest of "1" (OBS and CABS negative-energy states)
The no-pair strong-orthogonality projector is
    Q12 = (V+ + C+)(V+ + C+) - V+ V+ = (1 - O')(1 - O') - V+ V+,  O' = O + N.

Note (found with the c -> infinity singles anchor): defining C+- instead by
diagonalizing the Fock operator inside the complement of *all* OBS orbitals
forces the complement to be orthogonal to the OBS negative-energy states;
that admixes O(1/c) negative-energy character and shifts the CABS energies
by O(1) even for c -> infinity (Brown-Ravenhall-type contamination).
'''

import numpy
from functools import reduce
from pyscf import gto, scf
from socutils.rf12 import fourc


def make_ri_mol(mol, cabs_basis):
    auxmol = mol.copy()
    auxmol.basis = cabs_basis
    auxmol.build(False, False)
    ri = gto.conc_mol(mol, auxmol)
    ri._atm[mol.natm:, gto.CHARGE_OF] = 0       # conc_mol duplicates the atoms
    return ri


class DCSpace:
    def __init__(self, mf, cabs_basis, c, lindep=1e-8, lindep_cabs=1e-6):
        mol = mf.mol
        self.c = c
        ri = make_ri_mol(mol, cabs_basis)
        self.ri_mol = ri
        n2c, N2c = mol.nao_2c(), ri.nao_2c()
        # OBS DHF orbitals embedded in the ri RKB AO basis [L(N2c); S(N2c)]
        Cobs = numpy.zeros((2 * N2c, mf.mo_coeff.shape[1]), dtype=complex)
        Cobs[:n2c] = mf.mo_coeff[:n2c]
        Cobs[N2c:N2c + n2c] = mf.mo_coeff[n2c:]
        with fourc.light_speed(c):
            mfri = scf.DHF(ri)
            S = mfri.get_ovlp()
            h = mfri.get_hcore()
            dm = numpy.zeros((2 * N2c, 2 * N2c), dtype=complex)
            dm[:n2c, :n2c] = mf.make_rdm1()[:n2c, :n2c]
            dm[:n2c, N2c:N2c + n2c] = mf.make_rdm1()[:n2c, n2c:]
            dm[N2c:N2c + n2c, :n2c] = mf.make_rdm1()[n2c:, :n2c]
            dm[N2c:N2c + n2c, N2c:N2c + n2c] = mf.make_rdm1()[n2c:, n2c:]
            vj, vk = mfri.get_jk(ri, dm)
        # orthonormal RKB basis of the ri space (blocks scale independently)
        X = []
        for blk in (slice(0, N2c), slice(N2c, 2 * N2c)):
            e, U = numpy.linalg.eigh(S[blk, blk])
            # relative threshold: the small block carries the factor 1/(4c^2)
            keep = e > lindep * e.max()
            Xb = numpy.zeros((2 * N2c, keep.sum()), dtype=complex)
            Xb[blk] = U[:, keep] / numpy.sqrt(e[keep])
            X.append(Xb)
        X = numpy.hstack(X)
        fock_ao = h + vj - vk
        # Lambda+ of the whole "1"
        FX = reduce(numpy.dot, (X.conj().T, fock_ao, X))
        eX, uX = numpy.linalg.eigh(FX)
        Ppos = numpy.dot(X, uX[:, eX > -c * c])
        nocc = mol.nelectron
        pos_obs = mf.mo_energy > -c * c
        occ = numpy.arange(n2c, n2c + nocc)
        vplus = numpy.where(pos_obs)[0][nocc:]
        assert numpy.all(occ == numpy.where(pos_obs)[0][:nocc])
        Cov = Cobs[:, numpy.r_[occ, vplus]]
        # C+ = part of Lambda+ orthogonal to O + V+
        A = reduce(numpy.dot, (Cov.conj().T, S, Ppos))
        u, sv, vh = numpy.linalg.svd(A)
        nnz = numpy.count_nonzero(sv > lindep_cabs)
        Cp = numpy.dot(Ppos, vh[nnz:].conj().T)
        ec, uc = numpy.linalg.eigh(reduce(numpy.dot, (Cp.conj().T, fock_ao, Cp)))
        Cp = numpy.dot(Cp, uc)
        self.lambda_leak = numpy.sqrt(max(0, len(sv) - (sv ** 2).sum()))   # |O+V+ outside Lambda+|
        # N = orthogonal complement of O + V+ + C+ in "1"
        Cpos = numpy.hstack([Cov, Cp])
        B = reduce(numpy.dot, (Cpos.conj().T, S, X))
        u, sv, vh = numpy.linalg.svd(B)
        nnz = numpy.count_nonzero(sv > lindep_cabs)
        Cn = numpy.dot(X, vh[nnz:].conj().T)
        en, un = numpy.linalg.eigh(reduce(numpy.dot, (Cn.conj().T, fock_ao, Cn)))
        Cn = numpy.dot(Cn, un)
        C1 = numpy.hstack([Cov, Cp, Cn])
        self.C1 = C1
        n = [len(occ), len(vplus), Cp.shape[1], Cn.shape[1]]
        off = numpy.cumsum([0] + n)
        self.slices = {k: slice(off[i], off[i + 1]) for i, k in enumerate(('o', 'v+', 'c+', 'n'))}
        self.dims = dict(zip(('o', 'v+', 'c+', 'n'), n))
        self.nri = C1.shape[1]
        self.vj_ao = vj
        self.vk_ao = vk
        self.fock = reduce(numpy.dot, (C1.conj().T, fock_ao, C1))
        self.K = reduce(numpy.dot, (C1.conj().T, vk, C1))
        self.metric_error = abs(reduce(numpy.dot, (C1.conj().T, S, C1)) - numpy.eye(self.nri)).max()
        # projector masks in "1"
        self.mask_O = numpy.zeros(self.nri, bool)
        self.mask_Vp = numpy.zeros(self.nri, bool)
        self.mask_neg = numpy.zeros(self.nri, bool)
        self.mask_Cp = numpy.zeros(self.nri, bool)
        self.mask_O[self.slices['o']] = True
        self.mask_Vp[self.slices['v+']] = True
        self.mask_neg[self.slices['n']] = True
        self.mask_Cp[self.slices['c+']] = True

    def orbitals(self, cols):
        '''RKB orbitals (sph/derivative form) of "1" columns'''
        return fourc.orbitals(self.ri_mol, self.C1[:, cols], self.c)

    def cabs_singles(self):
        '''no-pair CABS singles: occupied vs V+ + C+ (negative-energy states excluded)'''
        F = self.fock
        o = self.slices['o']
        a = numpy.where(self.mask_Vp | self.mask_Cp)[0]
        eo, uo = numpy.linalg.eigh(F[o, o])
        ea, ua = numpy.linalg.eigh(F[numpy.ix_(a, a)])
        foa = reduce(numpy.dot, (uo.conj().T, F[o][:, a], ua))
        return numpy.sum(abs(foa) ** 2 / (eo[:, None] - ea[None, :])).real
