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
Dirac-Coulomb no-pair MP2-F12, closed-shell DHF reference, brute-force
operator form (stage 3).  Derivation: rf12/notes/stage3_derivation.md.

Pairs are spin-orbital (spinor) pairs i < j of active occupied spinors with
normalized antisymmetric reference pairs Phi_ij = (|ij> - |ji>)/sqrt2.
Ansatz A (Liu, "kinetic balance first"): the geminal acts on all components,
    u_ij = Q12 f12 S12 Phi_ij,     S12 = 3/8 - 1/8 P^Sigma   (on Phi),
P^Sigma = (1 + Sigma1.Sigma2)/2 with Sigma projected onto "1" (Sigma_c applied
to an RKB function leaves the RKB space, sigma_c sigma.p = 2 p_c - sigma.p
sigma_c; the projection changes S12 only at O(1/c^2) and keeps the
non-relativistic singlet 1/2 / triplet 1/4 amplitudes exact).  Expanded,
    S12 Phi_ij = 1/sqrt2 (1 - P12) [ 5/16 |ij> - 1/16 sum_c |Sigma_c i, Sigma_c j> ].
Hylleraas functional with fixed amplitudes, conventional doubles at MP2:
    E_ij = 2 Re Vt_ij + Bt_ij
    Vt = V - sum_{a<b} C_ab <ab||ij> / (e_a + e_b - e_i - e_j)
    Bt = B - (e_i + e_j) X - sum_{a<b} |C_ab|^2 / (e_a + e_b - e_i - e_j)
    V = <g|f Q g|Phi>,  X = <g|f Q f|g>,  C_ab = <g|f Q (F1+F2)|Phi_ab>,
    B = <g|f Q (F1+F2) Q f|g>  (approximation C, Sec. 3 of the notes):
        <f (F1+F2) f>_C = 1/2 <{f^2, (F+K)_1 + (F+K)_2}> - <f (K1+K2) f>
        (4c: [f, [c alpha.p, f]] = 0, no U term), and
        B = <fFf>_C - <f P F f> - <f F P f> + <f P F P f>,  P = 1 - Q12,
all insertions in the complete RKB space "1" (socutils.rf12.dc_space), where
Q12 = (V+ + C+)(V+ + C+) - V+ V+ is a 0/1 mask on pairs of "1" indices.
f12 = -exp(-gamma r12)/gamma.  Integrals: socutils.rf12.fourc.mo_eri.
'''

import numpy
from functools import reduce
from socutils.rf12 import fourc, dc_space

COEF = (5. / 16, -1. / 16, -1. / 16, -1. / 16)     # identity, Sigma_x, Sigma_y, Sigma_z


class DCMP2F12:
    def __init__(self, mf, cabs_basis, c, gamma=1.0, frozen=0, ansatz='A'):
        self.mf = mf
        self.cabs_basis = cabs_basis
        self.c = c
        self.gamma = gamma
        self.frozen = frozen
        self.ansatz = ansatz

    # ------------------------------------------------------------ helpers
    def _pairs(self, nact):
        return [(i, j) for i in range(nact) for j in range(i + 1, nact)]

    def _gem_bra(self, T):
        '''T[t][i, j, ...] (bra orbitals of type t) -> [p, ...] for the
        geminals S12 Phi_p (real coefficients; bra conjugation is in T)'''
        out = 0
        for t, ct in enumerate(COEF):
            if T[t] is None:
                continue
            A = T[t][self._pi, self._pj] - T[t][self._pj, self._pi]
            out = out + ct * A
        return out / numpy.sqrt(2)

    def _gem_ket(self, T):
        '''T[s][..., k, l] (ket orbitals of type s) -> [..., q]'''
        out = 0
        for s, cs in enumerate(COEF):
            A = T[s][..., self._pi, self._pj] - T[s][..., self._pj, self._pi]
            out = out + cs * A
        return out / numpy.sqrt(2)

    def kernel(self):
        c = self.c
        g = self.gamma
        sp = dc_space.DCSpace(self.mf, self.cabs_basis, c)
        self.space = sp
        ri = sp.ri_mol
        C1 = sp.C1
        nri = sp.nri
        no = sp.dims['o']
        nfz = self.frozen
        act = numpy.arange(nfz, no)
        nact = len(act)
        pairs = self._pairs(nact)
        self._pi = numpy.array([p[0] for p in pairs], dtype=int)
        self._pj = numpy.array([p[1] for p in pairs], dtype=int)
        F = sp.fock
        K = sp.K
        e = numpy.diag(F).real
        vp = numpy.where(sp.mask_Vp)[0]

        # geminal orbitals: W_0 = active occupied, W_c = projected Sigma_c W_0
        N2c = ri.nao_2c()
        sig = numpy.zeros((3, 2 * N2c, 2 * N2c), dtype=complex)
        sig[:, :N2c, :N2c] = ri.intor('int1e_sigma_spinor')
        sig[:, N2c:, N2c:] = ri.intor('int1e_spsigmasp_spinor') * (0.5 / c) ** 2
        W = [numpy.eye(nri)[:, act].astype(complex)]
        for x in range(3):
            W.append(reduce(numpy.dot, (C1.conj().T, sig[x], C1))[:, act])
        self.W = W
        O = [fourc.orbitals(ri, numpy.dot(C1, w), c) for w in W]
        ONE = fourc.orbitals(ri, C1, c)
        A0 = O[0]

        # <g_p| f |I J>
        GF = 0
        for t in range(4):
            T = [None] * 4
            T[t] = fourc.mo_eri(ri, 'F', O[t], O[t], ONE, ONE, g) * (-1 / g)
            GF = GF + self._gem_bra(T)
        # <I J| g |Phi_q>  and  <ab||ij>
        G0 = fourc.mo_eri(ri, 'G', A0, A0, ONE, ONE)              # <ij|g|IJ>
        gIJ = (G0[self._pi, self._pj] - G0[self._pj, self._pi]).conj().transpose(1, 2, 0) / numpy.sqrt(2)   # [I, J, q]
        g_ab = (gIJ[numpy.ix_(vp, vp)] - gIJ[numpy.ix_(vp, vp)].transpose(1, 0, 2)) / numpy.sqrt(2)  # [a,b,q] for Phi_ab
        # <g_p| f g |Phi_q>
        T = [fourc.mo_eri(ri, 'Y', O[t], O[t], A0, A0, g) * (-1 / g) for t in range(4)]
        VFG = self._gem_bra(T)
        VFG = (VFG[:, self._pi, self._pj] - VFG[:, self._pj, self._pi]) / numpy.sqrt(2)
        # <g_p| f^2 |g_q>  and  <g_p| f^2 |I, W_s l>
        X0 = 0
        M = []
        for s in range(4):
            Ts = [fourc.mo_eri(ri, 'F', O[t], O[t], O[s], O[s], 2 * g) / g ** 2 for t in range(4)]
            X0 = X0 + COEF[s] * self._gem_bra([(x[..., self._pi, self._pj] - x[..., self._pj, self._pi]) for x in Ts])
            Tm = [fourc.mo_eri(ri, 'F', O[t], O[t], ONE, O[s], 2 * g) / g ** 2 for t in range(4)]
            M.append(self._gem_bra(Tm))                           # [p, I, l]
        X0 = X0 / numpy.sqrt(2)

        # strong orthogonality masks on pairs of "1" indices
        posd = sp.mask_Vp | sp.mask_Cp
        Qm = (numpy.outer(posd, posd) & ~numpy.outer(sp.mask_Vp, sp.mask_Vp)).astype(float)
        Pm = 1.0 - Qm
        GQ = GF * Qm
        GP = GF * Pm

        # V, X
        V = VFG - numpy.einsum('pIJ,IJq->pq', GP, gIJ, optimize=True)
        X = X0 - numpy.einsum('pIJ,qIJ->pq', GP, GF.conj(), optimize=True)
        # C_{p,ab} = <g_p|f Q (F1+F2)|ab>, antisymmetrized
        Cab = numpy.einsum('pIb,Ia->pab', GQ[:, :, vp], F[:, vp], optimize=True) \
            + numpy.einsum('paJ,Jb->pab', GQ[:, vp, :], F[:, vp], optimize=True)
        Cab = (Cab - Cab.transpose(0, 2, 1)) / numpy.sqrt(2)
        # B, approximation C
        FK = F + K
        Tm = 0
        for s in range(4):
            FW = numpy.dot(FK, W[s])                              # [(F+K) W_s]_{I k}
            t1 = numpy.einsum('Ik,pIl->pkl', FW, M[s], optimize=True)
            Tm = Tm + COEF[s] * (t1[:, self._pi, self._pj] - t1[:, self._pj, self._pi])
        Tm = Tm / numpy.sqrt(2)                                   # <g_p| f^2 (F+K)_1 |g_q>
        B = Tm + Tm.conj().T
        B -= 2 * numpy.einsum('pIJ,IK,qKJ->pq', GF, K, GF.conj(), optimize=True)
        PFf = 2 * numpy.einsum('pIJ,IK,qKJ->pq', GP, F, GF.conj(), optimize=True)
        PFPf = 2 * numpy.einsum('pIJ,IK,qKJ->pq', GP, F, GP.conj(), optimize=True)
        B -= PFf + PFf.conj().T - PFPf
        self.V, self.X, self.B, self.C = V, X, B, Cab

        # energies
        ea = e[vp]
        eo = e[act]
        npair = len(pairs)
        a_, b_ = numpy.triu_indices(len(vp), 1)
        self.e_mp2 = 0.0
        self.e_f12 = 0.0
        self.pair_energies = {}
        for p, (i, j) in enumerate(pairs):
            d = ea[a_] + ea[b_] - eo[i] - eo[j]
            gab = g_ab[a_, b_, p]
            cab = Cab[p, a_, b_]
            self.e_mp2 += -numpy.sum(abs(gab) ** 2 / d)
            Vt = V[p, p] - numpy.sum(cab * gab / d)
            Bt = B[p, p] - (eo[i] + eo[j]) * X[p, p] - numpy.sum(abs(cab) ** 2 / d)
            ep = (2 * Vt + Bt).real
            self.pair_energies[(act[i], act[j])] = ep
            self.e_f12 += ep
        self.e_singles = sp.cabs_singles()
        self.e_hf = self.mf.e_tot
        self.e_tot = self.e_hf + self.e_singles + self.e_mp2 + self.e_f12
        return self.e_tot
