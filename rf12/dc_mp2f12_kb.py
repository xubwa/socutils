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
Dirac-Coulomb no-pair MP2-F12, ansatz B ("correlate, then kinetic balance"),
approximation C-KB, brute-force reference (stage 3).
Definitions: rf12/notes/stage3_derivation.md Sec. 4.

    gamma_p = K12 f12 A_p,   A_p = S12 Phi^L_p,   K = (1, sigma.p/2c)^T,
Phi^L built from the large components (2-spinors) of the DHF spinors and
S12 = 3/8 - 1/8 P^sigma acting exactly on the 2-spinors.  Every quantity
that survives c -> infinity is evaluated as in stage 2 (analytic f g, f^2,
U = f'^2, one-orbital RI for f^2 (T+V+J), pair RI for f K f), the O(1/c^2)
remainders by RI in the orthonormal 2-spinor space R (large-component span
of OBS + CABS).  Projector terms, C and K use the complete RKB space "1"
(socutils.rf12.dc_space) through the overlaps
    <I J | gamma> = <I~ J~ | f | A>,  I~ = I^L + (1/2c) p^2 xi_I,  I^S = sigma.p xi_I.
'''

import numpy
from functools import reduce
from socutils.rf12 import fourc, dc_space, twoc_ints
from socutils.rf12.dc_mp2f12 import COEF, DCMP2F12

PAULI = numpy.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


class DCMP2F12KB(DCMP2F12):
    def _spinor2(self, C2c):
        '''2c spinor-AO coefficients (n2c, n) -> sph/spin coefficients (n, 2, nao)'''
        ca, cb = self.space.ri_mol.sph2spinor_coeff()
        return numpy.stack([numpy.dot(ca, C2c).T, numpy.dot(cb, C2c).T], axis=1)

    def kernel(self):
        c = self.c
        g = self.gamma
        sp = dc_space.DCSpace(self.mf, self.cabs_basis, c)
        self.space = sp
        ri = sp.ri_mol
        N2c = ri.nao_2c()
        nri = sp.nri
        no = sp.dims['o']
        act = numpy.arange(self.frozen, no)
        nact = len(act)
        pairs = self._pairs(nact)
        self._pi = numpy.array([p[0] for p in pairs], dtype=int)
        self._pj = numpy.array([p[1] for p in pairs], dtype=int)
        F = sp.fock
        K = sp.K
        e = numpy.diag(F).real
        vp = numpy.where(sp.mask_Vp)[0]
        eri = lambda *a, **k: twoc_ints.eri2(ri, *a, **k)
        sF, sF2 = -1. / g, 1. / g ** 2

        # "1" orbitals: large part and small generator
        ONE = fourc.orbitals(ri, sp.C1, c)
        L1, X1 = ONE.L, ONE.Xi / (2 * c)                      # I~ = L1 + X1 'lap'
        # geminal 2-spinors W_t: active occupied large components, sigma_c rotated
        La = L1[act]
        Wt = [La] + [numpy.einsum('st,ntm->nsm', PAULI[x], La) for x in range(3)]
        Lo4 = La                                             # large comps of the 4c active spinors
        Xa = ONE.Xi[act]                                     # their small generators
        # 2-spinor RI space R
        S2 = ri.intor_symmetric('int1e_ovlp_spinor')
        ev, U = numpy.linalg.eigh(S2)
        keep = ev > 1e-8 * ev.max()
        R2c = U[:, keep] / numpy.sqrt(ev[keep])
        R = self._spinor2(R2c)
        nR = R.shape[0]
        Rt = lambda M: reduce(numpy.dot, (R2c.conj().T, M, R2c))
        J_LL = sp.vj_ao[:N2c, :N2c]
        W_R = Rt(ri.intor_symmetric('int1e_nuc_spinor') + J_LL)          # V + J on 2-spinors
        T_R = Rt(ri.intor_symmetric('int1e_kin_spinor'))
        # coefficients of W_t orbitals in R (exact: W_t lies in the large span)
        Ssph = ri.intor_symmetric('int1e_ovlp')
        def in_R(C):
            # <R_mu | w>, evaluated in sph/spin form
            ov = numpy.einsum('rsm,mn,wsn->rw', R.conj(), Ssph, C, optimize=True)
            return ov                                          # (nR, n)

        def gem2(T, bra=True, ket=True):
            '''T[t][s][i, j, k, l] -> [p, q] with bra/ket pair geminals'''
            out = 0
            for t in range(4):
                for s in range(4):
                    x = T[t][s]
                    if bra:
                        x = (x[self._pi, self._pj] - x[self._pj, self._pi]) * COEF[t]
                    if ket:
                        x = (x[..., self._pi, self._pj] - x[..., self._pj, self._pi]) * COEF[s]
                    out = out + x
            return out / 2

        # ---- <I J | gamma_q>  (4c "1" pairs)
        Ov = 0
        for s in range(4):
            Ws = (Wt[s], 'L')
            T = eri('F', (L1, 'L'), (L1, 'L'), Ws, Ws, g, contract=('k', 'k'))
            T = T + eri('F', (X1, 'lap'), (L1, 'L'), Ws, Ws, g, contract=('k', 'k'))
            T = T + eri('F', (L1, 'L'), (X1, 'lap'), Ws, Ws, g, contract=('k', 'k'))
            T = T + eri('F', (X1, 'lap'), (X1, 'lap'), Ws, Ws, g, contract=('k', 'k'))
            Ov = Ov + COEF[s] * (T[..., self._pi, self._pj] - T[..., self._pj, self._pi])
        Ov = (Ov * sF / numpy.sqrt(2)).conj().transpose(2, 0, 1)            # [p, I, J] = <gamma_p|IJ>

        # ---- <I J| g |Phi_q> (4c) and <ab||ij>
        A4 = fourc.orbitals(ri, sp.C1[:, act], c)
        G0 = fourc.mo_eri(ri, 'G', A4, A4, ONE, ONE)
        gIJ = (G0[self._pi, self._pj] - G0[self._pj, self._pi]).conj().transpose(1, 2, 0) / numpy.sqrt(2)
        g_ab = (gIJ[numpy.ix_(vp, vp)] - gIJ[numpy.ix_(vp, vp)].transpose(1, 0, 2)) / numpy.sqrt(2)

        # ---- V: analytic LL part <f A_p| g |Phi^LL_q> ...
        T = [[None] * 4 for _ in range(4)]
        for t in range(4):
            Tb = eri('Y', (Wt[t], 'L'), (Wt[t], 'L'), (Lo4, 'L'), (Lo4, 'L'), g)
            for s in range(4):
                T[t][s] = Tb if s == 0 else 0 * Tb
        VFG = 0
        for t in range(4):
            x = T[t][0]
            x = (x[self._pi, self._pj] - x[self._pj, self._pi]) * COEF[t]
            VFG = VFG + (x[..., self._pi, self._pj] - x[..., self._pj, self._pi])
        VFG = VFG * sF / 2
        # ... + O(1/c^2) cross terms with the small components of Phi (R x R RI)
        def ovR(opb1, opb2):
            '''<mu nu| op f A_q>: R x R overlaps with sigma.p on mu / nu, [mu, nu, q]'''
            out = 0
            for s in range(4):
                Ws = (Wt[s], 'L')
                Tq = eri('F', (R, opb1), (R, opb2), Ws, Ws, g, contract=('k', 'k'))
                out = out + COEF[s] * (Tq[..., self._pi, self._pj] - Tq[..., self._pj, self._pi])
            return out * sF / numpy.sqrt(2)
        ovSL = ovR('sp', 'L') / (2 * c)                     # <mu nu|(sigma1.p1/2c) f A>
        ovLS = ovR('L', 'sp') / (2 * c)
        ovSS = ovR('sp', 'sp') / (4 * c * c)
        gSL = eri('G', (R, 'L'), (R, 'L'), (Xa, 'sp'), (Lo4, 'L'))           # <mu nu|g|k^S l^L>
        gLS = eri('G', (R, 'L'), (R, 'L'), (Lo4, 'L'), (Xa, 'sp'))
        gSS = eri('G', (R, 'L'), (R, 'L'), (Xa, 'sp'), (Xa, 'sp'))
        asym = lambda x: (x[..., self._pi, self._pj] - x[..., self._pj, self._pi]) / numpy.sqrt(2)
        Vrel = (numpy.einsum('mnp,mnq->pq', ovSL.conj(), asym(gSL), optimize=True)
                + numpy.einsum('mnp,mnq->pq', ovLS.conj(), asym(gLS), optimize=True)
                + numpy.einsum('mnp,mnq->pq', ovSS.conj(), asym(gSS), optimize=True))

        # ---- direct metric and kinetic pieces (2-spinor, analytic)
        F2d = [[eri('F', (Wt[t], 'L'), (Wt[t], 'L'), (Wt[s], 'L'), (Wt[s], 'L'), 2 * g) * sF2
                for s in range(4)] for t in range(4)]
        Ud = [[eri('F', (Wt[t], 'L'), (Wt[t], 'L'), (Wt[s], 'L'), (Wt[s], 'L'), 2 * g)
               for s in range(4)] for t in range(4)]
        # <A| f^2 |(T A')> exact (Laplacian on the ket orbital of electron 1, x 1/2)
        F2T = [[0.5 * eri('F', (Wt[t], 'L'), (Wt[t], 'L'), (Wt[s], 'lap'), (Wt[s], 'L'), 2 * g) * sF2
                for s in range(4)] for t in range(4)]
        Xf2 = gem2(F2d)
        Uf = gem2(Ud)
        # <A|f^2 T1|A'> (both electrons equal by antisymmetry), exact
        F2T1 = gem2(F2T)
        # <fA|p1^2|fA'> = <A|f^2 T1|A'> + <T1 A|f^2|A'> + <A|f'^2|A'>  (per electron)
        p1sq = F2T1 + F2T1.conj().T + Uf
        psq = 2 * p1sq                                         # electrons 1 + 2
        # <fA|p1^2 p2^2|fA'> by R x R RI
        def ovRlap(which):
            out = 0
            for s in range(4):
                Ws = (Wt[s], 'L')
                ops = ('lap', 'L') if which == 1 else ('L', 'lap')
                Tq = eri('F', (R, ops[0]), (R, ops[1]), Ws, Ws, g, contract=('k', 'k'))
                out = out + COEF[s] * (Tq[..., self._pi, self._pj] - Tq[..., self._pj, self._pi])
            return out * sF / numpy.sqrt(2)                  # [mu, nu, q] = <mu nu| p_e^2 f A_q>
        ovP1 = ovRlap(1)
        ovP2 = ovRlap(2)
        p12 = numpy.einsum('mnp,mnq->pq', ovP1.conj(), ovP2, optimize=True)  # <fA|p1^2 p2^2|fA'>

        # ---- X
        Xdir = Xf2 + psq / (4 * c * c) + p12 / (16 * c ** 4)

        # ---- B_C
        # one-orbital RI: 1/2 <A|{f^2, W12}|A'> with W = T + V + J in R
        Wfull = T_R + W_R
        F2R = [[eri('F', (Wt[t], 'L'), (Wt[t], 'L'), (R, 'L'), (Wt[s], 'L'), 2 * g, contract=('b', 'b')) * sF2
                for s in range(4)] for t in range(4)]            # <W_t i, W_t j| f^2 |mu, W_s l>
        Tm = 0
        for s in range(4):
            cR = in_R(Wt[s])                                  # (nR, nact)
            WW = numpy.dot(Wfull, cR)                         # W applied to orbital, in R
            for t in range(4):
                x = numpy.einsum('ijml,mk->ijkl', F2R[t][s], WW, optimize=True)
                x = (x[self._pi, self._pj] - x[self._pj, self._pi]) * COEF[t]
                Tm = Tm + COEF[s] * (x[..., self._pi, self._pj] - x[..., self._pj, self._pi])
        Tm = Tm / 2                                           # <A|f^2 W_1|A'>
        # the kinetic part of Tm is replaced by the exact one: remove T from W
        TmT = 0
        for s in range(4):
            cR = in_R(Wt[s])
            TT = numpy.dot(T_R, cR)
            for t in range(4):
                x = numpy.einsum('ijml,mk->ijkl', F2R[t][s], TT, optimize=True)
                x = (x[self._pi, self._pj] - x[self._pj, self._pi]) * COEF[t]
                TmT = TmT + COEF[s] * (x[..., self._pi, self._pj] - x[..., self._pj, self._pi])
        TmT = TmT / 2
        self._kin_ri_vs_exact = abs(TmT - F2T1).max()
        BC = Uf + (Tm + Tm.conj().T)                          # stage-2 form: U + 1/2{f^2, W1 + W2}
        BC -= 2 * numpy.einsum('pIJ,IK,qKJ->pq', Ov, K, Ov.conj(), optimize=True)   # f K f, pair RI in "1"
        # relativistic remainders (R x R RI)
        BC += p12 / (4 * c * c)
        ovS1 = ovR('sp', 'L')                                 # <mu nu|sigma1.p1 f A>
        ovS2 = ovR('L', 'sp')
        ovSS1 = ovR('sp', 'sp')
        rel = (numpy.einsum('mnp,mk,knq->pq', ovS1.conj(), W_R, ovS1, optimize=True)      # e = 1, own
               + numpy.einsum('mnp,mk,knq->pq', ovS2.conj(), W_R, ovS2, optimize=True))   # e = 1, spectator S2
        rel = 2 * rel                                         # electron 2 (antisymmetry)
        rel4 = 2 * numpy.einsum('mnp,mk,knq->pq', ovSS1.conj(), W_R, ovSS1, optimize=True)
        BC += rel / (4 * c * c) + rel4 / (16 * c ** 4)

        # ---- projector, V, X, C, B
        posd = sp.mask_Vp | sp.mask_Cp
        if not getattr(self, 'project_negative', True):
            posd = ~sp.mask_O                          # diagnostic: N treated like C+ (not projected)
        Qm = (numpy.outer(posd, posd) & ~numpy.outer(sp.mask_Vp, sp.mask_Vp)).astype(float)
        self.ov_neg = abs(Ov[:, sp.mask_neg, :]).max() * c
        OvQ = Ov * Qm
        OvP = Ov * (1.0 - Qm)
        V = VFG + Vrel - numpy.einsum('pIJ,IJq->pq', OvP, gIJ, optimize=True)
        X = Xdir - numpy.einsum('pIJ,qIJ->pq', OvP, Ov.conj(), optimize=True)
        Cab = numpy.einsum('pIb,Ia->pab', OvQ[:, :, vp], F[:, vp], optimize=True) \
            + numpy.einsum('paJ,Jb->pab', OvQ[:, vp, :], F[:, vp], optimize=True)
        Cab = (Cab - Cab.transpose(0, 2, 1)) / numpy.sqrt(2)
        PFf = 2 * numpy.einsum('pIJ,IK,qKJ->pq', OvP, F, Ov.conj(), optimize=True)
        PFPf = 2 * numpy.einsum('pIJ,IK,qKJ->pq', OvP, F, OvP.conj(), optimize=True)
        B = BC - (PFf + PFf.conj().T - PFPf)
        self.V, self.X, self.B, self.C = V, X, B, Cab
        self.parts = dict(Vrel=Vrel, p12=p12, rel=rel, rel4=rel4, psq=psq, VFG=VFG, Xf2=Xf2, Uf=Uf,
                          Tm=Tm, fKf=2 * numpy.einsum('pIJ,IK,qKJ->pq', Ov, K, Ov.conj(), optimize=True),
                          PFf=PFf, PFPf=PFPf, Vproj=numpy.einsum('pIJ,IJq->pq', OvP, gIJ, optimize=True),
                          Xproj=numpy.einsum('pIJ,qIJ->pq', OvP, Ov.conj(), optimize=True))

        ea = e[vp]
        eo = e[act]
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
