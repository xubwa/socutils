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
Brute-force V, X, C, B of MP2-F12/3C(FIX) directly from their definitions
(rf12/notes/stage2_derivation.md, Sec. 3), with the strong-orthogonality
projector applied as an explicit operator on the pair space "1" x "1":

    P12 = O1 + O2 - O1 O2 + V1 V2,   Q12 = 1 - P12
    V = <kl| f Q12 g |ij>
    X = <kl| f Q12 f |mn>
    C = <kl| f Q12 (F1 + F2) |ab>
    B = <kl| f Q12 (F1 + F2) Q12 f |mn>           (approximation C:)
      = <f (F1+F2) f>_C - <f P12 F f> - <f F P12 f> + <f P12 F P12 f>
    <f (F1+F2) f>_C = U + 1/2 <{f^2, (F+K)_1 + (F+K)_2}> - <f (K1 + K2) f>

All insertions use the "1" = OBS + CABS resolution of the identity.  The
operator algebra is done with dense projector matrices; none of the
term-by-term formulas of mp2f12.MP2F12 are used, so agreement of the two
is a check of that derivation.  Only for small systems (He, Ne).
'''

import numpy


def intermediates(ints):
    d = ints.dims
    nocc, nfz, nmo, nri = d['nocc'], d['nfz'], d['nmo'], d['nri']
    Fk, K = ints.fock, ints.K
    one = numpy.eye(nri)
    O = numpy.zeros((nri, nri))
    O[:nocc, :nocc] = numpy.eye(nocc)
    Vp = numpy.zeros((nri, nri))
    Vp[nocc:nmo, nocc:nmo] = numpy.eye(nmo - nocc)

    def P12(T):
        '''apply P12 to the ket pair indices (last two) of T[..., x, y]'''
        return (numpy.einsum('...xy,zx->...zy', T, O) + numpy.einsum('...xy,zy->...xz', T, O)
                - numpy.einsum('...xy,zx,wy->...zw', T, O, O) + numpy.einsum('...xy,zx,wy->...zw', T, Vp, Vp))

    def F12op(T):
        '''apply F1 + F2 (Fock matrix in "1") to the ket pair of T[..., x, y]'''
        return numpy.einsum('...xy,zx->...zy', T, Fk) + numpy.einsum('...xy,zy->...xz', T, Fk)

    def K12op(T):
        return numpy.einsum('...xy,zx->...zy', T, K) + numpy.einsum('...xy,zy->...xz', T, K)

    F_aa11 = ints('F', 'aa11')          # <kl|f|xy>
    G_aa11 = ints('G', 'aa11')
    f_bra = F_aa11                       # <kl|f|xy>, x,y in "1"
    Pf = P12(f_bra)                      # <kl|f P12|xy>  (P12 Hermitian)

    # V and X: <kl|f Q g|ij> = FG - <kl|f P12|xy><xy|g|ij>
    a = slice(nfz, nocc)
    G_11aa = G_aa11.transpose(2, 3, 0, 1)
    V = ints('FG', 'aaaa') - numpy.einsum('klxy,xyij->klij', Pf, G_11aa)
    F_11aa = F_aa11.transpose(2, 3, 0, 1)
    X = ints('F2', 'aaaa') - numpy.einsum('klxy,xymn->klmn', Pf, F_11aa)

    # C: <kl|f Q12 (F1+F2)|ab>
    Qf = f_bra - Pf                      # <kl|f Q12|xy>
    C = F12op(Qf)[:, :, nocc:nmo, nocc:nmo]   # (F1+F2) acting to the left = Fock^T, Fock symmetric

    # B, approximation C
    U = ints('U', 'aaaa')
    F2_aaa1 = ints('F2', 'aaa1')
    FK = Fk + K
    t = numpy.einsum('klmI,In->klmn', F2_aaa1, FK[:, a])
    t = t + t.transpose(1, 0, 3, 2)
    fFf = U + 0.5 * (t + t.transpose(2, 3, 0, 1))
    fFf -= numpy.einsum('klxy,mnxy->klmn', K12op(f_bra), f_bra)
    # projector corrections, all within "1" x "1"
    PFf = numpy.einsum('klxy,mnxy->klmn', F12op(Pf), f_bra)     # <f P F f>
    FPf = PFf.transpose(2, 3, 0, 1)                              # <f F P f>
    PFPf = numpy.einsum('klxy,mnxy->klmn', F12op(Pf), Pf)       # <f P F P f>
    B = fFf - PFf - FPf + PFPf
    return V, X, C, B
