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
Non-relativistic closed-shell MP2-F12/3C(FIX), brute-force reference
(stage 2).  Derivation and conventions: rf12/notes/stage2_derivation.md.

Ingredients
-----------
geminal       f12 = -exp(-gamma r12) / gamma      (Slater type, exact)
ansatz 3      Q12 = (1 - O1)(1 - O2) - V1 V2,  O = occupied, V = OBS virtual
fixed amp.    SP ansatz (Ten-no): t^{ij}_{ij} = 3/8, t^{ij}_{ji} = 1/8
              (spin-orbital singlet 1/2, triplet 1/4)
approx. C     B from the commutator identity  f t f = (f^2 t + t f^2)/2 + f'^2
RI            complete-space insertions in the orthonormal OBS + CABS basis
              ("1" = p + a'), CABS by SVD projection (Valeev 2004)
CABS singles  second-order Fock-coupling correction to the HF reference

All two-electron integrals come from libcint through pyscf:
int2e (1/r12), int2e_stg (exp(-z r12)), int2e_yp (exp(-z r12)/r12).
    F   = <|f12|>           = -1/g  stg(g)
    FG  = <|f12 / r12|>     = -1/g  yp(g)
    F2  = <|f12^2|>         =  1/g^2 stg(2g)
    U   = <|(d f12/dr12)^2|> = stg(2g)
Notation of the 4-index arrays: A[p,q,r,s] = <pq|A|rs> (physicist).
'''

import numpy
from functools import reduce
from pyscf import gto, scf, ao2mo, lib


def make_cabs(mol, cabs_basis, mo_coeff, lindep_ri=1e-8, lindep_cabs=1e-6):
    '''Orthonormal complementary auxiliary basis.

    RI basis = OBS + CABS basis functions, canonically orthonormalized
    (overlap eigenvalues < lindep_ri dropped); CABS = null space of the
    OBS-MO / RI overlap (singular values < lindep_cabs).
    Returns (ri_mol, C_cabs) with C_cabs in the ri_mol AO basis.'''
    auxmol = mol.copy()
    auxmol.basis = cabs_basis
    auxmol.build(False, False)
    ri_mol = gto.conc_mol(mol, auxmol)
    # conc_mol appends the atoms of auxmol again: keep their basis functions
    # but remove the duplicated nuclear charges (one-electron integrals)
    ri_mol._atm[mol.natm:, gto.CHARGE_OF] = 0
    S = ri_mol.intor_symmetric('int1e_ovlp')
    e, U = numpy.linalg.eigh(S)
    keep = e > lindep_ri
    X = U[:, keep] / numpy.sqrt(e[keep])
    nao = mol.nao
    S_mo_ri = reduce(numpy.dot, (mo_coeff.T, S[:nao], X))
    u, s, vt = numpy.linalg.svd(S_mo_ri)
    nnull = numpy.count_nonzero(s > lindep_cabs)
    C_cabs = numpy.dot(X, vt[nnull:].T)
    return ri_mol, C_cabs


def _ints(ri_mol, intor, mos, zeta=None):
    '''<12|K|34> in the MO sets mos = (C1, C2, C3, C4) (ri_mol AO basis)'''
    if zeta is not None:
        ri_mol.set_f12_zeta(zeta)
    c1, c2, c3, c4 = mos
    # ao2mo works in chemist notation (13|24)
    eri = ao2mo.general(ri_mol, (c1, c3, c2, c4), intor=intor, compact=False, aosym='s1')
    eri = eri.reshape(c1.shape[1], c3.shape[1], c2.shape[1], c4.shape[1])
    return eri.transpose(0, 2, 1, 3)


class LibcintIntegrals:
    '''MO integrals over the orbital spaces of an RHF reference with libcint.

    Spaces (labels): 'a' active occupied, 'o' all occupied, 'v' OBS virtual,
    'P' all OBS MOs, 'c' CABS, '1' = 'P' + 'c'.
    Kernels: 'G' 1/r12, 'F' f12, 'FG' f12/r12, 'F2' f12^2, 'U' (df12/dr12)^2,
    f12 = -exp(-gamma r12)/gamma.'''
    def __init__(self, mf, cabs_basis, gamma=1.0, frozen=0):
        mol = mf.mol
        self.gamma = gamma
        nocc = numpy.count_nonzero(mf.mo_occ > 0)
        nmo = mf.mo_coeff.shape[1]
        ri_mol, C_cabs = make_cabs(mol, cabs_basis, mf.mo_coeff)
        nao_ri = ri_mol.nao
        nao = mol.nao
        C_obs = numpy.zeros((nao_ri, nmo))
        C_obs[:nao] = mf.mo_coeff
        C1 = numpy.hstack([C_obs, C_cabs])
        self.ri_mol = ri_mol
        self.spaces = {'a': C1[:, frozen:nocc], 'o': C1[:, :nocc], 'v': C1[:, nocc:nmo],
                       'P': C1[:, :nmo], 'c': C1[:, nmo:], '1': C1}
        self.dims = dict(nocc=nocc, nfz=frozen, nmo=nmo, nri=C1.shape[1])
        # Fock and exchange operators in the "1" basis
        dm_ri = numpy.zeros((nao_ri, nao_ri))
        dm_ri[:nao, :nao] = mf.make_rdm1()
        h = ri_mol.intor_symmetric('int1e_kin') + ri_mol.intor_symmetric('int1e_nuc')
        vj, vk = scf.hf.get_jk(ri_mol, dm_ri)
        self.K = reduce(numpy.dot, (C1.T, 0.5 * vk, C1))
        self.fock = reduce(numpy.dot, (C1.T, h + vj - 0.5 * vk, C1))
        check_f12_libcint(ri_mol)

    def __call__(self, kind, labels):
        g = self.gamma
        mos = [self.spaces[x] for x in labels]
        if kind == 'G':
            return _ints(self.ri_mol, 'int2e', mos)
        if kind == 'F':
            return _ints(self.ri_mol, 'int2e_stg', mos, g) * (-1 / g)
        if kind == 'FG':
            return _ints(self.ri_mol, 'int2e_yp', mos, g) * (-1 / g)
        if kind == 'F2':
            return _ints(self.ri_mol, 'int2e_stg', mos, 2 * g) / g ** 2
        if kind == 'U':
            return _ints(self.ri_mol, 'int2e_stg', mos, 2 * g)
        raise KeyError(kind)


def check_f12_libcint(mol):
    try:
        mol.set_f12_zeta(1.)
        mol.intor('int2e_stg', shls_slice=(0, 1, 0, 1, 0, 1, 0, 1))
    except AttributeError:
        raise RuntimeError('libcint was built without F12 integrals (int2e_stg/int2e_yp).  '
                           'Rebuild libcint v6.1.3 with -DWITH_F12=1 (see rf12/notes/stage2_derivation.md).')


class MP2F12:
    '''MP2-F12/3C(FIX) for a converged closed-shell pyscf RHF object.
    ints: optional integral provider (default LibcintIntegrals).'''
    def __init__(self, mf, cabs_basis=None, gamma=1.0, frozen=0, cabs_singles=True, ints=None):
        self.mf = mf
        self.cabs_basis = cabs_basis
        self.gamma = gamma
        self.frozen = frozen
        self.cabs_singles = cabs_singles
        self.ints = ints

    def kernel(self):
        if self.ints is None:
            self.ints = LibcintIntegrals(self.mf, self.cabs_basis, self.gamma, self.frozen)
        return self.assemble(self.ints, self.mf.e_tot)

    def assemble(self, ints, e_hf):
        dims = ints.dims
        nocc, nfz, nmo, nri = dims['nocc'], dims['nfz'], dims['nmo'], dims['nri']
        nact = nocc - nfz
        nvir = nmo - nocc
        self.dims = dict(nocc=nocc, nact=nact, nvir=nvir, nmo=nmo, ncabs=nri - nmo, nri=nri)
        Fk = ints.fock
        K = ints.K
        e = numpy.diag(Fk)
        G = lambda lab: ints('G', lab)
        F = lambda lab: ints('F', lab)
        F2 = lambda lab: ints('F2', lab)
        FG = ints('FG', 'aaaa')
        U = ints('U', 'aaaa')

        # ---- V and X intermediates (ansatz 3, complete-space RI in "1")
        F_aaPP = F('aaPP')
        G_aaPP = G('aaPP')
        F_aaoc = F('aaoc')
        G_aaoc = G('aaoc')
        V = FG.copy()
        V -= numpy.einsum('klpq,ijpq->klij', F_aaPP, G_aaPP)
        tmp = numpy.einsum('klmc,ijmc->klij', F_aaoc, G_aaoc)
        V -= tmp + tmp.transpose(1, 0, 3, 2)
        X = F2('aaaa')
        X -= numpy.einsum('klpq,ijpq->klij', F_aaPP, F_aaPP)
        tmp = numpy.einsum('klmc,ijmc->klij', F_aaoc, F_aaoc)
        X -= tmp + tmp.transpose(1, 0, 3, 2)

        # ---- C intermediate: C^{kl}_{ab} = sum_a' F_{kl,a a'} f_{b a'} + (1<->2)
        F_aavc = F('aavc')
        f_vc = Fk[nocc:nmo, nmo:]
        tmp = numpy.einsum('klaq,bq->klab', F_aavc, f_vc)
        Cm = tmp + tmp.transpose(1, 0, 3, 2)

        # ---- B intermediate, approximation C (term numbering as in
        #      rf12/notes/stage2_derivation.md, Sec. 4)
        sym = lambda t: t + t.transpose(1, 0, 3, 2)
        F_aa11 = F('aa11')
        F2_aaa1 = F2('aaa1')
        fk_a1 = (Fk + K)[nfz:nocc, :]
        B = U.copy()                                                     # (B1) f'^2
        B += sym(numpy.einsum('klmI,nI->klmn', F2_aaa1, fk_a1))           # (B2) f^2 (f+k)
        B -= sym(numpy.einsum('klPC,CA,mnPA->klmn', F_aa11, K, F_aa11, optimize=True))  # (B3) f k f
        F_aao1 = F_aa11[:, :, :nocc, :]
        B -= sym(numpy.einsum('kljC,CA,mnjA->klmn', F_aao1, Fk, F_aao1, optimize=True))  # (B4)
        F_aaco = F_aa11[:, :, nmo:, :nocc]
        B += sym(numpy.einsum('klpj,ji,mnpi->klmn', F_aaco, Fk[:nocc, :nocc], F_aaco, optimize=True))  # (B5)
        F_aavP = F_aa11[:, :, nocc:nmo, :nmo]
        B -= sym(numpy.einsum('klbr,rp,mnbp->klmn', F_aavP, Fk[:nmo, :nmo], F_aavP, optimize=True))  # (B6)
        F_aac1 = F_aa11[:, :, nmo:, :]
        B -= 2 * sym(numpy.einsum('klpI,jI,mnpj->klmn', F_aac1, Fk[:nocc, :], F_aaco, optimize=True))  # (B7)
        F_aavc = F_aa11[:, :, nocc:nmo, nmo:]
        B -= 2 * sym(numpy.einsum('klbr,rq,mnbq->klmn', F_aavP, Fk[:nmo, nmo:], F_aavc, optimize=True))  # (B8)
        B = 0.5 * (B + B.transpose(2, 3, 0, 1))
        self.V, self.X, self.B, self.C = V, X, B, Cm

        # ---- MP2
        G_aavv = G_aaPP[:, :, nocc:, nocc:]
        eo = e[nfz:nocc]
        ev = e[nocc:nmo]
        D = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
        t2 = G_aavv / D
        self.e_mp2 = numpy.einsum('ijab,ijab', t2, 2 * G_aavv - G_aavv.transpose(0, 1, 3, 2))

        # ---- F12 energy with fixed amplitudes (SP ansatz)
        # pair (i,j): Vt_kl = V_kl,ij - sum_ab C_kl,ab <ij|ab>/(e_a+e_b-e_i-e_j)
        #             Bt_klmn = B_klmn - (e_i+e_j) X_klmn - sum_ab C_kl,ab C_mn,ab/(..)
        ts, tt = 0.5, 0.25           # spin-orbital singlet / triplet amplitudes
        e_s = e_t = 0.0
        pair = {}
        for i in range(nact):
            for j in range(i, nact):
                d = -D[i, j]                                    # e_a + e_b - e_i - e_j > 0
                Vt = V[:, :, i, j] - numpy.einsum('klab,ab->kl', Cm, G_aavv[i, j] / d)
                Bt = B - (eo[i] + eo[j]) * X - numpy.einsum('klab,mnab->klmn', Cm, Cm / d)
                # singlet / triplet projections of the (i,j) pair, fixed amplitudes
                vs = 0.5 * (Vt[i, j] + Vt[j, i])
                bs = 0.25 * (Bt[i, j, i, j] + Bt[i, j, j, i] + Bt[j, i, i, j] + Bt[j, i, j, i])
                kd = 1 if i == j else 2
                if i == j:
                    es = 2 * ts * Vt[i, i] + ts ** 2 * Bt[i, i, i, i]
                    et = 0.0
                else:
                    # singlet pair function: ts (|ij> + |ji>)/..., normalisation as in derivation Sec. 5
                    es = kd * (2 * ts * vs + ts ** 2 * bs)
                    vt_ = 0.5 * (Vt[i, j] - Vt[j, i])
                    bt_ = 0.25 * (Bt[i, j, i, j] - Bt[i, j, j, i] - Bt[j, i, i, j] + Bt[j, i, j, i])
                    et = 3 * kd * (2 * tt * vt_ + tt ** 2 * bt_)
                e_s += es
                e_t += et
                pair[(i + nfz, j + nfz)] = (es, et)
        self.pair_energies = pair
        self.e_f12 = e_s + e_t
        self.e_f12_singlet = e_s
        self.e_f12_triplet = e_t

        # ---- CABS singles
        self.e_singles = 0.0
        if self.cabs_singles:
            eo_, Uo = numpy.linalg.eigh(Fk[:nocc, :nocc])
            ev_, Uv = numpy.linalg.eigh(Fk[nocc:, nocc:])
            fov = reduce(numpy.dot, (Uo.T, Fk[:nocc, nocc:], Uv))
            self.e_singles = 2 * numpy.sum(fov ** 2 / (eo_[:, None] - ev_[None, :]))
        self.e_hf = e_hf
        self.e_tot = self.e_hf + self.e_singles + self.e_mp2 + self.e_f12
        return self.e_tot
