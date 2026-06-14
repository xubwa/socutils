#
# Full spin-orbital CCSDT (iterative T3) for socutils two-component spinor CC.
#
# Efficient T1-dressed tensor implementation, consistent with pyscf's rccsdt:
# T1 is absorbed into the antisymmetrized integrals and Fock (x = 1 - t1,
# y = 1 + t1), so the T3 sector is T1-free.  Conventions follow pyscf gccsd /
# socutils zccsd: spin-orbital, antisymmetrized physicist integrals <pq||rs>,
# complex amplitudes, t2/t3 fully antisymmetric.
#
# VALIDATION: every residual (r1, r2, r3) matches the guaranteed-correct
# determinant-space oracle (socutils.cc._ccsdt_bruteforce) element-wise to
# ~1e-15 (random t1/t2/t3, t1=0 and t1!=0), and the converged correlation
# energy matches pyscf rccsdt_highm on H2O/HF/LiH (STO-3G, frozen core) to
# ~1e-9.  See cc/_t3model.py for the standalone term-by-term r3 check.
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from socutils.cc.zccsd import ZCCSD, _PhysicistsERIs
from socutils.cc._ccsdt_bruteforce import build_g

einsum = lib.einsum


def _asarray(x):
    return np.asarray(x)


# ---------- antisymmetrizers / partial permutation operators ----------

def fullasym(t):
    '''Full P(i/jk)*P(a/bc) antisymmetrizer (6x6 signed perms, no 1/36).'''
    a = (t + t.transpose(1, 2, 0, 3, 4, 5) + t.transpose(2, 0, 1, 3, 4, 5)
         - t.transpose(1, 0, 2, 3, 4, 5) - t.transpose(0, 2, 1, 3, 4, 5)
         - t.transpose(2, 1, 0, 3, 4, 5))
    a = (a + a.transpose(0, 1, 2, 4, 5, 3) + a.transpose(0, 1, 2, 5, 3, 4)
         - a.transpose(0, 1, 2, 4, 3, 5) - a.transpose(0, 1, 2, 3, 5, 4)
         - a.transpose(0, 1, 2, 5, 4, 3))
    return a


def _Pabc(t):
    return t - t.transpose(0, 1, 2, 4, 3, 5) - t.transpose(0, 1, 2, 5, 4, 3)


def _Pijk(t):
    return t - t.transpose(1, 0, 2, 3, 4, 5) - t.transpose(2, 1, 0, 3, 4, 5)


def _Pc_ab(t):
    return t - t.transpose(0, 1, 2, 5, 4, 3) - t.transpose(0, 1, 2, 3, 5, 4)


def _Pk_ij(t):
    return t - t.transpose(2, 1, 0, 3, 4, 5) - t.transpose(0, 2, 1, 3, 4, 5)


def _Pijk_Pabc(t):
    return _Pijk(_Pabc(t))


# ---------- T3 antisymmetric packing (store only i<j<k, a<b<c) ----------

_T3_TRI_CACHE = {}


def _t3_tri(nocc, nvir):
    '''Index arrays of the strictly-ordered occupied (i<j<k) and virtual
    (a<b<c) triples; cached per (nocc, nvir).'''
    key = (nocc, nvir)
    if key not in _T3_TRI_CACHE:
        occ = np.array([(i, j, k) for i in range(nocc)
                        for j in range(i + 1, nocc)
                        for k in range(j + 1, nocc)], dtype=int).reshape(-1, 3)
        vir = np.array([(a, b, c) for a in range(nvir)
                        for b in range(a + 1, nvir)
                        for c in range(b + 1, nvir)], dtype=int).reshape(-1, 3)
        _T3_TRI_CACHE[key] = (occ, vir)
    return _T3_TRI_CACHE[key]


def pack_t3(t3):
    '''Pack a fully-antisymmetric t3[i,j,k,a,b,c] into its unique components
    (i<j<k, a<b<c) -- a flat array ~36x smaller.'''
    nocc, nvir = t3.shape[0], t3.shape[3]
    occ, vir = _t3_tri(nocc, nvir)
    if len(occ) == 0 or len(vir) == 0:
        return np.zeros(0, dtype=t3.dtype)
    return t3[occ[:, 0][:, None], occ[:, 1][:, None], occ[:, 2][:, None],
              vir[:, 0][None, :], vir[:, 1][None, :], vir[:, 2][None, :]].ravel()


def unpack_t3(packed, nocc, nvir):
    '''Rebuild the full antisymmetric t3 from its unique components.  The unique
    entries are scattered into a seed tensor and ``fullasym`` fills the rest
    with the correct signs (no manual sign bookkeeping).'''
    occ, vir = _t3_tri(nocc, nvir)
    seed = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir),
                    dtype=np.result_type(packed, np.complex128))
    if len(occ) and len(vir):
        seed[occ[:, 0][:, None], occ[:, 1][:, None], occ[:, 2][:, None],
             vir[:, 0][None, :], vir[:, 1][None, :], vir[:, 2][None, :]] = \
            packed.reshape(len(occ), len(vir))
    return fullasym(seed)


# ---------- T1 dressing ----------

def _t1_dress(g, fock, t1, nocc, nmo):
    '''Generalized spin-orbital T1 dressing of the antisymmetrized integrals
    g[p,q,r,s]=<pq||rs> and the Fock matrix (x = 1 - t1, y = 1 + t1).'''
    x = np.eye(nmo, dtype=complex)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(nmo, dtype=complex)
    y[:nocc, nocc:] += t1
    ge = einsum('tvuw,pt->pvuw', g, x)
    ge = einsum('pvuw,rv->pruw', ge, x)
    ge = ge.transpose(2, 3, 0, 1)
    ge = einsum('uwpr,qu->qwpr', ge, y)
    ge = einsum('qwpr,sw->qspr', ge, y)
    ge = ge.transpose(2, 3, 0, 1)
    fdr = fock + einsum('risa,ia->rs', g[:, :nocc, :, nocc:], t1)
    fdr = x @ fdr @ y.T
    return ge, fdr


def _t3_residual(t2, t3, ge, fdr, nocc, nmo):
    '''Spin-orbital CCSDT T3 residual <Phi_ijk^abc|Hbar|0>, T1-free in the
    T1-dressed integrals ge (=<pq||rs> dressed) and Fock fdr.  Verified
    element-wise against the determinant-space oracle to ~1e-15.'''
    o = slice(0, nocc)
    v = slice(nocc, nmo)
    fvv = fdr[v, v]; foo = fdr[o, o]; fov = fdr[o, v]
    gvvvo = ge[v, v, v, o]; govoo = ge[o, v, o, o]
    vvvv = ge[v, v, v, v]; oooo = ge[o, o, o, o]; ovvo = ge[o, v, v, o]
    goovv = ge[o, o, v, v]; ovvv = ge[o, v, v, v]; ooov = ge[o, o, o, v]
    oovo = ge[o, o, v, o]

    # (1) T2 driving via t2-dressed W_vvvo / W_vooo (contains the linear-in-t2
    #     driving and the quadratic t2^2 driving)
    Wvvvo = gvvvo.copy()
    Wvvvo += 0.5 * einsum('mnei,mnab->abei', oovo, t2)              # hole ladder
    tmp = einsum('mbef,miaf->abei', ovvv, t2)
    Wvvvo -= (tmp - tmp.transpose(1, 0, 2, 3))                      # P(ab) particle
    Wvvvo -= einsum('me,miab->abei', fov, t2)                       # dressed-Fock_ov * t2
    Wvooo = govoo.copy()                                            # slot [m,a,j,i]
    tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
    Wvooo -= (tmp2 - tmp2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    Wvooo -= (0.5 * einsum('mbef,ijef->mbij', ovvv, t2)).transpose(0, 1, 3, 2)
    R = -0.25 * fullasym(einsum('abei,jkec->ijkabc', Wvvvo, t2)
                         + einsum('maji,mkbc->ijkabc', Wvooo, t2))

    # (2) linear in t3 (dressed integrals)
    R += _Pabc(einsum('ad,ijkdbc->ijkabc', fvv, t3))
    R -= _Pijk(einsum('mi,mjkabc->ijkabc', foo, t3))
    R += 0.5 * _Pc_ab(einsum('abde,ijkdec->ijkabc', vvvv, t3))
    R += 0.5 * _Pk_ij(einsum('mnij,mnkabc->ijkabc', oooo, t3))
    R += _Pijk_Pabc(einsum('madi,mjkdbc->ijkabc', ovvo, t3))

    # (3) bilinear t2*t3 coupling
    dWvvvo_t3 = einsum('lmef,ilmabf->abei', goovv, t3)
    R += -0.125 * fullasym(einsum('abei,jkec->ijkabc', dWvvvo_t3, t2))
    dFvv = einsum('mnaf,mndf->ad', t2, goovv)
    R += -0.5 * _Pabc(einsum('ad,ijkdbc->ijkabc', dFvv, t3))
    dWvvvv = einsum('mnab,mnde->abde', t2, goovv)
    R += 0.25 * _Pc_ab(einsum('abde,ijkdec->ijkabc', dWvvvv, t3))
    dWoooo = einsum('mnef,ijef->mnij', goovv, t2)
    R += 0.25 * _Pk_ij(einsum('mnij,mnkabc->ijkabc', dWoooo, t3))
    return R


def energy(cc, t1, t2, eris):
    '''CCSD-form correlation energy (T3 enters only through t1/t2).'''
    nocc, nvir = t1.shape
    fock = eris.fock
    e = einsum('ia,ia', fock[:nocc, nocc:], t1)
    eris_oovv = _asarray(eris.oovv)
    e += 0.25 * einsum('ijab,ijab', t2, eris_oovv)
    e += 0.5 * einsum('ia,jb,ijab', t1, t1, eris_oovv)
    return e.real


def update_amps(cc, t1, t2, t3, eris):
    '''Next amplitudes (t1new, t2new, t3new).  Efficient tensor residuals:
    the CCSD T1/T2 part from socutils.cc.zccsd (validated), plus the T3->T1/T2
    couplings and the T3 residual in the T1-dressed formalism.'''
    assert isinstance(eris, _PhysicistsERIs)
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    o = slice(0, nocc)
    v = slice(nocc, nmo)
    mo_e = eris.mo_energy
    eia = mo_e[:nocc][:, None] - (mo_e[nocc:] + cc.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    eijkabc = (eia[:, None, None, :, None, None]
               + eia[None, :, None, None, :, None]
               + eia[None, None, :, None, None, :])

    # The bare antisymmetrized integral tensor g[p,q,r,s]=<pq||rs> depends only
    # on the (fixed) MO integrals, so build it once and cache it on eris.  Only
    # the T1-dressing below is redone each iteration (t1 changes).
    g = getattr(eris, '_ccsdt_g', None)
    if g is None:
        g = build_g(eris, nocc, nmo)
        eris._ccsdt_g = g
    fock = _asarray(eris.fock)
    ge, fdr = _t1_dress(g, fock, t1, nocc, nmo)

    # CCSD T1/T2 part (socutils, validated == pyscf CCSD)
    t1new, t2new = ZCCSD.update_amps(cc, t1, t2, eris)

    # T3 -> T1
    t1new = t1new + (0.25 * einsum('mnef,imnaef->ia', ge[o, o, v, v], t3)) / eia

    # T3 -> T2
    r2t3 = einsum('me,ijmabe->ijab', fdr[o, v], t3)
    tmp = 0.5 * einsum('amef,ijmebf->ijab', ge[v, o, v, v], t3)
    r2t3 = r2t3 + (tmp - tmp.transpose(0, 1, 3, 2))            # P(ab)
    tmp = 0.5 * einsum('mnej,inmabe->ijab', ge[o, o, v, o], t3)
    r2t3 = r2t3 - (tmp - tmp.transpose(1, 0, 2, 3))            # P(ij)
    t2new = t2new + r2t3 / eijab

    # T3 residual
    r3 = _t3_residual(t2, t3, ge, fdr, nocc, nmo)
    t3new = t3 + r3 / eijkabc
    return t1new, t2new, t3new


class ZCCSDT(ZCCSD):
    '''Full spin-orbital CCSDT for two-component spinor CC (T1-dressed tensor).'''

    conv_tol = 1e-9
    conv_tol_normt = 1e-7
    max_cycle = 300

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None,
                 with_mmf=False, erifile=None):
        ZCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ,
                       with_mmf=with_mmf, erifile=erifile)
        self.t3 = None

    update_amps = update_amps
    energy = energy

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e = eris.mo_energy
        eia = mo_e[:nocc, None] - mo_e[None, nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        fov = eris.fock[:nocc, nocc:]
        t1 = fov.conj() / eia
        oovv = _asarray(eris.oovv).conj()
        t2 = oovv / eijab
        t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=t2.dtype)
        emp2 = 0.25 * einsum('ijab,ijab', t2, _asarray(eris.oovv))
        self.emp2 = emp2.real
        return self.emp2, t1, t2, t3

    def kernel(self, t1=None, t2=None, t3=None, eris=None):
        return self.ccsdt(t1, t2, t3, eris)

    def ccsdt(self, t1=None, t2=None, t3=None, eris=None):
        log = logger.new_logger(self)
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo(self.mo_coeff)
            self.eris = eris

        emp2, t1i, t2i, t3i = self.init_amps(eris)
        if t1 is None:
            t1 = t1i
        if t2 is None:
            t2 = t2i
        if t3 is None:
            t3 = t3i

        e_old = self.energy(t1, t2, eris)
        log.info('Init E_corr(CCSDT) = %.15g', e_old)

        adiis = lib.diis.DIIS()
        adiis.space = 8

        conv = False
        for icycle in range(self.max_cycle):
            t1new, t2new, t3new = self.update_amps(t1, t2, t3, eris)
            normt = (np.linalg.norm(t1new - t1)
                     + np.linalg.norm(t2new - t2)
                     + np.linalg.norm(t3new - t3))
            t1, t2, t3 = self.run_diis(t1new, t2new, t3new, adiis)
            e_new = self.energy(t1, t2, eris)
            de = e_new - e_old
            log.info('cycle = %d  E_corr(CCSDT) = %.15g  dE = %.3e  norm(t) = %.3e',
                     icycle + 1, e_new, de, normt)
            e_old = e_new
            if abs(de) < self.conv_tol and normt < self.conv_tol_normt:
                conv = True
                break

        self.converged = conv
        self.e_corr = e_old
        self.t1, self.t2, self.t3 = t1, t2, t3
        log.note('E_corr(CCSDT) = %.15g  converged = %s', e_old, conv)
        return self.e_corr, self.t1, self.t2, self.t3

    def amplitudes_to_vector(self, t1, t2, t3):
        # t3 is stored packed (unique i<j<k, a<b<c) -> ~36x smaller DIIS history
        return np.hstack((t1.ravel(), t2.ravel(), pack_t3(t3)))

    def vector_to_amplitudes(self, vec, nocc, nvir):
        n1 = nocc * nvir
        n2 = nocc * nocc * nvir * nvir
        t1 = vec[:n1].reshape(nocc, nvir)
        t2 = vec[n1:n1 + n2].reshape(nocc, nocc, nvir, nvir)
        t3 = unpack_t3(vec[n1 + n2:], nocc, nvir)
        return t1, t2, t3

    def run_diis(self, t1, t2, t3, adiis):
        nocc, nvir = t1.shape
        vec = self.amplitudes_to_vector(t1, t2, t3)
        vec = adiis.update(vec)
        return self.vector_to_amplitudes(vec, nocc, nvir)


if __name__ == '__main__':
    from pyscf import gto
    from socutils.scf import spinor_hf
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = spinor_hf.SCF(mol)
    mf.kernel()
    mycc = ZCCSDT(mf, frozen=2)
    print('E_corr(CCSDT) =', mycc.kernel()[0])
