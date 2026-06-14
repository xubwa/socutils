#
# Full spin-orbital CCSDT (iterative T3) for socutils two-component spinor CC.
#
# Conventions follow pyscf gccsd / socutils zccsd: spin-orbital, antisymmetrized
# physicist integrals <pq||rs>, complex amplitudes, t2/t3 fully antisymmetric.
#
# STATUS
# ------
# Phase-1 reference: correctness over speed.  The connected CCSDT residual is
# evaluated by the guaranteed-correct determinant-space engine in
# ``socutils.cc._ccsdt_bruteforce`` (it builds the full Hamiltonian and the T
# operator in Slater-determinant space and projects <Phi_n|e^{-T}H e^{T}|0>).
# It is EXACT -- validated against pyscf ``rccsdt_highm`` on three closed-shell
# molecules (STO-3G, frozen core):
#     H2O : -0.049482538   (pyscf -0.049482538,  diff 3e-10)
#     HF  : -0.025823451   (pyscf -0.025823452,  diff 1e-9)
#     LiH : -0.020231830   (pyscf -0.020231830,  diff 3e-10)
# Its cost is determinant-space, so use it only for small active spaces
# (validation / reference), NOT production.
#
# The efficient, pyscf-rccsdt-consistent T1-dressed *tensor* form is Phase 2.
# Most of it is already derived and verified term-by-term against the oracle
# (linear-in-t3 and bilinear t2*t3 sectors exact to ~1e-15); two localized gaps
# remain (the t2-dressed driving intermediates W_vvvo/W_ovoo, and their correct
# T1-dressing).  The determinant-space oracle is the term-by-term ground truth
# for finishing it.
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from socutils.cc.zccsd import ZCCSD, _PhysicistsERIs

einsum = lib.einsum


def _asarray(x):
    return np.asarray(x)


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
    '''Next amplitudes (t1new, t2new, t3new) = t + residual/denominator.

    Connected CCSDT residuals from the determinant-space oracle (guaranteed
    correct); cost is determinant-space -> small-system reference only.'''
    from socutils.cc import _ccsdt_bruteforce as bf
    assert isinstance(eris, _PhysicistsERIs)
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    mo_e = eris.mo_energy
    eia = mo_e[:nocc][:, None] - (mo_e[nocc:] + cc.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    eijkabc = (eia[:, None, None, :, None, None]
               + eia[None, :, None, None, :, None]
               + eia[None, None, :, None, None, :])

    g = bf.build_g(eris, nocc, nmo)
    fock = np.asarray(eris.fock)
    r1, r2, r3 = bf.ccsdt_residuals(fock, g, nocc, nmo, t1, t2, t3)
    return t1 + r1 / eia, t2 + r2 / eijab, t3 + r3 / eijkabc


class ZCCSDT(ZCCSD):
    '''Full spin-orbital CCSDT for two-component spinor CC (Phase-1 reference).'''

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
        return np.hstack((t1.ravel(), t2.ravel(), t3.ravel()))

    def vector_to_amplitudes(self, vec, nocc, nvir):
        n1 = nocc * nvir
        n2 = nocc * nocc * nvir * nvir
        t1 = vec[:n1].reshape(nocc, nvir)
        t2 = vec[n1:n1 + n2].reshape(nocc, nocc, nvir, nvir)
        t3 = vec[n1 + n2:].reshape(nocc, nocc, nocc, nvir, nvir, nvir)
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
