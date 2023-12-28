import time
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.ao2mo import r_outcore
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import gccsd
from pyscf.cc import gintermediates as imd
from pyscf.x2c import x2c
from pyscf import __config__
import QMCUtils

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

einsum = lib.einsum
#einsum = np.einsum


def makeCholERIMO(mf, mo_idx):
    mol = mf.mol
    ##mol aobasis --->  UHF j-spinors
    c = np.vstack(mol.sph2spinor_coeff())
    chol_vecs = QMCUtils.chunked_cholesky(mol, max_error=1.e-5).reshape(-1, mol.nao, mol.nao)
    naux, norb, orbs = chol_vecs.shape[0], chol_vecs.shape[1], c.dot(mf.mo_coeff[:,mo_idx])
    
    chol_mo  = lib.einsum('Lib, bj->Lij', lib.einsum('Lab,ai->Lib', chol_vecs, orbs[:norb,:].conj()), orbs[:norb, :])
    chol_mo += lib.einsum('Lib, bj->Lij', lib.einsum('Lab,ai->Lib', chol_vecs, orbs[norb:,:].conj()), orbs[norb:, :])
    return chol_mo


def cc_vvvv_chol(t1, t2, eris, chol):
    nocc, nvir = t1.shape
    tau = imd.make_tau(t2, t1, t1)

    tmp = einsum('ijcd, klcd->ijkl', t2, eris.oovv)
    t2new = einsum('ijkl,klab->ijab', tmp, tau)/8.

    cholVV = chol[:,nocc:,nocc:]
    chol2 = einsum('Pkc,ka->Pac', chol[:,:nocc,nocc:], t1).reshape(-1,nvir)

    cholVV_twiddle = (cholVV.transpose(0,2,1)).reshape(-1,nvir)
    for i in range(nocc):
        for j in range(i):

            T2nij, T2ij = t2new[i,j], t2[i,j] 
            
            T2ijAsym = T2ij - T2ij.T
            tmp = 0.5*(cholVV.reshape(-1,nvir).dot(T2ijAsym)).reshape(-1,nvir,nvir)
            tmp = (tmp.transpose(0,2,1)).reshape(-1, nvir)
            t2new[i,j] += tmp.T.dot(cholVV_twiddle)
            #t2new[i,j] += einsum('Pad,Pbd->ab', tmp, cholVV)
            
            
            tmp = -0.5*(chol2.dot(T2ijAsym)).reshape(-1,nvir,nvir)
            tmp = (tmp.transpose(0,2,1)).reshape(-1, nvir)
            tmp2 = tmp.T.dot(cholVV_twiddle)
            #tmp2 = einsum('Pad,Pbd->ab', tmp, cholVV)
            
            t2new[i,j] +=  tmp2 - tmp2.T

            t2new[j,i] = t2new[i,j].T
    return t2new


##DEFUNCT
def cc_Wvvvv_chol(t1, t2, eris):
    
    tau = imd.make_tau(t2, t1, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('mb,mafe->bafe', t1, eris_ovvv)
    Wabef = np.asarray(eris.vvvv) - tmp + tmp.transpose(1,0,2,3)
    Wabef += einsum('mnab,mnef->abef', tau, 0.25*np.asarray(eris.oovv))
    return Wabef

def cc_Fvv_chol(t1, t2, eris, chol):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]

    tau_tilde = imd.make_tau(t2, t1, t1,fac=0.5)
    Fae = fvv - 0.5*einsum('me,ma->ae',fov, t1)

    #eris_vovv = np.asarray(eris.ovvv).transpose(1,0,3,2)
    #Fae += einsum('mf,amef->ae', t1, eris_vovv)
    L = einsum('Lbi, ib->L', chol[:,nocc:, :nocc], t1)
    Fae += einsum('L,Lab->ba', L, chol[:,nocc:,nocc:])

    Lai = einsum('Lba,ib->Lia', chol[:,nocc:,nocc:], t1)
    Fae -= einsum('Lci, Lia->ac',  chol[:,nocc:, :nocc], Lai)

    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae


def cc_Wovvo_chol(t1, t2, eris, chol):
    nocc, nvir = t1.shape
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)

    tmp1 = einsum('ia, Pba->Pbi', t1, chol[:,nocc:, nocc:])
    Wmbej = einsum('Pjc, Pbi->jbci', chol[:,:nocc, nocc:], tmp1)
    tmp1 = einsum('ia, Pja->Pji', t1, chol[:,:nocc, nocc:])
    Wmbej -= einsum('Pji, Pbc->jbci', tmp1, chol[:,nocc:, nocc:])
    #Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)

    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej

def update_amps(cc, t1, t2, eris):
    assert(isinstance(eris, _PhysicistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = cc_Fvv_chol(t1, t2, eris, cc.eriChol)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    #Wvvvv = cc_Wvvvv_chol(t1, t2, eris)

    Wovvo = cc_Wovvo_chol(t1, t2, eris, cc.eriChol)
    #Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)

    tmp = -0.5*einsum('ijab,Pja->iPb', t2, cc.eriChol[:,:nocc, nocc:])
    t1new += einsum('iPb,Pcb->ic', tmp, cc.eriChol[:,nocc:,nocc:])
    tmp = 0.5*einsum('ijab,Pjb->iPa', t2, cc.eriChol[:,:nocc, nocc:])
    t1new += einsum('iPa,Pca->ic', tmp, cc.eriChol[:,nocc:,nocc:])
    #t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)

    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)

    t2new += cc_vvvv_chol(t1, t2, eris, cc.eriChol)
    #t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)

    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp

    tmp = einsum('ic,Pca->Pia', t1, cc.eriChol[:,nocc:, nocc:])
    tmp1 = einsum('Pjb,Pia->ijab', cc.eriChol[:,:nocc,nocc:], tmp)
    tmp = einsum('ic,Pcb->Pib', t1, cc.eriChol[:,nocc:, nocc:])
    tmp1 -= einsum('Pja,Pib->ijab', cc.eriChol[:,:nocc,nocc:], tmp)
    t2new += (tmp1 - tmp1.transpose(1,0,2,3))
    #tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    #t2new += (tmp - tmp.transpose(1,0,2,3))

    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new

class ZCCSD(gccsd.GCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        #if frozen !
        #assert(isinstance(mf, x2c.RHF))
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        mo_idx = ccsd.get_frozen_mask(self)
        self.eriChol = makeCholERIMO(mf, mo_idx)

    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        if mo_coeff is None:
            mo_coeff=self.mo_coeff
        return _make_eris_FromChol(self, mo_coeff)

class _PhysicistsERIs(gccsd._PhysicistsERIs):
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None
        self.orbspin = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = ccsd.get_frozen_mask(mycc)
        mo_coeff = mo_coeff[:,mo_idx]

        self.mo_coeff = mo_coeff

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        self.nocc = mycc.nocc
        self.mol = mycc.mol
        mo_e = self.mo_energy = self.fock.diagonal().real
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for ZCCSD', gap)
        return self 

def _make_eris_FromChol(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    chol_mo = mycc.eriChol
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    assert(eris.mo_coeff.dtype == complex)

    oooo = einsum('Lij,Lkl->ijkl', chol_mo[:,:nocc,:nocc], chol_mo[:,:nocc,:nocc])
    eris.oooo = oooo.transpose(0,2,1,3) - oooo.transpose(0,2,3,1)
    oooo = 0.
    
    ooov = einsum('Lij,Lkl->ijkl', chol_mo[:,:nocc,:nocc], chol_mo[:,:nocc,nocc:])
    eris.ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2, 0, 1, 3) 
    ooov = 0.
    
    oovv = einsum('Lij,Lkl->ijkl', chol_mo[:,:nocc,:nocc], chol_mo[:,nocc:,nocc:])
    ovov = einsum('Lij,Lkl->ijkl', chol_mo[:,:nocc,nocc:], chol_mo[:,:nocc,nocc:])
    ovvo = einsum('Lij,Lkl->ijkl', chol_mo[:,:nocc,nocc:], chol_mo[:,nocc:,:nocc])
    eris.oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
    eris.ovov = oovv.transpose(0,2,1,3) - ovvo.transpose(0,2,3,1)
    eris.ovvo = ovvo.transpose(0,2,1,3) - oovv.transpose(0,2,3,1)
    oovv, ovov, ovvo = 0., 0., 0.

    #ovvv = einsum('Lij,Lkl->ijkl', chol_mo[:,:nocc,nocc:], chol_mo[:,nocc:,nocc:])
    #eris.ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)
    #ovvv = 0.
    ##eris.ovvv, eris.vvvv will be using DF


    '''
    ###DELETE AFTER THIS
    feri = eris.feris = lib.H5TmpFile()
    dtype = np.result_type(eris.mo_coeff).char
    eris.oooo = feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), dtype)
    eris.ooov = feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), dtype)
    eris.oovv = feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), dtype)
    eris.ovov = feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), dtype)
    eris.ovvo = feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), dtype)
    eris.ovvv = feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), dtype)

    max_memory = mycc.max_memory-lib.current_memory()[0]
    blksize = min(nocc, max(2, int(max_memory*1e6/16/(nmo**3*2))))
    max_memory = max(MEMORYMIN, max_memory)

    mo = eris.mo_coeff
    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]
    print(mo.shape)
    print(orbo.shape, orbv.shape)

    Fswap = Lib.H5TmpFile()
    from pyscf import ao2mo
    eri = ao2mo.general(mycc.mol, (orbo, mo, mo, mo), fswap, 'eri', intor='int2e_spinor',max_memory=max_memory, verbose=log)
    #eri = ao2mo.general(mycc.mol, (orbo, orbv, orbv, orbv), intor='int2e_spinor',max_memory=max_memory, verbose=log)
    print(np.asarray(fswap['eri']).shape)
    for p0, p1 in lib.prange(0, nocc, blksize):
        tmp = np.asarray(fswap['eri'][p0*nmo:p1*nmo])
        print(p0, nmo, p1, nmo)
        tmp = tmp.reshape(p1-p0, nmo, nmo, nmo)
        eris.oooo[p0:p1] = (tmp[:,:nocc,:nocc,:nocc].transpose(0,2,1,3) - tmp[:,:nocc,:nocc,:nocc].transpose(0,2,3,1))
        eris.ooov[p0:p1] = (tmp[:,:nocc,:nocc,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,:nocc,:nocc].transpose(0,2,3,1))
        eris.ovvv[p0:p1] = (tmp[:,nocc:,nocc:,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,nocc:,nocc:].transpose(0,2,3,1))
        eris.oovv[p0:p1] = (tmp[:,nocc:,:nocc,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,:nocc,nocc:].transpose(0,2,3,1))
        eris.ovov[p0:p1] = (tmp[:,:nocc,nocc:,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,nocc:,:nocc].transpose(0,2,3,1))
        eris.ovvo[p0:p1] = (tmp[:,nocc:,nocc:,:nocc].transpose(0,2,1,3) - tmp[:,:nocc,nocc:,nocc:].transpose(0,2,3,1))
        tmp = None
    cpu0 = log.timer_debug1('transforming ovvv', *cput0)

    #eris.vvvv = feri.create_dataset('vvvv', (nvir, nvir, nvir, nvir), dtype)
    #tril2sq = lib.square_mat_in_trilu_indices(nvir)
    #fswap = lib.H5TmpFile()
    #r_outcore.general(mycc.mol, (orbv, orbv, orbv, orbv), fswap, 'vvvv', max_memory=max_memory, verbose=log)
    #for p0, p1 in lib.prange(0, nvir, blksize):
    #    tmp = np.asarray(fswap['vvvv'][p0*nvir:p1*nvir]).reshape(p1-p0, nvir, nvir, nvir)
    #    eris.vvvv[p0:p1] = tmp.transpose(0,2,1,3)-tmp.transpose(0,2,3,1)
    cput0 = log.timer_debug1('transforming vvvv', *cput0)
    '''
    return eris


if __name__ == '__main__':
    from pyscf import scf, gto
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.verbose = 4 
    mol.build()
    import x2camf_hf
    mf = x2camf_hf.RHF(mol).run()
    mycc = ZCCSD(mf, frozen=2)
    ecc, t1, t2 = mycc.kernel()
    e_corr = mycc.ccsd_t()
    print(e_corr, e_corr + ecc)
    exit()
    e,v = mycc.ipccsd(nroots=8)

    e,v = mycc.eaccsd(nroots=8)

    e,v = mycc.eeccsd(nroots=4)
    
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    e,v = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    e,v = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)

    e,v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
    
    mycc.ccsd_t()
