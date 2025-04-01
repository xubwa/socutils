#
# Author Xubo Wang <wangxubo0201@outlook.com
#

import time
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf import scf
from pyscf.ao2mo import r_outcore
from pyscf.ao2mo import nrr_outcore
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import gccsd
from socutils.cc import gintermediates as imd
from pyscf.x2c import x2c
from pyscf import __config__

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

einsum = lib.einsum

def update_amps(cc, t1, t2, eris, alg='new'):
    assert(isinstance(eris, _PhysicistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
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
    if (alg == 'old'):
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    elif (alg == 'new'):
        t2new += imd.update_t2_vvvv(t1, t2, tau, eris)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new

class ZCCSD(gccsd.GCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, with_mmf=False, erifile=None):
        #assert(isinstance(mf, x2c.RHF))
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.with_mmf = with_mmf
        self.feri = erifile
        self.eris = None
        if not isinstance(mf, scf.dhf.DHF) and self.with_mmf:
            print('WARNING! SCF reference is not four component, with_mmf option is doing nothing')

    update_amps = update_amps

    def ao2mo(self, mo_coeff=None, swapfile=None):
        nmo = self.nmo
        if mo_coeff is None:
            mo_coeff=self.mo_coeff
        self.eris = _make_eris_outcore(self, mo_coeff, self.feri, swapfile)
        return self.eris

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=None):
        if eris is None:
            if self.eris is None:
                eris = self.ao2mo(self.mo_coeff)
                self.eris = eris
            else:
                eris = self.eris

        if mbpt2:
            from pyscf.mp import gmp2
            pt = gmp2.GMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        # Initialize orbspin so that we can attach it to t1, t2
        if getattr(self.mo_coeff, 'orbspin', None) is None:
            orbspin = scf.ghf.guess_orbspin(self.mo_coeff)
            if not np.any(orbspin == -1):
                self.mo_coeff = lib.tag_array(self.mo_coeff, orbspin=orbspin)

        e_corr, self.t1, self.t2 = ccsd.CCSDBase.ccsd(self, t1, t2, eris)
        if getattr(eris, 'orbspin', None) is not None:
            self.t1 = lib.tag_array(self.t1, orbspin=eris.orbspin)
            self.t2 = lib.tag_array(self.t2, orbspin=eris.orbspin)
        return e_corr, self.t1, self.t2


    def ccsd_t(self, t1=None, t2=None, eris=None, alg='occ_loop'):
        from socutils.cc import gccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            if self.eris is None:
                eris = self.ao2mo(self.mo_coeff)
            else:
                eris = self.eris
        return gccsd_t.kernel(self, eris, t1, t2, self.verbose, alg=alg)

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

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        fock_ref = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        print(fock_ref.diagonal())
        import scipy
        #print(scipy.linalg.eigh(fockao, mycc._scf.get_ovlp())[0])
        if isinstance(mycc._scf, scf.dhf.DHF) and mycc.with_mmf:
            print('Initialize X2Cmmf Fock matrix')
            print('X2C transformation 4c FOCK matrix')
            n2c = fockao.shape[0]//2
            from socutils.somf.x2c_grad import x2c1e_hfw0_block
            ovlp = mycc._scf.get_ovlp()
            print(fockao[0,0])
            print(fockao[n2c,0])
            print(fockao[0,n2c])
            print(fockao[n2c,n2c])
            _, _, _, _, r, l, _, _ = x2c1e_hfw0_block(fockao[:n2c,:n2c], fockao[n2c:,:n2c], fockao[:n2c,n2c:],
                                                    fockao[n2c:,n2c:], ovlp[:n2c,:n2c], ovlp[n2c:,n2c:])
            fock_x2c = reduce(np.dot, (r.T.conj(), l, r))
            rinv = np.linalg.inv(r)
            mo_l = mo_coeff[:n2c,:]
            mo_x2c = np.dot(rinv,mo_l)
            mo_coeff = mo_x2c
        
            fockao = fock_x2c 

        mo_idx = ccsd.get_frozen_mask(mycc)
        mo_coeff = mo_coeff[:,mo_idx]
        self.mo_coeff = mo_coeff

        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        print(self.e_hf)
        self.nocc = mycc.nocc
        self.mol = mycc.mol
        mo_e = self.mo_energy = self.fock.diagonal().real
        #print(mo_e)
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for ZCCSD', gap)
        return self 

def _make_eris_outcore(mycc, mo_coeff=None, erifile=None, swapfile=None, mod='continue', transformed=[]):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    print(mo_coeff.shape)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    assert(eris.mo_coeff.dtype == complex)
    print(nao, nmo)
    print(nocc, nvir)
    print(swapfile)

    if erifile is not None:
        feri = eris.feris = lib.H5TmpFile(erifile)
    else:
        feri = eris.feris = lib.H5TmpFile()
    
    if len(feri.keys()) is not 0:
        feri.close()
        feri = eris.feris = lib.H5TmpFile(erifile, mode='r')
        eris.oooo = feri['oooo']
        eris.ooov = feri['ooov']
        eris.oovv = feri['oovv']
        eris.ovov = feri['ovov']
        eris.ovvo = feri['ovvo']
        eris.ovvv = feri['ovvv']
        eris.vvvv = feri['vvvv']
        return eris
    else:
        dtype = np.result_type(eris.mo_coeff).char
        eris.oooo = feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), dtype)
        eris.ooov = feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), dtype)
        eris.oovv = feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), dtype)
        eris.ovov = feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), dtype)
        eris.ovvo = feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), dtype)
        eris.ovvv = feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), dtype)
        eris.vvvv = feri.create_dataset('vvvv', (nvir, nvir, nvir, nvir), dtype)

    max_memory = mycc.max_memory-lib.current_memory()[0]
    blksize = min(nocc, max(2, int(max_memory*1e6/16/(nmo**3*2))))
    max_memory = max(MEMORYMIN, max_memory)

    mo = eris.mo_coeff
    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]
    print(mo.shape)
    print(orbo.shape, orbv.shape)

    from pyscf import ao2mo
    mf = mycc._scf
    mol = mycc.mol
    def fill_eris(ftmp, multip):
        for p0, p1 in lib.prange(0, nocc, blksize):
            tmp = multip*np.asarray(ftmp['eri'][p0*nmo:p1*nmo])
            tmp = tmp.reshape(p1-p0, nmo, nmo, nmo)
            eris.oooo[p0:p1] += (tmp[:,:nocc,:nocc,:nocc].transpose(0,2,1,3) - tmp[:,:nocc,:nocc,:nocc].transpose(0,2,3,1))
            eris.ooov[p0:p1] += (tmp[:,:nocc,:nocc,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,:nocc,:nocc].transpose(0,2,3,1))
            eris.ovvv[p0:p1] += (tmp[:,nocc:,nocc:,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,nocc:,nocc:].transpose(0,2,3,1))
            eris.oovv[p0:p1] += (tmp[:,nocc:,:nocc,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,:nocc,nocc:].transpose(0,2,3,1))
            eris.ovov[p0:p1] += (tmp[:,:nocc,nocc:,nocc:].transpose(0,2,1,3) - tmp[:,nocc:,nocc:,:nocc].transpose(0,2,3,1))
            eris.ovvo[p0:p1] += (tmp[:,nocc:,nocc:,:nocc].transpose(0,2,1,3) - tmp[:,:nocc,nocc:,nocc:].transpose(0,2,3,1))
            tmp = None
        for p0, p1 in lib.prange(nocc, nmo, blksize):
            tmp = multip*np.asarray(ftmp['eri'][p0*nmo:p1*nmo])
            tmp = tmp.reshape(p1-p0, nmo, nmo, nmo)
            eris.vvvv[p0-nocc:p1-nocc] += (tmp[:,nocc:,nocc:,nocc:].transpose(0,2,1,3)-tmp[:,nocc:,nocc:,nocc:].transpose(0,2,3,1))
            tmp = None

    if isinstance(mf, scf.dhf.DHF) and not mycc.with_mmf:
        c1 = 0.5 / lib.param.LIGHT_SPEED
        nao_2c = mo.shape[0]//2
        mo_l = mo[:nao_2c,:]
        mo_s = mo[nao_2c:,:]
        mos = [[mo_l,mo_l,mo_l,mo_l],#llll
               [mo_l,mo_l,mo_s,mo_s],#llss
               [mo_s,mo_s,mo_l,mo_l],#ssll
               [mo_s,mo_s,mo_s,mo_s]]
        if swapfile is not None:
            fswap = lib.H5TmpFile(swapfile)
        else:
            fswap = lib.H5TmpFile()
        if mf.with_gaunt:
            p = 'int2e_breit_' if mf.with_breit else 'int2e_'
            multips = (c1**2,)*4 if mf.with_breit else (-c1**2,)*4
            mos = [[mo_l, mo_s, mo_s, mo_l],
                   [mo_l, mo_s, mo_l, mo_s],
                   [mo_s, mo_l, mo_s, mo_l],
                   [mo_s, mo_l, mo_l, mo_s]]
            intors = [p+'ssp1sps2_spinor', p+'ssp1ssp2_spinor', p+'sps1sps2_spinor', p+'sps1ssp2_spinor']
            #lssl,lsls,slsl,slls
            for mo, intor, multip in zip(mos, intors, multips):
                if swapfile is not None:
                    fswap = lib.H5TmpFile(swapfile)
                else:
                    fswap = lib.H5TmpFile()
                eri = ao2mo.general(mol, mo, fswap, 'eri', intor=intor, aosym='s1', comp=1, max_memory=max_memory, verbose=mycc.verbose)
                fill_eris(fswap, multip)
                del fswap['eri']
        intors = ['int2e_spinor','int2e_spsp1_spinor','int2e_spsp2_spinor','int2e_spsp1spsp2_spinor']
        multips = [1.0, c1**2, c1**2, c1**4] 
        for mo, intor, multip in zip(mos, intors, multips):
            eri = ao2mo.general(mol, mo, fswap, 'eri', intor=intor, max_memory=max_memory, verbose=mycc.verbose)
            fill_eris(fswap, multip)
            del fswap['eri']
        del fswap
    else:
        if swapfile is not None:
            fswap = lib.H5TmpFile(swapfile)
        else:
            fswap = lib.H5TmpFile()
        #eri = ao2mo.general(mycc.mol, (mo, mo, mo, mo), fswap, 'eri', intor='int2e_spinor',max_memory=max_memory, verbose=mycc.verbose)
        sph2spinor = np.vstack(mycc.mol.sph2spinor_coeff())
        mog = np.dot(sph2spinor, mo)
        eri = nrr_outcore.general(mycc.mol, (mo, mo, mo, mo), fswap, 'eri', intor='int2e_sph', motype='j-spinor', max_memory=max_memory, verbose=mycc.verbose)
        fill_eris(fswap, 1.0)
        del fswap
    
    if erifile is not None: # reopen in read only mode to protect hdf5 file
        feri.close()
        feri = eris.feris = lib.H5TmpFile(erifile, mode='r')
        eris.oooo = feri['oooo']
        eris.ooov = feri['ooov']
        eris.oovv = feri['oovv']
        eris.ovov = feri['ovov']
        eris.ovvo = feri['ovvo']
        eris.ovvv = feri['ovvv']
        eris.vvvv = feri['vvvv']
        return eris
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
    mf = x2c.RHF(mol).run()
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
