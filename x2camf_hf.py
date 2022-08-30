from functools import reduce

import os
from turtle import shape
import numpy
import scipy

from pyscf import gto, lib, scf
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.lib import chkfile, logger
from pyscf.x2c import x2c
from pyscf.scf import hf, dhf, ghf

import somf, frac_dhf, writeInput, settings, zquatev

x2camf = None
try:
    import x2camf
except ImportError:
    pass

class X2CAMF(x2c.X2C):
    atom_gso_mf = None
    def __init__(self, mol, sfx2c=False, with_gaunt=False, with_breit=False, with_aoc=False, prog="sph_atm"):
        x2c.X2C.__init__(self, mol)
        self.sfx2c = sfx2c # this is still a spinor x2c object, only labels the flavor of soc integral.
        self.gaunt = with_gaunt
        self.breit = with_breit
        self.aoc = with_aoc
        self.prog = prog
        self.soc_matrix = None

    def initialize_x2camf(self):
        self.atom_gso_mf = {}
        xmol = self.get_xmol()[0]
        if os.path.isfile('amf.chk'):
            for atom in xmol.elements:
                mat1e = chkfile.load('amf.chk', atom)
                assert(mat1e is not None), \
                'chkfile to store amf integrals don\'t have the specified element, try delete amf.chk and rerun.'
                self.atom_gso_mf[atom] = mat1e
        else:
            for atom in set(xmol.elements):
                atm_id = gto.elements.charge(atom)
                spin = atm_id % 2
                mol_atom = gto.M(verbose=xmol.verbose,atom=[[atom, [0, 0, 0]]], basis=xmol.basis, spin=spin)
                atm_x2c = x2c.X2C(mol_atom)
                mol_atom = atm_x2c.get_xmol(mol_atom)[0]
                conf = gto.elements.CONFIGURATION[atm_id]
                # generate configuration for spherical symmetry atom
                if conf[0] % 2 == 0:
                    if conf[1] % 6 == 0:
                        if conf[2] % 10 == 0:
                            if conf[3] % 14 == 0:
                                nopen = 0
                                nact = 0
                            else:
                                nopen = 7
                                nact = conf[3] % 14
                        else:
                            nopen = 5
                            nact = conf[2] % 10
                    else:
                        nopen = 3
                        nact = conf[1] % 6
                else:
                    nopen = 1
                    nact = 1
                mf_atom = frac_dhf.FRAC_RDHF(mol_atom, nopen*2, nact)
                mf_atom.with_breit = self.breit
                mf_atom.with_gaunt = self.gaunt
                mf_atom.kernel()
                dm = mf_atom.make_rdm1()
                nao = mol_atom.nao_2c()
                cl = mf_atom.mo_coeff[:nao, nao:]
                cs = mf_atom.mo_coeff[nao:, nao:]
                x = numpy.linalg.solve(cl.T, cs.T).T
                s = mol_atom.intor_symmetric('int1e_ovlp_spinor')
                t = mol_atom.intor_symmetric('int1e_spsp_spinor') * .5
                s1 = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5 / LIGHT_SPEED**2)
                r = x2c._get_r(s, s1)

                mo2c = cl
                dm2c = numpy.dot(mo2c * mf_atom.mo_occ[nao:], mo2c.T.conj())

                vj_sf, vk_sf = somf.get_jk_sf_coulomb(mol_atom, dm, 1)
                vj, vk = dhf.get_jk_coulomb(mol_atom, dm, 1)
                opt_llll, opt_ssll, opt_ssss, opt_gaunt = mf_atom.opt
                if self.breit:
                    vj1, vk1 = dhf._call_veff_gaunt_breit(mol_atom, dm, 1, opt_gaunt, True)
                    vj += vj1
                    vk += vk1
                elif self.gaunt:
                    vj1, vk1 = dhf._call_veff_gaunt_breit(mol_atom, dm, 1, opt_gaunt, False)
                    vj += vj1
                    vk += vk1
 
                veff_sd = (vj-vk)-(vj_sf-vk_sf)

                g_ll = veff_sd[:nao, :nao]
                g_ls = veff_sd[:nao, nao:]
                g_sl = veff_sd[nao:, :nao]
                g_ss = veff_sd[nao:, nao:]
                g_so_mf = reduce(numpy.dot, (r.T.conj(), (g_ll+reduce(numpy.dot, (g_ls, x)) + reduce(numpy.dot, (x.T.conj(), g_sl))+reduce(numpy.dot, (x.T.conj(), g_ss, x))),r))
                chkfile.dump('amf.chk', atom, g_so_mf)
                self.atom_gso_mf[atom] = g_so_mf

    def get_soc_integrals(self, mol=None):
        c = LIGHT_SPEED
        
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol()
        if self.soc_matrix is not None:
            return self.soc_matrix

        soc_matrix = numpy.zeros((xmol.nao_2c(), xmol.nao_2c()), dtype=complex)

        if(self.prog == "sph_atm"):
            if self.sfx2c:
                spin_free = True
                two_c = True
            else:
                spin_free = False
                two_c = False
            soc_matrix = x2camf.amfi(self, spin_free, two_c, self.gaunt, self.breit)

        elif(self.prog == "sph_atm_legacy"): # keep this legacy interface for a sanity check.
            writeInput.write_input(self.mol, self.gaunt, self.breit, self.aoc)
            print(settings.AMFIEXE)
            os.system(settings.AMFIEXE)
            with open("amf_int","r") as ifs:
                lines = ifs.readlines()
            if(len(lines) != hcore.shape[0]**2):
                print("Something went wrong. The dimension of hcore and amfi calculations do NOT match.")
                exit()
            else:
                for ii in range(hcore.shape[0]):
                    for jj in range(hcore.shape[1]):
                        soc_matrix[ii][jj] = complex(lines[ii*hcore.shape[0]+jj])
        else: # should fall back to the expensive way.
            if self.atom_gso_mf is None:
                self.initialize_x2camf()

            atom_slices = xmol.aoslice_2c_by_atom()
            for ia in range(xmol.natm):
                ishl0, ishl1, c0, c1 = atom_slices[ia]
                soc_matrix[c0:c1, c0:c1] += self.atom_gso_mf[xmol.elements[ia]]
        self.soc_matrix = soc_matrix
        return soc_matrix

            

    def get_hcore(self, mol=None):
        c = LIGHT_SPEED

        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol()

        hcore = x2c.X2C.get_hcore(self, xmol)
        soc_matrix = self.get_soc_integrals()
        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
            s21 = gto.mole.intor_cross('int1e_ovlp_spinor', xmol, mol)
            c = lib.cho_solve(s22, s21)
            soc_matrix = reduce(numpy.dot, (c.T.conj(), soc_matrix, c))
        elif self.xuncontract:
            np, nc = contr_coeff_nr.shape
            contr_coeff = numpy.zeros((np * 2, nc * 2))
            contr_coeff[0::2, 0::2] = contr_coeff_nr
            contr_coeff[1::2, 1::2] = contr_coeff_nr
            soc_matrix = reduce(numpy.dot, (contr_coeff.T.conj(), soc_matrix, contr_coeff))

        return hcore + soc_matrix


class SCF(x2c.SCF):
    nopen = None
    nact = None
    def __init__(self, mol, nopen=0, nact=0, with_gaunt=False, with_breit=False, with_aoc=False, prog="sph_atm"):
        hf.SCF.__init__(self, mol)
        self.with_x2c = X2CAMF(mol, with_gaunt=with_gaunt, with_breit=with_breit, with_aoc=with_aoc, prog=prog)
        self._keys = self._keys.union(['with_x2c'])
        self.nopen = nopen 
        self.nact = nact

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        nopen = self.nopen
        nact = self.nact
        nclose = mol.nelectron - nact
        c = lib.param.LIGHT_SPEED
        n2c = len(mo_energy)
        mo_occ = numpy.zeros(n2c)

        if nopen == 0:
            mo_occ[:mol.nelectron] = 1
        else:
            mo_occ[:nclose] = 1
            mo_occ[nclose:nclose+nopen] = 1.*nact/nopen

        if self.verbose >= logger.INFO:
            if nopen == 0:
                homo_ndx = mol.nelectron
            else:
                homo_ndx = nclose + nopen
            logger.info(self, 'HOMO %d = %.12g  LUMO %d = %.12g',
                            homo_ndx, mo_energy[homo_ndx-1],
                            homo_ndx+1, mo_energy[homo_ndx])
            logger.debug(self, 'mo_energy = %s', mo_energy[:])
        return mo_occ

X2CAMF_SCF = SCF

class UHF(SCF):
    def to_ks(self, xc='HF'):
        from pyscf.x2c import sft
        mf = self.view(dft.UKS)
        mf.converged = False
        return mf

X2CAMF_UHF = UHF

class RHF(UHF):
    def __init__(self, mol, nopen=0, nact=0, with_gaunt=False, with_breit=False, with_aoc=False, prog="sph_atm"):
        super().__init__(mol)
        if dhf.zquatev is None:
            raise RuntimeError('zquatev library is required to perform Kramers-restricted X2C-RHF')

    def _eigh(self, h, s):
        return dhf.zquatev.solve_KR_FCSCE(self.mol, h, s)

    def to_ks(self, xc='HF'):
        from pyscf.x2c import dft
        mf = self.view(dft.RKS)
        mf.converged = False
        return mf

X2CAMF_RHF = RHF

def x2camf_ghf(mf, *args, **kwargs):
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = X2CAMF(mf.mol, *args, **kwargs)
            return mf
        elif not isinstance(mf.with_x2c, X2CAMF):
            mf.with_x2c = X2CAMF(mf.mol, *args, **kwargs)
            return mf
        else:
            return mf
        
    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__

    class X2CAMF_GSCF(x2c._X2C_SCF, mf_class):
        __doc__ = doc + '''
        Attributes for spin-orbital spin-orbit mean field X2C:
            with_x2c : X2C object
        '''
        def __init__(self, mol, *args, **kwargs):
            mf_class.__init__(self, mol)
            self.with_x2c = X2CAMF(mf.mol, *args, **kwargs)
            self._keys = self._keys.union(['with_x2c'])

        def get_hcore(self, mol=None):
            if mol is None: mol = self.mol
            hcore = with_x2c.get_hcore(mol)
            hso = numpy.zeros(hcore.shape, dtype=complex)
            nao = hcore.shape[0]//2
            # transform spinor orbital basis spin-orbit terms to spin orbital.
            ca, cb = mol.sph2spinor_coeff()
            hso[:nao,:nao] = reduce(numpy.dot,(ca, hcore, ca.conj().T))
            hso[nao:, nao:] = reduce(numpy.dot, (cb, hcore, cb.conj().T))
            hso[:nao,nao:] = reduce(numpy.dot, (ca, hcore, cb.conj().T))
            hso[nao:, :nao] = reduce(numpy.dot, (cb, hcore, ca.conj().T))
            return hso

    with_x2c = X2CAMF(mf.mol, *args, **kwargs)
    return mf.view(X2CAMF_GSCF).add_keys(with_x2c=with_x2c)

if __name__ == '__main__':
    mol = gto.M(verbose=3,
                atom=[["O", (0., 0., -0.12390941)], 
		              [1, (0., -1.42993701, 0.98326612)],
                      [1, (0.,  1.42993701, 0.98326612)]],
                basis='unc-ccpvdz',
                unit = 'Bohr')
    import os
    os.system('rm amf.chk')
    mf = X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
    e_spinor = mf.scf()
    os.system('rm amf.chk')
    mf = X2CAMF_RHF(mol, with_gaunt=True, with_breit=False)
    e_gaunt = mf.scf()
    os.system('rm amf.chk')
    mf = X2CAMF_RHF(mol, with_gaunt=True, with_breit=True)
    e_breit = mf.scf()
    gmf = x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
    e_ghf = gmf.kernel()
    print("Energy from spinor X2CAMF(Coulomb):    %16.10g" % e_spinor)
    print("Energy from spinor X2CAMF(Gaunt):      %16.10g" % e_gaunt)
    print("Energy from spinor X2CAMF(Breit):      %16.10g" % e_breit)
    print("Energy from ghf-based X2CAMF(Breit):   %16.10g" % e_ghf)
