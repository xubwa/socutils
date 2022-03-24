from functools import reduce

import os
from turtle import shape
import numpy
import scipy

from pyscf import gto, lib, scf
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.lib import chkfile
from pyscf.x2c import x2c
from pyscf.scf import dhf, ghf

import somf, frac_dhf, writeInput, settings

class X2CAMF(x2c.X2C):
    atom_gso_mf = None
    def __init__(self, mol, with_gaunt=False, with_breit=False, with_aoc=False, prog="mol"):
        x2c.X2C.__init__(self, mol)
        self.gaunt = with_gaunt
        self.breit = with_breit
        self.aoc = with_aoc
        self.prog = prog

    def build(self):
        self.atom_gso_mf = {}
        xmol = self.get_xmol()[0]
        if os.path.isfile('amf.chk'):
            for atom in xmol.elements:
                mat1e = chkfile.load('amf.chk', atom)
                assert(mat1e is not None), \
                'chkfile to store amf integrals don\'t have the specified element, try delete amf.chk and rerun.'
                self.atom_gso_mf[atom] = mat1e
        else:
            for atom in xmol.elements:
                print(atom)
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
                veff_sd = (vj-vk)-(vj_sf-vk_sf)

                g_ll = veff_sd[:nao, :nao]
                g_ls = veff_sd[:nao, nao:]
                g_sl = veff_sd[nao:, :nao]
                g_ss = veff_sd[nao:, nao:]
                g_so_mf = reduce(numpy.dot, (r.T.conj(), (g_ll+reduce(numpy.dot, (g_ls, x)) + reduce(numpy.dot, (x.T.conj(), g_sl))+reduce(numpy.dot, (x.T.conj(), g_ss, x))),r))
                chkfile.dump('amf.chk', atom, g_so_mf)
                self.atom_gso_mf[atom] = g_so_mf

    def get_hcore(self, mol=None):
        c = LIGHT_SPEED

        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol()

        hcore = x2c.X2C.get_hcore(self, xmol)

        if(self.prog == "mol"):
            if self.atom_gso_mf is None:
                self.build()

            atom_slices = xmol.aoslice_2c_by_atom()
            for ia in range(xmol.natm):
                ishl0, ishl1, c0, c1 = atom_slices[ia]
                hcore[c0:c1, c0:c1] += self.atom_gso_mf[xmol.elements[ia]]

            if self.basis is not None:
                s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
                s21 = gto.mole.intor_cross('int1e_ovlp_spinor', xmol, mol)
                c = lib.cho_solve(s22, s21)
                hcore = reduce(numpy.dot, (c.T.conj(), hcore, c))
            elif self.xuncontract:
                np, nc = contr_coeff_nr.shape
                contr_coeff = numpy.zeros((np * 2, nc * 2))
                contr_coeff[0::2, 0::2] = contr_coeff_nr
                contr_coeff[1::2, 1::2] = contr_coeff_nr
                hcore = reduce(numpy.dot, (contr_coeff.T.conj(), hcore, contr_coeff))
        elif(self.prog == "sph_atm"):
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
                        hcore[ii][jj] = hcore[ii][jj] + complex(lines[ii*hcore.shape[0]+jj])
        return hcore


class X2CAMF_RHF(x2c.X2C_RHF):
    def __init__(self, mol, with_gaunt=False, with_breit=False, with_aoc=False, prog="mol"):
        x2c.X2C_RHF.__init__(self, mol)
        self.with_x2c = X2CAMF(mol, with_gaunt, with_breit, with_aoc, prog)
        self._keys = self._keys.union(['with_x2c'])

def x2camf_ghf(mf):
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = X2CAMF(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, X2CAMF):
            mf.with_x2c = X2CAMF(mf.mol)
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
            mf_class.__init__(self, mol, *args, **kwargs)
            self.with_x2c = X2CAMF(mf.mol)
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

    with_x2c = X2CAMF(mf.mol)
    return mf.view(X2CAMF_GSCF).add_keys(with_x2c=with_x2c)

if __name__ == '__main__':
    mol = gto.M(verbose=3,
                atom=[["O", (0., 0., 0.)], [1, (0., -0.757, 0.587)],
                      [1, (0., 0.757, 0.587)]],
                basis='ccpvdz')
    mf = X2CAMF_RHF(mol)
    e_spinor = mf.scf()
    gmf = x2camf_ghf(scf.GHF(mol))
    e_ghf = gmf.kernel()
    print("Energy from spinor X2CAMF:    %16.8g" % e_spinor)
    print("Energy from ghf-based X2CAMF: %16.8g" % e_ghf)
