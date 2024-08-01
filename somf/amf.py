import os
import numpy
from functools import reduce
from pyscf import gto
from pyscf.scf import dhf
from pyscf.x2c import x2c
from pyscf.lib import chkfile
from pyscf.lib.parameters import LIGHT_SPEED
from socutils.somf import somf, writeInput, settings
from socutils.scf import frac_dhf

try:
    import x2camf
except ImportError:
    pass


def initialize_x2camf(x2cobj):
    x2cobj.atom_gso_mf = {}
    xmol = x2cobj.get_xmol()[0]
    if os.path.isfile('amf.chk'):
        for atom in xmol.elements:
            mat1e = chkfile.load('amf.chk', atom)
            assert(mat1e is not None), \
                'chkfile to store amf integrals don\'t have the specified element, try delete amf.chk and rerun.'
            x2cobj.atom_gso_mf[atom] = mat1e
    else:
        for atom in set(xmol.elements):
            atm_id = gto.elements.charge(atom)
            spin = atm_id % 2
            mol_atom = gto.M(verbose=xmol.verbose, atom=[[atom, [0, 0, 0]]], basis=xmol.basis, spin=spin)
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
            mf_atom = frac_dhf.FRAC_RDHF(mol_atom, nopen * 2, nact)
            mf_atom.with_breit = x2cobj.breit
            mf_atom.with_gaunt = x2cobj.gaunt
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
            # mo2c = cl
            # dm2c = numpy.dot(mo2c * mf_atom.mo_occ[nao:], mo2c.T.conj())
            vj_sf, vk_sf = somf.get_jk_sf_coulomb(mol_atom, dm, 1)
            vj, vk = dhf.get_jk_coulomb(mol_atom, dm, 1)
            opt_llll, opt_ssll, opt_ssss, opt_gaunt = mf_atom.opt
            if x2cobj.breit:
                vj1, vk1 = dhf._call_veff_gaunt_breit(mol_atom, dm, 1, opt_gaunt, True)
                vj += vj1
                vk += vk1
            elif x2cobj.gaunt:
                vj1, vk1 = dhf._call_veff_gaunt_breit(mol_atom, dm, 1, opt_gaunt, False)
                vj += vj1
                vk += vk1
            veff_sd = (vj - vk) - (vj_sf - vk_sf)
            g_ll = veff_sd[:nao, :nao]
            g_ls = veff_sd[:nao, nao:]
            g_sl = veff_sd[nao:, :nao]
            g_ss = veff_sd[nao:, nao:]
            g_so_mf = reduce(numpy.dot, (r.T.conj(),
                                         (g_ll
                                         + reduce(numpy.dot,(g_ls, x))
                                         + reduce(numpy.dot,(x.T.conj(), g_sl))
                                         + reduce(numpy.dot,(x.T.conj(), g_ss, x))),
                                        r))
            chkfile.dump('amf.chk', atom, g_so_mf)
            x2cobj.atom_gso_mf[atom] = g_so_mf


def get_soc_integrals(x2cobj, mol=None, prog="sph_atm", with_gaunt=False, with_breit=False, with_gaunt_sd=False, sfx2c=False, sph=False):
    if mol is None:
        mol = x2cobj.mol
    if mol.has_ecp():
        raise NotImplementedError
    xmol, contr_coeff_nr = x2cobj.get_xmol()
    nao_2c = xmol.nao_2c()
    if x2cobj.soc_matrix is not None:
        return x2cobj.soc_matrix
    soc_matrix = numpy.zeros((xmol.nao_2c(), xmol.nao_2c()), dtype=complex)
    if (x2cobj.prog == "sph_atm"):
        if x2cobj.sfx2c:
            spin_free = True
            two_c = True
        else:
            spin_free = False
            two_c = False
        print(x2cobj.gau_nuc)
        soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=x2cobj.gaunt, with_gauge=x2cobj.breit,
                                 with_gaunt_sd=x2cobj.gaunt_sd, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc, aoc=x2cobj.aoc)
    elif (x2c.prog == "sph_atm_legacy"):  # keep this legacy interface for a sanity check.
        writeInput.write_input(x2cobj.mol, x2cobj.gaunt, x2cobj.breit, x2cobj.aoc)
        print(settings.AMFIEXE)
        os.system(settings.AMFIEXE)
        with open("amf_int", "r") as ifs:
            lines = ifs.readlines()
        if (len(lines) != nao_2c**2):
            print("Something went wrong. The dimension of hcore and amfi calculations do NOT match.")
            exit()
        else:
            for ii in range(nao_2c):
                for jj in range(nao_2c):
                    soc_matrix[ii][jj] = complex(lines[ii * nao_2c + jj])
    else:  # should fall back to the expensive way.
        if x2cobj.atom_gso_mf is None:
            x2cobj.initialize_x2camf()
        atom_slices = xmol.aoslice_2c_by_atom()
        for ia in range(xmol.natm):
            ishl0, ishl1, c0, c1 = atom_slices[ia]
            soc_matrix[c0:c1, c0:c1] += x2cobj.atom_gso_mf[xmol.elements[ia]]

    if x2cobj.xuncontract:
        np, nc = contr_coeff_nr.shape
        contr_coeff = numpy.zeros((np * 2, nc * 2))
        contr_coeff[0::2, 0::2] = contr_coeff_nr
        contr_coeff[1::2, 1::2] = contr_coeff_nr
        soc_matrix = reduce(numpy.dot, (contr_coeff.T.conj(), soc_matrix, contr_coeff))

    if sph:
        mol = x2cobj.mol
        ca, cb = x2cobj.mol.sph2spinor_coeff()
        nao = mol.nao_nr()
        hso = numpy.zeros_like(soc_matrix, dtype=complex)
        hso[:nao, :nao] = reduce(numpy.dot, (ca, soc_matrix, ca.conj().T))
        hso[nao:, nao:] = reduce(numpy.dot, (cb, soc_matrix, cb.conj().T))
        hso[:nao, nao:] = reduce(numpy.dot, (ca, soc_matrix, cb.conj().T))
        hso[nao:, :nao] = reduce(numpy.dot, (cb, soc_matrix, ca.conj().T))
        soc_matrix = hso
    x2cobj.soc_matrix = soc_matrix
    return soc_matrix


class SpinorX2CAMFHelper(x2c.SpinorX2CHelper):
    atom_gso_mf = None

    def __init__(self, mol, sfx2c=False, with_gaunt=True, with_breit=True, with_gaunt_sd=False, with_aoc=False, with_pcc=False, prog="sph_atm"):
        x2c.X2C.__init__(self, mol)
        self.sfx2c = sfx2c  # this is still a spinor x2c object, only labels the flavor of soc integral.
        self.gaunt = with_gaunt
        self.breit = with_breit
        self.gaunt_sd = with_gaunt_sd
        self.aoc = with_aoc
        self.pcc = with_pcc
        self.prog = prog
        self.soc_matrix = None
        if mol.nucmod != {}:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        print(f'Gaussian nuclear model : {self.gau_nuc}')

    def initialize_x2camf(self):
        initialize_x2camf(self)

    def get_soc_integrals(self):
        return get_soc_integrals(self, self.mol, self.prog, self.gaunt, self.breit, self.sfx2c)

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol()
        hcore = x2c.X2C.get_hcore(self, self.mol)
        soc_matrix = self.get_soc_integrals()

        return hcore + soc_matrix


class SpinOrbitalX2CAMFHelper(x2c.SpinOrbitalX2CHelper):
    atom_gso_mf = None

    def __init__(self, mol, sfx2c=False, with_gaunt=True, with_breit=True, with_aoc=False, prog="sph_atm"):
        x2c.X2C.__init__(self, mol)
        self.sfx2c = sfx2c  # this is still a spinor x2c object, only labels the flavor of soc integral.
        self.pcc = False
        self.gaunt = with_gaunt
        self.gaunt_sd = False
        self.breit = with_breit
        print(f'gaunt:{self.gaunt}, breit:{self.breit}')
        self.aoc = with_aoc
        self.prog = prog
        self.soc_matrix = None
        if gto.mole._parse_nuc_mod(self.mol.nucmod) == 2:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        self.gau_nuc = False
        print(self.xuncontract)

    def initialize_x2camf(self):
        initialize_x2camf(self)

    def get_soc_integrals(self):
        so_amf = get_soc_integrals(self, self.mol, self.prog, self.gaunt, self.breit, self.sfx2c, sph=True)
        nao = so_amf.shape[-1] // 2
        # transform spinor orbital basis spin-orbit terms to spin orbital.
        hso = numpy.zeros((nao * 2, nao * 2), dtype=complex)
        ca, cb = self.mol.sph2spinor_coeff()
        hso[:nao, :nao] = reduce(numpy.dot, (ca, so_amf, ca.conj().T))
        hso[nao:, nao:] = reduce(numpy.dot, (cb, so_amf, cb.conj().T))
        hso[:nao, nao:] = reduce(numpy.dot, (ca, so_amf, cb.conj().T))
        hso[nao:, :nao] = reduce(numpy.dot, (cb, so_amf, ca.conj().T))
        return so_amf

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol()
        hcore = x2c.SpinOrbitalX2CHelper.get_hcore(self, self.mol)
        soc_matrix = self.get_soc_integrals()

        return hcore + soc_matrix


