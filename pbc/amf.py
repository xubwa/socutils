import numpy
from functools import reduce
from pyscf.x2c import x2c as mole_x2c
from pyscf.pbc.x2c import x2c1e as pbc_x2c1e
from pyscf import gto as mole_gto
from socutils.somf.amf import get_soc_integrals

class PBCSpinOrbitalX2CAMFHelper(pbc_x2c1e.SpinOrbitalX2C1EHelper):
    def __init__(self, cell, sfx2c=False, with_gaunt=True, with_breit=True, with_aoc=False, prog="sph_atm"):
        self.cell = cell
        mole_x2c.X2C.__init__(self, cell)
        self.sfx2c = sfx2c  # this is still a spinor x2c object, only labels the flavor of soc integral.
        self.pcc = False
        self.gaunt = with_gaunt
        self.gaunt_sd = False
        self.breit = with_breit
        self.aoc = with_aoc
        self.prog = prog
        self.soc_matrix = None
        if self.cell.nucmod == {}:
            self.gau_nuc = False
        elif mole_gto.mole._parse_nuc_mod(self.cell.nucmod) == 2:
            raise NotImplementedError("Gaussian nuclear model is not supported in PBC")
        else:
            self.gau_nuc = False

        print(f'gaunt:{self.gaunt}, breit:{self.breit}')
        print("xuncontracted",self.xuncontract)

    def get_soc_integrals(self):
        so_amf = get_soc_integrals(self, self.cell, self.prog, self.gaunt, self.breit, self.sfx2c, sph=True)
        nao = so_amf.shape[-1] // 2
        # transform spinor orbital basis spin-orbit terms to spin orbital.
        hso = numpy.zeros((nao * 2, nao * 2), dtype=complex)
        ca, cb = self.mol.sph2spinor_coeff()
        hso[:nao, :nao] = reduce(numpy.dot, (ca, so_amf, ca.conj().T))
        hso[nao:, nao:] = reduce(numpy.dot, (cb, so_amf, cb.conj().T))
        hso[:nao, nao:] = reduce(numpy.dot, (ca, so_amf, cb.conj().T))
        hso[nao:, :nao] = reduce(numpy.dot, (cb, so_amf, ca.conj().T))
        return hso

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if cell.has_ecp():
            raise NotImplementedError("ECP/PP is not supported with X2CAMF scheme")

        amf_so = self.get_soc_integrals()
        h1e = pbc_x2c1e.SpinOrbitalX2C1EHelper.get_hcore(self, cell, kpts)

        if kpts is None or numpy.shape(kpts) == (3,):
            h1e += amf_so
        else:
            print("Warning: X2CAMF for k-points might be wrong")
            for i in range(len(kpts)):
                h1e[i] += amf_so
        return h1e
        

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, dft

    cell = gto.Cell()
    cell.atom='''
    He 0.0 0.0 0.0
    '''
    cell.basis = 'sto-3g'
    cell.a = '''
    50.0 0.0 0.0
    0.0 50.0 0.0
    0.0 0.0 50.0
    '''
    cell.unit = 'B'
    cell.verbose = 4
    cell.max_memory = 8000
    cell.build()

    # mf = dft.GKS(cell, xc = "b3lyp").density_fit()
    # mf = pbc_x2c1e.x2c1e_gscf(mf)
    # mf.with_x2c = PBCSpinOrbitalX2CAMFHelper(cell, with_gaunt=True, with_breit=True)
    mf = dft.GKS(cell, xc="b3lyp")
    mf = pbc_x2c1e.x2c1e_gscf(mf)
    mf.with_x2c = PBCSpinOrbitalX2CAMFHelper(cell, with_gaunt=True, with_breit=True)
    mf.kernel()

    # mf = scf.GHF(cell).x2c1e()
    # mf.with_x2c = PBCSpinOrbitalX2CAMFHelper(cell, with_gaunt=True, with_breit=True)
    # mf.kernel()

