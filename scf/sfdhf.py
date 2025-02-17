# Spin-free Dirac Coulomb Hartree Fock with the hope that 
# Gaunt or even Breit integrals can also be treated.
import numpy

from pyscf import scf, gto, lib
from pyscf.scf import _vhf
from pyscf.lib import logger

from socutils.somf import somf
def get_ovlp(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    s = mol.intor_symmetric('int1e_ovlp_spinor')
    t = mol.intor_symmetric('int1e_kin_spinor') * 2.
    s1e = numpy.zeros((n4c, n4c), numpy.complex128)
    s1e[:n2c,:n2c] = s
    s1e[n2c:,n2c:] = t * (.5/c)**2
    return s1e

def get_hcore(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    t  = mol.intor_symmetric('int1e_kin_spinor')
    vn = mol.intor_symmetric('int1e_nuc_spinor')
    wn = mol.intor_symmetric('int1e_pnucp_spinor')
    h1e = numpy.empty((n4c, n4c), numpy.complex128)
    h1e[:n2c,:n2c] = vn
    h1e[n2c:,:n2c] = t
    h1e[:n2c,n2c:] = t
    h1e[n2c:,n2c:] = wn * (.25/c**2) - t
    return h1e

get_jk_coulomb = somf.get_jk_sf_coulomb

class SpinFreeDHF(scf.dhf.DHF):

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_hcore(mol)
    
    @lib.with_doc(get_ovlp.__doc__)
    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol)
    
    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        def set_vkscreen(opt, name):
            opt._this.r_vkscreen = _vhf._fpointer(name)

        cpu0 = (logger.process_clock(), logger.perf_counter())
        opt_llll = scf.dhf._VHFOpt(mol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                           'CVHFrkb_q_cond', 'CVHFrkb_dm_cond',
                           direct_scf_tol=self.direct_scf_tol)
        set_vkscreen(opt_llll, 'CVHFrkbllll_vkscreen')

        c1 = .5 / lib.param.LIGHT_SPEED
        opt_ssss = scf.dhf._VHFOpt(mol, 'int2e_pp1pp2_spinor',
                           'CVHFrkbllll_prescreen', 'CVHFrkb_q_cond',
                           'CVHFrkb_dm_cond',
                           direct_scf_tol=self.direct_scf_tol/c1**4)
        opt_ssss.direct_scf_tol = self.direct_scf_tol
        opt_ssss.q_cond *= c1**2
        set_vkscreen(opt_ssss, 'CVHFrkbllll_vkscreen')

        opt_ssll = scf.dhf._VHFOpt(mol, 'int2e_pp1_spinor',
                           'CVHFrkbssll_prescreen',
                           dmcondname='CVHFrkbssll_dm_cond',
                           direct_scf_tol=self.direct_scf_tol)
        opt_ssll.q_cond = numpy.array([opt_llll.q_cond, opt_ssss.q_cond])
        set_vkscreen(opt_ssll, 'CVHFrkbssll_vkscreen')
        logger.timer(self, 'init_direct_scf_coulomb', *cpu0)
        
        opt_gaunt_lsls = None
        opt_gaunt_lssl = None
        return None, None, None, None, None
        return opt_llll, opt_ssll, opt_ssss, opt_gaunt_lsls, opt_gaunt_lssl

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self)
        if self.direct_scf and self._opt.get(omega) is None:
            with mol.with_range_coulomb(omega):
                self._opt[omega] = self.init_direct_scf(mol)
        vhfopt = self._opt.get(omega)
        if vhfopt is None:
            opt_llll = opt_ssll = opt_ssss = opt_gaunt_lsls = opt_gaunt_lssl = None
        else:
            opt_llll, opt_ssll, opt_ssss, opt_gaunt_lsls, opt_gaunt_lssl = vhfopt
        if self.screening is False:
            opt_llll = opt_ssll = opt_ssss = opt_gaunt_lsls = opt_gaunt_lssl = None

        opt_gaunt = (opt_gaunt_lsls, opt_gaunt_lssl)
        vj, vk = get_jk_coulomb(mol, dm, hermi, self._coulomb_level,
                                opt_llll, opt_ssll, opt_ssss, omega, log)
        t1 = log.timer_debug1('Coulomb', *t0)
        if self.with_breit or self.with_gaunt:
            raise NotImplementedError('Gaunt and Breit term')
        log.timer_debug1('Gaunt and Breit term', *t1)

        log.timer('vj and vk', *t0)
        return vj, vk
    
if __name__ == '__main__':
    mol = gto.M(verbose=4,
                atom=[["O", (0., 0., 0.)], [1, (0., -0.757, 0.587)],
                      [1, (0., 0.757, 0.587)]],
                basis='unc-sto-3g')

    mf = SpinFreeDHF(mol).run()
    
