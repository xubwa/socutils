from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf, dhf, ghf, _vhf

def spinor2sph(mol, spinor):
    assert (spinor.shape[0] == mol.nao_2c()), "spinor integral must be of shape (nao_2c, nao_2c)"
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    ints_sph = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    return ints_sph

def sph2spinor(mol, sph):
    assert (sph.shape[0] == mol.nao_nr() * 2), "spherical integral must be of shape (nao_nr, nao_nr)"
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    ints_spinor = lib.einsum('ip,pq,qj->ij', c2.T.conj(), sph, c2)
    return ints_spinor

def _proj_dmll(mol_nr, dm_nr, mol):
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, numpy.eye(mol_nr.nao_nr()), mol)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm_ll = (dm_ll + dhf.time_reversal_matrix(mol, dm_ll)) * .5
    return dm_ll

def get_jk(mol, dm, hermi=1, mf_opt=None, with_j=True, with_k=True, omega=None):
    '''non-relativistic J/K matrices (without SSO,SOO etc) in the j-adapted
    spinor basis.
    '''
    vj, vk = _vhf.rdirect_mapdm('int2e_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dm, 1,
                                mol._atm, mol._bas, mol._env, mf_opt)
    vj = vj.reshape(dm.shape)
    vk = vk.reshape(dm.shape)
    return dhf._jk_triu_(mol, vj, vk, hermi)

make_rdm1 = hf.make_rdm1

def init_guess_by_minao(mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    dm = hf.init_guess_by_minao(mol)
    return _proj_dmll(mol, dm, mol)

#def init_guess_by_1e(mol):
#    '''Initial guess from one electron system.'''
#    mf = UHF(mol)
#    return mf.init_guess_by_1e(mol)

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    dm = hf.init_guess_by_atom(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    dm = dhf.init_guess_by_chkfile(mol, chkfile_name, project)
    n2c = dm.shape[0] // 2
    return dm[:n2c,:n2c].copy()

def get_init_guess(mol, key='minao'):
    if callable(key):
        return key(mol)
    #elif key.lower() == '1e':
    #    return init_guess_by_1e(mol)
    elif key.lower() == 'atom':
        return init_guess_by_atom(mol)
    elif key.lower() == 'chkfile':
        raise RuntimeError('Call pyscf.scf.hf.init_guess_by_chkfile instead')
    else:
        return init_guess_by_minao(mol)

def get_hcore(mol):
    '''Core Hamiltonian

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> scf.hf.get_hcore(mol)
    array([[-0.93767904, -0.59316327],
           [-0.59316327, -0.93767904]])
    '''
    h = mol.intor_symmetric('int1e_kin_spinor')

    if mol._pseudo:
        # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
        # may exist if mol is converted from cell object.
        from pyscf.gto import pp_int
        h += pp_int.get_gth_pp(mol)
    else:
        h+= mol.intor_symmetric('int1e_nuc_spinor')

    if len(mol._ecpbas) > 0:
        h += mol.intor_symmetric('ECPspinor')
    return h

# attempt to restructure the spinor hf code structure
# spinor hf should be parent of x2c hf using j-adapted spinor
class Spinor_SCF(hf.SCF):
    '''Nonrelativistic SCF under j-adapted spinor basis'''
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)

    def build(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.direct_scf:
            self.opt = self.init_direct_scf(mol)
        return self

    def dump_flags(self, verbose=None):
        hf.SCF.dump_flags(self, verbose)
        return self

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def _eigh(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        idx = numpy.argmax(abs(c.real), axis=0)
        c[:,c[idx,range(len(e))].real<0] *= -1
        return e, c

    @lib.with_doc(get_hcore.__doc__)
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_hcore(mol)
        

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return mol.intor_symmetric('int1e_ovlp_spinor')

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = mol.nelectron
        mo_occ[:nocc] = 1
        if nocc < len(mo_energy):
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])
            else:
                logger.info(self, 'nocc = %d  HOMO = %.12g  LUMO = %.12g',
                            nocc, mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(self, 'nocc = %d  HOMO = %.12g  no LUMO',
                        nocc, mo_energy[nocc-1])
        logger.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    make_rdm1 = lib.module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        if self.direct_scf and self.opt is None:
            self.opt = self.init_direct_scf(mol)
        dm_sph = spinor2sph(mol, dm)
        j_sph, k_sph = ghf.GHF.get_jk(self, mol, dm_sph)
        j_spinor = sph2spinor(mol, j_sph)
        k_spinor = sph2spinor(mol, k_sph)
        logger.timer(self, 'vj and vk', *t0)
        return j_spinor, k_spinor

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Dirac-Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last, copy=False) + vj - vk
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        return dhf.analyze(self, verbose)

    def newton(self):
        from pyscf.x2c.newton_ah import newton
        return newton(self)

    def stability(self, internal=None, external=None, verbose=None, return_status=False):
        '''
        X2C-HF/X2C-KS stability analysis.

        See also pyscf.scf.stability.rhf_stability function.

        Kwargs:
            return_status: bool
                Whether to return `stable_i` and `stable_e`

        Returns:
            If return_status is False (default), the return value includes
            two set of orbitals, which are more close to the stable condition.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.

            Else, another two boolean variables (indicating current status:
            stable or unstable) are returned.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.
        '''
        ''' Currently using GHF stability, and GHF stability is wrong '''
        from pyscf.x2c.stability import x2chf_stability
        return x2chf_stability(self, verbose, return_status)

    def nuc_grad_method(self):
        raise NotImplementedError
    
JHF = Spinor_SCF

if __name__ == '__main__':
    from pyscf import scf
    mol = mole.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [['Ne', (0, 0, 0)], ]
    mol.basis = 'ccpvdz'
    mol.build(0, 0)

##############
# SCF result
    method = Spinor_SCF(mol)
    #method.init_guess = '1e'
    energy = method.scf()
    print(method.mo_energy)
    print(energy)