from functools import reduce
import copy
import numpy
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf, dhf, ghf, _vhf
import re
from zquatev import solve_KR_FCSCE as eigkr

def symmetry_label(mol, symmetry=None):
    if symmetry is None:
        raise ValueError("Symmetry of orbital desired here.")
    if 'sph' not in symmetry and 'linear' not in symmetry:
        raise ValueError("Only spherical and linear symmetry supported for now.")
    else:
        labels = mol.spinor_labels()
        irrep = dict() 
        processed_labels = []
        for label in labels:
            label = label.split()
            match = re.search(r'[a-zA-Z](.*?),(.*?)$', label[2])
            if match:
                #print(match.group(0), match.group(1), match.group(2))
                full_label = match.group(0)
                j_val = match.group(1)
                jz_val = match.group(2)
            processed_labels.append([full_label, j_val, jz_val])
        if 'sph' in symmetry:
            print(f'Spherical symmetry assigned')
            for ilabel, label in enumerate(processed_labels):
                full_label = label[0]
                if irrep.get(full_label) is not None:
                    irrep[full_label].append(ilabel)
                else:
                    irrep[full_label] = [ilabel]
        elif 'linear' in symmetry:
            print(f'Linear symmetry assigned')
            for ilabel, label in enumerate(processed_labels):
                full_label=label[2]
                print(label)
                print("full_label",full_label)
                if irrep.get(full_label) is not None:
                    irrep[full_label].append(ilabel)
                else:
                    irrep[full_label] = [ilabel]
        return irrep

def eig(mf, h, s, irrep=None):
    '''Solve generalized eigenvalue problem, for each irrep.  The
    eigenvalues and eigenvectors are not sorted to ascending order.
    Instead, they are grouped based on irreps.
    '''
    print('symmetry adapted eigen value')
    mol = mf.mol

    if irrep is None:
        raise ValueError("An irrep is desired as input")
    cs = []
    es = []
    nmo = mf.mol.nao_2c()
    e = np.zeros(nmo)
    c = np.zeros((nmo, nmo), dtype=complex)
    irrep_tag = np.empty(nmo, dtype='U10')
    for ir in irrep:
        ir_idx = irrep[ir]
        hi = h[np.ix_(ir_idx, ir_idx)]
        si = s[np.ix_(ir_idx, ir_idx)]
        ei, ci = mf._eigh(h[np.ix_(ir_idx,ir_idx)], s[np.ix_(ir_idx,ir_idx)])
        c[np.ix_(ir_idx,ir_idx)] = ci
        e[ir_idx] = ei
        irrep_tag[ir_idx] = str(ir)
        angular_tag = None
    # process max contributing ao
    from scipy.linalg import sqrtm
    s_sqrtm = sqrtm(s)
    c_tilde = np.dot(s_sqrtm, c)
    labels = np.array(mol.spinor_labels())
    label_ang = np.array([re.search(r'[a-z]', label.split()[2]).group(0) for label in labels])
    angular_tag = np.empty(nmo, dtype='U1')
    for i in range(nmo):
        ci = abs(c_tilde[:,i])**2
        norm_ang = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        angs = np.array(['s','p', 'd', 'f', 'g'])
        for j, label in enumerate(angs):
            norm_ang[j] = sum(ci[np.where(label_ang == label)])
        angular_tag[i] = angs[np.argmax(norm_ang)]

    sort_idx = np.argsort(e)
    irrep_tag = irrep_tag[sort_idx]
    angular_tag = angular_tag[sort_idx]
    return lib.tag_array(e[sort_idx],irrep_tag=irrep_tag),\
           lib.tag_array(c[:,sort_idx], ang_tag=angular_tag)

def get_occ_symm(mf, irrep, occup, irrep_mo=None, mo_energy=None, mo_coeff=None):
    mol = mf.mol
    mo_occ = np.zeros_like(mo_energy, dtype=complex)
    ir_tag = irrep_mo
    if ir_tag is None:
        if mo_energy is None:
            mo_energy = mf.mo_energy
        ir_tag=mf.irrep_mo
    if mo_coeff is None and mf.mo_coeff is None:
        for ir in irrep:
            ir_mo = np.where(ir_tag==ir)
            if isinstance(occup[ir][0], int): 
                occupy = sum(occup[ir])
            else:
                occupy = occup[ir][0]
            mo_occ[ir_mo[:occupy]] = 1.0+0.j
            #mo_occ
            return mo_occ
    elif mo_coeff is None:
        mo_coeff = mf.mo_coeff
    angular_momentum = ['s', 'p', 'd', 'f']
    n_ang = 4
    for ir in irrep:
        ir_mo = np.where(ir_tag==ir)[0]
        if not ir in occup:
            continue
        occupy = occup[ir]
        if isinstance(occup[ir][0], int): 
            mo_occ[ir_mo[:occupy[0]]] = 1.0+0.j 
        else:
            mo_occ[ir_mo[occupy[0]]] = 1.0+0.j 
        for i in range(1, min(len(occupy), n_ang+1)):
            if occupy[i] == 0:
                continue
            else:
                ang = angular_momentum[i-1]
                indices = np.where(mo_coeff.ang_tag[ir_mo[occupy[0]:]] == ang)[0]
                if len(indices) < occupy[i]:
                    raise ValueError(f'Not enough mo with largest contribution from {ang} found')
                else:
                    mo_occ[ir_mo[occupy[0]:][indices[:occupy[i]]]] = 1.0+0.j
    occupied = np.where(mo_occ == 1.0+0.j)[0]
    count = [0,0,0,0]
    for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
        if ang == 's':
            count[0]+=1
    for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
        if ang == 'p':
            count[1]+=1
    for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
        if ang == 'd':
            count[2]+=1
    for idx, irrep, ene, ang in zip(occupied, ir_tag[occupied], mo_energy[occupied], mo_coeff.ang_tag[occupied]):
        if ang == 'f':
            count[3]+=1
    return mo_occ


def spinor2sph(mol, spinor):
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    #print(c2.shape)
    assert (spinor.shape[0] == c2.shape[1]), "spinor integral must be of shape (nao_2c, nao_2c)"
    ints_sph = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    return ints_sph

def sph2spinor(mol, sph):
    assert (sph.shape[-1] == sph.shape[-2] == mol.nao_nr() * 2), "spherical integral must be of shape (nao_2c, nao_2c)"
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    if len(sph.shape) == 3:
        ints_spinor = lib.einsum('ip,xpq,qj->xij', c2.T.conj(), sph, c2)
    elif len(sph.shape) == 2:
        ints_spinor = lib.einsum('ip,pq,qj->ij', c2.T.conj(), sph, c2)
    else:
        raise ValueError("spherical integral must be of shape (nao_nr, nao_nr) or (nao_nr, nao_nr, 3)")
    return ints_spinor

def _proj_dmll(mol_nr, dm_nr, mol):
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, numpy.eye(mol_nr.nao_nr()), mol)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm_ll = (dm_ll + dhf.time_reversal_matrix(mol, dm_ll)) * .5
    return dm_ll

'''
def get_jk(mol, dm, hermi=1, mf_opt=None, with_j=True, with_k=True, omega=None):
    def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
        if (not omega and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            return hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            return hf.SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
    dm_sph = spinor2sph(mol, dm)
    j_sph, k_sph = ghf.get_jk(mol, dm_sph, hermi=1, jkbuild=jkbuild)
    j_spinor = sph2spinor(mol, j_sph)
    k_spinor = sph2spinor(mol, k_sph)
    return j_spinor, k_spinor
'''

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
    n2c = dm.shape[0]
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

def get_hcore(mol, with_soc=None):
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
        ecp = mol.intor('ECPscalar')
        iden = numpy.eye(2)
        ints_sph = mol.intor('int1e_nuc_sph')
        ints_sph = numpy.einsum('ij,pq->ijpq', iden, ecp)

        c = mol.sph2spinor_coeff()

        ecp_spinor = numpy.einsum('ipa,ijpq,jqb->ab', numpy.conj(c), ints_sph,
                                  c)
        h += 1. * ecp_spinor
        if with_soc is True:
            mat_sph = mol.intor('ECPso')
            s = .5 * lib.PauliMatrices
            u = mol.sph2spinor_coeff()
            mat_spinor = numpy.einsum('sxy,spq,xpi,yqj->ij', s, mat_sph, u.conj(), u)
            h += -1.j*mat_spinor
    return h

# attempt to restructure the spinor hf code structure
# spinor hf should be parent of x2c hf using j-adapted spinor
class SpinorSCF(hf.SCF):
    '''Nonrelativistic SCF under j-adapted spinor basis'''

    _keys = {'with_soc', 'with_x2c'}
    
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.with_soc=False
        self.with_x2c=None
        self.so_contr=None
        self.new_energy_algo=False

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
        return eigkr(self.mol, h, s, debug=False)

    @lib.with_doc(get_hcore.__doc__)
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        if self.with_x2c is not None:
            hcore = self.with_x2c.get_hcore(mol)
        else:
            hcore = get_hcore(mol, self.with_soc)
        if getattr(mol, 'so_contr', None) is not None:
            hcore = reduce(numpy.dot, (mol.so_contr.T, hcore, mol.so_contr))
        return hcore
        
    def energy_tot(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm=self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None:
            vhf = self.get_veff()
        print('new_energy_algo', self.new_energy_algo, hasattr(self.with_x2c, 'soc_matrix'))
        if self.with_x2c is not None and hasattr(self.with_x2c, 'soc_matrix'):
            if self.new_energy_algo:
                return hf.energy_tot(self, dm, h1e - self.with_x2c.soc_matrix, vhf+self.with_x2c.soc_matrix)
            else:
                return hf.energy_tot(self, dm, h1e, vhf)
        else:
            return hf.energy_tot(self, dm, h1e, vhf)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        ovlp = mol.intor_symmetric('int1e_ovlp_spinor')
        if getattr(mol, 'so_contr', None) is not None:
            ovlp = reduce(numpy.dot, (mol.so_contr.T, ovlp, mol.so_contr))
        return ovlp

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
    

class SymmSpinorSCF(SpinorSCF):
    def __init__(self, mol, symmetry=None, occup=None):
        SpinorSCF.__init__(self, mol)
        if symmetry is 'linear' or symmetry is 'sph':
            self.irrep_ao = symmetry_label(mol, symmetry)
        else:
            raise NotImplementedError
        
        self.occupation = occup
        print('occup')
        print(self.occupation)
        
    def eig(self, h, s):
        e, c = eig(self, h, s, irrep=self.irrep_ao)
        self.irrep_mo = e.irrep_tag
        return e, c

    # when linear symmetry or spherical symmetry imposed,
    # kramers symmetry is adapted outside the eigensolver
    def _eigh(self, h, s):
        return scipy.linalg.eigh(h, s)
    
    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.occupation is None or mo_energy is None:
            return SpinorSCF.get_occ(self, mo_energy, mo_coeff)
        else:
            return get_occ_symm(self, self.irrep_ao, self.occupation, mo_energy=mo_energy, mo_coeff=mo_coeff)

JHF = SpinorSCF
SCF = SpinorSCF
SymmSCF = SymmJHF = SymmSpinorSCF

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
    method = SpinorSCF(mol)
    #method.init_guess = '1e'
    energy = method.scf()
    print(method.mo_energy)
    print(method.mo_coeff[:,0])
    print(method.mo_coeff[:,1])
    print(energy)
