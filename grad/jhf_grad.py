import pyscf.grad.rhf as rhf_grad
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.x2c.x2c import _block_diag
from pyscf.socutils import ghf_grad
import numpy
from pyscf.socutils.spinor_hf import spinor2sph, sph2spinor

def _block_diag_xyz(mat):
    '''
    [a b c]
    ->
    [[a 0] [b 0] [c 0]]
    [[0 a] [0 b] [0 c]]
    '''
    return numpy.asarray([_block_diag(mat_i) for mat_i in mat])

def get_ovlp(mol):
    return -mol.intor('int1e_ipovlp_spinor')


def hcore_generator(mf, mol=None):
    if mol is None: mol = mf.mol
    with_x2c = getattr(mf.base, 'with_x2c', None)
    if with_x2c:
        hcore_deriv = with_x2c.hcore_deriv_generator(deriv=1)
    else:
        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        aoslices = mol.aoslice_by_atom()
        h1 = mf.get_hcore(mol)
        c2 = numpy.vstack(mol.sph2spinor_coeff())
        def hcore_deriv(atm_id):
            shl0, shl1, p0, p1 = aoslices[atm_id]
            with mol.with_rinv_at_nucleus(atm_id):
                vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                vrinv *= -mol.atom_charge(atm_id)
                if with_ecp and atm_id in ecp_atoms:
                    vrinv += mol.intor('ECPscalar_iprinv', comp=3)
            vrinv[:,p0:p1] += h1[:,p0:p1]
            vrinv = _block_diag_xyz(vrinv + vrinv.transpose(0,2,1))
            vrinv = sph2spinor(mol, vrinv)
            print(vrinv.shape)
            return vrinv
    return hcore_deriv

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of j-adapted spinor HF/KS gradients

    Args:
        mf_grad : grad.jhf.Gradients or grad.jks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-GHF Coulomb repulsion')
    vhf_spinor = mf_grad.get_veff(mol, dm0)
    vhf = vhf_spinor
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_2c_by_atom()
    de = numpy.zeros((len(atmlst),3), dtype=complex)
    nao = mol.nao_2c()

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
# s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1])*2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1])*2

        de[k] += mf_grad.extra_force(ia, locals())
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
    return de.real

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    return rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

class Gradients(rhf_grad.GradientsMixin):
    '''Non-relativistic generalized Hartree-Fock gradients
    '''
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        dm_scalar = spinor2sph(mol, dm)
        veff_scalar = ghf_grad.get_veff(self, mol, dm_scalar)
        return sph2spinor(mol, veff_scalar)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    def hcore_generator(self, mol=None):
        if mol is None: mol = self.mol
        return hcore_generator(self, mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return get_ovlp(mol)

    grad_elec = grad_elec

Grad = Gradients

from pyscf import socutils
socutils.spinor_hf.Spinor_SCF.Gradients = lib.class_as_method(Gradients)
from pyscf import x2c
x2c.x2c.SCF.Gradients = lib.class_as_method(Gradients)