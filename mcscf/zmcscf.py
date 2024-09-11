import scipy
import numpy
from functools import reduce
from pyscf import __config__
from pyscf.lib import logger

from socutils.scf import spinor_hf
from . import zcasbase, zcasci
from .zmc_ao2mo import chunked_cholesky
from .zmc_superci import mcscf_superci

def eig(h, irrep=None):
    if irrep is None:
        e, c = scipy.linalg.eigh(h)
    else:
        ir_set = numpy.unique(irrep)
        e = numpy.zeros(h.shape[1])
        c = numpy.zeros(h.shape, dtype=complex)
        for ir in ir_set:
            ir_idx = numpy.where(irrep == ir)[0]
            print(ir_idx)
            hi = h[numpy.ix_(ir_idx, ir_idx)]
            ei, ci = scipy.linalg.eigh(hi)
            e[ir_idx] = ei
            c[numpy.ix_(ir_idx,ir_idx)] = ci
    return e, c

def expmat(a):
    return scipy.linalg.expm(a)

def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = casscf.view(zcasci.CASCI)
    mc.mo_coeff = mo

    if eris is None:
        return mc

    mc.get_h2eff = lambda *args: eris.aaaa
    return mc

def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
    if ci is None: ci = mc.ci
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    dm_core = numpy.dot(mo_coeff[:,:ncore], mo_coeff[:,:ncore].conj().T)
    mocas = mo_coeff[:,ncore:nocc]
    dm = dm_core + reduce(numpy.dot, (mocas, casdm1, mocas.conj().T))
    vj, vk = mc._scf.get_jk(mc.mol, dm)
    fock = mc.get_hcore() + vj - vk
    return fock

def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,     
                 cas_natorb=False, casdm1=None, verbose=logger.NOTE):
    log = logger.new_logger(mc, verbose)
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)
    
    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo_coeff.shape[1]
    fock_ao = mc.get_fock(mo_coeff, ci, eris, casdm1, verbose)

    if cas_natorb:
        raise NotImplementedError
    else:
        mo_coeff1 = mo_coeff.copy()
        log.info('Density matrix diagonal elements %s', casdm1.diagonal())

    mo_energy = numpy.einsum('pi,pi->i', mo_coeff.conj(), fock_ao.dot(mo_coeff1))

    irs = numpy.unique(mc.irrep)

    def _diag_subfock_(idx):
        if idx.size > 1:
            c = mo_coeff1[:,idx]
            fock = reduce(numpy.dot, (c.conj().T, fock_ao, c))
            ovlp = numpy.eye(idx.size, dtype=complex)
            w, c = eig(fock, irrep=mc.irrep[idx])

            mo_coeff1[:,idx] = mo_coeff1[:,idx].dot(c)
            mo_energy[idx] = w

    mask = numpy.ones(nmo, dtype=bool)
    frozen = getattr(mc, 'frozen', None)
    #if frozen is not None:
    
    #    if isinstance(frozen, (int, numpy.integer)):
    #        mask[:frozen] = False
    #    else:
    #        mask[frozen] = False
    core_idx = numpy.where(mask[:ncore])[0]
    vir_idx = numpy.where(mask[nocc:])[0] + nocc
    act_idx = numpy.where(mask[ncore:nocc])[0] + ncore
    #_diag_subfock_(core_idx)
    #_diag_subfock_(vir_idx)
    _diag_subfock_(act_idx)

    return mo_coeff1, ci, mo_energy


class CASSCF(zcasci.CASCI):

    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 20)
    irrep=None
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None, cholesky=True):
        zcasbase.CASBase.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen
        self.callback = None
        self._cderi = None
        self.max_stepsize = 0.2
        self.conv_tol = 1e-8
        self.conv_tol_grad = None
        self.freeze_pair = None

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
        return get_fock(self, mo_coeff, ci, eris, casdm1, verbose)

    canonicalize = canonicalize

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        # if self.internal_rotation:
        #     mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        # if self.extrasym is not None:
        #     extrasym = numpy.asarray(self.extrasym)
        #     # Allow rotation only if extra symmetry labels are the same
        #     extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
        #     mask = mask * extrasym_allowed
        if self.freeze_pair is not None:
            freeze_pair = self.freeze_pair
            set_i = freeze_pair[0]
            set_j = freeze_pair[1]
            for i in set_i:
                for j in set_j:
                    mask[i,j] = False
                    mask[j,i] = False
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        
        if self.irrep is not None: 
            irrep = self.irrep
            for i, iri in enumerate(irrep):
                for j, irj in enumerate(irrep):
                    if iri != irj:
                        mask[i,j] = False
        return mask

    def screen_irrep(self, mat):
        if self.irrep is not None:
            irrep = self.irrep
            for i, iri in enumerate(irrep):
                for j, irj in enumerate(irrep):
                    if iri != irj:
                        mat[i, j] = 0.
        return mat


    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        #print(idx[nocc:, :ncore])
        #print(idx[:ncore,fnocc:])
        return self.screen_irrep(mat)[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo), dtype=complex)
        mat[idx] = v
        all_indices = numpy.arange(nmo)
        if self.irrep is not None:
            irrep = self.irrep
            for i, iri in enumerate(irrep):
                for j, irj in enumerate(irrep):
                    if iri != irj:
                        mat[i, j] = 0.
        #frozen_pair = [[0,1,4,5,6,7,8,9],[10,11,12,13,14,15,16,17]]
        #set_i = frozen_pair[0]
        #set_j = frozen_pair[1]
        #for i in set_i:
        #    for j in set_j:
        #        mat[i,j] = 0.
        #        mat[j,i] = 0.
        
        return mat - mat.T.conj()

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return numpy.dot(u0, expmat(dr))

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        mci = self.view(zcasci.CASCI)
        return mci.kernel(mo_coeff, ci0=ci0, verbose=verbose)

    def superci(self, mo_coeff=None, ci0=None, callback=None, _kern=mcscf_superci):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else: # overwrite self.mo_coeff because it is needed in many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff, max_stepsize=self.max_stepsize,
                      conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad, verbose=self.verbose, cderi=self._cderi)
        logger.note(self, 'CASSCF energy = %#.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
