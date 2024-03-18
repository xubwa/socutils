import scipy
import numpy
from pyscf import __config__
from pyscf.lib import logger

from . import zcasbase, zcasci
from .zmc_ao2mo import chunked_cholesky
from .zmc_superci import mcscf_superci

def expmat(a):
    return scipy.linalg.expm(a)

def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = casscf.view(zcasci.CASCI)
    mc.mo_coeff = mo

    if eris is None:
        return mc

    mc.get_h2eff = lambda *args: eris.aaaa
    return mc

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

    def uniq_var_indices(self, nmo, ncore, ncas, frozen, freeze_pair=None):
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
        #print(idx[:ncore,nocc:])
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
