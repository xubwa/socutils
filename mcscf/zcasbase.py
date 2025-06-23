import numpy
import tempfile
from functools import reduce
from pyscf import fci, mcscf, lib, scf, gto
from pyscf.lib import logger
from pyscf import __config__


class CASBase(lib.StreamObject):

    natorb = getattr(__config__, 'mcscf_casci_CASCI_natorb', False)
    canonicalization = getattr(__config__, 'mcscf_casci_CASCI_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'mcscf_casci_CASCI_sorting_mo_energy', False)

    _keys = {
        'natorb', 'canonicalization', 'sorting_mo_energy', 'mol', 'max_memory',
        'ncas', 'nelecas', 'ncore', 'fcisolver', 'frozen', 'extrasym',
        'e_tot', 'e_cas', 'ci', 'mo_coeff', 'mo_energy', 'mo_occ', 'converged',
    }

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None):
        if isinstance(mf_or_mol, gto.Mole):
            mf = scf.GHF(mf_or_mol)
        else:
            mf = mf_or_mol

        mol = mf.mol
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas

        self._chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self.chkfile = self._chkfile.name

        if not isinstance(nelecas, (int, numpy.integer)):
            self.nelecas = sum(nelecas)
        else:
            self.nelecas = nelecas

        self.ncore = ncore
        self.fcisolver = fci.fci_dhf_slow.FCISolver(mol)
        # CI solver parameters are set in fcisolver object
        self.fcisolver.lindep = getattr(__config__, 'mcscf_casci_CASCI_fcisolver_lindep', 1e-12)
        self.fcisolver.max_cycle = getattr(__config__, 'mcscf_casci_CASCI_fcisolver_max_cycle', 200)
        self.fcisolver.conv_tol = getattr(__config__, 'mcscf_casci_CASCI_fcisolver_conv_tol', 1e-8)
        self.fcisolver.spin_square=None
        self.frozen = None

        ##################################################
        # don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.mo_occ = None
        self.converged = False

    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.mol.nelectron - self.nelecas
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            return ncorelec
        else:
            return self._ncore

    @ncore.setter
    def ncore(self, x):
        assert x is None or isinstance(x, (int, numpy.integer))
        assert x is None or x >= 0
        self._ncore = x

    def dump_flags(self, verbose=None):
        from pyscf.lib import logger
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** CASCI flags ********')
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de, %do), ncore = %d, nvir = %d', self.nelecas, ncas, ncore, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('max_memory %d (MB)', self.max_memory)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(log.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')

    def energy_nuc(self):
        return self._scf.energy_nuc()

    def get_hcore(self, mol=None):
        return self._scf.get_hcore(mol)

    @lib.with_doc(scf.hf.get_jk.__doc__)
    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self._scf.get_jk(mol, dm, hermi,
                                with_j=with_j, with_k=with_k, omega=omega)

    @lib.with_doc(scf.hf.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = numpy.dot(mocore, mocore.conj().T)
# don't call self._scf.get_veff because _scf might be DFT object
        vj, vk = self.get_jk(mol, dm, hermi)
        return vj - vk

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)

    def get_h2cas(self, mo_coeff=None):
        '''An alias of get_h2eff method'''
        return self.get_h2eff(mo_coeff)

    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        raise NotImplementedError

    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        '''An alias of get_h1eff method'''
        return self.get_h1eff(mo_coeff, ncas, ncore)

    get_h1eff = h1e_for_cas = get_h1cas 

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        raise NotImplementedError

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
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
        raise NotImplementedError

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        if isinstance(self.e_cas, (float, numpy.number)):
            log.note('CASCI E = %#.15g  E(CI) = %#.15g', self.e_tot, self.e_cas)
        else:
            for i, e in enumerate(self.e_cas):
                log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g',
                         i, self.e_tot[i], e)
        return self

    #@lib.with_doc(cas_natorb.__doc__)
    #def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False,
    #               casdm1=None, verbose=None, with_meta_lowdin=WITH_META_LOWDIN):
    #    return cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose,
    #                      with_meta_lowdin)


    #@lib.with_doc(cas_natorb.__doc__)
    #def cas_natorb_(self, mo_coeff=None, ci=None, eris=None, sort=False,
    #                casdm1=None, verbose=None, with_meta_lowdin=WITH_META_LOWDIN):
    #    self.mo_coeff, self.ci, self.mo_occ = cas_natorb(self, mo_coeff, ci, eris,
    #                                                     sort, casdm1, verbose)
    #    return self.mo_coeff, self.ci, self.mo_occ

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None,
                 verbose=None):
        return get_fock(self, mo_coeff, ci, eris, casdm1, verbose)

    #canonicalize = canonicalize

    # @lib.with_doc(canonicalize.__doc__)
    # def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
    #                   cas_natorb=False, casdm1=None, verbose=None,
    #                   with_meta_lowdin=WITH_META_LOWDIN):
    #     self.mo_coeff, ci, self.mo_energy = \
    #             canonicalize(self, mo_coeff, ci, eris,
    #                          sort, cas_natorb, casdm1, verbose, with_meta_lowdin)
    #     if cas_natorb:  # When active space is changed, the ci solution needs to be updated
    #         self.ci = ci
    #     return self.mo_coeff, ci, self.mo_energy
    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        '''One-particle density matrix in AO representation
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1 = self.fcisolver.make_rdm1(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1 = numpy.dot(mocore, mocore.conj().T)
        dm1 = dm1 + reduce(numpy.dot, (mocas, casdm1, mocas.conj().T))
        return dm1

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mcscf import df
        return df.density_fit(self, auxbasis, with_df)
