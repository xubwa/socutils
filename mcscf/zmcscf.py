import scipy
import numpy
from functools import reduce
from pyscf import __config__
from pyscf.lib import logger

from pyscf import lib
from socutils.scf import spinor_hf
from socutils.scf import double_group_symm
from socutils.mcscf import zcasbase, zcasci
# mcscf_superci is imported lazily inside superci() to avoid a circular import
# (zmc_superci imports zmcscf).
try:
    from socutils.lib import zquatev
except ImportError:
    zquatev = None

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

def eig_sub(mc, h, idx):
    '''Eigenvalues and eigenvectors of the matrix h given in the orbitals idx
    of mc.mo_coeff: within the irreps of the orbitals when they are known
    (Kramers pairs kept, see double_group_symm.eig_mo), otherwise with the
    eig of the SCF object.'''
    orbsym = mc._rot_irrep() if hasattr(mc, '_rot_irrep') else None
    eigh = lambda m: mc._scf.eig(m, None)
    if orbsym is None:
        return eigh(h)
    return double_group_symm.eig_mo(h, numpy.asarray(orbsym)[idx], eigh)

def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = casscf.view(zcasci.CASCI)
    mc.mo_coeff = mo
    # the irreps of the orbitals were set up by the orbital optimizer
    mc._orbsym_fixed = True

    if eris is None:
        return mc

    mc.get_h2eff = lambda *args: eris.aaaa

    # Precompute core JK from eris to avoid redundant get_jk in CASCI kernel
    if hasattr(eris, 'get_jk'):
        ncore = casscf.ncore
        ncas = casscf.ncas
        nocc = ncore + ncas
        mo_core = mo[:, :ncore]
        mo_cas = mo[:, ncore:nocc]
        core_occ = numpy.zeros(mo.shape[1])
        core_occ[:ncore] = 1
        dm_core_ao = numpy.dot(mo_core, mo_core.T.conj())
        vj_c, vk_c = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
        corevhf = vj_c - vk_c
        hcore = casscf.get_hcore()
        h1eff = reduce(numpy.dot, (mo_cas.T.conj(), hcore + corevhf, mo_cas))
        energy_core = casscf.energy_nuc()
        energy_core += numpy.einsum('ij,ji', dm_core_ao, hcore)
        energy_core += numpy.einsum('ij,ji', dm_core_ao, corevhf) * 0.5
        mc.get_h1eff = lambda *args, h1=h1eff, ec=energy_core: (h1, ec)

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
    # casdm1[p,q] = <p^dagger q>: the AO density takes its transpose
    dm = dm_core + reduce(numpy.dot, (mocas, casdm1.T, mocas.conj().T))
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
    

    def _diag_subfock_(idx):
        if idx.size > 1:
            c = mo_coeff1[:,idx]
            fock = reduce(numpy.dot, (c.conj().T, fock_ao, c))
            w, c = eig(fock)
            print(mo_energy[idx])
            print(w)
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
    print(act_idx)
    print(core_idx)
    c = mo_coeff1[:, core_idx]
    fock = reduce(numpy.dot, (c.conj().T, fock_ao, c))
    #print(fock)
    if mc._rot_irrep() is not None:
        w, c = eig_sub(mc, fock, core_idx)
    else:
        w, c = zquatev.eigh(fock)
    #print(w, mo_energy[core_idx])
    #print(c)
    #mo_coeff1[:, core_idx] = mo_coeff1[:, core_idx].dot(c)
    
    c = mo_coeff1[:, act_idx]
    fock = reduce(numpy.dot, (c.conj().T, fock_ao, c))
    if mc._rot_irrep() is not None:
        w, c = eig_sub(mc, fock, act_idx)
    else:
        w, c = zquatev.eigh(fock)
    #print(fock)
    #print(w)
    #print(c)
    mo_coeff1[:, act_idx] = mo_coeff1[:, act_idx].dot(c)
    #c = mo_coeff1[:, act_idx]
    #print(reduce(numpy.dot, (c.T.conj(), fock_ao, c)))
    #_diag_subfock_(core_idx)
    #_diag_subfock_(vir_idx)
    #_diag_subfock_(act_idx)

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
        # Not needed by the super-CI driver any more (its metric, Hessian and
        # preconditioner are basis covariant) and both rotate the orbital basis
        # between iterations, which discards the L-BFGS history and, through
        # the plain-eigh fallback, can scramble the Kramers pairing.
        self.canonicalize_ = False
        self.natorb = False

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

        orbsym = self._rot_irrep()
        if orbsym is not None:
            mask &= orbsym[:, None] == orbsym[None, :]
        return mask

    def _rot_irrep(self):
        '''Labels that restrict the orbital rotations: the legacy .irrep if
        set, else .orbsym (double-group irreps), else None.'''
        if self.irrep is not None:
            return numpy.asarray(self.irrep)
        return self._orbsym

    def screen_irrep(self, mat):
        '''Zero the elements of mat that couple different irreps (in place).'''
        orbsym = self._rot_irrep()
        if orbsym is not None:
            mat[orbsym[:, None] != orbsym[None, :]] = 0.
        return mat

    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo), dtype=complex)
        mat[idx] = v
        return mat - mat.T.conj()

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return numpy.dot(u0, expmat(dr))

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        mci = self.view(zcasci.CASCI)
        return mci.kernel(mo_coeff, ci0=ci0, verbose=verbose)

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        '''Optimize the CASSCF orbitals and CI vector.

        This drives the super-CI orbital optimizer directly (see
        :meth:`superci`); it is the CASSCF entry point analogous to PySCF's
        ``mc.kernel()``.
        '''
        return self.superci(mo_coeff, ci0=ci0, callback=callback)

    def superci(self, mo_coeff=None, ci0=None, callback=None, _kern=None):
        '''Super-CI CASSCF orbital optimization.

        Returns:
            Five elements -- total energy, active-space CI energy, the
            active-space FCI coefficients, and the MCSCF canonical orbital
            coefficients and orbital energies.  They are also stored as the
            attributes ``.e_tot``, ``.e_cas``, ``.ci``, ``.mo_coeff`` and
            ``.mo_energy``.
        '''
        # Lazy import to break the zmcscf <-> zmc_superci circular import.
        from socutils.mcscf.zmc_superci import mcscf_superci
        if _kern is None:
            _kern = mcscf_superci
        if zquatev is None:
            raise RuntimeError('zquatev library is required for spinor CASSCF '
                               'orbital optimization')
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:  # overwrite self.mo_coeff; it is used by many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback

        self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff, max_stepsize=self.max_stepsize,
                      conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      verbose=self.verbose, cderi=self._cderi)
        logger.note(self, 'CASSCF energy = %#.15g', self.e_tot)
        if self._orbsym is not None:
            self.check_orbsym(self.mo_coeff)
            self.mo_coeff = lib.tag_array(numpy.asarray(self.mo_coeff), orbsym=self._orbsym)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
