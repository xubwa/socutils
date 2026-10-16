#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

from functools import reduce
import copy
import numpy
import numpy as np
import scipy.linalg
from pyscf import df
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf, dhf, ghf, _vhf
import re
try:
    from socutils.lib import zquatev
except ImportError:
    zquatev = None
        
#from zquatev import solve_KR_FCSCE as eigkr


def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    if with_df is None:
        with_df = df.DF(mf.mol)
    with_df.max_memory = mf.max_memory
    with_df.auxbasis = auxbasis
    with_df.stdout = mf.stdout
    with_df.verbose = mf.verbose

    dfmf = _DFJHF(mf, with_df, only_dfj)
    dfmf.direct_scf = False   # the MO-based K path needs the full (tagged) DM
    return lib.set_class(dfmf, (_DFJHF, mf.__class__))


class _DFJHF(df.df_jk._DFHF):
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if mol is None: mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        t0 = (logger.process_clock(), logger.perf_counter())
        nao = mol.nao_nr()   # real spherical AO dimension (matches DF integrals)
        vj = vk = None
        mo_coeff = getattr(dm, 'mo_coeff', None)
        mo_occ   = getattr(dm, 'mo_occ',   None)
        # J and the MO-based K in one pass over the DF vectors (they are read
        # from disk for large systems): a single Hermitian density with MOs
        if (with_j and with_k and self.with_df and hermi and numpy.ndim(dm) == 2
                and mo_coeff is not None and mo_occ is not None):
            dm_sph   = spinor2sph(mol, dm)
            dm_trace = dm_sph[:nao, :nao] + dm_sph[nao:, nao:]
            # the imaginary part of a Hermitian D_aa + D_bb is antisymmetric
            # and does not contribute to J with symmetric (P|mu nu)
            vk, vj_ao = self._get_k_mo(mol, mo_coeff, mo_occ, omega, dm_j=dm_trace.real)
            vj_sph = numpy.zeros(dm_sph.shape, dtype=complex)
            vj_sph[:nao, :nao] = vj_sph[nao:, nao:] = vj_ao
            vj = sph2spinor(mol, vj_sph)
            logger.timer(self, 'vj and vk (one pass)', *t0)
            return vj, vk

        # --- J ---
        # J depends only on the total charge density dm_aa + dm_bb.
        # Pass real/imag parts of the trace DM directly to with_df.get_jk;
        # no need to go through ghf.get_jk's block splitting.
        if with_j:
            # ``dm`` may be a single (n2c, n2c) matrix or a batch
            # (nset, n2c, n2c) -- the TDDFT response feeds many trial densities
            # at once.  Slice the spin blocks with an ellipsis so the batch axis
            # (if any) is preserved.
            dm_sph   = spinor2sph(mol, dm)                                # (..., 2nao, 2nao)
            dm_trace = dm_sph[..., :nao, :nao] + dm_sph[..., nao:, nao:]  # (..., nao, nao)
            vj_ao, _ = self.with_df.get_jk(dm_trace.real, 1, True, False,
                                            self.direct_scf_tol, omega)
            if numpy.iscomplexobj(dm_trace):
                vj_i, _ = self.with_df.get_jk(dm_trace.imag, 1, True, False,
                                               self.direct_scf_tol, omega)
                vj_ao = vj_ao + 1j * vj_i
            vj_sph = numpy.zeros(dm_sph.shape, dtype=complex)
            vj_sph[..., :nao, :nao] = vj_sph[..., nao:, nao:] = vj_ao
            vj = sph2spinor(mol, vj_sph)

        # --- K ---
        if with_k and self.with_df:
            mo_coeff = getattr(dm, 'mo_coeff', None)
            mo_occ   = getattr(dm, 'mo_occ',   None)
            if mo_coeff is not None and mo_occ is not None:
                vk = self._get_k_mo(mol, mo_coeff, mo_occ, omega)
            else:
                # Fallback: DM-based K via ghf.get_jk (first iteration etc.)
                logger.warn(self, 'mo_coeff not tagged on dm, using DM-based K')
                if not with_j:
                    dm_sph = spinor2sph(mol, dm)

                def jkbuild(mol, dms, hermi, with_j, with_k, omega=None):
                    # Pass the whole batch of (real spherical) densities to the
                    # DF get_jk in one call instead of looping over them -- the
                    # TDDFT response hands us many trial densities at once.
                    dms = numpy.asarray(dms)
                    vj, vk = self.with_df.get_jk(dms.real, hermi, with_j, with_k,
                                                  self.direct_scf_tol, omega)
                    if numpy.iscomplexobj(dms):
                        vjI, vkI = self.with_df.get_jk(dms.imag, hermi,
                                                        with_j, with_k,
                                                        self.direct_scf_tol, omega)
                        if with_j and vj is not None: vj = vj + vjI * 1j
                        if with_k and vk is not None: vk = vk + vkI * 1j
                    return vj, vk

                _, k_sph = ghf.get_jk(mol, dm_sph, hermi, False, True, jkbuild, omega)
                vk = sph2spinor(mol, k_sph)

        logger.timer(self, 'vj and vk', *t0)
        return vj, vk

    def _get_k_mo(self, mol, mo_coeff, mo_occ, omega=None, dm_j=None):
        """Build K in spinor basis via MO half-transform of DF 3-index integrals.

        DF integrals (P|mu nu) live in the real spherical AO basis (nao = mol.nao_nr()).
        Spinor MOs are projected to the spherical GHF basis and split into
        alpha/beta spin blocks, each of dimension nao.

        Because the DF integrals are real but the spinor MOs are complex, we split
        each MO block into real and imaginary parts and call PySCF's real-only C
        half-transform (AO2MOnr_e2_drv) twice per block.  The four real
        half-transformed buffers

            LaR[Pi, mu] = sum_nu eri[P,mu,nu] Re(C_a[nu,i])
            LaI[Pi, mu] = sum_nu eri[P,mu,nu] Im(C_a[nu,i])

        are then combined with real BLAS calls:

            K_aa[m,n] = sum_{Pi} La[Pi,m] La*[Pi,n]
                      = LaR.T@LaR + LaI.T@LaI + i*(LaI.T@LaR - LaR.T@LaI)

        The C library reads the compressed-triangular eri directly (no unpack_tril).

        With dm_j (a real symmetric (nao, nao) density) the Coulomb matrix of
        dm_j is accumulated in the same pass and (vk, vj) is returned.
        """
        import ctypes
        from pyscf.ao2mo import _ao2mo

        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
        fmmm   = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
        fdrv   = _ao2mo.libao2mo.AO2MOnr_e2_drv
        null   = lib.c_null_ptr()

        # c2: (2*nao, n2c) unitary, projects spinor MOs -> spherical GHF basis
        c2  = numpy.vstack(mol.sph2spinor_coeff())
        nao = mol.nao_nr()
        n2c = mol.nao_2c()

        nocc_mask = mo_occ > 0
        nocc = int(nocc_mask.sum())
        if dm_j is not None:
            dj = dm_j + dm_j.T
            dj[numpy.diag_indices(nao)] *= .5
            dj_tril = lib.pack_tril(dj)
            vj_tril = numpy.zeros_like(dj_tril)
        if nocc == 0:
            vk0 = numpy.zeros((n2c, n2c), dtype=complex)
            if dm_j is None:
                return vk0
            for eri1 in self.with_df.loop():
                vj_tril += lib.dot(lib.dot(eri1, dj_tril[:, None]).T, eri1).ravel()
            return vk0, lib.unpack_tril(vj_tril)

        # Occupied spinor MOs weighted by sqrt(f_i), then projected to sph GHF basis
        C_occ     = mo_coeff[:, nocc_mask] * numpy.sqrt(mo_occ[nocc_mask])
        C_occ_sph = c2 @ C_occ   # (2*nao, nocc); first nao rows = alpha, last = beta
        # Kramers pairs (u, Theta u): only the unbarred half is transformed
        kramers = getattr(self, 'kramers_k', True) and _kramers_unbarred(C_occ_sph, nao)
        if kramers is not False:
            C_occ_sph = kramers
            nocc = C_occ_sph.shape[1]
            logger.debug(self, 'DF-K: Kramers-paired occupied spinors, %d unbarred', nocc)

        # Four real Fortran-order arrays for the C half-transform library
        # Each has shape (nao, nocc), matching the DF integral AO index
        C_aR = numpy.asfortranarray(C_occ_sph[:nao].real)
        C_aI = numpy.asfortranarray(C_occ_sph[:nao].imag)
        C_bR = numpy.asfortranarray(C_occ_sph[nao:].real)
        C_bI = numpy.asfortranarray(C_occ_sph[nao:].imag)

        vk_sph = numpy.zeros((2 * nao, 2 * nao), dtype=complex)

        max_memory = self.max_memory - lib.current_memory()[0]
        blockdim   = getattr(self.with_df, 'blockdim', 240)
        # 4 real buffers of (blksize*nocc, nao), 8 bytes/element
        blksize = max(4, int(min(blockdim,
                                 max_memory * 1e6 / 8 / (4 * nao * nocc))))
        logger.debug(self, 'DF-K MO half-transform: nocc=%d  blksize=%d', nocc, blksize)

        buf_aR = numpy.empty((blksize * nocc, nao))
        buf_aI = numpy.empty((blksize * nocc, nao))
        buf_bR = numpy.empty((blksize * nocc, nao))
        buf_bI = numpy.empty((blksize * nocc, nao))

        def _half(eri1, C_real, buf):
            """Half-transform compressed-triangular eri1 (naux, nao*(nao+1)/2)
            with real Fortran-order C_real (nao, nocc) -> buf (naux*nocc, nao)."""
            naux_ = eri1.shape[0]
            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 C_real.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_), ctypes.c_int(nao),
                 (ctypes.c_int * 4)(0, nocc, 0, nao),
                 null, ctypes.c_int(0))

        for eri1 in self.with_df.loop(blksize):
            if dm_j is not None:
                vj_tril += lib.dot(lib.dot(eri1, dj_tril[:, None]).T, eri1).ravel()
            naux = eri1.shape[0]
            LaR = buf_aR[:naux * nocc]
            LaI = buf_aI[:naux * nocc]
            LbR = buf_bR[:naux * nocc]
            LbI = buf_bI[:naux * nocc]

            _half(eri1, C_aR, LaR)
            _half(eri1, C_aI, LaI)
            _half(eri1, C_bR, LbR)
            _half(eri1, C_bI, LbI)

            # K_aa[m,n] = LaR.T@LaR + LaI.T@LaI + i*(LaI.T@LaR - LaR.T@LaI)
            IRa = lib.dot(LaI.T, LaR)
            vk_sph[:nao, :nao] += (lib.dot(LaR.T, LaR) + lib.dot(LaI.T, LaI)
                                   + 1j * (IRa - IRa.T))

            # K_bb[m,n] = LbR.T@LbR + LbI.T@LbI + i*(LbI.T@LbR - LbR.T@LbI)
            IRb = lib.dot(LbI.T, LbR)
            vk_sph[nao:, nao:] += (lib.dot(LbR.T, LbR) + lib.dot(LbI.T, LbI)
                                   + 1j * (IRb - IRb.T))

            # K_ab[m,n] = LaR.T@LbR + LaI.T@LbI + i*(LaI.T@LbR - LaR.T@LbI)
            K_ab = (lib.dot(LaR.T, LbR) + lib.dot(LaI.T, LbI)
                    + 1j * (lib.dot(LaI.T, LbR) - lib.dot(LaR.T, LbI)))
            vk_sph[:nao, nao:] += K_ab
            vk_sph[nao:, :nao] += K_ab.conj().T   # K_ba = K_ab†

        if kramers is not False:
            # unbarred sums A = K_aa, B = K_bb, C = K_ab; the barred partners
            # (Theta u: alpha = -u_b*, beta = u_a*, and real DF vectors) add
            # B* to K_aa and -C^T to K_ab
            A = vk_sph[:nao, :nao].copy()
            B = vk_sph[nao:, nao:].copy()
            C = vk_sph[:nao, nao:].copy()
            vk_sph[:nao, :nao] = A + B.conj()
            vk_sph[nao:, nao:] = B + A.conj()
            vk_sph[:nao, nao:] = C - C.T
            vk_sph[nao:, :nao] = vk_sph[:nao, nao:].conj().T

        if dm_j is not None:
            return sph2spinor(mol, vk_sph), lib.unpack_tril(vj_tril)
        return sph2spinor(mol, vk_sph)


def _kramers_unbarred(C, nao, tol=1e-8):
    '''Unbarred columns of an occupied set (sph GHF basis, (2 nao, nocc), columns
    weighted by sqrt(occupation)) made of Kramers pairs (u_k, b_k) in columns
    (2k, 2k+1) with b_k = Theta u_k up to a phase, Theta (a, b) = (-b*, a*),
    and equal occupations; False when the set is not paired that way.'''
    nocc = C.shape[1]
    if nocc == 0 or nocc % 2:
        return False
    u, b = C[:, 0::2], C[:, 1::2]
    tu = numpy.vstack((-u[nao:].conj(), u[:nao].conj()))
    nu = numpy.einsum('pk,pk->k', u.conj(), u).real
    nb = numpy.einsum('pk,pk->k', b.conj(), b).real
    ov = abs(numpy.einsum('pk,pk->k', b.conj(), tu))
    if (abs(nu - nb) > tol * nu.max()).any() or (abs(ov - numpy.sqrt(nu * nb)) > tol * nu.max()).any():
        return False
    return numpy.ascontiguousarray(u)


def spinor2sph(mol, spinor):
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    assert (spinor.shape[-2] == c2.shape[-1]), "spinor integral must be of shape (nao_2c, nao_2c)"
    if len(spinor.shape) == 3:
        ints_sph = lib.einsum('ip,xpq,qj->xij', c2, spinor, c2.T.conj())
    elif len(spinor.shape) == 2:
        ints_sph = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    else:
        raise ValueError("spherical integral must be of shape (nao_nr, nao_nr) or (nao_nr, nao_nr, 3)")
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

class SCF(hf.SCF):
    '''SCF in the j-adapted spinor basis (two-component).

    Without a relativistic Hamiltonian this is the nonrelativistic spinor SCF;
    .x2camf() / .x2cmp() attach the X2C spin-orbit Hamiltonian (with_x2c).
    The base class is Kramers unrestricted; :class:`RHF` is the
    Kramers-restricted variant.
    '''

    _keys = {'with_soc', 'with_x2c'}

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.with_soc=False
        self.with_x2c=None

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        return density_fit(self, auxbasis, with_df, only_dfj)

    def cholesky(self, tau=1e-4, sigma=1e-2, method='threeloop',
                 cderi=None, cderi_to_save=None, with_df=None, only_dfj=False):
        from socutils.cd.cd import cholesky
        return cholesky(self, tau=tau, sigma=sigma, method=method,
                        cderi=cderi, cderi_to_save=cderi_to_save,
                        with_df=with_df, only_dfj=only_dfj)

    def _attach_x2cmp(self, flavor, with_gaunt, with_breit, with_pcc, with_aoc):
        from socutils.somf.x2cmp import SpinorX2CMPHelper
        self.with_x2c = SpinorX2CMPHelper(
            self.mol, x2cmp=flavor, with_gaunt=with_gaunt, with_breit=with_breit,
            with_pcc=with_pcc, with_aoc=with_aoc)
        self._keys = self._keys.union(['with_x2c'])
        terms = ['Dirac-Coulomb']
        if with_gaunt: terms.append('Gaunt')
        if with_breit: terms.append('Breit')
        if with_pcc: terms.append('PCC')
        logger.info(self, 'X2C SOC Hamiltonian: flavor=%s, terms = %s',
                    flavor, ' + '.join(terms))
        return self

    def x2camf(self, flavor='x2camf', with_gaunt=True, with_breit=True,
               with_pcc=False, with_aoc=False):
        '''Attach an X2CAMF (atomic-mean-field) spin-orbit Hamiltonian to this
        spinor mean-field object, in place, and return it -- the spinor
        analogue of PySCF's ``mf.x2c()``.  The SOC enters through
        ``self.with_x2c`` and is applied in :meth:`get_hcore`; the
        Kramers-restriction of the solve is a property of the driver (use
        :class:`RHF` for a Kramers-restricted run).

        >>> mf = RHF(mol).x2camf()              # Gaunt + Breit
        >>> mf.kernel()
        '''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def x2cmp(self, flavor='x2cmp', with_gaunt=True, with_breit=True,
              with_pcc=False, with_aoc=False):
        '''Attach an X2C molecular-mean-field (molecular picture-change) spin-
        orbit Hamiltonian to this spinor mean-field object, in place, and
        return it.  Same usage as :meth:`x2camf`, with the molecular-mean-field
        flavor.

        >>> mf = RHF(mol).x2cmp()
        >>> mf.kernel()
        '''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def gen_response(self, *args, **kwargs):
        '''Linear-response (J - c_x K + f_xc) function for TDDFT/stability.

        The spinor reference is a GHF/GKS reference in a complex n2c basis, so
        the GHF response template applies directly (J/K from the spinor get_jk,
        f_xc from the 2-component numint).'''
        from pyscf.scf._response_functions import _gen_ghf_response
        if isinstance(self, hf.KohnShamDFT) and self.do_nlc():
            raise NotImplementedError('NLC response is not implemented for the '
                                      'spinor (j-adapted) basis')
        return _gen_ghf_response(self, *args, **kwargs)

    def TDA(self):
        from socutils.tdscf import gks
        return gks.TDA(self)

    def eig(self, h, s=None, *args, **kwargs):
        return hf.SCF.eig(self, h, s)

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

    #def _eigh(self, h, s):
    #    return eigkr(self.mol, h, s, debug=False)

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
            ddm = numpy.array(dm) - numpy.array(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last) + vj - vk
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

    def spin_square(self, mo_coeff=None, s=None):
        from pyscf.scf.ghf import spin_square as spin_square_ghf
        if mo_coeff is None: mo_coeff = self.mo_coeff[:,self.mo_occ>0]
        if s is None: s = self.get_ovlp()
        mo_coeff_ghf = np.dot(np.vstack(self.mol.sph2spinor_coeff()), mo_coeff)
        s_ghf = spinor2sph(self.mol, s)
        return spin_square_ghf(mo_coeff_ghf, s_ghf)

    def nuc_grad_method(self):
        raise NotImplementedError


class UHF(SCF):
    '''Kramers-unrestricted spinor SCF.'''

    def to_ks(self, xc='HF'):
        '''The spinor Kohn-Sham object of the same flavor (same with_x2c).'''
        from socutils.dft import dft
        mf = self.view(dft.UKS)
        mf.xc = xc
        mf.converged = False
        return mf


class RHF(SCF):
    '''Kramers-restricted spinor SCF (quaternion eigensolver zquatev).'''

    def __init__(self, mol):
        super().__init__(mol)
        if zquatev is None:
            raise RuntimeError('zquatev library is required to perform Kramers-restricted spinor SCF')
    def _eigh(self, h, s):
        return zquatev.solve_KR_FCSCE(self.mol, h, s)
    def eig(self, h, s=None, *args, **kwargs):
        if s is not None:
            return self._eigh(h, s)
        else:
            n = h.shape[0]
            if n % 2 != 0:
                return scipy.linalg.eigh(h)
            # Reorder from interleaved [u0,b0,u1,b1,...] to block [u0,u1,...,b0,b1,...]
            idx = numpy.hstack([numpy.arange(0, n, 2), numpy.arange(1, n, 2)])
            h_kr = h[numpy.ix_(idx, idx)]
            try:
                # the check compares Frobenius norms, which grow ~ eps*n for
                # per-element noise eps: scale the threshold with the block
                # dimension or large Kramers-clean blocks get rejected
                zquatev.check_kramers_structure(h_kr, thresh=1e-6 * max(1, n // 2))
            except AssertionError:
                logger.warn(self, 'Kramers structure check failed in eig, falling back to scipy.linalg.eigh')
                return scipy.linalg.eigh(h)
            e, v_kr = zquatev.eigh(h_kr, iop=1)
            # v_kr columns are in interleaved order (iop=1)
            # but rows are in block order, need to map back to interleaved
            v = numpy.zeros_like(v_kr)
            v[idx] = v_kr
            return e, v
    def to_ks(self, xc='HF'):
        '''The Kramers-restricted spinor Kohn-Sham object (same with_x2c).'''
        from socutils.dft import dft
        mf = self.view(dft.RKS)
        mf.xc = xc
        mf.converged = False
        return mf


# old names
SpinorSCF = JHF = SCF
KRHF = RHF


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
