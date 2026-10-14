#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Range-separated Gaussian density fitting at the Gamma point, for
# all-electron cells with compact (core) functions.
#
# The Coulomb kernel with its G = 0 component removed (pyscf's convention,
# exxdiv=None) is split as
#     v'(r) = erfc(w r)/r                         short range, real space
#           + erf(w r)/r restricted to G != 0     long range, reciprocal space
#           - pi/(w^2 V)                          the G = 0 part of the SR term
# and every two- and three-centre integral of the fit is evaluated with it:
#   * SR: libcint integrals over explicitly enumerated lattice images with a
#     per-shell-triple bound (lib/pbcdf/sr_int.c), no lattice-sum estimators;
#   * LR: analytic Fourier transforms of AO pairs (overlap-screened) and of the
#     auxiliary functions, exactly, up to |G| where exp(-G^2/4w^2) is negligible.
# The fit uses the metric of the same kernel, eigendecomposed with a relative
# threshold; the metric must come out positive semidefinite, which is checked.
#
# Gaussian nuclei go through the same kernel (as charge-normalised s shells),
# so electron-electron, electron-nucleus and nucleus-nucleus terms share one
# convention and the total Coulomb energy of the neutral cell is the tin-foil
# Ewald energy.
#
# Output: Gamma-point (real) Cholesky-like vectors L[P, mu, nu] with
#     (mu nu | lam sig)' ~= sum_P L[P, mu, nu] L[P, lam, sig],
# exposed through loop() in pyscf's packed layout, so socutils.mcscf._CDERIS
# and other molecular DF consumers can use the object as with_df.
#
# lr='gspace': only the SR part is fitted (metric and 3c of the erfc kernel,
# local); the LR part is not fitted at all but appended as its exact
# plane-wave factorisation, one real row per cos/sin of each +-G pair,
#     L_G[mu nu] = sqrt(v_G / V) Re/Im rho_mu nu(G),   G != 0,
# and the G = 0 part of the SR kernel, -c0 S_mn S_ls (c0 = pi / (w^2 V)), is
# kept out of the vectors.  The vectors then carry
#     (mn|ls)' = (mn|ls)_vectors - c0 S_mn S_ls,
# get_jk(with_g0=True) adds the c0 term (J: -c0 tr(DS) S, K: -c0 S D S); in an
# orthonormal MO basis it is -c0 delta_pq delta_rs, i.e. the constant
# -c0 N (N - 1) / 2 for any N-electron wavefunction (see g0_constant and
# socutils.pbc.spinor_scf).  The SR vectors are what a local (sparse) fit
# replaces later; the LR rows are exact (no fitting error) and their number
# ~ V (2 w)^3 shrinks with w.
#

import ctypes
import numpy as np
import scipy.linalg
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.df import addons as df_addons
from pyscf.gto.ft_ao import ft_aopair, ft_ao
from socutils.pbc.coulomb_reference import lattice_translations, reciprocal_vectors, cell_volume

_lib = None


def _libpbcdf():
    global _lib
    if _lib is None:
        from socutils.lib import load_library
        _lib = load_library('libsocpbcdf', 'SOCUTILS_PBCDF_LIBRARY')
    return _lib


def _mole_at(mol, shifts, basis=None):
    '''Molecule of the atoms of mol repeated at every shift (shift-major order).'''
    m = gto.Mole()
    m.atom = [(mol.atom_symbol(ia), mol.atom_coord(ia) + t) for t in shifts for ia in range(mol.natm)]
    m.unit = 'Bohr'
    m.basis = mol._basis if basis is None else basis
    m.cart = mol.cart
    m.verbose = 0
    m.spin = sum(gto.charge(a) for a, _ in m.atom) % 2
    m.build(False, False)
    return m


def _shell_params(mol):
    '''Most diffuse exponent and largest |normalised coefficient| of each shell.'''
    amin = np.empty(mol.nbas)
    cmax = np.empty(mol.nbas)
    for i in range(mol.nbas):
        amin[i] = mol.bas_exp(i).min()
        cmax[i] = abs(mol._libcint_ctr_coeff(i)).max()
    return amin, cmax


_ANG_CACHE = {}


def _angular_amplitude(lmax):
    '''ang[l] = max over directions of |chi| / (c r^l exp(-a r^2)) for a libcint
    spherical function of momentum l with coefficient c (env value).  Measured
    numerically so that libcint's normalisation conventions are absorbed; 5%
    safety margin for the finite direction sample.'''
    if lmax in _ANG_CACHE:
        return _ANG_CACHE[lmax]
    rng = np.random.default_rng(0)
    u = rng.normal(size=(20000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    ang = np.empty(lmax + 1)
    for l in range(lmax + 1):
        m = gto.M(atom='X 0 0 0', basis={'X': [[l, [1., 1.]]]}, unit='Bohr', verbose=0)
        c = m._env[m._bas[0, gto.PTR_COEFF]]
        ang[l] = 1.05 * abs(m.eval_gto('GTOval_sph', u)).max() / (c * np.exp(-1.))
    _ANG_CACHE[lmax] = ang
    return ang


def _shell_envelope(mol, ang, t=.5):
    '''Envelope of each shell: |chi| <= Q (q/pi)^1.5 exp(-q r^2) for every function.

    r^l exp(-a r^2) <= (l/(2(1-t)a e))^(l/2) exp(-t a r^2) for 0 < t < 1, so
    primitive k contributes the charge |c_k| ang[l] (l/(2(1-t)a_k e))^(l/2)
    (pi/a'_k)^1.5 with a' = a (l = 0) or t a (l > 0).  Returns the smallest envelope exponent q and the summed charge Q
    (bounding a contracted shell by its most diffuse exponent).'''
    qmin = np.empty(mol.nbas)
    charge = np.empty(mol.nbas)
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        a = mol.bas_exp(i)
        c = abs(mol._libcint_ctr_coeff(i)).max(axis=1)
        if l == 0:
            ae, fac = a, ang[0] * np.ones_like(a)
        else:
            ae, fac = t * a, ang[l] * (l / (2 * (1 - t) * a * np.e))**(.5 * l)
        qmin[i] = ae.min()
        charge[i] = (c * fac * (np.pi / ae)**1.5).sum()
    return qmin, charge


def _aux_coulomb_diag(auxmol):
    '''Smallest (P|P) of each aux shell.'''
    out = np.empty(auxmol.nbas)
    for i in range(auxmol.nbas):
        d = np.diag(auxmol.intor('int2c2e', shls_slice=(i, i + 1, i, i + 1)))
        out[i] = d.min()
    return out


def _gamma_ft_kernel(cell, precision):
    '''Analytic FT of lattice-summed AO pairs at Gamma, packed lower triangle
    (s2), from pyscf's C lattice sum (pbc.df.ft_ao): rows of shells [i0, i1),
    called as kern(Gv, kptjs=zeros((1, 3)), shls_slice=(i0, i1, 0, i1)) and
    returning (2, 1, ng, nij) real/imaginary parts.  Against an explicit sum
    over images the matrices agree to 2e-11 of max 1 (Y2O3), independent of
    the pyscf precision 1e-10..1e-14 (the explicit sum was the less converged
    one), at ~1/10 of the cost.'''
    from pyscf.pbc.df import ft_ao as pft
    c = cell.copy(deep=False)
    c.precision = min(cell.precision, precision)
    c.build(False, False)
    rs_cell = pft._RangeSeparatedCell.from_cell(c, pft.KECUT_THRESHOLD, pft.RCUT_THRESHOLD)
    rcut = pft.estimate_rcut(rs_cell)
    supmol = pft.ExtendedMole.from_cell(rs_cell, [1, 1, 1], rcut.max())
    supmol = supmol.strip_basis(rcut)
    return supmol.gen_ft_kernel('s2', return_complex=False)


def _sr_g(p, q, omega, d):
    '''SR (erfc) interaction of normalised s Gaussians, as in sr_int.c.'''
    from scipy.special import erf
    gamma = np.sqrt(p * q / (p + q))
    beta = 1. / np.sqrt(1. / omega**2 + 1. / p + 1. / q)
    d = np.asarray(d, dtype=float)
    out = np.where(d < 1e-8, 2 * (gamma - beta) / np.sqrt(np.pi),
                   (erf(gamma * d) - erf(beta * d)) / np.where(d < 1e-8, 1., d))
    return out


def _k_real_cderi(L, d):
    '''K_mn = sum_P sum_ls L[P,m,l] d[l,s] L[P,s,n] for real L and real d.'''
    naux, nao = L.shape[:2]
    X = (L.reshape(-1, nao) @ d).reshape(naux, nao, nao)          # X_P = L_P d
    # sum_P X_P L_P as one GEMM over the combined (P, s) index
    return X.transpose(1, 0, 2).reshape(nao, -1) @ L.reshape(-1, nao)


def _occ_factors(dm, nrow):
    '''Occupied factors C sqrt(n) (one (nrow, nocc) array per density) when dm
    carries pyscf's mo_coeff/mo_occ tags with non-negative occupations, so that
    dm = C n C^H; else None.'''
    c = getattr(dm, 'mo_coeff', None)
    n = getattr(dm, 'mo_occ', None)
    if c is None or n is None:
        return None
    c = np.asarray(c)
    n = np.asarray(n, dtype=float)
    c = c.reshape(-1, nrow, c.shape[-1])
    n = n.reshape(-1, n.shape[-1])
    if len(c) != len(n) or (n < -1e-14).any():
        return None
    return [ci[:, ni > 1e-14] * np.sqrt(ni[ni > 1e-14]) for ci, ni in zip(c, n)]


def _xxh(X, Y):
    '''sum_P X_P Y_P^H for (naux, nao, nocc) stacks.'''
    nao = X.shape[1]
    return X.transpose(1, 0, 2).reshape(nao, -1) @ Y.transpose(1, 0, 2).reshape(nao, -1).conj().T


def ghf_get_jk(get_jk, dm, hermi=0, with_j=True, with_k=True):
    '''Spin-orbital (2c) J/K from a scalar J/K builder.

    get_jk(dms, hermi, with_j, with_k) acts on a stack of (nao, nao) matrices.
    dm is (..., 2nao, 2nao) with [[aa, ab], [ba, bb]] blocks.  The alpha-beta
    exchange block uses D_ab = dm[:nao, nao:] (pyscf.pbc.scf.ghf takes the
    beta-alpha block there, which is wrong for non-collinear densities).'''
    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dms = dm.reshape(-1, nso, nso)
    daa, dbb = dms[:, :nao, :nao], dms[:, nao:, nao:]
    dab, dba = dms[:, :nao, nao:], dms[:, nao:, :nao]
    vj = vk = None
    if with_j:
        j = get_jk(daa + dbb, 1 if hermi else 0, True, False)[0]
        vj = np.zeros(dms.shape, dtype=np.result_type(j, dm))
        vj[:, :nao, :nao] = vj[:, nao:, nao:] = j
        vj = vj.reshape(dm.shape)
    if with_k:
        blocks = [daa, dbb, dab] if hermi else [daa, dbb, dab, dba]
        k = [get_jk(b, 0, False, True)[1] for b in blocks]
        vk = np.zeros(dms.shape, dtype=np.result_type(k[0], dm))
        vk[:, :nao, :nao] = k[0]
        vk[:, nao:, nao:] = k[1]
        vk[:, :nao, nao:] = k[2]
        vk[:, nao:, :nao] = k[2].conj().transpose(0, 2, 1) if hermi else k[3]
        vk = vk.reshape(dm.shape)
    return vj, vk


class RSDF(lib.StreamObject):
    '''Range-separated Gamma-point density fitting for a pyscf Cell.

    Args:
        cell : pyscf.pbc.gto.Cell
        auxbasis : auxiliary basis (anything pyscf.df.addons.make_auxmol accepts)
        omega : range-separation parameter; the result does not depend on it
            up to the screening thresholds
        precision : cutoff for the integral bounds and the G-space kernel
        lindep : relative eigenvalue threshold of the fitting metric
        nuc_zeta : Gaussian nuclear exponent(s); a very large value mimics
            point nuclei (default 1e12)
        lr : 'fit' (default) fits the whole kernel; 'gspace' fits the SR part
            only and appends the exact LR plane-wave rows (module notes)
    '''

    def __init__(self, cell, auxbasis, omega=0.5, precision=1e-10, lindep=1e-12, nuc_zeta=1e12, lr='fit'):
        if lr not in ('fit', 'gspace'):
            raise ValueError("lr must be 'fit' or 'gspace'")
        self.lr = lr
        self.c_g0 = 0.               # G = 0 coefficient kept out of the vectors (lr='gspace')
        self.nlr = 0                 # number of exact LR rows appended to the vectors
        self.cell = cell
        self.auxbasis = auxbasis
        self.omega = omega
        self.precision = precision
        self.lindep = lindep
        self.nuc_zeta = nuc_zeta
        self.verbose = cell.verbose
        self.stdout = cell.stdout
        self.max_memory = cell.max_memory
        self.grid_spacing = 4.0
        self.aux_bound = 'max'       # aux bound units: 'max' (default), 'l2' or 'coulomb'
        self.cderi_to_disk = False   # force HDF5 storage (default: only when larger than max_memory/2)
        self.cderi_file = None       # keep the vectors in this HDF5 file (RSDF.load reads it back)
        self.keep_raw = False        # keep the unfitted (aux|pair) and the aux metric (small cells; local fitting)
        self.metric_sr = None
        self.cderi = None            # (naux_kept, nao(nao+1)/2), pyscf packed pairs; ndarray or h5py dataset
        self.stats = {}

    # ------------------------------------------------------------------
    def _geometry(self):
        cell = self.cell
        a = cell.lattice_vectors()
        coords = cell.atom_coords()
        span = np.ptp(coords, axis=0).max() if cell.natm > 1 else 0.
        return a, cell_volume(a), coords, span

    def _ao_images(self):
        '''AO image molecule (cell 0 first), per-shell fold offsets and the
        overlap mask of (cell-0 shell, image shell) pairs.'''
        cell = self.cell
        a, vol, coords, span = self._geometry()
        lnp = -np.log(self.precision)
        amin0, cmax0 = _shell_params(cell)
        rpair = np.sqrt(lnp / (amin0.min() / 2))
        T = lattice_translations(a, rpair + span)
        T = T[np.argsort(np.linalg.norm(T, axis=1), kind='stable')]
        T = np.array([t for t in T
                      if np.min(np.linalg.norm(coords[:, None] - coords[None] - t, axis=2)) <= rpair])
        mol = _mole_at(cell, T)
        ao_loc0 = cell.ao_loc_nr()
        nbas0 = cell.nbas
        sh_fold = np.tile(ao_loc0[:-1], len(T)).astype(np.int32)
        amin, cmax = np.tile(amin0, len(T)), np.tile(cmax0, len(T))
        # overlap bound of shell pairs (cell-0 shell i, image shell j)
        ri = mol.atom_coords()[mol._bas[:nbas0, gto.ATOM_OF]]
        rj = mol.atom_coords()[mol._bas[:, gto.ATOM_OF]]
        r2 = np.einsum('ijx,ijx->ij', ri[:, None] - rj[None], ri[:, None] - rj[None])
        ai, aj = amin[:nbas0, None], amin[None, :]
        p = ai + aj
        kij = cmax[:nbas0, None] * cmax[None, :] * np.exp(-ai * aj / p * r2) * (np.pi / p)**1.5
        return mol, T, sh_fold, amin, cmax, kij

    def _aux_images(self, auxmol0, radius):
        a, vol, coords, span = self._geometry()
        T = lattice_translations(a, radius + span)
        T = T[np.argsort(np.linalg.norm(T, axis=1), kind='stable')]
        return _mole_at(auxmol0, T), T

    # ------------------------------------------------------------------
    def build(self):
        log = logger.new_logger(self)
        t0 = t1 = (logger.process_clock(), logger.perf_counter())
        cell = self.cell
        a, vol, coords, span = self._geometry()
        nao = cell.nao
        lnp = -np.log(self.precision)
        omega = self.omega

        # auxiliary functions of cell 0 plus the Gaussian nuclei as one more
        # "auxiliary" set of charge-normalised s shells
        mol0 = _mole_at(cell, [np.zeros(3)])
        auxmol0 = df_addons.make_auxmol(mol0, self.auxbasis)
        naux = auxmol0.nao
        zeta = np.broadcast_to(np.asarray(self.nuc_zeta, dtype=float), (cell.natm,))
        nucmol0 = gto.Mole()
        nucmol0.atom = [('X%d' % i, coords[i]) for i in range(cell.natm)]
        nucmol0.unit = 'Bohr'
        nucmol0.basis = {'X%d' % i: [[0, [zeta[i], 1.]]] for i in range(cell.natm)}
        nucmol0.verbose = 0
        nucmol0.build(False, False)
        # libcint-normalised s shells carry charge q; rescale them to the nuclear
        # charge Z so that the integral bounds are in physical units (a 1/q
        # rescaling after the fact would amplify the screened-out integrals)
        nucq = np.array([(np.pi / z)**1.5 * (2 * z / np.pi)**.75 for z in zeta])
        Z = cell.atom_charges().astype(float)
        nucscale = Z / nucq
        fitmol0 = gto.conc_mol(auxmol0, nucmol0)  # [aux | nuclei]
        nfit = fitmol0.nao
        naux_sh_cell = auxmol0.nbas

        aomol, Tao, sh_fold, amin, cmax, kij = self._ao_images()
        lmax = max(cell._bas[:, gto.ANG_OF].max(), fitmol0._bas[:, gto.ANG_OF].max())
        ang = _angular_amplitude(lmax)
        libp = _libpbcdf()
        env_t = np.zeros(8)
        nenv = libp.SOCpbc_nenv(env_t.ctypes.data_as(ctypes.c_void_p))
        env_t = env_t[:nenv]
        # one envelope (exponent, charge) per shell and envelope parameter t
        qmin0, cq0 = np.array([_shell_envelope(fitmol0, ang, t) for t in env_t]).transpose(1, 0, 2)
        # Auxiliary functions enter through fit coefficients, so their bound
        # depends on the units the coefficients are measured in.  'l2' takes the
        # libcint (L2) normalisation, 'coulomb' rescales to (P|P) = 1, 'max'
        # (default) the larger of the two.  The nuclei are not fitted and carry
        # their physical charge Z.
        if self.aux_bound in ('coulomb', 'max'):
            c_coul = cq0[:, :naux_sh_cell] / np.sqrt(_aux_coulomb_diag(auxmol0))
            if self.aux_bound == 'coulomb':
                cq0[:, :naux_sh_cell] = c_coul
            else:
                cq0[:, :naux_sh_cell] = np.maximum(cq0[:, :naux_sh_cell], c_coul)
        cq0[:, naux_sh_cell:] *= nucscale
        # aux images: far enough for the most diffuse pair with the most diffuse aux
        p_lo = 2 * amin.min()
        beta = 1. / np.sqrt(1. / omega**2 + 2. / p_lo + 1. / qmin0.min())
        r_sr = np.sqrt(lnp) / beta + np.sqrt(lnp / (amin.min() / 2))
        auxmol, Tfit = self._aux_images(fitmol0, r_sr)
        log.info('RSDF: omega %.3f  nao %d  naux %d  AO images %d  aux images %d  (pair radius %.1f, SR radius %.1f bohr)',
                 omega, nao, naux, len(Tao), len(Tfit), np.sqrt(lnp / (amin.min() / 2)), r_sr)

        # one libcint environment: [AO images | fit images]
        sysmol = gto.conc_mol(aomol, auxmol)
        env = sysmol._env.copy()
        env[gto.PTR_RANGE_OMEGA] = -omega
        atm = np.asarray(sysmol._atm, dtype=np.int32, order='C')
        bas = np.asarray(sysmol._bas, dtype=np.int32, order='C')
        nao_sh = aomol.nbas
        aux_sh0, naux_sh = aomol.nbas, auxmol.nbas
        nfit_sh0 = fitmol0.nbas
        # images of one nucleus share their coefficient slot: scale each slot once
        scaled = set()
        for k in range(aux_sh0, sysmol.nbas):
            ifit = (k - aux_sh0) % nfit_sh0
            if ifit >= naux_sh_cell:
                ptr_c = bas[k, gto.PTR_COEFF]
                if ptr_c not in scaled:
                    env[ptr_c] *= nucscale[ifit - naux_sh_cell]
                    scaled.add(ptr_c)
        fit_loc0 = fitmol0.ao_loc_nr()
        aux_fold = np.zeros(sysmol.nbas, dtype=np.int32)
        aux_fold[aux_sh0:] = np.tile(fit_loc0[:-1], len(Tfit))
        ao_fold = np.zeros(sysmol.nbas, dtype=np.int32)
        ao_fold[:nao_sh] = sh_fold
        qmin = np.zeros((nenv, sysmol.nbas))
        cq = np.zeros((nenv, sysmol.nbas))
        qmin[:, aux_sh0:] = np.tile(qmin0, len(Tfit))
        cq[:, aux_sh0:] = np.tile(cq0, len(Tfit))

        # bin the aux image shells on a uniform grid
        rk = sysmol.atom_coords()[bas[aux_sh0:, gto.ATOM_OF]]
        h = self.grid_spacing
        origin = rk.min(axis=0) - h
        gdims = (np.ceil((rk.max(axis=0) - origin) / h).astype(int) + 2).astype(np.int32)
        idx = np.floor((rk - origin) / h).astype(int)
        binid = (idx[:, 0] * gdims[1] + idx[:, 1]) * gdims[2] + idx[:, 2]
        order = np.argsort(binid, kind='stable')
        bin_shells = (order + aux_sh0).astype(np.int32)
        bin_start = np.searchsorted(binid[order], np.arange(np.prod(gdims) + 1)).astype(np.int32)

        cutoff = self.precision * 1e-2
        c_int, c_dbl = ctypes.c_int, ctypes.c_double
        ptr = lambda x: x.ctypes.data_as(ctypes.c_void_p)

        # ---- metric (fit x fit): SR as full - LR (libcint's erfc 2c integrals
        # are not accurate enough, see sr_int.c), LR in G space, G = 0 removal
        M = np.zeros((nfit, nfit))
        stats2 = np.zeros(2, dtype=np.int64)
        env_full, env_lr = env.copy(), env.copy()
        env_full[gto.PTR_RANGE_OMEGA] = 0.
        env_lr[gto.PTR_RANGE_OMEGA] = omega
        libp.SOCpbc_sr_int2c(
            ptr(M), c_int(nfit), c_int(aux_sh0), c_int(nfit_sh0), c_int(naux_sh),
            ptr(aux_fold), ptr(qmin), ptr(cq), c_dbl(omega), c_dbl(cutoff),
            ptr(atm), c_int(sysmol.natm), ptr(bas), c_int(sysmol.nbas), ptr(env_full), ptr(env_lr),
            ptr(stats2))
        t1 = log.timer('RSDF SR 2c', *t1)
        gspace = self.lr == 'gspace'
        if self.keep_raw or gspace:
            self.metric_sr = .5 * (M[:naux, :naux] + M[:naux, :naux].T)

        G = reciprocal_vectors(a, 2 * omega * np.sqrt(lnp + np.log(4 * np.pi)))
        # all densities are real, so the -G term is the complex conjugate of
        # the +G term: keep one vector of each +-G pair with weight 2
        b = 2 * np.pi * np.linalg.inv(a).T
        m = np.rint(G @ np.linalg.inv(b)).astype(int)         # integer coordinates
        half = (m[:, 0] > 0) | ((m[:, 0] == 0) & ((m[:, 1] > 0) | ((m[:, 1] == 0) & (m[:, 2] > 0))))
        G = G[half]
        G2 = np.einsum('gx,gx->g', G, G)
        vG = 2 * 4 * np.pi * np.exp(-G2 / (4 * omega**2)) / G2

        # aux FTs (ng, nfit); computed once for all G when small enough
        cache_fit = len(G) * nfit * 16 < .1 * self.max_memory * 1e6

        def fit_ft(g0, g1):
            if cache_fit:
                if 'all' not in fit_cache:
                    fit_cache['all'] = _fit_ft(0, len(G))
                return fit_cache['all'][g0:g1]
            return _fit_ft(g0, g1)

        def _fit_ft(g0, g1):
            fit = ft_ao(fitmol0, G[g0:g1]) / vol
            fit[:, naux:] *= nucscale
            return fit
        fit_cache = {}

        gblk_fit = len(G) if cache_fit else max(16, int(self.max_memory * .1e6 / (16 * nfit * 2)))
        for g0 in range(0, len(G), gblk_fit):
            g1 = min(g0 + gblk_fit, len(G))
            fit = fit_ft(g0, g1)
            M += ((fit.conj().T * (vG[g0:g1] * vol)) @ fit).real
        q = ft_ao(fitmol0, np.zeros((1, 3)))[0].real
        q[naux:] *= nucscale
        c0 = np.pi / (omega**2 * vol)
        M -= c0 * np.outer(q, q)
        M = .5 * (M + M.T)
        t1 = log.timer('RSDF LR 2c (%d G vectors)' % len(G), *t1)

        e_self = np.sum(Z**2 * np.sqrt(2 * zeta / np.pi)) / 2
        self.e_nuc = .5 * M[naux:, naux:].sum() - e_self
        self.metric = M[:naux, :naux].copy() if self.keep_raw else None
        # lr='gspace': the SR kernel (with its own G = 0 part) is fitted
        Mfit = self.metric_sr if gspace else M[:naux, :naux]
        if gspace and not self.keep_raw:
            self.metric_sr = None
        M = M[naux:, naux:] if gspace else M
        w, v = scipy.linalg.eigh(Mfit)
        Mfit = None
        self.metric_eig = (w.min(), w.max())
        if w.min() < -self.lindep * w.max() * 1e2:
            log.warn('RSDF: fitting metric has negative eigenvalue %.2e (max %.2e): integrals are inaccurate',
                     w.min(), w.max())
        keep = w > self.lindep * w.max()
        self.naux_kept = nkept = int(keep.sum())
        W = np.asarray((v[:, keep] / np.sqrt(w[keep])).T, order='C')  # (nkept, naux)
        log.info('RSDF: metric eigenvalues [%.2e, %.2e], kept %d of %d', w.min(), w.max(), nkept, naux)

        # ---- 3c, in blocks of cell-0 AO shells (rows), lower triangle: rows
        # mu of the block, columns nu <= mu, i.e. a contiguous range of
        # pyscf's packed (pack_tril) pair index
        npair = nao * (nao + 1) // 2
        self.c_g0 = c0 if gspace else 0.
        self.nlr = nlr = 2 * len(G) if gspace else 0
        nvec = nkept + nlr
        if gspace:
            log.info('RSDF lr=gspace: %d SR fitted vectors + %d exact LR rows (c0 = %.6g)', nkept, nlr, c0)
        self._cderi_file = None
        self._h5 = None
        if self.cderi_file or nvec * npair * 8 > .5 * self.max_memory * 1e6 or self.cderi_to_disk:
            import h5py
            if self.cderi_file:
                path = self.cderi_file
            else:
                import tempfile
                self._cderi_file = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR, suffix='.h5')
                path = self._cderi_file.name
            self._h5 = f = h5py.File(path, 'w')
            cderi = f.create_dataset('j3c', (nvec, npair), 'f8', chunks=(min(nvec, 64), min(npair, 4096)))
        else:
            cderi = np.empty((nvec, npair))
        self.cderi = cderi
        self.int3c = None            # (naux, npair) unfitted (aux|pair), with keep_raw
        self.int3c_sr = None         # its short-range (erfc) part, with keep_raw
        vnuc = np.zeros(npair)
        S = np.zeros(npair)

        ao_loc = cell.ao_loc_nr()
        # rows per block: B (nfit, nrow, nao) and LR work space (ng, nrow, nao nT, complex)
        nT = len(Tao)
        assert aomol.nao == nT * nao
        row_max = max(1, int(.25 * self.max_memory * 1e6 / (8 * nfit * nao * 2)))
        blocks, i0 = [], 0
        while i0 < cell.nbas:
            i1 = i0 + 1
            while i1 < cell.nbas and ao_loc[i1 + 1] - ao_loc[i0] <= row_max:
                i1 += 1
            blocks.append((i0, i1))
            i0 = i1
        ft_kern = _gamma_ft_kernel(cell, cutoff)
        sh_cell = np.tile(np.arange(cell.nbas, dtype=np.int32), len(Tao))
        stats = np.zeros(3, dtype=np.int64)
        tot = dict(pairs=0, triples=0, screened=0)
        t_sr = t_lr = 0.
        for ib, (i0, i1) in enumerate(blocks):
            r0, r1 = ao_loc[i0], ao_loc[i1]
            tsr0, tlr0 = t_sr, t_lr
            nrow = r1 - r0
            tb = logger.perf_counter()
            B = np.zeros((nfit, nrow, nao))
            libp.SOCpbc_sr_int3c(
                ptr(B), c_int(nao), c_int(nfit), c_int(i0), c_int(i1), c_int(r0), c_int(nrow),
                c_int(cell.nbas), c_int(nao_sh),
                c_int(aux_sh0), c_int(naux_sh), ptr(ao_fold), ptr(sh_cell), ptr(aux_fold),
                ptr(qmin), ptr(cq),
                ptr(origin), c_dbl(h), ptr(gdims), ptr(bin_start), ptr(bin_shells),
                c_dbl(omega), c_dbl(cutoff), ptr(ang),
                ptr(atm), c_int(sysmol.natm), ptr(bas), c_int(sysmol.nbas), ptr(env), ptr(stats))
            for k, key in enumerate(('pairs', 'triples', 'screened')):
                tot[key] += int(stats[k])
            t_sr += logger.perf_counter() - tb
            tb = logger.perf_counter()
            # lower triangle of the block rows -> packed pair range [p0, p1)
            tril = np.arange(nao)[None, :] <= np.arange(r0, r1)[:, None]
            p0, p1 = r0 * (r0 + 1) // 2, r1 * (r1 + 1) // 2
            Bp = B[:, tril]                                            # (nfit, p1 - p0)
            if self.keep_raw:
                if self.int3c_sr is None:
                    self.int3c_sr = np.zeros((naux, npair))
                self.int3c_sr[:, p0:p1] = Bp[:naux]
            B = None
            # long range: lattice-summed pair FTs at Gamma for the packed pair
            # range [p0, p1), real and imaginary parts (pyscf's C lattice sum)
            def pair_ft(Gb):
                F = ft_kern(Gb, kptjs=np.zeros((1, 3)), shls_slice=(i0, i1, 0, i1))[:, 0]
                return np.asarray(F[0], order='C') / vol, np.asarray(F[1], order='C') / vol

            gblk = len(G) if cache_fit else gblk_fit
            gblk = max(16, min(gblk, int(.2 * self.max_memory * 1e6 / (16 * nrow * nao * 3))))
            # lr='gspace': LR and G = 0 only for the nuclei (V_ne); the pairs'
            # LR goes into the exact rows
            f0 = naux if gspace else 0
            for g0 in range(0, len(G), gblk):
                g1 = min(g0 + gblk, len(G))
                PgR, PgI = pair_ft(G[g0:g1])                         # (ng, p1 - p0)
                fw = fit_ft(g0, g1)[:, f0:].T * (vG[g0:g1] * vol)      # (nfit - f0, ng)
                # Re(fw conj(Pg)) as two real GEMMs
                Bp[f0:] += np.asarray(fw.real, order='C') @ PgR
                Bp[f0:] += np.asarray(fw.imag, order='C') @ PgI
                if gspace:
                    sq = np.sqrt(vG[g0:g1] * vol)[:, None]
                    rows = np.empty((g1 - g0, 2, p1 - p0))
                    rows[:, 0] = sq * PgR
                    rows[:, 1] = sq * PgI
                    cderi[nkept + 2 * g0:nkept + 2 * g1, p0:p1] = rows.reshape(-1, p1 - p0)
                    rows = None
                PgR = PgI = fw = None
            # overlap at G = 0 (same pair list), for the SR G = 0 removal
            Sb = pair_ft(np.zeros((1, 3)))[0][0] * vol
            Bp[f0:] -= c0 * np.outer(q[f0:], Sb)
            t_lr += logger.perf_counter() - tb
            S[p0:p1] = Sb
            vnuc[p0:p1] = -Bp[naux:].sum(axis=0)
            tb = logger.perf_counter()
            cderi[:nkept, p0:p1] = W @ Bp[:naux]
            if self.keep_raw and not gspace:
                if self.int3c is None:
                    self.int3c = np.zeros((naux, npair))
                self.int3c[:, p0:p1] = Bp[:naux]
            Bp = None
            log.debug('RSDF block %d/%d (%d rows): SR %.1f s (%d triples), LR %.1f s, fit+store %.1f s',
                      ib + 1, len(blocks), nrow, t_sr - tsr0, int(stats[1]), t_lr - tlr0,
                      logger.perf_counter() - tb)
        self.stats['sr3c'] = tot
        log.info('RSDF 3c: %d row blocks, SR %.1f s (%d pairs, %d triples), LR %.1f s (%d G vectors)',
                 len(blocks), t_sr, tot['pairs'], tot['triples'], t_lr, len(G))
        self.S = lib.unpack_tril(S)
        self.vnuc = lib.unpack_tril(vnuc)
        if self.cderi_file:
            self._save_meta(self._h5)
            self._h5.flush()
        log.timer('RSDF build', *t0)
        return self

    # ------------------------------------------------------------------
    # persistent vectors: build once, reuse for SCF / CASCI / restarts
    def _fingerprint(self):
        cell = self.cell
        return np.hstack([cell.lattice_vectors().ravel(), cell.atom_coords().ravel(),
                          cell.atom_charges(), [cell.nao, cell.nbas]]).astype(float)

    def _save_meta(self, f):
        for k in ('S', 'vnuc'):
            if k in f:
                del f[k]
            f[k] = getattr(self, k)
        f.attrs['e_nuc'] = self.e_nuc
        f.attrs['metric_eig'] = np.asarray(self.metric_eig, dtype=float)
        f.attrs['naux_kept'] = self.naux_kept
        f.attrs['lr'] = self.lr
        f.attrs['c_g0'] = self.c_g0
        f.attrs['nlr'] = self.nlr
        f.attrs['omega'] = self.omega
        f.attrs['precision'] = self.precision
        f.attrs['nuc_zeta'] = np.atleast_1d(np.asarray(self.nuc_zeta, dtype=float))
        f.attrs['fingerprint'] = self._fingerprint()

    def save(self, path):
        '''Write the vectors and the one-electron/nuclear parts to an HDF5 file.'''
        import h5py
        if self.cderi is None:
            self.build()
        if self._h5 is not None and self.cderi_file and \
                __import__('os').path.abspath(path) == __import__('os').path.abspath(self.cderi_file):
            return self
        with h5py.File(path, 'w') as f:
            d = f.create_dataset('j3c', self.cderi.shape, 'f8',
                                 chunks=(min(self.cderi.shape[0], 64), min(self.cderi.shape[1], 4096)))
            for p0, blk in zip(range(0, self.cderi.shape[0], 256), self.loop(256)):
                d[p0:p0 + blk.shape[0]] = blk
            self._save_meta(f)
        return self

    @classmethod
    def load(cls, cell, path, check=True):
        '''RSDF from a file written by build (cderi_file) or save; the vectors
        stay on disk and are streamed.'''
        import h5py
        f = h5py.File(path, 'r')
        r = cls(cell, None, omega=float(f.attrs['omega']), precision=float(f.attrs['precision']),
                nuc_zeta=f.attrs['nuc_zeta'] if len(f.attrs['nuc_zeta']) > 1 else float(f.attrs['nuc_zeta'][0]))
        if check and not np.allclose(r._fingerprint(), f.attrs['fingerprint'], atol=1e-8):
            raise ValueError('%s was built for a different cell' % path)
        r._h5 = f
        r.cderi = f['j3c']
        r.S = f['S'][()]
        r.vnuc = f['vnuc'][()]
        r.e_nuc = float(f.attrs['e_nuc'])
        r.metric_eig = tuple(f.attrs['metric_eig'])
        r.naux_kept = int(f.attrs['naux_kept'])
        r.lr = str(f.attrs.get('lr', 'fit'))
        r.c_g0 = float(f.attrs.get('c_g0', 0.))
        r.nlr = int(f.attrs.get('nlr', 0))
        return r

    # ------------------------------------------------------------------
    def get_naoaux(self):
        if self.cderi is None:
            self.build()
        return self.cderi.shape[0]

    def loop(self, blksize=None):
        '''Yield packed (naux_blk, nao*(nao+1)/2) real blocks, like pyscf DF.loop.'''
        if self.cderi is None:
            self.build()
        if blksize is None:
            nao = self.cell.nao
            blksize = max(16, min(1024, int(.2 * self.max_memory * 1e6 / (8 * nao * nao * 3))))
        for p0 in range(0, self.cderi.shape[0], blksize):
            yield np.asarray(self.cderi[p0:p0 + blksize])

    def _j_density(self, dms):
        '''Packed (nset, npair) densities for J: sum_{mu>=nu} (D_mu nu + D_nu mu)(1 - delta/2).'''
        nao = self.cell.nao
        tril = np.tril_indices(nao)
        dsym = dms + dms.transpose(0, 2, 1)
        dsym[:, np.arange(nao), np.arange(nao)] *= .5
        return dsym[:, tril[0], tril[1]]

    def g0_constant(self, nelectron):
        '''Energy of the G = 0 term kept out of the vectors (lr='gspace') for
        any N-electron wavefunction: -c0 N (N - 1) / 2.'''
        return -.5 * self.c_g0 * nelectron * (nelectron - 1)

    def _g0_jk(self, vj, vk, dms):
        '''Add the -c0 S_mn S_ls term to J (packed) and K.'''
        c0 = self.c_g0
        if vj is not None:
            n = np.einsum('xij,ji->x', dms, self.S)
            vj -= c0 * n[:, None] * lib.pack_tril(self.S)[None]
        if vk is not None:
            vk -= c0 * np.einsum('ij,xjk,kl->xil', self.S, dms, self.S)

    def _exxdiv(self, vk, dms, exxdiv):
        if exxdiv == 'ewald':
            from pyscf.pbc import tools
            madelung = tools.pbc.madelung(self.cell, np.zeros((1, 3)))
            return vk + madelung * np.einsum('ij,xjk,kl->xil', self.S, dms, self.S)
        elif exxdiv is not None:
            raise NotImplementedError(exxdiv)
        return vk

    # pyscf/socutils DF interface (scf.spinor_hf._DFJHF, mcscf.zmc_ao2mo._CDERIS)
    blockdim = 240

    @property
    def mol(self):
        return self.cell

    @property
    def _cderi(self):
        return self.cderi

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=None, omega=None, exxdiv=None,
               with_g0=False):
        '''Gamma-point J and K (pyscf conventions; dm real or complex).

        J/K of the vectors.  For lr='gspace' that excludes the G = 0 part of
        the SR kernel, -c0 S S; with_g0=True adds it (J: -c0 tr(DS) S, K:
        -c0 S D S).  The default matches what loop() consumers (the spinor
        SCF's MO-based K, _CDERIS) see; spinor_scf adds the term to the mean
        field itself (g0_terms).

        The argument order is that of pyscf's DF.get_jk (direct_scf_tol is
        ignored; range-separated omega is not supported).

        When dm carries mo_coeff/mo_occ (pyscf tag_array, occupations >= 0,
        fractional allowed), K is built from the occupied factors (occ-RI):
        K = sum_P (L_P C)(L_P C)^H, naux nao^2 nocc instead of 2 naux nao^3.'''
        if omega:
            raise NotImplementedError('RSDF.get_jk with omega')
        if self.cderi is None:
            self.build()
        nao = self.cell.nao
        occ = _occ_factors(dm, nao) if with_k else None
        dm = np.asarray(dm)
        dms = dm.reshape(-1, nao, nao)
        nset = len(dms)
        if occ is not None and len(occ) != nset:
            occ = None
        dtril = self._j_density(dms)
        vj = np.zeros((nset, dtril.shape[1]), dtype=dtril.dtype) if with_j else None
        kdtype = np.result_type(dms, *(occ or []))
        vk = np.zeros(dms.shape, dtype=kdtype) if with_k else None
        for Lp in self.loop():
            if with_j:
                vj += (dtril @ Lp.T) @ Lp
            if with_k:
                L = lib.unpack_tril(Lp)
                for i, d in enumerate(dms):
                    if occ is not None:
                        X = L @ occ[i]                                 # (naux, nao, nocc)
                        vk[i] += _xxh(X, X)
                    else:
                        vk[i] += _k_real_cderi(L, d.real)
                        if np.iscomplexobj(d):
                            vk[i] += 1j * _k_real_cderi(L, d.imag)
                L = X = None
        if self.c_g0 and with_g0:
            self._g0_jk(vj, vk, dms)
        if with_j:
            vj = lib.unpack_tril(vj, lib.SYMMETRIC).reshape(dm.shape)
        if with_k:
            if not np.iscomplexobj(dms) and np.iscomplexobj(vk):
                vk = vk.real
            vk = self._exxdiv(vk, dms, exxdiv).reshape(dm.shape)
        return vj, vk

    def get_jk_ghf(self, dm, hermi=0, with_j=True, with_k=True, exxdiv=None, with_g0=False):
        '''J/K for a (2nao, 2nao) spin-orbital density matrix.  With
        mo_coeff/mo_occ tags K uses the occupied spinors (occ-RI): the spin
        blocks are K_st = sum_P (L_P C_s)(L_P C_t)^H, one pass over the vectors.'''
        nao = self.cell.nao
        occ = _occ_factors(dm, 2 * nao) if with_k else None
        if occ is None:
            return ghf_get_jk(lambda d, h, wj, wk: self.get_jk(d, h, wj, wk, exxdiv=exxdiv, with_g0=with_g0),
                              dm, hermi, with_j, with_k)
        if self.cderi is None:
            self.build()
        dm = np.asarray(dm)
        dms = dm.reshape(-1, 2 * nao, 2 * nao)
        nset = len(dms)
        dtril = self._j_density(dms[:, :nao, :nao] + dms[:, nao:, nao:])
        vj = np.zeros((nset, dtril.shape[1]), dtype=dtril.dtype) if with_j else None
        vk = np.zeros(dms.shape, dtype=np.result_type(dms, *occ))
        for Lp in self.loop():
            if with_j:
                vj += (dtril @ Lp.T) @ Lp
            L = lib.unpack_tril(Lp)
            for i, c in enumerate(occ):
                Xa = L @ c[:nao]
                Xb = L @ c[nao:]
                vk[i, :nao, :nao] += _xxh(Xa, Xa)
                vk[i, nao:, nao:] += _xxh(Xb, Xb)
                vk[i, :nao, nao:] += _xxh(Xa, Xb)
            L = Xa = Xb = None
        vk[:, nao:, :nao] = vk[:, :nao, nao:].conj().transpose(0, 2, 1)
        if self.c_g0 and with_g0:
            S2 = np.zeros((2 * nao, 2 * nao))
            S2[:nao, :nao] = S2[nao:, nao:] = self.S
            n = np.einsum('xij,ji->x', dms, S2)
            if with_j:
                vj -= self.c_g0 * n[:, None] * lib.pack_tril(self.S)[None]
            vk -= self.c_g0 * np.einsum('ij,xjk,kl->xil', S2, dms, S2)
        for i in range(nset):
            for s0, t0 in ((0, 0), (nao, nao), (0, nao), (nao, 0)):
                blk = vk[i:i + 1, s0:s0 + nao, t0:t0 + nao]
                vk[i, s0:s0 + nao, t0:t0 + nao] = self._exxdiv(
                    blk, dms[i:i + 1, s0:s0 + nao, t0:t0 + nao], exxdiv)[0]
        if with_j:
            j = lib.unpack_tril(vj, lib.SYMMETRIC)
            vj = np.zeros(dms.shape, dtype=j.dtype)
            vj[:, :nao, :nao] = vj[:, nao:, nao:] = j
            vj = vj.reshape(dm.shape)
        else:
            vj = None
        return vj, vk.reshape(dm.shape)

    def get_nuc(self):
        '''Electron-nucleus attraction with Gaussian nuclei, exact (not fitted).'''
        if self.cderi is None:
            self.build()
        return self.vnuc

    def energy_nuc(self):
        '''Nucleus-nucleus repulsion of the Gaussian nuclei without self energies.'''
        if self.cderi is None:
            self.build()
        return self.e_nuc
