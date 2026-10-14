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
    '''

    def __init__(self, cell, auxbasis, omega=0.5, precision=1e-10, lindep=1e-12, nuc_zeta=1e12):
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
        self.cderi = None
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
        stats = np.zeros(3, dtype=np.int64)
        B = np.zeros((nfit, nao, nao))
        sh_cell = np.tile(np.arange(cell.nbas, dtype=np.int32), len(Tao))
        libp.SOCpbc_sr_int3c(
            ptr(B), c_int(nao), c_int(nfit), c_int(cell.nbas), c_int(nao_sh),
            c_int(aux_sh0), c_int(naux_sh), ptr(ao_fold), ptr(sh_cell), ptr(aux_fold),
            ptr(qmin), ptr(cq),
            ptr(origin), c_dbl(h), ptr(gdims), ptr(bin_start), ptr(bin_shells),
            c_dbl(omega), c_dbl(cutoff), ptr(ang),
            ptr(atm), c_int(sysmol.natm), ptr(bas), c_int(sysmol.nbas), ptr(env), ptr(stats))
        # only shell pairs (i, j) with cell shell of j >= i were evaluated; the
        # rest follow from (mu_0 nu_T|P) = (nu_0 mu_-T|P)
        ao2sh = np.repeat(np.arange(cell.nbas), np.diff(cell.ao_loc_nr()))
        offdiag = (ao2sh[:, None] != ao2sh[None, :])
        B += B.transpose(0, 2, 1) * offdiag
        self.stats['sr3c'] = dict(pairs=int(stats[0]), triples=int(stats[1]), screened=int(stats[2]))
        t1 = log.timer('RSDF SR 3c (%d pairs, %d triples)' % (stats[0], stats[1]), *t1)
        M = np.zeros((nfit, nfit))
        stats2 = np.zeros(2, dtype=np.int64)
        libp.SOCpbc_sr_int2c(
            ptr(M), c_int(nfit), c_int(aux_sh0), c_int(nfit_sh0), c_int(naux_sh),
            ptr(aux_fold), ptr(qmin), ptr(cq), c_dbl(omega), c_dbl(cutoff),
            ptr(atm), c_int(sysmol.natm), ptr(bas), c_int(sysmol.nbas), ptr(env), ptr(stats2))
        t1 = log.timer('RSDF SR 2c', *t1)

        # long range, reciprocal space
        G = reciprocal_vectors(a, 2 * omega * np.sqrt(lnp + np.log(4 * np.pi)))
        # all densities are real, so the -G term is the complex conjugate of
        # the +G term: keep one vector of each +-G pair with weight 2
        b = 2 * np.pi * np.linalg.inv(a).T
        m = np.rint(G @ np.linalg.inv(b)).astype(int)         # integer coordinates
        half = (m[:, 0] > 0) | ((m[:, 0] == 0) & ((m[:, 1] > 0) | ((m[:, 1] == 0) & (m[:, 2] > 0))))
        G = G[half]
        G2 = np.einsum('gx,gx->g', G, G)
        vG = 2 * 4 * np.pi * np.exp(-G2 / (4 * omega**2)) / G2
        mask = np.asarray(kij > cutoff, dtype=np.int8, order='F')
        # image AOs are translation-major copies of the cell AOs: folding them
        # onto cell 0 is a sum over the translation axis
        nT = len(Tao)
        assert aomol.nao == nT * nao
        fold = lambda F: F.reshape(F.shape[0], nao, nT, nao).sum(axis=2)
        blk = max(16, int(self.max_memory * .1e6 / (16 * nao * aomol.nao * 3)))
        Bl = B.reshape(nfit, -1)
        S = np.zeros((nao, nao))
        for g0 in range(0, len(G), blk):
            g1 = min(g0 + blk, len(G))
            F = ft_aopair(aomol, G[g0:g1], shls_slice=(0, cell.nbas, 0, aomol.nbas), ovlp_mask=mask)
            Pg = fold(F) / vol                                         # (ng, nao, nao)
            fit = ft_ao(fitmol0, G[g0:g1]) / vol                       # (ng, nfit)
            fit[:, naux:] *= nucscale
            w = vG[g0:g1] * vol
            Bl += ((fit.T * w) @ Pg.reshape(g1 - g0, -1).conj()).real
            M += ((fit.conj().T * w) @ fit).real
        # overlap and charges at G = 0 (same pair list), for the SR G = 0 removal
        F0 = ft_aopair(aomol, np.zeros((1, 3)), shls_slice=(0, cell.nbas, 0, aomol.nbas), ovlp_mask=mask)
        S = fold(F0)[0].real
        q = ft_ao(fitmol0, np.zeros((1, 3)))[0].real
        q[naux:] *= nucscale
        c0 = np.pi / (omega**2 * vol)
        Bl -= c0 * np.outer(q, S.ravel())
        M -= c0 * np.outer(q, q)
        t1 = log.timer('RSDF LR (%d G vectors)' % len(G), *t1)

        B = Bl.reshape(nfit, nao, nao)
        self.sym_err = abs(B - B.transpose(0, 2, 1)).max()
        B = .5 * (B + B.transpose(0, 2, 1))
        M = .5 * (M + M.T)

        # split the auxiliary and the nuclear parts
        self.S = .5 * (S + S.T)
        # rows/columns naux: are already the nuclear charges Z_A g_A
        Baux = B[:naux]
        Maux = M[:naux, :naux]
        self.vnuc = -B[naux:].sum(axis=0)
        e_self = np.sum(Z**2 * np.sqrt(2 * zeta / np.pi)) / 2
        self.e_nuc = .5 * M[naux:, naux:].sum() - e_self

        w, v = scipy.linalg.eigh(Maux)
        self.metric_eig = (w.min(), w.max())
        if w.min() < -self.lindep * w.max() * 1e2:
            log.warn('RSDF: fitting metric has negative eigenvalue %.2e (max %.2e): integrals are inaccurate',
                     w.min(), w.max())
        keep = w > self.lindep * w.max()
        self.naux_kept = int(keep.sum())
        self.cderi = ((v[:, keep] / np.sqrt(w[keep])).T @ Baux.reshape(naux, -1)).reshape(-1, nao, nao)
        log.info('RSDF: metric eigenvalues [%.2e, %.2e], kept %d of %d; B symmetry error %.1e',
                 w.min(), w.max(), self.naux_kept, naux, self.sym_err)
        log.timer('RSDF build', *t0)
        return self

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
            blksize = 240
        for p0 in range(0, self.cderi.shape[0], blksize):
            yield lib.pack_tril(self.cderi[p0:p0 + blksize])

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, exxdiv=None):
        '''Gamma-point J and K (pyscf conventions; dm real or complex).'''
        if self.cderi is None:
            self.build()
        dm = np.asarray(dm)
        nao = self.cell.nao
        dms = dm.reshape(-1, nao, nao)
        L = self.cderi
        vj = vk = None
        if with_j:
            vj = np.array([np.einsum('P,Pij->ij', np.einsum('Pij,ji->P', L, d), L) for d in dms])
            vj = vj.reshape(dm.shape)
        if with_k:
            vk = np.array([_k_real_cderi(L, d.real) + (1j * _k_real_cderi(L, d.imag)
                           if np.iscomplexobj(d) else 0) for d in dms])
            if exxdiv == 'ewald':
                from pyscf.pbc import tools
                madelung = tools.pbc.madelung(self.cell, np.zeros((1, 3)))
                vk = vk + madelung * np.einsum('ij,xjk,kl->xil', self.S, dms, self.S)
            elif exxdiv is not None:
                raise NotImplementedError(exxdiv)
            vk = vk.reshape(dm.shape)
        return vj, vk

    def get_jk_ghf(self, dm, hermi=0, with_j=True, with_k=True, exxdiv=None):
        '''J/K for a (2nao, 2nao) spin-orbital density matrix.'''
        return ghf_get_jk(lambda d, h, wj, wk: self.get_jk(d, h, wj, wk, exxdiv=exxdiv),
                          dm, hermi, with_j, with_k)

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
