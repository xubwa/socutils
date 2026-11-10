#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Seminumerical exchange (chain of spheres, COSX; F. Neese, F. Wennmohs,
A. Hansen, U. Becker, Chem. Phys. 356, 98 (2009); overlap fitting:
R. Izsak, F. Neese, J. Chem. Phys. 135, 144105 (2011)) for real densities,
the kernel in lib/dft/cosx.c:

    K_mu nu = sum_g Q_mu g sum_lam A_nu lam(g) F_lam g,
    F_lam g = sum_kap D_kap lam phi_kap(g),   A_nu lam(g) = (nu lam | delta_g),
    Q = S S_g^-1 (w phi)          (fit=True; S_g = sum_g w phi phi^T on the grid)
      = w phi                     (fit=False)

the analytic one-electron potential integrals A of the AO pairs on every grid
point (McMurchie-Davidson, SIMD over the points of a block), the K of
symmetric densities symmetrised.

    k = cosx.COSX(mol)                  # grid: ORCA-like (dft.grids.Grids), see GRID
    vk = k.get_k(dm)                    # dm (nao, nao) or (nset, nao, nao)
    mf = cosx.apply(scf.RKS(mol, xc='pbe0').density_fit(only_dfj=True))
                                        # RIJ J (md_j) + COSX K in the SCF
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger

# the COSX grids, dft.grids.Grids attributes (intacc, angular_grid): ORCA 6's GRIDX1-3 (IntAcc 3.816 /
# 4.020 / 4.338, Lebedev-50 / 110 / 194 at most); the SCF runs on GRIDX2 (GRIDX1 while the density
# moves, SCF_GRID_SWITCH, off by default), the final exchange energy is recomputed on GRIDX3 (as ORCA)
GRIDX = {1: {'intacc': 3.816, 'angular_grid': 1},
         2: {'intacc': 4.020, 'angular_grid': 2},
         3: {'intacc': 4.338, 'angular_grid': 3}}
GRID = GRIDX[2]
BLK = 112               # grid points per task of the kernel (a multiple of 56)
AO_CUTOFF = 1e-10       # AO values below which an AO is dropped from a block
TOL = 1e-9              # screening of the shell-pair contributions, B_IJ max|F_J| sum_g |w phi| < TOL
TCUT = 36.              # p r^2 beyond which a one-centre primitive pair is in its far field
PRIM_TOL = 1e-14
PAIR_TOL = 1e-13


def _lib():
    from socutils.hessian.rhf_fast import _load_socdft
    libdft = _load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTcosx_k'):
        return None
    libdft.SOCDFTcosx_build.restype = ctypes.c_void_p
    return libdft


def available(mol):
    libdft = _lib()
    return (libdft is not None and not mol.cart
            and mol._bas[:, 1].max() <= libdft.SOCDFTcosx_lmax())


def make_grids(mol, **kwargs):
    '''the COSX grid (socutils.dft.grids.Grids with GRID, kwargs override), built with non0tab'''
    from socutils.dft import grids as sgrids
    sgrids.use_fast_grid_sort()
    g = sgrids.Grids(mol)
    g.verbose = mol.verbose
    for k, v in dict(GRID, **kwargs).items():
        setattr(g, k, v)
    g.build(with_non0tab=True)
    return g


class COSX(lib.StreamObject):
    def __init__(self, mol, grids=None, fit=True, tol=TOL):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.grids = grids
        self.fit = fit
        self.tol = tol
        self.ao_cutoff = AO_CUTOFF
        self.tcut = TCUT
        self.blk = BLK
        self._opt = None
        self._key = None
        self._proj = None
        self._libdft = _lib()

    def __del__(self):
        self._free()

    def _free(self):
        if self._opt is not None and self._libdft is not None:
            self._libdft.SOCDFTcosx_free(ctypes.c_void_p(self._opt))
        self._opt = None

    def build(self):
        mol = self.mol
        key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
        if self._key == key and self._opt is not None:
            return self
        t0 = (logger.process_clock(), logger.perf_counter())
        self._free()
        libdft = _lib()
        if self.grids is None or self._key is not None:
            self.grids = make_grids(mol)
        elif self.grids.coords is None or self.grids.non0tab is None:
            self.grids.build(with_non0tab=True)
        ao_loc = mol.ao_loc_nr().astype(numpy.int32)
        self._env = (numpy.asarray(mol._atm, dtype=numpy.int32, order='C'),
                     numpy.asarray(mol._bas, dtype=numpy.int32, order='C'),
                     numpy.asarray(mol._env, dtype=numpy.double, order='C'), ao_loc)
        atm, bas, env, _ = self._env
        self._opt = libdft.SOCDFTcosx_build(
            ctypes.c_double(PRIM_TOL), ctypes.c_double(PAIR_TOL), ao_loc.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p))
        if not self._opt:
            raise RuntimeError('COSX: l > %d or a shell of too many functions' % libdft.SOCDFTcosx_lmax())
        info = numpy.zeros(3)
        libdft.SOCDFTcosx_info(ctypes.c_void_p(self._opt), info.ctypes.data_as(ctypes.c_void_p))
        self._coords = numpy.ascontiguousarray(self.grids.coords.T)
        self._weights = numpy.ascontiguousarray(self.grids.weights)
        self._non0 = numpy.ascontiguousarray(self.grids.non0tab, dtype=numpy.uint8)
        self._proj = self._overlap_fit() if self.fit else None
        self._key = key
        logger.info(self, 'COSX: %d grid points, %d shell pairs, %d primitive pairs, E table %.0f MB',
                    self._weights.size, info[0], info[1], info[2] * 8e-6)
        logger.timer(self, 'COSX build', *t0)
        return self

    def _overlap_fit(self):
        '''S S_g^-1, S_g the overlap on the grid'''
        from socutils.dft.vxc_fast import eval_ao
        from pyscf.dft import numint
        mol = self.mol
        ni = numint.NumInt()
        sg = numpy.zeros((mol.nao, mol.nao))
        coords, w = self.grids.coords, self.grids.weights
        blk = 56 * 64
        for g0 in range(0, w.size, blk):
            g1 = min(w.size, g0 + blk)
            ao = eval_ao(ni, mol, coords[g0:g1], 0, non0tab=self.grids.non0tab[g0 // 56:])
            lib.dot(ao.T * w[g0:g1], ao, beta=1, c=sg)
        s = mol.intor_symmetric('int1e_ovlp')
        self._sg = sg
        return numpy.linalg.solve(sg.T, s.T).T          # s sg^-1 (sg symmetric)

    def get_k(self, dm, hermi=1):
        '''K of real densities dm ((nset,) nao, nao); hermi=1: symmetrised'''
        self.build()
        libdft = _lib()
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        dm = numpy.asarray(dm, dtype=numpy.double)
        shape = dm.shape
        nao = shape[-1]
        dms = numpy.ascontiguousarray(dm.reshape(-1, nao, nao))
        nset = dms.shape[0]
        vk = numpy.empty_like(dms)
        atm, bas, env, ao_loc = self._env
        t0 = (logger.process_clock(), logger.perf_counter())
        libdft.SOCDFTcosx_k(
            vk.ctypes.data_as(ctypes.c_void_p), dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nset),
            ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
            self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._weights.size), ctypes.c_int(self.blk), ctypes.c_double(self.ao_cutoff),
            ctypes.c_double(self.tol), ctypes.c_double(self.tcut),
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
            *_blas_ptrs(), None, None)
        if self._proj is not None:
            vk = lib.einsum('ij,xjk->xik', self._proj, vk)
        if hermi == 1:
            vk = (vk + vk.transpose(0, 2, 1)) * .5
        logger.timer_debug1(self, 'COSX K (%d densities)' % nset, *t0)
        return vk.reshape(shape)


    def energy(self, dm, tol=None):
        '''sum_mu nu D_mu nu K[D]_mu nu (= -4 E_x / hyb for a closed-shell density) of real densities
        ((nset,) nao, nao), without K: the pairs screened by the densities on both sides (tol, default
        self.tol)'''
        self.build()
        libdft = self._libdft
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        dm = numpy.asarray(dm, dtype=numpy.double)
        nao = dm.shape[-1]
        dms = numpy.ascontiguousarray(dm.reshape(-1, nao, nao))
        dmt = dms.transpose(0, 2, 1)
        if self._proj is not None:
            # tr(D P K) = tr((P^T D) K): the fit on the grid side
            dmh = numpy.ascontiguousarray(lib.einsum('ki,xkj->xij', self._proj, dmt))
        else:
            dmh = numpy.ascontiguousarray(dmt)
        out = numpy.zeros(dms.shape[0])
        atm, bas, env, ao_loc = self._env
        libdft.SOCDFTcosx_k(
            lib.c_null_ptr(), dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(dms)),
            ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
            self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._weights.size), ctypes.c_int(self.blk), ctypes.c_double(self.ao_cutoff),
            ctypes.c_double(self.tol if tol is None else tol), ctypes.c_double(self.tcut),
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
            *_blas_ptrs(), dmh.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
        return out[0] if dm.ndim == 2 else out


    def _plain_k(self, dms):
        """the plain K_plain[X] (unsymmetrised, no fit) of real matrices X ((nset,) nao, nao)"""
        proj, self._proj = self._proj, None
        try:
            return self.get_k(dms, hermi=0)
        finally:
            self._proj = proj

    def get_k_var(self, dm):
        """the Fock K of the variational COSX: the derivative of the energy tr(D P K_plain[D]) / 2,
        1/2 sym(P K_plain[D] + K_plain[P^T D]) (one pass, the integrals shared by both densities);
        the one-sided fitted K alone is not the derivative of its energy (fit=False: the two agree)"""
        self.build()
        dm = numpy.asarray(dm, dtype=numpy.double)
        if self._proj is None:
            return self.get_k(dm)
        nao = dm.shape[-1]
        dms = dm.reshape(-1, nao, nao)
        n = len(dms)
        m = lib.einsum('ki,xkj->xij', self._proj, dms)                  # P^T D
        kp = self._plain_k(numpy.concatenate([dms, m]))
        vk = .5 * (lib.einsum('ij,xjk->xik', self._proj, kp[:n]) + kp[n:])
        vk = (vk + vk.transpose(0, 2, 1)) * .5
        return vk.reshape(dm.shape)

    def grad(self, dm, tol=None):
        """d/dR of tr(D P K_plain[D]) (fit) or tr(D K_plain[D]) at fixed D (symmetric) on the fixed grid
        (no grid response), (natm, 3); with the fit the derivatives of S and of S_g in P = S S_g^-1"""
        self.build()
        libdft = self._libdft
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        atm, bas, env, ao_loc = self._env
        if not libdft.SOCDFTcosx_build_grad(ctypes.c_void_p(self._opt), atm.ctypes.data_as(ctypes.c_void_p),
                                           bas.ctypes.data_as(ctypes.c_void_p), env.ctypes.data_as(ctypes.c_void_p)):
            raise MemoryError('COSX gradient tables')
        dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
        de = numpy.zeros((mol.natm, 3))
        yg = None
        if self._proj is not None:
            P = self._proj
            m = numpy.ascontiguousarray(P.T @ dm)
            kp = self._plain_k(dm)
            z = dm @ kp.T
            y1 = numpy.linalg.solve(self._sg, z.T).T                    # Z S_g^-1
            y2 = P.T @ y1
            # the overlap: dS_mu nu / dR_X = -<d mu|nu> (mu on X) - <mu|d nu> (nu on X)
            ip = mol.intor('int1e_ipovlp', comp=3)
            y1s = y1 + y1.T
            aoslices = mol.aoslice_by_atom()
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia, 2:]
                de[ia] -= numpy.einsum('xij,ij->x', ip[:, p0:p1], y1s[p0:p1])
            yg = numpy.ascontiguousarray(-(y2 + y2.T))
        else:
            m = dm
        g = numpy.zeros((mol.natm, 3))
        libdft.SOCDFTcosx_grad(
            g.ctypes.data_as(ctypes.c_void_p), dm.ctypes.data_as(ctypes.c_void_p), m.ctypes.data_as(ctypes.c_void_p),
            lib.c_null_ptr() if yg is None else yg.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
            self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._weights.size), ctypes.c_int(self.blk), ctypes.c_double(self.ao_cutoff),
            ctypes.c_double(self.tol if tol is None else tol), ctypes.c_double(self.tcut),
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
            *_blas_ptrs())
        return de + g


def get_k(mol, dm, hermi=1, grids=None, fit=True):
    return COSX(mol, grids, fit).get_k(dm, hermi)


def stats(reset=True):
    '''the kernel's profiling counters: thread-summed seconds of [AO + F, pairs (integrals + contraction), -, K GEMM],
    with -DCX_PROFILE also counts [pairs, primitive pairs, far-field terms, sum of nl] and cycles'''
    out = numpy.zeros(32)
    _lib().SOCDFTcosx_stats(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(int(reset)))
    return out


TOL_ENERGY = 1e-9       # screening of the final-grid exchange energy
VARIATIONAL = True      # the SCF's Fock K the derivative of the COSX energy (for analytic gradients)
REBUILD_NSTEPS = 30     # incremental K builds between two full builds (a late full build is a whole K for nothing)
TOL_FULL = 1e-7         # screening of the full builds: their error is a fixed offset the increments do not
                        # touch (c16 / cc-pVDZ: E_x changes by 1e-6 against 1e-9; the increments keep tol)
SCF_GRID_SWITCH = 0     # max |D - D_prev| below which the SCF goes from GRIDX1 to GRIDX2 (0: GRIDX2 throughout;
                        # c16 / cc-pVDZ PBE0 with 1e-3: 18 cycles instead of 12, the grid switch upsets DIIS)


def apply(mf, grids=None, final_grid=3, incremental=True, fit=True, tol=None, variational=VARIATIONAL):
    """mf (RHF / RKS, hybrid or HF) with K from COSX: J as before (RIJ through
    socutils.scf.md_j for density_fit(only_dfj=True), else mf's own), K by
    cosx on GRIDX1 / GRIDX2 (``grids``: one grid, or None for the switch at
    SCF_GRID_SWITCH), incremental builds K[D] = K[D_prev] + K[D - D_prev]
    (pyscf's own incremental Fock build switched off, mf.direct_scf = False)
    with a full build every REBUILD_NSTEPS and at the grid switch;
    final_grid: after the SCF the exchange energy is recomputed on that
    grid (ORCA's GRIDX3) and mf.e_tot corrected (None: not).  Range-separated
    calls, complex densities and other molecules keep mf's get_jk.
    variational: the Fock K is the derivative of the COSX energy (COSX.get_k_var,
    +~20 % per build), so that the SCF is stationary for the energy whose analytic
    gradient grad_k computes; otherwise the one-sided fitted K (ORCA's).  Modifies
    and returns mf."""
    from pyscf.df import df_jk
    from socutils.scf import md_j
    mol0 = mf.mol
    if not available(mol0):
        return mf
    if isinstance(mf, df_jk._DFHF) and mf.only_dfj:
        mf = md_j.apply(mf)
    orig_jk = mf.get_jk
    mf.direct_scf = False
    if tol is None:
        # the skipped contributions of the incremental builds put a floor under the SCF gradient
        # (c16 / def2-TZVP PBE0 at 1e-7: |g| stalls at 7e-5); tol <= conv_tol / 10
        tol = min(TOL, .1 * getattr(mf, 'conv_tol', 1e-9))
    st = {'k': {}, 'last': None}

    def kobj(level):
        if level not in st['k']:
            g = grids if isinstance(grids, lib.StreamObject) else make_grids(mol0, **GRIDX[level])
            k = COSX(mol0, g, fit=fit, tol=tol)
            k.verbose = mf.verbose
            k.stdout = mf.stdout
            st['k'][level] = k.build()
        return st['k'][level]

    def get_k(dm, hermi):
        dm = numpy.asarray(dm)
        if not incremental or dm.ndim != 2 or hermi != 1:
            lev = 2 if grids is None else 0
            k = kobj(lev)
            return k.get_k_var(dm) if variational and hermi == 1 else k.get_k(dm, hermi)
        last = st['last']
        if grids is not None and not isinstance(grids, lib.StreamObject):
            lev = int(grids)
        elif isinstance(grids, lib.StreamObject):
            lev = 0
        elif last is None:
            lev = 1 if SCF_GRID_SWITCH > 0 else 2
        else:
            lev = last['lev']
            if lev == 1 and abs(dm - last['dm']).max() < SCF_GRID_SWITCH:
                lev = 2
        k = kobj(lev)
        if last is not None and last['lev'] == lev and last['n'] < REBUILD_NSTEPS:
            ddm = dm - last['dm']
            vk = last['vk'] + (k.get_k_var(ddm) if variational else k.get_k(ddm, hermi))
            n = last['n'] + 1
            logger.debug(mf, 'COSX: incremental K (%d since the full build), grid %d', n, lev)
        else:
            tol0 = k.tol
            k.tol = max(tol0, TOL_FULL)
            try:
                vk = k.get_k_var(dm) if variational else k.get_k(dm, hermi)
            finally:
                k.tol = tol0
            n = 0
            logger.debug(mf, 'COSX: full K, grid %d', lev)
        st['last'] = {'lev': lev, 'dm': dm.copy(), 'vk': vk.copy(), 'n': n}
        return vk

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if mol is None:
            mol = mol0
        if dm is None:
            dm = mf.make_rdm1()
        if not with_k or omega is not None or mol is not mol0 or numpy.iscomplexobj(dm):
            return orig_jk(mol, dm, hermi, with_j, with_k, omega)
        t0 = (logger.process_clock(), logger.perf_counter())
        vj = orig_jk(mol, dm, hermi, True, False, omega)[0] if with_j else None
        t1 = logger.timer_debug1(mf, 'vj', *t0)
        vk = get_k(dm, hermi)
        logger.timer_debug1(mf, 'vk (COSX)', *t1)
        return vj, vk

    mf.get_jk = get_jk
    st['variational'] = variational
    mf._cosx = st
    if final_grid is not None:
        kernel0 = mf.kernel

        def kernel(*args, **kwargs):
            st['last'] = None
            out = kernel0(*args, **kwargs)
            last = st['last']
            if last is None or last['lev'] == final_grid:
                return out
            t0 = (logger.process_clock(), logger.perf_counter())
            # e_tot is that of the density of the last Fock build, whose K st['last'] holds
            dm, k_scf = last['dm'], last['vk']
            # energy only, the pairs screened by the density on both sides (c16 / def2-TZVP GRIDX3:
            # 4.5 s against 8.5 s for K, tr(DK) within 1e-5)
            e_fin = kobj(final_grid).energy(dm, tol=TOL_ENERGY)
            hyb = 1.
            if hasattr(mf, 'xc'):
                hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)[2]
            de = -.25 * hyb * (e_fin - numpy.einsum('ij,ji', dm, k_scf))
            mf._cosx_de = de
            logger.info(mf, 'COSX: exchange energy recomputed on GRIDX%d: change %.9f, E = %.12f',
                        final_grid, de, mf.e_tot + de)
            logger.timer(mf, 'COSX final grid', *t0)
            mf.e_tot = mf.e_tot + de
            return mf.e_tot
        mf.kernel = kernel
    return mf


def grad_k(mf, dm=None):
    """the exchange part of the nuclear gradient, (natm, 3), of an SCF run through apply():
    d/dR of -(hyb / 4) tr(D P K_plain[D]) on the SCF's COSX grid (no grid response), the
    derivatives of the integrals, of the AOs on the grid and of the overlap fit; None if mf
    did not run COSX.  Exact for the variational SCF (apply(variational=True))."""
    st = getattr(mf, '_cosx', None)
    if not st or st.get('last') is None:
        return None
    k = st['k'][st['last']['lev']]
    if dm is None:
        dm = mf.make_rdm1()
    hyb = 1.
    if hasattr(mf, 'xc'):
        hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)[2]
    if not st.get('variational', False):
        logger.warn(mf, 'COSX gradient of a non-variational SCF (apply(variational=False)): '
                    'the orbital response is not exact')
    return -.25 * hyb * k.grad(numpy.asarray(dm))
