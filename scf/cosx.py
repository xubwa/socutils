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
    vk = k.get_k(dm)                    # dm (nao, nao) or (nset, nao, nao); many densities
                                        # share the integrals (the CPHF K, hessian.rhf_fast)
    e = k.energy(dm)                    # tr(D K) without K (screened by D on both sides)
    g = k.grad(dm)                      # d tr(D K) / dR at fixed D (no grid response)
    mf = cosx.apply(scf.RKS(mol, xc='pbe0').density_fit(only_dfj=True))
                                        # RIJ J (md_j) + COSX K in the SCF, the final
                                        # exchange energy on GRIDX3
    socutils.grad.rhf_fast.Gradients(mf).kernel()   # the K part from grad_k

The one-sided fit makes the bilinear form tr(D1 K[D2]) unsymmetric, so its K
is not the derivative of the energy tr(D K[D]); apply(variational=True) uses
that derivative (get_k_var) and the SCF gradient is then exact for the COSX
energy on the grid.
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
MULTI_MIN = 4           # densities from which the multi-density contraction runs
MULTI_FLOAT = False     # its contraction in single precision (set_multi_float)
HESS_FLOAT = False      # the Hessian passes' A C contraction (Fock1 / K second derivatives) in single precision
PROFILE = {}            # wall seconds of get_k's phases (accumulated)
MULTI_MAXSET = 512      # densities per pass (every pass computes the integrals again)
PRIM_TOL = 1e-14
PAIR_TOL = 1e-13


def _lib():
    from socutils.hessian.rhf_fast import _load_socdft
    libdft = _load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTcosx_k'):
        return None
    libdft.SOCDFTcosx_build.restype = ctypes.c_void_p
    if not getattr(libdft, '_cx_symm_set', False) and hasattr(libdft, 'SOCDFTcosx_set_symm'):
        # dsymm_ / ssymm_ of the BLAS pyscf uses (MKL) for the dense multi-density contraction (else the
        # record-wise one)
        try:
            for name in ('libmkl_rt.so.3', 'libmkl_rt.so.2', 'libmkl_rt.so'):
                try:
                    blas = ctypes.CDLL(name)
                    libdft.SOCDFTcosx_set_symm(ctypes.cast(blas.dsymm_, ctypes.c_void_p),
                                               ctypes.cast(blas.ssymm_, ctypes.c_void_p))
                    if hasattr(libdft, 'SOCDFTcosx_set_sgemm'):
                        libdft.SOCDFTcosx_set_sgemm(ctypes.cast(blas.sgemm_, ctypes.c_void_p))
                    break
                except (OSError, AttributeError):
                    continue
        finally:
            libdft._cx_symm_set = True
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

    def get_k(self, dm, hermi=1, _onepass=False):
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
        atm, bas, env, ao_loc = self._env
        t0 = (logger.process_clock(), logger.perf_counter())
        multi = nset >= MULTI_MIN
        if multi:
            # many densities: the densities as the vector dimension of the contraction, in as few passes as
            # the memory allows (every pass evaluates the integrals again).  Per density: the threads' F, G
            # (float: in float only), 2 blk nao each; here two nao^2 arrays alive at a time (the transposed
            # input and output, then the output and the result).  Fixed: the threads' stored pair integrals
            # of a block (dense contraction), at most nao^2/2 blk each
            nt = lib.num_threads()
            per = ((4 if libdft.SOCDFTcosx_mfloat() else 12) * 2 * self.blk * nao * nt + 8 * 2.5 * nao * nao) / 1e6
            if getattr(libdft, 'SOCDFTcosx_kpriv', None) is not None and libdft.SOCDFTcosx_kpriv():
                per += 8 * nao * nao * nt / 1e6         # the threads' private K
            fixed = (4 if libdft.SOCDFTcosx_mfloat() else 8) * (nao * nao / 2 * self.blk + nao * nao) * nt / 1e6
            # (at least 2 GB: a pass costs the integrals of the whole grid, ~25 densities' worth on c16 TZVP)
            avail = max(2000, mol.max_memory - lib.current_memory()[0] - fixed)
            nb = max(MULTI_MIN, min(nset, MULTI_MAXSET, int(.8 * avail / per)))
            if nb < nset and not _onepass:
                npass = (nset + nb - 1) // nb
                nb = (nset + npass - 1) // npass        # equal passes
                vk = numpy.concatenate([self.get_k(dms[i:i + nb], hermi, _onepass=True) for i in range(0, nset, nb)])
                return vk.reshape(shape)
            PROFILE['passes'] = PROFILE.get('passes', 0) + 1
            PROFILE['densities'] = PROFILE.get('densities', 0) + nset
            tp_ = logger.perf_counter()
            vl = 16 if libdft.SOCDFTcosx_mfloat() else 8
            nset8 = (nset + vl - 1) // vl * vl          # the kernel takes the densities 8 (16) at a time
            dtm = numpy.empty((nao, nao, nset8))
            libdft.SOCDFTcosx_transpose_in(dtm.ctypes.data_as(ctypes.c_void_p), dms.ctypes.data_as(ctypes.c_void_p),
                                           ctypes.c_int(nao), ctypes.c_int(nset), ctypes.c_int(nset8))
            vkt = numpy.empty((nao, nao, nset8))
            PROFILE['prep'] = PROFILE.get('prep', 0) + logger.perf_counter() - tp_
        tk_ = logger.perf_counter()
        vk = None if multi else numpy.empty_like(dms)
        libdft.SOCDFTcosx_k(
            (vkt if multi else vk).ctypes.data_as(ctypes.c_void_p), dms.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nset8 if multi else nset),
            ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
            self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._weights.size), ctypes.c_int(self.blk), ctypes.c_double(self.ao_cutoff),
            ctypes.c_double(self.tol), ctypes.c_double(self.tcut),
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
            *_blas_ptrs(), None, None, dtm.ctypes.data_as(ctypes.c_void_p) if multi else None)
        if libdft.SOCDFTcosx_error():
            raise MemoryError('COSX: a kernel buffer could not be allocated (%d densities, nao %d)' % (nset, nao))
        tp_ = logger.perf_counter()
        PROFILE['kernel'] = PROFILE.get('kernel', 0) + tp_ - tk_
        if multi:
            dtm = None
            vk = numpy.empty((nset, nao, nao))
            libdft.SOCDFTcosx_untranspose(vk.ctypes.data_as(ctypes.c_void_p), vkt.ctypes.data_as(ctypes.c_void_p),
                                          ctypes.c_int(nao), ctypes.c_int(nset), ctypes.c_int(nset8))
            vkt = None
        tq_ = logger.perf_counter()
        PROFILE['untranspose'] = PROFILE.get('untranspose', 0) + tq_ - tp_
        if self._proj is not None:
            vk = lib.einsum('ij,xjk->xik', self._proj, vk)
        tp_ = logger.perf_counter()
        PROFILE['proj'] = PROFILE.get('proj', 0) + tp_ - tq_
        if hermi == 1:
            vk = lib.hermi_sum(vk, axes=(0, 2, 1), inplace=True)
            vk *= .5
        PROFILE['sym'] = PROFILE.get('sym', 0) + logger.perf_counter() - tp_
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
            *_blas_ptrs(), dmh.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), None)
        return out[0] if dm.ndim == 2 else out


    def _plain_k(self, dms):
        """the plain K_plain[X] (unsymmetrised, no fit) of real matrices X ((nset,) nao, nao)"""
        proj, self._proj = self._proj, None
        try:
            return self.get_k(dms, hermi=0)
        finally:
            self._proj = proj

    def get_k_occ(self, dms, mocc):
        """K C_occ of many real densities ((nset,) nao, nao) (nset, nao, nocc), the one-sided fitted
        P K_plain[D] C (not symmetrised): A C per grid point shared by all densities and Aoc^T F per
        density, no K matrix; for the CPHF, whose response needs only K C_occ"""
        self.build()
        libdft = self._libdft
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        dms = numpy.asarray(dms, dtype=numpy.double)
        nao = dms.shape[-1]
        dms = dms.reshape(-1, nao, nao)
        nset = len(dms)
        nocc = mocc.shape[1]
        vl = 16 if libdft.SOCDFTcosx_mfloat() else 8
        no8 = (nocc + vl - 1) // vl * vl
        ns8 = (nset + vl - 1) // vl * vl
        # per density and thread: the gathered rows (nao^2) and K C (nao nocc), F (nao blk)
        per = 8 * ((nao * nao + nao * no8 + 2 * nao * self.blk) * lib.num_threads() + 3 * nao * nao) / 1e6
        nb = max(vl, min(ns8, int(.4 * max(500, mol.max_memory - lib.current_memory()[0]) / per) // vl * vl))
        if nb < nset:
            return numpy.concatenate([self.get_k_occ(dms[i:i + nb], mocc) for i in range(0, nset, nb)])
        co = numpy.zeros((nao, no8))
        co[:, :nocc] = mocc
        dtm = numpy.zeros((nao, nao, ns8))
        dtm[:, :, :nset] = dms.transpose(1, 2, 0)
        out = numpy.empty((nao, no8, ns8))
        atm, bas, env, ao_loc = self._env
        libdft.SOCDFTcosx_set_occ(co.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(no8))
        try:
            libdft.SOCDFTcosx_k(
                out.ctypes.data_as(ctypes.c_void_p), dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ns8),
                ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
                self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self._weights.size), ctypes.c_int(self.blk), ctypes.c_double(self.ao_cutoff),
                ctypes.c_double(self.tol), ctypes.c_double(self.tcut),
                ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
                *_blas_ptrs(), None, None, dtm.ctypes.data_as(ctypes.c_void_p))
        finally:
            libdft.SOCDFTcosx_set_occ(None, ctypes.c_int(0))
        if libdft.SOCDFTcosx_error():
            raise MemoryError('COSX: a kernel buffer could not be allocated (%d densities, nao %d)' % (nset, nao))
        kc = out[:, :nocc, :nset].transpose(2, 0, 1)
        if self._proj is not None:
            kc = lib.einsum('ij,xjk->xik', self._proj, kc)
        return numpy.ascontiguousarray(kc)

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

    def _ipsg(self):
        """sum_g w d_x phi_mu phi_nu on the grid (3, nao, nao), cached per geometry"""
        if getattr(self, '_ipsg_key', None) != self._key:
            from socutils.dft.vxc_fast import eval_ao
            from pyscf.dft import numint
            mol = self.mol
            ni = numint.NumInt()
            out = numpy.zeros((3, mol.nao, mol.nao))
            coords, w = self.grids.coords, self.grids.weights
            blk = 56 * 32
            for g0 in range(0, w.size, blk):
                g1 = min(w.size, g0 + blk)
                ao = eval_ao(ni, mol, coords[g0:g1], 1, non0tab=self.grids.non0tab[g0 // 56:])
                wao = ao[0] * w[g0:g1, None]
                for x in range(3):
                    lib.dot(ao[1 + x].T, wao, beta=1, c=out[x])
            self._ipsg_v = out
            self._ipsg_key = self._key
        return self._ipsg_v

    def _fock1_plain(self, mocc, yocc, tol=None):
        """V = d K_plain[D] / dR C, VT = (d K_plain[D] / dR)^T Y (natm, 3, nao, nocc), D = 2 C C^T, at
        fixed D on the fixed grid"""
        self.build()
        libdft = self._libdft
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        atm, bas, env, ao_loc = self._env
        if not libdft.SOCDFTcosx_build_grad(ctypes.c_void_p(self._opt), atm.ctypes.data_as(ctypes.c_void_p),
                                           bas.ctypes.data_as(ctypes.c_void_p), env.ctypes.data_as(ctypes.c_void_p)):
            raise MemoryError('COSX gradient tables')
        nao, nocc = mocc.shape
        vl = 16 if HESS_FLOAT and libdft.SOCDFTcosx_hfloat is not None else 8
        libdft.SOCDFTcosx_set_hfloat(int(vl == 16))
        n8 = (nocc + vl - 1) // vl * vl
        co = numpy.zeros((nao, n8))
        co[:, :nocc] = mocc
        yo = numpy.zeros((nao, n8))
        yo[:, :nocc] = yocc
        v = numpy.empty((mol.natm, 3, nao, n8))
        vt = numpy.empty((mol.natm, 3, nao, n8))
        err = libdft.SOCDFTcosx_fock1(
            v.ctypes.data_as(ctypes.c_void_p), vt.ctypes.data_as(ctypes.c_void_p), co.ctypes.data_as(ctypes.c_void_p),
            yo.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n8), ctypes.c_void_p(self._opt),
            self._coords.ctypes.data_as(ctypes.c_void_p), self._weights.ctypes.data_as(ctypes.c_void_p),
            self._non0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self._weights.size), ctypes.c_int(self.blk),
            ctypes.c_double(self.ao_cutoff), ctypes.c_double(self.tol if tol is None else tol),
            ctypes.c_double(self.tcut), ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p), *_blas_ptrs())
        if err:
            raise MemoryError('COSX Fock1: a kernel buffer could not be allocated')
        return v[..., :nocc], vt[..., :nocc]

    def _dproj(self, ia, x, ip, ipsg, aoslices):
        """dS and dS_g with respect to R_{ia,x} (nao, nao)"""
        p0, p1 = aoslices[ia, 2:]
        ds = numpy.zeros_like(ip[0])
        ds[p0:p1] -= ip[x, p0:p1]
        ds[:, p0:p1] -= ip[x, p0:p1].T
        dsg = numpy.zeros_like(ip[0])
        dsg[p0:p1] -= ipsg[x, p0:p1]
        dsg[:, p0:p1] -= ipsg[x, p0:p1].T
        return ds, dsg

    def fock1(self, mocc, tol=None):
        """d K / dR_{A,x} C (natm, 3, nao, nocc) of the K get_k returns (sym(P K_plain[D]), D = 2 C C^T)
        at fixed D on the fixed grid (no grid response), the derivatives of S and S_g in P included"""
        mocc = numpy.ascontiguousarray(mocc)
        mol = self.mol
        self.build()
        if self._proj is None:
            v, vt = self._fock1_plain(mocc, mocc, tol)
            return .5 * (v + vt)
        P = self._proj
        Q = numpy.linalg.inv(self._sg)
        Q = (Q + Q.T) * .5
        y = numpy.ascontiguousarray(P.T @ mocc)
        v, vt = self._fock1_plain(mocc, y, tol)
        dm = 2 * mocc @ mocc.T
        kp = self._plain_k(dm)
        u = Q @ (kp @ mocc)
        kq = kp.T @ Q
        ip = mol.intor('int1e_ipovlp', comp=3)
        ipsg = self._ipsg()
        aoslices = mol.aoslice_by_atom()
        out = numpy.empty_like(v)
        for ia in range(mol.natm):
            for x in range(3):
                ds, dsg = self._dproj(ia, x, ip, ipsg, aoslices)
                # dX C = dP (K C) + P V, dX^T C = VT + K^T dP^T C;  dP = dS Q - P dS_g Q
                dxc = ds @ u - P @ (dsg @ u) + P @ v[ia, x]
                dxtc = vt[ia, x] + kq @ (ds @ mocc - dsg @ y)
                out[ia, x] = .5 * (dxc + dxtc)
        return out

    def _sg2(self):
        """sum_g w d_x d_y phi_mu phi_nu (6, nao, nao: xx xy xz yy yz zz) and sum_g w d_x phi_mu d_y phi_nu
        (3, 3, nao, nao) on the grid, cached per geometry"""
        if getattr(self, '_sg2_key', None) != self._key:
            from socutils.dft.vxc_fast import eval_ao
            from pyscf.dft import numint
            mol = self.mol
            ni = numint.NumInt()
            nao = mol.nao
            gaa = numpy.zeros((6, nao, nao))
            gab = numpy.zeros((3, 3, nao, nao))
            coords, w = self.grids.coords, self.grids.weights
            blk = 56 * 16
            for g0 in range(0, w.size, blk):
                g1 = min(w.size, g0 + blk)
                ao = eval_ao(ni, mol, coords[g0:g1], 2, non0tab=self.grids.non0tab[g0 // 56:])
                wao = ao[0] * w[g0:g1, None]
                for c in range(6):
                    lib.dot(ao[4 + c].T, wao, beta=1, c=gaa[c])
                for x in range(3):
                    wx = ao[1 + x] * w[g0:g1, None]
                    for y in range(3):
                        lib.dot(wx.T, ao[1 + y], beta=1, c=gab[x, y])
            self._sg2_v = (gaa, gab)
            self._sg2_key = self._key
        return self._sg2_v

    @staticmethod
    def _d2_contract(mol, aa, ab, xs):
        """sum_mu nu d^2 M_mu nu / dR_a dR_b xs_mu nu for a symmetric xs and a matrix M of AO pairs whose
        second derivatives are <d d mu|nu> (aa, (3, 3) or (6) packed) and <d mu|d nu> (ab (3, 3)) as pyscf's
        overlap: 2 [delta_ab sum_{mu in a} aa xs + sum_{mu in a, nu in b} ab xs], (natm, natm, 3, 3)"""
        natm = mol.natm
        out = numpy.zeros((natm, natm, 3, 3))
        aoslices = mol.aoslice_by_atom()
        if aa.ndim == 3:
            idx = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
            full = numpy.empty((3, 3) + aa.shape[1:])
            for (x, y), c in idx.items():
                full[x, y] = full[y, x] = aa[c]
            aa = full
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            out[ia, ia] += 2 * numpy.einsum('xypq,pq->xy', aa[:, :, p0:p1], xs[p0:p1])
            for ja in range(natm):
                q0, q1 = aoslices[ja, 2:]
                out[ia, ja] += 2 * numpy.einsum('xypq,pq->xy', ab[:, :, p0:p1, q0:q1], xs[p0:p1, q0:q1])
        return out

    def _hess_plain(self, mocc, yocc, with_fock1=False, tol=None):
        """T1 = d^2 / dR dR' sum_g w H^T A F at fixed M = 2 Y C^T, D = 2 C C^T ((3 natm, 3 natm)), and with
        with_fock1 the plain V, VT of _fock1_plain from the same pass"""
        self.build()
        libdft = self._libdft
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        atm, bas, env, ao_loc = self._env
        if not libdft.SOCDFTcosx_build_hess(ctypes.c_void_p(self._opt), atm.ctypes.data_as(ctypes.c_void_p),
                                           bas.ctypes.data_as(ctypes.c_void_p), env.ctypes.data_as(ctypes.c_void_p)):
            raise MemoryError('COSX Hessian tables')
        nao, nocc = mocc.shape
        vl = 16 if HESS_FLOAT and libdft.SOCDFTcosx_hfloat is not None else 8
        libdft.SOCDFTcosx_set_hfloat(int(vl == 16))
        n8 = (nocc + vl - 1) // vl * vl
        co = numpy.zeros((nao, n8))
        co[:, :nocc] = mocc
        yo = numpy.zeros((nao, n8))
        yo[:, :nocc] = yocc
        hs = numpy.empty((3 * mol.natm, 3 * mol.natm))
        v = vt = None
        if with_fock1:
            v = numpy.empty((mol.natm, 3, nao, n8))
            vt = numpy.empty((mol.natm, 3, nao, n8))
        ptr = lambda x: lib.c_null_ptr() if x is None else x.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        err = libdft.SOCDFTcosx_hess(
            hs.ctypes.data_as(ctypes.c_void_p), ptr(v), ptr(vt), co.ctypes.data_as(ctypes.c_void_p),
            yo.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n8), ctypes.c_void_p(self._opt),
            self._coords.ctypes.data_as(ctypes.c_void_p), self._weights.ctypes.data_as(ctypes.c_void_p),
            self._non0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self._weights.size), ctypes.c_int(self.blk),
            ctypes.c_double(self.ao_cutoff), ctypes.c_double(self.tol if tol is None else tol),
            ctypes.c_double(self.tcut), ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p), *_blas_ptrs())
        if err:
            raise MemoryError('COSX Hessian: a kernel buffer could not be allocated')
        if with_fock1:
            v, vt = v[..., :nocc], vt[..., :nocc]
        return hs, v, vt

    def hess(self, mocc, with_fock1=False, tol=None, sum_rule=True):
        """d^2 tr(D P K_plain[D]) / dR dR' (natm, natm, 3, 3) at fixed D = 2 C C^T on the fixed grid (no grid
        response), the derivatives of S and S_g in the fit P = S S_g^-1 included (fit=False: of
        tr(D K_plain[D])); with with_fock1 also fock1(mocc) from the same kernel pass, (hess, fock1)"""
        mocc = numpy.ascontiguousarray(mocc)
        mol = self.mol
        natm = mol.natm
        self.build()
        if self._proj is None:
            hs, v, vt = self._hess_plain(mocc, mocc, with_fock1, tol)
            h = hs.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
            if sum_rule:
                h = _sum_rule(h)
            return (h, .5 * (v + vt)) if with_fock1 else h
        P = self._proj
        Q = numpy.linalg.inv(self._sg)
        Q = (Q + Q.T) * .5
        y = numpy.ascontiguousarray(P.T @ mocc)
        hs, v, vt = self._hess_plain(mocc, y, True, tol)
        h = hs.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3).copy()
        dm = 2 * mocc @ mocc.T
        kp = self._plain_k(dm)
        u = Q @ (kp @ mocc)
        kq = kp.T @ Q
        ip = mol.intor('int1e_ipovlp', comp=3)
        ipsg = self._ipsg()
        aoslices = mol.aoslice_by_atom()
        n3 = 3 * natm
        nao, nocc = mocc.shape
        lam = numpy.empty((n3, nao, nocc))      # Q dS C
        kap = numpy.empty((n3, nao, nocc))      # Q dS_g Y
        sig = numpy.empty((n3, nao, nocc))      # dS_g u
        f1 = numpy.empty((natm, 3, nao, nocc)) if with_fock1 else None
        for ia in range(natm):
            for x in range(3):
                ds, dsg = self._dproj(ia, x, ip, ipsg, aoslices)
                r = 3 * ia + x
                dsc = ds @ mocc
                dsgy = dsg @ y
                lam[r] = Q @ dsc
                kap[r] = Q @ dsgy
                sig[r] = dsg @ u
                if with_fock1:
                    dxc = ds @ u - P @ sig[r] + P @ v[ia, x]
                    dxtc = vt[ia, x] + kq @ (dsc - dsgy)
                    f1[ia, x] = .5 * (dxc + dxtc)
        def outer(a_, b_):
            m = lib.dot(a_.reshape(n3, -1), b_.reshape(n3, -1).T)
            return m.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
        # T2 + T2^T: 2 sum Lambda_R . V^{R'}, Lambda = Q (dS C - dS_g Y)
        t2 = outer(2 * (lam - kap), v.reshape(n3, nao, nocc))
        h += t2 + t2.transpose(1, 0, 3, 2)
        # T3: (ii) -2 lam_a . sig_b, (iii-a) 2 kap_a . sig_b, both orders
        t3 = outer(-2 * lam + 2 * kap, sig)
        h += t3 + t3.transpose(1, 0, 3, 2)
        # (i): tr(d2S Q Z0), Q Z0 = 2 u C^T;  (iii-b): -2 tr(d2S_g u Y^T)
        from pyscf.hessian import rhf as rhf_hess
        s1aa, s1ab, _ = rhf_hess.get_ovlp(mol)
        x1 = 2 * u @ mocc.T
        h += self._d2_contract(mol, s1aa, s1ab, (x1 + x1.T) * .5)
        gaa, gab = self._sg2()
        x2 = u @ y.T
        h -= 2 * self._d2_contract(mol, gaa, gab, (x2 + x2.T) * .5)
        if sum_rule:
            h = _sum_rule(h)
        return (h, f1) if with_fock1 else h

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


def _sum_rule(h):
    """the one-centre blocks from translational invariance, H_AA = -sum_{B != A} H_AB: on the grid fixed in
    space the second derivatives of the AOs on the grid break the invariance, the error almost entirely
    in H_AA (as the fixed-grid XC Hessian, socutils.dft.grids); the exact K second derivatives at fixed D
    obey it to 1e-14 (H2O / CH3 cc-pVDZ, GRIDX3: max |H/4 - ek| 0.36 without, 1.3e-5 with)"""
    h = numpy.array(h)
    for a in range(h.shape[0]):
        h[a, a] = 0
        h[a, a] = -h[a].sum(axis=0)
    return h


def get_k(mol, dm, hermi=1, grids=None, fit=True):
    return COSX(mol, grids, fit).get_k(dm, hermi)


def set_multi_float(on=True):
    """the multi-density contraction (CPHF K) in single precision (AVX-512 builds)"""
    global MULTI_FLOAT
    MULTI_FLOAT = bool(on)
    _lib().SOCDFTcosx_set_opt(2, int(on))


def stats(reset=True):
    '''the kernel's profiling counters: thread-summed seconds of [AO + F, pairs (integrals + contraction), -, K GEMM],
    with -DCX_PROFILE also counts [pairs, primitive pairs, far-field terms, sum of nl] and cycles'''
    out = numpy.zeros(32)
    _lib().SOCDFTcosx_stats(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(int(reset)))
    return out


TOL_ENERGY = 1e-9       # screening of the final-grid exchange energy
VARIATIONAL = False     # the SCF's Fock K the derivative of the COSX energy: analytic gradients exact to
                        # ~1e-7 (else ~5e-5, ORCA's situation); +15-25 % per K build (c16 / def2-TZVP SCF
                        # 68 s against 55-60 s), so on for geometry optimisations, off for energies
REBUILD_NSTEPS = 30     # incremental K builds between two full builds (a late full build is a whole K for nothing)
TOL_FULL = 1e-7         # screening of the full builds: their error is a fixed offset the increments do not
                        # touch (c16 / cc-pVDZ: E_x changes by 1e-6 against 1e-9; the increments keep tol)
SCF_GRID_SWITCH = 0     # max |D - D_prev| below which the SCF goes from GRIDX1 to GRIDX2 (0: GRIDX2 throughout;
                        # c16 / cc-pVDZ PBE0 with 1e-3: 18 cycles instead of 12, the grid switch upsets DIIS)


def apply(mf, grids=None, final_grid=3, incremental=True, fit=True, tol=None, variational=None):
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
    if variational is None:
        variational = VARIATIONAL
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
