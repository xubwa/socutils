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

# the COSX grid: dft.grids.Grids attributes (intacc, angular_grid); see the module docstring
GRID = {'intacc': 3.467, 'angular_grid': 1}
BLK = 112               # grid points per task of the kernel (a multiple of 56)
AO_CUTOFF = 1e-10       # AO values below which an AO is dropped from a block
TOL = 1e-11             # screening of the shell-pair contributions, B_IJ max|F_J| sum_g |w phi| < TOL
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

    def __del__(self):
        self._free()

    def _free(self):
        if self._opt is not None:
            libdft = _lib()
            if libdft is not None:
                libdft.SOCDFTcosx_free(ctypes.c_void_p(self._opt))
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
            *_blas_ptrs())
        if self._proj is not None:
            vk = lib.einsum('ij,xjk->xik', self._proj, vk)
        if hermi == 1:
            vk = (vk + vk.transpose(0, 2, 1)) * .5
        logger.timer_debug1(self, 'COSX K (%d densities)' % nset, *t0)
        return vk.reshape(shape)


def get_k(mol, dm, hermi=1, grids=None, fit=True):
    return COSX(mol, grids, fit).get_k(dm, hermi)


def stats(reset=True):
    '''the kernel's profiling counters: thread-summed seconds of [AO + F, integrals, contraction, K GEMM],
    counts [pairs, primitive pairs, far-field terms, sum of nl]'''
    out = numpy.zeros(16)
    _lib().SOCDFTcosx_stats(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(int(reset)))
    return out
