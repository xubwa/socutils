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
from pyscf import gto, lib
from pyscf.dft import radi
from pyscf.lib import logger

# the COSX grids, dft.grids.Grids attributes (intacc, angular_grid): ORCA 6's GRIDX1-3 (IntAcc 3.816 /
# 4.020 / 4.338, Lebedev-50 / 110 / 194 at most); the SCF runs on GRIDX2 (GRIDX1 while the density
# moves, SCF_GRID_SWITCH, off by default), the final exchange energy is recomputed on GRIDX3 (as ORCA)
# level 0: a CP-SCF operator grid (~1/5 of GRIDX2; C16H34 def2-TZVP 11872 points, CP-SCF K only: frequencies
# within 0.22 / rms 0.06 cm-1 of GRIDX2, caffeine cc-pVDZ 0.18; rhf_fast.CPHF_COSX_GRID = 0)
GRIDX = {0: {'intacc': 3.0, 'angular_grid': (14, 14, 26, 26, 14)},
         1: {'intacc': 3.816, 'angular_grid': 1},
         2: {'intacc': 4.020, 'angular_grid': 2},
         3: {'intacc': 4.338, 'angular_grid': 3}}
GRID = GRIDX[2]
HEAVY_Z = 18            # molecules with an atom beyond Z = HEAVY_Z get at least the Lebedev table HEAVY_ANGULAR on
HEAVY_ANGULAR = 3       # every GRIDX level (gridx_args): the tables below 3 leave 26 points in the region of a
                        # 3d shell.  Fe(CO)5 def2-SVP PBE0 frequencies against exact K on a fine grid: 16.6 cm-1
                        # with the plain levels (SCF COSX on GRIDX2 +2.8, CP-SCF operator on level 0 +3.2,
                        # Fock1 / explicit K on GRIDX2 +2.8, the rest the CP-SCF XC grid), 0.01 each with table
                        # 3 at the same intacc; the GRIDX2 exchange energy of Fe is 8.3e-4 Eh off


COSX_RADII_ADJUST = None        # the size adjustment of the COSX grids' Becke cells (make_grids): None -> that of
                                # socutils Grids (dft.grids.polar_pair_atomic_radii_adjust).  Against the Treutler
                                # adjustment for all pairs: K gradient on GRAD_GRID, 1e-5 Eh/bohr median / max,
                                # ionic MX 3.9 / 9.1 -> 0.4 / 2.0, p-block halides and oxides 2.6 / 10.3 -> 1.5 /
                                # 3.3, 3d d0 5.8 / 12 -> 1.9 / 2.7 (carbonyls 1.0 / 1.1 -> 1.5 / 2.3); |dE| of the
                                # default path, ionic 5.6e-5 / 2.3e-4 -> 0.8e-5 / 1.2e-5 Eh; frequencies unchanged
# the pruning boundaries of the COSX grids by element (dft.grids.Grids.alpha_scale): halved for the alkali and
# alkaline-earth metals and Sc, whose Bragg radii put the coarse inner Lebedev tables over the semicore shell and
# the bonding region.  K gradient at a fixed density, 1e-5 Eh/bohr, pruned -> halved, on (4.338, table 3): LiF 7.7
# -> 5.4, KF 34 -> 4.4, RbF 63 -> 6.6, BeF2 11 -> 6.2, MgF2 18 -> 5.4, CaF2 39 -> 17, SrF2 62 -> 19, ScF3 16 -> 10
# (CaO 2.6 -> 5.1, SrO 1.7 -> 4.0, NaCl 9.8 -> 9.2, the hydrides unchanged); on the SCF grid (4.02, table 2) LiF 285
# -> 18, BeF2 38 -> 12, MgF2 51 -> 21, NaCl 33 -> 22; on the K gradient grid (4.338, table 4) LiF 3.9 -> 1.2, RbF 9.1
# -> 3.4, KF 5.1 -> 3.9, MgF2 3.8 -> 3.1.  5 - 16 % more points on these atoms.  Not for Ga, Ge, As, In, Sn, Ti
# (halved, table 3: GaCl3 4.9 -> 25, GeH4 1.6 -> 6.3, AsF3 3.5 -> 18, InCl3 9.9 -> 13, TiCl4 7.2 -> 13).
# With the polar-pair size adjustment on the COSX grids the scaling adds almost nothing (default path, ionic MX:
# 3.4 / 15.9 -> 3.3 / 15.8) for its 5 - 16 % of points: off by default, ALPHA_SCALE = ALPHA_SCALE_SBLOCK turns it on.
ALPHA_SCALE_SBLOCK = {s: .5 for s in ('Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Sc')}
ALPHA_SCALE = {}
GRAD_ONEPASS = True     # COSX.grad: K_plain[D] of the fit terms from the gradient's own integral pass
GRAM_BLOCKS = 16        # 56-point grid blocks per GEMM of the grid Gram matrices (S_g, its derivatives): small
                        # enough that a block sees only the AOs near it (_local_ao_blocks)
GRAD_GRID = 'auto'      # the grid of the K gradient (grad_k): 'auto' (by the accuracy asked for, GRAD_GRIDS), a
                        # GRIDX level, a dict of Grids attributes, or None (the SCF's grid)
GRAD_GRID_DENSE = {'intacc': 4.338, 'angular_grid': 4}  # the dense tier (1.64 x the points of GRIDX3)
# the tiers of GRAD_GRID 'auto': (nominal error of the exchange part of the gradient, max component, Eh/bohr; grid),
# cheapest first; grad_k takes the cheapest tier whose nominal error is <= the accuracy asked for
GRAD_GRIDS = ((3e-4, 2), (1e-5, 3), (1e-6, GRAD_GRID_DENSE))
# Calibration, 87 molecules of the periodic-table set (PBE0 / def2-SVP, displaced geometries), error of the
# exchange part of the gradient against the (6.0, 7) grid, median / max over the molecules, Eh/bohr:
#                 GRIDX2              GRIDX3              dense
#   Z <= 10  (12) 1.0e-4 / 4.8e-4     0.8e-5 / 3.3e-5     1.2e-6 / 4.9e-6
#   Z 11-18  (14) 4.8e-5 / 3.2e-4     1.0e-5 / 4.8e-5     1.5e-6 / 1.8e-5
#   Z 19-36  (27) 2.5e-5 / 1.3e-4     1.6e-5 / 1.2e-4     1.2e-5 / 3.5e-5
#   Z 37-54  (23) 1.5e-5 / 1.1e-4     1.5e-5 / 1.0e-4     3.8e-6 / 2.1e-5
#   Z > 54   (11) 1.4e-5 / 8.5e-5     0.9e-5 / 8.1e-5     7.2e-6 / 1.7e-5
# Beyond Z = 18 GRIDX2 and GRIDX3 carry the same Lebedev table (HEAVY_ANGULAR) and differ by 13 % of the points:
# GRIDX3 gains nothing there (oxides and fluorides up to 1e-4), so for those molecules the GRIDX3 tier is the
# dense grid (GRAD_HEAVY_DENSE), as before.
GRAD_HEAVY_DENSE = True
GRAD_ACCURACY = 1e-5    # the accuracy of a plain gradient: GRIDX3 (ORCA 6 differentiates on its third grid too)
GRAD_CONV_FRACTION = .1 # an optimiser's max-force threshold times this is the accuracy asked for
                        # (set_grad_accuracy(conv=...)), and with auto=True the last gradient's max force times this
                        # while the forces are large
                        # On the SCF's GRIDX2 the K gradient is the
                        # largest error of covalent molecules (HF 2.5e-4, H2O 1.1e-4, CH4 1.0e-4 Eh/bohr; median /
                        # max over 28 hydrides 1.3e-5 / 2.5e-4, 10 ionic MX 3.8e-4 / 2.9e-3); GRIDX3: 0.3e-5 /
                        # 2.5e-5 and 1.5e-4 / 6.3e-4; this grid (1.64 x the points of GRIDX3): 0.1e-5 / 0.4e-5
                        # and 3.9e-5 / 9.1e-5 (ORCA 6 DefGrid3 total: 0.2e-5 / 1.8e-5 and 7.1e-5 / 2.0e-4)


def gridx_args(mol, level):
    '''the Grids attributes of the COSX grid level for mol: GRIDX[level], with the Lebedev table raised to
    HEAVY_ANGULAR when mol has an atom beyond Z = HEAVY_Z'''
    args = dict(GRIDX[level])
    # the element, not mol.atom_charges(): that is the valence charge of an ECP atom (Rb: 9, W: 14)
    if mol.natm and max(gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)) > HEAVY_Z:
        ang = args['angular_grid']
        if not isinstance(ang, (int, numpy.integer)) or ang < HEAVY_ANGULAR:
            args['angular_grid'] = HEAVY_ANGULAR
    return args
BLK = 112               # grid points per task of the kernel (a multiple of 56)
AO_CUTOFF = 1e-10       # AO values below which an AO is dropped from a block
TOL = 1e-9              # screening of the shell-pair contributions, B_IJ max|F_J| sum_g |w phi| < TOL
TCUT = 36.              # p r^2 beyond which a one-centre primitive pair is in its far field
MULTI_MIN = 4           # densities from which the multi-density contraction runs
MULTI_FLOAT = False     # its contraction in single precision (set_multi_float)
HESS_BLK = None         # grid points per task of the Hessian kernel: None -> self.blk, or 56 when the threads'
                        # buffers (~ nao nocc blk each) do not fit max_memory; an int forces it (multiple of 56)
HESS_FLOAT = False      # the Hessian passes' A C contraction (Fock1 / K second derivatives) in single precision
PROFILE = {}            # wall seconds of get_k's phases (accumulated)
MULTI_MAXSET = 512      # densities per pass (every pass computes the integrals again)
MULTI_MEM_FLOOR = .3    # fraction of max_memory a pass of the multi-density K may always use
DENSE_MIN = .4          # a grid block contracts densely when its kept pairs are >= this fraction of nao^2/2,
                        # else record-wise (the register-blocked sparse_contract on 16-point tiles); same results.
                        # Crossover ~0.42-0.47 (24 threads, float / double): trpn_1_5 def2-SVP kept 0.42 sparse 15.0 /
                        # 29.5 s vs dense 14.9 / 27.5 s, C16H34 def2-TZVP kept 0.47 6.4 / 12.8 vs 5.5 / 10.2 s, the
                        # Hessian's A C 28.7 vs 28.3 s (trpn_1_5); with the old record-wise loops it was ~0.2
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
    if COSX_RADII_ADJUST is not None:
        g.radii_adjust = COSX_RADII_ADJUST
    g.alpha_scale = ALPHA_SCALE
    g.element_intacc = {}       # the XC grids' radial rule for Li, Be, Na, Mg is not for the COSX grids
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
        # the molecule's own data only: the header slots of _env are scratch (pyscf's with_rinv_at_nucleus
        # leaves the atom index in AS_RINV_ORIG_ATOM, so every gradient / Hessian changes them)
        key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env[gto.PTR_ENV_START:].tobytes())
        if self._key == key and self._opt is not None:
            return self
        t0 = (logger.process_clock(), logger.perf_counter())
        self._free()
        libdft = _lib()
        if self.grids is None:
            self.grids = make_grids(mol)
        elif self._key is not None:
            # another geometry: this object's grid (its level, not the default GRID) on the new molecule
            if hasattr(self.grids, 'reset'):
                self.grids.reset(mol)
                self.grids.build(with_non0tab=True)
            else:
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

    def _local_ao_blocks(self, deriv, nblk=None):
        '''(g0, g1, idx, ao) over the grid in blocks of nblk x 56 points: idx the AOs of the shells that are
        nonzero on the block (grids.non0tab; None: all), ao ((ncomp,) ng, len(idx)) their values.  The
        grid-side Gram matrices (S_g and its derivatives) then cost the local AO count squared per block
        instead of nao^2 (Trp-cage def2-SVP, nao 2901: a few hundred AOs per block).'''
        from socutils.dft.vxc_fast import eval_ao
        from pyscf.dft import numint
        mol = self.mol
        ni = numint.NumInt()
        coords, w = self.grids.coords, self.grids.weights
        non0 = self.grids.non0tab
        ao_loc = mol.ao_loc_nr()
        nf = ao_loc[1:] - ao_loc[:-1]
        nao = mol.nao
        blk = 56 * (GRAM_BLOCKS if nblk is None else nblk)
        for g0 in range(0, w.size, blk):
            g1 = min(w.size, g0 + blk)
            ao = eval_ao(ni, mol, coords[g0:g1], deriv, non0tab=non0[g0 // 56:])
            shm = non0[g0 // 56:(g1 + 55) // 56].any(axis=0)
            if shm.all():
                yield g0, g1, None, ao
                continue
            idx = numpy.nonzero(numpy.repeat(shm, nf))[0]
            if idx.size == 0:
                continue
            yield g0, g1, idx, numpy.ascontiguousarray(ao[..., idx])
        assert nao == ao_loc[-1]

    def _gram(self, kind):
        '''the grid Gram matrices by the C kernel (SOCDFTcosx_gram, the local AOs of each block): kind 0 S_g
        (nao, nao), 1 sum_g w d_x phi phi (3, nao, nao), 2 the second-derivative ones (15, nao, nao); None
        without the kernel'''
        libdft = self._libdft if self._libdft is not None else _lib()
        if libdft is None or not hasattr(libdft, 'SOCDFTcosx_gram'):
            return None
        from socutils.dft.vxc_fast import _blas_ptrs
        mol = self.mol
        nao = mol.nao
        atm, bas, env, ao_loc = self._env
        out = numpy.empty(((1, 3, 15)[kind], nao, nao))
        # points per GEMM: the block's AO buffers are (1, 4, 10) nao blk per thread
        blk = 56 * (GRAM_BLOCKS if kind < 2 else max(1, GRAM_BLOCKS // 4))
        err = libdft.SOCDFTcosx_gram(
            out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(kind), self._coords.ctypes.data_as(ctypes.c_void_p),
            self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._weights.size), ctypes.c_int(blk), ao_loc.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p), *_blas_ptrs())
        if err:
            raise MemoryError('COSX grid Gram matrices')
        return out

    def _overlap_fit(self):
        '''S S_g^-1, S_g the overlap on the grid'''
        mol = self.mol
        sg = self._gram(0)
        if sg is not None:
            sg = lib.hermi_sum(sg[0], inplace=True)
            sg *= .5
        else:
            sg = numpy.zeros((mol.nao, mol.nao))
            w = self.grids.weights
            for g0, g1, idx, ao in self._local_ao_blocks(0):
                if idx is None:
                    lib.dot(ao.T * w[g0:g1], ao, beta=1, c=sg)
                else:
                    sg[idx[:, None], idx] += lib.dot(ao.T * w[g0:g1], ao)
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
            # of a block, (kept fraction) nao^2/2 blk each (the largest kept fraction of the previous call,
            # 1/2 before the first) and the dense A(g), nao^2
            nt = lib.num_threads()
            per = ((4 if libdft.SOCDFTcosx_mfloat() else 12) * 2 * self.blk * nao * nt + 8 * 2.5 * nao * nao) / 1e6
            if getattr(libdft, 'SOCDFTcosx_kpriv', None) is not None and libdft.SOCDFTcosx_kpriv():
                per += 8 * nao * nao * nt / 1e6         # the threads' private K
            kept = getattr(self, '_kept', .5)
            fixed = (4 if libdft.SOCDFTcosx_mfloat() else 8) * (kept * nao * nao / 2 * self.blk + nao * nao) * nt / 1e6
            # At least 2 GB and MULTI_MEM_FLOOR of max_memory, whatever the caller already holds: in a Hessian
            # the process is near max_memory during the CP-SCF, and the 2 GB floor alone gave ~47 densities
            # per pass (trpn_1_5 def2-SVP, 276 densities, 24 threads: 6 passes 27 s, 2 passes 19 s, 1 pass 16 s)
            avail = max(2000, MULTI_MEM_FLOOR * mol.max_memory, mol.max_memory - lib.current_memory()[0] - fixed)
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
            if hasattr(libdft, 'SOCDFTcosx_set_dense_min'):
                libdft.SOCDFTcosx_set_dense_min(ctypes.c_double(DENSE_MIN))
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
        if multi and getattr(libdft, 'SOCDFTcosx_kept', None) is not None:
            libdft.SOCDFTcosx_kept.restype = ctypes.c_double
            self._kept = max(.05, libdft.SOCDFTcosx_kept())
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
            mol = self.mol
            out = self._gram(1)
            if out is None:
                out = numpy.zeros((3, mol.nao, mol.nao))
                w = self.grids.weights
                for g0, g1, idx, ao in self._local_ao_blocks(1):
                    wao = ao[0] * w[g0:g1, None]
                    for x in range(3):
                        if idx is None:
                            lib.dot(ao[1 + x].T, wao, beta=1, c=out[x])
                        else:
                            out[x][idx[:, None], idx] += lib.dot(ao[1 + x].T, wao)
            self._ipsg_v = out
            self._ipsg_key = self._key
        return self._ipsg_v

    def _fock1_plain(self, mocc, yocc, tol=None, padded=False):
        """V = d K_plain[D] / dR C, VT = (d K_plain[D] / dR)^T Y (natm, 3, nao, nocc), D = 2 C C^T, at
        fixed D on the fixed grid (padded: with the kernel's padded occupied dimension, contiguous); the
        kernel's blk / A C form / threads from max_memory as for _hess_plain"""
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
        blk, dense, nt, (shared, per) = self._hess_config(nao, n8, True)
        logger.info(self, 'COSX Fock1 kernel: blk %d, %s A C, %d threads; memory estimate %.0f MB shared + '
                    '%.0f MB x %d threads = %.0f MB (max_memory %d, in use %.0f)', blk, 'dense' if dense else 'record-wise',
                    nt, shared, per, nt, shared + per * nt, mol.max_memory, lib.current_memory()[0])
        v = numpy.empty((mol.natm, 3, nao, n8))
        vt = numpy.empty((mol.natm, 3, nao, n8))
        if hasattr(libdft, 'SOCDFTcosx_set_dense_min'):
            libdft.SOCDFTcosx_set_dense_min(ctypes.c_double(DENSE_MIN))
        if not dense:
            libdft.SOCDFTcosx_set_dense(0)
        try:
            with lib.with_omp_threads(nt):
                err = libdft.SOCDFTcosx_fock1(
                    v.ctypes.data_as(ctypes.c_void_p), vt.ctypes.data_as(ctypes.c_void_p),
                    co.ctypes.data_as(ctypes.c_void_p), yo.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n8),
                    ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
                    self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(self._weights.size), ctypes.c_int(blk), ctypes.c_double(self.ao_cutoff),
                    ctypes.c_double(self.tol if tol is None else tol), ctypes.c_double(self.tcut),
                    ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                    bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
                    *_blas_ptrs())
        finally:
            if not dense:
                libdft.SOCDFTcosx_set_dense(1)
        if err:
            raise MemoryError('COSX Fock1: a kernel buffer could not be allocated')
        if padded:
            return v, vt
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
        # v, vt padded and contiguous; the result formed in place of vt (no third natm 3 nao nocc array)
        v, vt = self._fock1_plain(mocc, y, tol, padded=True)
        nocc = mocc.shape[1]
        dm = 2 * mocc @ mocc.T
        kp = self._plain_k(dm)
        u = Q @ (kp @ mocc)
        kq = kp.T @ Q
        ip = mol.intor('int1e_ipovlp', comp=3)
        ipsg = self._ipsg()
        aoslices = mol.aoslice_by_atom()
        for ia in range(mol.natm):
            for x in range(3):
                ds, dsg = self._dproj(ia, x, ip, ipsg, aoslices)
                # dX C = dP (K C) + P V, dX^T C = VT + K^T dP^T C;  dP = dS Q - P dS_g Q
                dxc = ds @ u - P @ (dsg @ u) + P @ v[ia, x, :, :nocc]
                dxtc = vt[ia, x, :, :nocc] + kq @ (ds @ mocc - dsg @ y)
                vt[ia, x, :, :nocc] = .5 * (dxc + dxtc)
        v = None
        return vt[..., :nocc]

    def _sg2(self):
        """sum_g w d_x d_y phi_mu phi_nu (6, nao, nao: xx xy xz yy yz zz) and sum_g w d_x phi_mu d_y phi_nu
        (3, 3, nao, nao) on the grid, cached per geometry"""
        if getattr(self, '_sg2_key', None) != self._key:
            from socutils.dft.vxc_fast import eval_ao
            from pyscf.dft import numint
            mol = self.mol
            ni = numint.NumInt()
            nao = mol.nao
            out = self._gram(2)
            if out is not None:
                gaa, gab = out[:6], out[6:].reshape(3, 3, nao, nao)
            else:
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

    def _hess_config(self, nao, n8, with_fock1):
        """(blk, dense, threads, estimate in MB) of the Hessian kernel within max_memory: the threads' buffers
        scale with nao nocc blk (A C of a block), so blk 112 -> 56, then the record-wise A C contraction
        (no stored pairs), then fewer threads.  The estimate: (shared V + VT, per thread)."""
        mol = self.mol
        natm = mol.natm
        nt0 = lib.num_threads()
        fl = 1.5 if HESS_FLOAT else 1.
        shared = (2 * natm * 3 * nao * n8 * 8 / 1e6) if with_fock1 else 0.

        def per(blk, dense):
            nd = min(3 * natm, 300)                             # local directions of a block
            dcb = max(3, min(nd, (1 << 27) // (8 * nao * blk)))
            m = nao * n8 * blk * fl                             # A C
            m += 4 * nd * n8 * blk                              # psi^d, w psi^d, w eta^d, P2
            m += dcb * nao * blk                                # Bd
            m += 2 * min(3 * natm, 300) * nao * blk             # d_Y G slabs (F and H)
            m += (.35 * nao * nao / 2 * (blk + 16) + nao * nao) * (fl - .5 if HESS_FLOAT else 1.) if dense else (2 << 20)
            m += 26 * nao * blk + 8 * nao * n8 + 2 * (1 << 22)  # AOs, gathers, U3 chunks
            return m * 8 / 1e6
        avail = mol.max_memory - lib.current_memory()[0] - shared
        if HESS_BLK is not None:
            cands = [(int(HESS_BLK), True)]
        else:
            cands = [(self.blk, True)] + ([(56, True), (56, False)] if self.blk > 56 else [(self.blk, False)])
        for blk, dense in cands:
            if nt0 * per(blk, dense) <= avail:
                return blk, dense, nt0, (shared, per(blk, dense))
        blk, dense = cands[-1]
        nt = int(max(1, min(nt0, avail // per(blk, dense))))
        return blk, dense, nt, (shared, per(blk, dense))

    def _hess_plain(self, mocc, yocc, with_fock1=False, tol=None, padded=False):
        """T1 = d^2 / dR dR' sum_g w H^T A F at fixed M = 2 Y C^T, D = 2 C C^T ((3 natm, 3 natm)), and with
        with_fock1 the plain V, VT of _fock1_plain from the same pass (padded: with the kernel's padded
        occupied dimension, contiguous)"""
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
        blk, dense, nt, (shared, per) = self._hess_config(nao, n8, with_fock1)
        logger.info(self, 'COSX Hessian kernel: blk %d, %s A C, %d threads; memory estimate %.0f MB shared + '
                    '%.0f MB x %d threads = %.0f MB (max_memory %d, in use %.0f)', blk, 'dense' if dense else 'record-wise',
                    nt, shared, per, nt, shared + per * nt, mol.max_memory, lib.current_memory()[0])
        if with_fock1:
            v = numpy.empty((mol.natm, 3, nao, n8))
            vt = numpy.empty((mol.natm, 3, nao, n8))
        ptr = lambda x: lib.c_null_ptr() if x is None else x.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        if hasattr(libdft, 'SOCDFTcosx_set_dense_min'):
            libdft.SOCDFTcosx_set_dense_min(ctypes.c_double(DENSE_MIN))
        if not dense:
            libdft.SOCDFTcosx_set_dense(0)
        try:
            with lib.with_omp_threads(nt):
                err = libdft.SOCDFTcosx_hess(
                    hs.ctypes.data_as(ctypes.c_void_p), ptr(v), ptr(vt), co.ctypes.data_as(ctypes.c_void_p),
                    yo.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n8), ctypes.c_void_p(self._opt),
                    self._coords.ctypes.data_as(ctypes.c_void_p), self._weights.ctypes.data_as(ctypes.c_void_p),
                    self._non0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self._weights.size), ctypes.c_int(blk),
                    ctypes.c_double(self.ao_cutoff), ctypes.c_double(self.tol if tol is None else tol),
                    ctypes.c_double(self.tcut), ao_loc.ctypes.data_as(ctypes.c_void_p),
                    atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p), *_blas_ptrs())
        finally:
            if not dense:
                libdft.SOCDFTcosx_set_dense(1)
        if err:
            raise MemoryError('COSX Hessian: a kernel buffer could not be allocated')
        if with_fock1 and not padded:
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
        # v, vt with the kernel's padded occupied dimension (contiguous); the Fock1 is formed in place of vt and
        # the work arrays over the 3 natm perturbations are (chunk, nao, nocc): no further natm 3 nao nocc arrays
        hs, v, vt = self._hess_plain(mocc, y, True, tol, padded=True)
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
        n8 = v.shape[-1]
        v2 = v.reshape(n3, nao * n8)
        # per perturbation R: a_R = 2 Q (dS C - dS_g Y) (2 Lambda);  T2 = a_R . V^{R'};
        # T3 = -a_R . (dS_g^{R'} u) = sum_{mu in R', nu} <d mu|nu>_g (M + M^T)_{mu nu}, M = a_R u^T
        nchunk = max(1, min(n3, int(2e8 / (nao * n8))))
        abuf = numpy.zeros((nchunk, nao, n8))
        t2 = numpy.empty((n3, n3))
        t3 = numpy.empty((n3, n3))
        ao_atom = numpy.repeat(numpy.arange(natm), aoslices[:, 3] - aoslices[:, 2])
        r0 = 0
        for ia in range(natm):
            for x in range(3):
                ds, dsg = self._dproj(ia, x, ip, ipsg, aoslices)
                r = 3 * ia + x
                dsc = ds @ mocc
                dsgy = dsg @ y
                a_r = 2 * (Q @ (dsc - dsgy))
                abuf[r - r0, :, :nocc] = a_r
                ms = a_r @ u.T
                ms = ms + ms.T
                for xx in range(3):
                    rowsum = numpy.einsum('pq,pq->p', ipsg[xx], ms)
                    t3[r, xx::3] = numpy.bincount(ao_atom, weights=rowsum, minlength=natm)
                if with_fock1:
                    dxc = ds @ u - P @ (dsg @ u) + P @ v[ia, x, :, :nocc]
                    dxtc = vt[ia, x, :, :nocc] + kq @ (dsc - dsgy)
                    vt[ia, x, :, :nocc] = .5 * (dxc + dxtc)
                if r - r0 + 1 == nchunk or r == n3 - 1:
                    nr = r - r0 + 1
                    t2[r0:r + 1] = lib.dot(abuf[:nr].reshape(nr, -1), v2.T)
                    r0 = r + 1
        v = v2 = abuf = None
        f1 = vt[..., :nocc] if with_fock1 else None

        def blocks(m):
            return m.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
        # T2 + T2^T: 2 sum Lambda_R . V^{R'}, Lambda = Q (dS C - dS_g Y)
        t2 = blocks(t2)
        h += t2 + t2.transpose(1, 0, 3, 2)
        # T3: (ii) -2 lam_a . sig_b, (iii-a) 2 kap_a . sig_b, both orders
        t3 = blocks(t3)
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

    def grad(self, dm, tol=None, grid_response=False):
        """d/dR of tr(D P K_plain[D]) (fit) or tr(D K_plain[D]) at fixed D (symmetric) on the fixed grid
        (no grid response), (natm, 3); with the fit the derivatives of S and of S_g in P = S S_g^-1.
        grid_response: the grid moving with the atoms (socutils.grad.grid_response.cosx_response)"""
        self.build()
        yg = None
        libdft = self._libdft
        mol = self.mol
        from socutils.dft.vxc_fast import _blas_ptrs
        atm, bas, env, ao_loc = self._env
        if not libdft.SOCDFTcosx_build_grad(ctypes.c_void_p(self._opt), atm.ctypes.data_as(ctypes.c_void_p),
                                           bas.ctypes.data_as(ctypes.c_void_p), env.ctypes.data_as(ctypes.c_void_p)):
            raise MemoryError('COSX gradient tables')
        dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
        nao = dm.shape[-1]
        de = numpy.zeros((mol.natm, 3))
        fit = self._proj is not None
        # one integral pass: the derivative of sum_g w H^T A F and, with the fit, K_plain[D] from its A F
        # (GRAD_ONEPASS False: K_plain[D] by a K build of its own, as the SCF's)
        kp = None
        if fit:
            P = self._proj
            m = numpy.ascontiguousarray(lib.dot(P.T, dm))
            if GRAD_ONEPASS:
                kp = numpy.empty((nao, nao))
        else:
            m = dm
        g = numpy.zeros((mol.natm, 3))
        # grid response: the per-point terms from the same pass (grad.grid_response.fused_kernel)
        fn, extra, pts = libdft.SOCDFTcosx_grad, (), None
        if grid_response:
            from socutils.grad import grid_response as gresp
            fn, extra, pts = gresp.fused_kernel(libdft, self)
        fn(
            g.ctypes.data_as(ctypes.c_void_p), dm.ctypes.data_as(ctypes.c_void_p), m.ctypes.data_as(ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_void_p(self._opt), self._coords.ctypes.data_as(ctypes.c_void_p),
            self._weights.ctypes.data_as(ctypes.c_void_p), self._non0.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._weights.size), ctypes.c_int(self.blk), ctypes.c_double(self.ao_cutoff),
            ctypes.c_double(self.tol if tol is None else tol), ctypes.c_double(self.tcut),
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
            *_blas_ptrs(), lib.c_null_ptr() if kp is None else kp.ctypes.data_as(ctypes.c_void_p), *extra)
        if fit:
            if kp is None:
                kp = self._plain_k(dm)
            z = lib.dot(dm, kp.T)
            y1 = numpy.linalg.solve(self._sg, z.T).T                    # Z S_g^-1
            y2 = lib.dot(P.T, y1)
            # the overlap: dS_mu nu / dR_X = -<d mu|nu> (mu on X) - <mu|d nu> (nu on X)
            ip = mol.intor('int1e_ipovlp', comp=3)
            y1s = y1 + y1.T
            aoslices = mol.aoslice_by_atom()
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia, 2:]
                de[ia] -= numpy.einsum('xij,ij->x', ip[:, p0:p1], y1s[p0:p1])
            # the grid overlap: d/dR of sum Yg S_g, an AO-only pass
            yg = numpy.ascontiguousarray(-(y2 + y2.T))
            gy = numpy.zeros((mol.natm, 3))
            err = libdft.SOCDFTcosx_grad_gram(
                gy.ctypes.data_as(ctypes.c_void_p), yg.ctypes.data_as(ctypes.c_void_p),
                self._coords.ctypes.data_as(ctypes.c_void_p), self._weights.ctypes.data_as(ctypes.c_void_p),
                self._non0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self._weights.size), ctypes.c_int(self.blk),
                ctypes.c_double(self.ao_cutoff), ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p), *_blas_ptrs())
            if err:
                raise MemoryError('COSX gradient: the grid overlap term')
            g += gy
        if grid_response:
            from socutils.grad import grid_response as gresp
            g += gresp.cosx_response(self, dm, m, yg, tol, pts)
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
            g = grids if isinstance(grids, lib.StreamObject) else make_grids(mol0, **gridx_args(mol0, level))
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


def set_grad_accuracy(mf, accuracy=None, conv=None, auto=None):
    """the accuracy the K gradient of mf (an SCF run through apply()) is needed at, for GRAD_GRID 'auto':
    accuracy  the error allowed in the exchange part of the gradient (Eh/bohr; None: GRAD_ACCURACY);
    conv      an optimiser's max-force convergence threshold: accuracy = GRAD_CONV_FRACTION conv;
    auto      True: while the forces are large the grid follows them, accuracy = max(the above,
              GRAD_CONV_FRACTION x the max force of the previous gradient of this mf) - coarse grid far from
              convergence, the grid of the threshold near it."""
    st = getattr(mf, '_cosx', None)
    if st is None:
        raise RuntimeError('set_grad_accuracy: mf did not go through cosx.apply')
    if conv is not None:
        accuracy = GRAD_CONV_FRACTION * conv if accuracy is None else min(accuracy, GRAD_CONV_FRACTION * conv)
    st['grad_accuracy'] = accuracy
    if auto is not None:
        st['grad_auto'] = bool(auto)
    return mf


def grad_grid(mf, accuracy=None):
    """(Grids arguments, key in mf._cosx['k'], accuracy asked for) of the K-gradient grid of mf; (None, None, None):
    the SCF's grid"""
    st = mf._cosx
    spec = GRAD_GRID
    acc = None
    if isinstance(spec, str):       # 'auto'
        acc = accuracy if accuracy is not None else st.get('grad_accuracy')
        if acc is None:
            acc = GRAD_ACCURACY
        if st.get('grad_auto') and st.get('last_gmax') is not None:
            acc = max(acc, GRAD_CONV_FRACTION * st['last_gmax'])
        spec = GRAD_GRIDS[-1][1]
        for err, sp in GRAD_GRIDS:
            if err <= acc * (1 + 1e-9):
                spec = sp
                break
        if (GRAD_HEAVY_DENSE and spec == 3 and mf.mol.natm
                and max(gto.charge(mf.mol.atom_pure_symbol(i)) for i in range(mf.mol.natm)) > HEAVY_Z):
            spec = GRAD_GRID_DENSE
    if spec is None:
        return None, None, acc
    if isinstance(spec, dict):
        return dict(spec), 'grad' + repr(sorted(spec.items())), acc
    return gridx_args(mf.mol, spec), spec, acc


def grad_k(mf, dm=None, accuracy=None, grid_response=False):
    """the exchange part of the nuclear gradient, (natm, 3), of an SCF run through apply():
    d/dR of -(hyb / 4) tr(D P K_plain[D]) (no grid response), the derivatives of the integrals, of the AOs
    on the grid and of the overlap fit; None if mf did not run COSX.  Exact for the variational SCF
    (apply(variational=True)).  The grid: GRAD_GRID; 'auto' (default): the cheapest tier of GRAD_GRIDS for
    the accuracy asked for (the argument, else set_grad_accuracy, else GRAD_ACCURACY -> GRIDX3)."""
    st = getattr(mf, '_cosx', None)
    if not st or st.get('last') is None:
        return None
    k = st['k'][st['last']['lev']]
    args, key, acc = grad_grid(mf, accuracy)
    if args is not None:
        # the same energy expression differentiated on its own grid (the SCF density, its own overlap fit)
        if key not in st['k']:
            kg = COSX(mf.mol, make_grids(mf.mol, **args), fit=k.fit, tol=k.tol)
            kg.verbose, kg.stdout = k.verbose, k.stdout
            st['k'][key] = kg.build()
        k = st['k'][key]
    st['grad_key'] = key
    logger.info(mf, 'COSX gradient grid: %s, %d points%s', 'the SCF grid' if args is None else key, k._weights.size,
                '' if acc is None else ' (accuracy asked for %.1e Eh/bohr)' % acc)
    if dm is None:
        dm = mf.make_rdm1()
    hyb = 1.
    if hasattr(mf, 'xc'):
        hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)[2]
    if not st.get('variational', False):
        logger.warn(mf, 'COSX gradient of a non-variational SCF (apply(variational=False)): '
                    'the orbital response is not exact')
    return -.25 * hyb * k.grad(numpy.asarray(dm), grid_response=grid_response)
