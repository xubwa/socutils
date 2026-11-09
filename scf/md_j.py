#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Exact J matrices from the McMurchie-Davidson J engine (lib/vhf/md_j.c
SOCVHFmd_j_matrix): the basis decontracted to Cartesian primitives, every
density in Hermite form once per primitive pair, the Hermite potentials of
all pairs in one pass over the pairs of pairs, J_ab = sum_t E^ab_t V^P_t; no
Cartesian (ab|cd) are formed (pyscf's direct J builds them per quartet as for
K).  Several densities share one pass.

    vj = md_j.get_j(mol, dms)       # dms (nao, nao) or (ndm, nao, nao)
    mf = md_j.apply(scf.RKS(mol))   # J-only builds of the SCF (pure functionals)
                                    # from the J engine, J + K builds (hybrids,
                                    # HF) from socutils.scf.jk_multi
'''

import ctypes
import numpy
from pyscf import gto, lib
from pyscf.lib import logger

LMAX = 4                # lib/vhf/md_j.c LMAX
TOL = 1e-13
_cache = {}


def _load():
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFmd_j_matrix'):
        return None
    return libv


def available(mol):
    return _load() is not None and mol._bas[:, gto.ANG_OF].max() <= LMAX


def _prim_basis(mol):
    '''decontracted Cartesian copy, the contraction matrix (nprim, nao) and the
    Schwarz factors of its shell pairs, cached per geometry and basis'''
    key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes(), mol.cart)
    if key not in _cache:
        from socutils.scf import jk_multi
        pmol, c = mol.decontract_basis(to_cart=True, aggregate=True)
        q = numpy.ascontiguousarray(jk_multi._q_cond(pmol))
        _cache.clear()
        _cache[key] = (pmol, numpy.ascontiguousarray(c), q)
    return _cache[key]


def get_j(mol, dm, hermi=1, tol=TOL, max_memory=None):
    '''J[D] = sum_cd (ab|cd) D_cd of one or several densities (the symmetric
    part of each, all J sees).  None if the library is unavailable or l > LMAX.'''
    libv = _load()
    if libv is None or mol._bas[:, gto.ANG_OF].max() > LMAX:
        return None
    dm = numpy.asarray(dm, dtype=numpy.double)
    shape = dm.shape
    nao = shape[-1]
    dms = dm.reshape(-1, nao, nao)
    if hermi != 1:
        dms = (dms + dms.transpose(0, 2, 1)) * .5
    pmol, c, q = _prim_basis(mol)
    npao = c.shape[0]
    if max_memory is None:
        max_memory = mol.max_memory
    # per density: the Hermite densities, nthreads potential buffers, the
    # primitive density and J
    ls = pmol._bas[:, gto.ANG_OF]
    lij = (ls[:, None] + ls)[numpy.tril_indices(pmol.nbas)]
    oh = int(((lij + 1) * (lij + 2) * (lij + 3) // 6).sum())
    per_dm = 8 * (oh * (lib.num_threads() + 1) + 2 * npao * npao) / 1e6
    nbatch = max(1, min(len(dms), int(max_memory * .5 / per_dm)))
    vj = numpy.empty_like(dms)
    ao_loc = pmol.ao_loc_nr().astype(numpy.int32)
    for d0 in range(0, len(dms), nbatch):
        d1 = min(len(dms), d0 + nbatch)
        dmp = numpy.ascontiguousarray(lib.einsum('pi,dij,qj->dpq', c, dms[d0:d1], c))
        vjp = numpy.empty_like(dmp)
        libv.SOCVHFmd_j_matrix(
            vjp.ctypes.data_as(ctypes.c_void_p), dmp.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(d1 - d0),
            q.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol),
            ao_loc.ctypes.data_as(ctypes.c_void_p), pmol._atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(pmol.natm), pmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pmol.nbas),
            pmol._env.ctypes.data_as(ctypes.c_void_p))
        vj[d0:d1] = lib.einsum('pi,dpq,qj->dij', c, vjp, c)
    return vj.reshape(shape)


REBUILD_NSTEPS = 20     # incremental builds between two full J / K builds
SCF_TOL = 1e-12         # quartet cutoff of the SCF builds (Gaussian's default 2e cutoff)
TOL_PER_CONV = 1e-2     # ... and at most TOL_PER_CONV * mf.conv_tol, so that the screening (and its
                        # accumulation over incremental builds) stays below the convergence threshold
J_BY_JK_MULTI_LMAX = 2  # J-only builds of bases with l <= this by jk_multi, not the MD J engine
INCORE = True           # J + K builds from in-memory s8 ERIs when they fit (mf._is_mem_enough, as pyscf)


RIJ_RESIDENT = True     # RIJ: L^{-1} (P|ij) built once and kept in memory when it fits, J = B^T (B d)
INCORE_RESERVE = 500    # MB of mf.max_memory kept for the other arrays of the SCF


def incore_fits(mf, mol):
    '''the s8 ERIs (8 npair (npair + 1) / 2 bytes) within mf.max_memory - INCORE_RESERVE, max_memory
    taken as the budget of the work arrays as Gaussian's %mem (which keeps caffeine / cc-pVDZ's
    3.7 GB in memory at %mem=4GB); the process's own ~1 GB (Python, pyscf, MKL) not counted.
    pyscf's _is_mem_enough asks nao^4 / 1e6 + the current RSS < 0.95 max_memory.'''
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    return npair * (npair + 1) // 2 * 8 / 1e6 < mf.max_memory - INCORE_RESERVE


def incore_eri(mol):
    '''the s8 ERIs: socutils.scf.jk_multi.eri_s8 (the generated kernels), else pyscf's
    mol.intor int2e s8 (caffeine / cc-pVDZ, 4 threads: libcint 8.7 s)'''
    from socutils.scf import jk_multi
    eri = jk_multi.eri_s8(mol)
    return mol.intor('int2e', aosym='s8') if eri is None else eri


SCF_TOL_LOOSE = 1e-9    # two-pass: the cutoff while the density still moves (1e-10 -> 1e-9: caffeine / C16H34
                        # HF / PBE0 SCF 5 - 11 % faster, same cycles, |dE| < 1e-9; 1e-8 costs cycles)
LOOSE_PER_CONV = 10     # ... and at most LOOSE_PER_CONV * mf.conv_tol (tight SCFs keep 1e-10)
LOOSE_UNTIL = 1e-4      # two-pass: max |D - D_prev| below which the builds are tight


def apply(mf, incremental=True, two_pass=True):
    '''mf with get_jk computing J-only builds by the MD J engine and J + K (or
    K) builds by jk_multi in one pass; range-separated calls, complex
    densities, density fitting and l > LMAX keep the original.  Modifies and
    returns mf.

    incremental: J, K of a single symmetric density from the previous call's
    J, K plus those of the density change, J[D] = J[D_prev] + J[D - D_prev]
    (exact by linearity; the density screening of the change makes the late
    SCF cycles cheap), with a full build every REBUILD_NSTEPS calls and
    whenever the molecule or the requested matrices change.

    two_pass (with incremental): the builds use the cutoff SCF_TOL_LOOSE
    until max |D - D_prev| < LOOSE_UNTIL, then one full build at SCF_TOL
    (which drops the accumulated loose errors) and SCF_TOL from there on, for
    this molecule (as Gaussian's integral accuracy switch near convergence).
    The tight cutoff is min(SCF_TOL, TOL_PER_CONV * mf.conv_tol): with conv_tol
    1e-11 the 1e-12 cutoff's noise (accumulated over the incremental builds)
    would keep the SCF from converging.'''
    from pyscf.df import df_jk
    from pyscf.scf import hf as hf_mod
    from socutils.scf import jk_multi
    df_j = isinstance(mf, df_jk._DFHF)
    if (df_j and not mf.only_dfj) or not available(mf.mol):
        return mf
    if df_j:
        # RIJ (density_fit(only_dfj=True)): J from the fitting, the exact K by the builds below
        orig_df_jk = mf.get_jk

        def orig_jk(mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
            return hf_mod.SCF.get_jk(mf, mol, dm, hermi, with_j, with_k, omega)
    else:
        orig_jk = mf.get_jk
    mol0 = mf.mol
    last = {'key': None}

    def tight_tol():
        return min(SCF_TOL, TOL_PER_CONV * getattr(mf, 'conv_tol', 1e-9))

    def build(mol, dm, hermi, with_j, with_k, omega, max_memory, tol=None):
        if tol is None:
            tol = tight_tol()
        t0 = (logger.process_clock(), logger.perf_counter())
        if with_k:
            # J and K from one multi-density direct pass; a separate MD J does
            # not pay, K costs nearly the whole pass
            if not jk_multi.available(mol):
                return orig_jk(mol, dm, hermi, with_j, with_k, omega)
            vj, vk = jk_multi.get_jk(mol, dm, with_j, True, tol=tol, hermi=hermi, max_memory=max_memory)
            logger.timer_debug1(mf, 'vj, vk (jk_multi)', *t0)
            return vj, vk
        if not with_j:
            return orig_jk(mol, dm, hermi, with_j, with_k, omega)
        if mol._bas[:, gto.ANG_OF].max() <= J_BY_JK_MULTI_LMAX and jk_multi.available(mol):
            # the generated J / K kernels beat the MD J engine for l <= 2 (cholesterol / cc-pVDZ,
            # 4 threads, one full J: 39.6 s against 73.9 s)
            vj = jk_multi.get_jk(mol, dm, True, False, tol=tol, hermi=hermi, max_memory=max_memory)[0]
            logger.timer_debug1(mf, 'vj (jk_multi)', *t0)
            return vj, None
        vj = get_j(mol, dm, hermi, tol=tol, max_memory=max_memory)
        if vj is None:
            return orig_jk(mol, dm, hermi, with_j, with_k, omega)
        logger.timer_debug1(mf, 'vj (MD J engine)', *t0)
        return vj, None

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if mol is None:
            mol = mol0
        if dm is None:
            dm = mf.make_rdm1()
        dm = numpy.asarray(dm)
        if omega is not None or numpy.iscomplexobj(dm):
            return orig_jk(mol, dm, hermi, with_j, with_k, omega)
        if INCORE and (with_j or with_k) and mol is mol0 and (mf._eri is not None or incore_fits(mf, mol)):
            # the ERIs fit in memory: pyscf's in-core J / K (the s8 ERIs once, then a
            # contraction per cycle), as Gaussian does then ("Keep R1 ints in memory");
            # caffeine / cc-pVDZ HF, 4 threads: 18.6 s against 24.9 s direct
            if mf._eri is None:
                t0 = (logger.process_clock(), logger.perf_counter())
                mf._eri = incore_eri(mol)
                logger.timer(mf, 'in-core ERIs', *t0)
            from pyscf.scf import hf as hf_mod
            return hf_mod.dot_eri_dm(mf._eri, dm, hermi, with_j, with_k)
        max_memory = max(1000, mf.max_memory - lib.current_memory()[0])
        if not (incremental and dm.ndim == 2 and hermi == 1):
            return build(mol, dm, hermi, with_j, with_k, omega, max_memory)
        key = (with_j, with_k, mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
        same = last['key'] == key
        ddm = dm - last['dm'] if same else None
        if not same:
            last['tight'] = not two_pass
        elif not last['tight'] and abs(ddm).max() < LOOSE_UNTIL:
            last['tight'] = True
        loose = min(SCF_TOL_LOOSE, LOOSE_PER_CONV * getattr(mf, 'conv_tol', 1e-9))
        tol = tight_tol() if last['tight'] else max(tight_tol(), loose)
        if same and last['n'] < REBUILD_NSTEPS and last['tol'] == tol:
            vj1, vk1 = build(mol, ddm, hermi, with_j, with_k, omega, max_memory, tol)
            vj = last['vj'] + vj1 if with_j else None
            vk = last['vk'] + vk1 if with_k else None
            last['n'] += 1
            logger.debug1(mf, 'incremental J / K (%d since the full build), cutoff %.0e', last['n'], tol)
        else:
            vj, vk = build(mol, dm, hermi, with_j, with_k, omega, max_memory, tol)
            last['n'] = 0
            logger.debug1(mf, 'full J / K build, cutoff %.0e', tol)
        # copies: callers scale the returned matrices in place (vk *= hyb)
        last.update(key=key, dm=dm.copy(), tol=tol, vj=None if vj is None else vj.copy(),
                    vk=None if vk is None else vk.copy())
        return vj, vk

    if df_j:
        cache = {'key': None, 'B': None}

        def rij_j(mol, dm, hermi):
            '''J from B = L^{-1} (P|ij) (lower-triangle pairs) kept in memory: c = B d, J = B^T c;
            None if B does not fit'''
            if not RIJ_RESIDENT or hermi != 1 or numpy.iscomplexobj(dm):
                return None
            key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
            if cache['key'] != key:
                from pyscf import df
                from socutils.hessian.rhf_fast import _df_auxmol
                auxmol = _df_auxmol(mf)
                nao = mol.nao
                npair = nao * (nao + 1) // 2
                if auxmol.nao * npair * 8 / 1e6 > mf.max_memory - lib.current_memory()[0] - INCORE_RESERVE:
                    cache.update(key=key, B=None)
                    return None
                t0 = (logger.process_clock(), logger.perf_counter())
                cache['B'] = df.incore.cholesky_eri(mol, auxmol=auxmol, int3c='int3c2e', aosym='s2ij',
                                                    max_memory=mf.max_memory)
                cache['key'] = key
                logger.timer(mf, 'RIJ: resident L^-1 (P|ij)', *t0)
            B = cache['B']
            if B is None:
                return None
            dms = numpy.asarray(dm)
            nao = dms.shape[-1]
            d = dms.reshape(-1, nao, nao)
            d = d + d.transpose(0, 2, 1)
            idx = numpy.arange(nao)
            d[:, idx, idx] *= .5
            dtril = lib.pack_tril(d)                                  # (nset, npair), off-diagonal doubled
            c = lib.dot(dtril, B.T)                                   # (nset, naux)
            vj = lib.unpack_tril(lib.dot(c, B))                       # (nset, nao, nao)
            return vj.reshape(dms.shape)

        def get_jk_rij(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
            if mol is None:
                mol = mol0
            if dm is None:
                dm = mf.make_rdm1()
            if omega is not None or mol is not mol0:
                return orig_df_jk(mol, dm, hermi, with_j, with_k, omega)
            vj = None
            if with_j:
                vj = rij_j(mol, dm, hermi)
                if vj is None:
                    vj = mf.with_df.get_jk(dm, hermi, with_j=True, with_k=False)[0]
            vk = get_jk(mol, dm, hermi, False, True, omega)[1] if with_k else None
            return vj, vk
        mf.get_jk = get_jk_rij
        return mf
    mf.get_jk = get_jk
    return mf
