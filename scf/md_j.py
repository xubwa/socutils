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


def apply(mf, incremental=True):
    '''mf with get_jk computing J-only builds by the MD J engine and J + K (or
    K) builds by jk_multi in one pass; range-separated calls, complex
    densities, density fitting and l > LMAX keep the original.  Modifies and
    returns mf.

    incremental: J, K of a single symmetric density from the previous call's
    J, K plus those of the density change, J[D] = J[D_prev] + J[D - D_prev]
    (exact by linearity; the density screening of the change makes the late
    SCF cycles cheap), with a full build every REBUILD_NSTEPS calls and
    whenever the molecule or the requested matrices change.'''
    from pyscf.df import df_jk
    from socutils.scf import jk_multi
    if isinstance(mf, df_jk._DFHF) or not available(mf.mol):
        return mf
    orig_jk = mf.get_jk
    mol0 = mf.mol
    last = {'key': None}

    def build(mol, dm, hermi, with_j, with_k, omega, max_memory):
        t0 = (logger.process_clock(), logger.perf_counter())
        if with_k:
            # J and K from one multi-density direct pass; a separate MD J does
            # not pay, K costs nearly the whole pass
            if not jk_multi.available(mol):
                return orig_jk(mol, dm, hermi, with_j, with_k, omega)
            vj, vk = jk_multi.get_jk(mol, dm, with_j, True, hermi=hermi, max_memory=max_memory)
            logger.timer_debug1(mf, 'vj, vk (jk_multi)', *t0)
            return vj, vk
        if not with_j:
            return orig_jk(mol, dm, hermi, with_j, with_k, omega)
        vj = get_j(mol, dm, hermi, max_memory=max_memory)
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
        max_memory = max(1000, mf.max_memory - lib.current_memory()[0])
        if not (incremental and dm.ndim == 2 and hermi == 1):
            return build(mol, dm, hermi, with_j, with_k, omega, max_memory)
        key = (with_j, with_k, mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
        if last['key'] == key and last['n'] < REBUILD_NSTEPS:
            vj1, vk1 = build(mol, dm - last['dm'], hermi, with_j, with_k, omega, max_memory)
            vj = last['vj'] + vj1 if with_j else None
            vk = last['vk'] + vk1 if with_k else None
            last['n'] += 1
            logger.debug1(mf, 'incremental J / K (%d since the full build)', last['n'])
        else:
            vj, vk = build(mol, dm, hermi, with_j, with_k, omega, max_memory)
            last['n'] = 0
        # copies: callers scale the returned matrices in place (vk *= hyb)
        last.update(key=key, dm=dm.copy(), vj=None if vj is None else vj.copy(),
                    vk=None if vk is None else vk.copy())
        return vj, vk

    mf.get_jk = get_jk
    return mf
