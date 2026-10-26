#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Direct J / K of many density matrices at once (lib/vhf/jk_multi.c): one
integral evaluation per unique shell quartet shared by all densities, which
are updated with contiguous vector operations.  For the many first-order
densities of CPHF / TDDFT responses, where pyscf's direct J / K is limited
by its per-density contraction.

    vj, vk = jk_multi.get_jk(mol, dms)          # dms (ndm, nao, nao), any symmetry
    vj, vk = jk_multi.get_jk(mol, dms, hermi=1) # symmetric dms: 2 J + 4 K updates
                                                # per integral instead of 8 + 8
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.scf import _vhf

TOL = 1e-13
_q_cache = {}


def _q_cond(mol):
    key = (id(mol), mol._bas.tobytes(), mol._env.tobytes())
    if key not in _q_cache:
        nbas = mol.nbas
        q = numpy.empty((nbas, nbas))
        ao_loc = mol.ao_loc_nr()
        with mol.with_integral_screen(TOL**2):
            _vhf.libcvhf.CVHFnr_int2e_q_cond(
                getattr(_vhf.libcvhf, mol._add_suffix('int2e')), lib.c_null_ptr(), q.ctypes,
                ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
        _q_cache.clear()
        _q_cache[key] = q
    return _q_cache[key]


def available(mol):
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    return bool(libv) and hasattr(libv, 'SOCVHFjk_multi') and not mol.cart


def get_jk(mol, dms, with_j=True, with_k=True, tol=TOL, hermi=0):
    '''J_ab = (ab|cd) D_dc, K_ad = (ab|cd) D_bc for dms (ndm, nao, nao) (real).
    hermi=1 asserts symmetric dms (as pyscf's get_jk).'''
    from pyscf.gto import moleintor
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    dms = numpy.asarray(dms, dtype=numpy.double)
    shape = dms.shape
    dms = dms.reshape(-1, shape[-2], shape[-1])
    ndm, nao = dms.shape[:2]
    d = numpy.ascontiguousarray(dms.transpose(1, 2, 0))
    vj = numpy.empty((nao, nao, ndm)) if with_j else numpy.empty(1)
    vk = numpy.empty((nao, nao, ndm)) if with_k else numpy.empty(1)
    q = _q_cond(mol)
    dcond = ghf_hess._dm_cond(mol, list(dms))
    atm, bas, env = mol._atm, mol._bas, mol._env
    opt = moleintor.make_cintopt(atm, bas, env, 'int2e_sph')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    drv = libv.SOCVHFjk_multi_sym if hermi == 1 else libv.SOCVHFjk_multi
    drv(
        vj.ctypes.data_as(ctypes.c_void_p), vk.ctypes.data_as(ctypes.c_void_p),
        d.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ndm),
        ctypes.c_int(int(with_j)), ctypes.c_int(int(with_k)),
        q.ctypes.data_as(ctypes.c_void_p), dcond.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(tol), opt,
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    out_j = vj.transpose(2, 0, 1).reshape(shape) if with_j else None
    out_k = vk.transpose(2, 0, 1).reshape(shape) if with_k else None
    return out_j, out_k
