'''Prototype: optimized spinor (j-spinor) AO->MO transform.

Drop-in for pyscf.ao2mo.nrr_outcore.general_iofree(..., motype='j-spinor'):
reuses all of nrr_outcore's buffering / kl-pass machinery but replaces the
e1 (bra) step with the non-relativistic s4 AO fill (i>=j, k>=l permutational
symmetry) plus a complex two-component MO contraction (libnrr_opt.so), instead
of the stock s1-only nrr path.
'''
import os
import ctypes
import _ctypes
import numpy as np
from pyscf import lib
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3
from pyscf.ao2mo import nrr_outcore

_libopt = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libnrr_opt.so'))
_libao2mo = nrr_outcore.libao2mo


def _fp(name, dso):
    return ctypes.c_void_p(_ctypes.dlsym(dso._handle, name))


def r_e1_opt(intor, mo_a, mo_b, orbs_slice, sh_range, atm, bas, env,
             tao, aosym='s1', comp=1, ao2mopt=None, out=None):
    '''Replacement for nrr_outcore.r_e1: s4 fill + complex 2-component mmm.'''
    assert aosym == 's4'
    intor = ascint3(intor)
    mo_a = np.asarray(mo_a, dtype=np.complex128, order='F')
    mo_b = np.asarray(mo_b, dtype=np.complex128, order='F')
    i0, i1, j0, j1 = orbs_slice
    ij_count = (i1 - i0) * (j1 - j0)
    c_atm = np.asarray(atm, dtype=np.int32)
    c_bas = np.asarray(bas, dtype=np.int32)
    c_env = np.asarray(env)
    klsh0, klsh1, nkl = sh_range

    out = np.ndarray((2 * comp, nkl, ij_count), dtype=np.complex128, buffer=out)
    if out.size == 0:
        return out[:comp]

    fill = _fp('AO2MOfill_nr_s4', _libao2mo)
    fmmm = _fp('AO2MOmmm_nrr_iltj', _libopt)
    if ao2mopt is not None:
        cintopt = ao2mopt._cintopt
        intor = ao2mopt._intor
    else:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
    cintor = _fp(intor, _libao2mo)
    tao = np.asarray(tao, dtype=np.int32)
    ao_loc = make_loc(c_bas, 'int2e_sph')

    _libopt.AO2MOnrr_opt_e1_drv(
        cintor, fill, fmmm,
        out.ctypes.data_as(ctypes.c_void_p),
        mo_a.ctypes.data_as(ctypes.c_void_p),
        mo_b.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(klsh0), ctypes.c_int(klsh1 - klsh0),
        ctypes.c_int(nkl), ctypes.c_int(comp),
        (ctypes.c_int * 4)(*orbs_slice), tao.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, lib.c_null_ptr(),
        c_atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_atm.shape[0]),
        c_bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_bas.shape[0]),
        c_env.ctypes.data_as(ctypes.c_void_p))
    for i in range(comp):
        out[i] += out[2 * i + 1]
    return out[:comp]


def _count_naopair_sph(mol, nao):
    # s4 AO-pair count in the *spherical* AO basis (the int2e_sph integrals),
    # not the 2-component basis that nrr_outcore._count_naopair assumes.
    ao_loc = mol.ao_loc_nr()
    npair = 0
    for i in range(mol.nbas):
        di = int(ao_loc[i + 1] - ao_loc[i])
        for j in range(i + 1):
            npair += di * int(ao_loc[j + 1] - ao_loc[j])
    return npair


def general_iofree_opt(mol, mo_coeffs, intor='int2e_sph', verbose=0):
    '''s4-accelerated spinor general_iofree (motype='j-spinor').'''
    orig_r_e1 = nrr_outcore.r_e1
    orig_cnt = nrr_outcore._count_naopair
    nrr_outcore.r_e1 = r_e1_opt
    nrr_outcore._count_naopair = _count_naopair_sph
    try:
        return nrr_outcore.general_iofree(
            mol, mo_coeffs, intor=intor, motype='j-spinor', aosym='s4',
            verbose=verbose)
    finally:
        nrr_outcore.r_e1 = orig_r_e1
        nrr_outcore._count_naopair = orig_cnt
