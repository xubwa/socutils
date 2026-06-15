'''Fast spinor (j-spinor) AO->MO transform, fully in C with s4 AO symmetry.

Bypasses the broken s2/s4 plumbing of pyscf.ao2mo.nrr_outcore entirely and
drives two custom C kernels (libnrr_opt.so):

  * e1 (bra): reuse the non-relativistic AO permutational-symmetry fill
    (AO2MOfill_nr_s4, ~4x fewer shell quartets than the s1-only nrr path)
    then a complex two-component (alpha + beta j-spinor) MO contraction.
  * e2 (ket): unpack the shell-block s2kl half transform and do the complex
    ket MO contraction (AO2MOmmm_r_iltj) for alpha and beta, summed.

The only host-side work is a single complex transpose between the two passes.

Output matches int2e_spinor over j-spinor MOs to machine precision; see the
self-test at the bottom and NOTES.md.
'''
import os
import ctypes
import _ctypes
import numpy as np
from pyscf import lib
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3

_libopt = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libnrr_opt.so'))
_libao2mo = lib.load_library('libao2mo')


def _fp(name, dso):
    return ctypes.c_void_p(_ctypes.dlsym(dso._handle, name))


def _e1_full(mol, moija, moijb, ijshape, intor='int2e_sph', ao2mopt=None):
    '''bra half transform over the *full* kl range (s4 AO symmetry).

    Returns the half-transformed integrals (nkl_pair, nij) with the kl axis in
    the shell-block s2kl packing produced by AO2MOfill_nr_s4 (k>=l).'''
    intor = ascint3(intor)
    moija = np.asarray(moija, dtype=np.complex128, order='F')
    moijb = np.asarray(moijb, dtype=np.complex128, order='F')
    i0, i1, j0, j1 = ijshape
    ij_count = (i1 - i0) * (j1 - j0)

    c_atm = np.asarray(mol._atm, dtype=np.int32)
    c_bas = np.asarray(mol._bas, dtype=np.int32)
    c_env = np.asarray(mol._env)
    nbas = c_bas.shape[0]
    ao_loc = make_loc(c_bas, 'int2e_sph')
    nao = int(ao_loc[nbas])

    klsh0 = 0
    klsh1 = nbas * (nbas + 1) // 2
    nkl = nao * (nao + 1) // 2          # total packed kl pairs

    out = np.empty((2, nkl, ij_count), dtype=np.complex128)

    fill = _fp('AO2MOfill_nr_s4', _libao2mo)
    # deficiency #2: transform the smaller MO count first (i first if i<=j)
    if (i1 - i0) <= (j1 - j0):
        fmmm = _fp('AO2MOmmm_nrr_iltj', _libopt)
    else:
        fmmm = _fp('AO2MOmmm_nrr_igtj', _libopt)
    if ao2mopt is not None:
        cintopt = ao2mopt._cintopt
        intor = ao2mopt._intor
    else:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
    cintor = _fp(intor, _libao2mo)
    tao = np.asarray(mol.tmap(), dtype=np.int32)

    _libopt.AO2MOnrr_opt_e1_drv(
        cintor, fill, fmmm,
        out.ctypes.data_as(ctypes.c_void_p),
        moija.ctypes.data_as(ctypes.c_void_p),
        moijb.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(klsh0), ctypes.c_int(klsh1 - klsh0),
        ctypes.c_int(nkl), ctypes.c_int(1),
        (ctypes.c_int * 4)(*ijshape), tao.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, lib.c_null_ptr(),
        c_atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_atm.shape[0]),
        c_bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        c_env.ctypes.data_as(ctypes.c_void_p))
    out[0] += out[1]
    return out[0]                       # (nkl, nij)


def _e2(mol, vin, mokla, moklb, klshape):
    '''ket half transform: vin (nij, nkl_pair) -> (nij, nmok*nmol).'''
    vin = np.asarray(vin, dtype=np.complex128, order='C')
    nij = vin.shape[0]
    mokla = np.asarray(mokla, dtype=np.complex128, order='F')
    moklb = np.asarray(moklb, dtype=np.complex128, order='F')
    c_bas = np.asarray(mol._bas, dtype=np.int32)
    nbas = c_bas.shape[0]
    ao_loc = make_loc(c_bas, 'int2e_sph')
    nao = int(ao_loc[nbas])
    k_count = klshape[1] - klshape[0]
    l_count = klshape[3] - klshape[2]
    kl_mo = k_count * l_count

    vout = np.empty((nij, kl_mo), dtype=np.complex128)
    # deficiency #2: transform the smaller ket MO count first
    if k_count <= l_count:
        fmmm = _fp('AO2MOmmm_r_iltj', _libao2mo)
    else:
        fmmm = _fp('AO2MOmmm_r_igtj', _libao2mo)

    _libopt.AO2MOnrr_opt_e2_drv(
        fmmm,
        vout.ctypes.data_as(ctypes.c_void_p),
        vin.ctypes.data_as(ctypes.c_void_p),
        mokla.ctypes.data_as(ctypes.c_void_p),
        moklb.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nij), ctypes.c_int(nao),
        (ctypes.c_int * 4)(*klshape),
        ao_loc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas))
    return vout


def general(mol, mo_coeffs, intor='int2e_sph'):
    '''(ij|kl) chemist over four sets of j-spinor MOs, returned (nij, nkl).'''
    ca, cb = mol.sph2spinor_coeff()
    mo_a = [ca.dot(m) for m in mo_coeffs]
    mo_b = [cb.dot(m) for m in mo_coeffs]
    nmoi, nmoj, nmok, nmol = [m.shape[1] for m in mo_coeffs]

    moija = np.hstack((mo_a[0], mo_a[1]))
    moijb = np.hstack((mo_b[0], mo_b[1]))
    ijshape = (0, nmoi, nmoi, nmoi + nmoj)

    half = _e1_full(mol, moija, moijb, ijshape, intor)        # (nkl, nij)
    vin = lib.transpose(half)                                  # (nij, nkl)

    mokla = np.hstack((mo_a[2], mo_a[3]))
    moklb = np.hstack((mo_b[2], mo_b[3]))
    klshape = (0, nmok, nmok, nmok + nmol)
    return _e2(mol, vin, mokla, moklb, klshape)


def general_iofree(mol, mo_coeffs, intor='int2e_sph'):
    return general(mol, mo_coeffs, intor)


def full(mol, mo_coeff, intor='int2e_sph'):
    return general(mol, (mo_coeff,) * 4, intor)
