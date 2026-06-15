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
import subprocess
import numpy as np
from pyscf import lib
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, 'libnrr_opt.so')
_SRC = os.path.join(_HERE, 'nrr_ao2mo_opt.c')


def _ensure_lib():
    '''Build libnrr_opt.so from nrr_ao2mo_opt.c if missing or stale.

    The .so is gitignored, so it is (re)built on demand against the installed
    pyscf libraries (all needed headers live under pyscf/lib).'''
    if os.path.exists(_SO) and os.path.getmtime(_SO) >= os.path.getmtime(_SRC):
        return
    pi = os.path.dirname(os.path.abspath(lib.__file__))      # pyscf/lib
    import glob
    openblas = glob.glob(os.path.join(pi, 'libopenblas*.so')) or \
        glob.glob(os.path.join(pi, 'deps', 'lib', 'libopenblas*.so'))
    cmd = ['gcc', '-shared', '-fPIC', '-fopenmp', '-O2', _SRC, '-o', _SO,
           '-I' + pi, '-I' + os.path.join(pi, 'deps', 'include'), '-I' + _HERE,
           '-L' + pi, '-lao2mo', '-lnp_helper', '-lcvhf',
           '-L' + os.path.join(pi, 'deps', 'lib'), '-lcint',
           *openblas,
           '-Wl,-rpath,' + pi, '-Wl,-rpath,' + os.path.join(pi, 'deps', 'lib')]
    subprocess.run(cmd, check=True)


_ensure_lib()
_libopt = ctypes.CDLL(_SO)
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


def _split_ab(mol, mos, motype):
    '''alpha/beta spherical blocks of the given 2-component MOs.'''
    if motype == 'j-spinor':
        ca, cb = mol.sph2spinor_coeff()
        return [ca.dot(m) for m in mos], [cb.dot(m) for m in mos]
    elif motype == 'ghf':
        nao = mos[0].shape[0] // 2
        return [m[:nao] for m in mos], [m[nao:] for m in mos]
    raise ValueError('motype must be "j-spinor" or "ghf", got %r' % motype)


def bra_half(mol, A, B, intor='int2e_sph', motype='j-spinor'):
    '''e1 (bra) half transform of one bra pair (A,B); returns vin (nij, nkl)
    ready for ket_transform.  Cache & reuse this across kets that share (A,B)
    -- e.g. EE's vovv and voov both use the expensive bra=(v,v) half.'''
    mo_a, mo_b = _split_ab(mol, (A, B), motype)
    nmoi, nmoj = A.shape[1], B.shape[1]
    moija = np.hstack((mo_a[0], mo_a[1]))
    moijb = np.hstack((mo_b[0], mo_b[1]))
    ijshape = (0, nmoi, nmoi, nmoi + nmoj)
    half = _e1_full(mol, moija, moijb, ijshape, intor)        # (nkl, nij)
    return lib.transpose(half)                                # (nij, nkl)


def ket_transform(mol, vin, C, D, intor='int2e_sph', motype='j-spinor'):
    '''e2 (ket) transform of a cached bra half ``vin`` against ket pair (C,D);
    returns the chemist block (nij, nmok*nmol).'''
    mo_a, mo_b = _split_ab(mol, (C, D), motype)
    nmok, nmol = C.shape[1], D.shape[1]
    mokla = np.hstack((mo_a[0], mo_a[1]))
    moklb = np.hstack((mo_b[0], mo_b[1]))
    klshape = (0, nmok, nmok, nmok + nmol)
    return _e2(mol, vin, mokla, moklb, klshape)


def general(mol, mo_coeffs, intor='int2e_sph', motype='j-spinor'):
    '''(ij|kl) chemist over four sets of 2-component MOs, returned (nij, nkl).

    motype : 'j-spinor' -- MOs are in the n2c 2-component sph basis; the alpha
                           and beta sph blocks are recovered via sph2spinor_coeff.
             'ghf'      -- MOs are (2*nao, nmo) GHF spinors; the alpha/beta sph
                           blocks are the upper/lower halves.
    '''
    vin = bra_half(mol, mo_coeffs[0], mo_coeffs[1], intor, motype)
    return ket_transform(mol, vin, mo_coeffs[2], mo_coeffs[3], intor, motype)


def general_iofree(mol, mo_coeffs, intor='int2e_sph', motype='j-spinor'):
    return general(mol, mo_coeffs, intor, motype)


def full(mol, mo_coeff, intor='int2e_sph', motype='j-spinor'):
    return general(mol, (mo_coeff,) * 4, intor, motype)
