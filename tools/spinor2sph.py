#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import PauliMatrices

paulix, pauliy, pauliz = lib.PauliMatrices
iden = numpy.eye(2)

'''
converts a spinor soc matrix to a spherical soc matrix with four components.
the returned matrix is like [scalar, socx, socy, socz]
'''
def pauli_decompose(ints_so, with_scalar=True):
    nao_2c = ints_so.shape[0]
    naonr = nao_2c // 2
    soaa = ints_so[:naonr,:naonr]
    sobb = ints_so[naonr:,naonr:]
    soab = ints_so[:naonr,naonr:]
    soba = ints_so[naonr:,:naonr]
    so_scalar = soaa.real
    so_sz = soaa.imag
    so_sy = soab.real
    so_sx = soab.imag
    if not numpy.allclose(sobb.real, so_scalar):
        print(f"Warning: sobb.real and soaa.real are not close, "
              f"norm(diff) = {numpy.linalg.norm(soaa.real - sobb.real):.6e}. "
              f"Symmetrizing so_scalar by (soaa.real+sobb.real)/2.")
        so_scalar = 0.5 * (soaa.real + sobb.real)
    if not numpy.allclose(sobb.imag, -so_sz):
        print(f"Warning: sobb.imag and -soaa.imag are not close, "
              f"norm(diff) = {numpy.linalg.norm(sobb.imag + so_sz):.6e}. "
              f"Symmetrizing so_sz by (soaa.imag-sobb.imag)/2.")
        so_sz = 0.5 * (soaa.imag - sobb.imag)
    if not numpy.allclose(soba.real, -so_sy):
        print(f"Warning: soba.real and -soab.real are not close, "
              f"norm(diff) = {numpy.linalg.norm(soba.real + so_sy):.6e}. "
              f"Symmetrizing so_sy by (soab.real-soba.real)/2.")
        so_sy = 0.5 * (soab.real - soba.real)
    if not numpy.allclose(soba.imag, so_sx):
        print(f"Warning: soba.imag and soab.imag are not close, "
              f"norm(diff) = {numpy.linalg.norm(soba.imag - so_sx):.6e}. "
              f"Symmetrizing so_sx by (soab.imag+soba.imag)/2.")
        so_sx = 0.5 * (soab.imag + soba.imag)

    if not with_scalar:
        return numpy.array([so_sx, so_sy, so_sz])
    else:
        return numpy.array([so_scalar, so_sx, so_sy, so_sz])

def spinor2sph_soc(mol, spinor, with_scalar=True):

    #assert (spinor.dtype == complex), "spinor integral must be complex"
    assert (spinor.shape[0] == mol.nao_2c()), "spinor integral must be of shape (nao_2c, nao_2c)"

    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    ints_so = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    
    return pauli_decompose(ints_so, with_scalar=with_scalar)

def spinor2spinor_sd(mol, spinor):
    r'''
    Extract spin-dependent part of the spinor integral through pauli decomposition.
    '''
    assert (spinor.shape[0] == mol.nao_2c()), "spinor integral must be of shape (nao_2c, nao_2c)"

    paulix, pauliy, pauliz = lib.PauliMatrices
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    ints_so = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    nao_2c = ints_so.shape[0]
    naonr = nao_2c // 2
    ints_so[:naonr,:naonr].real *= 0.0 # soaa
    ints_so[naonr:,naonr:].real *= 0.0 # sobb
    result = lib.einsum('pi,ij,jq->pq',c2.T.conj(), ints_so, c2)
    return result

def spinor2sph(mol, spinor):
    assert (spinor.shape[0] == mol.nao_2c()), "spinor integral must be of shape (nao_2c, nao_2c)"
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    ints_sph = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    return ints_sph
