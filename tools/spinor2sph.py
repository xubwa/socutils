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
def pauli_decompose(ints_so):
    nao_2c = ints_so.shape[0]
    naonr = nao_2c // 2
    soaa = ints_so[:naonr,:naonr]
    #sobb = ints_so[naonr:,naonr:]
    soab = ints_so[:naonr,naonr:]
    #soba = ints_so[naonr:,:naonr]
    so_scalar = soaa.real
    so_sz = soaa.imag
    so_sy = soab.real
    so_sx = soab.imag
    return numpy.array([so_scalar, so_sx, so_sy, so_sz])

def spinor2sph_soc(mol, spinor):

    #assert (spinor.dtype == complex), "spinor integral must be complex"
    assert (spinor.shape[0] == mol.nao_2c()), "spinor integral must be of shape (nao_2c, nao_2c)"

    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    ints_so = lib.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    
    return pauli_decompose(ints_so)

def spinor2spinor_sd(mol, spinor):
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
