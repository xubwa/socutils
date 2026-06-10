#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Exact full CI solver for spinor (relativistic) Hamiltonians.

The full CI Hamiltonian matrix is built explicitly in the determinant basis
and diagonalized with a dense Hermitian eigensolver.  It is a drop-in
replacement for the Dice-based interface (socutils.hci.shci) when the active
space is small enough for exact diagonalization, and it avoids the Davidson
iterations of pyscf.fci.fci_dhf_slow, which can be fragile for the (nearly)
degenerate roots arising from Kramers degeneracy.

Determinant ordering, integral conventions and RDM conventions follow
pyscf.fci.fci_dhf_slow, so the resulting CI vectors and density matrices are
interchangeable with that module.

Usage:
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.fcisolver.nroots = 3
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import fci_dhf_slow
from pyscf.fci.fci_dhf_slow import (absorb_h1e, contract_2e, make_hdiag,
                                    make_rdm1, make_rdm12, reorder_rdm)


def make_hmat(h1e, eri, norb, nelec, max_memory=2000):
    '''Explicitly build the full CI Hamiltonian matrix in the determinant
    basis spanned by cistring.gen_occslst(range(norb), nelec).

    The matrix is built by applying the Hamiltonian to the columns of the
    identity matrix, using the same link-table machinery as
    fci_dhf_slow.contract_2e but vectorized over batches of determinants.

    Args:
        h1e: complex (norb, norb) one-electron Hamiltonian in spinor MO basis
        eri: complex (norb, norb, norb, norb) two-electron integrals in
            chemists' notation, (pq|rs) without any permutational symmetry
        norb: number of spinor orbitals
        nelec: number of electrons

    Returns:
        complex (na, na) Hermitian matrix, na = C(norb, nelec)
    '''
    h2e = absorb_h1e(h1e, eri, norb, nelec, 0.5).reshape(norb**2, norb**2)
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    nlink = link_index.shape[1]

    hmat = numpy.zeros((na, na), dtype=numpy.complex128)
    if na * nlink == 0:
        return hmat

    # Link table: E_{ai} |str0> = sign |str1>
    addr_ai = link_index[:,:,0] * norb + link_index[:,:,1]  # (na, nlink)
    str1 = link_index[:,:,2]
    sign = link_index[:,:,3]

    # H e_j = sum_{bj'} E_{bj'} sum_{ai} h2e[bj',ai] E_{ai} e_j.  Only the
    # nlink determinants reachable from j by E_{ai} contribute, so for each
    # column only the corresponding nlink columns of h2e and their nlink
    # outgoing links need to be touched.
    per_col = 16 * nlink * (norb**2 + 6*nlink) + 16 * na
    bsize = max(1, min(na, int(max_memory*1e6 / per_col)))
    for p0, p1 in lib.prange(0, na, bsize):
        nb = p1 - p0
        nl = nb * nlink
        # m[:,k] = sign_k * h2e[:, (a_k,i_k)], the coefficient vector carried
        # by the intermediate determinant str1_k = E_{a_k i_k} |j>
        m = h2e[:, addr_ai[p0:p1].ravel()] * sign[p0:p1].ravel()
        # outgoing links of every intermediate determinant
        tab2 = link_index[str1[p0:p1].ravel()]                  # (nl, nlink, 4)
        rows2 = tab2[:,:,0] * norb + tab2[:,:,1]
        vals = m[rows2, numpy.arange(nl)[:,None]] * tab2[:,:,3]
        jcol = numpy.repeat(numpy.arange(nb), nlink)[:,None]
        hblk = numpy.zeros((na, nb), dtype=numpy.complex128)
        numpy.add.at(hblk.ravel(), (tab2[:,:,2] * nb + jcol).ravel(), vals.ravel())
        hmat[:, p0:p1] = hblk
    return hmat


def kernel(h1e, eri, norb, nelec, ecore=0, nroots=1, max_memory=2000,
           verbose=logger.NOTE):
    '''Solve the full CI problem by explicit construction and dense
    diagonalization of the Hamiltonian matrix.

    Returns:
        (energy, civec) for nroots == 1, otherwise
        (energies, [civec0, civec1, ...])
    '''
    hmat = make_hmat(h1e, eri, norb, nelec, max_memory)
    e, c = scipy.linalg.eigh(hmat)
    if nroots == 1:
        return e[0] + ecore, c[:,0]
    else:
        return e[:nroots] + ecore, [numpy.ascontiguousarray(c[:,i])
                                    for i in range(nroots)]


def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    '''Transition density matrix dm_pq = <bra|p^+ q|ket>, with the same
    convention as fci_dhf_slow.make_rdm1.'''
    if nelec == 0:
        return numpy.zeros((norb, norb), dtype=ciket.dtype)
    nd = cistring.num_strings(norb, nelec - 1)
    index_d = cistring.gen_des_str_index(range(norb), nelec)
    dbra = numpy.zeros((nd, norb), dtype=numpy.complex128)
    dket = numpy.zeros((nd, norb), dtype=numpy.complex128)
    for str0, tab in enumerate(index_d):
        for _, i, str1, sign in tab:
            dbra[str1, i] += sign * cibra[str0]
            dket[str1, i] += sign * ciket[str0]
    return dbra.conj().T @ dket


class FCISolver(fci_dhf_slow.FCISolver):
    '''Exact full CI solver for complex spinor integrals.

    Inherits the RDM machinery from pyscf.fci.fci_dhf_slow.FCISolver but
    replaces the Davidson kernel by explicit Hamiltonian construction plus
    dense diagonalization.  All roots are available at the cost of a single
    eigh call; set nroots to the number of states needed.
    '''

    def __init__(self, mol=None):
        fci_dhf_slow.FCISolver.__init__(self, mol)
        self.converged = True

    def make_hmat(self, h1e, eri, norb, nelec, max_memory=None):
        if max_memory is None:
            max_memory = self.max_memory - lib.current_memory()[0]
        return make_hmat(h1e, eri, norb, nelec, max_memory)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None,
               ecore=0, max_memory=None, verbose=None, **kwargs):
        # ci0 is ignored; the diagonalization is exact.
        if nroots is None: nroots = self.nroots
        if max_memory is None:
            max_memory = self.max_memory - lib.current_memory()[0]
        log = logger.new_logger(self, verbose)
        self.norb = norb
        self.nelec = nelec
        na = cistring.num_strings(norb, nelec)
        log.debug('Full CI space size %d, Hamiltonian matrix %.1f MB',
                  na, na**2*16e-6)
        self.eci, self.ci = kernel(h1e, eri.reshape(norb, norb, norb, norb),
                                   norb, nelec, ecore=ecore,
                                   nroots=min(nroots, na),
                                   max_memory=max_memory, verbose=log)
        self.converged = True
        return self.eci, self.ci

FCI = FCISolver


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto, scf, ao2mo

    mol = gto.M(atom='H 0 0 0; H 0 0 1.2; H 0 1.0 0; H 0 1.1 1.3',
                basis='sto-3g', verbose=0)
    mf = scf.GHF(mol).run(conv_tol=1e-12)

    norb = mf.mo_coeff.shape[1]
    h1e = reduce(numpy.dot, (mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))
    mo_a, mo_b = mf.mo_coeff[:mol.nao], mf.mo_coeff[mol.nao:]
    eri = ao2mo.restore(4, ao2mo.general(mf._eri, (mo_a, mo_a, mo_b, mo_b)), norb)
    eri = eri + eri.transpose(1, 0)
    eri += ao2mo.restore(4, ao2mo.full(mf._eri, mo_a), norb)
    eri += ao2mo.restore(4, ao2mo.full(mf._eri, mo_b), norb)
    eri = ao2mo.restore(1, eri, norb)
    nelec = mol.nelectron

    e_ref, c_ref = fci_dhf_slow.kernel(h1e, eri, norb, nelec)
    e, c = kernel(h1e, eri, norb, nelec)
    print('davidson  ', e_ref)
    print('exact diag', e, abs(e - e_ref))
    assert abs(e - e_ref) < 1e-9

    hmat = make_hmat(h1e + 0j, eri + 0j, norb, nelec)
    print('hermiticity', abs(hmat - hmat.conj().T).max())

    solver = FCISolver(mol)
    solver.nroots = 3
    es, cs = solver.kernel(h1e + 0j, eri + 0j, norb, nelec, ecore=mol.energy_nuc())
    dm1, dm2 = solver.make_rdm12(cs[0], norb, nelec)
    e_from_rdm = (numpy.einsum('pq,qp->', h1e, dm1)
                  + 0.5 * numpy.einsum('pqrs,pqrs->', eri, dm2.transpose(1, 0, 2, 3))
                  + mol.energy_nuc())
    print('roots     ', es)
    print('rdm energy', e_from_rdm.real, abs(e_from_rdm - es[0]))
    tdm = solver.trans_rdm1(cs[0], cs[0], norb, nelec)
    print('trans_rdm1 vs make_rdm1', abs(tdm - solver.make_rdm1(cs[0], norb, nelec)).max())
