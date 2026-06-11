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
    from socutils.fci import zfci
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.fcisolver.nroots = 3
'''

import sys
import itertools
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


###############################################################
# CI in a given list of determinants (selected CI)
###############################################################

def gen_ras_occslst(nras, nelec, max_hole=0, max_elec=0):
    '''Generate the determinant list of a RAS-type CI space, for use as the
    occslst of SelectedCI.

    The active space is partitioned into three contiguous blocks of spinor
    orbitals: RAS1 (orbitals 0 to n1-1), RAS2 (the next n2 orbitals) and
    RAS3 (the last n3 orbitals).  The list contains every determinant with
    at most max_hole holes in RAS1 and at most max_elec electrons in RAS3;
    occupation of RAS2 is unrestricted.

    Args:
        nras: (n1, n2, n3), number of spinor orbitals in each RAS block
        nelec: total number of electrons
        max_hole: maximum number of holes in RAS1
        max_elec: maximum number of electrons in RAS3

    With max_hole >= n1 and max_elec >= n3 this reproduces the complete
    CAS determinant space; with nras = (n1, 0, n3) and max_hole = max_elec
    = 2 it gives a CISD-type space, etc.
    '''
    n1, n2, n3 = nras
    ras1 = range(0, n1)
    ras2 = range(n1, n1 + n2)
    ras3 = range(n1 + n2, n1 + n2 + n3)
    occslst = []
    for nh in range(min(max_hole, n1) + 1):
        ne1 = n1 - nh
        for ne3 in range(min(max_elec, n3) + 1):
            ne2 = nelec - ne1 - ne3
            if not 0 <= ne2 <= n2:
                continue
            for o1 in itertools.combinations(ras1, ne1):
                for o2 in itertools.combinations(ras2, ne2):
                    for o3 in itertools.combinations(ras3, ne3):
                        occslst.append(o1 + o2 + o3)
    return numpy.asarray(occslst, dtype=numpy.int64).reshape(-1, nelec)

if hasattr(numpy, 'bitwise_count'):
    _popcount = numpy.bitwise_count
else:  # numpy < 2.0
    _popcount_tab = numpy.array([bin(i).count('1') for i in range(256)],
                                dtype=numpy.uint8)
    def _popcount(x):
        x = numpy.ascontiguousarray(x)
        return _popcount_tab[x.view(numpy.uint8)].reshape(x.shape + (x.dtype.itemsize,)).sum(axis=-1)

def _bitpos(x):
    '''Position of the (single) set bit in each element of x'''
    return _popcount(x - numpy.uint64(1)).astype(numpy.int64)

def _between_mask(p, q):
    '''Bit mask covering the positions strictly between p and q'''
    one = numpy.uint64(1)
    lo = numpy.minimum(p, q).astype(numpy.uint64)
    hi = numpy.maximum(p, q).astype(numpy.uint64)
    return ((one << hi) - one) ^ ((one << (lo + one)) - one)

def _to_strings(occslst, norb):
    '''Convert a list of determinants (each a list of occupied spinor-orbital
    indices) to sorted occupation lists and uint64 bit strings.'''
    occs = numpy.asarray(occslst, dtype=numpy.int64)
    if occs.ndim != 2:
        raise ValueError('occslst must be a (ndet, nelec) array of occupied orbitals')
    if norb > 64:
        raise NotImplementedError('more than 64 spinor orbitals')
    occs = numpy.sort(occs, axis=1)
    if occs.size > 0 and ((occs[:,0] < 0).any() or (occs[:,-1] >= norb).any()):
        raise ValueError('orbital index out of range')
    strs = numpy.zeros(len(occs), dtype=numpy.uint64)
    for k in range(occs.shape[1]):
        strs |= numpy.uint64(1) << occs[:,k].astype(numpy.uint64)
    if (_popcount(strs) != occs.shape[1]).any():
        raise ValueError('repeated orbital within a determinant')
    if len(numpy.unique(strs)) != len(strs):
        raise ValueError('duplicate determinants in occslst')
    return occs, strs

def _excitations(strs, p0, p1):
    '''Classify all pairs (I in [p0:p1), J in full list) by excitation level.

    Returns:
        singles: (row, col, a, i, phase) with |I> = phase * a_a^+ a_i |J>
        doubles: (row, col, a, b, i, j, phase) with
                 |I> = phase * a_b^+ a_j a_a^+ a_i |J>, a < b, i < j
        row is relative to p0.
    '''
    one = numpy.uint64(1)
    xor = strs[p0:p1, None] ^ strs[None, :]
    ndiff = _popcount(xor)

    ri, rj = numpy.nonzero(ndiff == 2)
    jstr = strs[rj]
    xp = xor[ri, rj]
    a = _bitpos(xp & strs[p0 + ri])
    i = _bitpos(xp & jstr)
    perm = _popcount(jstr & _between_mask(a, i))
    singles = (ri, rj, a, i, 1 - 2*(perm.astype(numpy.int64) & 1))

    ri2, rj2 = numpy.nonzero(ndiff == 4)
    jstr = strs[rj2]
    xp = xor[ri2, rj2]
    pmask = xp & strs[p0 + ri2]
    hmask = xp & jstr
    abit = pmask & (~pmask + one)
    ibit = hmask & (~hmask + one)
    a = _bitpos(abit)
    b = _bitpos(pmask ^ abit)
    i = _bitpos(ibit)
    j = _bitpos(hmask ^ ibit)
    perm = _popcount(jstr & _between_mask(a, i))
    perm += _popcount((jstr ^ (ibit | abit)) & _between_mask(b, j))
    doubles = (ri2, rj2, a, b, i, j, 1 - 2*(perm.astype(numpy.int64) & 1))
    return singles, doubles

def _pair_block_size(ndet, max_memory):
    # xor + popcount intermediates: ~3 arrays of (bsize, ndet) uint64
    bsize = int(max_memory*1e6 / (3 * 8 * ndet + 1))
    return max(1, min(ndet, bsize))

def make_hmat_dets(h1e, eri, norb, occslst, max_memory=2000):
    '''Build the CI Hamiltonian matrix in the basis of an arbitrary list of
    determinants, using Slater-Condon rules.

    Args:
        occslst: (ndet, nelec) array-like; each row lists the occupied spinor
            orbitals of one determinant.  A determinant is defined with its
            creation operators in ascending orbital order.

    The phase convention is identical to the cistring-based full CI basis, so
    when occslst covers the complete space in cistring order this returns the
    same matrix as make_hmat.
    '''
    occs, strs = _to_strings(occslst, norb)
    nd, ne = occs.shape
    hmat = numpy.zeros((nd, nd), dtype=numpy.complex128)
    if nd == 0 or ne == 0:
        return hmat

    # diagonal: sum_i h_ii + 1/2 sum_ij [(ii|jj) - (ij|ji)]
    jk = numpy.einsum('iijj->ij', eri) - numpy.einsum('ijji->ij', eri)
    diag = h1e.diagonal()[occs].sum(axis=1)
    diag = diag + 0.5 * jk[occs[:,:,None], occs[:,None,:]].sum(axis=(1,2))
    hmat[numpy.diag_indices(nd)] = diag

    # w[a,i,k] = (ai|kk) - (ak|ki); note w[a,i,i] = 0, so the sum over the
    # common occupied orbitals can run over all occupied orbitals of J
    w = numpy.einsum('aikk->aik', eri) - numpy.einsum('akki->aik', eri)

    for p0, p1 in lib.prange(0, nd, _pair_block_size(nd, max_memory)):
        (ri, rj, a, i, ph1), (ri2, rj2, a2, b2, i2, j2, ph2) = \
                _excitations(strs, p0, p1)
        wai = numpy.take_along_axis(w[a, i], occs[rj], axis=1).sum(axis=1)
        hmat[p0 + ri, rj] = ph1 * (h1e[a, i] + wai)
        hmat[p0 + ri2, rj2] = ph2 * (eri[a2, i2, b2, j2] - eri[a2, j2, b2, i2])
    return hmat

def kernel_dets(h1e, eri, norb, occslst, ecore=0, nroots=1, max_memory=2000,
                verbose=logger.NOTE):
    '''Diagonalize the Hamiltonian in the space spanned by the determinants
    in occslst.  Returns energies and CI vectors; the CI coefficients follow
    the order of occslst.'''
    hmat = make_hmat_dets(h1e, eri, norb, occslst, max_memory)
    e, c = scipy.linalg.eigh(hmat)
    if nroots == 1:
        return e[0] + ecore, c[:,0]
    else:
        return e[:nroots] + ecore, [numpy.ascontiguousarray(c[:,i])
                                    for i in range(nroots)]

def trans_rdm1_dets(cibra, ciket, norb, occslst, max_memory=2000):
    '''Transition density matrix dm_pq = <bra|p^+ q|ket> in a determinant
    list basis.'''
    occs, strs = _to_strings(occslst, norb)
    nd, ne = occs.shape
    rdm1 = numpy.zeros((norb, norb), dtype=numpy.complex128)
    if nd == 0 or ne == 0:
        return rdm1
    wdiag = cibra.conj() * ciket
    numpy.add.at(rdm1, (occs.ravel(), occs.ravel()), numpy.repeat(wdiag, ne))
    for p0, p1 in lib.prange(0, nd, _pair_block_size(nd, max_memory)):
        (ri, rj, a, i, ph), _ = _excitations(strs, p0, p1)
        numpy.add.at(rdm1, (a, i), ph * cibra[p0 + ri].conj() * ciket[rj])
    return rdm1

def make_rdm1_dets(fcivec, norb, occslst, max_memory=2000):
    return trans_rdm1_dets(fcivec, fcivec, norb, occslst, max_memory)

def make_rdm12_dets(fcivec, norb, occslst, max_memory=2000):
    '''1- and 2-particle density matrices in a determinant list basis,
    with the same conventions as fci_dhf_slow.make_rdm12:
    rdm1[p,q] = <p^+ q>, rdm2[p,q,r,s] = <p^+ r^+ s q>.'''
    occs, strs = _to_strings(occslst, norb)
    nd, ne = occs.shape
    rdm1 = numpy.zeros((norb, norb), dtype=numpy.complex128)
    rdm2 = numpy.zeros((norb,)*4, dtype=numpy.complex128)
    if nd == 0 or ne == 0:
        return rdm1, rdm2

    # diagonal: <k^+ k> and <k^+ l^+ l k> = -<k^+ l^+ k l> for k != l
    w = (fcivec.conj() * fcivec)
    numpy.add.at(rdm1, (occs.ravel(), occs.ravel()), numpy.repeat(w, ne))
    k = numpy.broadcast_to(occs[:,:,None], (nd, ne, ne))
    l = numpy.broadcast_to(occs[:,None,:], (nd, ne, ne))
    offdiag = (k != l).ravel()
    k = k.ravel()[offdiag]
    l = l.ravel()[offdiag]
    wkl = numpy.broadcast_to(w[:,None,None], (nd, ne, ne)).ravel()[offdiag]
    numpy.add.at(rdm2, (k, k, l, l), wkl)
    numpy.add.at(rdm2, (k, l, l, k), -wkl)

    for p0, p1 in lib.prange(0, nd, _pair_block_size(nd, max_memory)):
        (ri, rj, a, i, ph1), (ri2, rj2, a2, b2, i2, j2, ph2) = \
                _excitations(strs, p0, p1)
        # singles, with spectator orbital k in both determinants
        c1 = ph1 * fcivec[p0 + ri].conj() * fcivec[rj]
        numpy.add.at(rdm1, (a, i), c1)
        kk = occs[rj]                              # (ns, ne), includes k == i
        spect = (kk != i[:,None]).ravel()
        aa = numpy.repeat(a, ne)[spect]
        ii = numpy.repeat(i, ne)[spect]
        cc = numpy.repeat(c1, ne)[spect]
        kk = kk.ravel()[spect]
        numpy.add.at(rdm2, (aa, ii, kk, kk), cc)
        numpy.add.at(rdm2, (kk, kk, aa, ii), cc)
        numpy.add.at(rdm2, (aa, kk, kk, ii), -cc)
        numpy.add.at(rdm2, (kk, ii, aa, kk), -cc)
        # doubles
        c2 = ph2 * fcivec[p0 + ri2].conj() * fcivec[rj2]
        numpy.add.at(rdm2, (a2, i2, b2, j2), c2)
        numpy.add.at(rdm2, (b2, j2, a2, i2), c2)
        numpy.add.at(rdm2, (a2, j2, b2, i2), -c2)
        numpy.add.at(rdm2, (b2, i2, a2, j2), -c2)
    return rdm1, rdm2


class SelectedCI(lib.StreamObject):
    '''CI solver in an arbitrary, fixed list of determinants.

    The Hamiltonian is built with Slater-Condon rules directly in the basis
    of the given determinants and diagonalized exactly, so the full CI space
    never needs to be enumerated.  When occslst covers the complete
    determinant space (in cistring order) the result is identical to
    FCISolver.

    Attributes:
        occslst: (ndet, nelec) array-like; each row lists the occupied
            spinor orbitals of one determinant, e.g. the determinant list
            of a preceding SHCI calculation.  CI coefficients follow the
            order of this list.

    Usage:
        from socutils.fci import zfci
        mc = zmcscf.CASSCF(mf, ncas, nelecas)
        mc.fcisolver = zfci.SelectedCI(mol)
        mc.fcisolver.occslst = [[0,1,2,3], [0,1,4,5], ...]
    '''

    def __init__(self, mol=None, occslst=None):
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
            self.max_memory = lib.param.MAX_MEMORY
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
            self.max_memory = mol.max_memory
        self.mol = mol
        self.occslst = occslst
        self.nroots = 1
        self.converged = True
        self.eci = None
        self.ci = None

    def _check_occslst(self, nelec):
        if self.occslst is None:
            raise RuntimeError('SelectedCI.occslst is not set')
        occs = numpy.asarray(self.occslst)
        if not isinstance(nelec, (int, numpy.integer)):
            nelec = sum(nelec)
        if occs.shape[1] != nelec:
            raise ValueError(f'occslst has {occs.shape[1]} occupied orbitals '
                             f'per determinant, but nelec = {nelec}')
        return occs

    def make_hmat(self, h1e, eri, norb, nelec, max_memory=None):
        if max_memory is None:
            max_memory = self.max_memory - lib.current_memory()[0]
        return make_hmat_dets(h1e, eri, norb, self._check_occslst(nelec),
                              max_memory)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None,
               ecore=0, max_memory=None, verbose=None, **kwargs):
        if nroots is None: nroots = self.nroots
        if max_memory is None:
            max_memory = self.max_memory - lib.current_memory()[0]
        log = logger.new_logger(self, verbose)
        occs = self._check_occslst(nelec)
        self.norb = norb
        self.nelec = nelec
        log.debug('Selected CI space size %d, Hamiltonian matrix %.1f MB',
                  len(occs), len(occs)**2*16e-6)
        self.eci, self.ci = kernel_dets(h1e, eri.reshape(norb, norb, norb, norb),
                                        norb, occs, ecore=ecore,
                                        nroots=min(nroots, len(occs)),
                                        max_memory=max_memory, verbose=log)
        self.converged = True
        return self.eci, self.ci

    def make_rdm1(self, fcivec, norb, nelec, **kwargs):
        return make_rdm1_dets(fcivec, norb, self._check_occslst(nelec))

    def make_rdm12(self, fcivec, norb, nelec, **kwargs):
        return make_rdm12_dets(fcivec, norb, self._check_occslst(nelec))

    def make_rdm2(self, fcivec, norb, nelec, **kwargs):
        return self.make_rdm12(fcivec, norb, nelec, **kwargs)[1]

    def trans_rdm1(self, cibra, ciket, norb, nelec, **kwargs):
        return trans_rdm1_dets(cibra, ciket, norb, self._check_occslst(nelec))


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
