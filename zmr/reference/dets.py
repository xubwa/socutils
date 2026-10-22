'''
Determinants over an arbitrary orbital set and second-quantized operators
acting on them.

A determinant is a uint64 bit string (bit p set = orbital p occupied) with
the phase convention of docs/derivations/conventions.md section 4:

    |q_1 ... q_n> = a^+_{q_n} ... a^+_{q_1} |vac>,  q_1 < ... < q_n

so a^+_p and a_p carry the sign (-1)^(number of occupied orbitals above p),
identical to pyscf.fci.cistring.

An operator is a list of blocks (daggers, orbs, coeffs):

    daggers: tuple of bools, one per factor of the operator string
             o_1 o_2 ... o_k, True for a creation operator
    orbs:    (nterm, k) orbital indices, orbs[t, m] belongs to factor o_m
    coeffs:  (nterm,) complex coefficients

and represents sum_t coeffs[t] * o_1(orbs[t,0]) ... o_k(orbs[t,k-1]).  The
rightmost factor acts first.  Operators may change the electron number.

Everything here is written for clarity; it is the reference against which
the production code is validated, not a production code path.
'''

import itertools
import numpy
from scipy import sparse

MAX_ORB = 63    # keeps every shift of a uint64 below 64 bits

_ONE = numpy.uint64(1)
_CHUNK = 2**23  # max number of (term, determinant) pairs processed at once


def _popcount(x):
    return numpy.bitwise_count(x).astype(numpy.int64)


def occ_to_str(occ):
    s = 0
    for p in occ:
        if s >> p & 1:
            raise ValueError('orbital %d occupied twice' % p)
        s |= 1 << int(p)
    return numpy.uint64(s)


def str_to_occ(s):
    s = int(s)
    return [p for p in range(s.bit_length()) if s >> p & 1]


def gen_strings(orbs, nelec):
    '''All determinants with nelec electrons distributed over orbs, sorted in
    ascending integer order (the order of pyscf.fci.cistring for
    orbs = range(norb)).'''
    orbs = list(orbs)
    if max(orbs, default=0) > MAX_ORB:
        raise NotImplementedError('more than %d orbitals' % MAX_ORB)
    if nelec < 0 or nelec > len(orbs):
        return numpy.zeros(0, dtype=numpy.uint64)
    strs = [sum(1 << p for p in c) for c in itertools.combinations(orbs, nelec)]
    return numpy.asarray(sorted(strs), dtype=numpy.uint64)


def cas_strings(ncore, nact, nelec_act):
    '''Determinants with the first ncore orbitals occupied and nelec_act
    electrons in the next nact orbitals, in cistring order of the active
    part.'''
    core = numpy.uint64((1 << ncore) - 1)
    act = gen_strings(range(nact), nelec_act)
    return core | (act << numpy.uint64(ncore))


def apply_opstring(daggers, orbs, strs):
    '''Apply the operator strings o_1 ... o_k (one per row of orbs) to every
    determinant in strs.

    Returns:
        new: (nterm, ndet) uint64 result determinants
        sign: (nterm, ndet) int64, +-1, or 0 where the string annihilates the
            determinant (new is meaningless there)
    '''
    k = len(daggers)
    orbs = numpy.asarray(orbs, dtype=numpy.int64)
    orbs = orbs.reshape(orbs.shape[0] if orbs.ndim == 2 else -1, k)
    strs = numpy.asarray(strs, dtype=numpy.uint64)
    if orbs.size and (orbs.min() < 0 or orbs.max() > MAX_ORB):
        raise ValueError('orbital index out of range')
    new = numpy.repeat(strs[None, :], len(orbs), axis=0)
    sign = numpy.ones(new.shape, dtype=numpy.int64)
    for m in reversed(range(k)):
        p = orbs[:, m].astype(numpy.uint64)[:, None]
        bit = _ONE << p
        occupied = (new & bit) != 0
        if daggers[m]:
            ok = ~occupied
            new = new | bit
        else:
            ok = occupied
            new = new & ~bit
        parity = _popcount(new >> (p + _ONE)) & 1
        sign *= numpy.where(ok, 1 - 2 * parity, 0)
    return new, sign


def _term_chunks(nterm, ndet):
    step = max(1, _CHUNK // max(ndet, 1))
    for t0 in range(0, nterm, step):
        yield t0, min(nterm, t0 + step)


class SparseVector:
    '''A CI vector on an arbitrary, sorted set of determinants.'''

    def __init__(self, strs, coeffs):
        strs = numpy.asarray(strs, dtype=numpy.uint64)
        coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
        order = numpy.argsort(strs, kind='stable')
        strs = strs[order]
        coeffs = coeffs[order]
        if len(strs) > 1 and (strs[1:] == strs[:-1]).any():
            raise ValueError('duplicate determinants; use SparseVector.accumulate')
        self.strs = strs
        self.coeffs = coeffs

    @classmethod
    def accumulate(cls, strs, coeffs):
        '''Build a vector from possibly repeated determinants, summing the
        coefficients of repeated entries.'''
        strs = numpy.asarray(strs, dtype=numpy.uint64).ravel()
        coeffs = numpy.asarray(coeffs, dtype=numpy.complex128).ravel()
        uniq, inv = numpy.unique(strs, return_inverse=True)
        c = (numpy.bincount(inv, coeffs.real, len(uniq))
             + 1j * numpy.bincount(inv, coeffs.imag, len(uniq)))
        return cls(uniq, c)

    def __len__(self):
        return len(self.strs)

    def lookup(self, strs):
        '''Positions of strs in this vector, -1 where absent.'''
        strs = numpy.asarray(strs, dtype=numpy.uint64)
        return _lookup(self.strs, strs)

    def on(self, strs):
        '''Coefficients on the given determinants (0 where absent).'''
        idx = self.lookup(strs)
        return numpy.where(idx >= 0, self.coeffs[numpy.maximum(idx, 0)], 0)

    def vdot(self, other):
        '''<self|other>'''
        return numpy.vdot(self.on(other.strs), other.coeffs)

    def __add__(self, other):
        return SparseVector.accumulate(numpy.concatenate([self.strs, other.strs]),
                                       numpy.concatenate([self.coeffs, other.coeffs]))

    def __sub__(self, other):
        return self + other * -1

    def __mul__(self, a):
        return SparseVector(self.strs, self.coeffs * a)

    __rmul__ = __mul__

    def norm(self):
        return numpy.linalg.norm(self.coeffs)

    def pruned(self, tol=0.0):
        keep = abs(self.coeffs) > tol
        return SparseVector(self.strs[keep], self.coeffs[keep])


def _lookup(sorted_strs, strs):
    if len(sorted_strs) == 0:
        return numpy.full(numpy.shape(strs), -1, dtype=numpy.int64)
    idx = numpy.searchsorted(sorted_strs, strs)
    idx = numpy.minimum(idx, len(sorted_strs) - 1)
    return numpy.where(sorted_strs[idx] == strs, idx, -1)


def apply_operator(op, vec):
    '''op |vec>, returned as a SparseVector (exact zeros removed).'''
    strs_out = []
    coef_out = []
    for daggers, orbs, coeffs in op:
        coeffs = numpy.asarray(coeffs, dtype=numpy.complex128).ravel()
        orbs = numpy.asarray(orbs, dtype=numpy.int64).reshape(len(coeffs), len(daggers))
        for t0, t1 in _term_chunks(len(orbs), len(vec)):
            new, sign = apply_opstring(daggers, orbs[t0:t1], vec.strs)
            val = sign * coeffs[t0:t1, None] * vec.coeffs[None, :]
            nz = sign != 0
            strs_out.append(new[nz])
            coef_out.append(val[nz])
    if not strs_out:
        return SparseVector([], [])
    out = SparseVector.accumulate(numpy.concatenate(strs_out),
                                  numpy.concatenate(coef_out))
    return out.pruned()


def operator_matrix(op, bra_strs, ket_strs=None):
    '''Sparse matrix M[I, J] = <bra_I| op |ket_J> (scipy CSR).  Contributions
    that leave the bra determinant set are dropped.'''
    bra_strs = numpy.asarray(bra_strs, dtype=numpy.uint64)
    ket_strs = bra_strs if ket_strs is None else numpy.asarray(ket_strs, dtype=numpy.uint64)
    order = numpy.argsort(bra_strs, kind='stable')
    sorted_bra = bra_strs[order]
    if len(sorted_bra) > 1 and (sorted_bra[1:] == sorted_bra[:-1]).any():
        raise ValueError('duplicate determinants in bra_strs')
    rows, cols, vals = [], [], []
    jcol = numpy.arange(len(ket_strs))
    for daggers, orbs, coeffs in op:
        coeffs = numpy.asarray(coeffs, dtype=numpy.complex128).ravel()
        orbs = numpy.asarray(orbs, dtype=numpy.int64).reshape(len(coeffs), len(daggers))
        for t0, t1 in _term_chunks(len(orbs), len(ket_strs)):
            new, sign = apply_opstring(daggers, orbs[t0:t1], ket_strs)
            pos = _lookup(sorted_bra, new)
            keep = (sign != 0) & (pos >= 0)
            rows.append(order[pos[keep]])
            cols.append(numpy.broadcast_to(jcol, new.shape)[keep])
            vals.append((sign * coeffs[t0:t1, None])[keep])
    shape = (len(bra_strs), len(ket_strs))
    if not rows:
        return sparse.csr_matrix(shape, dtype=numpy.complex128)
    return sparse.coo_matrix((numpy.concatenate(vals),
                              (numpy.concatenate(rows), numpy.concatenate(cols))),
                             shape=shape).tocsr()


def opstring_elements(daggers, orbs, bra, ket):
    '''<bra| o_1 ... o_k |ket> for every row of orbs; returns (nterm,).'''
    orbs = numpy.asarray(orbs, dtype=numpy.int64).reshape(-1, len(daggers))
    out = numpy.zeros(len(orbs), dtype=numpy.complex128)
    for t0, t1 in _term_chunks(len(orbs), len(ket)):
        new, sign = apply_opstring(daggers, orbs[t0:t1], ket.strs)
        out[t0:t1] = (bra.on(new).conj() * sign * ket.coeffs[None, :]).sum(axis=1)
    return out


def hamiltonian_operator(h1, eri, ecore=0.0, tol=0.0):
    '''H = ecore + sum h_pq p^+ q + 1/2 sum (pq|rs) p^+ r^+ s q as an operator.
    Terms with |coefficient| <= tol are skipped (tol = 0 keeps every
    non-zero term).'''
    h1 = numpy.asarray(h1)
    eri = numpy.asarray(eri)
    p, q = numpy.nonzero(abs(h1) > tol)
    op = [((), numpy.zeros((1, 0), dtype=numpy.int64), numpy.array([ecore]))]
    op.append(((True, False), numpy.stack([p, q], axis=1), h1[p, q]))
    p, q, r, s = numpy.nonzero(abs(eri) > tol)
    op.append(((True, True, False, False), numpy.stack([p, r, s, q], axis=1),
               0.5 * eri[p, q, r, s]))
    return op


def trans_rdm1(bra, ket, norb):
    '''t[p, q] = <bra| p^+ q |ket>'''
    p, q = numpy.indices((norb, norb)).reshape(2, -1)
    return opstring_elements((True, False), numpy.stack([p, q], axis=1),
                             bra, ket).reshape(norb, norb)


def trans_rdm2(bra, ket, norb):
    '''T[p, q, r, s] = <bra| p^+ r^+ s q |ket>'''
    p, q, r, s = numpy.indices((norb,) * 4).reshape(4, -1)
    return opstring_elements((True, True, False, False),
                             numpy.stack([p, r, s, q], axis=1),
                             bra, ket).reshape((norb,) * 4)
