'''
Time reversal (Kramers pairs) in an orthonormal spinor basis.

An integral source may carry eris.time_reversal, a TimeReversal of its correlated
orbitals: T|p> = phase[p] |partner[p]>, with T antiunitary and T^2 = -1.  It comes
from the SCF or CASSCF orbitals (from_orbitals; a Kramers-restricted SCF gives
interleaved pairs) and is kept by every rotation that maps Kramers pairs onto Kramers
pairs (rotated, through symmetry_sectors.keep).  quasi-canonicalization keeps the pairs
(paired_eigh) when the averaged Fock matrix is time-reversal symmetric, which it is for
an ensemble that holds both members of every Kramers doublet.

With paired core and virtual orbitals the external sums of a perturbation theory are
equal for a label and its time-reversed label, so only one of the two is computed
(e.g. the core pairs (i, j) and (Ti, Tj) of ccvv).
'''

import numpy


class TimeReversal:
    '''T|p> = phase[p] |partner[p]> for orbitals p = 0 .. n-1.'''

    def __init__(self, partner, phase):
        self.partner = numpy.asarray(partner, dtype=numpy.int64)
        self.phase = numpy.asarray(phase, dtype=numpy.complex128)
        n = self.partner.size
        if self.phase.size != n or sorted(self.partner) != list(range(n)):
            raise ValueError('partner must be a permutation with one phase per orbital')
        if numpy.any(self.partner == numpy.arange(n)):
            raise ValueError('an orbital cannot be its own Kramers partner')
        # T^2 = -1: conj(phase[p]) phase[partner[p]] = -1
        if abs(self.phase.conj() * self.phase[self.partner] + 1).max() > 1e-6:
            raise ValueError('the phases do not give T^2 = -1')

    @property
    def size(self):
        return self.partner.size

    @classmethod
    def from_orbitals(cls, mol, mo_coeff, tol=1e-6):
        '''Time reversal of the orbitals mo_coeff (two-component spinors of mol), or None
        if some orbital has no Kramers partner among them.'''
        from socutils.scf import double_group_symm as dgs
        partner, _, phase = dgs.kramers_pairs(mol, mo_coeff, tol=tol, return_phase=True)
        if numpy.any(partner < 0):
            return None
        # kramers_pairs gives phase[i] = <partner|T|i>
        return cls(partner, phase)

    @classmethod
    def interleaved(cls, n):
        '''Pairs (2k, 2k+1) with T|2k> = |2k+1> (hence T|2k+1> = -|2k>).'''
        if n % 2:
            raise ValueError('an odd number of orbitals cannot be Kramers paired')
        partner = numpy.arange(n) ^ 1
        phase = numpy.where(numpy.arange(n) % 2 == 0, 1.0, -1.0).astype(complex)
        return cls(partner, phase)

    def matrix(self):
        '''Theta with T (sum_p c_p |p>) = sum_q (Theta conj(c))_q |q>.'''
        n = self.size
        th = numpy.zeros((n, n), dtype=numpy.complex128)
        th[self.partner, numpy.arange(n)] = self.phase
        return th

    def apply(self, c):
        '''Coefficients of T applied to the vectors c (columns, or one vector).'''
        c = numpy.asarray(c)
        out = numpy.zeros(c.shape, dtype=numpy.complex128)
        out[self.partner] = self.phase.reshape((-1,) + (1,) * (c.ndim - 1)) * c.conj()
        return out

    def rotated(self, u, tol=1e-8):
        '''Time reversal of the orbitals phi'_j = sum_p phi_p u_pj (u with orthonormal
        columns), or None if T does not map each new orbital onto one new orbital.'''
        u = numpy.asarray(u)
        m = u.conj().T @ self.apply(u)                 # m[k, j] = <phi'_k|T|phi'_j>
        n = m.shape[1]
        if m.shape[0] != n:
            return None
        partner = numpy.argmax(abs(m), axis=0)
        phase = m[partner, numpy.arange(n)]
        if abs(abs(phase) - 1).max() > tol or sorted(partner) != list(range(n)):
            return None
        try:
            return TimeReversal(partner, phase / abs(phase))
        except ValueError:
            return None

    def subset(self, idx):
        '''Time reversal of the orbitals idx (all partners among them), else None.'''
        idx = numpy.asarray(idx, dtype=numpy.int64)
        pos = numpy.full(self.size, -1)
        pos[idx] = numpy.arange(idx.size)
        partner = pos[self.partner[idx]]
        if numpy.any(partner < 0):
            return None
        return TimeReversal(partner, self.phase[idx])

    def symmetric(self, h, tol=1e-8):
        '''Whether the Hermitian matrix h commutes with T (Theta conj(h) Theta^+ = h).'''
        th = self.matrix()
        return float(abs(th @ numpy.conj(h) @ th.conj().T - h).max()) < tol * max(
            1.0, float(abs(h).max()))

    def representatives(self):
        '''One orbital of each pair (the lower index).'''
        return numpy.nonzero(numpy.arange(self.size) < self.partner)[0]


def keep(new, old, u):
    '''Give the integral source new (orbitals of old rotated by u) the time reversal of
    its orbitals, None if u does not keep the Kramers pairs.'''
    tr = getattr(old, 'time_reversal', None)
    new.time_reversal = None if tr is None else tr.rotated(u)
    return new


def paired_eigh(h, tr, orbsym=None, deg_tol=1e-7):
    '''Eigenvalues and eigenvectors of the time-reversal symmetric Hermitian h, the
    eigenvectors as Kramers pairs (v, Tv) in consecutive columns (eigenvalues
    ascending, the two of a pair equal), within the double-group sectors orbsym if
    given.  Returns (e, v), or None if h is not time-reversal symmetric.'''
    from socutils.zmr import symmetry_sectors
    if not tr.symmetric(h, tol=1e-9):
        return None
    e, v, _ = symmetry_sectors.eigh(h, orbsym)
    n = len(e)
    out_e, out_v = [], []
    k = 0
    while k < n:
        # the degenerate group starting at k
        g = k + 1
        while g < n and e[g] - e[g - 1] < deg_tol * max(1.0, abs(e[k])):
            g += 1
        if (g - k) % 2:
            return None
        space = v[:, k:g]
        left = space.copy()
        ee = float(numpy.mean(e[k:g]))
        for _ in range((g - k) // 2):
            norms = numpy.linalg.norm(left, axis=0)
            x = left[:, int(numpy.argmax(norms))]
            x = x / numpy.linalg.norm(x)
            tx = tr.apply(x)
            # Tx must lie in the degenerate space
            if numpy.linalg.norm(space.conj().T @ tx) < 1 - 1e-6:
                return None
            pair = numpy.stack([x, tx], axis=1)
            left = left - pair @ (pair.conj().T @ left)
            out_v.append(pair)
            out_e += [ee, ee]
        k = g
    vv = numpy.concatenate(out_v, axis=1) if out_v else numpy.zeros((n, 0), dtype=complex)
    return numpy.asarray(out_e), vv
