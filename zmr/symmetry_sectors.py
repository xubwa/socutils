'''
Double-group symmetry sectors of the orbitals.

The sector of an orbital is its irrep id of socutils.symm.double_group
(ir = 2 n + p: n the exponent of the cyclic generator, modulo 2 or 4 or unbounded
for the linear groups, p the parity), as tagged on the mo_coeff of a socutils
double-group SCF or CASSCF (mo_coeff.orbsym) together with mf.doublegroupname.
Every rule about sectors - which products of orbitals are totally symmetric,
the sector of a pair or of an operator - is taken from
socutils.symm.double_group (direct_prod, conj); nothing else in zmr adds or
reduces irrep ids itself.

An integral source carries the sectors of its orbitals as eris.orbsym (core,
active, virtual) and eris.doublegroupname; None means no symmetry.  A rotation
keeps them if it does not mix sectors (column_sectors), which the
diagonalizations of this module guarantee.
'''

import numpy

from socutils.symm import double_group as dg
from socutils.zmr.lib import linalg as la

# elements of a rotation below this count as zero when its sectors are determined
TOL = 1e-8


def eigh(h, orbsym=None, grouped=False):
    '''Eigenvalues and eigenvectors of the Hermitian matrix h given in orbitals of the
    sectors orbsym; every eigenvector lies in one sector.  Returns (e, v, sectors of the
    eigenvectors), ascending in e (degenerate eigenvalues of different sectors, e.g.
    Kramers partners, ordered by their sector), or with grouped sector by sector (the
    sectors ascending, e ascending within each: every sector a range); with orbsym None
    the plain eigenproblem and None.'''
    h = numpy.ascontiguousarray(h)
    if orbsym is None:
        e, v = la.eigh(h)
        return e, v, None
    orbsym = numpy.asarray(orbsym)
    n = orbsym.size
    vals, irs, cols = [], [], []
    v = numpy.zeros((n, n), dtype=numpy.result_type(h.dtype, numpy.complex128))
    for ir in numpy.unique(orbsym):
        idx = numpy.nonzero(orbsym == ir)[0]
        w, x = la.eigh(numpy.ascontiguousarray(h[numpy.ix_(idx, idx)]))
        for k in range(len(w)):
            vals.append(w[k])
            irs.append(ir)
            cols.append((idx, x[:, k]))
    keys = (numpy.asarray(irs), numpy.asarray(vals))
    order = numpy.lexsort(keys[::-1] if grouped else keys)
    e = numpy.asarray(vals)[order]
    for j, k in enumerate(order):
        idx, x = cols[k]
        v[idx, j] = x
    return e, v, numpy.asarray(irs)[order]


def column_sectors(u, orbsym, tol=TOL):
    '''Sectors of the orbitals phi'_j = sum_i phi_i u_ij, or None if a column mixes
    sectors (or orbsym is None).  A column without elements above tol takes the
    sector of its largest element.'''
    if orbsym is None:
        return None
    orbsym = numpy.asarray(orbsym)
    u = numpy.asarray(u)
    if u.shape[0] != orbsym.size:
        raise ValueError('u has %d rows for %d orbitals' % (u.shape[0], orbsym.size))
    out = numpy.empty(u.shape[1], dtype=orbsym.dtype)
    for j in range(u.shape[1]):
        a = abs(u[:, j])
        big = a > tol
        labs = numpy.unique(orbsym[big]) if big.any() else orbsym[[int(numpy.argmax(a))]]
        if labs.size != 1:
            return None
        out[j] = labs[0]
    return out


def keep(new, old, u):
    '''Give the integral source new (orbitals phi_i of old rotated by u) the sectors of
    its orbitals; drops the symmetry if u mixes sectors.  Returns new.'''
    sym = column_sectors(u, getattr(old, 'orbsym', None))
    new.orbsym = sym
    new.doublegroupname = getattr(old, 'doublegroupname', None) if sym is not None else None
    return new


def totally_symmetric(irreps, groupname):
    '''True if the direct product of the irreps is the totally symmetric irrep.'''
    out = 0
    for ir in irreps:
        out = dg.direct_prod(out, int(ir), groupname)
    return int(out) == 0


def violation(block, sectors, groupname):
    '''Largest |element| of a four-index block X[p, q, r, s] (storage of zmr.dsrg:
    p, q annihilated, r, s created) whose orbitals do not conserve symmetry:
    conj(p) x conj(q) x r x s not totally symmetric.  sectors: the sectors of the four
    index ranges.'''
    sp, sq, sr, ss = (numpy.asarray(x) for x in sectors)
    worst = 0.0
    for a in numpy.unique(sp):
        for b in numpy.unique(sq):
            for c in numpy.unique(sr):
                for d in numpy.unique(ss):
                    ca, cb = dg.conj(int(a), groupname), dg.conj(int(b), groupname)
                    if totally_symmetric((ca, cb, c, d), groupname):
                        continue
                    sub = block[numpy.ix_(sp == a, sq == b, sr == c, ss == d)]
                    if sub.size:
                        worst = max(worst, float(abs(sub).max()))
    return worst
