'''
Generalized Fock matrix from the (state-averaged) active 1-RDM and
quasi-canonical core / virtual orbitals (conventions.md section 7).
'''

import numpy

from socutils.zmr import symmetry_sectors
from socutils.zmr.lib import linalg as la


def _sectors(eris, label):
    sym = getattr(eris, 'orbsym', None)
    return None if sym is None else numpy.asarray(sym)[eris.spaces.slice(label)]


def generalized_fock(eris, gamma_act):
    '''f_pq = h_pq + sum_i [(pq|ii) - (pi|iq)] + sum_tu [(pq|tu) - (pu|tq)] gamma_tu

    gamma_act[t, u] = <t^+ u>, averaged over the model states.  Evaluated by
    the Hamiltonian's own integral source.'''
    return eris.generalized_fock(gamma_act)


class QuasiCanonical:
    '''Result of quasi_canonicalize.

    Attributes:
        eris: Hamiltonian in the quasi-canonical orbitals
        fock: generalized Fock matrix in those orbitals
        eps_core, eps_virt: diagonal of the core-core / virtual-virtual blocks
        u: orbital rotation, new orbitals phi'_p = sum_q phi_q u_qp
    '''

    def __init__(self, eris, fock, eps_core, eps_virt, u):
        self.eris = eris
        self.fock = fock
        self.eps_core = eps_core
        self.eps_virt = eps_virt
        self.u = u


# blocks whose off-diagonal elements are all below this (Eh) count as diagonal
QUASI_CANONICAL_TOL = 1e-9


def quasi_canonicalize(eris, gamma_act):
    '''Diagonalize the core-core and virtual-virtual blocks of the generalized
    Fock matrix; the active orbitals are left untouched.  Orbitals that are
    already quasi-canonical (off-diagonal elements of both blocks below
    QUASI_CANONICAL_TOL, or the quasi_canonical_tol of the integral source)
    are kept as they are, without the four-index rotation of the integrals.'''
    sp = eris.spaces
    f = generalized_fock(eris, gamma_act)
    tol = getattr(eris, 'quasi_canonical_tol', QUASI_CANONICAL_TOL)
    off = 0.0
    for label in 'cv':
        s = sp.slice(label)
        b = f[s, s]
        if b.size:
            off = max(off, float(abs(b - numpy.diag(b.diagonal())).max()))
    if off < tol:
        c, v = sp.slice('c'), sp.slice('v')
        return QuasiCanonical(eris, f, f.diagonal()[c].real.copy(), f.diagonal()[v].real.copy(),
                              numpy.eye(sp.nmo, dtype=f.dtype))
    u = numpy.eye(sp.nmo, dtype=f.dtype)
    eps = {}
    for label in 'cv':
        s = sp.slice(label)
        # within the double-group sectors of the orbitals, if any
        e, v, _ = symmetry_sectors.eigh(f[s, s], _sectors(eris, label))
        u[s, s] = v
        eps[label] = e
    new = eris.rotate(u)
    return QuasiCanonical(new, u.conj().T @ f @ u, eps['c'], eps['v'], u)


def canonical_core(eris, gamma_act):
    '''(eps, u): eigenvalues and eigenvectors of the core-core block of the
    generalized Fock matrix.  The frozen core is to be chosen among these
    orbitals (mo[:, :ncore] @ u), not among the orbitals of the SCF or
    CASSCF, which are defined up to a rotation.'''
    c = eris.spaces.slice('c')
    e, u, _ = symmetry_sectors.eigh(generalized_fock(eris, gamma_act)[c, c], _sectors(eris, 'c'))
    return e, u


def shells(eps, tol=5e-3):
    '''Groups of nearly degenerate energies: [(first, last + 1), ...] for the
    sorted energies eps, split where consecutive energies differ by more than
    tol * max(1, |eps|).'''
    eps = numpy.asarray(eps)
    if len(eps) == 0:
        return []
    gaps = numpy.diff(eps)
    scale = tol * numpy.maximum(1.0, abs(eps[:-1]))
    edges = [0] + list(numpy.nonzero(gaps > scale)[0] + 1) + [len(eps)]
    return list(zip(edges[:-1], edges[1:]))


def orbital_report(eris, gamma_act, stdout=None, tol=5e-3, tol_virt=0.1):
    '''Print the quasi-canonical core and virtual orbital energies shell by
    shell, with the gap above each shell: what one looks at to choose the
    frozen core (nfrozen = a value of the column "below") and the virtual
    orbitals to keep (truncate_virtuals(..., nkeep=) or emax).  Call it for
    the Hamiltonian without frozen core and with all virtuals.  Shells are
    separated where the gap exceeds tol * max(1, |eps|); the coarser tol_virt
    keeps the table of the virtuals short.  Returns (eps_core, eps_virt).'''
    import sys
    out = stdout or sys.stdout
    sp = eris.spaces
    f = generalized_fock(eris, gamma_act)
    eps = {}
    for label, name in (('c', 'core'), ('v', 'virtual')):
        s = sp.slice(label)
        e = (symmetry_sectors.eigh(f[s, s], _sectors(eris, label))[0] if s.stop > s.start
             else numpy.zeros(0))
        eps[label] = e
        out.write('\n Quasi-canonical %s orbitals (%d), energies in Eh\n' % (name, len(e)))
        out.write(' %7s %6s %8s  %14s %14s %12s\n'
                  % ('orbital', 'count', 'below', 'lowest', 'highest', 'gap above'))
        groups = shells(e, tol if label == 'c' else tol_virt)
        for n, (i, j) in enumerate(groups):
            gap = '%12.4f' % (e[j] - e[j - 1]) if j < len(e) else '%12s' % '-'
            out.write(' %3d-%-3d %6d %8d  %14.6f %14.6f %s\n'
                      % (i + 1, j, j - i, j, e[i], e[j - 1], gap))
    out.write('\n "below": orbitals up to and including this shell '
              '(nfrozen for the core, nkeep for the virtuals)\n')
    out.flush()
    return eps['c'], eps['v']


def truncate_virtuals(eris, gamma_act, emax=None, window=0.2, nkeep=None):
    '''Remove the virtual orbitals with energies above emax from the
    correlated space (the high-lying virtuals of an uncontracted basis), or
    keep exactly the nkeep lowest ones (a choice made by looking at
    orbital_report).

    The energies are the eigenvalues of the virtual-virtual block of the
    generalized Fock matrix for the active density gamma_act; the kept
    orbitals are its eigenvectors.  The cut is placed in the largest gap
    between consecutive energies that lies within emax * (1 -+ window), so
    that it never separates nearly degenerate orbitals (the components of an
    atomic shell).

    Returns (eris', kept energies, removed energies).'''
    sp = eris.spaces
    v = sp.slice('v')
    e, c, _ = symmetry_sectors.eigh(generalized_fock(eris, gamma_act)[v, v], _sectors(eris, 'v'))
    n = len(e)
    if nkeep is not None:
        if not 0 <= nkeep <= n:
            raise ValueError('nkeep = %d, there are %d virtual orbitals' % (nkeep, n))
        return eris.with_virtuals(c[:, :nkeep]), e[:nkeep], e[nkeep:]
    if emax is None:
        raise ValueError('give emax or nkeep')
    keep = int(numpy.searchsorted(e, emax))
    if keep < n:
        lo, hi = emax * (1 - window), emax * (1 + window)
        cuts = [k for k in range(1, n) if e[k] > lo and e[k - 1] < hi]
        if cuts:
            keep = max(cuts, key=lambda k: e[k] - e[k - 1])
    return eris.with_virtuals(c[:, :keep]), e[:keep], e[keep:]
