'''
One-body symmetries of the active-space Hamiltonian and symmetry-adapted
active orbitals.

A one-body operator A = sum_tu A_tu t^+ u commutes with

    H_act = sum_tu h_tu t^+ u + 1/4 sum_tvuw g_tvuw t^+ v^+ w u,
    g_tvuw = <tv||uw> = (tu|vw) - (tw|vu),

if and only if

    [A, h] = 0,
    sum_p (A_tp g_pvuw + A_vp g_tpuw - g_tvpw A_pu - g_tvup A_pw) = 0.

These equations are linear in A.  Their Hermitian solutions form the Lie
algebra of the one-body symmetry group of H_act: the particle number always,
J_x, J_y, J_z for an atomic shell, J_z for a linear molecule, the spin
components for spin-free integrals.  A generic element of that algebra lies
in one Cartan subalgebra (maximal set of commuting generators); in its
eigenbasis every determinant has a definite charge (the sum of the orbital
eigenvalues), H_act is block diagonal in the charge, and so are the sector
Hamiltonians, the Gram matrices of the perturber functions and the Koopmans
matrices of every class.

find() returns the adapted orbitals and their charges; adapt() rotates the
active orbitals of an eris object and records the charges, after which
ActiveSpace removes the charge-violating integrals (they vanish up to the
accuracy to which the input integrals are symmetric; the largest removed
element is reported) and all dense eigenproblems of the active space are
solved block by block (zmr.lib.blocks).  Results are unchanged: NEVPT2 with
the Dyall Hamiltonian is invariant under rotations of the active orbitals.
'''

import numpy

from socutils.zmr import lib as zlib

# charges closer than this (relative to their spread) are the same charge
CHARGE_TOL = 1e-6


class ActiveSymmetry:
    '''Symmetry-adapted active orbitals.

    Attributes:
        u: (nact, nact) unitary, adapted orbitals phi'_t = sum_s phi_s u_st
        charges: (nact,) eigenvalues of the generic symmetry generator; equal
            charges are exactly equal
        ngen: dimension of the symmetry algebra (1: particle number only)
        error: largest charge-violating element of h or <tv||uw> in the
            adapted orbitals
    '''

    def __init__(self, u, charges, ngen, error):
        self.u = u
        self.charges = charges
        self.ngen = ngen
        self.error = error


def antisymmetrized(aaaa):
    '''g[t, v, u, w] = <tv||uw> = (tu|vw) - (tw|vu)'''
    return aaaa.transpose(0, 2, 1, 3) - aaaa.transpose(0, 2, 3, 1)


def _clusters(values, tol):
    '''Groups of indices whose sorted values are chained by gaps <= tol.'''
    order = numpy.argsort(values, kind='stable')
    cuts = numpy.nonzero(numpy.diff(values[order]) > tol)[0] + 1
    return numpy.split(order, cuts)


def _commutator_gram(e, g, clusters):
    '''Gram matrix of the linear map A -> ([A, h], D_A g) on the matrices A
    that are block diagonal in the clusters, in a basis where h = diag(e).
    Returns (gram, index pairs (a, b) of the basis matrices E_ab).'''
    n = len(e)
    pairs = numpy.array([(a, b) for c in clusters for a in c for b in c])
    npar = len(pairs)
    gram = numpy.zeros((npar, npar), dtype=numpy.complex128)
    # one-body part: [E_ab, h] = (e_b - e_a) E_ab
    gram[numpy.arange(npar), numpy.arange(npar)] = (e[pairs[:, 1]] - e[pairs[:, 0]])**2
    # two-body part, accumulated over slabs of the first index t
    slab = numpy.empty((npar, n, n, n), dtype=numpy.complex128)
    for t in range(n):
        slab[:] = 0
        for k, (a, b) in enumerate(pairs):
            if a == t:
                slab[k] += g[b]
            slab[k, a, :, :] += g[t, b, :, :]
            slab[k, :, b, :] -= g[t, :, a, :]
            slab[k, :, :, b] -= g[t, :, :, a]
        m = slab.reshape(npar, n**3)
        gram += m.conj() @ m.T
    return gram, pairs


def allowed_masks(charges, tol=CHARGE_TOL):
    '''Charge-conserving elements of h[t, u] and (tu|vw).'''
    q = numpy.asarray(charges, dtype=float)
    spread = max(q.max() - q.min(), 1.0) if len(q) else 1.0
    d1 = q[:, None] - q[None, :]
    mask1 = abs(d1) <= tol * spread
    mask2 = abs(d1[:, :, None, None] + d1[None, None, :, :]) <= tol * spread
    return mask1, mask2


def violation(heff, aaaa, charges):
    '''Largest charge-violating element of h and <tv||uw>.'''
    mask1, mask2 = allowed_masks(charges)
    g = antisymmetrized(aaaa)
    forbidden = ~mask2.transpose(0, 2, 1, 3)
    return max(abs(heff[~mask1]).max(initial=0.0), abs(g[forbidden]).max(initial=0.0))


def project(heff, aaaa, charges):
    '''(h, (tu|vw)) with the charge-violating elements set to zero.'''
    mask1, mask2 = allowed_masks(charges)
    return numpy.where(mask1, heff, 0), numpy.where(mask2, aaaa, 0)


def _snap(values, tol):
    '''Replace nearly equal values by their mean.'''
    out = numpy.array(values, dtype=float)
    for c in _clusters(out, tol):
        out[c] = out[c].mean()
    return out


def find(heff, aaaa, tol=1e-8, seed=0):
    '''Symmetry-adapted active orbitals, or None if the particle number is
    the only one-body symmetry of the Hamiltonian.

    Args:
        heff, aaaa: active-space Hamiltonian (ActiveSpace conventions)
        tol: accuracy to which a symmetry has to hold: the largest
            charge-violating integral element (hartree) that may be
            discarded.  A symmetry broken by more than that is not used.
        seed: seed of the generic combination of the generators
    '''
    heff = numpy.asarray(heff, dtype=numpy.complex128)
    n = heff.shape[0]
    if n < 2:
        return None
    aaaa = numpy.asarray(aaaa, dtype=numpy.complex128).reshape((n,) * 4)
    e, c = numpy.linalg.eigh(heff)
    scale = max(e[-1] - e[0], abs(antisymmetrized(aaaa)).max(), 1e-300)
    # levels that may be degenerate up to the accepted symmetry breaking
    clusters = _clusters(e, max(1e-6 * scale, 10 * tol))
    # integrals in the eigenbasis of heff
    eri = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', aaaa, c.conj(), c, c.conj(), c, optimize=True)
    gram, pairs = _commutator_gram(e, antisymmetrized(eri), clusters)
    w, x = numpy.linalg.eigh(gram)
    wmax = max(w[-1], 1e-300)
    rng = numpy.random.default_rng(seed)
    # generators: null space of the Gram matrix, i.e. commutator norm below
    # ~10 tol (but not below the round-off of the Gram matrix).  A generator
    # that turns out to be broken by more than tol is removed by tightening
    # the threshold.
    start = max((10 * tol)**2 / wmax, 1e-10)
    for thr in (start, 1e-2 * start, 1e-4 * start):
        null = x[:, w <= max(thr, 1e-13) * wmax]
        ngen = null.shape[1]
        if ngen <= 1:
            return None
        # generic Hermitian element of the algebra, without its trace
        a0 = numpy.zeros((n, n), dtype=numpy.complex128)
        for k in range(ngen):
            ak = numpy.zeros((n, n), dtype=numpy.complex128)
            ak[pairs[:, 0], pairs[:, 1]] = null[:, k]
            a0 += rng.normal() * (ak + ak.conj().T) + rng.normal() * 1j * (ak - ak.conj().T)
        a0 -= numpy.trace(a0) / n * numpy.eye(n)
        norm = numpy.linalg.norm(a0, 2)
        if norm < 1e-12:
            continue
        a0 /= norm
        # diagonalize it inside every cluster, so that heff stays diagonal
        u = numpy.zeros((n, n), dtype=numpy.complex128)
        charges = numpy.zeros(n)
        for idx in clusters:
            q, v = numpy.linalg.eigh(a0[numpy.ix_(idx, idx)])
            u[numpy.ix_(idx, idx)] = v
            charges[idx] = q
        u = c @ u
        charges = _snap(charges, CHARGE_TOL * max(charges.max() - charges.min(), 1.0))
        if len(numpy.unique(charges)) == 1:
            return None
        h2 = u.conj().T @ heff @ u
        e2 = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', aaaa, u.conj(), u, u.conj(), u,
                          optimize=True)
        error = violation(h2, e2, charges)
        if error <= tol:
            return ActiveSymmetry(u, charges, ngen, error)
    return None


@zlib.single_threaded_numpy
def adapt(eris, tol=1e-8, seed=0):
    '''Rotate the active orbitals of eris to symmetry-adapted ones.

    Returns (eris', symmetry): eris' carries active_charges, which makes
    ActiveSpace drop the charge-violating active integrals; symmetry is the
    ActiveSymmetry found, or None (then eris' is eris).'''
    heff, aaaa, _ = eris.active_hamiltonian()
    sym = find(heff, aaaa, tol=tol, seed=seed)
    if sym is None:
        return eris, None
    sp = eris.spaces
    u = numpy.eye(sp.nmo, dtype=numpy.complex128)
    a = sp.slice('a')
    u[a, a] = sym.u
    new = eris.rotate(u)
    new.active_charges = sym.charges
    return new, sym
