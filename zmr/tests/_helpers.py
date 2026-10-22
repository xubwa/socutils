'''Shared helpers for the test suite.'''
import numpy


def random_hamiltonian(norb, seed=0, scale=1.0):
    '''Random complex h1 and (pq|rs) with exactly the symmetries of
    conventions.md section 2 and nothing more:
    h = h^+, (pq|rs) = (rs|pq), (pq|rs)^* = (qp|sr).'''
    rng = numpy.random.default_rng(seed)
    h = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    h = (h + h.conj().T) * scale
    a = rng.normal(size=(norb,) * 4) + 1j * rng.normal(size=(norb,) * 4)
    eri = a + a.transpose(2, 3, 0, 1)
    eri = (eri + eri.transpose(1, 0, 3, 2).conj()) * (0.25 * scale)
    return h, eri


def random_vector(n, seed=0):
    rng = numpy.random.default_rng(seed)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return v / numpy.linalg.norm(v)


def model_eris(ncore, nact, nvirt, seed=0, coupling=0.08):
    '''Random complex Hamiltonian with a physical orbital hierarchy (core
    low, active near zero, virtual high) and weak random complex couplings,
    so that every perturber class is non-zero and there are no intruders.'''
    from socutils.zmr.integrals import ERIs
    from socutils.zmr.spaces import OrbitalSpaces
    sp = OrbitalSpaces(ncore, nact, nvirt)
    n = sp.nmo
    rng = numpy.random.default_rng(seed)
    h, eri = random_hamiltonian(n, seed=seed, scale=coupling)
    level = numpy.concatenate([rng.uniform(-3.0, -2.0, ncore),
                               rng.uniform(-0.4, 0.4, nact),
                               rng.uniform(1.5, 2.5, nvirt)])
    h = h + numpy.diag(level)
    # a positive "Coulomb-like" diagonal keeps the two-electron part realistic
    for p in range(n):
        for q in range(n):
            eri[p, p, q, q] += 0.3
    return ERIs(-1.0, h, eri, sp)


def spinfree_model_eris(ncore, nact, nvirt, seed=0, coupling=0.08, hide=True):
    '''Spin-orbital Hamiltonian of a random spin-free one with ncore, nact,
    nvirt spatial orbitals (every space has twice as many spin orbitals).  It
    commutes with the three spin components.  With hide=True the spin
    orbitals of every space are mixed by a random unitary, which hides the
    symmetry (nothing is block diagonal any more) without removing it.'''
    from scipy.stats import unitary_group
    from socutils.zmr.integrals import ERIs
    from socutils.zmr.spaces import OrbitalSpaces
    n = ncore + nact + nvirt
    rng = numpy.random.default_rng(seed)
    h, eri = random_hamiltonian(n, seed=seed, scale=coupling)
    level = numpy.concatenate([rng.uniform(-3.0, -2.0, ncore),
                               rng.uniform(-0.4, 0.4, nact),
                               rng.uniform(1.5, 2.5, nvirt)])
    h = h + numpy.diag(level)
    for p in range(n):
        for q in range(n):
            eri[p, p, q, q] += 0.3
    spin = numpy.eye(2)
    h_so = numpy.kron(h, spin)
    eri_so = numpy.einsum('pqrs,ab,cd->paqbrcsd', eri, spin, spin).reshape((2 * n,) * 4)
    sp = OrbitalSpaces(2 * ncore, 2 * nact, 2 * nvirt)
    out = ERIs(-1.0, h_so, eri_so, sp)
    if hide:
        u = numpy.zeros((2 * n, 2 * n), dtype=complex)
        for k, label in enumerate('cav'):
            sl = sp.slice(label)
            m = sl.stop - sl.start
            if m:
                u[sl, sl] = unitary_group.rvs(m, random_state=seed + 100 + k) if m > 1 else 1.0
        out = out.rotate(u)
    return out


def complete_levels(lam, nmin, gap=1e-6):
    '''Smallest number of states >= nmin that does not cut a degenerate level
    of the ascending spectrum lam.'''
    for n in range(nmin, len(lam)):
        if lam[n] - lam[n - 1] > gap:
            return n
    return len(lam)


def nr_model_eris(ncore, nact, nvirt, seed=0, coupling=0.25):
    '''Random real Hamiltonian over spatial orbitals with the eightfold
    symmetry of real orbitals (a density-fitting-like sum of squares), a
    physical orbital hierarchy and no intruders (zmr.nr.integrals.ERIs).'''
    from socutils.zmr.nr.integrals import ERIs
    from socutils.zmr.spaces import OrbitalSpaces
    sp = OrbitalSpaces(ncore, nact, nvirt)
    n = sp.nmo
    rng = numpy.random.default_rng(seed)
    level = numpy.concatenate([rng.uniform(-3.0, -2.0, ncore),
                               rng.uniform(-0.4, 0.4, nact),
                               rng.uniform(1.5, 2.5, nvirt)])
    h = rng.normal(size=(n, n)) * 0.08
    h = h + h.T + numpy.diag(level)
    b = rng.normal(size=(2 * n, n, n)) * coupling / numpy.sqrt(n)
    b = b + b.transpose(0, 2, 1)
    b[0] += numpy.eye(n) * 0.5                 # Coulomb-like diagonal
    eri = numpy.einsum('lpq,lrs->pqrs', b, b)
    return ERIs(-1.0, h, eri, sp)
