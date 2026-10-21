'''Small benchmark crystals and an all-electron basis helper for the periodic
J/K accuracy and cost benchmarks in this directory.'''
import numpy as np
from pyscf import gto as mgto
from pyscf.pbc import gto as pgto

BOHR = 0.52917721092


def hexagonal(a, c):
    return np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])


def expand(A, sites, ops):
    '''Apply symmetry operations (fractional) to the asymmetric unit; Cartesian output.'''
    out = []
    for el, f in sites:
        for op in ops:
            g = np.mod(np.array(op(*f)), 1.0)
            if not any(e == el and np.allclose(np.mod(g - h + .5, 1) - .5, 0, atol=1e-4) for e, h in out):
                out.append((el, g))
    return [(e, g @ A) for e, g in out]


# alpha-quartz, P3_2 21, COD 9013321 (a = 4.913437, c = 5.405118 A); 9 atoms
_q_ops = [lambda x, y, z: (x, y, z), lambda x, y, z: (y, x, -z),
          lambda x, y, z: (-y, x - y, 2 / 3 + z), lambda x, y, z: (-x, -x + y, 2 / 3 - z),
          lambda x, y, z: (-x + y, -x, 1 / 3 + z), lambda x, y, z: (x - y, -y, 1 / 3 - z)]
_QA = hexagonal(4.913437, 5.405118)
QUARTZ = (_QA, expand(_QA, [('Si', (0.47, 0, 2 / 3)), ('O', (0.4146, 0.2678, 0.78543))], _q_ops))

# A-type (La2O3-type) Y2O3, P-3m1; 5 atoms.  Lattice and u, v are approximate
# (scaled from La2O3): good enough for numerical benchmarks, not for physics.
_YA = hexagonal(3.70, 5.95)
_frac = [('Y', (1 / 3, 2 / 3, 0.245)), ('Y', (2 / 3, 1 / 3, 0.755)), ('O', (0, 0, 0)),
         ('O', (1 / 3, 2 / 3, 0.645)), ('O', (2 / 3, 1 / 3, 0.355))]
Y2O3_A = (_YA, [(e, np.array(f) @ _YA) for e, f in _frac])


def _yso_primitive():
    '''X2-Y2SiO5 (C2/c in the B 1 1 2/b setting), Michel, Buisson, Bertaut 1967
    (COD 1001838).  Low-precision 1967 coordinates (O to ~0.1 A); benchmarks only.
    Returns the 32-atom primitive cell.'''
    a, b, c, gam = 14.59, 10.52, 6.82, np.radians(122.25)
    A = np.array([[a, 0, 0], [b * np.cos(gam), b * np.sin(gam), 0], [0, 0, c]])
    sites = [('Y', (0.537, -0.031, 0.241)), ('Y', (0.357, -0.335, 0.880)), ('Si', (0.316, -0.304, 0.414)),
             ('O', (0.374, -0.334, 0.213)), ('O', (0.407, -0.156, 0.492)), ('O', (0.204, -0.325, 0.372)),
             ('O', (0.312, -0.404, 0.594)), ('O', (0.519, -0.106, 0.915))]
    ops = [lambda x, y, z: (x, y, z), lambda x, y, z: (-x, .5 - y, z), lambda x, y, z: (-x, -y, -z),
           lambda x, y, z: (x, .5 + y, -z), lambda x, y, z: (.5 + x, y, .5 + z),
           lambda x, y, z: (.5 - x, .5 - y, .5 + z), lambda x, y, z: (.5 - x, -y, .5 - z),
           lambda x, y, z: (.5 + x, .5 + y, .5 - z)]
    conv = expand(A, sites, ops)            # 64 atoms
    P = np.array([A[0], A[1], (A[0] + A[2]) / 2])  # centring (1/2, 0, 1/2)
    Pinv = np.linalg.inv(P)
    prim = []
    for el, x in conv:
        f = np.mod(x @ Pinv, 1.0)
        if not any(e == el and np.allclose(np.mod(f - g + .5, 1) - .5, 0, atol=1e-4) for e, g in prim):
            prim.append((el, f))
    return P, [(e, f @ P) for e, f in prim]


YSO = _yso_primitive()


def substitute(system, element='Eu', host='Y', index=0):
    '''The structure with the index-th host atom replaced (dopant on a host site).'''
    A, atoms = system
    out, k = [], 0
    for e, x in atoms:
        if e == host:
            if k == index:
                e = element
            k += 1
        out.append((e, x))
    return A, out


def _reduced_basis(L, nmax=6):
    '''Three shortest lattice vectors of L spanning the same lattice.'''
    import itertools
    T = np.array(list(itertools.product(range(-nmax, nmax + 1), repeat=3)))
    V = T @ L
    order = np.argsort(np.linalg.norm(V, axis=1))
    vol = abs(np.linalg.det(L))
    basis = []
    for i in order:
        if np.linalg.norm(V[i]) < 1e-8:
            continue
        trial = basis + [V[i]]
        if len(trial) < 3:
            if np.linalg.matrix_rank(np.array(trial), tol=1e-6) == len(trial):
                basis = trial
        elif abs(abs(np.linalg.det(np.array(trial))) - vol) < 1e-6 * vol:
            basis = trial
            break
    B = np.array(basis)
    return B if np.linalg.det(B) > 0 else B[[1, 0, 2]]


def supercell(system, M):
    '''Supercell with lattice vectors M @ A (M an integer matrix, or a mesh
    (n1, n2, n3) for a diagonal one), returned on a reduced (short) basis.'''
    A, atoms = system
    A = np.asarray(A, dtype=float)
    M = np.asarray(M)
    if M.ndim == 1:
        M = np.diag(M)
    L = _reduced_basis(M @ A)
    n = round(abs(np.linalg.det(M)))
    Linv = np.linalg.inv(L)
    span = int(np.abs(M).sum()) + 1
    pts = []
    for t in np.array(np.meshgrid(*[np.arange(-span, span + 1)] * 3, indexing='ij')).reshape(3, -1).T @ A:
        f = t @ Linv
        if np.all(f > -1e-8) and np.all(f < 1 - 1e-8):
            pts.append(t)
    assert len(pts) == n, (len(pts), n)
    out = []
    for t in pts:
        for e, x in atoms:
            out.append((e, np.mod((np.asarray(x) + t) @ Linv, 1.) @ L))
    return L, out


def min_image_distance(system, element):
    '''Shortest distance (A) between atoms of element and their periodic images
    (or each other): the dopant-dopant separation of a doped supercell.'''
    A, atoms = system
    A = np.asarray(A, dtype=float)
    X = np.array([x for e, x in atoms if e == element])
    T = np.array([[i, j, k] for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)
                  for k in (-2, -1, 0, 1, 2)]) @ A
    d = np.linalg.norm(X[:, None, None, :] + T[None, None] - X[None, :, None, :], axis=-1)
    d[d < 1e-6] = np.inf
    return d.min()


def from_cif(path, primitive=True, tol=0.01):
    '''(A, atoms) from a CIF (pymatgen): primitive cell (or the conventional cell
    of the file), Cartesian coordinates in Angstrom, plain element symbols.'''
    from pymatgen.core import Structure
    st = Structure.from_file(path)
    if not st.is_ordered:
        raise ValueError('%s: partially occupied sites; resolve the disorder first' % path)
    if primitive:
        st = st.get_primitive_structure(tolerance=tol)
    A = np.array(st.lattice.matrix)
    return A, [(sp.specie.symbol, np.array(sp.coords)) for sp in st]


def from_cif_relaxed(cif, xyz, tol=0.05):
    '''Cell of a CIF (as read by ASE, no reduction) with the positions of the
    first atoms of a relaxed supercell xyz built from it (e.g. the CP2K H-only
    r2SCAN optimisations of the Eu hydrates: the first image of the 2x2x1
    cell, as the clusters use); heavy atoms are checked against the CIF.'''
    from ase.io import read
    prim = read(cif)
    sup = read(xyz, index=-1)
    n = len(prim)
    sym = prim.get_chemical_symbols()
    assert sup.get_chemical_symbols()[:n] == sym
    heavy = [i for i, e in enumerate(sym) if e != 'H']
    dev = np.linalg.norm(sup.positions[heavy] - prim.positions[heavy], axis=1).max()
    if dev > tol:
        raise ValueError('heavy atoms of %s deviate %.3f A from %s' % (xyz, dev, cif))
    return np.array(prim.cell.array), [(e, np.array(x)) for e, x in zip(sym, sup.positions[:n])]


def _hnf(n):
    '''All 3x3 Hermite normal forms of determinant n (lower triangular).'''
    for a in range(1, n + 1):
        if n % a:
            continue
        for c in range(1, n // a + 1):
            if (n // a) % c:
                continue
            f = n // (a * c)
            # row-style HNF: entries below the diagonal bounded by the diagonal of their column
            for b in range(a):
                for d in range(a):
                    for e in range(c):
                        yield np.array([[a, 0, 0], [b, c, 0], [d, e, f]])


def shortest_vector(L):
    return np.linalg.norm(_reduced_basis(np.asarray(L, float)), axis=1).min()


def best_supercells(system, nmax=8):
    '''For each size n <= nmax, the supercell matrix M (Hermite normal form)
    with the longest shortest lattice vector, i.e. the largest distance between
    a site and its own periodic images (the active dopant and its copies).
    Returns [(n, M, image distance in A)].'''
    A = np.asarray(system[0], float)
    out = []
    for n in range(1, nmax + 1):
        best = max(_hnf(n), key=lambda M: shortest_vector(M @ A))
        out.append((n, best, shortest_vector(best @ A)))
    return out


def site_distances(system, element='Eu'):
    '''Shortest distances (A) from each element site to any other atom of that
    element and to its own images: (min over other sites, own-image distance).'''
    A, atoms = system
    A = np.asarray(A, float)
    X = np.array([x for e, x in atoms if e == element])
    T = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]) @ A
    d = np.linalg.norm(X[:, None, None, :] + T[None, None] - X[None, :, None, :], axis=-1)
    d[d < 1e-6] = np.inf
    own = min(d[i, i].min() for i in range(len(X)))
    other = min((d[i, j].min() for i in range(len(X)) for j in range(len(X)) if i != j), default=np.inf)
    return other, own


SYSTEMS = {'quartz': QUARTZ, 'y2o3A': Y2O3_A, 'yso': YSO,
           'eu_y2o3A': substitute(Y2O3_A), 'eu_yso': substitute(YSO),
           # Supercells of the primitive cell with the largest Eu-Eu image
           # distance for their size (all Hermite normal forms enumerated; the
           # primitive cell already contains a 6.82 A vector, -a1 + 2 a3 = c):
           #   x2, 64 atoms: 8.05 A (1 Eu / 16 Y);  x3, 96 atoms: 10.52 A;
           #   (x4, 128 atoms: 12.54 A)
           'eu_yso_x2': substitute(supercell(YSO, (2, 1, 1))),
           'eu_yso_x3': substitute(supercell(YSO, (1, 1, 3)))}


def trim_basis(basname, elements, emin=0.1):
    '''Drop primitives with exponent < emin (and shells/contractions left empty).
    pyscf's cell.exp_to_discard crashes when a whole shell is removed.'''
    out = {}
    for el in set(elements):
        new = []
        for sh in mgto.load(basname, el):
            rows = [r for r in sh[1:] if r[0] >= emin]
            if not rows:
                continue
            arr = np.array(rows)
            keep = [j for j in range(1, arr.shape[1]) if abs(arr[:, j]).max() > 1e-8]
            if keep:
                new.append([sh[0]] + [[r[0]] + [r[j] for j in keep] for r in rows])
        out[el] = new
    return out


def make_cell(name, precision=1e-8, basis='x2c-svpall', emin=0.1, **kwargs):
    A, atoms = SYSTEMS[name] if isinstance(name, str) else name
    c = pgto.Cell()
    c.a = A
    c.unit = 'A'
    c.atom = [(e, tuple(x)) for e, x in atoms]
    if isinstance(basis, dict):     # explicit {symbol: pyscf basis}, used as given
        c.basis = {e: basis[e] for e in set(e for e, _ in atoms)}
    else:
        c.basis = trim_basis(basis, [e for e, _ in atoms], emin)
    c.precision = precision
    c.verbose = 0
    for k, v in kwargs.items():
        setattr(c, k, v)
    c.build()
    return c


def nearest_neighbours(name):
    '''Print nearest-neighbour distances (sanity check of the structures).'''
    A, atoms = SYSTEMS[name] if isinstance(name, str) else name
    T = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ A
    X = np.array([x for _, x in atoms])
    E = [e for e, _ in atoms]
    print(name, len(atoms), 'atoms', {e: E.count(e) for e in sorted(set(E))})
    for i, e in enumerate(E):
        d = np.linalg.norm(X[:, None, :] + T[None] - X[i], axis=-1)
        d[d < 1e-6] = 1e9
        j = np.unravel_index(d.argmin(), d.shape)[0]
        print('   %s%d nearest %s %.3f A' % (e, i, E[j], d.min()))


if __name__ == '__main__':
    for n in SYSTEMS:
        nearest_neighbours(n)
