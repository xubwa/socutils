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
SYSTEMS = {'quartz': QUARTZ, 'y2o3A': Y2O3_A, 'yso': YSO}


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
    A, atoms = SYSTEMS[name]
    c = pgto.Cell()
    c.a = A
    c.unit = 'A'
    c.atom = [(e, tuple(x)) for e, x in atoms]
    c.basis = trim_basis(basis, [e for e, _ in atoms], emin)
    c.precision = precision
    c.verbose = 0
    for k, v in kwargs.items():
        setattr(c, k, v)
    c.build()
    return c


def nearest_neighbours(name):
    '''Print nearest-neighbour distances (sanity check of the structures).'''
    A, atoms = SYSTEMS[name]
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
