import numpy
from pyscf import gto
from socutils.rf12 import ints4c
# four atoms, one shell each (s, p, d, p), all on different centres
def mk(shift=None):
    coords = numpy.array([[0, 0, 0], [0.9, 0.2, -0.3], [-0.4, 1.1, 0.5], [0.3, -0.6, 1.2]])
    if shift is not None:
        a, d, h = shift
        coords = coords.copy(); coords[a, d] += h
    atom = [['X%d' % (n + 1), c] for n, c in enumerate(coords)]
    basis = {'X1': [[0, [1.3, 1.0]]], 'X2': [[1, [0.9, 1.0]]], 'X3': [[2, [1.1, 1.0]]], 'X4': [[1, [0.7, 1.0]]]}
    m = gto.M(atom=atom, basis=basis, unit='bohr', spin=None, verbose=0)
    return m
mol = mk()
z = 1.3
for kernel in ('G', 'F', 'Y'):
    I0 = ints4c.full(mol, kernel, 'plain', z)[0, 0, 0, 0]
    # pyscf reference for the plain class
    if kernel == 'G':
        ref = mol.intor('int2e')
    else:
        mol.set_f12_zeta(z); ref = mol.intor('int2e_stg' if kernel == 'F' else 'int2e_yp')
    print(kernel, 'plain vs pyscf', abs(I0 - ref).max())
    # ipvip1: (d_a i d_b j | k l) = d/dR_i,a d/dR_j,b (ij|kl)   (i on atom 0, j on atom 1)
    I1 = ints4c.full(mol, kernel, 'ipvip1', z)
    h = 1e-4
    ao = mol.ao_loc_nr()
    err = 0
    for a in range(3):
        for b in range(3):
            fd = 0
            for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                m = mk((0, a, sa * h)); m = mk(None) if False else m
                coords = m.atom_coords(); coords[1, b] += sb * h
                m = m.set_geom_(coords, unit='bohr', inplace=False)
                fd += sa * sb * ints4c.full(m, kernel, 'plain', z)[0, 0, 0, 0]
            fd /= 4 * h * h
            blk = (slice(ao[0], ao[1]), slice(ao[1], ao[2]), slice(ao[2], ao[3]), slice(ao[3], ao[4]))
            err = max(err, abs(I1[a, b, 0, 0][blk] - fd[blk]).max())
    print(kernel, 'ipvip1 vs FD (i on atom0, j on atom1)', err)
    # ipvip1ipvip2 vs FD of ipvip1 over atoms 2 (k) and 3 (l)
    I2 = ints4c.full(mol, kernel, 'ipvip1ipvip2', z)
    err = 0
    for c in range(3):
        for d in range(3):
            fd = 0
            for sc, sd in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                coords = mol.atom_coords().copy(); coords[2, c] += sc * h; coords[3, d] += sd * h
                m = mol.set_geom_(coords, unit='bohr', inplace=False)
                fd += sc * sd * ints4c.full(m, kernel, 'ipvip1', z)
            fd /= 4 * h * h
            blk = (slice(ao[0], ao[1]), slice(ao[1], ao[2]), slice(ao[2], ao[3]), slice(ao[3], ao[4]))
            err = max(err, abs(I2[:, :, c, d][(slice(None), slice(None)) + blk] - fd[:, :, 0, 0][(slice(None), slice(None)) + blk]).max())
    print(kernel, 'ipvip1ipvip2 vs FD of ipvip1', err, ' max|I| =', abs(I2).max())
# Yukawa at small zeta -> Coulomb (same class)
for cls in ('ipvip1', 'ipvip1ipvip2', 'ipip1', 'ipip1ipip2'):
    z = 2e-3
    yp = 2 * ints4c.full(mol, 'Y', cls, z) - ints4c.full(mol, 'Y', cls, 2 * z)   # Richardson, O(z^2)
    g = ints4c.full(mol, 'G', cls)
    print(cls, 'yp(zeta->0, Richardson) vs Coulomb', abs(yp - g).max(), ' max|I| =', abs(g).max())
# ipip1ipip2 vs FD of ipip1 (electron-2 function k on atom 2, two derivatives on the same centre)
I = ints4c.full(mol, 'F', 'ipip1ipip2', 1.3)
I1 = ints4c.full(mol, 'F', 'ipip1', 1.3)
err = 0
for c in range(3):
    for d in range(3):
        fd = 0
        for sc, sd in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            coords = mol.atom_coords().copy(); coords[2, c] += sc * h; coords[2, d] += sd * h
            m = mol.set_geom_(coords, unit='bohr', inplace=False)
            fd += sc * sd * ints4c.full(m, 'F', 'ipip1', 1.3)
        fd /= 4 * h * h
        blk = (slice(None), 0, 0, 0, slice(ao[0], ao[1]), slice(ao[1], ao[2]), slice(ao[2], ao[3]), slice(ao[3], ao[4]))
        err = max(err, abs(I[:, 0, 3 * c + d, 0][(slice(None),) + blk[4:]] - fd[blk]).max())
print('stg ipip1ipip2 vs FD of ipip1', err, abs(I).max())
