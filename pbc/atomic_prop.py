'''One-centre property operators for a periodic (Gamma) spinor calculation.

An atomic operator (e.g. the CFOUR X2C two-component Schiff/NSM integrals,
file SOSFMO2C, computed for the isolated dopant atom in its own basis) is put
in the diagonal block of that atom in the cell's spinor AO basis; every other
block is zero.  This is the same one-centre approximation as the X2CAMF
Hamiltonian, and the same operator as the cluster calculations
(prop/props_bypass.py of the Eu:aimp work), so W_NSM is comparable like for like.

The dopant basis in the cell must be the same basis (same GENBAS entry) as the
CFOUR atomic run; ``atomic_block_operator`` checks the shells.
'''
import os
import numpy
from pyscf import gto
from socutils.tools import basis_parser, label


def load_cfour_atom(atomdir, symbol='Eu', charge=3, fname='SOSFMO2C', ncomp=3):
    '''(atom Mole, operator[ncomp, n2c, n2c] in the pyscf spinor order).

    atomdir holds config.py (BASIS_MAP, GENBAS) and the CFOUR file ``fname``.
    The transformation is that of props_bypass.py.'''
    cfg = {}
    exec(open(os.path.join(atomdir, 'config.py')).read(), cfg)
    bas = basis_parser.parse_genbas(cfg['BASIS_MAP'][symbol], cfg['GENBAS'])
    mol = gto.M(atom='%s 0 0 0' % symbol, basis={symbol: bas}, charge=charge, unit='B', nucmod='G', verbose=0)
    n = mol.nao_2c()
    c2p = label.Labels.spinor_cfour2pyscf(mol)
    op = label.carr_f2py(mol, filename=os.path.join(atomdir, fname)).reshape(ncomp, n, n).conj()
    op = numpy.array([c2p.conj().T @ o @ c2p for o in op])
    return mol, op


def _shells(mol, ish0, ish1):
    out = []
    for i in range(ish0, ish1):
        l = mol.bas_angular(i)
        e = mol.bas_exp(i)
        c = mol._libcint_ctr_coeff(i)
        out.append((l, tuple(numpy.round(e, 10)), numpy.round(c, 8).tobytes()))
    return out


def atomic_block_operator(cell, iatm, molatom, op):
    '''Embed op (from load_cfour_atom) in the (iatm, iatm) spinor block of cell.'''
    sh0, sh1, p0, p1 = cell.aoslice_2c_by_atom()[iatm]
    if p1 - p0 != molatom.nao_2c() or _shells(cell, sh0, sh1) != _shells(molatom, 0, molatom.nbas):
        raise ValueError('basis of atom %d in the cell differs from the atomic operator basis' % iatm)
    n2c = cell.nao_2c()
    out = numpy.zeros((op.shape[0], n2c, n2c), dtype=numpy.complex128)
    out[:, p0:p1, p0:p1] = op
    return out


def expectation(op, dm):
    '''<O>_x = Re tr(O_x D) for D in the pyscf AO convention (Tr(O D) = sum O_uv D_vu).'''
    return numpy.einsum('xuv,vu->x', op, dm).real


def image_cluster(cell, iatm, rmax):
    '''Molecule of all atom images within rmax (bohr) of atom iatm (iatm at the
    origin, same basis and nuclear model as the cell), the map of its spinor
    AOs onto the cell's, and the distance of each spinor AO's atom from iatm.'''
    a = numpy.asarray(cell.lattice_vectors())
    x = cell.atom_coords()
    n = int(numpy.ceil(rmax / min(numpy.linalg.norm(a, axis=1)))) + 1
    t = numpy.array([[i, j, k] for i in range(-n, n + 1) for j in range(-n, n + 1)
                     for k in range(-n, n + 1)]) @ a
    imgs = [(ia, x[ia] + tt - x[iatm]) for tt in t for ia in range(cell.natm)
            if numpy.linalg.norm(x[ia] + tt - x[iatm]) < rmax]
    imgs.sort(key=lambda s: numpy.linalg.norm(s[1]))
    mol = gto.M(atom=[(cell.atom_symbol(ia), tuple(v)) for ia, v in imgs], basis=cell._basis, unit='B',
                nucmod=cell.nucmod, spin=None, verbose=0)
    cs, ms = cell.aoslice_2c_by_atom(), mol.aoslice_2c_by_atom()
    cmap = numpy.concatenate([numpy.arange(cs[ia][2], cs[ia][3]) for ia, _ in imgs])
    dist = numpy.concatenate([numpy.full(ms[i][3] - ms[i][2], numpy.linalg.norm(v)) for i, (_, v) in enumerate(imgs)])
    return mol, cmap, dist


def local_dipole(cell, iatm, dm, rcs=(0., 2.6, 2.9, 3.5), rmax=None):
    '''Dipole of a density localised at atom iatm (e.g. a 4f-4f transition
    density) about the nucleus of iatm: sum over the image AOs whose atoms lie
    within rc of iatm of <a| r - R |b> dm[cell(a), cell(b)], for each rc (bohr;
    rc = 0 is the atom's own block).  Returns [(rc, d[3], q)], q = the charge of
    dm in the same region (zero for a complete transition density; an origin
    shift Delta changes d by -q Delta).'''
    if rmax is None:
        rmax = max(rcs) + 1e-6
    mol, cmap, dist = image_cluster(cell, iatm, rmax)
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r_spinor', comp=3)
    s = mol.intor('int1e_ovlp_spinor')
    out = []
    for rc in rcs:
        i = numpy.where(dist <= rc + 1e-8)[0]
        t = dm[numpy.ix_(cmap[i], cmap[i])]
        d = numpy.array([numpy.einsum('ij,ji->', r[x][numpy.ix_(i, i)], t) for x in range(3)])
        q = numpy.einsum('ij,ji->', s[numpy.ix_(i, i)], t)
        out.append((rc, d, q))
    return out


def real_axis(v):
    '''Remove the arbitrary phase of a transition moment (the phase of its
    largest component), as props_bypass.py.'''
    v = numpy.asarray(v)
    return (v * numpy.exp(-1j * numpy.angle(v[numpy.argmax(abs(v))]))).real
