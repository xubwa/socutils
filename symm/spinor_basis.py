#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Double-group symmetry-adapted spinor basis (the AO layer).

The SALCs are built for PySCF's j-adapted spinor GTOs.  All operations used
here (C2z, i, sigma_h and the rotations C(phi) about z) act on a spinor GTO
|atom A, kappa, m_j> as an atom permutation times a phase:

    C2z      exp(+i pi m_j)
    i        (-1)^l
    sigma_h  (-1)^l exp(+i pi m_j)          (sigma_h = i C2z)
    C(phi)   exp(+i phi m_j)

(passive convention R(phi) = exp(+i phi J_z), as in the Altmann-Herzig tables
and DIRAC; see socutils.symm.double_group).

provided the rotation axis is the lab z axis.  The molecule therefore has to be
in its symmetry frame (z axis along the main axis); :func:`to_symm_frame`
puts it there.  Groups without an axis (C1, Ci) work in any orientation.

Kramers partners are generated with the time-reversal operator instead of
being projected independently: for a pair of complex-conjugate irreps
(Gamma, conj Gamma) the columns of the partner block are T applied to the
columns of the first block, one by one.  For a self-conjugate irrep (C1, Ci)
the block is laid out as [X | T X], the order expected by the quaternion
eigensolver (zquatev).

Realification.  If the molecule has, besides the double group, an operation
G with G C2z G^-1 = C2z^-1 (C2y for D2h and D2, sigma_xz for C2v and the
linear groups), the antiunitary K = G T maps every fermion irrep onto itself
and K^2 = +1.  Kramers-restricted orbitals can then be chosen K invariant
(:func:`realify_mo`), which makes all their one- and two-electron integrals
real.  On the spinor GTOs

    C2y       (-1)^(j - m_j)          |j, -m_j>
    sigma_xz  (-1)^l (-1)^(j - m_j)   |j, -m_j>        (sigma_xz = i C2y)

For 4-component calculations :func:`dirac_symm_orb` adds the small component.
The small-component basis is sigma.p acting on the same spinor GTOs.  sigma.p
is odd under inversion, but so is the intrinsic parity of the small component
(inversion acts on 4-spinors as beta P), so the small-component coefficients
of an irrep Gamma are the SALCs of the same Gamma.
'''

import weakref
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from socutils.symm import double_group as dg

# 3x3 matrices of the operations in the symmetry frame
_OP_MATRIX = {
    'E' : numpy.eye(3),
    'C2': numpy.diag([-1., -1., 1.]),
    'i' : -numpy.eye(3),
    'sh': numpy.diag([1., 1., -1.]),
    'C2y': numpy.diag([-1., 1., -1.]),
    'sxz': numpy.diag([1., -1., 1.]),
}

# group elements (barred elements omitted: for fermion irreps they give the
# same contribution to the projector) used for the projection
_PROJECTION_OPS = {
    'C1': ('E',), 'Ci': ('E', 'i'), 'C2': ('E', 'C2'), 'Cs': ('E', 'sh'),
    'C2h': ('E', 'C2', 'i', 'sh'), 'Coov': ('E',), 'Dooh': ('E', 'i'),
}

_LINEAR = ('Coov', 'Dooh')

# groups whose operations involve the z axis
_NEEDS_AXIS = ('C2', 'Cs', 'C2h', 'Coov', 'Dooh')

FRAME_TOL = 1e-6

# realification: the operation G (by point group) for K = G T, and the double
# groups whose conjugate pairs (1E, 2E or m_j, -m_j) G interchanges
_REALIFY_OP = {'D2h': 'C2y', 'D2': 'C2y', 'C2v': 'sxz', 'Coov': 'sxz', 'Dooh': 'sxz'}
_REALIFY_GROUPS = ('C2', 'C2h', 'Coov', 'Dooh')


def _spinor_ao_info(mol):
    '''(atom id, l, 2*m_j) of each spinor AO.'''
    labels = mol.spinor_labels(fmt=False)
    atom = numpy.array([x[0] for x in labels])
    l = numpy.array(['spdfghiklmnortu'.index(x[2].lstrip('0123456789')[0]) for x in labels])
    two_mj = numpy.array([int(x[3].split('/')[0]) for x in labels])
    return atom, l, two_mj


def _op_phase(op, l, two_mj):
    if op == 'E':
        return numpy.ones(l.size, dtype=complex)
    if op == 'C2':
        return numpy.exp(.5j * numpy.pi * two_mj)
    if op == 'i':
        return (-1.)**l + 0j
    if op == 'sh':
        return (-1.)**l * numpy.exp(.5j * numpy.pi * two_mj)
    raise KeyError(op)


def realification_op(mol, doublegroupname):
    '''The operation G of K = G T for realifying the orbitals of
    ``doublegroupname``, or None.'''
    if not mol.symmetry or doublegroupname not in _REALIFY_GROUPS:
        return None
    op = _REALIFY_OP.get(mol.groupname)
    if op is None:
        return None
    orig = numpy.asarray(mol._symm_orig)
    try:
        _atom_permutation(mol, op, orig, max(symm.geom.TOLERANCE * 10, 1e-4))
    except symm.PointGroupSymmetryError:
        return None
    return op


def apply_k(mol, op, c):
    '''Coefficients of K f = G T f (columns of ``c``).'''
    labels = mol.spinor_labels(fmt=False)
    l = numpy.array(['spdfghiklmnortu'.index(x[2].lstrip('0123456789')[0]) for x in labels])
    two_j = numpy.array([int(x[2].lstrip('0123456789')[1:].split('/')[0]) for x in labels])
    two_mj = numpy.array([int(x[3].split('/')[0]) for x in labels])
    phase = (-1.)**((two_j - two_mj) // 2)
    if op == 'sxz':
        phase *= (-1.)**l
    # G maps the AO (A, l, j, m_j) to (G A, l, j, -m_j); the -m_j AO is the
    # time-reversal partner
    flip = abs(numpy.asarray(mol.time_reversal_map())) - 1
    dest = _ao_permutation(mol, _atom_permutation(
        mol, op, numpy.asarray(mol._symm_orig), max(symm.geom.TOLERANCE * 10, 1e-4)))[flip]
    tc = time_reversal_coeff(mol, c)
    kc = numpy.empty_like(tc)
    kc[dest] = phase.reshape((-1,) + (1,) * (tc.ndim - 1)) * tc
    return kc


def realify_mo(mol, mo_coeff, s, op, mo_energy=None, degen_tol=1e-8):
    """Phases of Kramers-paired orbitals such that the first orbital u of each
    pair is K invariant (K u = u, K = G T); the second is replaced by T u.

    mo_coeff : (n, nmo) in the interleaved (u, T u) layout, 2c (n = nao_2c)
        or 4c (n = 2 nao_2c, [L; S]).  s is the metric of the same dimension.
    With an orbital energy degenerate within the same K-invariant block
    (accidental), the degenerate u are rotated together, U -> U sqrt(U^+ S K U).
    Returns the new coefficients (same layout).
    """
    n2c = mol.nao_2c()
    c = numpy.array(mo_coeff, dtype=complex)
    four_c = c.shape[0] == 2 * n2c

    def tr(x):
        if four_c:
            return numpy.vstack((time_reversal_coeff(mol, x[:n2c]), time_reversal_coeff(mol, x[n2c:])))
        return time_reversal_coeff(mol, x)

    def k(x):
        if four_c:
            return numpy.vstack((apply_k(mol, op, x[:n2c]), apply_k(mol, op, x[n2c:])))
        return apply_k(mol, op, x)

    u = c[:, 0::2]
    if abs(tr(u) - c[:, 1::2]).max() > 1e-6:
        raise ValueError('orbitals are not in the interleaved Kramers-pair layout (u, T u)')
    lam = numpy.einsum('pi,pq,qi->i', u.conj(), s, k(u))
    done = numpy.zeros(u.shape[1], dtype=bool)
    for i in range(u.shape[1]):
        if done[i]:
            continue
        if abs(abs(lam[i]) - 1) < 1e-6:
            u[:, i] *= numpy.sqrt(lam[i])
            done[i] = True
            continue
        if mo_energy is None:
            raise RuntimeError('orbital %d mixes under K; pass mo_energy to group degenerate orbitals' % (2 * i))
        e = numpy.asarray(mo_energy)[0::2]
        grp = numpy.where(~done & (abs(e - e[i]) < degen_tol))[0]
        ug = u[:, grp]
        m = ug.conj().T.dot(s).dot(k(ug))
        # M is symmetric unitary (K antiunitary, K^2 = 1); its principal square
        # root W is symmetric unitary and K (U W) = U M W* = U W
        if abs(m.dot(m.conj()) - numpy.eye(grp.size)).max() > 1e-6:
            raise RuntimeError('degenerate orbitals %s do not span a K-invariant space' % (2 * grp))
        u[:, grp] = ug.dot(scipy.linalg.sqrtm(m))
        done[grp] = True
    c[:, 0::2] = u
    c[:, 1::2] = tr(u)
    return c


def _atom_permutation(mol, op, orig, tol):
    '''perm[a] = atom b at the image of atom a under ``op`` (about ``orig``).'''
    coords = mol.atom_coords() - orig
    images = coords.dot(_OP_MATRIX[op].T)
    charges = mol.atom_charges()
    perm = numpy.empty(mol.natm, dtype=int)
    for a in range(mol.natm):
        dist = numpy.linalg.norm(coords - images[a], axis=1)
        b = int(numpy.argmin(dist))
        if dist[b] > tol or charges[b] != charges[a] or \
           mol.atom_symbol(b) != mol.atom_symbol(a):
            raise symm.PointGroupSymmetryError(
                'Atom %d has no symmetry image under %s' % (a, op))
        perm[a] = b
    return perm


def _ao_permutation(mol, atom_perm):
    '''AO index map for an atom permutation (equivalent atoms carry the same
    basis, so the local AO order is the same).'''
    aoslice = mol.aoslice_2c_by_atom()
    dest = numpy.empty(mol.nao_2c(), dtype=int)
    for a, b in enumerate(atom_perm):
        p0, p1 = aoslice[a, 2:]
        q0, q1 = aoslice[b, 2:]
        assert p1 - p0 == q1 - q0
        dest[p0:p1] = numpy.arange(q0, q1)
    return dest


def time_reversal_coeff(mol, c):
    '''Coefficients of T f for the functions f with coefficients ``c``
    (columns), T being the time-reversal operator on the spinor AOs.'''
    tao = numpy.asarray(mol.time_reversal_map())
    idx = abs(tao) - 1
    sign = numpy.where(tao < 0, -1., 1.)
    c = numpy.asarray(c)
    tc = numpy.empty_like(c, dtype=complex)
    tc[idx] = sign.reshape((-1,) + (1,) * (c.ndim - 1)) * c.conj()
    return tc


def symm_frame_ok(mol, tol=FRAME_TOL):
    '''Whether the symmetry operations of the double group are defined with
    respect to the lab z axis.'''
    if not mol.symmetry:
        return True
    gname = dg.double_group_of(mol.groupname)
    if gname not in _NEEDS_AXIS:
        return True
    return abs(abs(mol._symm_axes[2, 2]) - 1) < tol


def to_symm_frame(mol, symmetrize=True, verbose=None):
    '''A copy of ``mol`` with the coordinates in its symmetry frame,
    r_s = R (r - orig), R a proper rotation.  With ``symmetrize`` every atom is
    averaged over its images under the point group, removing the deviations
    from exact symmetry allowed by ``symm.geom.TOLERANCE``.  The frame is kept
    in ``mol.symm_frame = (orig, R)``.'''
    coords, orig, axes, disp = _symm_frame_coords(mol, symmetrize)
    log = logger.new_logger(mol, verbose)
    log.info('Molecule moved to its symmetry frame (origin %s, axes %s); '
             'max displacement from symmetrization %.2e Bohr', orig, axes.tolist(), disp)
    mol_s = mol.set_geom_(coords, unit='Bohr', inplace=False)
    # plain lists keep mol.dumps() (chkfile) working
    mol_s.symm_frame = (orig.tolist(), axes.tolist())
    return mol_s


def _symm_frame_coords(mol, symmetrize=True):
    if not mol.symmetry:
        raise RuntimeError('mol.symmetry is not enabled')
    orig = numpy.asarray(mol._symm_orig, dtype=float)
    axes = numpy.array(mol._symm_axes, dtype=float)
    if numpy.linalg.det(axes) < 0:
        axes[0] *= -1
    coords = (mol.atom_coords() - orig).dot(axes.T)
    sym = coords.copy()
    if symmetrize:
        gname = mol.groupname
        if gname in ('Coov', 'Dooh', 'SO3'):
            sym[:, :2] = 0
            if gname == 'SO3':
                sym[:] = 0
            ops = ['E', 'i'] if gname in ('Dooh', 'SO3') else ['E']
        else:
            ops = symm.param.OPERATOR_TABLE[gname]
        opdic = symm.geom.symm_ops(gname if gname in symm.param.POINTGROUP else 'D2h')
        tol = max(symm.geom.TOLERANCE * 10, 1e-4)
        acc = numpy.zeros_like(sym)
        for op in ops:
            mat = opdic[op] * numpy.eye(3)
            images = sym.dot(mat.T)
            for a in range(mol.natm):
                dist = numpy.linalg.norm(sym - images[a], axis=1)
                b = int(numpy.argmin(dist))
                if dist[b] > tol or mol.atom_charge(b) != mol.atom_charge(a):
                    raise symm.PointGroupSymmetryError(
                        'Atom %d has no image under %s (distance %.2e)' % (a, op, dist[b]))
                acc[a] += mat.T.dot(sym[b])
        sym = acc / len(ops)
    disp = abs(sym - coords).max() if mol.natm else 0.
    return sym, orig, axes, disp


def frame_error_message(mol):
    '''Explanation, fix and ready-to-use coordinates for a molecule that is not
    in its symmetry frame.'''
    coords, orig, axes, disp = _symm_frame_coords(mol)
    lines = [
        'Double-group symmetry %s needs the molecule in its symmetry frame: the'
        % dg.double_group_of(mol.groupname),
        'spinor m_j labels and the symmetry phases are defined along the lab z axis,',
        'and lab-frame symmetry-adapted spinors are not implemented yet.',
        'The symmetry z axis of this molecule is %s.' % numpy.round(mol._symm_axes[2], 8).tolist(),
        'Fix:  mol = socutils.symm.to_symm_frame(mol)',
        'or use these coordinates (Bohr; symmetrized, max change %.2e Bohr):' % disp,
    ]
    for a in range(mol.natm):
        lines.append('  %-4s %20.12f %20.12f %20.12f' % ((mol.atom_symbol(a),) + tuple(coords[a])))
    lines.append('Vectors, tensors and m_j labels then refer to this frame.')
    return '\n'.join(lines)


def check_symm_frame(mol):
    if not symm_frame_ok(mol):
        raise symm.PointGroupSymmetryError(frame_error_message(mol))


def _project(mol, gname, irrep, aos, ops, perms, phases):
    '''SALCs P_irrep |mu> for the AOs ``aos`` (one column per nonzero result).'''
    n2c = mol.nao_2c()
    chars = [numpy.conj(dg.character(gname, irrep, op)) for op in ops]
    cols = []
    for mu in aos:
        c = numpy.zeros(n2c, dtype=complex)
        for chi, dest, ph in zip(chars, perms, phases):
            c[dest[mu]] += chi * ph[mu]
        norm = numpy.linalg.norm(c)
        if norm > 1e-8:
            cols.append(c / norm)
    return cols


def symm_adapted_basis_spinor(mol, doublegroupname=None):
    '''Symmetry-adapted spinor basis.

    Returns:
        so : list of (n2c, n_ir) complex arrays, one per fermion irrep; the
            columns of all blocks together form a unitary matrix.
        irrep_id : list of int, double-group irrep ids of the blocks.
    '''
    if doublegroupname is None:
        doublegroupname = dg.double_group_of(mol.groupname if mol.symmetry else 'C1')
    gname = doublegroupname
    if gname in _NEEDS_AXIS:
        check_symm_frame(mol)
    orig = numpy.zeros(3) if not mol.symmetry else numpy.asarray(mol._symm_orig)
    tol = max(symm.geom.TOLERANCE * 10, 1e-4)

    atom, l, two_mj = _spinor_ao_info(mol)
    ops = _PROJECTION_OPS[gname]
    perms = [_ao_permutation(mol, _atom_permutation(mol, op, orig, tol)) for op in ops]
    phases = [_op_phase(op, l, two_mj) for op in ops]
    # one representative atom per orbit
    atom_orbit_rep = numpy.array([min(_atom_permutation(mol, op, orig, tol)[a] for op in ops)
                                  for a in range(mol.natm)])
    rep_ao = atom_orbit_rep[atom] == atom
    tao = numpy.asarray(mol.time_reversal_map())

    if gname in _LINEAR:
        # C(phi) is diagonal on the AOs: n = 2 m_j is an AO label
        n_ao = two_mj
        first = sorted({n for n in n_ao if n > 0})
        p_list = (0, 1) if gname == 'Dooh' else (0,)
        blocks = {}
        for n in first:
            for p in p_list:
                ir = 2 * n + p
                aos = numpy.where(rep_ao & (n_ao == n))[0]
                cols = _project(mol, gname, ir, aos, ops, perms, phases)
                if cols:
                    x = numpy.array(cols).T
                    blocks[ir] = x
                    blocks[dg.conj(ir, gname)] = time_reversal_coeff(mol, x)
        order = sorted(blocks, key=lambda ir: (ir % 2, abs(ir // 2), -(ir // 2)))
    else:
        blocks = {}
        done = set()
        for ir in dg.irrep_ids(gname, 'fermion'):
            if ir in done:
                continue
            partner = dg.conj(ir, gname)
            if partner == ir:
                # self-conjugate: [X | T X] from the AOs with T f_i = +f_j
                aos = numpy.where(rep_ao & (tao > 0))[0]
                cols = _project(mol, gname, ir, aos, ops, perms, phases)
                if cols:
                    x = numpy.array(cols).T
                    blocks[ir] = numpy.hstack((x, time_reversal_coeff(mol, x)))
            else:
                aos = numpy.where(rep_ao)[0]
                cols = _project(mol, gname, ir, aos, ops, perms, phases)
                if cols:
                    x = numpy.array(cols).T
                    blocks[ir] = x
                    blocks[partner] = time_reversal_coeff(mol, x)
            done.update((ir, partner))
        order = [ir for ir in dg.irrep_ids(gname, 'fermion') if ir in blocks]

    so = [blocks[ir] for ir in order]
    nso = sum(x.shape[1] for x in so)
    if nso != mol.nao_2c():
        raise RuntimeError('Symmetry-adapted spinor basis has %d functions, expected %d'
                           % (nso, mol.nao_2c()))
    return so, order


def dirac_symm_orb(mol, so=None, irrep_id=None, doublegroupname=None):
    '''Symmetry-adapted 4-component basis (large component first, as in
    pyscf.scf.dhf).  The small-component block of irrep Gamma uses the 2c
    SALCs of the same Gamma.  For self-conjugate irreps the [X | T X] layout
    is kept: columns [X_L, X_S, T X_L, T X_S].'''
    if doublegroupname is None:
        doublegroupname = dg.double_group_of(mol.groupname if mol.symmetry else 'C1')
    gname = doublegroupname
    if so is None:
        so, irrep_id = symm_adapted_basis_spinor(mol, gname)
    lookup = dict(zip(irrep_id, so))
    n2c = mol.nao_2c()

    def lift(xl, xs):
        zl = numpy.zeros((n2c, xs.shape[1]), dtype=complex)
        zs = numpy.zeros((n2c, xl.shape[1]), dtype=complex)
        return numpy.hstack((numpy.vstack((xl, zs)), numpy.vstack((zl, xs))))

    def time_reversal_4c(x):
        return numpy.vstack((time_reversal_coeff(mol, x[:n2c]), time_reversal_coeff(mol, x[n2c:])))

    blocks = {}
    for ir in irrep_id:
        if ir in blocks:
            continue
        xl = lookup[ir]
        xs = xl
        partner = dg.conj(ir, gname)
        if partner == ir:
            # keep the [X | T X] layout: [X_L, X_S, T X_L, T X_S]
            x = lift(xl[:, :xl.shape[1] // 2], xs[:, :xs.shape[1] // 2])
            blocks[ir] = numpy.hstack((x, time_reversal_4c(x)))
        else:
            x = lift(xl, xs)
            blocks[ir] = x
            blocks[partner] = time_reversal_4c(x)
    return [blocks[ir] for ir in irrep_id], list(irrep_id)


# SALCs are kept outside the Mole object: mol.dumps() (used for chkfiles) can
# serialize only JSON-like attributes.  In PySCF mainline they would become a
# Mole attribute excluded in dumps(), like mol.symm_orb.
_SALC_CACHE = weakref.WeakKeyDictionary()


def _geometry_key(mol):
    return (mol.atom_coords().round(10).tobytes(), mol.nao_2c())


def _default_group(mol):
    return dg.double_group_of(mol.groupname if mol.symmetry else 'C1')


def build(mol, doublegroupname=None):
    '''Attach the double-group attributes to ``mol``: doublegroupname,
    irrep_id_spinor, irrep_name_spinor.  The SALCs are available from
    :func:`get_symm_orb` (cf. mol.symm_orb).'''
    if doublegroupname is None:
        doublegroupname = _default_group(mol)
    so, irrep_id = get_symm_orb(mol, '2c', doublegroupname)
    mol.doublegroupname = doublegroupname
    mol.irrep_id_spinor = list(irrep_id)
    mol.irrep_name_spinor = [dg.irrep_id2name(doublegroupname, ir) for ir in irrep_id]
    return mol


def get_symm_orb(mol, kind='2c', doublegroupname=None):
    '''SALCs of ``mol`` for ``doublegroupname`` (default: the double group of
    mol.groupname), built on first use.  kind '2c' gives the spinor basis of
    :func:`symm_adapted_basis_spinor`, '4c' that of :func:`dirac_symm_orb`.
    Returns (so, irrep_id).'''
    if doublegroupname is None:
        doublegroupname = getattr(mol, 'doublegroupname', None) or _default_group(mol)
    groups = _SALC_CACHE.setdefault(mol, {})
    entry = groups.get(doublegroupname)
    if entry is None or entry['key'] != _geometry_key(mol):
        so, irrep_id = symm_adapted_basis_spinor(mol, doublegroupname)
        entry = groups[doublegroupname] = {'key': _geometry_key(mol), 'so': so,
                                           'irrep_id': [int(i) for i in irrep_id]}
    if kind == '2c':
        return entry['so'], entry['irrep_id']
    if kind == '4c':
        if 'so4' not in entry:
            entry['so4'] = dirac_symm_orb(mol, entry['so'], entry['irrep_id'], doublegroupname)[0]
        return entry['so4'], entry['irrep_id']
    raise KeyError(kind)


def symmetrize_matrix(mat, so):
    '''Blocks so[i]^dagger mat so[i] (cf. pyscf.symm.symmetrize_matrix).'''
    return [lib.dot(c.conj().T, lib.dot(mat, c)) for c in so]
