#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Abelian double groups for spinor (2c) and Dirac (4c) orbitals.

The functions mirror :mod:`pyscf.symm` (``irrep_id2name``, ``irrep_name2id``,
``direct_prod``) and take the double-group name as an argument; nothing but
the name is stored on ``mol`` (``mol.doublegroupname``).

Supported groups and their generators (symmetry frame, main axis z):

    C1             -
    Ci             i
    C2             C2z
    Cs             sigma_h (mirror perpendicular to z)
    C2h            C2z, i
    Coov           rotations C(phi) about z
    Dooh           rotations C(phi) about z, i

D2h is handled through C2h and C2v/D2 through C2 (see :func:`double_group_of`).

Irrep ids
---------
Every irrep is labelled by two numbers:

* ``n``: the character of the cyclic generator is exp(+i pi n / 2) for
  C2z (C2, C2h) or sigma_h (Cs), and exp(+i phi n / 2) for C(phi) (linear
  groups, where n = 2 m_j or 2 m_l).  This is the passive convention,
  R(phi) = exp(+i phi J_z), of the Altmann-Herzig tables and of DIRAC, so that
  m_j = +1/2 belongs to 1E.  n is taken modulo N, with N = 2 for C1/Ci,
  N = 4 for C2/Cs/C2h, and unbounded for the linear groups.  Odd n are fermion
  irreps, even n are boson irreps.
* ``p``: parity under inversion (0 gerade, 1 ungerade), 0 for groups without i.

and stored as the integer ``ir = 2*n + p`` (negative for linear groups with
n < 0).  Direct product: n adds (mod N), p is XOR-ed.  Complex conjugation
(the irrep of the Kramers partner) is n -> -n, p unchanged.

Names
-----
Fermion irreps of C2, Cs, C2h use the Altmann-Herzig short names that DIRAC
prints: ``1E`` has character +i under C2z (Cs: sigma_h), ``2E`` has -i.  1E
contains m_j = 1/2, -3/2, 5/2, ..., which are DIRAC's unbarred spinors (DIRAC
LINSYM prints MJ = 1/2, -3/2, 5/2 for them, and moltra/traone.F puts unbarred
spinors into 1E).  C1 and
Ci fermion irreps are ``A1/2``, ``A1/2g``, ``A1/2u`` (DIRAC's ``A``, ``AG``,
``AU`` would clash with PySCF's boson names).  Boson irreps use PySCF's
single-group names.  Linear groups use m_j (fermion) and m_l (boson) values:
``1/2``, ``-3/2``, ``0``, ``2``, with a ``g``/``u`` suffix for Dooh.
'''

import numpy

DOUBLE_GROUPS = ('C1', 'Ci', 'C2', 'Cs', 'C2h', 'Coov', 'Dooh')

# modulus of n; None for the linear groups
_MODULUS = {'C1': 2, 'Ci': 2, 'C2': 4, 'Cs': 4, 'C2h': 4, 'Coov': None, 'Dooh': None}
_HAS_PARITY = {'C1': False, 'Ci': True, 'C2': False, 'Cs': False, 'C2h': True,
               'Coov': False, 'Dooh': True}

# single group (pyscf mol.groupname) -> double group used for spinors
_FROM_SINGLE_GROUP = {'D2h': 'C2h', 'C2v': 'C2', 'D2': 'C2', 'SO3': 'Dooh'}

# names of the finite groups, keyed by (n, p)
_NAMES = {
    'C1' : {(0, 0): 'A', (1, 0): 'A1/2'},
    'Ci' : {(0, 0): 'Ag', (0, 1): 'Au', (1, 0): 'A1/2g', (1, 1): 'A1/2u'},
    'C2' : {(0, 0): 'A', (2, 0): 'B', (1, 0): '1E', (3, 0): '2E'},
    'Cs' : {(0, 0): "A'", (2, 0): 'A"', (1, 0): '1E', (3, 0): '2E'},
    'C2h': {(0, 0): 'Ag', (2, 0): 'Bg', (0, 1): 'Au', (2, 1): 'Bu',
            (1, 0): '1Eg', (3, 0): '2Eg', (1, 1): '1Eu', (3, 1): '2Eu'},
}
_IDS = {g: {name: 2*n+p for (n, p), name in tab.items()} for g, tab in _NAMES.items()}

# group operations whose characters are defined for the finite groups;
# a leading 'b' denotes the barred element (times the 2pi rotation E-bar)
OPERATOR_TABLE = {
    'C1' : ('E',),
    'Ci' : ('E', 'i'),
    'C2' : ('E', 'C2'),
    'Cs' : ('E', 'sh'),
    'C2h': ('E', 'C2', 'i', 'sh'),
}


def _check_group(groupname):
    if groupname not in _MODULUS:
        raise KeyError('Unknown double group %s. Supported: %s' % (groupname, DOUBLE_GROUPS))


def double_group_of(groupname):
    '''Double group used for spinors, given PySCF's single-group name
    (``mol.groupname``).'''
    groupname = _FROM_SINGLE_GROUP.get(groupname, groupname)
    _check_group(groupname)
    return groupname


def _split(ir):
    ir = numpy.asarray(ir)
    return ir // 2, ir % 2


def _join(n, p, groupname):
    modulus = _MODULUS[groupname]
    if modulus is not None:
        n = n % modulus
    if not _HAS_PARITY[groupname]:
        p = 0 * p
    return 2 * n + p


def _as_result(x, *inputs):
    if all(numpy.ndim(a) == 0 for a in inputs):
        return int(x)
    return x


def is_fermion(ir):
    '''True for fermion (odd n) irreps.'''
    fermion = _split(ir)[0] % 2 == 1
    return fermion if numpy.ndim(ir) else bool(fermion)


def conj(ir, groupname):
    '''Complex-conjugate irrep, i.e. the irrep of the Kramers partner.'''
    _check_group(groupname)
    n, p = _split(ir)
    return _as_result(_join(-n, p, groupname), ir)


def direct_prod(orbsym1, orbsym2, groupname):
    '''Direct product of irreps.  Like :func:`pyscf.symm.direct_prod`, two
    1D arrays give the 2D table ``prod[i,j] = orbsym1[i] x orbsym2[j]``;
    scalars give an int.

    For a matrix element <p|O|q> use ``direct_prod(conj(p), q)``.'''
    _check_group(groupname)
    a = numpy.asarray(orbsym1)
    b = numpy.asarray(orbsym2)
    if a.ndim and b.ndim:
        a = a.reshape(a.shape + (1,) * b.ndim)
    na, pa = _split(a)
    nb, pb = _split(b)
    return _as_result(_join(na + nb, pa ^ pb, groupname), orbsym1, orbsym2)


def irrep_ids(groupname, kind='all', mj_max=None):
    '''Irrep ids of a finite double group, ordered fermions first (as DIRAC
    does).  For the linear groups ``mj_max`` (the largest |m_j|, or |m_l| for
    bosons) bounds the list.  ``kind`` is 'all', 'fermion' or 'boson'.'''
    _check_group(groupname)
    modulus = _MODULUS[groupname]
    if modulus is None:
        if mj_max is None:
            raise ValueError('mj_max is required for linear group %s' % groupname)
        nmax = int(round(2 * mj_max))
        ns = list(range(-nmax, nmax + 1))
    else:
        ns = list(range(modulus))
    ps = (0, 1) if _HAS_PARITY[groupname] else (0,)
    fermions = [2*n+p for n in ns if n % 2 == 1 for p in ps]
    bosons = [2*n+p for n in ns if n % 2 == 0 for p in ps]
    if modulus is not None:
        # DIRAC order: 1Eg 2Eg 1Eu 2Eu, then Ag Bg Au Bu
        fermions = sorted(fermions, key=lambda ir: (ir % 2, ir // 2))
        bosons = sorted(bosons, key=lambda ir: (ir % 2, ir // 2))
    if kind == 'fermion':
        return fermions
    elif kind == 'boson':
        return bosons
    return fermions + bosons


def _check_id(groupname, ir):
    ir = int(ir)
    if ((_MODULUS[groupname] is not None and ir not in _IDS[groupname].values())
            or (not _HAS_PARITY[groupname] and ir % 2)):
        raise KeyError('Irrep id %d not in double group %s' % (ir, groupname))
    return ir


def irrep_id2name(groupname, ir):
    _check_group(groupname)
    ir = _check_id(groupname, ir)
    n, p = ir // 2, ir % 2
    if _MODULUS[groupname] is not None:
        return _NAMES[groupname][(n, p)]
    if n % 2:
        name = '%d/2' % n
    else:
        name = '%d' % (n // 2)
    if _HAS_PARITY[groupname]:
        name += 'gu'[p]
    return name


def irrep_name2id(groupname, name):
    _check_group(groupname)
    if _MODULUS[groupname] is not None:
        try:
            return _IDS[groupname][name]
        except KeyError:
            raise KeyError('Irrep %s not in double group %s. Irreps: %s' %
                           (name, groupname, list(_IDS[groupname])))
    s = name.strip()
    p = 0
    if _HAS_PARITY[groupname]:
        if s[-1:] not in ('g', 'u'):
            raise KeyError('Irrep %s of %s needs a g/u suffix' % (name, groupname))
        p = 'gu'.index(s[-1])
        s = s[:-1]
    try:
        if s.endswith('/2'):
            n = int(s[:-2])
            if n % 2 == 0:
                raise ValueError
        else:
            n = 2 * int(s)
    except ValueError:
        raise KeyError('Irrep %s not in double group %s' % (name, groupname))
    return 2 * n + p


def character(groupname, ir, op):
    '''Character of a finite-group irrep for operation ``op`` (one of
    OPERATOR_TABLE[groupname], or the same with a leading 'b' for the barred
    element).  For the linear groups ``op`` may also be ('C', phi).'''
    _check_group(groupname)
    n, p = int(ir) // 2, int(ir) % 2
    if isinstance(op, tuple):
        assert op[0] == 'C'
        return numpy.exp(.5j * op[1] * n)
    sign = 1
    if op.startswith('b'):
        op = op[1:]
        sign = -1 if n % 2 else 1
    if op == 'E':
        chi = 1
    elif op == 'i':
        chi = (-1) ** p
    elif op == 'C2':
        chi = numpy.exp(.5j * numpy.pi * n)
    elif op == 'sh':
        if groupname == 'Cs':
            chi = numpy.exp(.5j * numpy.pi * n)
        else:  # sigma_h = i * C2
            chi = (-1) ** p * numpy.exp(.5j * numpy.pi * n)
    else:
        raise KeyError('Operation %s' % op)
    return sign * complex(numpy.round(chi, 14))


def descend(ir, groupname, subgroup):
    '''Map irrep ids of ``groupname`` onto its subgroup ``subgroup``.

    Supported: Dooh -> Coov, C2h, Ci, C2, Cs, C1;  Coov -> C2, C1;
    C2h -> Ci, C2, Cs, C1;  C2, Cs, Ci -> C1.'''
    _check_group(groupname)
    _check_group(subgroup)
    n, p = _split(ir)
    linear = groupname in ('Coov', 'Dooh')
    if subgroup == groupname:
        return ir
    if groupname == 'Dooh' or groupname == 'C2h' or (linear and subgroup in ('C2', 'C1')) \
       or (groupname in ('C2', 'Cs', 'Ci') and subgroup == 'C1'):
        pass
    else:
        raise KeyError('%s is not a supported subgroup of %s' % (subgroup, groupname))
    if subgroup == 'Coov':
        out = _join(n, p, 'Coov')
    elif subgroup == 'C2h':
        out = _join(n, p, 'C2h')
    elif subgroup == 'C2':
        out = _join(n, p, 'C2')
    elif subgroup == 'Ci':
        out = _join(n, p, 'Ci')
    elif subgroup == 'Cs':
        # sigma_h = i * C2: exp(i pi n_s/2) = (-1)^p exp(i pi n/2)
        out = _join(n + 2 * p, 0, 'Cs')
    elif subgroup == 'C1':
        out = _join(n, 0, 'C1')
    else:
        raise KeyError(subgroup)
    return _as_result(out, ir)
