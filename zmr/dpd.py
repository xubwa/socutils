'''
Direct product decomposition (DPD) of tensors over orbital spaces with double-group
symmetry, shared by the methods of zmr (Stanton, Gauss, Watts, Bartlett, J. Chem. Phys.
94, 4334 (1991); the buf4 of Psi4's libdpd).

Irreps are those of socutils.symm.double_group (ir = 2 n + p); every product, conjugate
and comparison is taken from there (Group), so the cyclic groups with their modulus, the
linear groups with unbounded n and the parity are all the same code.  The trivial group
(name None) has one irrep, 0: C1, or no symmetry.

Leg
    An index of a tensor: its orbitals in sectors of one irrep each, contiguous (the
    orbitals of a space sorted by sector).  A leg with one sector of irrep None carries
    no symmetry (e.g. an auxiliary index that is not symmetry adapted): a tensor with
    such a leg does not conserve symmetry and is stored dense in the sectors.

Tensor
    legs, the direction of every index (+1 created, -1 annihilated: an element is
    nonzero only if the product of the irreps of its orbitals, the annihilated ones
    conjugated, is the irrep `irrep` of the tensor, the totally symmetric one by
    default) and the split of the legs into row legs and column legs.  The data are,
    for every irrep h of the row multi-index, the matrix M[h] over the row multi-indices
    of composite irrep h and the column multi-indices of composite irrep conj(h) x irrep,
    all in one flat buffer.  Within M[h] the multi-indices are ordered by their sector
    tuple (C order over the legs), then by their elements (C order), so every block of
    sectors is a rectangle of M[h].

    contract(a, b): the column legs of a are the row legs of b (equal legs, opposite
    directions): one matrix product per irrep, C[h] = A[h] B[h].
    sort(t, order, split): the legs permuted and split anew (the "sort" of DPD codes).
'''

import itertools

import numpy

from socutils.symm import double_group as dg


class Group:
    '''Products and conjugates of irreps of a double group of socutils (name None: the
    trivial group).  Arrays are taken elementwise.'''

    def __init__(self, name=None):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Group) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def conj(self, a):
        if self.name is None:
            return numpy.zeros_like(a) if numpy.ndim(a) else 0
        return dg.conj(a, self.name)

    def prod(self, a, b):
        if self.name is None:
            return numpy.zeros(numpy.broadcast(a, b).shape, dtype=int) if (
                numpy.ndim(a) or numpy.ndim(b)) else 0
        if not numpy.ndim(a) and not numpy.ndim(b):
            return dg.direct_prod(int(a), int(b), self.name)
        a, b = numpy.broadcast_arrays(numpy.asarray(a), numpy.asarray(b))
        ua, ia = numpy.unique(a, return_inverse=True)
        ub, ib = numpy.unique(b, return_inverse=True)
        table = numpy.asarray(dg.direct_prod(ua, ub, self.name)).reshape(ua.size, ub.size)
        return table[ia.reshape(a.shape), ib.reshape(b.shape)]

    def combine(self, irreps, dirs):
        '''Irrep of a product of orbitals: dirs[k] = +1 takes irreps[k], -1 its conjugate.'''
        out = 0
        for ir, d in zip(irreps, dirs):
            out = self.prod(out, ir if d > 0 else self.conj(ir))
        return out


class Leg:
    '''An index: sectors of one irrep each, in order (irrep None: no symmetry).'''

    def __init__(self, irreps, sizes, name=''):
        self.irreps = tuple(None if x is None else int(x) for x in irreps)
        self.sizes = tuple(int(n) for n in sizes)
        self.offsets = numpy.concatenate([[0], numpy.cumsum(self.sizes, dtype=int)])
        self.name = name

    @classmethod
    def from_orbsym(cls, orbsym, name=''):
        '''The leg of orbitals with irreps orbsym, which must come in runs of equal
        irreps (sorted by sector); None: one sector without symmetry.'''
        orbsym = numpy.asarray(orbsym)
        if orbsym.size == 0:
            return cls([], [], name)
        starts = [0] + [k for k in range(1, orbsym.size) if orbsym[k] != orbsym[k - 1]]
        irreps = [int(orbsym[k]) for k in starts]
        if len(set(irreps)) != len(irreps):
            raise ValueError('the orbitals of a leg must be sorted by irrep')
        sizes = numpy.diff(starts + [orbsym.size])
        return cls(irreps, sizes, name)

    @classmethod
    def unsymmetric(cls, size, name=''):
        return cls([None], [size], name)

    @property
    def size(self):
        return int(self.offsets[-1])

    @property
    def symmetric(self):
        return None not in self.irreps

    def __len__(self):
        return len(self.sizes)

    def key(self):
        return (self.irreps, self.sizes)

    def __eq__(self, other):
        return isinstance(other, Leg) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return 'Leg(%s, %s%s)' % (list(self.irreps), list(self.sizes),
                                 ', %r' % self.name if self.name else '')


class _Side:
    '''The multi-indices of a group of legs, grouped by their composite irrep: for every
    irrep h, the sector tuples of irrep h in C order with their offset in the rows (or
    columns) of h.'''

    def __init__(self, legs, dirs, group, symmetric):
        # symmetric: whether the tensor as a whole conserves symmetry (no leg without it);
        # otherwise all multi-indices are in one group, h = None
        self.groups = {}           # h -> [(sectors, offset, size)]
        self.total = {}            # h -> number of multi-indices
        self.where = {}            # sectors -> (h, offset, size)
        for sec in itertools.product(*(range(len(leg)) for leg in legs)):
            size = int(numpy.prod([leg.sizes[s] for leg, s in zip(legs, sec)], dtype=int))
            if size == 0:
                continue
            if symmetric:
                h = int(group.combine([leg.irreps[s] for leg, s in zip(legs, sec)], dirs))
            else:
                h = None
            off = self.total.get(h, 0)
            self.groups.setdefault(h, []).append((sec, off, size))
            self.where[sec] = (h, off, size)
            self.total[h] = off + size


class Structure:
    '''Row and column multi-indices and the matrices M[h] of a Tensor; shared by all
    tensors of the same legs, directions, split, group and irrep.'''

    _cache = {}

    def __new__(cls, legs, dirs, split, group, irrep=0):
        key = (tuple(legs), tuple(dirs), split, group, irrep)
        obj = cls._cache.get(key)
        if obj is not None:
            return obj
        obj = super().__new__(cls)
        obj.legs, obj.dirs, obj.split, obj.group, obj.irrep = key[0], key[1], split, group, irrep
        obj.symmetric = all(leg.symmetric for leg in obj.legs)
        obj.rows = _Side(obj.legs[:split], obj.dirs[:split], group, obj.symmetric)
        obj.cols = _Side(obj.legs[split:], obj.dirs[split:], group, obj.symmetric)
        obj.matrices = {}          # h -> (row h, column h, offset, nrow, ncol)
        off = 0
        for h, nrow in obj.rows.total.items():
            hc = obj.col_irrep(h)
            ncol = obj.cols.total.get(hc, 0)
            if ncol:
                obj.matrices[h] = (h, hc, off, nrow, ncol)
                off += nrow * ncol
        obj.size = off
        cls._cache[key] = obj
        return obj

    def col_irrep(self, h):
        '''Composite irrep of the columns of M[h]: conj(h) x irrep.'''
        if h is None or not self.symmetric:
            return None
        return int(self.group.prod(self.group.conj(h), self.irrep))

    @property
    def shape(self):
        return tuple(leg.size for leg in self.legs)


class Tensor:
    '''A tensor in the DPD layout (see the module documentation).'''

    def __init__(self, legs, dirs, split, group=None, irrep=0, data=None):
        group = group if isinstance(group, Group) else Group(group)
        self.structure = Structure(tuple(legs), tuple(int(d) for d in dirs), split, group,
                                   irrep)
        st = self.structure
        if data is None:
            data = numpy.zeros(st.size, dtype=numpy.complex128)
        if data.size != st.size:
            raise ValueError('data of size %d for a structure of size %d' % (data.size, st.size))
        self.data = data

    @property
    def legs(self):
        return self.structure.legs

    @property
    def dirs(self):
        return self.structure.dirs

    @property
    def split(self):
        return self.structure.split

    @property
    def group(self):
        return self.structure.group

    @property
    def shape(self):
        return self.structure.shape

    def matrix(self, h):
        '''M[h] as a view of the buffer (None if empty).'''
        m = self.structure.matrices.get(h)
        if m is None:
            return None
        _, _, off, nrow, ncol = m
        return self.data[off:off + nrow * ncol].reshape(nrow, ncol)

    def matrices(self):
        for h in self.structure.matrices:
            yield h, self.matrix(h)

    # ------------------------------------------------------------ dense arrays
    def _index_arrays(self, side, legs, h):
        '''Flat positions (C order over legs) of the multi-indices of irrep h.'''
        dims = tuple(leg.size for leg in legs)
        out = []
        for sec, _, _ in side.groups[h]:
            ranges = [numpy.arange(leg.offsets[s], leg.offsets[s + 1])
                      for leg, s in zip(legs, sec)]
            out.append(numpy.ravel_multi_index(numpy.ix_(*ranges), dims).ravel()
                       if ranges else numpy.zeros(1, dtype=numpy.int64))
        return numpy.concatenate(out)

    @classmethod
    def from_dense(cls, a, legs, dirs, split, group=None, irrep=0):
        '''The elements of a allowed by symmetry (the others are dropped).'''
        t = cls(legs, dirs, split, group, irrep)
        st = t.structure
        rl, cl = st.legs[:split], st.legs[split:]
        a2 = numpy.asarray(a, dtype=numpy.complex128).reshape(
            int(numpy.prod([x.size for x in rl], dtype=int)),
            int(numpy.prod([x.size for x in cl], dtype=int)))
        for h, (_, hc, _, _, _) in st.matrices.items():
            ri = t._index_arrays(st.rows, rl, h)
            ci = t._index_arrays(st.cols, cl, hc)
            t.matrix(h)[...] = a2[numpy.ix_(ri, ci)]
        return t

    def to_dense(self):
        st = self.structure
        rl, cl = st.legs[:st.split], st.legs[st.split:]
        nr = int(numpy.prod([x.size for x in rl], dtype=int))
        nc = int(numpy.prod([x.size for x in cl], dtype=int))
        out = numpy.zeros((nr, nc), dtype=numpy.complex128)
        for h, (_, hc, _, _, _) in st.matrices.items():
            ri = self._index_arrays(st.rows, rl, h)
            ci = self._index_arrays(st.cols, cl, hc)
            out[numpy.ix_(ri, ci)] = self.matrix(h)
        return out.reshape(st.shape)

    # ------------------------------------------------------------------ boxes
    def boxes(self):
        '''{sector tuple (one sector per leg): (base, strides, sizes)} of every block:
        the block is the box of self.data at element base with the element strides and
        sizes of its legs (in the order of the legs).  Kept on the structure.'''
        st = self.structure
        cached = st.__dict__.get('_boxes')
        if cached is not None:
            return cached
        out = {}
        for h, (_, hc, off, nrow, ncol) in st.matrices.items():
            for rsec, roff, _ in st.rows.groups[h]:
                rs = [st.legs[k].sizes[x] for k, x in enumerate(rsec)]
                for csec, coff, _ in st.cols.groups[hc]:
                    cs = [st.legs[st.split + k].sizes[x] for k, x in enumerate(csec)]
                    strides = []
                    for k in range(len(rs)):
                        strides.append(ncol * int(numpy.prod(rs[k + 1:], dtype=int)))
                    for k in range(len(cs)):
                        strides.append(int(numpy.prod(cs[k + 1:], dtype=int)))
                    out[rsec + csec] = (off + roff * ncol + coff, tuple(strides), tuple(rs + cs))
        st._boxes = out
        return out

    # --------------------------------------------------------------- algebra
    def copy(self):
        st = self.structure
        return Tensor(st.legs, st.dirs, st.split, st.group, st.irrep, self.data.copy())

    def conj(self):
        '''Complex conjugate; the directions are reversed (and the irrep conjugated).'''
        st = self.structure
        return Tensor(st.legs, tuple(-d for d in st.dirs), st.split, st.group,
                      st.group.conj(st.irrep), self.data.conj())

    def norm(self):
        return float(numpy.linalg.norm(self.data))

    def __iadd__(self, other):
        if other.structure is not self.structure:
            raise ValueError('add needs tensors of one structure (sort first)')
        self.data += other.data
        return self

    def __imul__(self, x):
        self.data *= x
        return self


def sort(t, order, split):
    '''t with its legs in the order `order` (a permutation of range(rank)) and the first
    `split` of them as row legs.'''
    st = t.structure
    legs = tuple(st.legs[k] for k in order)
    dirs = tuple(st.dirs[k] for k in order)
    out = Tensor(legs, dirs, split, st.group, st.irrep)
    new = out.structure
    rank = len(order)
    for h, (_, hc, _, _, _) in st.matrices.items():
        m = t.matrix(h)
        for rsec, roff, rsize in st.rows.groups[h]:
            for csec, coff, csize in st.cols.groups[hc]:
                sec = rsec + csec
                blk = m[roff:roff + rsize, coff:coff + csize].reshape(
                    [st.legs[k].sizes[sec[k]] for k in range(rank)])
                nsec = tuple(sec[k] for k in order)
                h2, r2, n2 = new.rows.where[nsec[:split]]
                hc2, c2, m2 = new.cols.where[nsec[split:]]
                out.matrix(h2)[r2:r2 + n2, c2:c2 + m2] = blk.transpose(order).reshape(n2, m2)
    return out


def contract(a, b):
    '''a b over the column legs of a, which are the row legs of b (equal legs, opposite
    directions): the tensor of the row legs of a and the column legs of b.'''
    sa, sb = a.structure, b.structure
    ca, rb = sa.legs[sa.split:], sb.legs[:sb.split]
    if ca != rb:
        raise ValueError('the column legs of a are not the row legs of b')
    if any(x != -y for x, y in zip(sa.dirs[sa.split:], sb.dirs[:sb.split])):
        raise ValueError('contracted legs must have opposite directions')
    if sa.group != sb.group:
        raise ValueError('different groups')
    irrep = sa.group.prod(sa.irrep, sb.irrep) if sa.symmetric else 0
    out = Tensor(sa.legs[:sa.split] + sb.legs[sb.split:],
                 sa.dirs[:sa.split] + sb.dirs[sb.split:], sa.split, sa.group, irrep)
    for h, ma in a.matrices():
        if ma is None:
            continue
        # the columns of a (irrep hc of a) are rows of b of the conjugate irrep with
        # the reversed directions, i.e. of irrep conj(hc) in b
        hc = sa.matrices[h][1]
        hb = None if hc is None else int(sa.group.conj(hc))
        mb = b.matrix(hb)
        mo = out.matrix(h)
        if mb is None or mo is None:
            continue
        mo += ma @ mb
    return out


def contract_symmetric(a, b, structure):
    '''a b (column legs of a = row legs of b, as in contract) for operands that do not
    conserve symmetry (a leg without it, e.g. the auxiliary index of three-index
    integrals) into a product that does: only the matrices of `structure` (the
    symmetric structure of the result) are formed, from the rows of a and the columns
    of b of the irreps they couple.'''
    sa, sb = a.structure, b.structure
    if sa.legs[sa.split:] != sb.legs[:sb.split]:
        raise ValueError('the column legs of a are not the row legs of b')
    ma, mb = a.matrix(None), b.matrix(None)
    out = Tensor(structure.legs, structure.dirs, structure.split, structure.group,
                 structure.irrep)
    st = out.structure
    if st.legs[:st.split] != sa.legs[:sa.split] or st.legs[st.split:] != sb.legs[sb.split:]:
        raise ValueError('structure does not match the free legs of a and b')
    # positions of the multi-indices of every irrep in the single groups of a and b
    rpos = {sec: off for sec, off, _ in sa.rows.groups[None]}
    cpos = {sec: off for sec, off, _ in sb.cols.groups[None]}
    for h, (_, hc, _, _, _) in st.matrices.items():
        rows = numpy.concatenate([numpy.arange(rpos[sec], rpos[sec] + n)
                                  for sec, _, n in st.rows.groups[h]])
        cols = numpy.concatenate([numpy.arange(cpos[sec], cpos[sec] + n)
                                  for sec, _, n in st.cols.groups[hc]])
        out.matrix(h)[...] = ma[rows] @ mb[:, cols]
    return out


def identity(leg, group=None, dirs=(1, -1)):
    '''delta[p, q] on one leg (a density-like tensor: p created, q annihilated).'''
    t = Tensor((leg, leg), dirs, 1, group)
    for h, m in t.matrices():
        m[...] = numpy.eye(m.shape[0])
    return t


def map_blocks(t, fn):
    '''Apply fn(sectors, block) in place to every block (a view of shape
    (sizes of the sectors of every leg)), e.g. a denominator from orbital energies.'''
    st = t.structure
    rank = len(st.legs)
    for h, (_, hc, _, _, _) in st.matrices.items():
        m = t.matrix(h)
        for rsec, roff, rsize in st.rows.groups[h]:
            for csec, coff, csize in st.cols.groups[hc]:
                sec = rsec + csec
                shape = [st.legs[k].sizes[sec[k]] for k in range(rank)]
                view = m[roff:roff + rsize, coff:coff + csize]
                blk = view.reshape(shape) if view.flags.c_contiguous else None
                if blk is None:
                    blk = view.copy().reshape(shape)
                    fn(sec, blk)
                    view[...] = blk.reshape(rsize, csize)
                else:
                    fn(sec, blk)


def einsum(spec, a, b, out=None, alpha=1.0, split=None):
    '''Two-operand einsum without traces: letters shared by a and b and absent from the
    output are contracted (their legs equal, directions opposite).  The result has the
    legs in the order of the output letters, split after `split` of them (default: the
    split of out, else after the free letters of a); with out it is added, times alpha.'''
    lhs, rhs = spec.replace(' ', '').split('->')
    sa_, sb_ = lhs.split(',')
    con = [x for x in sa_ if x in sb_ and x not in rhs]
    aown = [x for x in sa_ if x not in con]
    bown = [x for x in sb_ if x not in con]
    if set(aown) & set(bown) or len(set(sa_)) != len(sa_) or len(set(sb_)) != len(sb_):
        raise ValueError('einsum: no traces or Hadamard products (%s)' % spec)
    if sorted(aown + bown) != sorted(rhs):
        raise ValueError('einsum: output letters %r' % rhs)
    ta = sort(a, [sa_.index(x) for x in aown + con], len(aown))
    tb = sort(b, [sb_.index(x) for x in con + bown], len(con))
    c = contract(ta, tb)
    cur = aown + bown
    order = [cur.index(x) for x in rhs]
    if split is None:
        split = out.split if out is not None else len(aown)
    c = sort(c, order, split) if (order != list(range(len(order))) or split != c.split) else c
    if out is None:
        if alpha != 1.0:
            c *= alpha
        return c
    if c.structure is not out.structure:
        raise ValueError('einsum: the result does not have the structure of out')
    out.data += alpha * c.data
    return out


# [number, bytes] of the expansions of Views to dense arrays (a diagnostic)
DENSIFIED = [0, 0]


class View:
    '''t with its legs permuted (axis k of the view is leg perm[k] of t) and optionally
    complex conjugated, without a copy; numpy.asarray gives the dense array.'''

    def __init__(self, t, perm=None, conj=False):
        self.t = t
        self.perm = tuple(range(len(t.legs))) if perm is None else tuple(perm)
        self.conjugated = bool(conj)

    @property
    def shape(self):
        return tuple(self.t.legs[k].size for k in self.perm)

    @property
    def ndim(self):
        return len(self.perm)

    @property
    def size(self):
        return int(numpy.prod(self.shape, dtype=int))

    dtype = numpy.dtype(numpy.complex128)

    def transpose(self, perm):
        return View(self.t, [self.perm[k] for k in perm], self.conjugated)

    def conj(self):
        return View(self.t, self.perm, not self.conjugated)

    def __array__(self, dtype=None, copy=None):
        # counted: an expansion is a fallback that costs memory and time
        DENSIFIED[0] += 1
        DENSIFIED[1] += 16 * self.size
        d = self.t.to_dense().transpose(self.perm)
        d = d.conj() if self.conjugated else d
        return numpy.ascontiguousarray(d, dtype=dtype)


def _permuted_boxes(t, perm):
    '''{sector tuple in the order perm: (base, strides, sizes in the order perm)}; kept
    on the structure for every perm.'''
    st = t.structure
    cache = st.__dict__.setdefault('_permuted', {})
    perm = tuple(perm)
    if perm in cache:
        return cache[perm]
    out = {}
    for sec, (base, st, sz) in t.boxes().items():
        out[tuple(sec[k] for k in perm)] = (base, tuple(st[k] for k in perm),
                                            tuple(sz[k] for k in perm))
    cache[perm] = out
    return out


def add_permuted(dst, src, perm, alpha=1.0, conj=False, assign=False):
    '''dst += alpha (conj) src.transpose(perm), block by block (assign: dst = ...);
    axis k of the permuted src is leg perm[k] of src.  The blocks of src must be blocks
    of dst (same legs after the permutation).'''
    from socutils.zmr.lib import kernels
    sboxes = _permuted_boxes(src, perm)
    dboxes = dst.boxes()
    keys = [k for k in sboxes if k in dboxes]
    if not keys:
        return dst
    rank = len(perm)
    nrow = max(1, rank // 2) if rank > 1 else rank
    nrow = min(nrow, rank)
    doff, ld, off = [], [], 0
    for k in keys:
        sz = sboxes[k][2]
        nc = int(numpy.prod(sz[nrow:], dtype=int))
        doff.append(off)
        ld.append(nc)
        off += int(numpy.prod(sz, dtype=int))
    buf = numpy.empty(off, dtype=numpy.complex128)
    dims = [sboxes[k][2] for k in keys]
    kernels.box_copy_strided(0, src.data, buf, nrow, [sboxes[k][0] for k in keys],
                             [sboxes[k][1] for k in keys], dims, doff, ld, conj=conj)
    kernels.box_copy_strided(1 if assign else 2, dst.data, buf, nrow,
                             [dboxes[k][0] for k in keys], [dboxes[k][1] for k in keys],
                             dims, doff, ld, alpha=alpha)
    return dst
