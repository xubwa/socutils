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


class XorGroup:
    '''The single groups D2h and its subgroups of PySCF: irrep ids (pyscf.symm,
    mo_coeff.orbsym) multiply by XOR, every irrep is its own conjugate.'''

    name = 'xor'

    def __eq__(self, other):
        return isinstance(other, XorGroup)

    def __hash__(self):
        return hash('xor')

    def conj(self, a):
        return a

    def prod(self, a, b):
        return numpy.bitwise_xor(a, b) if (numpy.ndim(a) or numpy.ndim(b)) else int(a) ^ int(b)

    def combine(self, irreps, dirs):
        out = 0
        for ir in irreps:
            out = self.prod(out, ir)
        return out


class U1Group:
    '''An additive integer quantum number (m_l of a linear molecule, 2 S_z, ...):
    products add, the conjugate is the negative.'''

    name = 'u1'

    def __eq__(self, other):
        return isinstance(other, U1Group)

    def __hash__(self):
        return hash('u1')

    def conj(self, a):
        return -numpy.asarray(a) if numpy.ndim(a) else -int(a)

    def prod(self, a, b):
        return numpy.add(a, b) if (numpy.ndim(a) or numpy.ndim(b)) else int(a) + int(b)

    def combine(self, irreps, dirs):
        out = 0
        for ir, d in zip(irreps, dirs):
            out = self.prod(out, ir if d > 0 else self.conj(ir))
        return out


class ProductGroup:
    '''The direct product of groups (e.g. a spatial point group and 2 S_z for spin
    orbitals): irreps are tuples, one irrep of every factor.'''

    def __init__(self, *groups):
        self.groups = tuple(groups)
        self.name = 'x'.join(str(getattr(g, 'name', g)) for g in groups)

    def __eq__(self, other):
        return isinstance(other, ProductGroup) and self.groups == other.groups

    def __hash__(self):
        return hash(self.groups)

    def conj(self, a):
        if a == 0:
            return 0
        return tuple(g.conj(x) for g, x in zip(self.groups, a))

    def prod(self, a, b):
        if a == 0:
            return b
        if b == 0:
            return a
        return tuple(g.prod(x, y) for g, x, y in zip(self.groups, a, b))

    def combine(self, irreps, dirs):
        out = tuple(0 for _ in self.groups)
        for ir, d in zip(irreps, dirs):
            out = self.prod(out, ir if d > 0 else self.conj(ir))
        return out

    def identity(self):
        return tuple(0 for _ in self.groups)


def _irrep(x):
    '''An irrep as a hashable key: int, or a tuple of ints for a product group.'''
    if isinstance(x, tuple):
        return tuple(int(y) for y in x)
    return int(x)


def as_group(group):
    '''A group object: one with prod / conj / combine as is, a socutils double-group name
    or None (no symmetry) as Group.'''
    if hasattr(group, 'prod') and hasattr(group, 'combine'):
        return group
    return Group(group)


class Leg:
    '''An index: sectors of one irrep each, in order (irrep None: no symmetry).'''

    def __init__(self, irreps, sizes, name=''):
        self.irreps = tuple(None if x is None else (tuple(int(y) for y in x)
                                                     if isinstance(x, tuple) else int(x))
                            for x in irreps)
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
                h = _irrep(group.combine([leg.irreps[s] for leg, s in zip(legs, sec)], dirs))
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
        return _irrep(self.group.prod(self.group.conj(h), self.irrep))

    @property
    def shape(self):
        return tuple(leg.size for leg in self.legs)


class Tensor:
    '''A tensor in the DPD layout (see the module documentation).'''

    def __init__(self, legs, dirs, split, group=None, irrep=0, data=None,
                 dtype=numpy.complex128):
        group = as_group(group)
        self.structure = Structure(tuple(legs), tuple(int(d) for d in dirs), split, group,
                                   irrep)
        st = self.structure
        if data is None:
            data = numpy.zeros(st.size, dtype=dtype)
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
    def from_dense(cls, a, legs, dirs, split, group=None, irrep=0, dtype=None):
        '''The elements of a allowed by symmetry (the others are dropped).'''
        a = numpy.asarray(a)
        dtype = dtype or (numpy.float64 if a.dtype.kind in 'fiu' else numpy.complex128)
        t = cls(legs, dirs, split, group, irrep, dtype=dtype)
        st = t.structure
        rl, cl = st.legs[:split], st.legs[split:]
        a2 = numpy.asarray(a, dtype=dtype).reshape(
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
        out = numpy.zeros((nr, nc), dtype=self.data.dtype)
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

    @property
    def dtype(self):
        return self.data.dtype

    def conj(self):
        '''Complex conjugate; the directions are reversed (and the irrep conjugated).'''
        st = self.structure
        return Tensor(st.legs, tuple(-d for d in st.dirs), st.split, st.group,
                      st.group.conj(st.irrep), self.data.conj())

    def flipped(self):
        '''The same elements with all directions reversed (and the irrep conjugated):
        the blocks of the reversed directions are those of the conjugate irreps, in the
        same order, so the buffer is shared.'''
        st = self.structure
        return Tensor(st.legs, tuple(-d for d in st.dirs), st.split, st.group,
                      _irrep(st.group.conj(st.irrep)) if st.symmetric else st.irrep,
                      self.data)

    def norm(self):
        return float(numpy.linalg.norm(self.data))

    @property
    def nbytes(self):
        return self.data.nbytes

    def __iadd__(self, other):
        if other.structure is not self.structure:
            raise ValueError('add needs tensors of one structure (sort first)')
        self.data += other.data
        return self

    def __imul__(self, x):
        self.data *= x
        return self


def _one_piece(x):
    '''Whether the elements of x fill one contiguous piece of memory from its first
    element on (positive strides that are a permutation of those of a dense array).'''
    if x.flags.c_contiguous:
        return True
    dims = sorted((st, n) for st, n in zip(x.strides, x.shape) if n > 1)
    want = x.itemsize
    for st, n in dims:
        if st != want:
            return False
        want *= n
    return True


def _flat(a, dtype):
    '''(flat buffer, element strides, offset) of a: a view of an array in one contiguous
    piece of memory (C or Fortran order, axes in any order) keeps its base,
    anything else is copied.'''
    base = a
    while isinstance(base, numpy.ndarray) and isinstance(base.base, numpy.ndarray):
        base = base.base
    if (a.dtype == dtype and isinstance(base, numpy.ndarray) and base.dtype == dtype
            and _one_piece(base) and all(st % a.itemsize == 0 for st in a.strides)):
        flat = numpy.lib.stride_tricks.as_strided(base, shape=(base.size,),
                                                  strides=(base.itemsize,))
        off = a.__array_interface__['data'][0] - flat.__array_interface__['data'][0]
        if off % a.itemsize == 0 and 0 <= off // a.itemsize < max(flat.size, 1):
            return flat, [st // a.itemsize for st in a.strides], off // a.itemsize
    a = numpy.ascontiguousarray(a, dtype=dtype)
    return a.reshape(-1), [st // a.itemsize for st in a.strides], 0


def _dense_plan(t, stride):
    '''(dense bases, dense strides, tensor bases, tensor strides, sizes) of the blocks of
    t as int64 arrays, a dense array of the element strides `stride` in the order of
    the legs; kept on the structure.'''
    st = t.structure
    cache = st.__dict__.setdefault('_dense_plans', {})
    key = tuple(stride)
    hit = cache.get(key)
    if hit is not None:
        return hit
    boxes = t.boxes()
    keys = list(boxes)
    legs = t.legs
    i64 = lambda x, n: numpy.asarray(x, dtype=numpy.int64).reshape(n)
    nb, nd = len(keys), len(legs)
    hit = (i64([sum(int(legs[k].offsets[sec[k]]) * stride[k] for k in range(nd))
                for sec in keys], (nb,)),
           numpy.ascontiguousarray(numpy.broadcast_to(i64(stride, (nd,)), (nb, nd))),
           i64([boxes[k][0] for k in keys], (nb,)),
           i64([boxes[k][1] for k in keys], (nb, nd)),
           i64([boxes[k][2] for k in keys], (nb, nd)))
    cache[key] = hit
    return hit


def dense_to_tensor(a, t, alpha=1.0, beta=0.0, conj=False, perm=None):
    '''t = beta t + alpha (conj) a on the blocks of t, a dense array whose axis perm[k]
    is leg k of t (default the same order); a view of a C-contiguous array is read
    in place (one native call).'''
    from socutils.zmr.lib import kernels
    a = numpy.asarray(a)
    if perm is not None:
        a = a.transpose(perm)
    if not t.boxes():
        if beta != 1.0:
            t.data *= beta
        return t
    flat, stride, off = _flat(a, t.data.dtype)
    dbase, dstride, tbase, tstride, dims = _dense_plan(t, stride)
    kernels.dpd_copy(flat, t.data, dbase + off, dstride, tbase, tstride, dims,
                     alpha, beta, conj)
    return t


def tensor_to_dense(t, out=None, alpha=1.0, beta=0.0, conj=False, perm=None):
    '''out = beta out + alpha (conj) t on the blocks of t (zero elsewhere for a new out;
    beta only scales the blocks of t); leg k of t is axis perm[k] of out (default the
    same order); out any view of a C-contiguous array, written in place.'''
    from socutils.zmr.lib import kernels
    if out is None:
        shape = t.shape if perm is None else tuple(
            t.shape[k] for k in numpy.argsort(perm))
        out = numpy.zeros(shape, dtype=t.data.dtype)
        beta = 0.0
    if not t.boxes():
        return out
    view = out if perm is None else out.transpose(perm)
    flat, stride, off = _flat(view, t.data.dtype)
    if not numpy.shares_memory(flat, out):
        raise ValueError('tensor_to_dense: out must be a view of a C-contiguous array')
    dbase, dstride, tbase, tstride, dims = _dense_plan(t, stride)
    kernels.dpd_copy(t.data, flat, tbase, tstride, dbase + off, dstride, dims,
                     alpha, beta, conj)
    return out


def sort_axpy(dst, src, perm, alpha=1.0, beta=1.0, conj=False):
    '''dst = beta dst + alpha (conj) src.transpose(perm), block by block, one native call
    (axis k of the permuted src is leg perm[k] of src).  Every block of src must be a
    block of dst after the permutation; the blocks of dst that src does not have are
    only scaled by beta.'''
    from socutils.zmr.lib import kernels
    perm = tuple(perm)
    plan = _sort_plan(src, perm, dst)
    if plan is None:
        if beta != 1.0:
            dst.data *= beta
        return dst
    complete = plan[5]
    if beta not in (0.0, 1.0) and not complete:
        dst.data *= beta
        beta = 1.0
    if beta == 0.0 and not complete:
        dst.data[...] = 0
    kernels.dpd_copy(src.data, dst.data, *plan[:5], alpha, beta, conj)
    return dst


_SORT_PLANS = {}


def _sort_plan(src, perm, dst):
    '''(src bases, src strides, dst bases, dst strides, sizes, whether every block of dst
    is written) of sort_axpy as int64 arrays, None if no block is shared; cached by
    the two structures and perm.'''
    key = (src.structure, perm, dst.structure)
    if key in _SORT_PLANS:
        return _SORT_PLANS[key]
    sb = _permuted_boxes(src, perm)
    db = dst.boxes()
    keys = [k for k in sb if k in db]
    plan = None
    if keys:
        nb, nd = len(keys), len(perm)
        i64 = lambda x, n: numpy.asarray(x, dtype=numpy.int64).reshape(n)
        plan = (i64([sb[k][0] for k in keys], (nb,)), i64([sb[k][1] for k in keys], (nb, nd)),
                i64([db[k][0] for k in keys], (nb,)), i64([db[k][1] for k in keys], (nb, nd)),
                i64([db[k][2] for k in keys], (nb, nd)), len(keys) == len(db))
    _SORT_PLANS[key] = plan
    return plan


def sort(t, order, split, conj=False):
    '''t with its legs in the order `order` (a permutation of range(rank)) and the first
    `split` of them as row legs (the "sort" of DPD codes); one native call.'''
    st = t.structure
    legs = tuple(st.legs[k] for k in order)
    dirs = tuple(st.dirs[k] for k in order)
    if conj:
        dirs = tuple(-d for d in dirs)
    out = Tensor(legs, dirs, split, st.group, st.group.conj(st.irrep) if conj else st.irrep,
                 dtype=t.data.dtype)
    return sort_axpy(out, t, order, 1.0, 0.0, conj)


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
    from socutils.zmr.lib import linalg as la
    out = Tensor(sa.legs[:sa.split] + sb.legs[sb.split:],
                 sa.dirs[:sa.split] + sb.dirs[sb.split:], sa.split, sa.group, irrep,
                 dtype=numpy.result_type(a.data.dtype, b.data.dtype))
    products = []
    for h, ma in a.matrices():
        if ma is None:
            continue
        # the columns of a (irrep hc of a) are rows of b of the conjugate irrep with
        # the reversed directions, i.e. of irrep conj(hc) in b
        hc = sa.matrices[h][1]
        hb = None if hc is None else _irrep(sa.group.conj(hc))
        mb = b.matrix(hb)
        mo = out.matrix(h)
        if mb is None or mo is None:
            continue
        if ma.dtype != out.data.dtype:
            ma = ma.astype(out.data.dtype)
        if mb.dtype != out.data.dtype:
            mb = mb.astype(out.data.dtype)
        products.append((ma, mb, mo))
    la.gemm_batch(products, beta=1.0)
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


def _sorted(t, order, split):
    '''t itself if its legs are already in the order and split, else sort(t, ...).'''
    if list(order) == list(range(len(order))) and split == t.split:
        return t
    return sort(t, order, split)


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
    ta = _sorted(a, [sa_.index(x) for x in aown + con], len(aown))
    tb = _sorted(b, [sb_.index(x) for x in con + bown], len(con))
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
    axis k of the permuted src is leg perm[k] of src (sort_axpy).'''
    return sort_axpy(dst, src, perm, alpha, 0.0 if assign else 1.0, conj)
