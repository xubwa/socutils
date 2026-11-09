'''
Storage of blocked operators shared by the methods of zmr (DSRG, NEVPT, ...).

An operator of a method is a mapping space label -> block ("ccvv", "aavv", ...).  A
Layout decides how each block is held: in the DPD layout (zmr.dpd.Tensor, only the
elements allowed by the double-group symmetry) when the method uses the double-group
sectors and every space of the label has them, else as a dense array.  The method code
then only writes its equations; these helpers do the rest for either storage:

    layout = Layout(dims, use_dpd=True)         dims: zmr.dsrg.commutator.Dims
    t = layout.zeros('ccvv')                    a zero block
    t = layout.block('ccvv', x, alpha)          a copy of alpha x (x as returned by get:
                                                array, Tensor, View, (block, sign[, conj]))
    layout.add(t, x, alpha)                     t += alpha x
    layout.fill('ccvv', fn)                     block of values fn(ranges): ranges[k] the
                                                orbital indices (within its space) of
                                                leg k of a box; e.g. denominators
    layout.denominators('ccvv', eps, s)         (1 - exp(-s D^2)) / D and exp(-s D^2)
    data(t)                                     the stored elements (for elementwise
                                                algebra, norms, DIIS vectors)
    vdot(a, b)                                  sum conj(a) b over the stored elements

Blocks of one label made by one Layout share their layout, so elementwise products of
data(...) of such blocks are elementwise products of the operators.  A block's
directions (which indices are created, +1, or annihilated, -1) are given by `signs`,
in the convention of zmr.dsrg.commutator (X[annihilated..., created...]): TWO_BODY
for two-body operators and amplitudes, ONE_BODY for one-body ones.
'''

import numpy

from socutils.zmr import dpd

ONE_BODY = (-1, 1)
TWO_BODY = (-1, -1, 1, 1)
_DEFAULT_SIGNS = {2: ONE_BODY, 4: TWO_BODY}


def data(x):
    '''The stored elements of a block: the buffer of a dpd.Tensor, a dense array itself.'''
    return x.data if isinstance(x, dpd.Tensor) else x


def vdot(a, b):
    '''sum conj(a) b of two blocks of one label and layout.'''
    return numpy.vdot(data(a).ravel(), data(b).ravel())


def add_block(t, x, alpha=1.0, beta=1.0):
    '''t = beta t + alpha x for a block t (dpd.Tensor or dense array) and x as returned by
    get (an array, a dpd.Tensor or View, or (block, sign) / (block, sign, conjugate)) in
    the order of the indices of t; None adds nothing.  A Tensor t receives only its
    blocks (the elements allowed by symmetry).'''
    if x is None:
        if beta != 1.0:
            data(t)[...] *= beta
        return t
    if isinstance(x, tuple):
        blk, sign, cj = x[0], x[1], len(x) == 3 and bool(x[2])
    else:
        blk, sign, cj = x, 1.0, False
    if isinstance(blk, dpd.Tensor):
        blk = dpd.View(blk)
    if isinstance(t, dpd.Tensor):
        if isinstance(blk, dpd.View):
            return dpd.sort_axpy(t, blk.t, blk.perm, alpha * sign, beta,
                                 conj=cj != blk.conjugated)
        return dpd.dense_to_tensor(numpy.asarray(blk), t, alpha * sign, beta, conj=cj)
    if isinstance(blk, dpd.View):
        inv = numpy.argsort(blk.perm)
        return dpd.tensor_to_dense(blk.t, t, alpha * sign, beta, conj=cj != blk.conjugated,
                                   perm=list(inv))
    blk = numpy.asarray(blk)
    if cj:
        blk = blk.conj()
    if beta == 0.0:
        t[...] = alpha * sign * blk
    else:
        if beta != 1.0:
            t *= beta
        t += alpha * sign * blk
    return t


class Layout:
    '''How the blocks of the operators of one calculation are stored (see the module
    documentation).  dims: the dimensions of the spaces with their double-group sectors
    (Dims.sym, Dims.legs); use_dpd False keeps every block dense.'''

    def __init__(self, dims, use_dpd=True):
        self.dims = dims
        sym = getattr(dims, 'sym', None)
        self.group = sym[0] if sym is not None else None
        self.legs = getattr(dims, 'legs', None) or {}
        self.use_dpd = bool(use_dpd) and self.group is not None and bool(self.legs)
        self._cache = {}

    def is_dpd(self, lab, signs=None):
        '''Whether the block lab is held in the DPD layout.'''
        signs = _DEFAULT_SIGNS.get(len(lab)) if signs is None else signs
        return (self.use_dpd and signs is not None and len(lab) == len(signs) and len(lab) >= 2
                and all(x in self.legs for x in lab))

    def shape(self, lab):
        return tuple(self.dims[x] for x in lab)

    def zeros(self, lab, signs=None, dtype=numpy.complex128):
        signs = _DEFAULT_SIGNS.get(len(lab)) if signs is None else signs
        if self.is_dpd(lab, signs):
            return dpd.Tensor([self.legs[x] for x in lab], signs, len(lab) // 2, self.group,
                              dtype=dtype)
        return numpy.zeros(self.shape(lab), dtype=dtype)

    def block(self, lab, x, alpha=1.0, signs=None):
        '''A new block lab holding alpha x (x as add_block takes it; None: zero).'''
        t = self.zeros(lab, signs)
        return add_block(t, x, alpha, beta=0.0)

    add = staticmethod(add_block)

    def fill(self, lab, fn, signs=None, dtype=numpy.complex128):
        '''A block lab with the values fn(ranges), box by box: ranges[k] is the range of
        orbital indices (within the space lab[k]) of leg k of the box, fn returns an
        array of the box shape (broadcastable).  Dense: one box, the whole block.'''
        t = self.zeros(lab, signs, dtype)
        if isinstance(t, dpd.Tensor):
            legs = t.legs

            def put(sec, blk):
                blk[...] = fn([range(legs[k].offsets[s], legs[k].offsets[s + 1])
                               for k, s in enumerate(sec)])
            dpd.map_blocks(t, put)
        else:
            t[...] = fn([range(n) for n in t.shape])
        return t

    def denominators(self, lab, eps, s, flip=False):
        '''((1 - exp(-s D^2)) / D, exp(-s D^2)) as blocks of lab, D = sum of eps of the
        first half of the indices less those of the second half (flip: the opposite,
        the D of the excitation of a de-excitation label); kept for (lab, eps, s, flip).'''
        key = ('den', lab, id(eps), s, flip)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is eps:
            return hit[1]
        n = len(lab) // 2
        sign = -1.0 if flip else 1.0

        def delta(ranges):
            d = 0.0
            for k, (x, rg) in enumerate(zip(lab, ranges)):
                shape = [1] * len(lab)
                shape[k] = -1
                e = numpy.asarray(eps[x])[rg.start:rg.stop]
                d = d + sign * (e if k < n else -e).reshape(shape)
            return d

        def r(ranges):
            d = delta(ranges)
            small = abs(d) < 1e-7
            return numpy.where(small, s * d, (1.0 - numpy.exp(-s * d * d))
                               / numpy.where(small, 1.0, d))

        out = (self.fill(lab, r, dtype=numpy.float64),
               self.fill(lab, lambda rg: numpy.exp(-s * delta(rg) ** 2), dtype=numpy.float64))
        self._cache[key] = (eps, out)
        return out
