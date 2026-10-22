'''
The product basis of the MRCI space.

A block B (zmr.wick.mrci.BLOCKS) has the functions E_e |Phi_c b_k>: external
strings E_e = a^+ b^+ j i and vectors b_k of the active sector N + dn(B),
the columns of Block.b.  The active vectors of a block are either

    'full'    every determinant of the sector (no contraction), or
    'pc'      an orthonormal basis of the span of O_X |Psi_J> over the active
              operators O_X of the class (zmr.uc.classes) and the model
              states J: the internally contracted functions,

and for the internal block 'cas' either every CAS determinant ('full', the
reference coefficients relax) or the model states ('states').

Coefficients are stored as arrays C[a, b, i, j, k] over the full ranges of
the external indices, antisymmetric in equivalent indices; pack / unpack
convert to and from the vector of the independent coefficients (a < b,
i < j), in which the basis is orthonormal.
'''

import numpy

from socutils.zmr.pc import spectral
from socutils.zmr.uc import classes, engine
from socutils.zmr.wick import mrci as terms

# which blocks are contracted
SCHEMES = {
    'fic': {name: 'pc' for name in terms.ORDER if name != 'cas'},
    'wk': {name: ('pc' if terms.BLOCKS[name][1] == 2 else 'full')
           for name in terms.ORDER if name != 'cas'},
    'none': {name: 'full' for name in terms.ORDER if name != 'cas'},
}


class Block:
    def __init__(self, name, nelec, b, ncore, nvirt):
        self.name = name
        self.nh, self.np = terms.BLOCKS[name]
        self.nelec = nelec                  # electrons of the active sector
        self.b = b                          # (dim of the sector, m)
        self.m = b.shape[1]
        self.shape = (nvirt,) * self.np + (ncore,) * self.nh + (self.m,)
        self._virt = _combinations(nvirt, self.np)
        self._core = _combinations(ncore, self.nh)
        self.size = len(self._virt[0]) * len(self._core[0]) * self.m

    def pack(self, c):
        '''Independent coefficients of C[..., a, b, i, j, k] (leading axes kept).'''
        lead = c.shape[:c.ndim - len(self.shape)]
        v, o = self._virt, self._core
        idx = tuple(x[:, None] for x in v[1:]) + tuple(x[None, :] for x in o[1:])
        out = c[(Ellipsis,) + idx + (slice(None),)] if idx else c
        return out.reshape(lead + (self.size,))

    def antisymmetrize(self, c):
        '''Projection of C[..., a, b, i, j, k] onto the arrays antisymmetric
        in equivalent external indices.'''
        n = c.ndim - len(self.shape)
        if self.np == 2:
            c = 0.5 * (c - c.swapaxes(n, n + 1))
        if self.nh == 2:
            c = 0.5 * (c - c.swapaxes(n + self.np, n + self.np + 1))
        return c

    def unpack(self, x):
        '''Antisymmetric array from the independent coefficients.'''
        lead = x.shape[:-1]
        v, o = self._virt, self._core
        x = x.reshape(lead + (len(v[0]), len(o[0]), self.m))
        out = numpy.zeros(lead + self.shape, dtype=x.dtype)
        for sv, pv in _permutations(v[1:]):
            for so, po in _permutations(o[1:]):
                idx = tuple(a[:, None] for a in pv) + tuple(a[None, :] for a in po)
                if idx:
                    out[(Ellipsis,) + idx + (slice(None),)] = sv * so * x
                else:
                    out[...] = x.reshape(lead + self.shape)
        return out


def _combinations(n, k):
    '''(positions, index arrays...) of the k-subsets p < q of range(n).'''
    if k == 0:
        return (numpy.zeros(1, dtype=int),)
    if k == 1:
        return (numpy.arange(n), numpy.arange(n))
    p, q = numpy.triu_indices(n, k=1)
    return (numpy.arange(len(p)), p, q)


def _permutations(arrays):
    if len(arrays) < 2:
        return [(1, arrays)]
    return [(1, arrays), (-1, arrays[::-1])]


class ProductBasis:
    '''Blocks of the MRCI space of the model states civecs (columns).

    contraction: 'fic' (all external blocks contracted), 'wk' (only the
        doubly external ones, Werner-Knowles type), 'none', or a dict
        block -> 'pc' | 'full'
    internal: 'cas' or 'states'
    '''

    def __init__(self, eris, act, nelec, civecs, contraction='fic', internal='cas',
                 lindep=1e-10):
        sp = eris.spaces
        scheme = SCHEMES[contraction] if isinstance(contraction, str) else dict(contraction)
        self.blocks = {}
        civecs = numpy.asarray(civecs)
        for name in terms.ORDER:
            nh, np_ = terms.BLOCKS[name]
            n = act.target(nelec, terms.dn(name))
            if nh > sp.ncore or np_ > sp.nvirt or n < 0 or n > act.norb:
                continue
            dim = act.dim(n)
            mode = scheme.get(name, 'full')
            if name == 'cas' and internal == 'states':
                rows = civecs.T
            elif name == 'cas' and internal != 'cas':
                raise ValueError('internal must be cas or states')
            elif mode == 'pc':
                pclass = classes.CLASS_BY_NAME[name]
                rows = engine.class_functions(pclass, act, civecs, nelec).reshape(-1, dim)
            elif mode == 'full':
                rows = None
            else:
                raise ValueError('unknown contraction %r' % mode)
            # the basis diagonalizes H_act within the span: energies mu
            if rows is None:
                sec = act.sector(n)
                b, mu = sec.vecs, sec.lam
            else:
                b, mu = spectral.contracted_space(rows,
                                                  lambda q, n=n: act.apply_h(n, q),
                                                  lambda n=n: act.sector(n), lindep)
            if b.shape[1]:
                blk = Block(name, n, numpy.ascontiguousarray(b), sp.ncore, sp.nvirt)
                blk.mu = numpy.asarray(mu, dtype=float)
                self.blocks[name] = blk
        self.names = list(self.blocks)
        sizes = [self.blocks[n].size for n in self.names]
        self.offsets = numpy.concatenate([[0], numpy.cumsum(sizes)]).astype(int)
        self.dim = int(self.offsets[-1])

    def unpack(self, x):
        '''Vector(s) x (..., dim) -> {block: C[..., a, b, i, j, k]}'''
        return {n: self.blocks[n].unpack(x[..., self.offsets[k]:self.offsets[k + 1]])
                for k, n in enumerate(self.names)}

    def pack(self, c):
        return numpy.concatenate([self.blocks[n].pack(c[n]) for n in self.names], axis=-1)

    def diagonal(self, eps_core, eps_virt, e_core):
        '''Zeroth-order (Dyall) energies of the basis functions, packed:
        E_core + mu_k + sum eps_a - sum eps_i.'''
        out = []
        for n in self.names:
            blk = self.blocks[n]
            d = numpy.zeros(blk.shape)
            for axis in range(blk.np):
                d += eps_virt.reshape([-1 if x == axis else 1 for x in range(d.ndim)])
            for axis in range(blk.np, blk.np + blk.nh):
                d -= eps_core.reshape([-1 if x == axis else 1 for x in range(d.ndim)])
            out.append(blk.pack(d + blk.mu + e_core))
        return numpy.concatenate(out)
