'''
Derivation of the MRCI sigma equations in the product basis.

Basis functions (docs/derivations/mrci.md): for each of the nine blocks,

    |B; e, k> = E_e |Phi_c b_k>,     E_e = a^+ b^+ j i   (virtual creators,
                                                           core annihilators)

with the external strings below acting on the core-filled, virtual-empty
state whose active part is the vector b_k of the sector N + dn(B).  With
the coefficients C[e, k] antisymmetric in equivalent external indices and
summed over their full ranges,

    sigma^{B'}[e', k'] = sum_{B} 1 / (n_h! n_p!) sum_{e, k}
                         <Phi_c b'_k'| E_e'^+ H E_e |Phi_c b_k> C^B[e, k].

The core and virtual operators are contracted completely
(zmr.wick.algebra); what is left of H in the active space is a
normal-ordered string, whose matrix elements between the block bases,

    Gamma[k', t.., u.., k] = <b'_k'| t^+ .. u .. |b_k>,

take the place of the density matrices of the NEVPT2 derivation.  A term is
(coefficient, deltas, tensors, (creators, annihilators)); the tensors are
'h' (bare one-electron integrals), 'eri' ((pq|rs)) and 'C' (the ket
coefficients, virtual indices first).

The part of H with active indices only commutes with the external strings:
it is not expanded but kept as one operator, the terms of overlap(B', B)
with Gamma replaced by <b'_k'| H_aa |b_k>.
'''

import functools
import itertools
import math
from fractions import Fraction

from socutils.zmr.wick import algebra as w

# block -> (core holes, virtual particles)
BLOCKS = {'cas': (0, 0), 'caaa': (1, 0), 'ccaa': (2, 0), 'aaav': (0, 1), 'aavv': (0, 2),
          'cava': (1, 1), 'ccav': (2, 1), 'cavv': (1, 2), 'ccvv': (2, 2)}
ORDER = ('cas', 'caaa', 'ccaa', 'aaav', 'aavv', 'cava', 'ccav', 'cavv', 'ccvv')


def dn(block):
    '''Change of the number of active electrons.'''
    nh, np_ = BLOCKS[block]
    return nh - np_


def ext_indices(block, bra):
    '''(virtual indices, core indices) of the external string.'''
    nh, np_ = BLOCKS[block]
    v, c = ('AB', 'IJ') if bra else ('ab', 'ij')
    return ([w.Index(v[m], 'v') for m in range(np_)], [w.Index(c[m], 'c') for m in range(nh)])


def ext_string(virt, core):
    '''a^+ b^+ j i'''
    return (tuple(w.Op(x, True) for x in virt)
            + tuple(w.Op(x, False) for x in reversed(core)))


def hamiltonian():
    '''H without its all-active part, as terms with dummy indices p, q, r, s
    in every combination of spaces.'''
    out = []
    for s in itertools.product('cav', repeat=2):
        if set(s) == {'a'}:
            continue
        p, q = (w.Index(n, x) for n, x in zip('pq', s))
        out.append(w.Term(1, tensors=(('h', (p, q)),), ops=(w.Op(p, True), w.Op(q, False))))
    for s in itertools.product('cav', repeat=4):
        if set(s) == {'a'}:
            continue
        p, q, r, t = (w.Index(n, x) for n, x in zip('pqrs', s))
        out.append(w.Term(Fraction(1, 2), tensors=(('eri', (p, q, r, t)),),
                          ops=(w.Op(p, True), w.Op(r, True), w.Op(t, False), w.Op(q, False))))
    return out


def _evaluate(term, free, change):
    '''As algebra.evaluate for bra and ket active vectors whose electron
    numbers differ by change.'''
    sign, ext, act = w._split_spaces(term.ops)
    out = []
    for s1, d1 in w.external_expectation(ext):
        for s2, d2, cre, ann in w.normal_order(act):
            if len(cre) - len(ann) != change:
                continue
            t = term.copy(coef=term.coef * sign * s1 * s2, deltas=term.deltas + d1 + d2,
                          rdm=(cre, ann), ops=())
            t = w.substitute_deltas(t, free)
            if t is not None:
                out.append(t)
    return out


def _matrix_elements(bra, ket, operators):
    bv, bc = ext_indices(bra, True)
    kv, kc = ext_indices(ket, False)
    free = {x.name for x in bv + bc}
    left = w.adjoint_string(ext_string(bv, bc))
    right = ext_string(kv, kc)
    nh, np_ = BLOCKS[ket]
    sym = Fraction(1, math.factorial(nh) * math.factorial(np_))
    out = []
    for h in operators:
        t = w.Term(h.coef * sym, tensors=h.tensors + (('C', tuple(kv + kc)),),
                   ops=left + h.ops + right)
        out += _evaluate(t, free, dn(bra) - dn(ket))
    return w.collect(out, free)


def _swaps(block, bra):
    '''Permutations of equivalent external indices: [(sign, mapping)].'''
    virt, core = ext_indices(block, bra)
    out = [(1, {})]
    for pair in (virt, core):
        if len(pair) == 2:
            a, b = pair
            out = out + [(-s, {**m, a: b, b: a}) for s, m in out]
    return out


def _permute_c(term, perm):
    tensors = tuple((n, tuple(idx[k] for k in perm) if n == 'C' else idx)
                    for n, idx in term.tensors)
    return term.copy(tensors=tensors)


def reduce(terms, bra, ket):
    '''One representative of every set of terms that are equal after
    antisymmetrization: C is antisymmetric in equivalent ket indices, and
    the result is antisymmetrized in equivalent bra indices afterwards
    (sigma = A[sum of the representatives]).'''
    free = set(free_indices(bra))
    nh, np_ = BLOCKS[ket]
    cperms = [(1, tuple(range(np_ + nh)))]
    if np_ == 2:
        cperms += [(-s, (1, 0) + p[2:]) for s, p in cperms]
    if nh == 2:
        cperms += [(-s, p[:np_] + (p[np_ + 1], p[np_])) for s, p in cperms]
    acc, order = {}, []
    for t in terms:
        images = {}
        vanish = False
        for sb, mapping in _swaps(bra, True):
            tb = w._rename(t, mapping) if mapping else t
            for sc, perm in cperms:
                cand = w.canonicalize(_permute_c(tb, perm).copy(coef=tb.coef * sb * sc), free)
                if cand is None:
                    continue
                k = w._key(cand)
                if k in images and images[k].coef != cand.coef:
                    vanish = True          # equal to minus itself
                images[k] = cand
        if vanish or not images:
            continue
        k = min(images)
        if k in acc:
            acc[k] = acc[k].copy(coef=acc[k].coef + images[k].coef)
        else:
            acc[k] = images[k]
            order.append(k)
    return [acc[k] for k in sorted(order) if acc[k].coef != 0]


@functools.lru_cache(maxsize=None)
def reduced(bra, ket):
    '''(terms of H without its all-active part, overlap terms), reduced.'''
    over = overlap(bra, ket) if bra == ket else []
    return reduce(derive(bra, ket), bra, ket), reduce(over, bra, ket)


def derive(bra, ket):
    '''Terms of sigma^{bra} from the coefficients of block ket, without the
    all-active part of H.'''
    return _matrix_elements(bra, ket, hamiltonian())


def overlap(bra, ket):
    '''Terms of <bra; e', k'| ket; e, k> C[e, k] (non-zero only for bra =
    ket): the Gamma factor is the overlap of the block bases, or the matrix
    of the all-active part of H.'''
    return _matrix_elements(bra, ket, [w.Term(1)])


def free_indices(block):
    '''Names of the free (bra) indices in output order: virtual, then core.'''
    bv, bc = ext_indices(block, True)
    return [x.name for x in bv + bc]


def _format(term):
    cre, ann = term.rdm
    def show(i):
        # dummies carry their space: d0c, d1a, d2v
        return i.name + (i.space if i.name.startswith('d') else '')
    parts = ['%s[%s]' % (n, ','.join(show(i) for i in idx)) for n, idx in term.tensors]
    parts += ['delta[%s,%s]' % (a.name, b.name) for a, b in term.deltas]
    gamma = ' '.join([show(i) + '^+' for i in cre] + [show(i) for i in ann])
    return '%6s  %s  <%s>' % (term.coef, ' '.join(parts), gamma)


def generate(path):
    '''Write the terms of all block pairs to a Markdown file.'''
    lines = ['# MRCI sigma terms in the product basis', '',
             'Generated by `python -m socutils.zmr.wick.mrci`; do not edit.  Notation of',
             '`docs/derivations/mrci.md`: `sigma^{bra}[A,B,I,J; k\'] += coef * tensors *',
             '<b\'_k\'| active string |b_k>`, summed over the dummies `d0c, d1a, d2v, ...`',
             '(the last letter is the space), the ket coefficients `C` (virtual',
             'indices first) and `k`.  `<>` is the overlap of the block bases.  The',
             'all-active part of H is not listed: it is the overlap terms with',
             '`<b\'_k\'| H_aa |b_k>`.  Of terms that are equal after antisymmetrization',
             '(C is antisymmetric in equivalent indices) one representative is listed;',
             'the sum is to be antisymmetrized in the equivalent bra indices.', '']
    total = 0
    for bra in ORDER:
        for ket in ORDER:
            t = reduce(derive(bra, ket), bra, ket)
            if not t:
                continue
            total += len(t)
            lines += ['## %s <- %s (%d terms)' % (bra, ket, len(t)), '', '```']
            lines += [_format(x) for x in t] + ['```', '']
    lines += ['## Overlaps', '', '```']
    for b in ORDER:
        lines += ['%s:' % b] + [_format(x) for x in reduce(overlap(b, b), b, b)]
    lines += ['```', '', 'Total: %d terms.' % total, '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return total


if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, '..', 'docs', 'derivations', 'generated', 'mrci_sigma_terms.md')
    print('%d terms -> %s' % (generate(out), os.path.normpath(out)))
