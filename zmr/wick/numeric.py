'''
Numerical evaluation of evaluated Wick terms.

    value(terms, binding, dms)      all indices bound to orbital numbers
    tensor(terms, out_indices, ...) dummies summed with numpy.einsum

dms[k] holds transition RDMs dm_k[I, J, p1, q1, ..., pk, qk] (zmr.active.rdm
layout); an RDM factor <I| c_1^+ ... c_k^+ a_1 ... a_k |J> is
dm_k[I, J, c_1, a_k, c_2, a_(k-1), ..., c_k, a_1].
'''

import string

import numpy


def rdm_layout(rdm):
    '''Index tuple (p1, q1, ..., pk, qk) of an RDM factor (creators,
    annihilators).'''
    cre, ann = rdm
    k = len(cre)
    out = []
    for m in range(k):
        out += [cre[m], ann[k - 1 - m]]
    return tuple(out)


def value(terms, binding, dms, tensors=None):
    '''Value of a sum of evaluated terms with every index bound
    (binding: name -> orbital number, active orbitals counted within the
    active space).  Returns an (nstate, nstate) array.'''
    tensors = tensors or {}
    nstate = dms[0].shape[0]
    total = numpy.zeros((nstate, nstate), dtype=numpy.complex128)
    for t in terms:
        f = complex(t.coef)
        for a, b in t.deltas:
            if binding[a.name] != binding[b.name]:
                f = 0
                break
        if f == 0:
            continue
        for name, idx in t.tensors:
            f = f * tensors[name][tuple(binding[i.name] for i in idx)]
        lay = rdm_layout(t.rdm)
        total = total + f * dms[len(t.rdm[0])][(slice(None), slice(None))
                                                + tuple(binding[i.name] for i in lay)]
    return total


def tensor(terms, out_indices, dms, tensors, n):
    '''Array T[I, J, out...] = sum over dummies of the terms; all indices are
    active with dimension n.  out_indices: names of the free indices in
    output order.'''
    letters = iter(string.ascii_letters.replace('I', '').replace('J', ''))
    names = {}

    def letter(name):
        if name not in names:
            names[name] = next(letters)
        return names[name]

    out_sub = ''.join(letter(x) for x in out_indices)
    nstate = dms[0].shape[0]
    result = numpy.zeros((nstate, nstate) + (n,) * len(out_indices), dtype=numpy.complex128)
    eye = numpy.eye(n)
    for t in terms:
        subs, ops = [], []
        lay = rdm_layout(t.rdm)
        subs.append('IJ' + ''.join(letter(i.name) for i in lay))
        ops.append(dms[len(t.rdm[0])])
        for a, b in t.deltas:
            subs.append(letter(a.name) + letter(b.name))
            ops.append(eye)
        for name, idx in t.tensors:
            subs.append(''.join(letter(i.name) for i in idx))
            ops.append(tensors[name])
        # free indices that appear in no factor (cannot happen for valid
        # expressions) would be broadcast; check
        used = set(''.join(subs))
        missing = [c for c in out_sub if c not in used]
        if missing:
            raise ValueError('free index not present in term %r' % t)
        expr = ','.join(subs) + '->IJ' + out_sub
        result += complex(t.coef) * numpy.einsum(expr, *ops, optimize=True)
    return result
