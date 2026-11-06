'''Kramers-paired spinors for a time-reversal symmetric reference.

Time reversal acts on a two-component spinor with the components (a, b) in a
real scalar AO basis as

    K (a, b) = (-b*, a*),        K^2 = -1,

and commutes with the Fock operator of a closed-shell X2C / GHF calculation
without a magnetic field: the spinors come in degenerate pairs (p, Kp).  In a
basis of such pairs every tensor of the theory - integrals, amplitudes,
intermediates - obeys

    X[Kp, Kq, ...] = s_p s_q ... conj(X[p, q, ...]),

with s = +1 for the first member of a pair ("unbarred") and -1 for the
second ("barred", K maps it to minus the first): half of the elements
determine the other half.

A two-fold degenerate level is a Kramers pair in any orthonormal basis up
to a phase; levels that are degenerate for other reasons as well (spatial
symmetry) are not.  pair() rotates the spinors within each degenerate level
into exact pairs, partner = K(first), and orders the result as

    [unbarred, barred]      (the barred ones in the order of their partners),

so that K is the shift by half the dimension.  The Fock matrix stays
diagonal up to the deviation of the reference from time-reversal symmetry
(its convergence).  If a level is not closed under K (an open-shell or
symmetry-broken reference, a magnetic field), pair() returns None.
'''

import itertools

import numpy as np


def time_reverse(c):
    '''K c for spinor coefficients c[(spin, AO), ...] in a real AO basis.'''
    nao = c.shape[0] // 2
    return np.concatenate((-c[nao:].conj(), c[:nao].conj()))


def pair(c, s, e, tol=1e-7, sym_tol=1e-5):
    '''Kramers pairs from the spinors c[(spin, AO), n] (orthonormal in the
    metric s of the (spin, AO) basis) with orbital energies e (ascending).

    Returns (cp, order) or None.  cp[(spin, AO), n] are the paired spinors,
    [unbarred..., barred...] with cp[:, k + n/2] = K cp[:, k] exactly,
    orthonormal in s; they span the space of c and mix only spinors whose
    energies agree within ``tol`` (up to the deviation of the reference from
    time-reversal symmetry, which must be below ``sym_tol``).  ``order[k]``
    is the index of the level the new spinor k belongs to: e[order] are the
    energies.  None if a level has an odd dimension or is not closed under
    time reversal.'''
    n = c.shape[1]
    if n % 2:
        return None
    sc = s @ c
    first, order = [], []
    k = 0
    while k < n:
        m = k + 1
        while m < n and abs(e[m] - e[k]) < tol:
            m += 1
        if (m - k) % 2:
            return None
        mine = []                                   # the unbarred spinors of the level
        cand = [c[:, j].copy() for j in range(k, m)]
        while 2 * len(mine) < m - k:
            # the candidate with the largest part outside the pairs found so far
            for w in mine[-1:]:
                kw = time_reverse(w)
                sw, skw = s @ w, s @ kw
                cand = [x - w * (sw.conj() @ x) - kw * (skw.conj() @ x) for x in cand]
            norms = [np.sqrt((x.conj() @ s @ x).real) for x in cand]
            j = int(np.argmax(norms))
            if norms[j] < 1e-3:
                return None
            v = cand.pop(j) / norms[j]
            # K v must stay in the level
            inside = np.linalg.norm(sc[:, k:m].conj().T @ time_reverse(v))
            if abs(inside - 1.0) > sym_tol:
                return None
            mine.append(v)
        first += mine
        order += [k] * len(mine)
        k = m
    cu = np.array(first).T
    cp = np.hstack((cu, time_reverse(cu)))
    # symmetric orthonormalization: it keeps the pairs (the overlap of a
    # time-reversal closed set has the structure [[A, B], [-B*, A*]])
    ovlp = cp.conj().T @ s @ cp
    if abs(ovlp - np.eye(n)).max() > 10 * sym_tol:
        return None
    w, x = np.linalg.eigh(ovlp)
    cp = cp @ (x / np.sqrt(w)) @ x.conj().T
    h = n // 2
    cp[:, h:] = time_reverse(cp[:, :h])             # exact partners
    return cp, np.array(order + order)


def partner_signs(n):
    '''(partner index, sign) of the time reversal on the paired ordering
    [unbarred, barred] of n spinors: K p = s_p partner(p).'''
    h = n // 2
    return np.r_[np.arange(h, n), np.arange(h)], np.r_[np.ones(h), -np.ones(h)]


def symmetry_error(x):
    '''max |X[Kp, Kq, ...] - s_p s_q ... conj(X[p, q, ...])| for an array
    whose axes are spinor indices, each in the paired ordering [unbarred,
    barred] of its own space (occupied or virtual).'''
    y = x
    for ax, n in enumerate(x.shape):
        kp, sg = partner_signs(n)
        y = np.take(y, kp, axis=ax) * sg.reshape([-1 if a == ax else 1 for a in range(x.ndim)])
    return abs(y.conj() - x).max()


def pair_maps(n):
    '''(partner, sign) of the time reversal on the packed pairs p > q
    (numpy.tril_indices(n, -1) order) of n paired spinors:
    X[partner[P], ...] = sign[P] ... conj(X[P, ...]); sign[partner[P]] = sign[P].'''
    kp, sg = partner_signs(n)
    i, j = np.tril_indices(n, -1)
    ki, kj = kp[i], kp[j]
    hi, lo = np.maximum(ki, kj), np.minimum(ki, kj)
    return hi * (hi - 1) // 2 + lo, sg[i] * sg[j] * np.where(ki > kj, 1.0, -1.0)


def _axis_maps(x, pairs):
    '''(partner, sign) for every axis of x in the form "destination from
    source": y[p'] = sign[p'] conj(x[partner[p']]); ``pairs``: the axes that
    are packed pairs p > q of spinors (the others are single spinors).'''
    maps = []
    for ax, n in enumerate(x.shape):
        if ax in pairs:
            m = int(round((1 + np.sqrt(1 + 8 * n)) / 2))
            maps.append(pair_maps(m))
        else:
            kp, sg = partner_signs(n)
            maps.append((kp, -sg))
    return maps


def image(x, pairs=()):
    '''The time-reversed array y[Kp, Kq, ...] = s_p s_q ... conj(x[p, q, ...])
    (an even number of spinor indices; packed pairs count as two).  A
    symmetric x is its own image.'''
    y = x.conj()
    for ax, (kp, sg) in enumerate(_axis_maps(x, pairs)):
        y = np.take(y, kp, axis=ax) * sg.reshape([-1 if a == ax else 1 for a in range(x.ndim)])
    return y


def fill_barred(x, pairs=()):
    '''x[n/2:] along the first axis (a single spinor index: the barred
    half) from x[:n/2] by time reversal, in place.  ``pairs``: the axes that
    are packed pairs of spinors.

    On a single spinor index K is a swap of the two halves with a sign
    (-1 into the unbarred half), so the image is written half by half along
    these axes, one pass over the array and no index arrays.'''
    h = x.shape[0] // 2
    src, dst = x[:h], x[h:]
    single = [ax for ax in range(1, x.ndim) if ax not in pairs]
    pmaps = [(ax, _axis_maps(x, pairs)[ax]) for ax in sorted(pairs)]
    for halves in itertools.product((0, 1), repeat=len(single)):
        into, out_of, sign = [slice(None)] * x.ndim, [slice(None)] * x.ndim, 1.0
        for ax, upper in zip(single, halves):
            m = x.shape[ax] // 2
            into[ax] = slice(m, None) if upper else slice(0, m)
            out_of[ax] = slice(0, m) if upper else slice(m, None)
            sign = sign if upper else -sign
        y, block = src[tuple(out_of)], dst[tuple(into)]
        if not pmaps:
            np.conjugate(y, out=block)
            if sign < 0:
                np.negative(block, out=block)
            continue
        for ax, (kp, sg) in pmaps:
            y = np.take(y, kp, axis=ax)
            sign = sign * sg.reshape([-1 if a == ax else 1 for a in range(x.ndim)])
        np.conjugate(y, out=y)
        np.multiply(y, sign, out=block)
    return x


def symmetric(t1, t2, tol=1e-6):
    '''Do the amplitudes t1[i,a] and t2 (t2[i,j,a,b], or packed
    [(i>j), (a>b)]) of paired spinors obey the time-reversal symmetry
    t[Kp Kq, Kr Ks] = +-conj(t[pq, rs])?  All of t1 and a few rows (i>j) of
    t2 are compared with their images, relative to the largest of them.'''
    o, v = t1.shape
    if o % 2 or v % 2 or o < 2 or v < 2:
        return False
    kpo, sgo = pair_maps(o)
    kpv, sgv = pair_maps(v)
    npo = kpo.size
    if t2.ndim == 2:
        row = lambda p: np.asarray(t2[p])                           # noqa: E731
    else:
        io, jo = np.tril_indices(o, -1)
        av, bv = np.tril_indices(v, -1)
        row = lambda p: np.asarray(t2[io[p], jo[p]])[av, bv]        # noqa: E731
    rows = [(row(p), row(kpo[p]), sgo[p])
            for p in sorted({0, npo // 3, npo // 2, (2 * npo) // 3, npo - 1})]
    scale = max([abs(t1).max(), 1e-300] + [abs(r).max() for r, _, _ in rows])
    if abs(image(t1) - t1).max() > tol * scale:
        return False
    return all(abs(rk - sg * sgv * r[kpv].conj()).max() <= tol * scale for r, rk, sg in rows)
