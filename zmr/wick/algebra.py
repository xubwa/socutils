'''
Second-quantized algebra for spin-orbital / spinor operator strings and
their matrix elements between multi-determinant states.

Objects
    Index(name, space)          space 'c' (core), 'a' (active), 'v' (virtual)
    Op(index, dagger)
    Term(coef, deltas, tensors, rdm, ops)
        coef      Fraction
        deltas    tuple of (Index, Index), Kronecker deltas between free indices
        tensors   tuple of (name, tuple of Index), e.g. ('heff', (p, q))
        rdm       None, or (creators, annihilators) of an active expectation
                  value <I| c_1^+ ... c_k^+ a_1 ... a_k |J>
        ops       remaining operator string (evaluated terms have none)

Matrix elements <Phi_c Psi_I| ops |Phi_c Psi_J> (core filled, virtuals empty,
active part Psi) factorize exactly into a core/virtual vacuum expectation
value and an active matrix element (docs/derivations/wick.md): every core
operator comes in a pair, so the active phases cancel.  The core/virtual part
is evaluated by full contraction (core: <i^+ j> = delta_ij, virtual:
<a b^+> = delta_ab), the active string is normal-ordered (creators to the
left) and becomes an RDM element.

Indices whose name is in the set of free indices of an expression are kept;
all others are summed (dummies).  Deltas involving a dummy are eliminated by
substitution.
'''

import itertools
from fractions import Fraction


class Index(tuple):
    '''Orbital index: (name, space).'''

    def __new__(cls, name, space):
        if space not in ('c', 'a', 'v'):
            raise ValueError('unknown space %r' % space)
        return tuple.__new__(cls, (name, space))

    name = property(lambda self: self[0])
    space = property(lambda self: self[1])

    def __repr__(self):
        return self.name


class Op(tuple):
    '''Creation (dagger=True) or annihilation operator.'''

    def __new__(cls, index, dagger):
        return tuple.__new__(cls, (index, bool(dagger)))

    index = property(lambda self: self[0])
    dagger = property(lambda self: self[1])

    def adjoint(self):
        return Op(self.index, not self.dagger)

    def __repr__(self):
        return self.index.name + ('^+' if self.dagger else '')


def adjoint_string(ops):
    '''(o_1 ... o_m)^+ = o_m^+ ... o_1^+'''
    return tuple(o.adjoint() for o in reversed(ops))


class Term:
    __slots__ = ('coef', 'deltas', 'tensors', 'rdm', 'ops')

    def __init__(self, coef=1, deltas=(), tensors=(), rdm=None, ops=()):
        self.coef = Fraction(coef)
        self.deltas = tuple(deltas)
        self.tensors = tuple(tensors)
        self.rdm = rdm
        self.ops = tuple(ops)

    def copy(self, **kw):
        new = Term(self.coef, self.deltas, self.tensors, self.rdm, self.ops)
        for k, v in kw.items():
            setattr(new, k, v)
        return new

    def __repr__(self):
        parts = ['%s' % self.coef]
        parts += ['d(%s,%s)' % (a.name, b.name) for a, b in self.deltas]
        parts += ['%s[%s]' % (n, ','.join(i.name for i in idx)) for n, idx in self.tensors]
        if self.rdm is not None:
            c, a = self.rdm
            parts.append('<%s>' % ' '.join([x.name + '+' for x in c] + [x.name for x in a]))
        if self.ops:
            parts.append(' '.join(map(repr, self.ops)))
        return ' '.join(parts)


# ----------------------------------------------------------------------
# products and commutators

def product(*terms):
    '''Product of terms (operator strings concatenated left to right).'''
    out = Term()
    for t in terms:
        if t.rdm is not None:
            raise ValueError('cannot multiply evaluated terms')
        out = out.copy(coef=out.coef * t.coef, deltas=out.deltas + t.deltas,
                       tensors=out.tensors + t.tensors, ops=out.ops + t.ops)
    return out


def _delta(a, b):
    '''Kronecker delta between two indices: None if identically zero
    (different spaces), () if identically one, else ((a, b),).'''
    if a.space != b.space:
        return None
    if a == b:
        return ()
    return ((a, b),)


def commutator_with_op(h_ops, o):
    '''[E, o] for an even string E = h_ops (one- or two-body, p^+ q or
    p^+ r^+ s q) and a single operator o.  Returns a list of
    (sign, delta, ops).'''
    if len(h_ops) == 2:
        p, q = h_ops
        if o.dagger:          # [p^+ q, t^+] = delta_qt p^+
            pairs = [(+1, q.index, (p,))]
        else:                 # [p^+ q, t] = -delta_pt q
            pairs = [(-1, p.index, (q,))]
    elif len(h_ops) == 4:
        p, r, s, q = h_ops
        if o.dagger:          # [p^+ r^+ s q, t^+] = d_qt p^+ r^+ s - d_st p^+ r^+ q
            pairs = [(+1, q.index, (p, r, s)), (-1, s.index, (p, r, q))]
        else:                 # [p^+ r^+ s q, t] = -d_pt r^+ s q + d_rt p^+ s q
            pairs = [(-1, p.index, (r, s, q)), (+1, r.index, (p, s, q))]
    else:
        raise ValueError('commutator_with_op needs a one- or two-body string')
    out = []
    for sign, idx, ops in pairs:
        d = _delta(idx, o.index)
        if d is not None:
            out.append((sign, d, ops))
    return out


def commutator(h_term, o_ops):
    '''[h_term, o_1 ... o_m] = sum_i o_1 ... [h, o_i] ... o_m for an even
    operator h_term (coefficient, tensors and a one- or two-body string).'''
    out = []
    for i, o in enumerate(o_ops):
        for sign, d, ops in commutator_with_op(h_term.ops, o):
            out.append(h_term.copy(coef=h_term.coef * sign, deltas=h_term.deltas + d,
                                   ops=tuple(o_ops[:i]) + ops + tuple(o_ops[i + 1:])))
    return out


# ----------------------------------------------------------------------
# evaluation

def _split_spaces(ops):
    '''Stable partition into (external ops, active ops) with the sign of the
    permutation.'''
    ext, act = [], []
    sign = 1
    for o in ops:
        if o.index.space == 'a':
            act.append(o)
        else:
            # moving o to the left past all active operators already seen
            if len(act) % 2:
                sign = -sign
            ext.append(o)
    return sign, ext, act


def _full_contractions(ops, space):
    '''Vacuum expectation value of a string of core ('c', filled vacuum) or
    virtual ('v', empty vacuum) operators: list of (sign, deltas).'''
    if not ops:
        return [(1, ())]
    if len(ops) % 2:
        return []
    first = ops[0]
    out = []
    for j in range(1, len(ops)):
        other = ops[j]
        if space == 'c':
            ok = first.dagger and not other.dagger       # <i^+ j> = delta_ij
        else:
            ok = (not first.dagger) and other.dagger     # <a b^+> = delta_ab
        if not ok:
            continue
        d = _delta(first.index, other.index)
        if d is None:
            continue
        rest = ops[1:j] + ops[j + 1:]
        sign = -1 if (j - 1) % 2 else 1
        for s2, d2 in _full_contractions(rest, space):
            out.append((sign * s2, d + d2))
    return out


def external_expectation(ext_ops):
    '''<Phi_c| ext_ops |Phi_c> for a string of core and virtual operators.'''
    core, virt = [], []
    sign = 1
    for o in ext_ops:
        if o.index.space == 'c':
            if len(virt) % 2:
                sign = -sign
            core.append(o)
        else:
            virt.append(o)
    out = []
    for s1, d1 in _full_contractions(tuple(core), 'c'):
        for s2, d2 in _full_contractions(tuple(virt), 'v'):
            out.append((sign * s1 * s2, d1 + d2))
    return out


def normal_order(ops):
    '''Normal-order an active string (all creators to the left), keeping the
    relative order of the creators and of the annihilators.
    Returns a list of (sign, deltas, creators, annihilators).'''
    ops = tuple(ops)
    for i in range(len(ops) - 1):
        a, b = ops[i], ops[i + 1]
        if (not a.dagger) and b.dagger:
            out = []
            d = _delta(a.index, b.index)
            if d is not None:      # a b^+ = delta - b^+ a
                for s, dd, c, n in normal_order(ops[:i] + ops[i + 2:]):
                    out.append((s, d + dd, c, n))
            for s, dd, c, n in normal_order(ops[:i] + (b, a) + ops[i + 2:]):
                out.append((-s, dd, c, n))
            return out
    creators = tuple(o.index for o in ops if o.dagger)
    annihilators = tuple(o.index for o in ops if not o.dagger)
    return [(1, (), creators, annihilators)]


def evaluate(term, free):
    '''<Phi_c Psi_I| term |Phi_c Psi_J> as a list of evaluated terms (no
    operators left; one RDM factor each).  free: names of free indices.'''
    sign, ext, act = _split_spaces(term.ops)
    out = []
    for s1, d1 in external_expectation(ext):
        for s2, d2, cre, ann in normal_order(act):
            if len(cre) != len(ann):      # bra and ket have the same N
                continue
            t = term.copy(coef=term.coef * sign * s1 * s2, deltas=term.deltas + d1 + d2,
                          rdm=(cre, ann), ops=())
            t = substitute_deltas(t, free)
            if t is not None:
                out.append(t)
    return out


# ----------------------------------------------------------------------
# simplification

def _rename(term, mapping):
    def r(i):
        return mapping.get(i, i)
    deltas = tuple((r(a), r(b)) for a, b in term.deltas)
    tensors = tuple((n, tuple(r(i) for i in idx)) for n, idx in term.tensors)
    rdm = None
    if term.rdm is not None:
        rdm = (tuple(r(i) for i in term.rdm[0]), tuple(r(i) for i in term.rdm[1]))
    return term.copy(deltas=deltas, tensors=tensors, rdm=rdm)


def substitute_deltas(term, free):
    '''Eliminate deltas that involve a dummy index; drop identically zero
    terms.  Returns the new term or None.'''
    deltas = list(term.deltas)
    kept = []
    while deltas:
        a, b = deltas.pop()
        if a.space != b.space:
            return None
        if a == b:
            continue
        if a.name not in free or b.name not in free:
            dummy, other = (a, b) if a.name not in free else (b, a)
            term = _rename(term.copy(deltas=tuple(deltas) + tuple(kept)), {dummy: other})
            deltas = list(term.deltas)
            kept = []
            continue
        kept.append(tuple(sorted((a, b))))
    term = term.copy(deltas=tuple(sorted(set(kept))))
    if len(set(kept)) != len(kept):
        # delta_ab^2 = delta_ab
        pass
    return term


def _perm_sign(seq):
    '''Sign of the permutation that sorts seq (seq has distinct items).'''
    seq = list(seq)
    sign = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign


def canonical_rdm(term):
    '''Sort creators and annihilators of the RDM factor (antisymmetry);
    returns None if the element vanishes by the Pauli principle.'''
    if term.rdm is None:
        return term
    cre, ann = term.rdm
    if len(set(cre)) != len(cre) or len(set(ann)) != len(ann):
        return None
    sign = _perm_sign(cre) * _perm_sign(ann)
    return term.copy(coef=term.coef * sign, rdm=(tuple(sorted(cre)), tuple(sorted(ann))))


def _canonical_tensor(name, idx):
    if name == 'eri':           # (pq|rs) = (rs|pq)
        p, q, r, s = idx
        return name, min((p, q, r, s), (r, s, p, q))
    return name, tuple(idx)


def _key(term):
    tensors = tuple(sorted(_canonical_tensor(n, i) for n, i in term.tensors))
    return (term.deltas, tensors, term.rdm)


def canonicalize(term, free):
    '''Canonical form: sorted RDM indices and tensors, dummies renamed to the
    lexicographically smallest assignment.'''
    term = canonical_rdm(term)
    if term is None:
        return None
    names = set()
    for n, idx in term.tensors:
        names.update(idx)
    if term.rdm is not None:
        names.update(term.rdm[0])
        names.update(term.rdm[1])
    dummies = sorted(i for i in names if i.name not in free)
    if not dummies:
        return term.copy(tensors=tuple(sorted(_canonical_tensor(n, i) for n, i in term.tensors)))
    best = None
    targets = [Index('d%d' % k, d.space) for k, d in enumerate(dummies)]
    for perm in itertools.permutations(targets):
        if any(p.space != d.space for p, d in zip(perm, dummies)):
            continue
        cand = canonical_rdm(_rename(term, dict(zip(dummies, perm))))
        cand = cand.copy(tensors=tuple(sorted(_canonical_tensor(n, i) for n, i in cand.tensors)))
        k = _key(cand)
        if best is None or k < best[0]:
            best = (k, cand)
    return best[1]


def collect(terms, free):
    '''Canonicalize and sum terms with identical structure.'''
    acc = {}
    order = []
    for t in terms:
        t = canonicalize(t, free)
        if t is None or t.coef == 0:
            continue
        k = _key(t)
        if k in acc:
            acc[k] = acc[k].copy(coef=acc[k].coef + t.coef)
        else:
            acc[k] = t
            order.append(k)
    return [acc[k] for k in sorted(order) if acc[k].coef != 0]
