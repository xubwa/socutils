'''
Derivation of the PC-NEVPT2 metric and zeroth-order matrices in RDM form,
and generation of the code and documentation from the same terms.

For each perturber class the contracted functions are E_e O_X |Phi_c Psi>
with the active operator blocks OPERATORS below, here with unrestricted
indices; zmr.uc.classes uses the non-redundant strings among them
(PerturberClass.subset, checked numerically by the test suite).  For a pair
of blocks (bx, by):

    S^{IJ}[X, Y]  = <I| O_X^+ O_Y |J>
    Kc^{IJ}[X, Y] = <I| O_X^+ [H_act, O_Y] |J>,
        H_act = sum heff_pq p^+ q + 1/2 sum (pq|rs) p^+ r^+ s q

and, for a model space invariant under H_act,

    K^{IJ}[X, Y] = <I| O_X^+ H_act O_Y |J> = Kc^{IJ} + sum_L S^{IL} h^{LJ},
    h^{LJ} = <L| H_act |J>.

The commutator keeps the RDM rank at (|O_X| + |O_Y| + 2) / 2, i.e. at most
four, and four only for two three-operator strings (the semi-internal
blocks of caaa and aaav).  There the 4-RDM is avoided with intermediate
states (Chatterjee and Sokolov, J. Chem. Theory Comput. 16, 6343 (2020),
appendix): with O_Y = y_1 y_2 y_3,

    [H, O_Y] = sum_k (-1)^(3-k) (O_Y without y_k) [H, y_k]
               + sum_k y_1 .. y_(k-1) [[H, y_k], y_(k+1) .. y_3}

where [ , } is the commutator for an even and the anticommutator for an odd
number of operators on the right.  The first sum acts on |J> through the
intermediate states [H, y_k] |J> (one vector per active orbital and state),
and its matrix elements <I| O_X^+ (two operators) |intermediate> are
transition quantities of 3-RDM cost; the second sum has at most three
operators and gives 3-RDM terms.  The terms record the first sum as the
list 'I' of (sign, k).

Run `python -m socutils.zmr.wick.pc_nevpt2` to regenerate zmr/pc/generated_terms.py,
docs/derivations/generated/pc_nevpt2_terms.md and the explicit 4-RDM form
zmr/reference/pc_rdm4_terms.py, which is kept for checks only.
'''

import os
import string
from fractions import Fraction

from socutils.zmr.wick import algebra as w
from socutils.zmr.wick.numeric import rdm_layout

# class -> list of operator blocks (operator string, index layout of X)
OPERATORS = {
    'ccvv': [((), ())],
    'ccav': [((('t', True),), ('t',))],
    'cavv': [((('t', False),), ('t',))],
    'ccaa': [((('t', True), ('u', True)), ('t', 'u'))],
    'aavv': [((('u', False), ('t', False)), ('t', 'u'))],
    'cava': [((), ()), ((('t', True), ('u', False)), ('t', 'u'))],
    'caaa': [((('t', True),), ('t',)),
             ((('t', True), ('u', True), ('v', False)), ('t', 'u', 'v'))],
    'aaav': [((('t', False),), ('t',)),
             ((('u', True), ('v', False), ('t', False)), ('t', 'u', 'v'))],
}
CLASS_ORDER = ('ccvv', 'ccav', 'cavv', 'ccaa', 'aavv', 'cava', 'caaa', 'aaav')


def _act(name):
    return w.Index(name, 'a')


def block_ops(block, prefix):
    ops, layout = block
    names = {n: prefix + n for n in layout}
    return (tuple(w.Op(_act(names[n]), d) for n, d in ops),
            tuple(names[n] for n in layout))


def hamiltonian_terms():
    p, q, r, s = (_act(x) for x in 'pqrs')
    one = w.Term(tensors=(('heff', (p, q)),), ops=(w.Op(p, 1), w.Op(q, 0)))
    two = w.Term(coef=Fraction(1, 2), tensors=(('eri', (p, q, r, s)),),
                 ops=(w.Op(p, 1), w.Op(r, 1), w.Op(s, 0), w.Op(q, 0)))
    return [one, two]


def derive(class_name, explicit=False):
    '''{(bx, by): {'S': terms, 'K': terms, 'I': [(sign, k)], 'xlay': ..., 'ylay': ...}}

    explicit=True: every term of K as an RDM term, up to the 4-RDM.'''
    blocks = OPERATORS[class_name]
    out = {}
    for bx, xb in enumerate(blocks):
        xops, xlay = block_ops(xb, 'x')
        bra = w.Term(ops=w.adjoint_string(xops))
        for by, yb in enumerate(blocks):
            yops, ylay = block_ops(yb, 'y')
            free = set(xlay) | set(ylay)
            s_terms = w.collect(w.evaluate(w.product(bra, w.Term(ops=yops)), free), free)
            k_terms, inter = [], []
            if explicit or len(xops) + len(yops) < 6:
                for h in hamiltonian_terms():
                    for c in w.commutator(h, yops):
                        k_terms += w.evaluate(w.product(bra, c), free)
            else:
                for k, y in enumerate(yops):
                    prefix, suffix = yops[:k], yops[k + 1:]
                    sign = (-1) ** len(suffix)
                    inter.append((sign, k))
                    for h in hamiltonian_terms():
                        for g in w.commutator(h, (y,)):
                            # prefix (g suffix - sign suffix g)
                            ahead = w.product(bra, w.Term(ops=prefix), g, w.Term(ops=suffix))
                            behind = w.product(bra, w.Term(coef=-sign, ops=prefix + suffix), g)
                            k_terms += w.evaluate(ahead, free) + w.evaluate(behind, free)
            k_terms = w.collect(k_terms, free)
            if not explicit and any(len(t.rdm[0]) > 3 for t in k_terms):
                raise RuntimeError('4-RDM terms left in %s (%d, %d)' % (class_name, bx, by))
            out[(bx, by)] = {'S': s_terms, 'K': k_terms, 'I': inter, 'xlay': xlay, 'ylay': ylay}
    return out


# ----------------------------------------------------------------------
# code and documentation generation

def _einsum_spec(term, out_names):
    '''(subscripts, operand names) for numpy.einsum; 'IJ' are the state
    indices of the RDM factor.'''
    letters = iter(c for c in string.ascii_lowercase)
    names = {}

    def letter(n):
        if n not in names:
            names[n] = next(letters)
        return names[n]

    for n in out_names:
        letter(n)
    subs, ops = [], []
    lay = rdm_layout(term.rdm)
    subs.append('IJ' + ''.join(letter(i.name) for i in lay))
    ops.append('dm%d' % len(term.rdm[0]))
    for a, b in term.deltas:
        subs.append(letter(a.name) + letter(b.name))
        ops.append('eye')
    for name, idx in term.tensors:
        subs.append(''.join(letter(i.name) for i in idx))
        ops.append(name)
    return ','.join(subs) + '->IJ' + ''.join(names[n] for n in out_names), tuple(ops)


def _pretty(term):
    coef = term.coef
    sign = '-' if coef < 0 else '+'
    mag = abs(coef)
    parts = [] if mag == 1 else [str(mag)]
    for a, b in term.deltas:
        parts.append('d(%s,%s)' % (a.name, b.name))
    for name, idx in term.tensors:
        parts.append('%s[%s]' % (name, ','.join(i.name for i in idx)))
    cre, ann = term.rdm
    ops = ' '.join([c.name + '+' for c in cre] + [a.name for a in ann])
    parts.append('<I|%s|J>' % ops if ops else '<I|J>')
    return '%s %s' % (sign, ' '.join(parts))


def generate(explicit=False):
    '''Python source of zmr/pc/generated_terms.py and the markdown document
    (explicit=True: of zmr/reference/pc_rdm4_terms.py, with the 4-RDM).'''
    py = ['# Generated by `python -m socutils.zmr.wick.pc_nevpt2` -- do not edit.',
          '"""PC-NEVPT2 metric (S) and commutator (K) terms in RDM form.',
          '',
          'TERMS[(class, block_x, block_y)] = {"S": [...], "K": [...], "x": layout, "y": layout};',
          'each term is (coefficient, einsum subscripts, operand names) with operands',
          'dm<k> (transition k-RDMs, zmr.active.rdm layout, leading state indices IJ),',
          'eye, heff and eri (active-space integrals).  Output indices: IJ, then the',
          'X layout, then the Y layout."""',
          '',
          'from fractions import Fraction',
          '',
          'TERMS = {']
    md = ['# PC-NEVPT2 metric and commutator matrices (generated)',
          '',
          'Generated by `python -m socutils.zmr.wick.pc_nevpt2` from the same terms as',
          '`zmr/pc/generated_terms.py`; do not edit.  Notation: `<I|...|J>` is an',
          'active-space transition matrix element, `d(a,b)` a Kronecker delta,',
          '`heff[p,q]` and `eri[p,q,r,s] = (pq|rs)` the active-space integrals; repeated',
          'non-free indices (`d0`, `d1`, ...) are summed.  For blocks `bx`, `by` of a class,',
          '',
          '    S[X, Y] = <I| O_X^+ O_Y |J>',
          '    K[X, Y] = <I| O_X^+ [H_act, O_Y] |J>     (full K = K + S h_model)',
          '']
    for name in CLASS_ORDER:
        derived = derive(name, explicit)
        md.append('## %s' % name)
        md.append('')
        md.append('Operator blocks: ' + '; '.join(
            '%d: `%s`' % (b, ' '.join(n + ('+' if d else '') for n, d in ops) or '1')
            for b, (ops, _) in enumerate(OPERATORS[name])))
        md.append('')
        for (bx, by), d in sorted(derived.items()):
            out_names = d['xlay'] + d['ylay']
            py.append('    (%r, %d, %d): {' % (name, bx, by))
            py.append('        "x": %r, "y": %r,' % (d['xlay'], d['ylay']))
            for kind in ('S', 'K'):
                py.append('        %r: [' % kind)
                for t in d[kind]:
                    subs, ops = _einsum_spec(t, out_names)
                    py.append('            (Fraction(%d, %d), %r, %r),'
                              % (t.coef.numerator, t.coef.denominator, subs, ops))
                py.append('        ],')
            py.append('        "I": %r,' % (d['I'],))
            py.append('    },')
            if d['I']:
                md.append('**intermediate states, blocks (%d, %d)**: K gets in addition '
                          '`sum_k sign_k <I| O_X^+ (O_Y without y_k) [H, y_k] |J>`, '
                          '(sign, k) = %s' % (bx, by, ', '.join(map(str, d['I']))))
                md.append('')
            for kind in ('S', 'K'):
                md.append('**%s, blocks (%d, %d)**, X = (%s), Y = (%s):' % (
                    kind, bx, by, ', '.join(d['xlay']), ', '.join(d['ylay'])))
                md.append('')
                md.append('```')
                md += [_pretty(t) for t in d[kind]] or ['0']
                md.append('```')
                md.append('')
    py.append('}')
    py.append('')
    return '\n'.join(py), '\n'.join(md) + '\n'


def paths():
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # socutils/zmr
    return (os.path.join(pkg, 'pc', 'generated_terms.py'),
            os.path.join(pkg, 'docs', 'derivations', 'generated', 'pc_nevpt2_terms.md'),
            os.path.join(pkg, 'reference', 'pc_rdm4_terms.py'))


if __name__ == '__main__':
    py_src, md_src = generate()
    py_path, md_path, ref_path = paths()
    for path, text in ((py_path, py_src), (md_path, md_src), (ref_path, generate(True)[0])):
        with open(path, 'w') as f:
            f.write(text)
        print('wrote', path)
