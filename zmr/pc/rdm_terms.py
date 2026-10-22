'''
PC-NEVPT2 metric and zeroth-order matrices from RDMs up to rank three,
evaluated from the generated terms in zmr.pc.generated_terms
(zmr.wick.pc_nevpt2).  The 4-RDM contributions to K of the semi-internal
classes are evaluated through the intermediate states [H_act, y] |J> of
Chatterjee and Sokolov (intermediate_terms); the 4-RDM is never built.  The
explicit 4-RDM formulas are in zmr.reference.pc_rdm4 for checks.

    S[I, J, X, Y] = <I| O_X^+ O_Y |J>
    K[I, J, X, Y] = <I| O_X^+ H_act O_Y |J> = Kc + sum_L S[I, L] h_model[L, J]

The second form requires a model space invariant under H_act (true for CASCI
eigenstates and their unitary mixtures); check_invariance verifies it.
'''

import numpy

from socutils.zmr.pc.generated_terms import TERMS
from socutils.zmr.uc.classes import _all


def blocks_of(class_name, terms=TERMS):
    return sorted({bx for (c, bx, by) in terms if c == class_name})


def max_rank(terms=TERMS):
    return max(int(op[2:]) for d in terms.values() for kind in ('S', 'K')
               for _, _, ops in d[kind] for op in ops if op.startswith('dm'))


def intermediate_terms(class_name, items, act, civecs, nelec):
    '''sum_k sign_k <I| O_X^+ (O_Y without y_k) [H_act, y_k] |J> for the
    three-operator block of a semi-internal class, as an array
    (nstate, nstate, n^3, n^3) in the layouts of X and Y.

    The intermediate states g[J, a] = [H_act, y(a)] |Psi_J> = H_act y |J> -
    y H_act |J> are n vectors per state; the two remaining operators of O_Y
    are applied to them and the result is projected on the functions
    O_X |Psi_I>.  Cost and memory are those of the 3-RDM.'''
    from socutils.zmr.wick.pc_nevpt2 import OPERATORS
    ops, layout = OPERATORS[class_name][1]
    n = act.norb
    civecs = numpy.asarray(civecs)
    nstate = civecs.shape[1]
    out = numpy.zeros((nstate, nstate) + (n,) * 6, dtype=numpy.complex128)
    ntarget = nelec + sum(1 if d else -1 for _, d in ops)
    if not 0 <= ntarget <= n:
        return out.reshape(nstate, nstate, n**3, n**3)
    daggers = tuple(d for _, d in ops)
    column = [layout.index(name) for name, _ in ops]
    phi = act.apply_strings(civecs.T, nelec, [(daggers, _all(n, 3)[:, column])])
    hc = act.apply_h(nelec, civecs)
    for sign, k in items:
        name, dag = ops[k]
        nmid = nelec + (1 if dag else -1)
        if not 0 <= nmid <= n:
            continue
        single = [((dag,), _all(n, 1))]
        y = act.apply_strings(civecs.T, nelec, single)             # (J, a, dim)
        g = act.apply_h(nmid, y.reshape(nstate * n, -1).T).T.reshape(y.shape)
        g -= act.apply_strings(hc.T, nelec, single)
        rest = [o for i, o in enumerate(ops) if i != k]
        ket = act.apply_strings(g, nmid, [(tuple(d for _, d in rest), _all(n, 2))])
        t = numpy.einsum('ixd,jabd->ijxab', phi.conj(), ket, optimize=True)
        t = t.reshape((nstate, nstate) + (n,) * 6)
        # axes after X: (y_k, rest_1, rest_2) -> the layout of Y
        names = [name] + [r for r, _ in rest]
        order = [0, 1, 2, 3, 4] + [5 + names.index(x) for x in layout]
        out += sign * t.transpose(order)
    return out.reshape(nstate, nstate, n**3, n**3)


def _evaluate(terms, operands, shape):
    out = numpy.zeros(shape, dtype=numpy.complex128)
    for coef, subs, ops in terms:
        out += float(coef) * numpy.einsum(subs, *[operands[o] for o in ops], optimize=True)
    return out


def class_matrices(class_name, dms, heff, eri, subset=None, intermediate=None, terms=TERMS):
    '''(S, Kc) as arrays (nstate, nstate, nX, nX); X runs over the operator
    strings of zmr.wick.pc_nevpt2.OPERATORS with unrestricted indices, or
    over the positions subset among them (PerturberClass.subset).
    dms: {k: transition k-RDMs}.  intermediate(class_name, items) evaluates
    the intermediate-state terms of a block (intermediate_terms).'''
    n = heff.shape[0]
    nstate = dms[0].shape[0]
    operands = dict(dms=None, eye=numpy.eye(n), heff=heff, eri=eri)
    for k, d in dms.items():
        operands['dm%d' % k] = d
    blocks = blocks_of(class_name, terms)
    sizes = [n ** len(terms[(class_name, b, b)]['x']) for b in blocks]
    offsets = numpy.concatenate([[0], numpy.cumsum(sizes)])
    nx = offsets[-1]
    s = numpy.zeros((nstate, nstate, nx, nx), dtype=numpy.complex128)
    kc = numpy.zeros_like(s)
    for bx in blocks:
        for by in blocks:
            d = terms[(class_name, bx, by)]
            shape = (nstate, nstate) + (n,) * (len(d['x']) + len(d['y']))
            sx = slice(offsets[bx], offsets[bx + 1])
            sy = slice(offsets[by], offsets[by + 1])
            flat = (nstate, nstate, sizes[bx], sizes[by])
            s[:, :, sx, sy] = _evaluate(d['S'], operands, shape).reshape(flat)
            kc[:, :, sx, sy] = _evaluate(d['K'], operands, shape).reshape(flat)
            if d.get('I'):
                kc[:, :, sx, sy] += intermediate(class_name, d['I'])
    if subset is not None:
        s = numpy.ascontiguousarray(s[:, :, subset][:, :, :, subset])
        kc = numpy.ascontiguousarray(kc[:, :, subset][:, :, :, subset])
    return s, kc


def full_k(s, kc, hmodel_act):
    return kc + numpy.einsum('ilxy,lj->ijxy', s, hmodel_act)


def check_invariance(act, civecs, nelec, tol=1e-8):
    '''Largest residual of H_act C - C (C^+ H_act C).'''
    hc = act.apply_h(nelec, civecs)
    return abs(hc - civecs @ (civecs.conj().T @ hc)).max()


def contracted_space(smat, kmat, lindep):
    '''Orthonormal, H_act-diagonal combinations of the contracted functions
    from their metric and H_act matrices: columns of C (nfunc, m) and the
    energies mu.'''
    s, u = numpy.linalg.eigh(smat)
    keep = s > lindep
    x = u[:, keep] / numpy.sqrt(s[keep])
    mu, wv = numpy.linalg.eigh(x.conj().T @ kmat @ x)
    return x @ wv, mu
