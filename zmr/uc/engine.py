'''
Class-independent contraction of the second-order effective Hamiltonian.

For one perturber class the active functions O_X |Psi_J> (CI vectors phi of
the target sector), an orthonormal basis b_k of the first-order space with
H_act-energies lam_k (the sector eigenstates for UC, the contracted space
for PC) and right-hand-side coefficients v[e, X] give

    a_IJ = <I|V R(E0_J) V|J>
         = - sum_{e,k} conj(u^I_ek) u^J_ek / (lam_k + delta_e - s_J),
    u^J_ek = <b_k| sum_X v[e, X] O_X |Psi_J>,   s_J = E0_J - E_core.

u is formed in whichever order is cheaper:

    'T'  project the functions first, T^J[k, X] = <b_k|O_X|Psi_J>, then
         contract with v            (few operators, many external labels)
    'R'  contract with the integrals first, r^J_e = sum_X v[e, X] O_X |Psi_J>,
         then project               (many operators, few external labels:
                                     caaa, aaav)

and the sum over (e, k) is done by OpenMP kernels: weighted_gram for 'R';
for 'T' either contract_t (u is formed for one k at a time and never stored)
or contract_w (the external labels are summed first into W_k[X, Y], whose
cost does not grow with the number of states), see gram_route.  Real input
(the spin-restricted code) stays real throughout.  The Laplace path lives in
zmr.uc.laplace_engine.
'''

import os

import numpy

from socutils.zmr import output
from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la


def class_functions(pclass, act, civecs, nelec):
    '''phi[J, X, :] = O_X |Psi_J> in the determinant basis of the target
    sector, or None if that sector is empty.'''
    if act.dim(act.target(nelec, pclass.dn)) == 0:
        return None
    return numpy.ascontiguousarray(pclass.apply(act, numpy.asarray(civecs).T, nelec))


def project_functions(basis, phi):
    '''T[J, k, X] = <b_k|O_X|Psi_J> for functions phi[J, X, :] and basis
    vectors given as the columns of basis (dim, nk).'''
    nstate, nx, _ = phi.shape
    t = numpy.empty((nstate, basis.shape[1], nx), dtype=numpy.result_type(basis, phi))
    for j in range(nstate):
        t[j] = la.gemm(basis, phi[j], 'C', 'T')
    return t


def transition_amplitudes(pclass, act, civecs, nelec):
    '''T[J, k, X] = <k|O_X|Psi_J> over the eigenstates k of the target sector.
    Returns (T, sector) or (None, sector) if the target sector is empty.'''
    sector = act.sector(act.target(nelec, pclass.dn))
    phi = class_functions(pclass, act, civecs, nelec)
    if phi is None:
        return None, sector
    return project_functions(sector.vecs, phi), sector


def class_blocks(pclass, ctx):
    '''External blocks (delta, v) of a class: a list, or an ExternalBlocks
    that can also deliver the external sums directly (factorized integrals).'''
    if hasattr(pclass, 'external'):
        ext = pclass.external(ctx)
        if ext is not None:
            return ext
    return [(numpy.asarray(d), numpy.asarray(v)) for d, v in pclass.blocks(ctx)]


def report_denominator(blocks, lam, shifts, col=None):
    '''Tell the output module the smallest denominator of this contraction,
    min(delta) + min(lam) - max(shift).'''
    if len(lam) == 0:
        return
    if hasattr(blocks, 'delta_min'):
        dmin = blocks.delta_min
    else:
        mins = [d.min() for d, _ in blocks if len(d)]
        if not mins:
            return
        dmin = min(mins)
    shifts = numpy.asarray(shifts, dtype=float)
    top = shifts[col] if col is not None else shifts.max()
    output.denominator(dmin + numpy.min(lam) - top)


def nlabel(blocks):
    if hasattr(blocks, 'nlabel'):
        return blocks.nlabel
    return sum(len(d) for d, _ in blocks)


def direct_contract_t(blocks, t, lam, shifts):
    '''a[I, J] from external sums W at the interpolation nodes that the
    blocks deliver themselves (ExternalBlocks.w_nodes); None if that is not
    possible (no kernel, intruder, too many nodes).'''
    if getattr(blocks, 'w_nodes', None) is None:
        return None
    nstate, nk, nx = t.shape
    cheb = resolvent_nodes(lam[:, None] - shifts[None, :], blocks.delta_min,
                           amplitudes_present(t))
    if cheb is None:
        return None
    nodes, weights = cheb
    w_all = blocks.w_nodes(nodes)
    if w_all is None:
        return None
    dtype = numpy.result_type(t, w_all)
    tj = numpy.ascontiguousarray(t).reshape(nstate * nk, nx)
    acc = numpy.zeros((nstate, nk, nx), dtype=dtype)
    for w, wgt in zip(w_all, weights):
        y = la.gemm(tj, numpy.ascontiguousarray(w), 'N', 'T').reshape(nstate, nk, nx)
        acc += y * wgt.T[:, :, None]
    return -la.gemm(tj.reshape(nstate, nk * nx).conj(), acc.reshape(nstate, nk * nx), 'N', 'T')


def choose_route(blocks, nx, dim, nk):
    '''Cheaper order of the two contractions (flop estimate).'''
    ne = nlabel(blocks)
    cost_t = nx * dim * nk + ne * nx * nk
    cost_r = ne * dim * (nx + nk)
    return 'T' if cost_t <= cost_r else 'R'


def gram_route(nstate, ne, nx, shifts, col=None):
    '''Cheaper form of the sum over (k, e) for projected amplitudes: 'rows'
    (u = t v^T for one k at a time and the weighted Gram matrix of its rows,
    cost nstate ne (nX + ncol) per k) or 'w' (the external labels first:
    W_k[X, Y], cost ne nX^2 / 2 per k and distinct shift), which wins for
    many states and few operators.'''
    ncol = nstate if col is None else 1
    nshift = len(numpy.unique(shifts)) if col is None else 1
    rows = nstate * ne * (nx + ncol)
    w = nshift * ne * nx * (nx + 1) / 2 + ncol * nx * (nx + nstate)
    return 'rows' if rows <= w else 'w'


# Chebyshev interpolation of the resolvent in the energy argument: at most
# this many nodes, chosen for an interpolation error of exp(-INTERP_DIGITS)
INTERP_MAX_NODES = 96
INTERP_DIGITS = 40.0
# False: every contraction uses the exact denominators (validation of the
# interpolation; also set by the environment variable ZMR_NO_INTERPOLATION)
INTERPOLATE = not os.environ.get('ZMR_NO_INTERPOLATION')


def chebyshev_weights(d, dmin):
    '''(nodes, weights) of the Chebyshev interpolant in the argument d of
    functions with poles at or below -dmin: g(d[k, j]) = sum_m weights[m, k, j]
    g(nodes[m]) to round-off.  None if the pole is inside the interval of d
    or more than INTERP_MAX_NODES nodes would be needed.'''
    if not INTERPOLATE:
        return None
    lo, hi = d.min(), d.max()
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    if dmin + lo <= 0:
        return None
    if half < 1e-13 * max(abs(mid), 1.0):
        return numpy.array([mid]), numpy.ones((1,) + d.shape)
    z = (dmin + mid) / half
    n = int(numpy.ceil(INTERP_DIGITS / numpy.log(z + numpy.sqrt(z * z - 1)))) + 1
    if n > INTERP_MAX_NODES:
        return None
    theta = (2 * numpy.arange(n) + 1) * numpy.pi / (2 * n)
    nodes = mid + half * numpy.cos(theta)
    # barycentric Lagrange weights of the Chebyshev points of the first kind
    bary = (-1.0)**numpy.arange(n) * numpy.sin(theta)
    diff = d[None, :, :] - nodes[:, None, None]
    hit = diff == 0
    diff[hit] = 1.0
    weights = bary[:, None, None] / diff
    weights /= weights.sum(axis=0)[None]
    on_node = hit.any(axis=0)
    if on_node.any():
        weights[:, on_node] = hit[:, on_node]
    return nodes, weights


def resolvent_nodes(d, dmin, used=None):
    '''(nodes, weights) for the arguments d[k, J] of the external sums:
    the distinct arguments themselves where amplitudes are present (used[k,
    J]) if these are no more than the interpolation nodes -- the exact sums,
    e.g. ccvv with a few states, whose functions have components only on the
    model states -- and chebyshev_weights otherwise.'''
    cheb = chebyshev_weights(d, dmin)
    if used is None:
        used = numpy.ones(d.shape, dtype=bool)
    if dmin + d.min() <= 0:
        return cheb
    scale = max(abs(d).max(), 1.0)
    values, index = numpy.unique(numpy.round(d[used] / scale, 12), return_inverse=True)
    if len(values) == 0 or (cheb is not None and len(values) >= len(cheb[0])) \
            or (cheb is None and len(values) > INTERP_MAX_NODES):
        return cheb
    nodes = numpy.zeros(len(values))
    numpy.add.at(nodes, index, d[used])
    nodes /= numpy.bincount(index, minlength=len(values))
    weights = numpy.zeros((len(values),) + d.shape)
    k, j = numpy.nonzero(used)
    weights[index, k, j] = 1.0
    return nodes, weights


def amplitudes_present(t):
    '''used[k, J]: t[J, k, :] does not vanish.'''
    size = abs(t).max(axis=2).T
    return size > 1e-14 * max(size.max(initial=0.0), 1e-300)


def interpolated_block_t(t, lam, delta, v, shifts, col=None, max_cost=None):
    '''a[I, J] of one block with the external labels contracted first,

        a[I, J] = - sum_k t[I, k, :]^+ W(lam_k - s_J) t[J, k, :],
        W(d)[X, Y] = sum_e conj(v[e, X]) v[e, Y] / (d + delta_e),

    where W(d) is interpolated in d.  W is analytic in d up to its nearest
    pole at -min(delta), so its Chebyshev interpolant on the interval of the
    needed arguments converges geometrically; the number of nodes is chosen
    for an error at round-off level, and the cost in the external labels is
    one matrix W per node instead of one per (k, distinct shift).  Returns
    None if the interval reaches the pole (intruder) or too many nodes would
    be needed.'''
    nstate, nk, nx = t.shape
    js = numpy.arange(nstate) if col is None else numpy.array([col])
    cheb = chebyshev_weights(lam[:, None] - shifts[None, js], delta.min())
    if cheb is None:
        return None
    nodes, weights = cheb
    if max_cost is not None:
        cost = (len(nodes) * nx * nx * (len(delta) + len(js) * nk)
                + nstate * len(js) * nk * nx)
        if cost > max_cost:
            return None
    dtype = numpy.result_type(t, v)
    tj = numpy.ascontiguousarray(t[js]).reshape(len(js) * nk, nx)
    tall = numpy.ascontiguousarray(t).reshape(nstate, nk * nx)
    acc = numpy.zeros((len(js), nk, nx), dtype=dtype)
    for m, (x, wgt) in enumerate(zip(nodes, weights)):
        output.progress('interpolation node', m + 1, len(nodes))
        w = la.gemm(v, v / (x + delta)[:, None], 'C', 'N')          # W[X, Y]
        y = la.gemm(tj, w, 'N', 'T').reshape(len(js), nk, nx)       # (W t_J)[k, X]
        acc += y * wgt.T[:, :, None]
    a = numpy.zeros((nstate, nstate), dtype=dtype)
    a[:, js] = -la.gemm(tall.conj(), acc.reshape(len(js), nk * nx), 'N', 'T')
    return a


def exact_block_t(t, lam, delta, v, shifts, col=None, gram=None):
    '''a[I, J] of one block of external labels from projected amplitudes
    t (nstate, nk, nX).'''
    nstate, nk, nx = t.shape
    dtype = numpy.result_type(t, v)
    if nk == 0 or len(delta) == 0:
        return numpy.zeros((nstate, nstate), dtype=dtype)
    if nx == 1:
        # a single operator (ccvv): u^J_ek = v_e t_Jk, so
        # a[I, J] = -sum_k conj(t_Ik) t_Jk f_k(s_J), f_k(s) = sum_e |v_e|^2 / D_ek(s)
        w2 = abs(v[:, 0])**2
        tk = t[:, :, 0]
        a = numpy.zeros((nstate, nstate), dtype=dtype)
        for s in numpy.unique(shifts):
            js = numpy.nonzero(shifts == s)[0]
            if col is not None:
                js = js[js == col]
            if len(js) == 0:
                continue
            f = (w2[:, None] / (lam[None, :] + delta[:, None] - s)).sum(axis=0)
            a[:, js] = -(tk.conj() * f[None, :]) @ tk[js].T
        return a
    if gram is None:
        gram = gram_route(nstate, len(delta), nx, shifts, col)
    if gram == 'w':
        return kernels.contract_w(t, v, lam, delta, shifts, col)
    # u[J, k, e] = sum_X t[J, k, X] v[e, X], formed inside the kernel
    return kernels.contract_t(t, v, lam, delta, shifts, col)


def single_operator_contract(blocks, t, lam, shifts, col=None):
    '''a[I, J] of a class with one active operator (ccvv), all external
    blocks at once: u^J_ek = v_e t_Jk, so

        a[I, J] = - sum_k conj(t_Ik) t_Jk f[k, J],
        f[k, J] = sum_e |v_e|^2 / (lam_k + delta_e - s_J),

    and f is needed only where t_Jk does not vanish (for eigenstate model
    vectors: k in the level of J).'''
    tk = numpy.ascontiguousarray(t[:, :, 0])
    nstate, nk = tk.shape
    mask = abs(tk.T) > 1e-14 * max(abs(tk).max(initial=0.0), 1e-300)
    if col is not None:
        keep = numpy.zeros(nstate, dtype=bool)
        keep[col] = True
        mask &= keep[None, :]
    blocks = [(d, v) for d, v in blocks if len(d)]
    f = numpy.zeros((nk, nstate))
    if not blocks:
        return f.astype(tk.dtype)
    delta = numpy.concatenate([d for d, _ in blocks])
    w = numpy.concatenate([abs(v[:, 0])**2 for _, v in blocks])
    cheb = resolvent_nodes(lam[:, None] - shifts[None, :], delta.min(), mask)
    if cheb is not None:
        nodes, weights = cheb
        for m, (x, wgt) in enumerate(zip(nodes, weights)):
            output.progress('node', m + 1, len(nodes))
            f += wgt * (w / (x + delta)).sum()
    else:
        kernels.shifted_sums(w, delta, lam, shifts, mask, f)
    return -la.gemm(tk.conj(), numpy.where(mask, f * tk.T, 0))


def class_contract_t(blocks, t, lam, shifts, col=None):
    '''a[I, J] of a class from projected amplitudes: all external labels of
    the class in one interpolated contraction (the matrices W of different
    blocks add up), block by block with exact denominators otherwise.'''
    if col is None and t.shape[1]:
        a = direct_contract_t(blocks, numpy.ascontiguousarray(t), lam, shifts)
        if a is not None:
            return a
    if t.shape[2] == 1:
        return single_operator_contract(blocks, t, lam, shifts, col)
    blocks = [(d, v) for d, v in blocks if len(d)]
    if not blocks or t.shape[1] == 0:
        return 0
    delta = numpy.concatenate([d for d, _ in blocks])
    v = numpy.concatenate([v for _, v in blocks], axis=0)
    # operation count of the block-wise kernels (the cheaper of their two forms)
    nstate, nk, nx = t.shape
    ncol = nstate if col is None else 1
    nshift = len(numpy.unique(shifts)) if col is None else 1
    rows = nk * nstate * len(delta) * (nx + ncol)
    cols = nk * (nshift * len(delta) * nx * (nx + 1) / 2 + ncol * nx * (nx + nstate))
    a = interpolated_block_t(t, lam, delta, v, shifts, col, max_cost=min(rows, cols))
    if a is None:
        a = 0
        for m, (d, v) in enumerate(blocks):
            output.progress('external block', m + 1, len(blocks))
            a = a + exact_block_t(t, lam, d, v, shifts, col)
    return a


def exact_block_r(basis, phi, lam, delta, v, shifts, col=None):
    '''a[I, J] of one block, contracting the integrals with the functions
    first: r^J_e = sum_X v[e, X] phi[J, X, :], u^J[k, e] = <b_k|r^J_e>.'''
    nstate = phi.shape[0]
    nk = basis.shape[1]
    dtype = numpy.result_type(basis, phi, v)
    if nk == 0 or len(delta) == 0:
        return numpy.zeros((nstate, nstate), dtype=dtype)
    u = numpy.empty((nstate, nk, len(delta)), dtype=dtype)
    for j in range(nstate):
        r = la.gemm(v, phi[j])                       # (e, dim)
        u[j] = la.gemm(basis, r, 'C', 'T')           # (k, e)
    return kernels.weighted_gram(u, lam, delta, shifts, col)


def _dtype(blocks, *arrays):
    if hasattr(blocks, 'nlabel'):          # blocks that are not listed
        return numpy.result_type(blocks.dtype, *arrays)
    return numpy.result_type(*arrays, *[v for _, v in blocks[:1]])


# symmetric window for the decay channels of core-hole states: terms with
# |lam_k + delta_e - s_J| < WINDOW are left out (None: no window); the number
# of terms left out and the sum of their weights |u|^2 accumulate in
# window_stats
WINDOW = None
window_stats = [0, 0.0]


def windowed_contract_t(blocks, t, lam, shifts, col, tau):
    '''a[I, J] from projected amplitudes with exact denominators, leaving
    out the terms with |denominator| < tau.  Plain numpy, for the small
    decay-channel blocks only.'''
    nstate, nk, nx = t.shape
    js = range(nstate) if col is None else [col]
    a = numpy.zeros((nstate, nstate), dtype=_dtype(blocks, t))
    for delta, v in blocks:
        if len(delta) == 0 or nk == 0:
            continue
        u = numpy.einsum('jkx,ex->jke', t, v)
        for j in js:
            d = lam[:, None] + delta[None, :] - shifts[j]
            inside = abs(d) < tau
            w = numpy.where(inside, 0.0, 1.0 / numpy.where(inside, 1.0, d))
            a[:, j] -= numpy.einsum('ike,ke->i', u.conj(), u[j] * w)
            window_stats[0] += int(inside.sum())
            window_stats[1] += float((abs(u[j])**2)[inside].sum())
    return a


def exact_contract(blocks, basis, phi, lam, shifts, route=None, col=None):
    '''a[I, J] of one class, exact sum over the first-order space spanned by
    the columns of basis (dim, nk) with energies lam.  col: only that column
    J is computed (state-specific first-order spaces); the rest is zero.'''
    nstate, nx, dim = phi.shape
    nk = basis.shape[1]
    a = numpy.zeros((nstate, nstate), dtype=_dtype(blocks, basis, phi))
    if nk == 0 or not blocks:
        return a
    shifts = numpy.asarray(shifts, dtype=float)
    lam = numpy.asarray(lam, dtype=float)
    report_denominator(blocks, lam, shifts, col)
    if WINDOW is not None:
        t = project_functions(basis, phi)
        return a + windowed_contract_t(blocks, t, lam, shifts, col, WINDOW)
    if route is None:
        route = 'T' if nx == 1 else choose_route(blocks, nx, dim, nk)
    if route == 'T':
        t = project_functions(basis, phi)
        a += class_contract_t(blocks, t, lam, shifts, col)
    else:
        listed = list(blocks)
        for m, (delta, v) in enumerate(listed):
            output.progress('external block', m + 1, len(listed))
            a += exact_block_r(basis, phi, lam, delta, v, shifts, col)
    return a


def exact_contract_t(blocks, t, lam, shifts, col=None):
    '''As exact_contract for amplitudes t (nstate, nk, nX) that are already
    projected (RDM-based contracted spaces).'''
    nstate = t.shape[0]
    a = numpy.zeros((nstate, nstate), dtype=_dtype(blocks, t))
    shifts = numpy.asarray(shifts, dtype=float)
    lam = numpy.asarray(lam, dtype=float)
    t = numpy.ascontiguousarray(t)
    if blocks:
        report_denominator(blocks, lam, shifts, col)
    if t.shape[1] and blocks:
        if WINDOW is not None:
            return a + windowed_contract_t(blocks, t, lam, shifts, col, WINDOW)
        a += class_contract_t(blocks, t, lam, shifts, col)
    return a


def exact_class(pclass, ctx, act, civecs, nelec, shifts):
    '''a[I, J] = <I|V R(E0_J) V|J> restricted to one class, evaluated exactly
    in the complete target sector (uncontracted).'''
    nstate = numpy.asarray(civecs).shape[1]
    blocks = class_blocks(pclass, ctx)
    phi = class_functions(pclass, act, civecs, nelec)
    if not blocks or phi is None:
        return numpy.zeros((nstate, nstate), dtype=numpy.asarray(civecs).dtype)
    sector = act.sector(act.target(nelec, pclass.dn))
    return exact_contract(blocks, sector.vecs, phi, sector.lam, shifts)
