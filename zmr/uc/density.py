'''
Unrelaxed (response) one-particle densities of NEVPT2: derivatives of the second-order
energy with respect to the one-electron Hamiltonian at fixed orbitals and fixed model
states, so that for h -> h + lambda O

    dE2 / dlambda = sum_pq D[p, q] O[p, q]

(finite field with frozen orbitals and CI vectors).  A perturbation of h changes the
generalized Fock matrix by O itself; the quasi-canonical orbitals and their energies
follow it, and the energy of a class, written without canonical orbitals, depends on
the core-core and virtual-virtual blocks of the Fock matrix through its denominators.

ccvv (state independent), with t[ij, ab] = v[ij, ab] / Delta, v = <ab||ij> and all
orderings of the indices (E = -1/4 sum |v|^2 / Delta):

    D_vv[a, a'] =  1/2 sum_{ij b} conj(t[ij, ab]) t[ij, a'b]
    D_cc[i, i'] = -1/2 sum_{j ab} conj(t[ij, ab]) t[i'j, ab]

in the quasi-canonical orbitals.  The sum runs over the core orbitals j in the outer
loop, so that the amplitudes t[ij] of all i are at hand for D_cc: every pair is formed
twice (twice the cost of the energy alone).
'''

import numpy

from socutils.zmr.lib import linalg as la
from socutils.zmr.uc import classes
from socutils.zmr.uc.classes import _slice


def ccvv(ctx, kramers=True):
    '''(E_ccvv, D_cc, D_vv) in the orbitals of ctx (zmr.uc.classes.Context of
    quasi-canonical orbitals, factorized integrals): the energy and its derivatives
    with respect to the core-core and virtual-virtual blocks of the Fock matrix,
    dE = sum dF_cc * D_cc + sum dF_vv * D_vv (elementwise).  Native kernel; with Kramers
    pairs (eris.time_reversal, kramers) the core orbitals j of one member of every pair
    only, the others from time reversal: E twice, D + Theta conj(D) Theta^+.'''
    from socutils.zmr.lib import kernels
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    if nc < 2 or nv < 2:
        return (0.0, numpy.zeros((nc, nc), dtype=numpy.complex128),
                numpy.zeros((nv, nv), dtype=numpy.complex128))
    jlist = th_c = th_v = None
    if kramers and classes.kramers_pairs(ctx) is not None:
        sp = ctx.sp
        tr = ctx.eris.time_reversal
        core = tr.subset(numpy.arange(sp.ncore))
        th_c = core.matrix()
        th_v = tr.subset(numpy.arange(sp.ncore + sp.nact, sp.nmo)).matrix()
        jlist = core.representatives()
    res = kernels.d_ccvv(ctx.eris.cd_block('vc'), ctx.eps_v, ctx.eps_c, jlist=jlist)
    if res is None:
        return ccvv_reference(ctx)
    e, dcc, dvv = res
    if jlist is not None:
        e = 2 * e
        dcc = dcc + th_c @ dcc.conj() @ th_c.conj().T
        dvv = dvv + th_v @ dvv.conj() @ th_v.conj().T
    return e, dcc, dvv


def ccvv_reference(ctx):
    '''ccvv as in ccvv, in Python (all pairs; for checks and without libzmr).'''
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    dcc = numpy.zeros((nc, nc), dtype=numpy.complex128)
    dvv = numpy.zeros((nv, nv), dtype=numpy.complex128)
    if nc < 2 or nv < 2:
        return 0.0, dcc, dvv
    bvc = ctx.eris.cd_block('vc')                  # (L, a, i)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    dab = ev[:, None] + ev[None, :]
    slices = [_slice(bvc, i) for i in range(nc)]   # B[:, :, i] as (L, nv)
    e = 0.0
    t = numpy.zeros((nc, nv, nv), dtype=numpy.complex128)
    for j in range(nc):
        for i in range(nc):
            if i == j:
                t[i] = 0.0
                continue
            m = la.gemm(slices[i], slices[j], 'T', 'N')     # (ai|bj)
            v = m - m.T                                       # <ab||ij>
            t[i] = v / (dab - ec[i] - ec[j])
            e -= 0.25 * float(numpy.vdot(v, t[i]).real)
        flat = t.reshape(nc, nv * nv)
        dcc -= 0.5 * la.gemm(flat.conj(), flat, 'N', 'T')
        for i in range(nc):
            if i != j:
                dvv += 0.5 * la.gemm(t[i].conj(), t[i], 'N', 'T')
    return e, dcc, dvv


# ----------------------------------------------------------------------
# classes with active operators (Python reference: all amplitudes of a class in memory)

def first_order_space(pt, pclass, state=0):
    '''(b, mu, t, s) of the class for one model state of a finished PC-NEVPT2 pt: the
    orthonormal basis b (D, nk) of the first-order space in the determinants of the
    target sector, its H_act energies mu, the projected functions
    t[k, X] = <b_k|O_X|Psi>, and the shift s = E0 - E_core of the state.'''
    from socutils.zmr.pc import spectral
    from socutils.zmr.uc import engine
    # kept on pt: the density, the orbital and the CI gradients ask for the same space
    cache = pt.__dict__.setdefault('_first_order_spaces', {})
    key = (pclass.name, state)
    if key in cache:
        return cache[key]
    act, civecs = pt.active_space, pt.civecs
    ntarget = act.target(pt.nelec, pclass.dn)
    phi = engine.class_functions(pclass, act, civecs[:, [state]], pt.nelec)
    b, mu = spectral.contracted_space(phi.reshape(-1, phi.shape[2]),
                                      lambda q: act.apply_h(ntarget, q), lambda: None, pt.lindep)
    t = engine.project_functions(b, phi)[0]
    cache[key] = (b, mu, t, float(pt.shifts[state]))
    return cache[key]


def _active_block(pt, pclass, b, g):
    '''D_aa[t, u] = sum_kk' g[k, k'] <b_k'|t^+ u|b_k> - tr(g) <Psi|t^+ u|Psi>: the
    derivative of the class energy with respect to the active block of h through
    H_act in the first-order space and the zeroth-order energy of the model state
    (g[k, k'] = sum_e c[e, k] conj(c[e, k'])).'''
    from socutils.zmr.active import rdm as active_rdm
    act = pt.active_space
    ntarget = act.target(pt.nelec, pclass.dn)
    # g = sum_m w_m y_m y_m^+ (rank at most the number of external labels): the 1-RDMs
    # of the vectors b y_m only, not the transition RDMs of all pairs of basis functions
    w, y = numpy.linalg.eigh(0.5 * (g + g.conj().T))
    keep = w > 1e-14 * max(1.0, float(w.max()) if len(w) else 1.0)
    d = numpy.zeros((act.norb, act.norb), dtype=numpy.complex128)
    if keep.any() and ntarget >= 1:
        vecs = numpy.ascontiguousarray(b @ y[:, keep])
        amp = active_rdm.annihilation_amplitudes(act, vecs, ntarget, 1)
        amp = amp.reshape(vecs.shape[1], act.norb, -1)
        d = numpy.einsum('m,mpk,mqk->pq', w[keep], amp.conj(), amp, optimize=True)
    gamma = active_rdm.make_rdm(act, pt.civecs[:, [0]], pt.nelec, 1)[0, 0]
    return d - numpy.trace(g) * gamma


# external labels of the classes: (core indices, virtual indices), the layout of the
# labels within a block of PerturberClass.blocks, and the Fock elements in v:
# (block of the density, sign, number of leading operators X they multiply)
_LABELS = {'ccav': (2, 1), 'cavv': (1, 2), 'ccaa': (2, 0), 'aavv': (0, 2),
           'cava': (1, 1), 'caaa': (1, 0), 'aaav': (0, 1)}


def _amplitudes(pt, ctx, pclass, state):
    '''(E, C, b, t): the amplitudes c[e, k] = u[e, k] / (mu_k + delta_e - s) of a class
    as C[i, j, a, b, k], antisymmetric in the core and in the virtual indices (axes of
    length 1 for the indices the class does not have), the basis b of the first-order
    space and the projected functions t[k, X].'''
    b, mu, t, s = first_order_space(pt, pclass, state)
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    ncore, nvirt = _LABELS[pclass.name]
    nk = len(mu)
    shape = [nc if ncore > 0 else 1, nc if ncore > 1 else 1,
             nv if nvirt > 0 else 1, nv if nvirt > 1 else 1, nk]
    c = numpy.zeros(shape, dtype=numpy.complex128)
    ii, jj = numpy.triu_indices(nc, 1)
    aa, bb = numpy.triu_indices(nv, 1)
    e = 0.0
    for n, (delta, v) in enumerate(pclass.blocks(ctx)):
        u = v @ t.T
        x = u / (delta[:, None] + mu[None, :] - s)
        e -= float(numpy.vdot(u, x).real)
        name = pclass.name
        if name == 'ccav':                       # block: pair (i < j), labels a
            c[ii[n], jj[n], :, 0] = x
            c[jj[n], ii[n], :, 0] = -x
        elif name == 'cavv':                     # block: i, labels a < b
            c[n, 0, aa, bb] = x
            c[n, 0, bb, aa] = -x
        elif name == 'ccaa':                     # one block, labels i < j
            c[ii, jj, 0, 0] = x
            c[jj, ii, 0, 0] = -x
        elif name == 'aavv':                     # one block, labels a < b
            c[0, 0, aa, bb] = x
            c[0, 0, bb, aa] = -x
        elif name == 'cava':                     # block: i, labels a
            c[n, 0, :, 0] = x
        elif name == 'caaa':                     # one block, labels i
            c[:, 0, 0, 0] = x
        elif name == 'aaav':                     # one block, labels a
            c[0, 0, :, 0] = x
    return e, c, b, t


def class_reference(pt, ctx, name, state=0):
    '''(E, D) of one class with active operators for one model state: D the blocks
    {'cc', 'aa', 'vv', 'vc', 'ac', 'va'} of the unrelaxed density that the class has,
    dE = sum_pq O[p, q] D[p, q] for h -> h + lambda O with Hermitian O (the block 'vc'
    is D[a, i]; D[i, a] is its conjugate, likewise 'ac' and 'va').  Python reference:
    all amplitudes of the class in memory.

    The energy depends on h through the denominators (core-core and virtual-virtual
    blocks of the Fock matrix), through H_act in the first-order space and the
    zeroth-order energy of the model state (active block), and through the Fock
    elements in the perturbation (cava: f_ai, caaa: f_ti, aaav: f_at).'''
    pclass = [p for p in pt.perturber_classes() if p.name == name][0]
    e, c, b, t = _amplitudes(pt, ctx, pclass, state)
    ncore, nvirt = _LABELS[name]
    fc = 0.5 if ncore == 2 else 1.0
    fv = 0.5 if nvirt == 2 else 1.0
    na = ctx.sp.nact
    d = {'aa': _active_block(pt, pclass, b, fc * fv * numpy.einsum('ijabk,ijabl->kl', c, c.conj(), optimize=True))}
    if ncore:
        d['cc'] = -fv * numpy.einsum('ijabk,ljabk->il', c.conj(), c, optimize=True)
    if nvirt:
        d['vv'] = fc * numpy.einsum('ijabk,ijdbk->ad', c.conj(), c, optimize=True)
    if name == 'cava':                           # v[(i, a), 0] = f_ai
        d['vc'] = -numpy.einsum('iak,k->ai', c[:, 0, :, 0].conj(), t[:, 0])
    elif name == 'caaa':                         # v[i, t] = -f_ti
        d['ac'] = numpy.einsum('ik,kt->ti', c[:, 0, 0, 0].conj(), t[:, :na])
    elif name == 'aaav':                         # v[a, t] = f_at
        d['va'] = -numpy.einsum('ak,kt->at', c[0, 0, :, 0].conj(), t[:, :na])
    return e, d


def assemble(sp, blocks):
    '''The (nmo, nmo) density from blocks {'cc': ..., 'vc': ..., ...} (the off-diagonal
    blocks with their conjugates).'''
    out = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    for lab, x in blocks.items():
        p, q = sp.slice(lab[0]), sp.slice(lab[1])
        out[p, q] += x
        if lab[0] != lab[1]:
            out[q, p] += x.conj().T
    return out


def context(pt):
    '''The Context of the quasi-canonical orbitals of a finished NEVPT2 (with the core
    Fock matrix, as the driver builds it).'''
    from socutils.zmr import fock
    qc = pt.quasi_canonical
    na = qc.eris.spaces.nact
    fcore = fock.generalized_fock(qc.eris, numpy.zeros((na, na), dtype=numpy.complex128))
    return classes.Context(qc.eris, qc.eps_core, qc.eps_virt, fcore)


def make_rdm1(pt, state=0, max_memory=4000, timings=None):
    '''(D_ref, D_pt2) of one model state of a finished single-state PC-NEVPT2, in the
    correlated orbitals of the input integrals (pt.eris0), gamma convention
    D[p, q] = <p^+ q>: for h -> h + lambda O, dE_ref = sum O[p, q] D_ref[p, q] (the
    reference: core occupied, the active 1-RDM of the model state) and
    dE2 = sum O[p, q] D_pt2[p, q] at fixed orbitals and CI vector (unrelaxed).
    ccvv from the native kernel, cavv and ccav in segments within max_memory (MB), the
    other classes with their amplitudes in memory.  timings: dict that receives the
    seconds per class.'''
    import time
    from socutils.zmr.active import rdm as active_rdm
    qc = pt.quasi_canonical
    sp = qc.eris.spaces
    ctx = context(pt)
    d2 = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    for name in ('ccvv',) + tuple(_LABELS):
        t0 = time.time()
        if name == 'ccvv':
            _, dcc, dvv = ccvv(ctx)
            blocks = {'cc': dcc, 'vv': dvv}
        elif name == 'cavv':
            blocks = cavv(pt, ctx, state, max_memory)[1]
        elif name == 'ccav':
            blocks = ccav(pt, ctx, state, max_memory)[1]
        else:
            blocks = class_reference(pt, ctx, name, state)[1]
        d2 += assemble(sp, blocks)
        if timings is not None:
            timings[name] = time.time() - t0
    # quasi-canonical orbitals phi' = phi u: sum O' D' = sum O D with D = conj(u) D' u^T
    u = numpy.asarray(qc.u)
    d2 = u.conj() @ d2 @ u.T
    ref = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    c, a = sp.slice('c'), sp.slice('a')
    ref[c, c] = numpy.eye(sp.ncore)
    ref[a, a] = active_rdm.make_rdm(pt.active_space, pt.civecs[:, [state]], pt.nelec, 1)[0, 0]
    return ref, d2


# ----------------------------------------------------------------------
# cavv and ccav for large external spaces: the amplitudes of one segment of an external
# index at a time, from the three-index integrals (matrix products only)

def _segments(n, per_index, max_memory):
    '''Ranges of an external index such that six arrays of per_index complex numbers
    per index fit max_memory (MB).'''
    step = max(1, min(n, int(max_memory * 1e6 / (6 * 16.0 * max(per_index, 1)))))
    return [(p, min(p + step, n)) for p in range(0, n, step)]


def cavv(pt, ctx, state=0, max_memory=4000):
    '''(E, D) of cavv as class_reference, for one model state, in segments of the second
    virtual index b: C[a, i, b, k] for all a, i and the b of a segment, antisymmetric in
    (a, b),

        V[a, i, b, t] = -((ai|bt) - (bi|at)),   C = (V t^T) / (e_a + e_b - e_i + mu_k - s),

    two matrix products with the three-index integrals per segment (twice those of the
    energy, which needs a < b only) and the Gram matrices of C over its other indices.'''
    pclass = classes.CLASS_BY_NAME['cavv']
    b, mu, t, s = first_order_space(pt, pclass, state)
    sp = ctx.sp
    nc, na, nv = sp.ncore, sp.nact, sp.nvirt
    nk = len(mu)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    bvc = numpy.ascontiguousarray(ctx.eris.cd_block('vc'))        # (L, a, i)
    bva = numpy.ascontiguousarray(ctx.eris.cd_block('va'))        # (L, a, t)
    naux = bvc.shape[0]
    left = bvc.reshape(naux, nv * nc)
    right = bva.reshape(naux, nv * na)
    tt = numpy.ascontiguousarray(t.T)                              # (t, k)
    e = 0.0
    g = numpy.zeros((nk, nk), dtype=numpy.complex128)
    dcc = numpy.zeros((nc, nc), dtype=numpy.complex128)
    dvv = numpy.zeros((nv, nv), dtype=numpy.complex128)
    for b0, b1 in _segments(nv, nv * nc * max(na, nk), max_memory):
        nb = b1 - b0
        # m1[a, i, b, t] = (ai|bt), m2[b, i, a, t] = (bi|at)
        m1 = la.gemm(left, numpy.ascontiguousarray(bva[:, b0:b1]).reshape(naux, nb * na),
                     'T', 'N').reshape(nv, nc, nb, na)
        m2 = la.gemm(numpy.ascontiguousarray(bvc[:, b0:b1]).reshape(naux, nb * nc), right,
                     'T', 'N').reshape(nb, nc, nv, na)
        v = m2.transpose(2, 1, 0, 3) - m1
        del m1, m2
        u = la.gemm(numpy.ascontiguousarray(v).reshape(-1, na), tt).reshape(nv, nc, nb, nk)
        del v
        den = (ev[:, None, None, None] - ec[None, :, None, None]
               + ev[None, None, b0:b1, None] + (mu - s)[None, None, None, :])
        c = u / den
        e -= 0.5 * float(numpy.vdot(u, c).real)
        del u, den
        flat = c.reshape(-1, nk)
        g += 0.5 * la.gemm(flat, flat.conj(), 'T', 'N')
        rows = c.reshape(nv, -1)
        dvv += la.gemm(rows.conj(), rows, 'N', 'T')
        rows = numpy.ascontiguousarray(c.transpose(1, 0, 2, 3)).reshape(nc, -1)
        dcc -= 0.5 * la.gemm(rows.conj(), rows, 'N', 'T')
        del c, rows, flat
    return e, {'cc': dcc, 'vv': dvv, 'aa': _active_block(pt, pclass, b, g)}


def ccav(pt, ctx, state=0, max_memory=4000):
    '''(E, D) of ccav as class_reference, for one model state, in segments of the second
    core index j: C[a, i, j, k], antisymmetric in (i, j),

        V[a, i, j, t] = (ai|tj) - (aj|ti),   C = (V t^T) / (e_a - e_i - e_j + mu_k - s).'''
    pclass = classes.CLASS_BY_NAME['ccav']
    b, mu, t, s = first_order_space(pt, pclass, state)
    sp = ctx.sp
    nc, na, nv = sp.ncore, sp.nact, sp.nvirt
    nk = len(mu)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    bvc = numpy.ascontiguousarray(ctx.eris.cd_block('vc'))        # (L, a, i)
    bac = numpy.ascontiguousarray(ctx.eris.cd_block('ac'))        # (L, t, i)
    naux = bvc.shape[0]
    left = bvc.reshape(naux, nv * nc)
    right = bac.reshape(naux, na * nc)
    tt = numpy.ascontiguousarray(t.T)
    e = 0.0
    g = numpy.zeros((nk, nk), dtype=numpy.complex128)
    dcc = numpy.zeros((nc, nc), dtype=numpy.complex128)
    dvv = numpy.zeros((nv, nv), dtype=numpy.complex128)
    for j0, j1 in _segments(nc, nv * nc * max(na, nk), max_memory):
        nj = j1 - j0
        # m1[a, i, t, j] = (ai|tj), m2[a, j, t, i] = (aj|ti)
        m1 = la.gemm(left, numpy.ascontiguousarray(bac[:, :, j0:j1]).reshape(naux, na * nj),
                     'T', 'N').reshape(nv, nc, na, nj)
        m2 = la.gemm(numpy.ascontiguousarray(bvc[:, :, j0:j1]).reshape(naux, nv * nj), right,
                     'T', 'N').reshape(nv, nj, na, nc)
        v = m1.transpose(0, 1, 3, 2) - m2.transpose(0, 3, 1, 2)          # (a, i, j, t)
        del m1, m2
        u = la.gemm(numpy.ascontiguousarray(v).reshape(-1, na), tt).reshape(nv, nc, nj, nk)
        del v
        den = (ev[:, None, None, None] - ec[None, :, None, None]
               - ec[None, None, j0:j1, None] + (mu - s)[None, None, None, :])
        c = u / den
        e -= 0.5 * float(numpy.vdot(u, c).real)
        del u, den
        flat = c.reshape(-1, nk)
        g += 0.5 * la.gemm(flat, flat.conj(), 'T', 'N')
        rows = c.reshape(nv, -1)
        dvv += 0.5 * la.gemm(rows.conj(), rows, 'N', 'T')
        rows = numpy.ascontiguousarray(c.transpose(1, 0, 2, 3)).reshape(nc, -1)
        dcc -= la.gemm(rows.conj(), rows, 'N', 'T')
        del c, rows, flat
    return e, {'cc': dcc, 'vv': dvv, 'aa': _active_block(pt, pclass, b, g)}
