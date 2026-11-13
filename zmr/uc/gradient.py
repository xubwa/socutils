'''
Orbital gradient of the PC-NEVPT2 second-order energy at fixed CI vectors for large
external spaces (real orbitals; the derivation and the reference with all amplitudes in
memory are in zmr.reference.orbital_gradient).

For a derivative G[P, a, i] = dE/dB[P, a, i] of the core-virtual block of the three-index
integrals, its part of the gradient sum_P B[P] (G[P] + G[P]^T) is

    T1[r, i] = sum_{P a} B[P, r, a] G[P, a, i]    (any r: through the AO vectors, the
                                                    virtual index of G back-transformed,
                                                    so no B[P, r, a] block is formed)
    T2[r, a] = sum_{P i} B[P, r, i] G[P, a, i]    (B[P, r, i]: cd_pc of the source)

at the cost of one exchange-type build (N_aux N_ao^2 n_core).
'''

import numpy
from pyscf import lib as pyscf_lib


def ao_back(eris, gam, space='v', out_space='c'):
    '''T1[r, i] = sum_{P x} B[P, r, x] G[P, x, i] for all orbitals r, G[P, x, i] with x in
    space (back-transformed to the AO basis) and i in out_space, from the AO vectors of
    the factorized source, block by block.'''
    fn = eris.eri_fn
    c_all = fn.components(eris.mo_coeff)
    c_x = fn.components(eris.coeff(space))
    nao = fn.nao
    ni = gam.shape[2]
    y = [numpy.zeros((nao, ni), dtype=numpy.result_type(gam, *c_x)) for _ in c_x]
    for p0, p1, packed in fn.loop(nao * ni):
        ao = pyscf_lib.unpack_tril(numpy.asarray(packed))         # (np, nao, nao)
        g = gam[p0:p1]
        for k, cx in enumerate(c_x):
            x = numpy.einsum('mx,Pxi->Pmi', cx, g, optimize=True)  # (np, nao, ni)
            y[k] += numpy.einsum('Pnm,Pmi->ni', ao, x, optimize=True)
    return sum(c.conj().T @ yk for c, yk in zip(c_all, y))


def explicit_vc(eris, gam):
    '''sum_P B[P] (G[P] + G[P]^T) for G nonzero only in the virtual-core block, G[P, a, i]:
    an (nmo, nmo) matrix with T1 in the core columns and T2 in the virtual columns.'''
    sp = eris.spaces
    n = sp.nmo
    out = numpy.zeros((n, n), dtype=numpy.complex128)
    out[:, sp.slice('c')] += ao_back(eris, gam, 'v')
    out[:, sp.slice('v')] += numpy.einsum('Pri,Pai->ra', eris.cd_pc, gam, optimize=True)
    return out


def ccvv(ctx):
    '''(E, D_cc, D_vv, explicit part of the orbital gradient) of ccvv from the native
    kernel: the explicit part is sum_P B[P] (G[P] + G[P]^T) with
    G[P, a, i] = -2 sum_jb t[ij, ab] B[P, b, j].'''
    from socutils.zmr.lib import kernels
    res = kernels.d_ccvv(ctx.eris.cd_block('vc'), ctx.eps_v, ctx.eps_c, gamma=True)
    if res is None:
        raise NotImplementedError('needs libzmr')
    e, dcc, dvv, gam = res
    return e, dcc, dvv, explicit_vc(ctx.eris, gam)


# ----------------------------------------------------------------------
# all classes

def _b_all(eris, space):
    '''B[P, r, x] for all orbitals r and x in the core or active space.'''
    return eris.cd_pc if space == 'c' else eris.cd_pa


def explicit_block(eris, gam, x, y):
    '''sum_P B[P] (G[P] + G[P]^T) for G nonzero only in the block (x, y), G[P, x, y]:
    B[P, all, c / a] from the source, B[P, all, v] through the AO vectors.'''
    sp = eris.spaces
    n = sp.nmo
    out = numpy.zeros((n, n), dtype=numpy.complex128)
    # (B G)[r, q in y] = sum_s B[P, r, s in x] G[P, s, q]
    if x == 'v':
        out[:, sp.slice(y)] += ao_back(eris, gam, 'v')
    else:
        out[:, sp.slice(y)] += numpy.einsum('Prs,Psq->rq', _b_all(eris, x), gam, optimize=True)
    # (B G^T)[r, s in x] = sum_q B[P, r, q in y] G[P, s, q]
    gt = numpy.ascontiguousarray(gam.transpose(0, 2, 1))
    if y == 'v':
        out[:, sp.slice(x)] += ao_back(eris, gt, 'v')
    else:
        out[:, sp.slice(x)] += numpy.einsum('Prq,Psq->rs', _b_all(eris, y), gam, optimize=True)
    return out


def _add(gams, key, value):
    gams[key] = gams[key] + value if key in gams else value


def gammas_from_antisym(eris, labels, a, gams):
    '''The three-index derivative blocks of dE/d<pr||qs> = A[p, r, q, s] (labels p, r, q, s;
    <pr||qs> = (pq|rs) - (ps|rq)), added to gams {(x, y): G[P, x, y]}.'''
    lp, lr, lq, ls = labels
    b = eris.cd_block
    # (pq|rs) with dE/d(pq|rs) = A[p, r, q, s]
    _add(gams, (lp, lq), numpy.einsum('prqs,Prs->Ppq', a, b(lr + ls), optimize=True))
    _add(gams, (lr, ls), numpy.einsum('prqs,Ppq->Prs', a, b(lp + lq), optimize=True))
    # (ps|rq) with dE/d(ps|rq) = -A[p, r, q, s]
    _add(gams, (lp, ls), -numpy.einsum('prqs,Prq->Pps', a, b(lr + lq), optimize=True))
    _add(gams, (lr, lq), -numpy.einsum('prqs,Pps->Prq', a, b(lp + ls), optimize=True))


def _q_update(q, y, v, delta, f):
    '''Q_S += f (y^T v + v^T y - y^T diag(delta) y), Q_K += f y^T y over labels e (rows).'''
    q_s, q_k = q
    yv = y.T @ v
    q_s += f * (yv + yv.T - (y.T * delta[None, :]) @ y)
    q_k += f * (y.T @ y)


def cavv_pass(pt, ctx, gams, state=0, max_memory=4000):
    '''One pass over cavv in segments of b: the density blocks, dE/dB (dE/d(ai|bt) =
    -W[a, i, b, t], W = -2 c t, all a != b) into the blocks (v, c) and (v, a), the active
    part (b, g) and the matrices (Q_S, Q_K) of the CI gradient.  Returns
    (blocks, b, g, q).'''
    from socutils.zmr.lib import linalg as la
    from socutils.zmr.uc import classes, density
    pclass = classes.CLASS_BY_NAME['cavv']
    b, mu, t, s = density.first_order_space(pt, pclass, state)
    sp = ctx.sp
    nc, na, nv = sp.ncore, sp.nact, sp.nvirt
    nk = len(mu)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    bvc = numpy.ascontiguousarray(ctx.eris.cd_block('vc'))
    bva = numpy.ascontiguousarray(ctx.eris.cd_block('va'))
    naux = bvc.shape[0]
    left = bvc.reshape(naux, nv * nc)
    right = bva.reshape(naux, nv * na)
    tt = numpy.ascontiguousarray(t.T)
    tp = numpy.ascontiguousarray(numpy.linalg.pinv(t).T)            # (k, X)
    c16 = numpy.complex128
    g = numpy.zeros((nk, nk), dtype=c16)
    g_vc = numpy.zeros((naux, nv, nc), dtype=c16)
    g_va = numpy.zeros((naux, nv, na), dtype=c16)
    dcc = numpy.zeros((nc, nc), dtype=c16)
    dvv = numpy.zeros((nv, nv), dtype=c16)
    q = (numpy.zeros((na, na), dtype=c16), numpy.zeros((na, na), dtype=c16))
    for b0, b1 in density._segments(nv, nv * nc * max(na, nk), max_memory):
        nb = b1 - b0
        bseg = numpy.ascontiguousarray(bva[:, b0:b1]).reshape(naux, nb * na)
        m1 = la.gemm(left, bseg, 'T', 'N').reshape(nv, nc, nb, na)
        m2 = la.gemm(numpy.ascontiguousarray(bvc[:, b0:b1]).reshape(naux, nb * nc), right,
                     'T', 'N').reshape(nb, nc, nv, na)
        v = numpy.ascontiguousarray(m2.transpose(2, 1, 0, 3) - m1).reshape(-1, na)   # (aib, t)
        del m1, m2
        u = la.gemm(v, tt)
        delta = (ev[:, None, None] - ec[None, :, None] + ev[None, None, b0:b1]).reshape(-1)
        c = u / (delta[:, None] + (mu - s)[None, :])
        del u
        g += 0.5 * la.gemm(c, c.conj(), 'T', 'N')
        c4 = c.reshape(nv, nc, nb, nk)
        rows = c4.reshape(nv, -1)
        dvv += la.gemm(rows.conj(), rows, 'N', 'T')
        rows = numpy.ascontiguousarray(c4.transpose(1, 0, 2, 3)).reshape(nc, -1)
        dcc -= 0.5 * la.gemm(rows.conj(), rows, 'N', 'T')
        # every unordered (a, b) twice: 1/2
        _q_update(q, la.gemm(c, tp), v, delta, 0.5)
        x = 2 * la.gemm(c, t)                                       # dE/d(ai|bt) = -W
        xm = x.reshape(nv * nc, nb * na)
        g_vc += la.gemm(bseg, xm, 'N', 'T').reshape(naux, nv, nc)
        g_va[:, b0:b1] += la.gemm(left, xm).reshape(naux, nb, na)
        del c, c4, rows, x, xm, v
    _add(gams, ('v', 'c'), g_vc)
    _add(gams, ('v', 'a'), g_va)
    blocks = {'cc': dcc, 'vv': dvv, 'aa': density._active_block(pt, pclass, b, g)}
    return blocks, b, g, q


def ccav_pass(pt, ctx, gams, state=0, max_memory=4000):
    '''One pass over ccav in segments of j (dE/d(ai|tj) = W[a, i, j, t], all i != j, into
    the blocks (v, c) and (a, c)); returns (blocks, b, g, q) as cavv_pass.'''
    from socutils.zmr.lib import linalg as la
    from socutils.zmr.uc import classes, density
    pclass = classes.CLASS_BY_NAME['ccav']
    b, mu, t, s = density.first_order_space(pt, pclass, state)
    sp = ctx.sp
    nc, na, nv = sp.ncore, sp.nact, sp.nvirt
    nk = len(mu)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    bvc = numpy.ascontiguousarray(ctx.eris.cd_block('vc'))      # (P, a, i)
    bac = numpy.ascontiguousarray(ctx.eris.cd_block('ac'))      # (P, t, j)
    naux = bvc.shape[0]
    left = bvc.reshape(naux, nv * nc)
    right = bac.reshape(naux, na * nc)
    tt = numpy.ascontiguousarray(t.T)
    tp = numpy.ascontiguousarray(numpy.linalg.pinv(t).T)
    c16 = numpy.complex128
    g = numpy.zeros((nk, nk), dtype=c16)
    g_vc = numpy.zeros((naux, nv, nc), dtype=c16)
    g_ac = numpy.zeros((naux, na, nc), dtype=c16)
    dcc = numpy.zeros((nc, nc), dtype=c16)
    dvv = numpy.zeros((nv, nv), dtype=c16)
    q = (numpy.zeros((na, na), dtype=c16), numpy.zeros((na, na), dtype=c16))
    for j0, j1 in density._segments(nc, nv * nc * max(na, nk), max_memory):
        nj = j1 - j0
        bseg = numpy.ascontiguousarray(bac[:, :, j0:j1])            # (P, t, j)
        m1 = la.gemm(left, bseg.reshape(naux, na * nj), 'T', 'N').reshape(nv, nc, na, nj)
        m2 = la.gemm(numpy.ascontiguousarray(bvc[:, :, j0:j1]).reshape(naux, nv * nj), right,
                     'T', 'N').reshape(nv, nj, na, nc)
        v = numpy.ascontiguousarray(m1.transpose(0, 1, 3, 2) - m2.transpose(0, 3, 1, 2)
                                    ).reshape(-1, na)                # (a i j, t)
        del m1, m2
        u = la.gemm(v, tt)
        delta = (ev[:, None, None] - ec[None, :, None] - ec[None, None, j0:j1]).reshape(-1)
        c = u / (delta[:, None] + (mu - s)[None, :])
        del u
        g += 0.5 * la.gemm(c, c.conj(), 'T', 'N')
        c4 = c.reshape(nv, nc, nj, nk)
        rows = c4.reshape(nv, -1)
        dvv += 0.5 * la.gemm(rows.conj(), rows, 'N', 'T')
        rows = numpy.ascontiguousarray(c4.transpose(1, 0, 2, 3)).reshape(nc, -1)
        dcc -= la.gemm(rows.conj(), rows, 'N', 'T')
        _q_update(q, la.gemm(c, tp), v, delta, 0.5)
        x = -2 * la.gemm(c, t)                                      # dE/d(ai|tj)
        xm = x.reshape(nv * nc, nj * na)
        bjt = numpy.ascontiguousarray(bseg.transpose(0, 2, 1)).reshape(naux, nj * na)
        g_vc += la.gemm(bjt, xm, 'N', 'T').reshape(naux, nv, nc)
        g_ac[:, :, j0:j1] += la.gemm(left, xm).reshape(naux, nj, na).transpose(0, 2, 1)
        del c, c4, rows, x, xm, v
    _add(gams, ('v', 'c'), g_vc)
    _add(gams, ('a', 'c'), g_ac)
    blocks = {'cc': dcc, 'vv': dvv, 'aa': density._active_block(pt, pclass, b, g)}
    return blocks, b, g, q


def active_two_body(pt, pclass, b, g, state=0):
    '''dE/d(tu|vw) of a class through H_act = ... + 1/2 sum (tu|vw) t^+ v^+ w u in the
    first-order space and in the zeroth-order energy, (na,) * 4: 1/2 [sum_m w_m
    Gamma(b y_m) - tr(g) Gamma(Psi)] with g = sum_m w_m y_m y_m^+.'''
    from socutils.zmr.active import rdm as active_rdm
    act = pt.active_space
    ntarget = act.target(pt.nelec, pclass.dn)
    na = act.norb
    w, y = numpy.linalg.eigh(0.5 * (g + g.conj().T))
    keep = w > 1e-14 * max(1.0, float(abs(w).max()) if len(w) else 1.0)
    d2 = numpy.zeros((na,) * 4, dtype=numpy.complex128)
    if keep.any() and ntarget >= 2:
        vecs = numpy.ascontiguousarray(b @ y[:, keep])
        for m, wm in enumerate(w[keep]):
            d2 += wm * active_rdm.make_rdm(act, vecs[:, m], ntarget, 2, bra=0, ket=0)
    if pt.nelec >= 2:
        d2 -= numpy.trace(g) * active_rdm.make_rdm(act, pt.civecs[:, [state]], pt.nelec, 2,
                                                   bra=0, ket=0)
    return 0.5 * d2.real


def derivatives(pt, state=0, max_memory=4000):
    '''One pass over the classes of a finished PC-NEVPT2 of one model state: the
    unrelaxed density (quasi-canonical orbitals), the orbital gradient at fixed CI vector
    (quasi-canonical orbitals), and the vector g of the CI gradient, dE2 = 2 <dPsi|g>
    (determinants of the spinor sector; zmr.reference.ci_gradient).  Real orbitals.
    Returns {'d', 'grad', 'g_ci', 'u'}.'''
    from socutils.zmr.reference import ci_gradient as cg
    from socutils.zmr.reference import orbital_gradient as og
    from socutils.zmr.uc import classes, density
    qc = pt.quasi_canonical
    eris = qc.eris
    sp = eris.spaces
    ctx = density.context(pt)
    n = sp.nmo
    c16 = numpy.complex128
    d_gen = numpy.zeros((n, n), dtype=c16)
    d_core = numpy.zeros((n, n), dtype=c16)
    gams = {}
    x_aa = numpy.zeros((sp.nact,) * 4)
    g_ci = numpy.zeros(pt.civecs.shape[0], dtype=c16)
    _, dcc, dvv, g_ccvv = ccvv(ctx)
    d_gen += density.assemble(sp, {'cc': dcc, 'vv': dvv})
    grad = g_ccvv.real.copy()
    for name in density._LABELS:
        pclass = classes.CLASS_BY_NAME[name]
        if name in ('cavv', 'ccav'):
            run = cavv_pass if name == 'cavv' else ccav_pass
            dblk, b, g, q = run(pt, ctx, gams, state, max_memory)
            g_ci += cg.g_from_q(pt, pclass, q[0], q[1], state)
        else:
            _, dblk = density.class_reference(pt, ctx, name, state)
            blocks, b, g = og._w_blocks(pt, ctx, pclass, state)
            gammas_from_antisym(eris, *og._scatter(name, ctx, blocks), gams)
            g_ci += cg.class_g(pt, ctx, name, state)
        for lab, x in dblk.items():
            target = d_gen if lab in ('cc', 'vv') else d_core
            target += density.assemble(sp, {lab: x})
        x_aa += active_two_body(pt, pclass, b, g, state)
    baa = eris.cd_block('aa')
    _add(gams, ('a', 'a'), numpy.einsum('tuvw,Pvw->Ptu', x_aa + x_aa.transpose(2, 3, 0, 1), baa))
    for (x, y), gam in gams.items():
        grad += explicit_block(eris, numpy.asarray(gam).real, x, y).real
    gam_gen = numpy.zeros((n, n))
    gam_gen[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    gam_gen[sp.slice('a'), sp.slice('a')] = numpy.asarray(pt.gamma).real
    gam_core = numpy.zeros((n, n))
    gam_core[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    # the generalized Fock matrix of the denominators and the core Fock matrix are known
    g_f, w_gen = og.fock_part(eris, d_gen.real, gam_gen, fock=qc.fock, return_w=True)
    grad += g_f + og.fock_part(eris, d_core.real, gam_core, fock=ctx.fcore)
    g_ci += cg.fock_g(pt, ctx, d_gen.real, state, w=w_gen)
    return {'d': d_gen + d_core, 'grad': grad, 'g_ci': g_ci, 'u': numpy.asarray(qc.u)}


def gradient(pt, state=0, max_memory=4000):
    '''(dE2/dK in the quasi-canonical orbitals of pt, u) at fixed CI vector, all classes,
    without MO four-index arrays or virtual-virtual three-index blocks (derivatives).'''
    out = derivatives(pt, state, max_memory)
    return out['grad'], out['u']
