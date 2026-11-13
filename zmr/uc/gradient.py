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


def _pairs(v, delta, mu, tts, s, rhos):
    '''(c, l, r) of zmr.uc.density.pair_amplitudes with u^J = v t^J^T by BLAS (tts[J] =
    t[J]^T), the targets rhos (L, J, J): c (J, e, k), l and r (L, J, e, k).'''
    from socutils.zmr.lib import linalg as la
    u = numpy.array([la.gemm(v, tj) for tj in tts])
    d = delta[None, :, None] + mu[None, None, :] - s[:, None, None]
    c = u / d
    if len(tts) == 1:
        w = rhos[:, 0, 0][:, None, None, None]
        return c, u[None] * (w.conj() / d[None]), w * c[None]
    l = numpy.tensordot(rhos.conj(), u, axes=(1, 0)) / d[None]
    return c, l, numpy.tensordot(rhos, c, axes=(2, 0))


def _herm(x):
    return 0.5 * (x + x.conj().T)


def _external_pass(name, pt, ctx, gams, state, max_memory, rho):
    '''One pass over cavv (segments of b) or ccav (segments of j): the density blocks,
    dE/dB into gams (cavv: dE/d(ai|bt) = -W[a, i, b, t], W = -sum_J (l^J + r^J) t^J, all
    a != b, into (v, c) and (v, a); ccav: dE/d(ai|tj) = W[a, i, j, t], all i != j, into
    (v, c) and (a, c)), the active part (b, g, traces) and the accumulated Q matrices of
    the CI gradient (zmr.reference.ci_gradient.QAccumulator).  rho: the target
    sum_IJ rho_IJ h2_IJ (zmr.uc.density.model_spaces); rho (L, J, J): every target at once
    (the integrals and the amplitudes of the states once), gams a list of L dicts.
    Returns ([(blocks, b, g, traces)] per target, acc).'''
    from socutils.zmr.lib import linalg as la
    from socutils.zmr.reference import ci_gradient as cg
    from socutils.zmr.uc import classes, density
    pclass = classes.CLASS_BY_NAME[name]
    b, mu, t, s, rho_, states = density.model_spaces(pt, pclass, state, rho)
    rhos, _ = density.stack_levels(rho_)
    nl, nj = len(rhos), len(states)
    sp = ctx.sp
    nc, na, nv = sp.ncore, sp.nact, sp.nvirt
    nk = len(mu)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    bvc = numpy.ascontiguousarray(ctx.eris.cd_block('vc'))      # (P, a, i)
    other = 'va' if name == 'cavv' else 'ac'
    bx = numpy.ascontiguousarray(ctx.eris.cd_block(other))      # (P, b, t) or (P, t, j)
    naux = bvc.shape[0]
    left = bvc.reshape(naux, nv * nc)
    right = bx.reshape(naux, -1)
    tts = [numpy.ascontiguousarray(tj.T) for tj in t]
    acc = cg.QAccumulator(pt, pclass, t, s, rho_, states)
    c16 = numpy.complex128
    gj = numpy.zeros((nl, nj, nk, nk), dtype=c16)
    g_vc = numpy.zeros((nl, naux, nv, nc), dtype=c16)
    g_x = numpy.zeros((nl,) + bx.shape, dtype=c16)
    dcc = numpy.zeros((nl, nc, nc), dtype=c16)
    dvv = numpy.zeros((nl, nv, nv), dtype=c16)
    nseg = nv if name == 'cavv' else nc
    # the cavv labels (a, i, b) with all a != b (1/2 each), ccav (a, i, j) all i != j
    fvv, fcc = (1.0, 0.5) if name == 'cavv' else (0.5, 1.0)
    for p0, p1 in density._segments(nseg, nv * nc * max(na, (2 * nl * nj + nj + 1) * nk),
                                    max_memory):
        npart = p1 - p0
        if name == 'cavv':
            bseg = numpy.ascontiguousarray(bx[:, p0:p1]).reshape(naux, npart * na)
            m1 = la.gemm(left, bseg, 'T', 'N').reshape(nv, nc, npart, na)
            m2 = la.gemm(numpy.ascontiguousarray(bvc[:, p0:p1]).reshape(naux, npart * nc),
                         right, 'T', 'N').reshape(npart, nc, nv, na)
            v = numpy.ascontiguousarray(m2.transpose(2, 1, 0, 3) - m1).reshape(-1, na)
            delta = (ev[:, None, None] - ec[None, :, None] + ev[None, None, p0:p1]).reshape(-1)
            sign = 1.0                                               # dE/d(ai|bt) = -W
        else:
            bseg = numpy.ascontiguousarray(bx[:, :, p0:p1])         # (P, t, j)
            m1 = la.gemm(left, bseg.reshape(naux, na * npart), 'T', 'N').reshape(nv, nc, na, npart)
            m2 = la.gemm(numpy.ascontiguousarray(bvc[:, :, p0:p1]).reshape(naux, nv * npart),
                         right, 'T', 'N').reshape(nv, npart, na, nc)
            v = numpy.ascontiguousarray(m1.transpose(0, 1, 3, 2) - m2.transpose(0, 3, 1, 2)
                                        ).reshape(-1, na)            # (a i j, t)
            delta = (ev[:, None, None] - ec[None, :, None] - ec[None, None, p0:p1]).reshape(-1)
            sign = -1.0                                              # dE/d(ai|tj) = W
        del m1, m2
        c, l, r = _pairs(v, delta, mu, tts, s, rhos)
        acc.add(c, l, v, delta, 0.5)
        if name == 'cavv':
            bjt = None
        else:
            bjt = numpy.ascontiguousarray(bseg.transpose(0, 2, 1)).reshape(naux, npart * na)
        for lv in range(nl):
            x = 0
            for j in range(nj):
                gj[lv, j] += 0.5 * la.gemm(c[j], l[lv, j].conj(), 'T', 'N')
                c4 = c[j].reshape(nv, nc, npart, nk)
                l4 = l[lv, j].reshape(nv, nc, npart, nk)
                dvv[lv] += fvv * la.gemm(l4.reshape(nv, -1).conj(), c4.reshape(nv, -1), 'N', 'T')
                dcc[lv] -= fcc * la.gemm(
                    numpy.ascontiguousarray(l4.transpose(1, 0, 2, 3)).reshape(nc, -1).conj(),
                    numpy.ascontiguousarray(c4.transpose(1, 0, 2, 3)).reshape(nc, -1), 'N', 'T')
                x = x + la.gemm(l[lv, j] + r[lv, j], t[j])
            xm = (sign * x).reshape(nv * nc, npart * na)
            if name == 'cavv':
                g_vc[lv] += la.gemm(bseg, xm, 'N', 'T').reshape(naux, nv, nc)
                g_x[lv][:, p0:p1] += la.gemm(left, xm).reshape(naux, npart, na)
            else:
                g_vc[lv] += la.gemm(bjt, xm, 'N', 'T').reshape(naux, nv, nc)
                g_x[lv][:, :, p0:p1] += la.gemm(left, xm).reshape(naux, npart, na).transpose(0, 2, 1)
        del c, l, r, v
    res = []
    for lv in range(nl):
        _add(gams[lv], ('v', 'c'), g_vc[lv])
        _add(gams[lv], ('v', 'a') if name == 'cavv' else ('a', 'c'), g_x[lv])
        g = gj[lv].sum(axis=0)
        traces = [(numpy.trace(y), j) for y, j in zip(gj[lv], states)]
        blocks = {'cc': _herm(dcc[lv]), 'vv': _herm(dvv[lv]),
                  'aa': density._active_block(pt, pclass, b, g, traces)}
        res.append((blocks, b, g, traces))
    return res, acc


def active_two_body(pt, pclass, b, g, state=0, traces=None):
    '''dE/d(tu|vw) of a class through H_act = ... + 1/2 sum (tu|vw) t^+ v^+ w u in the
    first-order space and in the zeroth-order energy, (na,) * 4: 1/2 [sum_m w_m
    Gamma(b y_m) - tr(g) Gamma(Psi)] with g = sum_m w_m y_m y_m^+.'''
    from socutils.zmr.active import rdm as active_rdm
    act = pt.active_space
    ntarget = act.target(pt.nelec, pclass.dn)
    na = act.norb
    w, y = numpy.linalg.eigh(0.5 * (g + g.conj().T))
    keep = abs(w) > 1e-14 * max(1.0, float(abs(w).max()) if len(w) else 1.0)
    d2 = numpy.zeros((na,) * 4, dtype=numpy.complex128)
    if keep.any() and ntarget >= 2:
        vecs = numpy.ascontiguousarray(b @ y[:, keep])
        for m, wm in enumerate(w[keep]):
            d2 += wm * active_rdm.make_rdm(act, vecs[:, m], ntarget, 2, bra=0, ket=0)
    if traces is None:
        traces = [(numpy.trace(g), state)]
    if pt.nelec >= 2:
        for tr, j in traces:
            d2 -= tr * active_rdm.make_rdm(act, pt.civecs[:, [j]], pt.nelec, 2, bra=0, ket=0)
    return 0.5 * d2.real


def derivatives(pt, state=0, max_memory=4000, rho=None):
    '''One pass over the classes of a finished PC-NEVPT2 of one model state: the
    unrelaxed density (quasi-canonical orbitals), the orbital gradient at fixed CI vector
    (quasi-canonical orbitals), and the vector g of the CI gradient, dE2 = 2 <dPsi|g>
    (determinants of the spinor sector; zmr.reference.ci_gradient).  rho: instead of one
    state the target sum_IJ rho_IJ h2_IJ of a multi-state PC-NEVPT2 (union contraction;
    a level: rho_IJ = conj(c_I) c_J), g the vectors g[J] of all states; rho (L, J, J):
    several targets (all levels) in the same pass, a list of the results.  Real orbitals.
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
    levels = rho is not None and numpy.ndim(rho) == 3
    nl = len(rho) if levels else 1

    def per_level(x):
        return list(x) if levels else [x]
    d_gen = [numpy.zeros((n, n), dtype=c16) for _ in range(nl)]
    d_core = [numpy.zeros((n, n), dtype=c16) for _ in range(nl)]
    gams = [{} for _ in range(nl)]
    x_aa = [numpy.zeros((sp.nact,) * 4) for _ in range(nl)]
    nstate = 1 if rho is None else pt.civecs.shape[1]
    g_ci = numpy.zeros((nl, nstate, pt.civecs.shape[0]), dtype=c16)
    _, dcc, dvv, g_ccvv = ccvv(ctx)
    for lv in range(nl):
        d_gen[lv] += density.assemble(sp, {'cc': dcc, 'vv': dvv})
    grad = [g_ccvv.real.copy() for _ in range(nl)]
    if rho is not None:
        # several model states: ccvv also through H_act and the CI vectors
        pclass = classes.CLASS_BY_NAME['ccvv']
        for lv, (d_aa, b, g, traces) in enumerate(per_level(density.ccvv_active(pt, ctx, rho))):
            d_core[lv] += density.assemble(sp, {'aa': d_aa})
            x_aa[lv] += active_two_body(pt, pclass, b, g, traces=traces)
        g_ci += numpy.reshape(cg.class_g(pt, ctx, 'ccvv', rho=rho), g_ci.shape)
    for name in density._LABELS:
        pclass = classes.CLASS_BY_NAME[name]
        if name in ('cavv', 'ccav'):
            res, acc = _external_pass(name, pt, ctx, gams, state, max_memory, rho)
            g_ci += numpy.reshape(acc.g(), g_ci.shape)
            dblks = [x[0] for x in res]
            act = [x[1:] for x in res]
        else:
            dblks = per_level(density.class_reference(pt, ctx, name, state, rho)[1])
            wb = per_level(og._w_blocks(pt, ctx, pclass, state, rho))
            for lv, (blocks, _, _, _) in enumerate(wb):
                gammas_from_antisym(eris, *og._scatter(name, ctx, blocks), gams[lv])
            act = [x[1:] for x in wb]
            g_ci += numpy.reshape(cg.class_g(pt, ctx, name, state, rho), g_ci.shape)
        for lv in range(nl):
            for lab, x in dblks[lv].items():
                target = d_gen[lv] if lab in ('cc', 'vv') else d_core[lv]
                target += density.assemble(sp, {lab: x})
            b, g, traces = act[lv]
            x_aa[lv] += active_two_body(pt, pclass, b, g, state, traces)
    baa = eris.cd_block('aa')
    gam_gen = numpy.zeros((n, n))
    gam_gen[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    gam_gen[sp.slice('a'), sp.slice('a')] = numpy.asarray(pt.gamma).real
    gam_core = numpy.zeros((n, n))
    gam_core[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    out = []
    for lv in range(nl):
        _add(gams[lv], ('a', 'a'), numpy.einsum('tuvw,Pvw->Ptu',
                                                x_aa[lv] + x_aa[lv].transpose(2, 3, 0, 1), baa))
        for (x, y), gam in gams[lv].items():
            grad[lv] += explicit_block(eris, numpy.asarray(gam).real, x, y).real
        # the generalized Fock matrix of the denominators and the core Fock matrix are known
        g_f, w_gen = og.fock_part(eris, d_gen[lv].real, gam_gen, fock=qc.fock, return_w=True)
        grad[lv] += g_f + og.fock_part(eris, d_core[lv].real, gam_core, fock=ctx.fcore)
        if rho is None:
            g_ci[lv, 0] += cg.fock_g(pt, ctx, d_gen[lv].real, state, w=w_gen)
        else:
            g_ci[lv] += cg.fock_g(pt, ctx, d_gen[lv].real, w=w_gen, states=range(nstate))
        out.append({'d': d_gen[lv] + d_core[lv], 'grad': grad[lv],
                    'g_ci': g_ci[lv, 0] if rho is None else g_ci[lv], 'u': numpy.asarray(qc.u)})
    return out if levels else out[0]


def gradient(pt, state=0, max_memory=4000):
    '''(dE2/dK in the quasi-canonical orbitals of pt, u) at fixed CI vector, all classes,
    without MO four-index arrays or virtual-virtual three-index blocks (derivatives).'''
    out = derivatives(pt, state, max_memory)
    return out['grad'], out['u']
