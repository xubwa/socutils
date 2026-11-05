#!/usr/bin/env python
'''
Generator of lib/vhf/grad_vrr_gen.c: per canonical angular class (la lb | lc ld)
(la >= lb, lc >= ld, l <= LMAX, Cartesian) the first-derivative kernel of
SOCVHFgen_grad2e (jk_gen.c): the energy derivatives

  g[x][A] = sum dd_abcd ((a + 1_x b|cd)_{2 alpha_A} - a_x (a - 1_x b|cd)),  likewise B, C,

with dd the symmetrized density weights of the quartet (the caller's), the
fourth centre by translational invariance.  Head-Gordon-Pople as
gen_jk_vrr.py, at total order L + 1:

  per primitive quartet: F_m, the vertical recursions to [e0|f0]^(0) for the
  union of the targets of four streams, S0 (weight 1; the (a - 1) / (b - 1) /
  (c - 1) shapes), SA (2 alpha_A; (a + 1)), SB (2 alpha_B; (b + 1)),
  SC (2 alpha_C, on the ket; (c + 1)), each contracted with its weights;

  per contraction combination: the horizontal recursions of the six shapes
  from their streams, and the contraction with dd (loops over the Cartesian
  components with constant tables).

    python gen_grad_vrr.py <dir> [nfiles]     # grad_vrr_gen.h, grad_vrr_gen_<k>.c

The kernels go into nfiles translation units (balanced by VRR size) so that
they compile in parallel; grad_vrr_gen.h has their prototypes and the
dispatch table (included by jk_gen.c).
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_jk_vrr import comps, band, sub, add, first_nz, tag, vname, hrr_plan, X   # noqa

LMAX = 2
LSUM_MAX = 6         # classes of total l above are left to the Rys kernel (compile time)


def vrr_plan_targets(targets):
    '''VRR nodes (e, f, m) for the (e, f, 0) targets, in dependency order'''
    exprs, order = {}, []

    def visit(e, f, m):
        key = (e, f, m)
        if key in exprs:
            return
        if any(f):
            k = first_nz(f)
            f1 = sub(f, k)
            visit(e, f1, m)
            visit(e, f1, m + 1)
            terms = ['QC%s * %s' % (X[k], vname(e, f1, m)), 'WQ%s * %s' % (X[k], vname(e, f1, m + 1))]
            if f[k] >= 2:
                f2 = sub(f, k, 2)
                visit(e, f2, m)
                visit(e, f2, m + 1)
                terms.append('%d * h2q * (%s - rq * %s)' % (f[k] - 1, vname(e, f2, m), vname(e, f2, m + 1)))
            if e[k] >= 1:
                e1 = sub(e, k)
                visit(e1, f1, m + 1)
                terms.append('%d * h2pq * %s' % (e[k], vname(e1, f1, m + 1)))
        elif any(e):
            k = first_nz(e)
            e1 = sub(e, k)
            visit(e1, f, m)
            visit(e1, f, m + 1)
            terms = ['PA%s * %s' % (X[k], vname(e1, f, m)), 'WP%s * %s' % (X[k], vname(e1, f, m + 1))]
            if e[k] >= 2:
                e2 = sub(e, k, 2)
                visit(e2, f, m)
                visit(e2, f, m + 1)
                terms.append('%d * h2p * (%s - rp * %s)' % (e[k] - 1, vname(e2, f, m), vname(e2, f, m + 1)))
        else:
            terms = ['F%d' % m]
        exprs[key] = ' + '.join(terms)
        order.append(key)

    for (e, f) in targets:
        visit(e, f, 0)
    return order, exprs


def shapes(la, lb, lc, ld):
    '''the derivative shapes: name -> (stream, la', lb', lc', ld')'''
    s = {'IA': ('SA', la + 1, lb, lc, ld), 'IB': ('SB', la, lb + 1, lc, ld), 'IC': ('SC', la, lb, lc + 1, ld)}
    if la > 0:
        s['Ia'] = ('S0', la - 1, lb, lc, ld)
    if lb > 0:
        s['Ib'] = ('S0', la, lb - 1, lc, ld)
    if lc > 0:
        s['Ic'] = ('S0', la, lb, lc - 1, ld)
    return s


def stream_targets(la, lb, lc, ld):
    '''per stream the (e, f) the HRR of its shapes read: e of degree la' .. la' + lb',
    f of degree lc' .. lc' + ld' '''
    out = {'S0': [], 'SA': [], 'SB': [], 'SC': []}
    for nm, (st, a, b, c, d) in shapes(la, lb, lc, ld).items():
        for e in band(a, a + b):
            for f in band(c, c + d):
                if (e, f) not in out[st]:
                    out[st].append((e, f))
    return out


def emit_hrr(w, src, sidx, la_, lb_, lc_, ld_, out):
    '''code for the Cartesian (la' lb' | lc' ld') of one combination from the
    stream row src (src[sidx[(e, f)]]) into out[(c ND + d) NAB + a NB + b]'''
    A, Bc, Cc, Dc = comps(la_), comps(lb_), comps(lc_), comps(ld_)
    NA, NB, NC, ND = len(A), len(Bc), len(Cc), len(Dc)
    NAB = NA * NB
    Fb = band(lc_, lc_ + ld_)
    NFT = len(Fb)
    fidx = {f: i for i, f in enumerate(Fb)}
    # bra: HB[(ia NB + ib) NFT + f]
    hb = '%s_hb' % out
    w('                double %s[%d];' % (hb, NAB * NFT))
    if lb_ == 0:
        for ia, a in enumerate(A):
            for jf, f in enumerate(Fb):
                w('                %s[%d] = %s[%d];' % (hb, ia * NFT + jf, src, sidx[(a, f)]))
    else:
        border, bdeps = hrr_plan(la_, lb_, None)
        nm = {}
        for (a, b) in border:
            if bdeps[(a, b)] is None:
                continue
            if sum(a) == la_ and sum(b) == lb_:
                nm[(a, b)] = '%s + %d' % (hb, (A.index(a) * NB + Bc.index(b)) * NFT)
            else:
                nm[(a, b)] = '%s_b%s_%s' % (out, tag(a), tag(b))
                w('                double %s[%d];' % (nm[(a, b)], NFT))

        def ref(n):
            if bdeps[n] is None:
                return None
            return nm[n]
        for (a, b) in border:
            d = bdeps[(a, b)]
            if d is None:
                continue
            k, n1, n2 = d
            for jf, f in enumerate(Fb):
                t1 = '(%s)[%d]' % (ref(n1), jf) if ref(n1) else '%s[%d]' % (src, sidx[(n1[0], f)])
                t2 = '(%s)[%d]' % (ref(n2), jf) if ref(n2) else '%s[%d]' % (src, sidx[(n2[0], f)])
                w('                (%s)[%d] = %s + AB[%d] * %s;' % (nm[(a, b)], jf, t1, k, t2))
    # ket: out[(ic ND + id) NAB + ab]
    w('                double %s[%d];' % (out, NC * ND * NAB))
    if ld_ == 0:
        w('                for (i = 0; i < %d; i++) {' % NAB)
        for ic in range(NC):
            w('                        %s[%d + i] = %s[i * %d + %d];' % (out, ic * NAB, hb, NFT, ic))
        w('                }')
    else:
        korder, kdeps = hrr_plan(lc_, ld_, None)
        kn = {}
        for (c, d_) in korder:
            dd = kdeps[(c, d_)]
            if dd is None:
                kn[(c, d_)] = '%s_k%s' % (out, tag(c))
                w('                double %s[%d];' % (kn[(c, d_)], NAB))
                w('                for (i = 0; i < %d; i++) %s[i] = %s[i * %d + %d];' % (
                    NAB, kn[(c, d_)], hb, NFT, fidx[c]))
                continue
            k, n1, n2 = dd
            if sum(c) == lc_ and sum(d_) == ld_:
                kn[(c, d_)] = '%s + %d' % (out, (Cc.index(c) * ND + Dc.index(d_)) * NAB)
            else:
                kn[(c, d_)] = '%s_k%s_%s' % (out, tag(c), tag(d_))
                w('                double %s[%d];' % (kn[(c, d_)], NAB))
            w('                for (i = 0; i < %d; i++) (%s)[i] = (%s)[i] + CD[%d] * (%s)[i];' % (
                NAB, kn[(c, d_)], kn[n1], k, kn[n2]))


def kernel(la, lb, lc, ld, raw=False):
    '''raw: the derivative integrals out[(slot 3 + x) nd + index] (slots A, B, C of the kernel,
    index by the caller's strides ds) instead of their contraction with dd'''
    L = la + lb + lc + ld + 1
    stg = stream_targets(la, lb, lc, ld)
    # the targets ordered so that those of each bra weight are one interval:
    # X = S0 + SC (weight c), Y = SB (2 c a_B), Z = SA (2 c a_A), Z within Y;
    # groups X\Y, (X^Y)\Z, X^Y^Z, Z\X, Y\(X+Z)
    Xs = []
    for t in stg['S0'] + stg['SC']:
        if t not in Xs:
            Xs.append(t)
    Ys, Zs = list(stg['SB']), list(stg['SA'])
    assert all(t in Ys for t in Zs)
    inX, inY, inZ = set(Xs), set(Ys), set(Zs)
    allt = Xs + [t for t in Ys if t not in inX]
    groups = [[t for t in allt if t in inX and t not in inY],
              [t for t in allt if t in inX and t in inY and t not in inZ],
              [t for t in allt if t in inX and t in inZ],
              [t for t in allt if t in inZ and t not in inX],
              [t for t in allt if t in inY and t not in inX and t not in inZ]]
    union = sum(groups, [])
    assert len(union) == len(allt)
    uidx = {t: i for i, t in enumerate(union)}
    nt = len(union)
    g0 = [0]
    for g in groups:
        g0.append(g0[-1] + len(g))
    nX, nY, nZ = g0[3], g0[5] - g0[1], g0[4] - g0[2]
    xb, yb, zb = 0, g0[1], g0[2]        # interval starts in the union
    order, exprs = vrr_plan_targets(union)
    # per combination the rows X (S0, weight 1), XC (SC, 2 a_C), Z (SA), Y (SB)
    n0, nA, nB, nC = nX, nZ, nY, nX
    nw = 2 * nX + nZ + nY
    sidx = {'S0': {t: uidx[t] - xb for t in stg['S0']}, 'SC': {t: uidx[t] - xb for t in stg['SC']},
            'SA': {t: uidx[t] - zb for t in stg['SA']}, 'SB': {t: uidx[t] - yb for t in stg['SB']}}
    fn = ('ggi_%d%d%d%d' if raw else 'ggd_%d%d%d%d') % (la, lb, lc, ld)
    o = []
    w = o.append
    w('/* (%d%d|%d%d) first derivatives: %d VRR nodes, %d targets */' % (la, lb, lc, ld, len(order), nt))
    if raw:
        w('int %s(double *out, size_t nd, const size_t *ds, const int *pos,' % fn)
    else:
        w('int %s(double *sum, const double *dd, int ngroup, size_t nd, const size_t *ds, const int *pos,' % fn)
    w('               const GPair *B, int bs, const GPair *K, int ks, double cut, const double *tab, double *wk)')
    w('{')
    w('        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;')
    w('        const double *pax = bs ? B->rjx : B->rix, *pay = bs ? B->rjy : B->riy, *paz = bs ? B->rjz : B->riz;')
    w('        const double *qcx = ks ? K->rjx : K->rix, *qcy = ks ? K->rjy : K->riy, *qcz = ks ? K->rjz : K->riz;')
    w('        const double *eA = bs ? B->ej : B->ei, *eB = bs ? B->ei : B->ej, *eC = ks ? K->ej : K->ei;')
    w('        double *v = wk, *w0 = v + (size_t)%d * nb;' % nt)
    w('        double *wA = w0 + (size_t)%d * ncb, *wB = wA + (size_t)%d * ncb, *SK = wB + (size_t)%d * ncb;' % (
        nX, nZ, nY))
    w('        double *FT = SK + (size_t)nck * ncb * %d;' % nw)
    w('        int t, b, cb, kc, i, any = 0;')
    w('        memset(SK, 0, sizeof(double) * nck * ncb * %d);' % nw)
    w('        for (t = 0; t < nk; t++) {')
    w('                if (K->bnd[t] < cut) continue;')
    w('                any = 1;')
    w('                const int nbt = n_above(B->beta, nb, cut * B->bmax / K->bnd[t]);')
    w('                if (nbt == 0) continue;')
    w('                const double q = K->p[t], qx = K->px[t], qy = K->py[t], qz = K->pz[t];')
    w('                const double QCx = qcx[t], QCy = qcy[t], QCz = qcz[t];')
    w('                const double h2q = .5 / q;')
    w('                (void)QCx; (void)QCy; (void)QCz; (void)h2q;')
    w('#pragma omp simd')
    w('                for (b = 0; b < nbt; b++) {')
    w('                        const double p = B->p[b], s = 1. / (p + q);')
    w('                        const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;')
    w('                        const double T = p * q * s * (dx * dx + dy * dy + dz * dz);')
    w('                        const int inr = T < GT_TMAX;')
    w('                        const int k = inr ? (int)(T * (1. / GT_STEP) + .5) : 0;')
    w('                        const double dt = k * GT_STEP - T, *f = tab + k * GT_TN;')
    w('                        const double Ta = inr ? GT_TMAX : T, h = .5 / Ta;')
    w('                        double a = .886226925452758014 / sqrt(Ta);')
    for m in range(L + 1):
        if m > 0:
            w('                        a *= %d * h;' % (2 * m - 1))
        w('                        FT[%d * nb + b] = inr ? taylor(f + %d, dt) : a;' % (m, m))
    w('                }')
    w('#pragma omp simd')
    w('                for (b = 0; b < nbt; b++) {')
    w('                        const double p = B->p[b], s = 1. / (p + q);')
    w('                        const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;')
    w('                        const double rho = p * q * s;')
    for m in range(L + 1):
        w('                        const double F%d = FT[%d * nb + b];' % (m, m))
    w('                        const double PAx = pax[b], PAy = pay[b], PAz = paz[b];')
    w('                        const double WPx = -q * s * dx, WPy = -q * s * dy, WPz = -q * s * dz;')
    w('                        const double WQx = p * s * dx, WQy = p * s * dy, WQz = p * s * dz;')
    w('                        const double h2p = .5 / p, rp = rho / p, rq = rho / q, h2pq = .5 * s;')
    w('                        (void)PAx; (void)PAy; (void)PAz; (void)WPx; (void)WPy; (void)WPz;')
    w('                        (void)WQx; (void)WQy; (void)WQz; (void)h2p; (void)rp; (void)rq; (void)h2pq;')
    w('                        const double g = B->kp[b] * sqrt(s);')
    for key in order:
        w('                        const double %s = %s;' % (vname(*key), exprs[key]))
    for i, (e, f) in enumerate(union):
        w('                        v[%d * nb + b] = g * %s;' % (i, vname(e, f, 0)))
    w('                }')
    # bra contractions over the target groups, v read once per target:
    # w0 = sum_b c v on X, wB = sum_b 2 c a_B v on Y, wA = sum_b 2 c a_A v on Z
    gsums = [('0',), ('0', 'B'), ('0', 'B', 'A'), ('B', 'A'), ('B',)]
    dst = {'0': ('w0', xb, nX), 'A': ('wA', zb, nZ), 'B': ('wB', yb, nY)}
    w('                for (cb = 0; cb < ncb; cb++) {')
    w('                        const double *c = B->c + (size_t)cb * B->ld;')
    w('                        double c0[nbt], cA[nbt], cB[nbt];')
    w('                        for (b = 0; b < nbt; b++) {')
    w('                                c0[b] = c[b];')
    w('                                cA[b] = 2 * c[b] * eA[b];')
    w('                                cB[b] = 2 * c[b] * eB[b];')
    w('                        }')
    w('                        double *y0 = w0 + cb * %d, *yA = wA + cb * %d, *yB = wB + cb * %d;' % (nX, nZ, nY))
    w('                        (void)y0; (void)yA; (void)yB;')
    for gi, sums in enumerate(gsums):
        lo, hi = g0[gi], g0[gi + 1]
        if hi == lo:
            continue
        outs = ['y%s[i - %d]' % (s, dst[s][1]) for s in sums]
        w('                        if (nb == 1) {')
        w('                                const double %s;' % ', '.join('k%s = c%s[0]' % (s, s) for s in sums))
        w('#pragma omp simd')
        w('                                for (i = %d; i < %d; i++) {' % (lo, hi))
        w('                                        const double x = v[i];')
        for s, ot in zip(sums, outs):
            w('                                        %s = k%s * x;' % (ot, s))
        w('                                }')
        w('                        } else {')
        w('                                for (i = %d; i < %d; i++) {' % (lo, hi))
        w('                                        const double *vi = v + (size_t)i * nb;')
        w('                                        double %s;' % ', '.join('s%s = 0' % s for s in sums))
        w('                                        for (b = 0; b < nbt; b++) {')
        w('                                                const double x = vi[b];')
        for s in sums:
            w('                                                s%s += c%s[b] * x;' % (s, s))
        w('                                        }')
        for s, ot in zip(sums, outs):
            w('                                        %s = s%s;' % (ot, s))
        w('                                }')
        w('                        }')
    w('                }')
    # ket accumulation: SK[(kc ncb + cb) nw + ...] rows X (S0) | XC (SC) | Z (SA) | Y (SB)
    w('                const double kq = K->kp[t], ac2 = 2 * eC[t];')
    w('                for (kc = 0; kc < nck; kc++) {')
    w('                        const double f = K->c[(size_t)kc * K->ld + t] * kq, fc = f * ac2;')
    w('                        for (cb = 0; cb < ncb; cb++) {')
    w('                                double *so = SK + ((size_t)kc * ncb + cb) * %d;' % nw)
    w('                                const double *x0 = w0 + cb * %d, *xa = wA + cb * %d, *xb = wB + cb * %d;' % (
        nX, nZ, nY))
    w('#pragma omp simd')
    w('                                for (i = 0; i < %d; i++) {' % nX)
    w('                                        so[i] += f * x0[i];')
    w('                                        so[%d + i] += fc * x0[i];' % nX)
    w('                                }')
    w('#pragma omp simd')
    w('                                for (i = 0; i < %d; i++) so[%d + i] += f * xa[i];' % (nZ, 2 * nX))
    w('#pragma omp simd')
    w('                                for (i = 0; i < %d; i++) so[%d + i] += f * xb[i];' % (nY, 2 * nX + nZ))
    w('                        }')
    w('                }')
    w('        }')
    w('        if (!any) return 0;')
    w('        double AB[3], CD[3];')
    w('        for (i = 0; i < 3; i++) {')
    w('                AB[i] = bs ? -B->rij[i] : B->rij[i];')
    w('                CD[i] = ks ? -K->rij[i] : K->rij[i];')
    w('        }')
    w('        (void)AB; (void)CD;')
    # tables of the Cartesian components
    A, Bc, Cc, Dc = comps(la), comps(lb), comps(lc), comps(ld)
    NA, NB, NC, ND = len(A), len(Bc), len(Cc), len(Dc)

    def tables(nm, cs, l):
        up, dn, pw = [], [], []
        cup, cdn = comps(l + 1), comps(l - 1) if l > 0 else []
        for e in cs:
            up.append([cup.index(add(e, x)) for x in range(3)])
            dn.append([cdn.index(sub(e, x)) if e[x] > 0 else 0 for x in range(3)])
            pw.append([e[x] for x in range(3)])
        fl = lambda tb: ', '.join('{%s}' % ', '.join(str(v) for v in r) for r in tb)
        w('        static const int %s_up[%d][3] = {%s};' % (nm, len(cs), fl(up)))
        w('        static const int %s_dn[%d][3] = {%s};' % (nm, len(cs), fl(dn)))
        w('        static const int %s_pw[%d][3] = {%s};' % (nm, len(cs), fl(pw)))
    tables('tA', A, la)
    tables('tB', Bc, lb)
    tables('tC', Cc, lc)
    sh = shapes(la, lb, lc, ld)
    w('        int ci, cj, ck, cl, ia, ib, ic, id, x, gi;')
    w('        for (cl = 0; cl < K->ncj; cl++)')
    w('        for (ck = 0; ck < K->nci; ck++)')
    w('        for (cj = 0; cj < B->ncj; cj++)')
    w('        for (ci = 0; ci < B->nci; ci++) {')
    w('                const int cpos[4] = {ci, cj, ck, cl};')
    w('                const double *sk = SK + ((size_t)(cl * K->nci + ck) * ncb + cj * B->nci + ci) * %d;' % nw)
    w('                const double *s0 = sk, *sC = sk + %d, *sA = sk + %d, *sB = sk + %d;' % (
        nX, 2 * nX, 2 * nX + nZ))
    w('                (void)s0;')
    srcname = {'S0': 's0', 'SA': 'sA', 'SB': 'sB', 'SC': 'sC'}
    for nm, (st, a_, b_, c_, d_) in sh.items():
        emit_hrr(w, srcname[st], sidx[st], a_, b_, c_, d_, nm)
    w('                const size_t base = cpos[pos[0]] * %d * ds[0] + cpos[pos[1]] * %d * ds[1]' % (NA, NB))
    w('                                  + cpos[pos[2]] * %d * ds[2] + cpos[pos[3]] * %d * ds[3];' % (NC, ND))
    # shape strides: X[(ic ND' + id) NAB' + ia NB' + ib]
    nAp, nBp, nCp = len(comps(la + 1)), len(comps(lb + 1)), len(comps(lc + 1))
    nAm = len(comps(la - 1)) if la > 0 else 0
    nBm = len(comps(lb - 1)) if lb > 0 else 0
    nCm = len(comps(lc - 1)) if lc > 0 else 0
    # profiling builds: -DGGD_PROF=1 the HRR kept, the contraction replaced by a plain sum;
    # -DGGD_PROF=2 the vertical recursions only
    if raw:
        w('                for (ic = 0; ic < %d; ic++)' % NC)
        w('                for (id = 0; id < %d; id++)' % ND)
        w('                for (ia = 0; ia < %d; ia++)' % NA)
        w('                for (ib = 0; ib < %d; ib++) {' % NB)
        w('                        const size_t o = base + ia * ds[0] + ib * ds[1] + ic * ds[2] + id * ds[3];')
        w('                        for (x = 0; x < 3; x++) {')
        w('                                double vA = IA[(ic * %d + id) * %d + tA_up[ia][x] * %d + ib];' % (ND, nAp * NB, NB))
        if la > 0:
            w('                                if (tA_pw[ia][x]) vA -= tA_pw[ia][x] * Ia[(ic * %d + id) * %d + tA_dn[ia][x] * %d + ib];' % (
                ND, nAm * NB, NB))
        w('                                double vB = IB[(ic * %d + id) * %d + ia * %d + tB_up[ib][x]];' % (ND, NA * nBp, nBp))
        if lb > 0:
            w('                                if (tB_pw[ib][x]) vB -= tB_pw[ib][x] * Ib[(ic * %d + id) * %d + ia * %d + tB_dn[ib][x]];' % (
                ND, NA * nBm, nBm))
        w('                                double vC = IC[(tC_up[ic][x] * %d + id) * %d + ia * %d + ib];' % (ND, NA * NB, NB))
        if lc > 0:
            w('                                if (tC_pw[ic][x]) vC -= tC_pw[ic][x] * Ic[(tC_dn[ic][x] * %d + id) * %d + ia * %d + ib];' % (
                ND, NA * NB, NB))
        w('                                out[x * nd + o] = vA;')
        w('                                out[(3 + x) * nd + o] = vB;')
        w('                                out[(6 + x) * nd + o] = vC;')
        w('                        }')
        w('                }')
        w('        }')
        w('        return 1;')
        w('}')
        code = '\n'.join(o)
        return code.replace('int ci, cj, ck, cl, ia, ib, ic, id, x, gi;', 'int ci, cj, ck, cl, ia, ib, ic, id, x;'), fn, nt, nw
    w('#if defined(GGD_PROF) && GGD_PROF == 1')
    shp = [nm for nm in sh]
    w('                { double s_ = 0;')
    for nm in shp:
        st_, a_, b_, c_, d_ = sh[nm]
        n_ = len(comps(a_)) * len(comps(b_)) * len(comps(c_)) * len(comps(d_))
        w('                  for (i = 0; i < %d; i++) s_ += %s[i];' % (n_, nm))
    w('                  sum[0] += s_ * 1e-300; }')
    w('                continue;')
    w('#elif defined(GGD_PROF) && GGD_PROF == 2')
    w('                sum[0] += sk[0] * 1e-300;')
    w('                continue;')
    w('#endif')
    w('                for (gi = 0; gi < ngroup; gi++) {')
    w('                        const double *dg = dd + gi * nd + base;')
    w('                        double gs[9] = {0};')
    w('                        for (ic = 0; ic < %d; ic++)' % NC)
    w('                        for (id = 0; id < %d; id++)' % ND)
    w('                        for (ia = 0; ia < %d; ia++)' % NA)
    w('                        for (ib = 0; ib < %d; ib++) {' % NB)
    w('                                const double wd = dg[ia * ds[0] + ib * ds[1] + ic * ds[2] + id * ds[3]];')
    w('                                if (wd == 0) continue;')
    w('                                for (x = 0; x < 3; x++) {')
    # A
    w('                                        double vA = IA[(ic * %d + id) * %d + tA_up[ia][x] * %d + ib];' % (
        ND, nAp * NB, NB))
    if la > 0:
        w('                                        if (tA_pw[ia][x]) vA -= tA_pw[ia][x] * Ia[(ic * %d + id) * %d + tA_dn[ia][x] * %d + ib];' % (
            ND, nAm * NB, NB))
    w('                                        double vB = IB[(ic * %d + id) * %d + ia * %d + tB_up[ib][x]];' % (
        ND, NA * nBp, nBp))
    if lb > 0:
        w('                                        if (tB_pw[ib][x]) vB -= tB_pw[ib][x] * Ib[(ic * %d + id) * %d + ia * %d + tB_dn[ib][x]];' % (
            ND, NA * nBm, nBm))
    w('                                        double vC = IC[(tC_up[ic][x] * %d + id) * %d + ia * %d + ib];' % (
        ND, NA * NB, NB))
    if lc > 0:
        w('                                        if (tC_pw[ic][x]) vC -= tC_pw[ic][x] * Ic[(tC_dn[ic][x] * %d + id) * %d + ia * %d + ib];' % (
            ND, NA * NB, NB))
    w('                                        gs[x * 3] += wd * vA;')
    w('                                        gs[x * 3 + 1] += wd * vB;')
    w('                                        gs[x * 3 + 2] += wd * vC;')
    w('                                }')
    w('                        }')
    w('                        for (x = 0; x < 9; x++) sum[gi * 9 + x] += gs[x];')
    w('                }')
    w('        }')
    w('        (void)nCp; (void)nCm;')
    w('        return 1;')
    w('}')
    code = '\n'.join(o)
    code = code.replace("        (void)nCp; (void)nCm;\n", '')
    return code, fn, nt, nw


def classes():
    pairs = [(a, b) for a in range(LMAX + 1) for b in range(a + 1)]
    return [(a, b, c, d) for (a, b) in pairs for (c, d) in pairs if a + b + c + d <= LSUM_MAX]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    nfiles = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    kern = []
    mx_nt = mx_nw = 0
    for cls in classes():
        for raw in (False, True):
            code, fn, nt, nw = kernel(*cls, raw=raw)
            kern.append((cls, fn, code))
            mx_nt, mx_nw = max(mx_nt, nt), max(mx_nw, nw)
    # balance the files by code size, largest first
    files = [[] for _ in range(nfiles)]
    size = [0] * nfiles
    for cls, fn, code in sorted(kern, key=lambda k: -len(k[2])):
        k = size.index(min(size))
        files[k].append(code)
        size[k] += len(code)
    for k, codes in enumerate(files):
        with open(os.path.join(outdir, 'grad_vrr_gen_%d.c' % k), 'w') as f:
            f.write('/* Generated by gen_grad_vrr.py (LMAX = %d); do not edit. */\n' % LMAX)
            f.write('#include "jk_gen.h"\n\n')
            f.write('\n\n'.join(codes) + '\n')
    n = LMAX + 1
    h = ['/* Generated by gen_grad_vrr.py (LMAX = %d); do not edit.  Included by jk_gen.c. */' % LMAX, '']
    for cls, fn, code in kern:
        if fn.startswith('ggi'):
            h.append('int %s(double *out, size_t nd, const size_t *ds, const int *pos,' % fn)
        else:
            h.append('int %s(double *sum, const double *dd, int ngroup, size_t nd, const size_t *ds, const int *pos,' % fn)
        h.append('        const GPair *B, int bs, const GPair *K, int ks, double cut, const double *tab, double *wk);')
    h.append('typedef int (*ggd_fn)(double *, const double *, int, size_t, const size_t *, const int *,')
    h.append('                      const GPair *, int, const GPair *, int, double, const double *, double *);')
    h.append('#define GGD_LMAX %d' % LMAX)
    h.append('#define GGD_LSUM_MAX %d   /* classes of larger total l: NULL (the Rys kernel) */' % LSUM_MAX)
    h.append('#define GGD_NT %d       /* the largest union of VRR targets */' % mx_nt)
    h.append('#define GGD_NW %d       /* the largest stream row per combination */' % mx_nw)
    h.append('static const ggd_fn ggd_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn, code in kern:
        if fn.startswith('ggd'):
            h.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    h.append('};')
    h.append('typedef int (*ggi_fn)(double *, size_t, const size_t *, const int *,')
    h.append('                      const GPair *, int, const GPair *, int, double, const double *, double *);')
    h.append('static const ggi_fn ggi_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn, code in kern:
        if fn.startswith('ggi'):
            h.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    h.append('};')
    with open(os.path.join(outdir, 'grad_vrr_gen.h'), 'w') as f:
        f.write('\n'.join(h) + '\n')


if __name__ == '__main__':
    main()
