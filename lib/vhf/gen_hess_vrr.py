#!/usr/bin/env python
'''
Generator of lib/vhf/hess_vrr_gen.c: per canonical angular class (la lb | lc ld)
(la >= lb, lc >= ld, l <= LMAX, Cartesian) the second-derivative kernel of
SOCVHFgen_hess2e (jk_gen.c): the energy second derivatives

  h[PQ][x][y] = sum dd_abcd d^2 (ab|cd) / dP_x dQ_y,   PQ = AA, AB, AC, BB, BC, CC,

with dd the symmetrized density weights of the quartet (the caller's), the
fourth centre by translational invariance (the caller's).  With
d/dP_x phi_a = 2 alpha_P phi_(a + 1_x) - a_x phi_(a - 1_x):

  P != Q:  (+1, +1) 4 a_P a_Q,  (+1, -1) 2 a_P,  (-1, +1) 2 a_Q,  (-1, -1) 1;
  P == Q:  4 a_P^2 phi_(a + 1_x + 1_y) - 2 a_P (a_y phi_(a + 1_x - 1_y) + a_x phi_(a - 1_x + 1_y)
           + delta_xy phi_a) + a_x (a - 1_x)_y phi_(a - 1_x - 1_y).

Head-Gordon-Pople as gen_grad_vrr.py, at total order L + 2: per primitive
quartet F_m and the vertical recursions to [e0|f0]^(0) for the union of the
targets of the ten streams (bra weight 1, 2a_A, 2a_B, 4a_A^2, 4a_A a_B, 4a_B^2
times ket weight 1, 2a_C, 4a_C^2, total degree <= 2), the bra contraction per
bra weight on the targets that weight's streams read (the union ordered by that
signature), the ket one per stream; per contraction combination the
horizontal recursions of the shapes and the contraction with dd.

    python gen_hess_vrr.py <dir> [nfiles]     # hess_vrr_gen.h, hess_vrr_gen_<k>.c
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_jk_vrr import comps, band, sub, add, tag   # noqa
from gen_grad_vrr import vrr_plan_targets, emit_hrr, vname   # noqa

LMAX = 2
LSUM_MAX = 1         # classes of total l above are left to the Rys kernel (compile time)

BRAW = ['0', 'A', 'B', 'AA', 'AB', 'BB']
KETW = ['0', 'C', 'CC']
BLOCKS = [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]


def stream_of(raised):
    '''the stream (bra weight, ket weight) of the multiset of raised centres'''
    bw = ''.join(sorted(c for c in raised if c in 'AB')) or '0'
    kw = ''.join(c for c in raised if c == 'C') or '0'
    return bw, kw


def block_terms(P, Q):
    '''terms of the block (P, Q) element (x, y): (sign, stream, shifts {centre: dl},
    {centre: index kind}, coefficient kind); index kinds up_x, dn_x, up_y, dn_y,
    uu, ud, du, self, d2; coefficients pwx (P's a_x), pwy (Q's a_y), pwxy, d, c2'''
    out = []
    if P != Q:
        for sp in (1, -1):
            for sq in (1, -1):
                raised = [c for c, s in ((P, sp), (Q, sq)) if s > 0]
                coef = []
                if sp < 0:
                    coef.append('pwx')
                if sq < 0:
                    coef.append('pwy')
                out.append((sp * sq, stream_of(raised), {P: sp, Q: sq},
                            {P: 'up_x' if sp > 0 else 'dn_x', Q: 'up_y' if sq > 0 else 'dn_y'}, coef))
    else:
        out.append((1, stream_of([P, P]), {P: 2}, {P: 'uu'}, []))
        out.append((-1, stream_of([P]), {P: 0}, {P: 'ud'}, ['pwy']))
        out.append((-1, stream_of([P]), {P: 0}, {P: 'du'}, ['pwx']))
        out.append((-1, stream_of([P]), {P: 0}, {P: 'self'}, ['d']))
        out.append((1, stream_of([]), {P: -2}, {P: 'd2'}, ['c2']))
    return out


def shapes(la, lb, lc, ld):
    '''name -> (stream, la', lb', lc', ld') of every shape the blocks read'''
    l0 = {'A': la, 'B': lb, 'C': lc}
    out = {}
    for P, Q in BLOCKS:
        for sign, st, shift, idx, coef in block_terms(P, Q):
            l = dict(l0)
            for c, s in shift.items():
                l[c] += s
            if min(l.values()) < 0:
                continue
            key = (st, l['A'], l['B'], l['C'], ld)
            if key not in out.values():
                out['S%s_%s_%d%d%d' % (st[0], st[1], l['A'], l['B'], l['C'])] = key
    return out


def shape_name(sh, st, a, b, c, d):
    for nm, k in sh.items():
        if k == (st, a, b, c, d):
            return nm
    return None


def stream_targets(la, lb, lc, ld):
    out = {}
    for nm, (st, a, b, c, d) in shapes(la, lb, lc, ld).items():
        lst = out.setdefault(st, [])
        for e in band(a, a + b):
            for f in band(c, c + d):
                if (e, f) not in lst:
                    lst.append((e, f))
    return out


def kernel(la, lb, lc, ld):
    L = la + lb + lc + ld + 2
    stg = stream_targets(la, lb, lc, ld)
    streams = [(bw, kw) for bw in BRAW for kw in KETW if (bw, kw) in stg]
    # the union ordered by signature (the set of bra weights reading the target)
    sig = {}
    for (bw, kw) in streams:
        for t in stg[(bw, kw)]:
            sig.setdefault(t, set()).add(bw)
    sigs = sorted(set(frozenset(s) for s in sig.values()), key=lambda s: (-len(s), sorted(BRAW.index(w) for w in s)))
    union = []
    for s in sigs:
        union += [t for t in sig if frozenset(sig[t]) == s]
    uidx = {t: i for i, t in enumerate(union)}
    nt = len(union)
    groups = []
    i0 = 0
    for s in sigs:
        n = sum(1 for t in union if frozenset(sig[t]) == s)
        groups.append((i0, i0 + n, [w for w in BRAW if w in s]))
        i0 += n
    braws = [w for w in BRAW if any(w in g[2] for g in groups)]
    order, exprs = vrr_plan_targets(union)
    # stream rows of SK per combination: union index range [lo, hi) of the stream's targets
    srange, soff = {}, {}
    nw = 0
    for s in streams:
        ii = [uidx[t] for t in stg[s]]
        srange[s] = (min(ii), max(ii) + 1)
        soff[s] = nw
        nw += srange[s][1] - srange[s][0]
    fn = 'ghd_%d%d%d%d' % (la, lb, lc, ld)
    o = []
    w = o.append
    w('/* (%d%d|%d%d) second derivatives: %d VRR nodes, %d targets, %d streams */' % (
        la, lb, lc, ld, len(order), nt, len(streams)))
    w('int %s(double *sum, const double *dd, int ngroup, size_t nd, const size_t *ds, const int *pos,' % fn)
    w('               const GPair *B, int bs, const GPair *K, int ks, double cut, const double *tab, double *wk)')
    w('{')
    w('        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;')
    w('        const double *pax = bs ? B->rjx : B->rix, *pay = bs ? B->rjy : B->riy, *paz = bs ? B->rjz : B->riz;')
    w('        const double *qcx = ks ? K->rjx : K->rix, *qcy = ks ? K->rjy : K->riy, *qcz = ks ? K->rjz : K->riz;')
    w('        const double *eA = bs ? B->ej : B->ei, *eB = bs ? B->ei : B->ej, *eC = ks ? K->ej : K->ei;')
    w('        double *v = wk, *Y = v + (size_t)%d * nb;' % nt)
    w('        double *SK = Y + (size_t)%d * ncb, *FT = SK + (size_t)nck * ncb * %d;' % (len(braws) * nt, nw))
    w('        int t, b, cb, kc, i, any = 0;')
    w('        memset(SK, 0, sizeof(double) * nck * ncb * %d);' % nw)
    w('        memset(Y, 0, sizeof(double) * ncb * %d);' % (len(braws) * nt))
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
    # bra contraction per signature group: Y[(w ncb + cb) nt + i] = sum_b c_w[b] v[i][b]
    wexpr = {'0': 'c[b]', 'A': '2 * c[b] * eA[b]', 'B': '2 * c[b] * eB[b]', 'AA': '4 * c[b] * eA[b] * eA[b]',
             'AB': '4 * c[b] * eA[b] * eB[b]', 'BB': '4 * c[b] * eB[b] * eB[b]'}
    widx = {bw: k for k, bw in enumerate(braws)}
    w('                for (cb = 0; cb < ncb; cb++) {')
    w('                        const double *c = B->c + (size_t)cb * B->ld;')
    w('                        double %s;' % ', '.join('c%s[nbt]' % bw for bw in braws))
    w('                        for (b = 0; b < nbt; b++) {')
    for bw in braws:
        w('                                c%s[b] = %s;' % (bw, wexpr[bw]))
    w('                        }')
    for bw in braws:
        w('                        double *y%s = Y + ((size_t)%d * ncb + cb) * %d;' % (bw, widx[bw], nt))
    for lo, hi, ws in groups:
        if hi == lo:
            continue
        w('                        if (nb == 1) {')
        w('                                const double %s;' % ', '.join('k%s = c%s[0]' % (s, s) for s in ws))
        w('#pragma omp simd')
        w('                                for (i = %d; i < %d; i++) {' % (lo, hi))
        w('                                        const double x = v[i];')
        for s in ws:
            w('                                        y%s[i] = k%s * x;' % (s, s))
        w('                                }')
        w('                        } else {')
        w('                                for (i = %d; i < %d; i++) {' % (lo, hi))
        w('                                        const double *vi = v + (size_t)i * nb;')
        w('                                        double %s;' % ', '.join('s%s = 0' % s for s in ws))
        w('                                        for (b = 0; b < nbt; b++) {')
        w('                                                const double x = vi[b];')
        for s in ws:
            w('                                                s%s += c%s[b] * x;' % (s, s))
        w('                                        }')
        for s in ws:
            w('                                        y%s[i] = s%s;' % (s, s))
        w('                                }')
        w('                        }')
    w('                }')
    # ket accumulation per stream
    w('                const double kq = K->kp[t], ac2 = 2 * eC[t];')
    w('                for (kc = 0; kc < nck; kc++) {')
    w('                        const double f0 = K->c[(size_t)kc * K->ld + t] * kq, fC = f0 * ac2, fCC = fC * ac2;')
    w('                        (void)fC; (void)fCC;')
    w('                        for (cb = 0; cb < ncb; cb++) {')
    w('                                double *so = SK + ((size_t)kc * ncb + cb) * %d;' % nw)
    for (bw, kw) in streams:
        lo, hi = srange[(bw, kw)]
        w('                                {')
        w('                                        const double *x = Y + ((size_t)%d * ncb + cb) * %d;' % (widx[bw], nt))
        w('#pragma omp simd')
        w('                                        for (i = %d; i < %d; i++) so[%d + i] += f%s * x[i];' % (
            lo, hi, soff[(bw, kw)] - lo, kw))
        w('                                }')
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
    A, Bc, Cc, Dc = comps(la), comps(lb), comps(lc), comps(ld)
    NA, NB, NC, ND = len(A), len(Bc), len(Cc), len(Dc)
    lcen = {'A': la, 'B': lb, 'C': lc}

    def fl1(tb):
        return ', '.join('{%s}' % ', '.join(str(v) for v in r) for r in tb)

    def fl2(tb):
        return ', '.join('{%s}' % fl1(r) for r in tb)

    def tables(nm, l):
        cs = comps(l)
        cup = comps(l + 1)
        cdn = comps(l - 1) if l > 0 else []
        cuu = comps(l + 2)
        cd2 = comps(l - 2) if l > 1 else []
        up, dn, pw, uu, ud, d2, c2 = [], [], [], [], [], [], []
        for e in cs:
            up.append([cup.index(add(e, x)) for x in range(3)])
            dn.append([cdn.index(sub(e, x)) if e[x] > 0 else 0 for x in range(3)])
            pw.append([e[x] for x in range(3)])
            uu.append([[cuu.index(add(add(e, x), y)) for y in range(3)] for x in range(3)])
            ud.append([[cs.index(sub(add(e, x), y)) if add(e, x)[y] > 0 else 0 for y in range(3)] for x in range(3)])
            r2, rc = [], []
            for x in range(3):
                rr, cc = [], []
                for y in range(3):
                    if e[x] > 0 and sub(e, x)[y] > 0:
                        rr.append(cd2.index(sub(sub(e, x), y)))
                        cc.append(e[x] * sub(e, x)[y])
                    else:
                        rr.append(0)
                        cc.append(0)
                r2.append(rr)
                rc.append(cc)
            d2.append(r2)
            c2.append(rc)
        n = len(cs)
        w('        static const int %s_up[%d][3] = {%s};' % (nm, n, fl1(up)))
        w('        static const int %s_dn[%d][3] = {%s};' % (nm, n, fl1(dn)))
        w('        static const int %s_pw[%d][3] = {%s};' % (nm, n, fl1(pw)))
        w('        static const int %s_uu[%d][3][3] = {%s};' % (nm, n, fl2(uu)))
        w('        static const int %s_ud[%d][3][3] = {%s};' % (nm, n, fl2(ud)))
        w('        static const int %s_d2[%d][3][3] = {%s};' % (nm, n, fl2(d2)))
        w('        static const int %s_c2[%d][3][3] = {%s};' % (nm, n, fl2(c2)))
    tables('tA', la)
    tables('tB', lb)
    tables('tC', lc)
    sh = shapes(la, lb, lc, ld)
    w('        int ci, cj, ck, cl, ia, ib, ic, id, x, y, gi;')
    w('        for (cl = 0; cl < K->ncj; cl++)')
    w('        for (ck = 0; ck < K->nci; ck++)')
    w('        for (cj = 0; cj < B->ncj; cj++)')
    w('        for (ci = 0; ci < B->nci; ci++) {')
    w('                const int cpos[4] = {ci, cj, ck, cl};')
    w('                const double *sk = SK + ((size_t)(cl * K->nci + ck) * ncb + cj * B->nci + ci) * %d;' % nw)
    for s in streams:
        w('                const double *s%s_%s = sk + %d;' % (s[0], s[1], soff[s]))
    for nm, (st, a_, b_, c_, d_) in sh.items():
        lo = srange[st][0]
        sidx = {t: uidx[t] - lo for t in stg[st]}
        emit_hrr(w, 's%s_%s' % st, sidx, a_, b_, c_, d_, nm)
    w('                const size_t base = cpos[pos[0]] * %d * ds[0] + cpos[pos[1]] * %d * ds[1]' % (NA, NB))
    w('                                  + cpos[pos[2]] * %d * ds[2] + cpos[pos[3]] * %d * ds[3];' % (NC, ND))
    w('                for (gi = 0; gi < ngroup; gi++) {')
    w('                        const double *dg = dd + gi * nd + base;')
    w('                        double hs[54] = {0};')
    w('                        for (ic = 0; ic < %d; ic++)' % NC)
    w('                        for (id = 0; id < %d; id++)' % ND)
    w('                        for (ia = 0; ia < %d; ia++)' % NA)
    w('                        for (ib = 0; ib < %d; ib++) {' % NB)
    w('                                const double wd = dg[ia * ds[0] + ib * ds[1] + ic * ds[2] + id * ds[3]];')
    w('                                if (wd == 0) continue;')
    w('                                for (x = 0; x < 3; x++)')
    w('                                for (y = 0; y < 3; y++) {')
    cvar = {'A': 'ia', 'B': 'ib', 'C': 'ic'}
    tnm = {'A': 'tA', 'B': 'tB', 'C': 'tC'}

    def cidx(c, kind, P, Q):
        t, v = tnm[c], cvar[c]
        return {'up_x': '%s_up[%s][x]' % (t, v), 'dn_x': '%s_dn[%s][x]' % (t, v),
                'up_y': '%s_up[%s][y]' % (t, v), 'dn_y': '%s_dn[%s][y]' % (t, v),
                'uu': '%s_uu[%s][x][y]' % (t, v), 'ud': '%s_ud[%s][x][y]' % (t, v),
                'du': '%s_ud[%s][y][x]' % (t, v), 'self': v, 'd2': '%s_d2[%s][x][y]' % (t, v)}[kind]

    for bi, (P, Q) in enumerate(BLOCKS):
        w('                                        double h%d = 0;' % bi)
        for sign, st, shift, idx, coef in block_terms(P, Q):
            l = dict(lcen)
            for c, s_ in shift.items():
                l[c] += s_
            if min(l.values()) < 0:
                continue
            nm = shape_name(sh, st, l['A'], l['B'], l['C'], ld)
            nA, nB, nC = len(comps(l['A'])), len(comps(l['B'])), len(comps(l['C']))
            ii = {c: (cidx(c, idx[c], P, Q) if c in idx else cvar[c]) for c in 'ABC'}
            ref = '%s[(%s * %d + id) * %d + %s * %d + %s]' % (nm, ii['C'], ND, nA * nB, ii['A'], nB, ii['B'])
            cf = []
            for k in coef:
                if k == 'pwx':
                    cf.append('%s_pw[%s][x]' % (tnm[P], cvar[P]))
                elif k == 'pwy':
                    cf.append('%s_pw[%s][y]' % (tnm[Q], cvar[Q]))
                elif k == 'd':
                    cf.append('(x == y)')
                elif k == 'c2':
                    cf.append('%s_c2[%s][x][y]' % (tnm[P], cvar[P]))
            op = '+=' if sign > 0 else '-='
            if cf:
                w('                                        if (%s) h%d %s %s * %s;' % (
                    ' && '.join(cf), bi, op, ' * '.join(cf), ref))
            else:
                w('                                        h%d %s %s;' % (bi, op, ref))
        w('                                        hs[%d + x * 3 + y] += wd * h%d;' % (bi * 9, bi))
    w('                                }')
    w('                        }')
    w('                        for (x = 0; x < 54; x++) sum[gi * 54 + x] += hs[x];')
    w('                }')
    w('        }')
    w('        return 1;')
    w('}')
    return '\n'.join(o), fn, nt, nw, len(braws) * nt, L


def classes():
    pairs = [(a, b) for a in range(LMAX + 1) for b in range(a + 1)]
    return [(a, b, c, d) for (a, b) in pairs for (c, d) in pairs if a + b + c + d <= LSUM_MAX]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    nfiles = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    kern = []
    mx_nt = mx_nw = mx_ny = mx_L = 0
    for cls in classes():
        code, fn, nt, nw, ny, L = kernel(*cls)
        kern.append((cls, fn, code))
        mx_nt, mx_nw, mx_ny, mx_L = max(mx_nt, nt), max(mx_nw, nw), max(mx_ny, ny), max(mx_L, L)
    files = [[] for _ in range(nfiles)]
    size = [0] * nfiles
    for cls, fn, code in sorted(kern, key=lambda k: -len(k[2])):
        k = size.index(min(size))
        files[k].append(code)
        size[k] += len(code)
    for k, codes in enumerate(files):
        with open(os.path.join(outdir, 'hess_vrr_gen_%d.c' % k), 'w') as f:
            f.write('/* Generated by gen_hess_vrr.py (LMAX = %d, LSUM_MAX = %d); do not edit. */\n' % (LMAX, LSUM_MAX))
            f.write('#include "jk_gen.h"\n\n')
            f.write('\n\n'.join(codes) + '\n')
    n = LMAX + 1
    h = ['/* Generated by gen_hess_vrr.py (LMAX = %d); do not edit.  Included by jk_gen.c. */' % LMAX, '']
    for cls, fn, code in kern:
        h.append('int %s(double *sum, const double *dd, int ngroup, size_t nd, const size_t *ds, const int *pos,' % fn)
        h.append('        const GPair *B, int bs, const GPair *K, int ks, double cut, const double *tab, double *wk);')
    h.append('#define GHD_LMAX %d' % LMAX)
    h.append('#define GHD_LSUM_MAX %d   /* classes of larger total l: NULL (the Rys kernel) */' % LSUM_MAX)
    h.append('#define GHD_NT %d       /* the largest union of VRR targets */' % mx_nt)
    h.append('#define GHD_NY %d       /* the largest bra-contracted row (all bra weights) */' % mx_ny)
    h.append('#define GHD_NW %d       /* the largest stream row per combination */' % mx_nw)
    h.append('#define GHD_NF %d       /* the largest Boys order + 1 */' % (mx_L + 1))
    h.append('static const ggd_fn ghd_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn, code in kern:
        h.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    h.append('};')
    with open(os.path.join(outdir, 'hess_vrr_gen.h'), 'w') as f:
        f.write('\n'.join(h) + '\n')


if __name__ == '__main__':
    main()
