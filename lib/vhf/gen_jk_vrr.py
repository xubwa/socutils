#!/usr/bin/env python
'''
Generator of lib/vhf/jk_vrr_gen.c: per canonical angular class (la lb | lc ld)
(la >= lb, lc >= ld, l <= LMAX) one complete Head-Gordon-Pople kernel of
jk_gen.c, unrolled into straight-line C:

  per primitive quartet: F_m(T), m <= L (Taylor tables), the vertical
  recursions [e0|00]^(m) (Obara-Saika on A) and their transfer to the ket
  [e0|f0]^(m) (on C), only the (e, f, m) the targets need (deg e in la ..
  la + lb, deg f in lc .. lc + ld, m = 0), in dependency order; the bra
  primitives as the inner loop; the bra then the ket contraction into
  SK[(kc * ncb + cb) * NT + e * NFT + f];

  per contraction combination: the bra horizontal recursion (a, b + 1_k) =
  (a + 1_k, b) + AB_k (a, b) as loops of constant length over the ket
  components, the ket one over the bra pairs, the Cartesian-to-spherical
  transform of the d shells with libcint's coefficients as straight-line
  sums, the results written at the role strides st[] (scaled by fac).

A = the bra shell of the higher l (the pair's shells swapped if needed),
likewise C; PA = P - R_A, QC = Q - R_C, AB = R_A - R_B, CD = R_C - R_D.

    python gen_jk_vrr.py > jk_vrr_gen.c
'''
import sys

LMAX = 3            # shells up to f
LMAX_ALL = 2        # every class of l <= LMAX_ALL; with f shells only those of total l <= F_LSUM_MAX
F_LSUM_MAX = 6      # (above: libcint; at total l 7 the generated ones only tie it, caffeine / cc-pVTZ)
NFFILES = 8         # translation units of the f kernels (jk_vrr_f_<k>.c, compiled in parallel)
X = 'xyz'
# libcint's Cartesian-to-spherical coefficients (xx xy xz yy yz zz -> xy yz z2 xz x2-y2)
C2S = {0: [[1.]], 1: [[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
       2: [[0, 1.092548430592079070, 0, 0, 0, 0],
           [0, 0, 0, 0, 1.092548430592079070, 0],
           [-0.315391565252520002, 0, 0, -0.315391565252520002, 0, 0.630783130505040012],
           [0, 0, 1.092548430592079070, 0, 0, 0],
           [0.546274215296039535, 0, 0, -0.546274215296039535, 0, 0]],
       # xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz (pyscf.gto.cart2sph(3), libcint's)
       3: [[0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -0.5900435899266435, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0],
           [0.0, 0.0, -1.1195289977703462, 0.0, 0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 0.7463526651802308],
           [-0.4570457994644657, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 1.4453057213202771, 0.0, 0.0, 0.0, 0.0, -1.4453057213202771, 0.0, 0.0],
           [0.5900435899266435, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}


def comps(d):
    return [(x, y, d - x - y) for x in range(d, -1, -1) for y in range(d - x, -1, -1)]


def band(lo, hi):
    out = []
    for d in range(lo, hi + 1):
        out += comps(d)
    return out


def sub(e, k, n=1):
    e = list(e)
    e[k] -= n
    return tuple(e)


def add(e, k):
    e = list(e)
    e[k] += 1
    return tuple(e)


def boys_lines(L, ind):
    '''C lines (indent ind) defining F0 .. FL = F_m(T) of the double T in scope from the
    tables tab (jk_gen.c): F_L by its Taylor series at the nearest grid point, e^-T =
    e^-T_k e^dt, then downwards (stable), F_m = (2 T F_(m+1) + e^-T) / (2 m + 1); the
    asymptotic form for T >= GT_TMAX.  Branch-free, the table read by indexed loads and
    selects, which GCC vectorises (with gathers) given -fno-math-errno -fno-trapping-math.'''
    o = []
    w = o.append
    w(ind + 'const int inr = T < GT_TMAX;')
    w(ind + 'const int k = inr ? (int)(T * (1. / GT_STEP) + .5) : 0;')
    w(ind + 'const int o = k * GT_TN;')
    w(ind + 'const double dt = inr ? k * GT_STEP - T : 0;')
    w(ind + 'const double Ta = inr ? GT_TMAX : T, h = .5 / Ta;')
    w(ind + 'const double a0 = .886226925452758014 / sqrt(Ta);')
    for m in range(1, L + 1):
        w(ind + 'const double a%d = a%d * (%d * h);' % (m, m - 1, 2 * m - 1))
    t = 'tab[o + %d]' % (L + 6)
    for j in range(6, 0, -1):
        t = 'tab[o + %d] + dt * (1. / %d) * (%s)' % (L + j - 1, j, t)
    w(ind + 'const double f%d = %s;' % (L, t))
    if L > 0:
        w(ind + 'const double ex = tab[GT_N * GT_TN + k] * (1 + dt * (1 + dt * (1. / 2) * (1 + dt * (1. / 3)'
          ' * (1 + dt * (1. / 4) * (1 + dt * (1. / 5) * (1 + dt * (1. / 6)))))));')
        w(ind + 'const double T2 = 2 * T;')
        for m in range(L - 1, -1, -1):
            w(ind + 'const double f%d = (T2 * f%d + ex) * (1. / %d);' % (m, m + 1, 2 * m + 1))
    for m in range(L + 1):
        w(ind + 'const double F%d = inr ? f%d : a%d;' % (m, m, m))
    return o


def first_nz(e):
    return next(k for k in range(3) if e[k])


def tag(e):
    return '%d%d%d' % e


def vname(e, f, m):
    return 'x%s_%s_%d' % (tag(e), tag(f), m)


def vrr_plan(la, lb, lc, ld):
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

    E, Fb = band(la, la + lb), band(lc, lc + ld)
    for e in E:
        for f in Fb:
            visit(e, f, 0)
    return order, exprs, E, Fb


def hrr_plan(l1, l2, base_of):
    '''nodes (a, b), deg a >= l1, b of degree <= l2, for the targets deg a = l1, deg b = l2;
    returns (order, deps) with deps[(a, b)] = (k, (a + 1_k, b - 1_k), (a, b - 1_k)) or None (base)'''
    deps, order = {}, []

    def visit(a, b):
        if (a, b) in deps:
            return
        if not any(b):
            deps[(a, b)] = None
        else:
            k = first_nz(b)
            b1 = sub(b, k)
            visit(add(a, k), b1)
            visit(a, b1)
            deps[(a, b)] = (k, (add(a, k), b1), (a, b1))
        order.append((a, b))

    for a in comps(l1):
        for b in comps(l2):
            visit(a, b)
    return order, deps


def hrr_fwd_lines(ind, l1, l2, base, AB, tgt, pre='h'):
    '''scalar forward HRR (a, b + 1_k) = (a + 1_k, b) + AB_k (a, b): base(a) the C expression of
    (a, 0), the results tgt(a, b) for deg a = l1, deg b = l2 (C lvalue)'''
    order, deps = hrr_plan(l1, l2, None)
    o = []
    nm = {}
    for (a, b) in order:
        d = deps[(a, b)]
        if d is None:
            nm[(a, b)] = base(a)
            continue
        k, n1, n2 = d
        if sum(a) == l1 and sum(b) == l2:
            nm[(a, b)] = tgt(a, b)
            o.append(ind + '%s = %s + %s[%d] * %s;' % (nm[(a, b)], nm[n1], AB, k, nm[n2]))
        else:
            nm[(a, b)] = '%s%s_%s' % (pre, tag(a), tag(b))
            o.append(ind + 'const double %s = %s + %s[%d] * %s;' % (nm[(a, b)], nm[n1], AB, k, nm[n2]))
    return o


def hrr_adj_lines(ind, l1, l2, src, AB, base, pre='g'):
    '''the adjoint of hrr_fwd_lines: src(a, b) the C expression of the target weights (deg a = l1,
    deg b = l2); base(a) the C lvalue receiving the weight of (a, 0) (assigned)'''
    order, deps = hrr_plan(l1, l2, None)
    o = []
    nm = {(a, b): '%s%s_%s' % (pre, tag(a), tag(b)) for (a, b) in order}
    for (a, b) in order:
        if sum(a) == l1 and sum(b) == l2:
            o.append(ind + 'double %s = %s;' % (nm[(a, b)], src(a, b)))
        else:
            o.append(ind + 'double %s = 0;' % nm[(a, b)])
    for (a, b) in reversed(order):
        d = deps[(a, b)]
        if d is None:
            continue
        k, n1, n2 = d
        o.append(ind + '%s += %s;' % (nm[n1], nm[(a, b)]))
        o.append(ind + '%s += %s[%d] * %s;' % (nm[n2], AB, k, nm[(a, b)]))
    for (a, b) in order:
        if deps[(a, b)] is None:
            o.append(ind + '%s = %s;' % (base(a), nm[(a, b)]))
    return o


def jpost_lines(la, lb, lc, ld, E, Fb, eidx, fidx):
    '''J-only post part (names B, bs, K, ks, SK, db, dk, jb, jk, DI, DK, fac in scope): per contraction
    combination the ket density (D_cd) taken through the transposed c2s and ket HRR to the f band,
    W_e = sum_f [e|f] Dk_f, then the bra HRR and c2s of W into jb (J_ab += fac (ab|cd) D_cd); the same
    the other way into jk.  db / jb (DJ, DI) and dk / jk (DL, DK): the pairs' blocks, index i (k) fastest.'''
    NET, NFT = len(E), len(Fb)
    nt = NET * NFT
    A, Bc, Cc, Dc = comps(la), comps(lb), comps(lc), comps(ld)
    cA, cB, cC, cD = C2S[la], C2S[lb], C2S[lc], C2S[ld]
    o = []
    w = o.append
    I = ' ' * 16
    w('        double AB[3], CD[3];')
    w('        for (i = 0; i < 3; i++) {')
    w('                AB[i] = bs ? -B->rij[i] : B->rij[i];')
    w('                CD[i] = ks ? -K->rij[i] : K->rij[i];')
    w('        }')
    w('        (void)AB; (void)CD;')
    # element (m_A, m_B) of the bra block at mA * sA + mB * sB from the combination's base
    w('        const size_t sA = bs ? DI : 1, sB = bs ? 1 : DI, sC = ks ? DK : 1, sD = ks ? 1 : DK;')
    w('        int ci, cj, ck, cl, e, f;')
    w('        for (cl = 0; cl < K->ncj; cl++)')
    w('        for (ck = 0; ck < K->nci; ck++)')
    w('        for (cj = 0; cj < B->ncj; cj++)')
    w('        for (ci = 0; ci < B->nci; ci++) {')
    w('                const double *sk = SK + ((size_t)(cl * K->nci + ck) * ncb + cj * B->nci + ci) * %d;' % nt)
    w('                const size_t ob = ci * (2 * B->li + 1) + cj * (2 * B->lj + 1) * DI;')
    w('                const size_t ok = ck * (2 * K->li + 1) + cl * (2 * K->lj + 1) * DK;')
    w('                const double *dbb = db + ob, *dkb = dk + ok;')
    w('                double *jbb = jb + ob, *jkb = jk + ok;')
    w('                double Dk[%d] = {0}, Db[%d] = {0}, W[%d], V[%d];' % (NFT, NET, NET, NFT))
    # ket density: cartesian dc[ic][id] = sum_mc,md C2S_c[mc][ic] C2S_d[md][id] D[mc][md]
    for side, (l1, l2, c1, c2, s1, s2, src, dst, X1, X2, band, idx, ABn) in enumerate([
            (lc, ld, cC, cD, 'sC', 'sD', 'dkb', 'Dk', Cc, Dc, Fb, fidx, 'CD'),
            (la, lb, cA, cB, 'sA', 'sB', 'dbb', 'Db', A, Bc, E, eidx, 'AB')]):
        def cart(x1, x2, c1=c1, c2=c2, s1=s1, s2=s2, src=src, X1=X1, X2=X2):
            i1, i2 = X1.index(x1), X2.index(x2)
            t = []
            for m1 in range(len(c1)):
                for m2 in range(len(c2)):
                    cf = c1[m1][i1] * c2[m2][i2]
                    if cf != 0:
                        t.append('%.18g * %s[%d * %s + %d * %s]' % (cf, src, m1, s1, m2, s2))
            return '(%s)' % (' + '.join(t) if t else '0')
        if l2 == 0:
            for x1 in X1:
                w(I + '%s[%d] = %s;' % (dst, idx[x1], cart(x1, (0, 0, 0))))
        else:
            for line in hrr_adj_lines(I, l1, l2, cart, ABn, lambda a, dst=dst, idx=idx: '%s[%d]' % (dst, idx[a]),
                                      pre='g%d_' % side):
                w(line)
    w(I + 'for (e = 0; e < %d; e++) {' % NET)
    w(I + '        double y = 0;')
    w(I + '        for (f = 0; f < %d; f++) y += sk[e * %d + f] * Dk[f];' % (NFT, NFT))
    w(I + '        W[e] = fac * y;')
    w(I + '}')
    w(I + 'for (f = 0; f < %d; f++) {' % NFT)
    w(I + '        double y = 0;')
    w(I + '        for (e = 0; e < %d; e++) y += Db[e] * sk[e * %d + f];' % (NET, NFT))
    w(I + '        V[f] = fac * y;')
    w(I + '}')
    # forward HRR + c2s into the J blocks
    for (l1, l2, c1, c2, s1, s2, src, dst, X1, X2, idx, ABn) in [
            (la, lb, cA, cB, 'sA', 'sB', 'W', 'jbb', A, Bc, eidx, 'AB'),
            (lc, ld, cC, cD, 'sC', 'sD', 'V', 'jkb', Cc, Dc, fidx, 'CD')]:
        tgt = {}
        if l2 == 0:
            for x1 in X1:
                tgt[(x1, (0, 0, 0))] = '%s[%d]' % (src, idx[x1])
        else:
            nmz = lambda a, b, ABn=ABn: 't%s%s_%s' % (ABn, tag(a), tag(b))
            for x1 in X1:
                for x2 in X2:
                    w(I + 'double %s;' % nmz(x1, x2))
                    tgt[(x1, x2)] = nmz(x1, x2)
            for line in hrr_fwd_lines(I, l1, l2, lambda a, src=src, idx=idx: '%s[%d]' % (src, idx[a]), ABn,
                                      nmz, pre='h%s' % ABn):
                w(line)
        for m1 in range(len(c1)):
            for m2 in range(len(c2)):
                t = []
                for x1 in X1:
                    for x2 in X2:
                        cf = c1[m1][X1.index(x1)] * c2[m2][X2.index(x2)]
                        if cf != 0:
                            t.append('%.18g * %s' % (cf, tgt[(x1, x2)]))
                w(I + '%s[%d * %s + %d * %s] += %s;' % (dst, m1, s1, m2, s2, ' + '.join(t)))
    w('        }')
    return o


def kernel(la, lb, lc, ld, jonly=False, storage='static int'):
    order, exprs, E, Fb = vrr_plan(la, lb, lc, ld)
    L = la + lb + lc + ld
    NET, NFT = len(E), len(Fb)
    nt = NET * NFT
    eidx = {e: i for i, e in enumerate(E)}
    fidx = {f: i for i, f in enumerate(Fb)}
    A, Bc, Cc, Dc = comps(la), comps(lb), comps(lc), comps(ld)
    NA, NB, NC, ND = len(A), len(Bc), len(Cc), len(Dc)
    NAB = NA * NB
    fn = ('jjg_%d%d%d%d' if jonly else 'jkg_%d%d%d%d') % (la, lb, lc, ld)
    o = []
    w = o.append
    w('/* (%d%d|%d%d)%s: %d VRR nodes, %d targets */' % (la, lb, lc, ld, ' J only' if jonly else '', len(order), nt))
    if jonly:
        w('static int %s(double *jb, double *jk, const double *db, const double *dk, size_t DI, size_t DK,' % fn)
        w('                const GPair *B, int bs, const GPair *K, int ks, double cut, double fac,')
        w('                const double *tab, double *wk)')
    else:
        w('%s %s(double *out, const size_t *st, const int *pos, const GPair *B, int bs,' % (storage, fn))
        w('                const GPair *K, int ks, double cut, double fac, const double *tab, double *wk)')
    w('{')
    w('        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;')
    w('        const double *pax = bs ? B->rjx : B->rix, *pay = bs ? B->rjy : B->riy, *paz = bs ? B->rjz : B->riz;')
    w('        const double *qcx = ks ? K->rjx : K->rix, *qcy = ks ? K->rjy : K->riy, *qcz = ks ? K->rjz : K->riz;')
    w('        double *v = wk, *w = v + (size_t)%d * nb, *SK = w + (size_t)%d * ncb;' % (nt, nt))
    w('        double *FT = SK + (size_t)nck * ncb * %d;     /* F_m of the bra primitives, FT[m * nb + b] */' % nt)
    w('        int t, b, cb, kc, i, any = 0;')
    w('        memset(SK, 0, sizeof(double) * nck * ncb * %d);' % nt)
    w('        for (t = 0; t < nk; t++) {')
    w('                if (K->bnd[t] < cut) continue;')
    w('                any = 1;')
    w('                const int nbt = n_above(B->beta, nb, cut * B->bmax / K->bnd[t]);')
    w('                const double q = K->p[t], qx = K->px[t], qy = K->py[t], qz = K->pz[t];')
    w('                const double QCx = qcx[t], QCy = qcy[t], QCz = qcz[t];')
    w('                const double h2q = .5 / q;')
    w('                (void)QCx; (void)QCy; (void)QCz; (void)h2q;')
    # pass 1: T and F_0 .. F_L of every bra primitive, branch-free (vectorised with gathers)
    w('#pragma omp simd')
    w('                for (b = 0; b < nbt; b++) {')
    w('                        const double p = B->p[b], s = 1. / (p + q);')
    w('                        const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;')
    w('                        const double T = p * q * s * (dx * dx + dy * dy + dz * dz);')
    for line in boys_lines(L, ' ' * 24):
        w(line)
    for m in range(L + 1):
        w('                        FT[%d * nb + b] = F%d;' % (m, m))
    w('                }')
    # pass 2: the vertical recursions
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
    i = 0
    for e in E:
        for f in Fb:
            w('                        v[%d * nb + b] = g * %s;' % (i, vname(e, f, 0)))
            i += 1
    w('                }')
    w('                for (cb = 0; cb < ncb; cb++) {')
    w('                        const double *c = B->c + (size_t)cb * B->ld;')
    w('                        for (i = 0; i < %d; i++) {' % nt)
    w('                                const double *vi = v + (size_t)i * nb;')
    w('                                double y = 0;')
    w('                                for (b = 0; b < nbt; b++) y += c[b] * vi[b];')
    w('                                w[cb * %d + i] = y;' % nt)
    w('                        }')
    w('                }')
    w('                const double kq = K->kp[t];')
    w('                for (kc = 0; kc < nck; kc++) {')
    w('                        const double f = K->c[(size_t)kc * K->ld + t] * kq;')
    w('                        double *so = SK + (size_t)kc * ncb * %d;' % nt)
    w('                        for (i = 0; i < ncb * %d; i++) so[i] += f * w[i];' % nt)
    w('                }')
    w('        }')
    w('        if (!any) return 0;')
    if jonly:
        o += jpost_lines(la, lb, lc, ld, E, Fb, eidx, fidx)
        w('        return 1;')
        w('}')
        return '\n'.join(o), fn, nt
    # ---- per contraction combination: HRR, c2s, placement
    w('        double AB[3], CD[3];')
    w('        for (i = 0; i < 3; i++) {')
    w('                AB[i] = bs ? -B->rij[i] : B->rij[i];')
    w('                CD[i] = ks ? -K->rij[i] : K->rij[i];')
    w('        }')
    w('        (void)AB; (void)CD;')
    w('        int ci, cj, ck, cl;')
    w('        for (cl = 0; cl < K->ncj; cl++)')
    w('        for (ck = 0; ck < K->nci; ck++)')
    w('        for (cj = 0; cj < B->ncj; cj++)')
    w('        for (ci = 0; ci < B->nci; ci++) {')
    w('                const int cpos[4] = {ci, cj, ck, cl};')
    w('                const double *sk = SK + ((size_t)(cl * K->nci + ck) * ncb + cj * B->nci + ci) * %d;' % nt)
    w('                double *o = out + cpos[pos[0]] * %d * st[0] + cpos[pos[1]] * %d * st[1]' % (2 * la + 1, 2 * lb + 1))
    w('                          + cpos[pos[2]] * %d * st[2] + cpos[pos[3]] * %d * st[3];' % (2 * lc + 1, 2 * ld + 1))
    # bra HRR: HB[(ia * NB + ib) * NFT + f]
    if lb == 0:
        w('                const double *HB = sk;')
    else:
        border, bdeps = hrr_plan(la, lb, None)
        names = {}
        for (a, b) in border:
            d = bdeps[(a, b)]
            if d is None:
                names[(a, b)] = 'sk + %d' % (eidx[a] * NFT)
                continue
            nm = 'hb%s_%s' % (tag(a), tag(b))
            names[(a, b)] = nm
        w('                double HB[%d];' % (NAB * NFT))
        for (a, b) in border:
            d = bdeps[(a, b)]
            if d is None:
                continue
            k, n1, n2 = d
            tgt = deg_tgt = (sum(a) == la and sum(b) == lb)
            if tgt:
                dst = 'HB + %d' % ((A.index(a) * NB + Bc.index(b)) * NFT)
            else:
                w('                double %s[%d];' % (names[(a, b)], NFT))
                dst = names[(a, b)]
            w('                for (i = 0; i < %d; i++) (%s)[i] = (%s)[i] + AB[%d] * (%s)[i];' % (
                NFT, dst, ref(names, n1, la, lb, A, Bc, NB, NFT), k, ref(names, n2, la, lb, A, Bc, NB, NFT)))
    # ket HRR: R[(ic * ND + id) * NAB + ab]
    if ld == 0:
        # R = HB transposed: R[ic][ab] = HB[ab * NFT + ic]
        w('                double R[%d];' % (NC * NAB))
        w('                for (i = 0; i < %d; i++) {' % NAB)
        for ic in range(NC):
            w('                        R[%d + i] = HB[i * %d + %d];' % (ic * NAB, NFT, ic))
        w('                }')
    else:
        korder, kdeps = hrr_plan(lc, ld, None)
        knames = {}
        w('                double R[%d];' % (NC * ND * NAB))
        for (c, d_) in korder:
            dd = kdeps[(c, d_)]
            if dd is None:
                # base: column fidx[c] of HB, gathered
                nm = 'kb%s' % tag(c)
                knames[(c, d_)] = nm
                w('                double %s[%d];' % (nm, NAB))
                w('                for (i = 0; i < %d; i++) %s[i] = HB[i * %d + %d];' % (NAB, nm, NFT, fidx[c]))
                continue
            k, n1, n2 = dd
            if sum(c) == lc and sum(d_) == ld:
                dst = 'R + %d' % ((Cc.index(c) * ND + Dc.index(d_)) * NAB)
                knames[(c, d_)] = dst
            else:
                nm = 'hk%s_%s' % (tag(c), tag(d_))
                w('                double %s[%d];' % (nm, NAB))
                dst = nm
                knames[(c, d_)] = nm
            w('                for (i = 0; i < %d; i++) (%s)[i] = (%s)[i] + CD[%d] * (%s)[i];' % (
                NAB, dst, knames[n1], k, knames[n2]))
    # c2s + placement: out[ma st0 + mb st1 + mc st2 + md st3] = fac * sum
    cA, cB, cC, cD = C2S[la], C2S[lb], C2S[lc], C2S[ld]
    for mc in range(len(cC)):
        for md in range(len(cD)):
            # T[ab] = sum_{c,d} cC[mc][c] cD[md][d] R[(c ND + d) NAB + ab]
            terms = [(cC[mc][ic] * cD[md][id_], (ic * ND + id_) * NAB)
                     for ic in range(NC) for id_ in range(ND) if cC[mc][ic] * cD[md][id_] != 0]
            for ma in range(len(cA)):
                for mb in range(len(cB)):
                    sm = []
                    for ia in range(NA):
                        for ib in range(NB):
                            cab = cA[ma][ia] * cB[mb][ib]
                            if cab == 0:
                                continue
                            for cf, off in terms:
                                sm.append('%.18g * R[%d]' % (cab * cf, off + ia * NB + ib))
                    w('                o[%d * st[0] + %d * st[1] + %d * st[2] + %d * st[3]] = fac * (%s);' % (
                        ma, mb, mc, md, ' + '.join(sm)))
    w('        }')
    w('        return 1;')
    w('}')
    return '\n'.join(o), fn, nt


def ref(names, n, la, lb, A, Bc, NB, NFT):
    a, b = n
    nm = names[n]
    if nm.startswith('sk'):
        return nm
    if sum(a) == la and sum(b) == lb:
        return 'HB + %d' % ((A.index(a) * NB + Bc.index(b)) * NFT)
    return nm


def classes(lmax_all=False):
    '''the generated classes: every one of l <= LMAX_ALL, those with an f shell up to total l
    F_LSUM_MAX (lmax_all: only the former; the lane and J-only forms)'''
    pairs = [(a, b) for a in range(LMAX + 1) for b in range(a + 1)]
    out = []
    for (a, b) in pairs:
        for (c, d) in pairs:
            if max(a, b, c, d) <= LMAX_ALL or (not lmax_all and a + b + c + d <= F_LSUM_MAX):
                out.append((a, b, c, d))
    return out


def main():
    out = ['/* Generated by gen_jk_vrr.py (LMAX = %d); do not edit. */' % LMAX,
           '/* included by jk_gen.c, which defines GPair, boys_n */', '']
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    table = []
    fk = []
    # -DJKG_NO_TRAP: the per-quartet kernels with no-trapping-math too (their F_m loops then
    # vectorise, with gathers)
    out += ['#ifdef JKG_NO_TRAP', '#pragma GCC push_options', '#pragma GCC optimize ("no-trapping-math")',
            '#endif', '']
    for cls in classes():
        if max(cls) <= LMAX_ALL:
            code, fn, nt = kernel(*cls)
            out.append(code)
            out.append('')
        else:
            code, fn, nt = kernel(*cls, storage='int')
            fk.append((code, fn))
        table.append((cls, fn, nt))
    out += ['#ifdef JKG_NO_TRAP', '#pragma GCC pop_options', '#endif', '']
    # the f kernels: own translation units, balanced by size; prototypes in jk_vrr_f.h
    files = [[] for _ in range(NFFILES)]
    size = [0] * NFFILES
    for code, fn in sorted(fk, key=lambda k: -len(k[0])):
        k = size.index(min(size))
        files[k].append(code)
        size[k] += len(code)
    for k, codes in enumerate(files):
        with open(os.path.join(here, 'jk_vrr_f_%d.c' % k), 'w') as f:
            f.write('/* Generated by gen_jk_vrr.py: the classes with f shells; do not edit. */\n')
            f.write('#include "jk_gen.h"\n\n')
            f.write('\n\n'.join(codes) + '\n')
    with open(os.path.join(here, 'jk_vrr_f.h'), 'w') as f:
        f.write('/* Generated by gen_jk_vrr.py: prototypes of the f-shell kernels (jk_vrr_f_<k>.c). */\n')
        for code, fn in fk:
            f.write('int %s(double *out, const size_t *st, const int *pos, const GPair *B, int bs,\n'
                    '        const GPair *K, int ks, double cut, double fac, const double *tab, double *wk);\n' % fn)
    out.insert(2, '#include "jk_vrr_f.h"')
    n = LMAX + 1
    out.append('typedef int (*jkg_fn)(double *, const size_t *, const int *, const GPair *, int, const GPair *,')
    out.append('                      int, double, double, const double *, double *);')
    out.append('#define JKV_LMAX %d' % LMAX)
    out.append('#define JKG_NT %d       /* the most VRR targets of a generated class */' % max(t[2] for t in table))
    out.append('#define JKG_LSUM %d     /* the largest total l of a generated class */' % max(sum(t[0]) for t in table))
    out.append('static const jkg_fn jkg_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn, nt in table:
        out.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    out.append('};')
    # the J-only forms (jk_gen.c SOCVHFgen_j2e)
    jt = []
    for cls in classes(lmax_all=True):
        code, fn, nt = kernel(*cls, jonly=True)
        out.append(code)
        out.append('')
        jt.append((cls, fn))
    out.append('typedef int (*jjg_fn)(double *, double *, const double *, const double *, size_t, size_t,')
    out.append('                      const GPair *, int, const GPair *, int, double, double, const double *, double *);')
    out.append('static const jjg_fn jjg_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn in jt:
        out.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    out.append('};')
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
