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

LMAX = 2
X = 'xyz'
# libcint's Cartesian-to-spherical coefficients (xx xy xz yy yz zz -> xy yz z2 xz x2-y2)
C2S = {0: [[1.]], 1: [[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
       2: [[0, 1.092548430592079070, 0, 0, 0, 0],
           [0, 0, 0, 0, 1.092548430592079070, 0],
           [-0.315391565252520002, 0, 0, -0.315391565252520002, 0, 0.630783130505040012],
           [0, 0, 1.092548430592079070, 0, 0, 0],
           [0.546274215296039535, 0, 0, -0.546274215296039535, 0, 0]]}


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
            terms = ['F[%d]' % m]
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


def kernel(la, lb, lc, ld):
    order, exprs, E, Fb = vrr_plan(la, lb, lc, ld)
    L = la + lb + lc + ld
    NET, NFT = len(E), len(Fb)
    nt = NET * NFT
    eidx = {e: i for i, e in enumerate(E)}
    fidx = {f: i for i, f in enumerate(Fb)}
    A, Bc, Cc, Dc = comps(la), comps(lb), comps(lc), comps(ld)
    NA, NB, NC, ND = len(A), len(Bc), len(Cc), len(Dc)
    NAB = NA * NB
    fn = 'jkg_%d%d%d%d' % (la, lb, lc, ld)
    o = []
    w = o.append
    w('/* (%d%d|%d%d): %d VRR nodes, %d targets */' % (la, lb, lc, ld, len(order), nt))
    w('static int %s(double *out, const size_t *st, const int *pos, const GPair *B, int bs,' % fn)
    w('                const GPair *K, int ks, double cut, double fac, const double *tab, double *wk)')
    w('{')
    w('        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;')
    w('        const double *pax = bs ? B->rjx : B->rix, *pay = bs ? B->rjy : B->riy, *paz = bs ? B->rjz : B->riz;')
    w('        const double *qcx = ks ? K->rjx : K->rix, *qcy = ks ? K->rjy : K->riy, *qcz = ks ? K->rjz : K->riz;')
    w('        double *v = wk, *w = v + (size_t)%d * nb, *SK = w + (size_t)%d * ncb;' % (nt, nt))
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
    w('                for (b = 0; b < nbt; b++) {')
    w('                        const double p = B->p[b], s = 1. / (p + q);')
    w('                        const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;')
    w('                        const double rho = p * q * s, T = rho * (dx * dx + dy * dy + dz * dz);')
    w('                        double F[%d];' % (L + 1))
    w('                        boys_n(tab, T, %d, F);' % L)
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


def classes():
    pairs = [(a, b) for a in range(LMAX + 1) for b in range(a + 1)]
    return [(a, b, c, d) for (a, b) in pairs for (c, d) in pairs]


def main():
    out = ['/* Generated by gen_jk_vrr.py (LMAX = %d); do not edit. */' % LMAX,
           '/* included by jk_gen.c, which defines GPair, boys_n */', '']
    table = []
    for cls in classes():
        code, fn, nt = kernel(*cls)
        out.append(code)
        out.append('')
        table.append((cls, fn, nt))
    n = LMAX + 1
    out.append('typedef int (*jkg_fn)(double *, const size_t *, const int *, const GPair *, int, const GPair *,')
    out.append('                      int, double, double, const double *, double *);')
    out.append('#define JKV_LMAX %d' % LMAX)
    out.append('static const jkg_fn jkg_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn, nt in table:
        out.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    out.append('};')
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
