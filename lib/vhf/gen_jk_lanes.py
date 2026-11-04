#!/usr/bin/env python
'''
Generator of lib/vhf/jk_lane_gen.c: the lane form of the kernels of
gen_jk_vrr.py.  One call evaluates GL_NL shell quartets (A B | C_l D_l),
l < GL_NL, sharing the bra pair and the ket shell types (the same exponents
and contraction coefficients, the ket shells' centres differing): the
primitive loops run in lockstep, the SIMD lanes are the quartets.  Every
quantity of the bra and of the ket exponents (p, q, 1 / (p + q), rho, PA,
the coefficient products) is a scalar, only the ket geometry (Q, Q - C,
K_cd, C - D) a vector; the vertical recursions, the contractions, the
horizontal recursions and the Cartesian-to-spherical transform are all
vectorised over the lanes.

out[e * GL_NL + l]: element e of quartet l at the role strides st[] (the
int2e_sph layout of the quartets, the same for all lanes).

    python gen_jk_lanes.py > jk_lane_gen.c
'''
import sys
from gen_jk_vrr import (LMAX, X, C2S, comps, band, sub, add, first_nz, tag, vname, vrr_plan, hrr_plan,
                        classes, boys_lines)

NL = 'GL_NL'
LANE_NO_TRAP = True    # -fno-trapping-math for the lane kernels only (GCC optimize pragma)


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
    fn = 'jkl_%d%d%d%d' % (la, lb, lc, ld)
    o = []
    w = o.append
    w('/* (%d%d|%d%d) on GL_NL lanes: %d VRR nodes, %d targets */' % (la, lb, lc, ld, len(order), nt))
    w('static int %s(double *out, const size_t *st, const int *pos, const GPair *B, int bs,' % fn)
    w('                const LaneK *K, double cut, double fac, const double *tab, double *wk)')
    w('{')
    w('        const int nb = B->n, nk = K->nk, ncb = B->nc, nck = K->nck;')
    w('        const double *pax = bs ? B->rjx : B->rix, *pay = bs ? B->rjy : B->riy, *paz = bs ? B->rjz : B->riz;')
    w('        double *v = wk, *SK = v + (size_t)%d * %s;' % (nt, NL))
    w('        int t, b, cb, kc, i, l, any = 0;')
    w('        memset(SK, 0, sizeof(double) * nck * ncb * %d * %s);' % (nt, NL))
    w('        for (t = 0; t < nk; t++) {')
    w('                if (K->bmx[t] < cut) continue;')
    w('                any = 1;')
    w('                const int nbt = n_above(B->beta, nb, cut * B->bmax / K->bmx[t]);')
    w('                const double q = K->q[t], h2q = .5 / q;')
    w('                const double *QX = K->qx[t], *QY = K->qy[t], *QZ = K->qz[t], *KP = K->kp[t];')
    w('                const double *QCX = K->qcx[t], *QCY = K->qcy[t], *QCZ = K->qcz[t];')
    w('                (void)h2q; (void)QCX; (void)QCY; (void)QCZ;')
    w('                for (b = 0; b < nbt; b++) {')
    w('                        const double p = B->p[b], s = 1. / (p + q), rho = p * q * s;')
    w('                        const double h2p = .5 / p, rp = rho / p, rq = rho / q, h2pq = .5 * s;')
    w('                        const double ws = -q * s, wq = p * s;')
    w('                        const double PAx = pax[b], PAy = pay[b], PAz = paz[b];')
    w('                        const double Px = B->px[b], Py = B->py[b], Pz = B->pz[b];')
    w('                        const double gb = B->kp[b] * sqrt(s);')
    w('                        (void)h2p; (void)rp; (void)rq; (void)h2pq; (void)ws; (void)wq;')
    w('                        (void)PAx; (void)PAy; (void)PAz;')
    w('#pragma omp simd')
    w('                        for (l = 0; l < %s; l++) {' % NL)
    w('                                const double dx = Px - QX[l], dy = Py - QY[l], dz = Pz - QZ[l];')
    w('                                const double T = rho * (dx * dx + dy * dy + dz * dz);')
    for line in boys_lines(L, ' ' * 32):
        w(line)
    w('                                const double WPx = ws * dx, WPy = ws * dy, WPz = ws * dz;')
    w('                                const double WQx = wq * dx, WQy = wq * dy, WQz = wq * dz;')
    w('                                const double QCx = QCX[l], QCy = QCY[l], QCz = QCZ[l];')
    w('                                (void)WPx; (void)WPy; (void)WPz; (void)WQx; (void)WQy; (void)WQz;')
    w('                                (void)QCx; (void)QCy; (void)QCz;')
    w('                                const double g = gb * KP[l];')
    for key in order:
        w('                                const double %s = %s;' % (vname(*key), exprs[key]))
    i = 0
    for e in E:
        for f in Fb:
            w('                                v[%d * %s + l] = g * %s;' % (i, NL, vname(e, f, 0)))
            i += 1
    w('                        }')
    w('                        for (cb = 0; cb < ncb; cb++) {')
    w('                                const double cbv = B->c[(size_t)cb * B->ld + b];')
    w('                                for (kc = 0; kc < nck; kc++) {')
    w('                                        const double wc = cbv * K->c[kc * GL_MAXP + t];')
    w('                                        double *so = SK + (size_t)(kc * ncb + cb) * %d * %s;' % (nt, NL))
    w('#pragma omp simd')
    w('                                        for (i = 0; i < %d * %s; i++) so[i] += wc * v[i];' % (nt, NL))
    w('                                }')
    w('                        }')
    w('                }')
    w('        }')
    w('        if (!any) return 0;')
    w('        double AB[3];')
    w('        for (i = 0; i < 3; i++) AB[i] = bs ? -B->rij[i] : B->rij[i];')
    w('        (void)AB;')
    w('        const double *CDx = K->cd[0], *CDy = K->cd[1], *CDz = K->cd[2];')
    w('        const double *CD[3] = {CDx, CDy, CDz};')
    w('        (void)CD;')
    w('        int ci, cj, ck, cl;')
    w('        for (cl = 0; cl < K->ncj; cl++)')
    w('        for (ck = 0; ck < K->nci; ck++)')
    w('        for (cj = 0; cj < B->ncj; cj++)')
    w('        for (ci = 0; ci < B->nci; ci++) {')
    w('                const int cpos[4] = {ci, cj, ck, cl};')
    w('                const double *sk = SK + ((size_t)(cl * K->nci + ck) * ncb + cj * B->nci + ci) * %d * %s;'
      % (nt, NL))
    w('                double *o = out + (cpos[pos[0]] * %d * st[0] + cpos[pos[1]] * %d * st[1]'
      % (2 * la + 1, 2 * lb + 1))
    w('                          + cpos[pos[2]] * %d * st[2] + cpos[pos[3]] * %d * st[3]) * %s;'
      % (2 * lc + 1, 2 * ld + 1, NL))
    # bra HRR: HB[((ia * NB + ib) * NFT + f) * NL + l]
    if lb == 0:
        w('                const double *HB = sk;')
    else:
        border, bdeps = hrr_plan(la, lb, None)
        names = {}
        for (a, b) in border:
            if bdeps[(a, b)] is None:
                names[(a, b)] = 'sk + %d * %s' % (eidx[a] * NFT, NL)
            elif sum(a) == la and sum(b) == lb:
                names[(a, b)] = 'HB + %d * %s' % ((A.index(a) * NB + Bc.index(b)) * NFT, NL)
            else:
                names[(a, b)] = 'hb%s_%s' % (tag(a), tag(b))
        w('                double HB[%d * %s];' % (NAB * NFT, NL))
        for (a, b) in border:
            d = bdeps[(a, b)]
            if d is None:
                continue
            k, n1, n2 = d
            if not (sum(a) == la and sum(b) == lb):
                w('                double %s[%d * %s];' % (names[(a, b)], NFT, NL))
            w('#pragma omp simd')
            w('                for (i = 0; i < %d * %s; i++) (%s)[i] = (%s)[i] + AB[%d] * (%s)[i];' % (
                NFT, NL, names[(a, b)], names[n1], k, names[n2]))
    # ket HRR: R[((ic * ND + id) * NAB + ab) * NL + l]
    if ld == 0:
        w('                double R[%d * %s];' % (NC * NAB, NL))
        w('                for (i = 0; i < %d; i++) {' % NAB)
        w('#pragma omp simd')
        w('                        for (l = 0; l < %s; l++) {' % NL)
        for ic in range(NC):
            w('                                R[(%d + i) * %s + l] = HB[(i * %d + %d) * %s + l];' % (
                ic * NAB, NL, NFT, ic, NL))
        w('                        }')
        w('                }')
    else:
        korder, kdeps = hrr_plan(lc, ld, None)
        knames = {}
        w('                double R[%d * %s];' % (NC * ND * NAB, NL))
        for (c, d_) in korder:
            dd = kdeps[(c, d_)]
            if dd is None:
                nm = 'kb%s' % tag(c)
                knames[(c, d_)] = nm
                w('                double %s[%d * %s];' % (nm, NAB, NL))
                w('                for (i = 0; i < %d; i++)' % NAB)
                w('#pragma omp simd')
                w('                        for (l = 0; l < %s; l++) %s[i * %s + l] = HB[(i * %d + %d) * %s + l];' % (
                    NL, nm, NL, NFT, fidx[c], NL))
                continue
            k, n1, n2 = dd
            if sum(c) == lc and sum(d_) == ld:
                dst = 'R + %d * %s' % ((Cc.index(c) * ND + Dc.index(d_)) * NAB, NL)
            else:
                dst = 'hk%s_%s' % (tag(c), tag(d_))
                w('                double %s[%d * %s];' % (dst, NAB, NL))
            knames[(c, d_)] = dst
            w('                for (i = 0; i < %d; i++)' % NAB)
            w('#pragma omp simd')
            w('                        for (l = 0; l < %s; l++) (%s)[i * %s + l] = (%s)[i * %s + l]'
              ' + CD[%d][l] * (%s)[i * %s + l];' % (NL, dst, NL, knames[n1], NL, k, knames[n2], NL))
    # c2s + placement
    cA, cB, cC, cD = C2S[la], C2S[lb], C2S[lc], C2S[ld]
    w('                {')
    w('                const size_t s0 = st[0] * %s, s1 = st[1] * %s, s2 = st[2] * %s, s3 = st[3] * %s;'
      % (NL, NL, NL, NL))
    w('#pragma omp simd')
    w('                for (l = 0; l < %s; l++) {' % NL)
    for mc in range(len(cC)):
        for md in range(len(cD)):
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
                                sm.append('%.18g * R[%d * %s + l]' % (cab * cf, off + ia * NB + ib, NL))
                    w('                        o[%d * s0 + %d * s1 + %d * s2 + %d * s3 + l] = fac * (%s);' % (
                        ma, mb, mc, md, ' + '.join(sm)))
    w('                }')
    w('                }')
    w('        }')
    w('        return 1;')
    w('}')
    return '\n'.join(o), fn, nt


def main():
    out = ['/* Generated by gen_jk_lanes.py (LMAX = %d); do not edit. */' % LMAX,
           '/* included by jk_gen.c after jk_vrr_gen.c (GPair, LaneK, taylor, n_above) */', '']
    if LANE_NO_TRAP:
        # the lane loops' selects if-converted, so that they vectorise over the GL_NL quartets;
        # here only (the per-quartet kernels' F_m gathers are slower vectorised)
        out += ['#pragma GCC push_options', '#pragma GCC optimize ("no-trapping-math")', '']
    table = []
    mx = 0
    for cls in classes(lmax_all=True):
        code, fn, nt = kernel(*cls)
        out.append(code)
        out.append('')
        table.append((cls, fn))
        mx = max(mx, nt)
    if LANE_NO_TRAP:
        out += ['#pragma GCC pop_options', '']
    n = LMAX + 1
    out.append('typedef int (*jkl_fn)(double *, const size_t *, const int *, const GPair *, int, const LaneK *,')
    out.append('                      double, double, const double *, double *);')
    out.append('#define JKL_NT %d       /* the largest number of VRR targets */' % mx)
    out.append('#define JKL_LMAX %d     /* the lanes take shells up to this l */' % max(max(c) for c, f in table))
    out.append('static const jkl_fn jkl_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn in table:
        out.append('        [%d][%d][%d][%d] = %s,' % (cls + (fn,)))
    out.append('};')
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
