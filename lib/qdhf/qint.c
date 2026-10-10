/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 *
 * Gaussian integrals in quadruple precision (IEEE binary128, __float128)
 * using the McMurchie-Davidson scheme.
 *
 * The integrals are evaluated over contracted Cartesian Gaussian shells
 *      g(r) = (x-Ax)^lx (y-Ay)^ly (z-Az)^lz sum_k c_k exp(-a_k |r-A|^2)
 * The normalization factors are included in the contraction coefficients
 * c_k by the caller.  Cartesian components are ordered as in libcint
 * (xx, xy, xz, yy, yz, zz for d shells).
 *
 * Shell data are stored in a libcint-like layout
 *      bas[ish*BAS_SLOTS+ANG_OF]    angular momentum
 *      bas[ish*BAS_SLOTS+NPRIM_OF]  number of primitives
 *      bas[ish*BAS_SLOTS+PTR_EXP]   offset of the exponents in env
 *      bas[ish*BAS_SLOTS+PTR_COEFF] offset of the coefficients in env
 *      bas[ish*BAS_SLOTS+PTR_COORD] offset of the center in env
 */

#include <stdlib.h>
#include <string.h>
#include <quadmath.h>

typedef __float128 quad;

#define MAXL            5
#define LPAIR           (2*MAXL+2)
#define BAS_SLOTS       5
#define ANG_OF          0
#define NPRIM_OF        1
#define PTR_EXP         2
#define PTR_COEFF       3
#define PTR_COORD       4
#define NCART(l)        (((l)+1)*((l)+2)/2)
/* primitive pairs with exp(-mu |A-B|^2) below this value are dropped */
#define PAIR_CUTOFF     1e-80Q

typedef struct {
        quad p;
        quad P[3];
        quad fac;
        /* Hermite expansion coefficients E[xyz][i][j][t]; E000 of the
         * x-component includes the factor exp(-mu |A-B|^2) */
        quad E[3][MAXL+1][MAXL+1][LPAIR];
} PrimPair;

static void cart_comps(int l, int *lx, int *ly, int *lz)
{
        int i, j, n = 0;
        for (i = l; i >= 0; i--) {
        for (j = l - i; j >= 0; j--) {
                lx[n] = i;
                ly[n] = j;
                lz[n] = l - i - j;
                n++;
        } }
}

/* Boys function F_m(t), m = 0 .. mmax */
static void boys_function(int mmax, quad t, quad *f)
{
        int m, k;
        quad et = expq(-t);
        if (t < mmax + 40) {
                /* F_m(t) = e^{-t} sum_k (2t)^k / ((2m+1)(2m+3)...(2m+2k+1)) */
                quad term = 1 / (quad)(2*mmax+1);
                quad sum = term;
                for (k = 1; k < 5000; k++) {
                        term *= 2 * t / (2*mmax + 2*k + 1);
                        sum += term;
                        if (term < sum * 1e-38Q) {
                                break;
                        }
                }
                f[mmax] = et * sum;
                for (m = mmax; m > 0; m--) {
                        f[m-1] = (2 * t * f[m] + et) / (2*m - 1);
                }
        } else {
                quad st = sqrtq(t);
                f[0] = sqrtq(M_PIq) / (2 * st) * erfq(st);
                for (m = 0; m < mmax; m++) {
                        f[m+1] = ((2*m+1) * f[m] - et) / (2 * t);
                }
        }
}

static void hermite_E(quad E[MAXL+1][MAXL+1][LPAIR], int la, int lb,
                      quad PA, quad PB, quad oo2p, quad e000)
{
        int i, j, t;
        memset(E, 0, sizeof(quad) * (MAXL+1) * (MAXL+1) * LPAIR);
        E[0][0][0] = e000;
        for (i = 1; i <= la; i++) {
                for (t = 0; t <= i; t++) {
                        quad v = PA * E[i-1][0][t] + (t+1) * E[i-1][0][t+1];
                        if (t > 0) {
                                v += oo2p * E[i-1][0][t-1];
                        }
                        E[i][0][t] = v;
                }
        }
        for (j = 1; j <= lb; j++) {
        for (i = 0; i <= la; i++) {
                for (t = 0; t <= i+j; t++) {
                        quad v = PB * E[i][j-1][t] + (t+1) * E[i][j-1][t+1];
                        if (t > 0) {
                                v += oo2p * E[i][j-1][t-1];
                        }
                        E[i][j][t] = v;
                }
        } }
}

static int make_pairs(PrimPair *pairs, int ish, int jsh,
                      const int *bas, const quad *env)
{
        const int *bi = bas + ish * BAS_SLOTS;
        const int *bj = bas + jsh * BAS_SLOTS;
        int li = bi[ANG_OF];
        int lj = bj[ANG_OF];
        int npi = bi[NPRIM_OF];
        int npj = bj[NPRIM_OF];
        const quad *ai = env + bi[PTR_EXP];
        const quad *aj = env + bj[PTR_EXP];
        const quad *ci = env + bi[PTR_COEFF];
        const quad *cj = env + bj[PTR_COEFF];
        const quad *ra = env + bi[PTR_COORD];
        const quad *rb = env + bj[PTR_COORD];
        quad ab[3], rr;
        int ip, jp, d, n = 0;
        for (d = 0; d < 3; d++) {
                ab[d] = ra[d] - rb[d];
        }
        rr = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2];
        for (ip = 0; ip < npi; ip++) {
        for (jp = 0; jp < npj; jp++) {
                quad a = ai[ip];
                quad b = aj[jp];
                quad p = a + b;
                quad kab = expq(-a * b / p * rr);
                if (kab < PAIR_CUTOFF) {
                        continue;
                }
                PrimPair *pp = pairs + n;
                pp->p = p;
                pp->fac = ci[ip] * cj[jp];
                for (d = 0; d < 3; d++) {
                        pp->P[d] = (a * ra[d] + b * rb[d]) / p;
                        hermite_E(pp->E[d], li, lj, pp->P[d] - ra[d],
                                  pp->P[d] - rb[d], 1 / (2 * p),
                                  (d == 0) ? kab : 1);
                }
                n++;
        } }
        return n;
}

/*
 * Hermite Coulomb integrals R_{tuv} (order n = 0), t+u+v <= L, stored in
 * R[(t*(L+1)+u)*(L+1)+v].  buf has the same size as R.
 */
static void hermite_R(quad *R, quad *buf, int L, quad alpha,
                      quad X, quad Y, quad Z, const quad *F)
{
        int L1 = L + 1;
        int n, s, t, u, v;
        quad *cur = R;
        quad *prev = buf;
        quad *tmp;
        quad m2a = -2 * alpha;
        quad fac[4*MAXL+2];
        fac[0] = 1;
        for (n = 1; n <= L; n++) {
                fac[n] = fac[n-1] * m2a;
        }
        /* make sure that the final result is stored in R */
        if (L % 2 == 1) {
                cur = buf;
                prev = R;
        }
#define IDX(t,u,v)      (((t)*L1+(u))*L1+(v))
        for (n = L; n >= 0; n--) {
                cur[0] = fac[n] * F[n];
                for (s = 1; s <= L - n; s++) {
                for (t = s; t >= 0; t--) {
                for (u = s - t; u >= 0; u--) {
                        v = s - t - u;
                        quad val;
                        if (t > 0) {
                                val = X * prev[IDX(t-1,u,v)];
                                if (t > 1) val += (t-1) * prev[IDX(t-2,u,v)];
                        } else if (u > 0) {
                                val = Y * prev[IDX(t,u-1,v)];
                                if (u > 1) val += (u-1) * prev[IDX(t,u-2,v)];
                        } else {
                                val = Z * prev[IDX(t,u,v-1)];
                                if (v > 1) val += (v-1) * prev[IDX(t,u,v-2)];
                        }
                        cur[IDX(t,u,v)] = val;
                } } }
                tmp = cur;
                cur = prev;
                prev = tmp;
        }
#undef IDX
}

static int max_prims(const int *shls, int n, const int *bas)
{
        int i, m = 1;
        for (i = 0; i < n; i++) {
                if (bas[shls[i]*BAS_SLOTS+NPRIM_OF] > m) {
                        m = bas[shls[i]*BAS_SLOTS+NPRIM_OF];
                }
        }
        return m;
}

static int max_ang(const int *shls, int n, const int *bas)
{
        int i, m = 0;
        for (i = 0; i < n; i++) {
                if (bas[shls[i]*BAS_SLOTS+ANG_OF] > m) {
                        m = bas[shls[i]*BAS_SLOTS+ANG_OF];
                }
        }
        return m;
}

static void shell_offsets(int *offs, const int *shls, int n, const int *bas)
{
        int i;
        offs[0] = 0;
        for (i = 0; i < n; i++) {
                offs[i+1] = offs[i] + NCART(bas[shls[i]*BAS_SLOTS+ANG_OF]);
        }
}

/* ------------------------------------------------------------------ */
/* one-electron integrals                                             */
/* ------------------------------------------------------------------ */
/*
 * kind = 0: overlap <i|j>
 * kind = 2, 3, 4: dipole <i| x | j>, <i| y |j>, <i| z |j> (origin at 0)
 * kind = 1: nuclear attraction <i| -sum_C Z_C/|r-R_C| |j>.  zetas[C] > 0
 *           selects the Gaussian charge distribution
 *           rho_C(r) = Z_C (zeta/pi)^{3/2} exp(-zeta |r-R_C|^2), for which
 *           1/|r-R_C| is replaced by erf(sqrt(zeta) |r-R_C|)/|r-R_C|.
 *           zetas can be NULL (point charges).
 * out[Ni,Nj], Ni, Nj = number of Cartesian functions in shls_i, shls_j
 * Returns 0 on success, -1 if the angular momentum is not supported.
 */
int QDint1e(int kind, quad *out, const int *shls_i, int nsi,
            const int *shls_j, int nsj, const int *bas, const quad *env,
            int natm, const quad *charges, const quad *coords,
            const quad *zetas)
{
        if (max_ang(shls_i, nsi, bas) > MAXL ||
            max_ang(shls_j, nsj, bas) > MAXL) {
                return -1;
        }
        int *offi = malloc(sizeof(int) * (nsi + nsj + 2));
        int *offj = offi + nsi + 1;
        shell_offsets(offi, shls_i, nsi, bas);
        shell_offsets(offj, shls_j, nsj, bas);
        int Nj = offj[nsj];
        int mpi = max_prims(shls_i, nsi, bas);
        int mpj = max_prims(shls_j, nsj, bas);

#pragma omp parallel
{
        PrimPair *pairs = malloc(sizeof(PrimPair) * mpi * mpj);
        int L1max = 2 * MAXL + 1;
        quad *R = malloc(sizeof(quad) * L1max * L1max * L1max);
        quad *Rbuf = malloc(sizeof(quad) * L1max * L1max * L1max);
        quad F[4*MAXL+2];
        int lxa[NCART(MAXL)], lya[NCART(MAXL)], lza[NCART(MAXL)];
        int lxb[NCART(MAXL)], lyb[NCART(MAXL)], lzb[NCART(MAXL)];
        int ij, ish, jsh, a, b, n, npair, ic, t, u, v;
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nsi * nsj; ij++) {
                ish = ij / nsj;
                jsh = ij % nsj;
                int I = shls_i[ish];
                int J = shls_j[jsh];
                int li = bas[I*BAS_SLOTS+ANG_OF];
                int lj = bas[J*BAS_SLOTS+ANG_OF];
                int nci = NCART(li);
                int ncj = NCART(lj);
                int L = li + lj;
                int L1 = L + 1;
                cart_comps(li, lxa, lya, lza);
                cart_comps(lj, lxb, lyb, lzb);
                for (a = 0; a < nci; a++) {
                for (b = 0; b < ncj; b++) {
                        out[(long)(offi[ish]+a)*Nj+offj[jsh]+b] = 0;
                } }
                npair = make_pairs(pairs, I, J, bas, env);
                for (n = 0; n < npair; n++) {
                        PrimPair *pp = pairs + n;
                        quad p = pp->p;
                        if (kind == 0) {
                                quad fac = pp->fac * powq(M_PIq / p, 1.5Q);
                                for (a = 0; a < nci; a++) {
                                for (b = 0; b < ncj; b++) {
                                        out[(long)(offi[ish]+a)*Nj+offj[jsh]+b] += fac
                                                * pp->E[0][lxa[a]][lxb[b]][0]
                                                * pp->E[1][lya[a]][lyb[b]][0]
                                                * pp->E[2][lza[a]][lzb[b]][0];
                                } }
                                continue;
                        }
                        if (kind >= 2 && kind <= 4) {
                                /* r_d = (r_d - P_d) + P_d:
                                 * int x_C Lambda_t = sqrt(pi/p) (P_C delta_t0 + delta_t1) */
                                int dd = kind - 2;
                                quad fac = pp->fac * powq(M_PIq / p, 1.5Q);
                                for (a = 0; a < nci; a++) {
                                for (b = 0; b < ncj; b++) {
                                        int la[3] = {lxa[a], lya[a], lza[a]};
                                        int lb[3] = {lxb[b], lyb[b], lzb[b]};
                                        quad v = fac;
                                        int d1;
                                        for (d1 = 0; d1 < 3; d1++) {
                                                const quad *E1 = pp->E[d1][la[d1]][lb[d1]];
                                                if (d1 == dd) {
                                                        v *= E1[1] + pp->P[d1] * E1[0];
                                                } else {
                                                        v *= E1[0];
                                                }
                                        }
                                        out[(long)(offi[ish]+a)*Nj+offj[jsh]+b] += v;
                                } }
                                continue;
                        }
                        for (ic = 0; ic < natm; ic++) {
                                if (charges[ic] == 0) {
                                        continue;
                                }
                                quad X = pp->P[0] - coords[ic*3+0];
                                quad Y = pp->P[1] - coords[ic*3+1];
                                quad Z = pp->P[2] - coords[ic*3+2];
                                quad aeff = p;
                                quad fac = -charges[ic] * pp->fac * 2 * M_PIq / p;
                                if (zetas != NULL && zetas[ic] > 0) {
                                        quad zeta = zetas[ic];
                                        aeff = p * zeta / (p + zeta);
                                        fac *= sqrtq(zeta / (p + zeta));
                                }
                                boys_function(L, aeff * (X*X+Y*Y+Z*Z), F);
                                hermite_R(R, Rbuf, L, aeff, X, Y, Z, F);
                                for (a = 0; a < nci; a++) {
                                for (b = 0; b < ncj; b++) {
                                        int nx = lxa[a] + lxb[b];
                                        int ny = lya[a] + lyb[b];
                                        int nz = lza[a] + lzb[b];
                                        quad s = 0;
                                        for (t = 0; t <= nx; t++) {
                                        for (u = 0; u <= ny; u++) {
                                                quad exy = pp->E[0][lxa[a]][lxb[b]][t]
                                                         * pp->E[1][lya[a]][lyb[b]][u];
                                                for (v = 0; v <= nz; v++) {
                                                        s += exy * pp->E[2][lza[a]][lzb[b]][v]
                                                           * R[(t*L1+u)*L1+v];
                                                }
                                        } }
                                        out[(long)(offi[ish]+a)*Nj+offj[jsh]+b] += fac * s;
                                } }
                        }
                }
        }
        free(pairs);
        free(R);
        free(Rbuf);
}
        free(offi);
        return 0;
}

/* ------------------------------------------------------------------ */
/* two-electron repulsion integrals                                   */
/* ------------------------------------------------------------------ */
static void eri_quartet(quad *buf, int li, int lj, int lk, int ll,
                        PrimPair *bra, int nbra, PrimPair *ket, int nket,
                        quad *G, quad *R, quad *Rbuf)
{
        int lxa[NCART(MAXL)], lya[NCART(MAXL)], lza[NCART(MAXL)];
        int lxb[NCART(MAXL)], lyb[NCART(MAXL)], lzb[NCART(MAXL)];
        int lxc[NCART(MAXL)], lyc[NCART(MAXL)], lzc[NCART(MAXL)];
        int lxd[NCART(MAXL)], lyd[NCART(MAXL)], lzd[NCART(MAXL)];
        quad F[4*MAXL+2];
        int nci = NCART(li);
        int ncj = NCART(lj);
        int nck = NCART(lk);
        int ncl = NCART(ll);
        int L = li + lj + lk + ll;
        int L1 = L + 1;
        int Lab = li + lj;
        int B1 = Lab + 1;
        int B3 = B1 * B1 * B1;
        int ib, ik, a, b, c, d, t, u, v, tau, nu, phi;
        cart_comps(li, lxa, lya, lza);
        cart_comps(lj, lxb, lyb, lzb);
        cart_comps(lk, lxc, lyc, lzc);
        cart_comps(ll, lxd, lyd, lzd);
        memset(buf, 0, sizeof(quad) * nci * ncj * nck * ncl);
        quad pi52 = 2 * powq(M_PIq, 2.5Q);

        for (ib = 0; ib < nbra; ib++) {
        for (ik = 0; ik < nket; ik++) {
                PrimPair *pb = bra + ib;
                PrimPair *pk = ket + ik;
                quad p = pb->p;
                quad q = pk->p;
                quad alpha = p * q / (p + q);
                quad X = pb->P[0] - pk->P[0];
                quad Y = pb->P[1] - pk->P[1];
                quad Z = pb->P[2] - pk->P[2];
                boys_function(L, alpha * (X*X+Y*Y+Z*Z), F);
                hermite_R(R, Rbuf, L, alpha, X, Y, Z, F);
                quad pref = pi52 / (p * q * sqrtq(p + q)) * pb->fac * pk->fac;

                /* G[cd][t,u,v] = sum_{tau,nu,phi} (-1)^{tau+nu+phi}
                 *                E^{cd}_{tau nu phi} R_{t+tau,u+nu,v+phi} */
                for (c = 0; c < nck; c++) {
                for (d = 0; d < ncl; d++) {
                        quad *Gcd = G + (long)(c * ncl + d) * B3;
                        int mx = lxc[c] + lxd[d];
                        int my = lyc[c] + lyd[d];
                        int mz = lzc[c] + lzd[d];
                        const quad *Ex = pk->E[0][lxc[c]][lxd[d]];
                        const quad *Ey = pk->E[1][lyc[c]][lyd[d]];
                        const quad *Ez = pk->E[2][lzc[c]][lzd[d]];
                        for (t = 0; t <= Lab; t++) {
                        for (u = 0; u <= Lab - t; u++) {
                        for (v = 0; v <= Lab - t - u; v++) {
                                quad s = 0;
                                for (tau = 0; tau <= mx; tau++) {
                                for (nu = 0; nu <= my; nu++) {
                                        quad exy = Ex[tau] * Ey[nu];
                                        if ((tau + nu) & 1) exy = -exy;
                                        const quad *Rp = R + ((t+tau)*L1+(u+nu))*L1 + v;
                                        quad s1 = 0;
                                        for (phi = 0; phi <= mz; phi++) {
                                                if (phi & 1) {
                                                        s1 -= Ez[phi] * Rp[phi];
                                                } else {
                                                        s1 += Ez[phi] * Rp[phi];
                                                }
                                        }
                                        s += exy * s1;
                                } }
                                Gcd[(t*B1+u)*B1+v] = s;
                        } } }
                } }

                for (a = 0; a < nci; a++) {
                for (b = 0; b < ncj; b++) {
                        int nx = lxa[a] + lxb[b];
                        int ny = lya[a] + lyb[b];
                        int nz = lza[a] + lzb[b];
                        const quad *Ex = pb->E[0][lxa[a]][lxb[b]];
                        const quad *Ey = pb->E[1][lya[a]][lyb[b]];
                        const quad *Ez = pb->E[2][lza[a]][lzb[b]];
                        quad *pbuf = buf + (long)(a * ncj + b) * nck * ncl;
                        for (c = 0; c < nck * ncl; c++) {
                                const quad *Gcd = G + (long)c * B3;
                                quad s = 0;
                                for (t = 0; t <= nx; t++) {
                                for (u = 0; u <= ny; u++) {
                                        quad exy = Ex[t] * Ey[u];
                                        const quad *Gp = Gcd + (t*B1+u)*B1;
                                        quad s1 = 0;
                                        for (v = 0; v <= nz; v++) {
                                                s1 += Ez[v] * Gp[v];
                                        }
                                        s += exy * s1;
                                } }
                                pbuf[c] += pref * s;
                        }
                } }
        } }
}

/*
 * out[Ni,Nj,Nk,Nl] = (ij|kl) over Cartesian functions of the four shell
 * lists.  shls = concatenation of the four shell lists, nshl = their sizes.
 * sym bits:
 *      1: list i == list j, (ij|kl) = (ji|kl)
 *      2: list k == list l, (ij|kl) = (ij|lk)
 *      4: (list i, list j) == (list k, list l), (ij|kl) = (kl|ij)
 * Returns 0 on success, -1 if the angular momentum is not supported.
 */
int QDint2e(quad *out, const int *shls, const int *nshl, int sym,
            const int *bas, const quad *env)
{
        const int *si = shls;
        const int *sj = si + nshl[0];
        const int *sk = sj + nshl[1];
        const int *sl = sk + nshl[2];
        int ni = nshl[0], nj = nshl[1], nk = nshl[2], nl = nshl[3];
        int lmax = max_ang(shls, ni+nj+nk+nl, bas);
        if (lmax > MAXL) {
                return -1;
        }
        int *offs = malloc(sizeof(int) * (ni+nj+nk+nl+4));
        int *offi = offs;
        int *offj = offi + ni + 1;
        int *offk = offj + nj + 1;
        int *offl = offk + nk + 1;
        shell_offsets(offi, si, ni, bas);
        shell_offsets(offj, sj, nj, bas);
        shell_offsets(offk, sk, nk, bas);
        shell_offsets(offl, sl, nl, bas);
        long Nj = offj[nj];
        long Nk = offk[nk];
        long Nl = offl[nl];
        int sym_ij = sym & 1;
        int sym_kl = sym & 2;
        int sym_ijkl = sym & 4;

        int npij = 0, npkl = 0;
        int *pij = malloc(sizeof(int) * 2 * ni * nj);
        int *pkl = malloc(sizeof(int) * 2 * nk * nl);
        int I, J;
        for (I = 0; I < ni; I++) {
        for (J = 0; J < nj; J++) {
                if (!sym_ij || J <= I) {
                        pij[npij*2+0] = I;
                        pij[npij*2+1] = J;
                        npij++;
                }
        } }
        for (I = 0; I < nk; I++) {
        for (J = 0; J < nl; J++) {
                if (!sym_kl || J <= I) {
                        pkl[npkl*2+0] = I;
                        pkl[npkl*2+1] = J;
                        npkl++;
                }
        } }
        int mpij = max_prims(si, ni, bas) * max_prims(sj, nj, bas);
        int mpkl = max_prims(sk, nk, bas) * max_prims(sl, nl, bas);

#pragma omp parallel
{
        int L1max = 4 * MAXL + 1;
        int B1max = 2 * MAXL + 1;
        PrimPair *bra = malloc(sizeof(PrimPair) * mpij);
        PrimPair *ket = malloc(sizeof(PrimPair) * mpkl);
        quad *R = malloc(sizeof(quad) * L1max * L1max * L1max);
        quad *Rbuf = malloc(sizeof(quad) * L1max * L1max * L1max);
        quad *G = malloc(sizeof(quad) * NCART(lmax) * NCART(lmax)
                         * B1max * B1max * B1max);
        quad *buf = malloc(sizeof(quad) * NCART(lmax) * NCART(lmax)
                           * NCART(lmax) * NCART(lmax));
        int ij, kl, a, b, c, d, nbra, nket;
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < npij; ij++) {
                int ish = pij[ij*2+0];
                int jsh = pij[ij*2+1];
                int ISH = si[ish];
                int JSH = sj[jsh];
                int li = bas[ISH*BAS_SLOTS+ANG_OF];
                int lj = bas[JSH*BAS_SLOTS+ANG_OF];
                int nci = NCART(li);
                int ncj = NCART(lj);
                nbra = make_pairs(bra, ISH, JSH, bas, env);
                int klmax = sym_ijkl ? ij + 1 : npkl;
                for (kl = 0; kl < klmax; kl++) {
                        int ksh = pkl[kl*2+0];
                        int lsh = pkl[kl*2+1];
                        int KSH = sk[ksh];
                        int LSH = sl[lsh];
                        int lk = bas[KSH*BAS_SLOTS+ANG_OF];
                        int ll = bas[LSH*BAS_SLOTS+ANG_OF];
                        int nck = NCART(lk);
                        int ncl = NCART(ll);
                        nket = make_pairs(ket, KSH, LSH, bas, env);
                        eri_quartet(buf, li, lj, lk, ll, bra, nbra, ket, nket,
                                    G, R, Rbuf);
                        for (a = 0; a < nci; a++) {
                        for (b = 0; b < ncj; b++) {
                        for (c = 0; c < nck; c++) {
                        for (d = 0; d < ncl; d++) {
                                quad val = buf[((a*ncj+b)*nck+c)*ncl+d];
                                long i = offi[ish] + a;
                                long j = offj[jsh] + b;
                                long k = offk[ksh] + c;
                                long l = offl[lsh] + d;
#define PUT(p,q,r,s)    out[(((p)*Nj+(q))*Nk+(r))*Nl+(s)] = val
                                PUT(i,j,k,l);
                                if (sym_ij) PUT(j,i,k,l);
                                if (sym_kl) PUT(i,j,l,k);
                                if (sym_ij && sym_kl) PUT(j,i,l,k);
                                if (sym_ijkl) {
                                        PUT(k,l,i,j);
                                        if (sym_ij) PUT(k,l,j,i);
                                        if (sym_kl) PUT(l,k,i,j);
                                        if (sym_ij && sym_kl) PUT(l,k,j,i);
                                }
#undef PUT
                        } } } }
                }
        }
        free(bra);
        free(ket);
        free(R);
        free(Rbuf);
        free(G);
        free(buf);
}
        free(offs);
        free(pij);
        free(pkl);
        return 0;
}


/* ------------------------------------------------------------------ */
/* integral-direct J and K for a real symmetric density matrix        */
/* ------------------------------------------------------------------ */
/*
 * vj[i,j] = sum_kl (ij|kl) dm[l,k],   vk[i,l] = sum_jk (ij|kl) dm[j,k]
 * over the Cartesian functions of the shell list shls (one list for all
 * four indices, 8-fold permutational symmetry).  The ERIs are evaluated on
 * the fly in quad precision; memory O(N^2) per thread.
 * Returns 0 on success, -1 if the angular momentum is not supported.
 */
int QDjk_direct(quad *vj, quad *vk, const quad *dm, const int *shls, int nsh,
                const int *bas, const quad *env)
{
        int lmax = max_ang(shls, nsh, bas);
        if (lmax > MAXL) {
                return -1;
        }
        int *offs = malloc(sizeof(int) * (nsh + 1));
        shell_offsets(offs, shls, nsh, bas);
        long N = offs[nsh];
        int npair = nsh * (nsh + 1) / 2;
        int *pairs = malloc(sizeof(int) * 2 * npair);
        int I, J, n = 0;
        for (I = 0; I < nsh; I++) {
        for (J = 0; J <= I; J++) {
                pairs[n*2+0] = I;
                pairs[n*2+1] = J;
                n++;
        } }
        int mp = max_prims(shls, nsh, bas);
        memset(vj, 0, sizeof(quad) * N * N);
        memset(vk, 0, sizeof(quad) * N * N);

#pragma omp parallel
{
        int L1max = 4 * MAXL + 1;
        int B1max = 2 * MAXL + 1;
        PrimPair *bra = malloc(sizeof(PrimPair) * mp * mp);
        PrimPair *ket = malloc(sizeof(PrimPair) * mp * mp);
        quad *R = malloc(sizeof(quad) * L1max * L1max * L1max);
        quad *Rbuf = malloc(sizeof(quad) * L1max * L1max * L1max);
        quad *G = malloc(sizeof(quad) * NCART(lmax) * NCART(lmax)
                         * B1max * B1max * B1max);
        quad *buf = malloc(sizeof(quad) * NCART(lmax) * NCART(lmax)
                           * NCART(lmax) * NCART(lmax));
        quad *jt = calloc(N * N, sizeof(quad));
        quad *kt = calloc(N * N, sizeof(quad));
        int ij, kl, a, b, c, d, nbra, nket, ip;
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < npair; ij++) {
                int ish = pairs[ij*2+0];
                int jsh = pairs[ij*2+1];
                int li = bas[shls[ish]*BAS_SLOTS+ANG_OF];
                int lj = bas[shls[jsh]*BAS_SLOTS+ANG_OF];
                int nci = NCART(li);
                int ncj = NCART(lj);
                nbra = make_pairs(bra, shls[ish], shls[jsh], bas, env);
                for (kl = 0; kl <= ij; kl++) {
                        int ksh = pairs[kl*2+0];
                        int lsh = pairs[kl*2+1];
                        int lk = bas[shls[ksh]*BAS_SLOTS+ANG_OF];
                        int ll = bas[shls[lsh]*BAS_SLOTS+ANG_OF];
                        int nck = NCART(lk);
                        int ncl = NCART(ll);
                        nket = make_pairs(ket, shls[ksh], shls[lsh], bas, env);
                        eri_quartet(buf, li, lj, lk, ll, bra, nbra, ket, nket,
                                    G, R, Rbuf);
                        /* distinct shell-level permutations of (IJ|KL) */
                        int perm[8][4];
                        int np = 0;
                        int s_ij = (ish != jsh);
                        int s_kl = (ksh != lsh);
                        int s_ijkl = (ij != kl);
                        int x, y, z;
                        for (z = 0; z <= s_ijkl; z++) {
                        for (x = 0; x <= s_ij; x++) {
                        for (y = 0; y <= s_kl; y++) {
                                /* perm[] stores which of (i,j,k,l) goes to each slot */
                                int q[4] = {0, 1, 2, 3};
                                if (x) { q[0] = 1; q[1] = 0; }
                                if (y) { q[2] = 3; q[3] = 2; }
                                if (z) {
                                        int t0 = q[0], t1 = q[1];
                                        q[0] = q[2]; q[1] = q[3];
                                        q[2] = t0; q[3] = t1;
                                }
                                perm[np][0] = q[0]; perm[np][1] = q[1];
                                perm[np][2] = q[2]; perm[np][3] = q[3];
                                np++;
                        } } }
                        for (a = 0; a < nci; a++) {
                        for (b = 0; b < ncj; b++) {
                        for (c = 0; c < nck; c++) {
                        for (d = 0; d < ncl; d++) {
                                quad val = buf[((a*ncj+b)*nck+c)*ncl+d];
                                long idx[4];
                                idx[0] = offs[ish] + a;
                                idx[1] = offs[jsh] + b;
                                idx[2] = offs[ksh] + c;
                                idx[3] = offs[lsh] + d;
                                for (ip = 0; ip < np; ip++) {
                                        long p0 = idx[perm[ip][0]];
                                        long p1 = idx[perm[ip][1]];
                                        long p2 = idx[perm[ip][2]];
                                        long p3 = idx[perm[ip][3]];
                                        jt[p0*N+p1] += val * dm[p3*N+p2];
                                        kt[p0*N+p3] += val * dm[p1*N+p2];
                                }
                        } } } }
                }
        }
#pragma omp critical
        {
                long m;
                for (m = 0; m < N*N; m++) {
                        vj[m] += jt[m];
                        vk[m] += kt[m];
                }
        }
        free(bra);
        free(ket);
        free(R);
        free(Rbuf);
        free(G);
        free(buf);
        free(jt);
        free(kt);
}
        free(offs);
        free(pairs);
        return 0;
}
