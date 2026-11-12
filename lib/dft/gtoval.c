/*
 * Spherical GTO values (and first derivatives) on a block of grid points, the
 * drop-in of pyscf's GTOval_sph / GTOval_sph_deriv1 (the same signature,
 * layout, normalisation and non0table convention) for the XC passes of
 * vxc_rks.c.  pyscf's evaluators call the scalar libm exp per primitive and
 * point (cc-pVDZ C: 9 s primitives); here the radial parts run as SIMD loops
 * over the points (this file is compiled with -ffast-math so that exp maps to
 * glibc's vector libmvec variants), the angular parts and the cart -> sph
 * transform as well.  Shells of l <= GV_LMAX (the caller keeps pyscf's
 * evaluator otherwise).
 *
 *   ao (ncomp, nao, ngrids), nao = ao_loc[sh1] - ao_loc[sh0]; coord (3, ngrids);
 *   non0table[(g / 56) * nbas + sh]: 0 -> the shell's values are zero on those
 *   56 points (as pyscf).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gtoval_c2s.h"

#define ATM_SLOTS 6
#define PTR_COORD 1
#define BAS_SLOTS 8
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define PTR_EXP 5
#define PTR_COEFF 6
#define GV_BLK 56
#define GV_LMAX 3
#define GV_NCMAX 16
#define GV_EXPCUT 50.        /* exp(-50) = 2e-22: primitives below on a whole 56-point block are skipped */

int SOCDFTgtoval_lmax() { return GV_LMAX; }

static inline double common_fac_sp(int l)
{
        return l == 0 ? 0.282094791773878143 : (l == 1 ? 0.488602511902919921 : 1.);
}

/* ld: the row stride between components (pyscf: nao of the slice), row0: the row of the slice's
 * first AO */
/* ostride: the point stride of the output rows, cstride: of the coordinate components (both ngrids
 * for pyscf's convention) */
static void eval_shells_s(int deriv, int ngrids, size_t ostride, size_t cstride, int *shls_slice, int *ao_loc,
                          double *ao, size_t ld, size_t row0, double *coord, uint8_t *non0table, int *atm, int *bas,
                          int nbas, double *env, double *rowmax);

static void eval_shells(int deriv, int ngrids, int *shls_slice, int *ao_loc, double *ao, size_t ld, size_t row0,
                        double *coord, uint8_t *non0table, int *atm, int *bas, int nbas, double *env,
                        double *rowmax)
{
        eval_shells_s(deriv, ngrids, ngrids, ngrids, shls_slice, ao_loc, ao, ld, row0, coord, non0table, atm, bas,
                      nbas, env, rowmax);
}

static void eval_shells_s(int deriv, int ngrids, size_t ostride, size_t cstride, int *shls_slice, int *ao_loc,
                          double *ao, size_t ld, size_t row0, double *coord, uint8_t *non0table, int *atm, int *bas,
                          int nbas, double *env, double *rowmax)
{
        const int sh0 = shls_slice[0], sh1 = shls_slice[1];
        const size_t nao = ld;
        const int ncomp = deriv == 3 ? 20 : (deriv == 2 ? 10 : (deriv ? 4 : 1));
        const int nblk = (ngrids + GV_BLK - 1) / GV_BLK;
        const double *gx = coord, *gy = coord + cstride, *gz = coord + 2 * cstride;
        double rad[GV_NCMAX][GV_BLK], rad1[GV_NCMAX][GV_BLK], rad2[GV_NCMAX][GV_BLK], rad3[GV_NCMAX][GV_BLK];
        double cart[20][10][GV_BLK];
        int sh, b;
        for (sh = sh0; sh < sh1; sh++) {
                const int *bs = bas + sh * BAS_SLOTS;
                const int l = bs[ANG_OF], np = bs[NPRIM_OF], nc = bs[NCTR_OF];
                const int deg = 2 * l + 1, ncart = (l + 1) * (l + 2) / 2;
                const double *al = env + bs[PTR_EXP], *cf = env + bs[PTR_COEFF];
                const double *R = env + atm[bs[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double fac = common_fac_sp(l);
                const size_t aoid = row0 + ao_loc[sh] - ao_loc[sh0];
                for (b = 0; b < nblk; b++) {
                        const int g0 = b * GV_BLK, bg = g0 + GV_BLK <= ngrids ? GV_BLK : ngrids - g0;
                        int c, k, m, g, p;
                        if (!non0table[(size_t)b * nbas + sh]) {
                                for (k = 0; k < ncomp; k++)
                                        for (c = 0; c < nc * deg; c++)
                                                memset(ao + ((size_t)k * nao + aoid + c) * ostride + g0, 0, sizeof(double) * bg);
                                continue;
                        }
                        double dx[GV_BLK], dy[GV_BLK], dz[GV_BLK], r2[GV_BLK];
#pragma omp simd
                        for (g = 0; g < bg; g++) {
                                dx[g] = gx[g0 + g] - R[0];
                                dy[g] = gy[g0 + g] - R[1];
                                dz[g] = gz[g0 + g] - R[2];
                                r2[g] = dx[g] * dx[g] + dy[g] * dy[g] + dz[g] * dz[g];
                        }
                        double xp[GV_LMAX + 1][GV_BLK], yp[GV_LMAX + 1][GV_BLK], zp[GV_LMAX + 1][GV_BLK];
                        {
                                int t;
#pragma omp simd
                                for (g = 0; g < bg; g++) { xp[0][g] = yp[0][g] = zp[0][g] = 1; }
                                for (t = 1; t <= l; t++) {
#pragma omp simd
                                        for (g = 0; g < bg; g++) {
                                                xp[t][g] = xp[t - 1][g] * dx[g];
                                                yp[t][g] = yp[t - 1][g] * dy[g];
                                                zp[t][g] = zp[t - 1][g] * dz[g];
                                        }
                                }
                        }
                        double r2min = r2[0];
                        for (g = 1; g < bg; g++) r2min = r2[g] < r2min ? r2[g] : r2min;
                        for (c = 0; c < nc; c++)
                                for (g = 0; g < bg; g++) rad[c][g] = rad1[c][g] = rad2[c][g] = rad3[c][g] = 0;
                        for (p = 0; p < np; p++) {
                                const double a = al[p], m2a = -2 * a;
                                /* the primitive below e^-GV_EXPCUT on the whole block: skipped */
                                if (a * r2min > GV_EXPCUT) continue;
                                double e[GV_BLK];
#pragma omp simd
                                for (g = 0; g < bg; g++) e[g] = exp(-a * r2[g]) * fac;
                                for (c = 0; c < nc; c++) {
                                        const double cc = cf[c * np + p], cd = cc * m2a, cd2 = cd * m2a, cd3 = cd2 * m2a;
                                        if (deriv == 3) {
#pragma omp simd
                                                for (g = 0; g < bg; g++) {
                                                        rad[c][g] += cc * e[g];
                                                        rad1[c][g] += cd * e[g];
                                                        rad2[c][g] += cd2 * e[g];
                                                        rad3[c][g] += cd3 * e[g];
                                                }
                                        } else if (deriv == 2) {
#pragma omp simd
                                                for (g = 0; g < bg; g++) {
                                                        rad[c][g] += cc * e[g];
                                                        rad1[c][g] += cd * e[g];
                                                        rad2[c][g] += cd2 * e[g];
                                                }
                                        } else {
#pragma omp simd
                                                for (g = 0; g < bg; g++) {
                                                        rad[c][g] += cc * e[g];
                                                        rad1[c][g] += cd * e[g];
                                                }
                                        }
                                }
                        }
                        for (c = 0; c < nc; c++) {
                                const double *f = rad[c], *f1 = rad1[c], *f2 = rad2[c], *f3 = rad3[c];
                                /* cartesian x^i y^j z^k f (and the derivatives) in pyscf's order, from the
                                 * power tables of the block */
                                int ic = 0, i, j;
                                for (i = l; i >= 0; i--)
                                for (j = l - i; j >= 0; j--, ic++) {
                                        const int kk = l - i - j;
                                        const double *xi = xp[i], *yj = yp[j], *zk = zp[kk];
                                        const double *xm = xp[i > 0 ? i - 1 : 0], *ym = yp[j > 0 ? j - 1 : 0];
                                        const double *zm = zp[kk > 0 ? kk - 1 : 0];
                                        const double fi = i, fj = j, fk = kk;
                                        if (deriv == 3) {
                                                /* the polynomial's derivatives Pd[tx][ty][tz] up to order 3, then
                                                 * d^n (P f) by the chain rule (f_n = sum c (-2a)^n e) */
                                                double Pd[4][4][4][GV_BLK];
                                                int tx, ty, tz;
                                                for (tx = 0; tx <= 3; tx++)
                                                for (ty = 0; ty <= 3 - tx; ty++)
                                                for (tz = 0; tz <= 3 - tx - ty; tz++) {
                                                        double cx = 1, cy = 1, cz = 1;
                                                        int t;
                                                        for (t = 0; t < tx; t++) cx *= i - t;
                                                        for (t = 0; t < ty; t++) cy *= j - t;
                                                        for (t = 0; t < tz; t++) cz *= kk - t;
                                                        const double cxyz = cx * cy * cz;
                                                        double *o_ = Pd[tx][ty][tz];
                                                        if (cxyz == 0) {
                                                                for (g = 0; g < bg; g++) o_[g] = 0;
                                                                continue;
                                                        }
                                                        const double *px = xp[i - tx], *py = yp[j - ty], *pz = zp[kk - tz];
#pragma omp simd
                                                        for (g = 0; g < bg; g++) o_[g] = cxyz * px[g] * py[g] * pz[g];
                                                }
                                                /* components: 0; x y z; xx xy xz yy yz zz; xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz */
                                                static const int ax3[10][3] = {{0,0,0},{0,0,1},{0,0,2},{0,1,1},{0,1,2},{0,2,2},
                                                                               {1,1,1},{1,1,2},{1,2,2},{2,2,2}};
                                                static const int ax2[6][2] = {{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};
                                                const double *X[3] = {dx, dy, dz};
#define PD1(a) Pd[(a) == 0][(a) == 1][(a) == 2]
#define PD2(a, b) Pd[((a) == 0) + ((b) == 0)][((a) == 1) + ((b) == 1)][((a) == 2) + ((b) == 2)]
#define PD3(a, b, c_) Pd[((a) == 0) + ((b) == 0) + ((c_) == 0)][((a) == 1) + ((b) == 1) + ((c_) == 1)][((a) == 2) + ((b) == 2) + ((c_) == 2)]
                                                const double *P0 = Pd[0][0][0];
#pragma omp simd
                                                for (g = 0; g < bg; g++) cart[0][ic][g] = P0[g] * f[g];
                                                int q;
                                                for (q = 0; q < 3; q++) {
                                                        const double *Pa = PD1(q), *Xa = X[q];
#pragma omp simd
                                                        for (g = 0; g < bg; g++) cart[1 + q][ic][g] = Pa[g] * f[g] + Xa[g] * P0[g] * f1[g];
                                                }
                                                for (q = 0; q < 6; q++) {
                                                        const int a = ax2[q][0], b2 = ax2[q][1];
                                                        const double *Pab = PD2(a, b2), *Pa = PD1(a), *Pb = PD1(b2);
                                                        const double *Xa = X[a], *Xb = X[b2];
                                                        const double dab = a == b2;
#pragma omp simd
                                                        for (g = 0; g < bg; g++)
                                                                cart[4 + q][ic][g] = Pab[g] * f[g] + (Xa[g] * Pb[g] + Xb[g] * Pa[g] + dab * P0[g]) * f1[g]
                                                                                   + Xa[g] * Xb[g] * P0[g] * f2[g];
                                                }
                                                for (q = 0; q < 10; q++) {
                                                        const int a = ax3[q][0], b2 = ax3[q][1], c3 = ax3[q][2];
                                                        const double *Pabc = PD3(a, b2, c3), *Pab = PD2(a, b2), *Pac = PD2(a, c3);
                                                        const double *Pbc = PD2(b2, c3), *Pa = PD1(a), *Pb = PD1(b2), *Pc = PD1(c3);
                                                        const double *Xa = X[a], *Xb = X[b2], *Xc = X[c3];
                                                        const double dab = a == b2, dac = a == c3, dbc = b2 == c3;
#pragma omp simd
                                                        for (g = 0; g < bg; g++)
                                                                cart[10 + q][ic][g] = Pabc[g] * f[g]
                                                                        + (Pab[g] * Xc[g] + Pac[g] * Xb[g] + Pbc[g] * Xa[g]
                                                                           + dbc * Pa[g] + dac * Pb[g] + dab * Pc[g]) * f1[g]
                                                                        + (Pa[g] * Xb[g] * Xc[g] + Pb[g] * Xa[g] * Xc[g] + Pc[g] * Xa[g] * Xb[g]
                                                                           + P0[g] * (dab * Xc[g] + dac * Xb[g] + dbc * Xa[g])) * f2[g]
                                                                        + P0[g] * Xa[g] * Xb[g] * Xc[g] * f3[g];
                                                }
#undef PD1
#undef PD2
#undef PD3
                                                continue;
                                        }
                                        if (deriv == 2) {
                                                /* d2 (P f) = P'' f + (x_a P_b + x_b P_a + delta_ab P) f1 + x_a x_b P f2 */
                                                const double *xmm = xp[i > 1 ? i - 2 : 0], *ymm = yp[j > 1 ? j - 2 : 0];
                                                const double *zmm = zp[kk > 1 ? kk - 2 : 0];
                                                const double fii = i * (i - 1), fjj = j * (j - 1), fkk = kk * (kk - 1);
                                                const double fij = i * j, fik = i * kk, fjk = j * kk;
#pragma omp simd
                                                for (g = 0; g < bg; g++) {
                                                        const double X = dx[g], Y = dy[g], Z = dz[g];
                                                        const double F = f[g], F1 = f1[g], F2 = f2[g];
                                                        const double P = xi[g] * yj[g] * zk[g];
                                                        const double Px = fi * xm[g] * yj[g] * zk[g];
                                                        const double Py = fj * xi[g] * ym[g] * zk[g];
                                                        const double Pz = fk * xi[g] * yj[g] * zm[g];
                                                        const double Pxx = fii * xmm[g] * yj[g] * zk[g];
                                                        const double Pyy = fjj * xi[g] * ymm[g] * zk[g];
                                                        const double Pzz = fkk * xi[g] * yj[g] * zmm[g];
                                                        const double Pxy = fij * xm[g] * ym[g] * zk[g];
                                                        const double Pxz = fik * xm[g] * yj[g] * zm[g];
                                                        const double Pyz = fjk * xi[g] * ym[g] * zm[g];
                                                        cart[0][ic][g] = P * F;
                                                        cart[1][ic][g] = Px * F + X * P * F1;
                                                        cart[2][ic][g] = Py * F + Y * P * F1;
                                                        cart[3][ic][g] = Pz * F + Z * P * F1;
                                                        cart[4][ic][g] = Pxx * F + (2 * X * Px + P) * F1 + X * X * P * F2;
                                                        cart[5][ic][g] = Pxy * F + (Y * Px + X * Py) * F1 + X * Y * P * F2;
                                                        cart[6][ic][g] = Pxz * F + (Z * Px + X * Pz) * F1 + X * Z * P * F2;
                                                        cart[7][ic][g] = Pyy * F + (2 * Y * Py + P) * F1 + Y * Y * P * F2;
                                                        cart[8][ic][g] = Pyz * F + (Z * Py + Y * Pz) * F1 + Y * Z * P * F2;
                                                        cart[9][ic][g] = Pzz * F + (2 * Z * Pz + P) * F1 + Z * Z * P * F2;
                                                }
                                                continue;
                                        }
#pragma omp simd
                                        for (g = 0; g < bg; g++) {
                                                const double yz = yj[g] * zk[g], P = xi[g] * yz;
                                                cart[0][ic][g] = P * f[g];
                                                if (deriv) {
                                                        const double Pf1 = P * f1[g];
                                                        cart[1][ic][g] = fi * xm[g] * yz * f[g] + dx[g] * Pf1;
                                                        cart[2][ic][g] = fj * xi[g] * ym[g] * zk[g] * f[g] + dy[g] * Pf1;
                                                        cart[3][ic][g] = fk * xi[g] * yj[g] * zm[g] * f[g] + dz[g] * Pf1;
                                                }
                                        }
                                }
                                for (k = 0; k < ncomp; k++) {
                                        double *o = ao + ((size_t)k * nao + aoid + (size_t)c * deg) * ostride + g0;
                                        for (m = 0; m < deg; m++) {
                                                double *om = o + (size_t)m * ostride, mx = 0;
                                                if (l <= 1) {
#pragma omp simd reduction(max:mx)
                                                        for (g = 0; g < bg; g++) {
                                                                const double v = cart[k][m][g];
                                                                om[g] = v;
                                                                mx = fabs(v) > mx ? fabs(v) : mx;
                                                        }
                                                } else if (l == 2) {
#pragma omp simd reduction(max:mx)
                                                        for (g = 0; g < bg; g++) {
                                                                double v = 0;
                                                                for (ic = 0; ic < 6; ic++) v += C2S_L2[m][ic] * cart[k][ic][g];
                                                                om[g] = v;
                                                                mx = fabs(v) > mx ? fabs(v) : mx;
                                                        }
                                                } else {
#pragma omp simd reduction(max:mx)
                                                        for (g = 0; g < bg; g++) {
                                                                double v = 0;
                                                                for (ic = 0; ic < 10; ic++) v += C2S_L3[m][ic] * cart[k][ic][g];
                                                                om[g] = v;
                                                                mx = fabs(v) > mx ? fabs(v) : mx;
                                                        }
                                                }
                                                if (rowmax) {
                                                        double *rm = rowmax + aoid + (size_t)c * deg + m;
                                                        if (mx > *rm) *rm = mx;
                                                }
                                        }
                                }
                                (void)ncart;
                        }
                }
        }
}

void SOCDFTgtoval_sph(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord, uint8_t *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        (void)natm;
        eval_shells(0, ngrids, shls_slice, ao_loc, ao, ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]], 0, coord,
                    non0table, atm, bas, nbas, env, NULL);
}

void SOCDFTgtoval_sph_deriv2(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord,
                             uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env)
{
        (void)natm;
        eval_shells(2, ngrids, shls_slice, ao_loc, ao, ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]], 0, coord,
                    non0table, atm, bas, nbas, env, NULL);
}

void SOCDFTgtoval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord,
                             uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env)
{
        (void)natm;
        eval_shells(1, ngrids, shls_slice, ao_loc, ao, ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]], 0, coord,
                    non0table, atm, bas, nbas, env, NULL);
}

/* the shells [sh0, sh1) into rows row0 .. of ao (ncomp, ld, ngrids) */
void SOCDFTgtoval_ld(int deriv, int ngrids, int sh0, int sh1, int *ao_loc, double *ao, size_t ld, size_t row0,
                     double *coord, uint8_t *non0table, int *atm, int *bas, int nbas, double *env, double *rowmax)
{
        int sl[2] = {sh0, sh1};
        eval_shells(deriv, ngrids, sl, ao_loc, ao, ld, row0, coord, non0table, atm, bas, nbas, env, rowmax);
}

/* pyscf's eval_ao (GTOval_sph(_deriv1/2)) of all shells on ngrids points, OpenMP over runs of 56-point
 * blocks: ao (ncomp, nao, ngrids), coord (3, ngrids), non0table (ngrids / 56, nbas) or NULL (all) */
void SOCDFTeval_ao(int deriv, int ngrids, double *ao, double *coord, uint8_t *non0table, int *ao_loc, int *atm,
                   int *bas, int nbas, double *env)
{
        const int nblk = (ngrids + GV_BLK - 1) / GV_BLK, per = 4;      /* 4 blocks of 56 points per task */
        (void)0;
        const size_t nao = ao_loc[nbas];
        uint8_t *ones = NULL;
        if (non0table == NULL) {
                ones = malloc((size_t)nblk * nbas);
                memset(ones, 1, (size_t)nblk * nbas);
                non0table = ones;
        }
        int t;
#pragma omp parallel for schedule(dynamic, 1)
        for (t = 0; t < (nblk + per - 1) / per; t++) {
                const int b0 = t * per, g0 = b0 * GV_BLK;
                const int n = g0 + per * GV_BLK <= ngrids ? per * GV_BLK : ngrids - g0;
                int sl[2] = {0, nbas};
                eval_shells_s(deriv, n, ngrids, ngrids, sl, ao_loc, ao + g0, nao, 0, coord + g0,
                              non0table + (size_t)b0 * nbas, atm, bas, nbas, env, NULL);
        }
        free(ones);
}
