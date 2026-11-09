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
static void eval_shells(int deriv, int ngrids, int *shls_slice, int *ao_loc, double *ao, size_t ld, size_t row0,
                        double *coord, uint8_t *non0table, int *atm, int *bas, int nbas, double *env,
                        double *rowmax)
{
        const int sh0 = shls_slice[0], sh1 = shls_slice[1];
        const size_t nao = ld;
        const int ncomp = deriv ? 4 : 1;
        const int nblk = (ngrids + GV_BLK - 1) / GV_BLK;
        const double *gx = coord, *gy = coord + ngrids, *gz = coord + 2 * ngrids;
        double rad[GV_NCMAX][GV_BLK], rad1[GV_NCMAX][GV_BLK];
        double cart[4][10][GV_BLK];
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
                                                memset(ao + ((size_t)k * nao + aoid + c) * ngrids + g0, 0, sizeof(double) * bg);
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
                                for (g = 0; g < bg; g++) rad[c][g] = rad1[c][g] = 0;
                        for (p = 0; p < np; p++) {
                                const double a = al[p], m2a = -2 * a;
                                /* the primitive below e^-GV_EXPCUT on the whole block: skipped */
                                if (a * r2min > GV_EXPCUT) continue;
                                double e[GV_BLK];
#pragma omp simd
                                for (g = 0; g < bg; g++) e[g] = exp(-a * r2[g]) * fac;
                                for (c = 0; c < nc; c++) {
                                        const double cc = cf[c * np + p], cd = cc * m2a;
#pragma omp simd
                                        for (g = 0; g < bg; g++) {
                                                rad[c][g] += cc * e[g];
                                                rad1[c][g] += cd * e[g];
                                        }
                                }
                        }
                        for (c = 0; c < nc; c++) {
                                const double *f = rad[c], *f1 = rad1[c];
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
                                        double *o = ao + ((size_t)k * nao + aoid + (size_t)c * deg) * ngrids + g0;
                                        for (m = 0; m < deg; m++) {
                                                double *om = o + (size_t)m * ngrids, mx = 0;
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
