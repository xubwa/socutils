/*
 * Becke partition weights of one atom's grid (socutils.dft.grids.get_partition):
 * pyscf's VXCgen_grid (Becke 1988 cell functions, three iterations of
 * f(s) = (3 - s^2) s / 2, optional size adjustment a_ij (1 - mu^2)), the same
 * arithmetic, but
 *   - only the owner atom's normalised weight P_owner / sum_i P_i per point is
 *     returned (no natm x ngrid array, no Python normalisation),
 *   - blocks of BLK points per OpenMP thread, the pair loop over the block's
 *     points vectorised (#pragma omp simd), the 1 / R_ij table shared.
 * The cost is still natm^2 / 2 per point.
 *
 * w[g] (n): out; coords (3, n) (Fortran order as pyscf passes them, x then y
 * then z rows); atm (natm, 3); radii (natm, natm) or NULL.
 */
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef BLK
#define BLK 64
#endif

void SOCDFTbecke_owner(double *w, const double *coords, int n, int owner,
                       const double *atm, const double *radii, int natm)
{
        double *inv = malloc(sizeof(double) * natm * natm);
        int i, j;
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        const double dx = atm[i * 3] - atm[j * 3], dy = atm[i * 3 + 1] - atm[j * 3 + 1],
                                     dz = atm[i * 3 + 2] - atm[j * 3 + 2];
                        inv[i * natm + j] = 1 / sqrt(dx * dx + dy * dy + dz * dz);
                }
        }
        const size_t N = n;
#pragma omp parallel
{
        double *r = malloc(sizeof(double) * natm * BLK);
        double *P = malloc(sizeof(double) * natm * BLK);
        int b;
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < (n + BLK - 1) / BLK; b++) {
                const int g0 = b * BLK, ng = g0 + BLK <= n ? BLK : n - g0;
                int ii, jj, g;
                for (ii = 0; ii < natm; ii++) {
                        double *ri = r + (size_t)ii * BLK, *pi = P + (size_t)ii * BLK;
                        const double ax = atm[ii * 3], ay = atm[ii * 3 + 1], az = atm[ii * 3 + 2];
#pragma omp simd
                        for (g = 0; g < ng; g++) {
                                const double dx = coords[g0 + g] - ax;
                                const double dy = coords[N + g0 + g] - ay;
                                const double dz = coords[2 * N + g0 + g] - az;
                                ri[g] = sqrt(dx * dx + dy * dy + dz * dz);
                                pi[g] = 1;
                        }
                }
                for (ii = 0; ii < natm; ii++) {
                        const double *ri = r + (size_t)ii * BLK;
                        double *pi = P + (size_t)ii * BLK;
                        for (jj = 0; jj < ii; jj++) {
                                const double *rj = r + (size_t)jj * BLK;
                                double *pj = P + (size_t)jj * BLK;
                                const double fac = inv[ii * natm + jj];
                                const double a = radii != NULL ? radii[ii * natm + jj] : 0;
#pragma omp simd
                                for (g = 0; g < ng; g++) {
                                        double s = (ri[g] - rj[g]) * fac;
                                        s += a * (1 - s * s);
                                        s = (3 - s * s) * s * .5;
                                        s = (3 - s * s) * s * .5;
                                        s = ((3 - s * s) * s * .5) * .5;
                                        pi[g] *= .5 - s;
                                        pj[g] *= .5 + s;
                                }
                        }
                }
                for (g = 0; g < ng; g++) {
                        double sum = 0;
                        for (ii = 0; ii < natm; ii++) sum += P[(size_t)ii * BLK + g];
                        w[g0 + g] = P[(size_t)owner * BLK + g] / sum;
                }
        }
        free(r);
        free(P);
}
        free(inv);
}
