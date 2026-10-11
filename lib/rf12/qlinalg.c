/* Copyright 2026 The socutils developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Real symmetric (generalized) eigensolver in IEEE binary128 for socutils.rf12.
 *
 * Householder tridiagonalization + implicit QL with shifts (the EISPACK
 * tred2/tql2 pair, following the public-domain JAMA translation), and a
 * Cholesky reduction for H c = e S c.  All matrices are row-major n x n.
 */
#include <stdlib.h>
#include <string.h>
#include <quadmath.h>

typedef __float128 quad;
#define V(i,j) v[(long)(i)*n+(j)]

static void tred2(int n, quad *v, quad *d, quad *e)
{
        int i, j, k;
        for (j = 0; j < n; j++) d[j] = V(n-1, j);
        for (i = n-1; i > 0; i--) {
                quad scale = 0, h = 0;
                for (k = 0; k < i; k++) scale += fabsq(d[k]);
                if (scale == 0) {
                        e[i] = d[i-1];
                        for (j = 0; j < i; j++) {
                                d[j] = V(i-1, j);
                                V(i, j) = 0;
                                V(j, i) = 0;
                        }
                } else {
                        for (k = 0; k < i; k++) {
                                d[k] /= scale;
                                h += d[k] * d[k];
                        }
                        quad f = d[i-1];
                        quad g = sqrtq(h);
                        if (f > 0) g = -g;
                        e[i] = scale * g;
                        h = h - f * g;
                        d[i-1] = f - g;
                        /* e = A d with A the active lower triangle V(k,j), k >= j */
#pragma omp parallel for schedule(dynamic, 8) private(k)
                        for (j = 0; j < i; j++) {
                                quad gj = 0;
                                for (k = 0; k <= j; k++) gj += V(j, k) * d[k];
                                for (k = j+1; k <= i-1; k++) gj += V(k, j) * d[k];
                                e[j] = gj;
                        }
                        for (j = 0; j < i; j++) V(j, i) = d[j];
                        f = 0;
                        for (j = 0; j < i; j++) {
                                e[j] /= h;
                                f += e[j] * d[j];
                        }
                        quad hh = f / (h + h);
                        for (j = 0; j < i; j++) e[j] -= hh * d[j];
#pragma omp parallel for schedule(dynamic, 8) private(k)
                        for (j = 0; j < i; j++) {
                                quad fj = d[j];
                                quad gj = e[j];
                                for (k = j; k <= i-1; k++) {
                                        V(k, j) -= (fj * e[k] + gj * d[k]);
                                }
                        }
                        for (j = 0; j < i; j++) {
                                d[j] = V(i-1, j);
                                V(i, j) = 0;
                        }
                }
                d[i] = h;
        }
        for (i = 0; i < n-1; i++) {
                V(n-1, i) = V(i, i);
                V(i, i) = 1;
                quad h = d[i+1];
                if (h != 0) {
                        for (k = 0; k <= i; k++) d[k] = V(k, i+1) / h;
#pragma omp parallel for schedule(static) private(k)
                        for (j = 0; j <= i; j++) {
                                quad g = 0;
                                for (k = 0; k <= i; k++) g += V(k, i+1) * V(k, j);
                                for (k = 0; k <= i; k++) V(k, j) -= g * d[k];
                        }
                }
                for (k = 0; k <= i; k++) V(k, i+1) = 0;
        }
        for (j = 0; j < n; j++) {
                d[j] = V(n-1, j);
                V(n-1, j) = 0;
        }
        V(n-1, n-1) = 1;
        e[0] = 0;
}

static int tql2(int n, quad *v, quad *d, quad *e)
{
        int i, j, k, l, m, iter;
        for (i = 1; i < n; i++) e[i-1] = e[i];
        e[n-1] = 0;
        quad f = 0, tst1 = 0;
        quad eps = FLT128_EPSILON;
        quad *cs = malloc(sizeof(quad) * n);
        quad *sn = malloc(sizeof(quad) * n);
        for (l = 0; l < n; l++) {
                quad t = fabsq(d[l]) + fabsq(e[l]);
                if (t > tst1) tst1 = t;
                m = l;
                while (m < n) {
                        if (fabsq(e[m]) <= eps * tst1) break;
                        m++;
                }
                if (m > l) {
                        iter = 0;
                        do {
                                if (++iter > 200) { free(cs); free(sn); return -1; }
                                quad g = d[l];
                                quad p = (d[l+1] - g) / (2 * e[l]);
                                quad r = hypotq(p, 1);
                                if (p < 0) r = -r;
                                d[l] = e[l] / (p + r);
                                d[l+1] = e[l] * (p + r);
                                quad dl1 = d[l+1];
                                quad h = g - d[l];
                                for (i = l+2; i < n; i++) d[i] -= h;
                                f += h;
                                p = d[m];
                                quad c = 1, c2 = 1, c3 = 1;
                                quad el1 = e[l+1];
                                quad s = 0, s2 = 0;
                                for (i = m-1; i >= l; i--) {
                                        c3 = c2;
                                        c2 = c;
                                        s2 = s;
                                        g = c * e[i];
                                        h = c * p;
                                        r = hypotq(p, e[i]);
                                        e[i+1] = s * r;
                                        s = e[i] / r;
                                        c = p / r;
                                        p = c * d[i] - s * g;
                                        d[i+1] = h + s * (c * g + s * d[i]);
                                        cs[i] = c;
                                        sn[i] = s;
                                }
                                /* the rotations of one sweep act on every row of V
                                 * independently: apply them row by row */
#pragma omp parallel for schedule(static) private(i, h)
                                for (k = 0; k < n; k++) {
                                        quad *vk = v + (long)k * n;
                                        for (i = m-1; i >= l; i--) {
                                                h = vk[i+1];
                                                vk[i+1] = sn[i] * vk[i] + cs[i] * h;
                                                vk[i] = cs[i] * vk[i] - sn[i] * h;
                                        }
                                }
                                p = -s * s2 * c3 * el1 * e[l] / dl1;
                                e[l] = s * p;
                                d[l] = c * p;
                        } while (fabsq(e[l]) > eps * tst1);
                }
                d[l] = d[l] + f;
                e[l] = 0;
        }
        free(cs);
        free(sn);
        /* sort ascending */
        for (i = 0; i < n-1; i++) {
                k = i;
                quad p = d[i];
                for (j = i+1; j < n; j++) {
                        if (d[j] < p) { k = j; p = d[j]; }
                }
                if (k != i) {
                        d[k] = d[i];
                        d[i] = p;
                        for (j = 0; j < n; j++) {
                                p = V(j, i); V(j, i) = V(j, k); V(j, k) = p;
                        }
                }
        }
        return 0;
}

/* a: symmetric, overwritten by eigenvectors (column j = vector j); w ascending */
int QRsyev(int n, quad *a, quad *w)
{
        quad *e = malloc(sizeof(quad) * n);
        tred2(n, a, w, e);
        int ierr = tql2(n, a, w, e);
        free(e);
        return ierr;
}

/* lower Cholesky factor in place (upper triangle zeroed) */
int QRpotrf(int n, quad *v)
{
        int i, j, k;
        for (j = 0; j < n; j++) {
                quad s = V(j, j);
                for (k = 0; k < j; k++) s -= V(j, k) * V(j, k);
                if (s <= 0) return j + 1;
                V(j, j) = sqrtq(s);
                for (i = j+1; i < n; i++) {
                        quad t = V(i, j);
                        for (k = 0; k < j; k++) t -= V(i, k) * V(j, k);
                        V(i, j) = t / V(j, j);
                }
                for (i = 0; i < j; i++) V(i, j) = 0;
        }
        return 0;
}

/* H c = e S c.  h is overwritten by the S-orthonormal eigenvectors
 * (column j), s by its Cholesky factor. */
int QRsygv(int n, quad *h, quad *s, quad *w)
{
        int i, j, k;
        int ierr = QRpotrf(n, s);
        if (ierr) return ierr;
        quad *v = s;  /* L, for the V() macro */
        /* h <- L^-1 h (columns) */
#pragma omp parallel for schedule(static) private(i, k)
        for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                        quad t = h[(long)i*n+j];
                        for (k = 0; k < i; k++) t -= V(i, k) * h[(long)k*n+j];
                        h[(long)i*n+j] = t / V(i, i);
                }
        }
        /* h <- h L^-T (rows) */
#pragma omp parallel for schedule(static) private(j, k)
        for (i = 0; i < n; i++) {
                quad *hi = h + (long)i*n;
                for (j = 0; j < n; j++) {
                        quad t = hi[j];
                        for (k = 0; k < j; k++) t -= hi[k] * V(j, k);
                        hi[j] = t / V(j, j);
                }
        }
        /* symmetrize against round-off */
        for (i = 0; i < n; i++) {
                for (j = 0; j < i; j++) {
                        quad t = (h[(long)i*n+j] + h[(long)j*n+i]) / 2;
                        h[(long)i*n+j] = t;
                        h[(long)j*n+i] = t;
                }
        }
        ierr = QRsyev(n, h, w);
        if (ierr) return -1;
        /* c = L^-T y */
#pragma omp parallel for schedule(static) private(i, k)
        for (j = 0; j < n; j++) {
                for (i = n-1; i >= 0; i--) {
                        quad t = h[(long)i*n+j];
                        for (k = i+1; k < n; k++) t -= V(k, i) * h[(long)k*n+j];
                        h[(long)i*n+j] = t / V(i, i);
                }
        }
        return 0;
}
