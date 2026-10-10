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
 * Quadruple precision (IEEE binary128, __float128) dense linear algebra
 * kernels used by the quad precision Dirac-Hartree-Fock program
 * (pyscf.qdhf).  Arrays are row-major and contiguous.  Complex arrays are
 * stored as two separate real arrays (real part, imaginary part).
 */

#include <stdlib.h>
#include <string.h>
#include <quadmath.h>

typedef __float128 quad;

/* ------------------------------------------------------------------ */
/* conversions                                                        */
/* ------------------------------------------------------------------ */
void QDd2q(quad *out, const double *in, long n)
{
        long i;
        for (i = 0; i < n; i++) {
                out[i] = (quad)in[i];
        }
}

void QDq2d(double *out, const quad *in, long n)
{
        long i;
        for (i = 0; i < n; i++) {
                out[i] = (double)in[i];
        }
}

/* parse n whitespace separated decimal numbers */
int QDparse(quad *out, const char *buf, long n)
{
        long i;
        const char *p = buf;
        char *end;
        for (i = 0; i < n; i++) {
                out[i] = strtoflt128(p, &end);
                if (end == p) {
                        return -1;
                }
                p = end;
        }
        return 0;
}

/* format n numbers with 40 significant digits, each in a 64-byte slot */
void QDformat(char *buf, const quad *x, long n)
{
        long i;
        for (i = 0; i < n; i++) {
                quadmath_snprintf(buf+i*64, 64, "%.39Qe", x[i]);
        }
}

/* ------------------------------------------------------------------ */
/* elementwise operations                                             */
/* ------------------------------------------------------------------ */
/* out = a * x + b * y.  y can be NULL */
void QDaxpby(quad *out, const quad *x, const quad *y,
             const quad *a, const quad *b, long n)
{
        long i;
        quad a0 = *a;
        if (y == NULL) {
#pragma omp parallel for schedule(static)
                for (i = 0; i < n; i++) {
                        out[i] = a0 * x[i];
                }
        } else {
                quad b0 = *b;
#pragma omp parallel for schedule(static)
                for (i = 0; i < n; i++) {
                        out[i] = a0 * x[i] + b0 * y[i];
                }
        }
}

void QDmul(quad *out, const quad *x, const quad *y, long n)
{
        long i;
#pragma omp parallel for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = x[i] * y[i];
        }
}

void QDdot(quad *out, const quad *x, const quad *y, long n)
{
        long i;
        quad s = 0;
        for (i = 0; i < n; i++) {
                s += x[i] * y[i];
        }
        *out = s;
}

void QDsum(quad *out, const quad *x, long n)
{
        long i;
        quad s = 0;
        for (i = 0; i < n; i++) {
                s += x[i];
        }
        *out = s;
}

void QDmaxabs(quad *out, const quad *x, long n)
{
        long i;
        quad s = 0;
        for (i = 0; i < n; i++) {
                if (fabsq(x[i]) > s) {
                        s = fabsq(x[i]);
                }
        }
        *out = s;
}

/* op: 0 sqrt, 1 reciprocal, 2 1/sqrt, 3 negative, 4 abs */
void QDunary(quad *out, const quad *x, long n, int op)
{
        long i;
        switch (op) {
        case 0: for (i = 0; i < n; i++) out[i] = sqrtq(x[i]); break;
        case 1: for (i = 0; i < n; i++) out[i] = 1 / x[i]; break;
        case 2: for (i = 0; i < n; i++) out[i] = 1 / sqrtq(x[i]); break;
        case 3: for (i = 0; i < n; i++) out[i] = -x[i]; break;
        case 4: for (i = 0; i < n; i++) out[i] = fabsq(x[i]); break;
        }
}

/* ------------------------------------------------------------------ */
/* matrix multiplication                                              */
/* ------------------------------------------------------------------ */
/*
 * C[m,n] = alpha * op(A)[m,k] * op(B)[k,n] + beta * C[m,n]   (row-major)
 * transa != 0: A is stored as [k,m];  transb != 0: B is stored as [n,k]
 * beta == NULL means beta = 0 (C is not read).
 */
void QDgemm(int transa, int transb, int m, int n, int k,
            const quad *alpha, const quad *a, const quad *b,
            const quad *beta, quad *c)
{
        long i, j, p;
        quad *abuf = NULL;
        quad *bbuf = NULL;
        const quad *pa, *pb;
        if (transa) {
                abuf = malloc(sizeof(quad) * m * k);
                for (p = 0; p < k; p++) {
                for (i = 0; i < m; i++) {
                        abuf[i*k+p] = a[p*m+i];
                } }
                pa = abuf;
        } else {
                pa = a;
        }
        if (!transb) {
                bbuf = malloc(sizeof(quad) * n * k);
                for (p = 0; p < k; p++) {
                for (j = 0; j < n; j++) {
                        bbuf[j*k+p] = b[p*n+j];
                } }
                pb = bbuf;
        } else {
                pb = b;
        }
        quad alp = *alpha;

#pragma omp parallel for schedule(dynamic) private(j, p)
        for (i = 0; i < m; i++) {
                const quad *ai = pa + i * k;
                for (j = 0; j < n; j++) {
                        const quad *bj = pb + j * k;
                        quad s0 = 0, s1 = 0;
                        for (p = 0; p+1 < k; p+=2) {
                                s0 += ai[p  ] * bj[p  ];
                                s1 += ai[p+1] * bj[p+1];
                        }
                        if (p < k) {
                                s0 += ai[p] * bj[p];
                        }
                        if (beta == NULL) {
                                c[i*n+j] = alp * (s0 + s1);
                        } else {
                                c[i*n+j] = alp * (s0 + s1) + (*beta) * c[i*n+j];
                        }
                }
        }
        if (abuf != NULL) free(abuf);
        if (bbuf != NULL) free(bbuf);
}

/* ------------------------------------------------------------------ */
/* complex Hermitian eigenvalue problem, cyclic Jacobi method          */
/* ------------------------------------------------------------------ */
/*
 * A = are + i aim is a Hermitian n x n matrix (row-major, destroyed on
 * output).  Eigenvalues are returned in w (ascending order) and the
 * eigenvectors in the columns of V = vre + i vim.
 * Returns the number of sweeps, or -1 if not converged.
 */
int QDzheev(int n, quad *are, quad *aim, quad *w, quad *vre, quad *vim)
{
        const quad eps = FLT128_EPSILON;
        long i, j, k, p, q;
        int sweep;
        int converged = 0;

        memset(vre, 0, sizeof(quad) * n * n);
        memset(vim, 0, sizeof(quad) * n * n);
        for (i = 0; i < n; i++) {
                vre[i*n+i] = 1;
                aim[i*n+i] = 0;
        }

        quad fro = 0;
        for (i = 0; i < (long)n*n; i++) {
                fro += are[i] * are[i] + aim[i] * aim[i];
        }
        fro = sqrtq(fro);
        if (fro == 0) {
                for (i = 0; i < n; i++) w[i] = 0;
                return 0;
        }

        for (sweep = 0; sweep < 100; sweep++) {
                quad off = 0;
                for (p = 0; p < n; p++) {
                for (q = p+1; q < n; q++) {
                        off += are[p*n+q] * are[p*n+q] + aim[p*n+q] * aim[p*n+q];
                } }
                off = sqrtq(off);
                if (off <= 1e-2Q * eps * fro) {
                        converged = 1;
                        break;
                }

                for (p = 0; p < n-1; p++) {
                for (q = p+1; q < n; q++) {
                        quad xr = are[p*n+q];
                        quad xi = aim[p*n+q];
                        quad r = hypotq(xr, xi);
                        if (r <= 1e-4Q * eps * fro) {
                                are[p*n+q] = 0; aim[p*n+q] = 0;
                                are[q*n+p] = 0; aim[q*n+p] = 0;
                                continue;
                        }
                        quad app = are[p*n+p];
                        quad aqq = are[q*n+q];
                        /* off-diagonal element is negligible relative to
                         * the diagonal elements */
                        if (sweep > 3 &&
                            fabsq(app) + 1e5Q * r == fabsq(app) &&
                            fabsq(aqq) + 1e5Q * r == fabsq(aqq)) {
                                are[p*n+q] = 0; aim[p*n+q] = 0;
                                are[q*n+p] = 0; aim[q*n+p] = 0;
                                continue;
                        }
                        /* e^{-i phi} = conj(a_pq) / |a_pq| */
                        quad er = xr / r;
                        quad ei = -xi / r;
                        quad theta = (aqq - app) / (2 * r);
                        quad t;
                        if (fabsq(theta) > 1e60Q) {
                                t = 1 / (2 * theta);
                        } else {
                                t = 1 / (fabsq(theta) + sqrtq(theta*theta + 1));
                                if (theta < 0) t = -t;
                        }
                        quad c = 1 / sqrtq(t*t + 1);
                        quad s = t * c;
                        /* U_pp = c, U_pq = s, U_qp = -s e^{-i phi}, U_qq = c e^{-i phi}
                         * A <- A U */
                        for (k = 0; k < n; k++) {
                                quad kpr = are[k*n+p], kpi = aim[k*n+p];
                                quad kqr = are[k*n+q], kqi = aim[k*n+q];
                                /* e^{-i phi} * A_kq */
                                quad zr = er * kqr - ei * kqi;
                                quad zi = er * kqi + ei * kqr;
                                are[k*n+p] = c * kpr - s * zr;
                                aim[k*n+p] = c * kpi - s * zi;
                                are[k*n+q] = s * kpr + c * zr;
                                aim[k*n+q] = s * kpi + c * zi;
                        }
                        /* A <- U^dagger A */
                        for (k = 0; k < n; k++) {
                                quad pkr = are[p*n+k], pki = aim[p*n+k];
                                quad qkr = are[q*n+k], qki = aim[q*n+k];
                                /* e^{+i phi} * A_qk */
                                quad zr = er * qkr + ei * qki;
                                quad zi = er * qki - ei * qkr;
                                are[p*n+k] = c * pkr - s * zr;
                                aim[p*n+k] = c * pki - s * zi;
                                are[q*n+k] = s * pkr + c * zr;
                                aim[q*n+k] = s * pki + c * zi;
                        }
                        are[p*n+q] = 0; aim[p*n+q] = 0;
                        are[q*n+p] = 0; aim[q*n+p] = 0;
                        are[p*n+p] = app - t * r;
                        are[q*n+q] = aqq + t * r;
                        aim[p*n+p] = 0;
                        aim[q*n+q] = 0;
                        /* V <- V U */
                        for (k = 0; k < n; k++) {
                                quad kpr = vre[k*n+p], kpi = vim[k*n+p];
                                quad kqr = vre[k*n+q], kqi = vim[k*n+q];
                                quad zr = er * kqr - ei * kqi;
                                quad zi = er * kqi + ei * kqr;
                                vre[k*n+p] = c * kpr - s * zr;
                                vim[k*n+p] = c * kpi - s * zi;
                                vre[k*n+q] = s * kpr + c * zr;
                                vim[k*n+q] = s * kpi + c * zi;
                        }
                } }
        }

        /* sort eigenvalues in ascending order */
        int *idx = malloc(sizeof(int) * n);
        for (i = 0; i < n; i++) {
                idx[i] = i;
                w[i] = are[i*n+i];
        }
        for (i = 0; i < n; i++) {
                k = i;
                for (j = i+1; j < n; j++) {
                        if (w[idx[j]] < w[idx[k]]) k = j;
                }
                int tmp = idx[i]; idx[i] = idx[k]; idx[k] = tmp;
        }
        quad *buf = malloc(sizeof(quad) * n * n);
        for (i = 0; i < n; i++) {
                are[i] = w[idx[i]];
        }
        for (i = 0; i < n; i++) {
                w[i] = are[i];
        }
        memcpy(buf, vre, sizeof(quad) * n * n);
        for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
                vre[k*n+i] = buf[k*n+idx[i]];
        } }
        memcpy(buf, vim, sizeof(quad) * n * n);
        for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
                vim[k*n+i] = buf[k*n+idx[i]];
        } }
        free(buf);
        free(idx);
        if (converged) {
                return sweep;
        } else {
                return -1;
        }
}

/* ------------------------------------------------------------------ */
/* real linear equations A X = B, Gaussian elimination with partial    */
/* pivoting.  A[n,n] is destroyed, B[n,nrhs] is overwritten by X.      */
/* Returns 0 on success, -1 for a singular matrix.                     */
/* ------------------------------------------------------------------ */
int QDgesv(int n, int nrhs, quad *a, quad *b)
{
        long i, j, k, piv;
        for (k = 0; k < n; k++) {
                piv = k;
                for (i = k+1; i < n; i++) {
                        if (fabsq(a[i*n+k]) > fabsq(a[piv*n+k])) piv = i;
                }
                if (a[piv*n+k] == 0) {
                        return -1;
                }
                if (piv != k) {
                        for (j = 0; j < n; j++) {
                                quad t = a[k*n+j]; a[k*n+j] = a[piv*n+j]; a[piv*n+j] = t;
                        }
                        for (j = 0; j < nrhs; j++) {
                                quad t = b[k*nrhs+j]; b[k*nrhs+j] = b[piv*nrhs+j]; b[piv*nrhs+j] = t;
                        }
                }
                quad inv = 1 / a[k*n+k];
                for (i = k+1; i < n; i++) {
                        quad f = a[i*n+k] * inv;
                        if (f == 0) continue;
                        for (j = k; j < n; j++) a[i*n+j] -= f * a[k*n+j];
                        for (j = 0; j < nrhs; j++) b[i*nrhs+j] -= f * b[k*nrhs+j];
                }
        }
        for (k = n-1; k >= 0; k--) {
                quad inv = 1 / a[k*n+k];
                for (j = 0; j < nrhs; j++) {
                        quad s = b[k*nrhs+j];
                        for (i = k+1; i < n; i++) s -= a[k*n+i] * b[i*nrhs+j];
                        b[k*nrhs+j] = s * inv;
                }
        }
        return 0;
}
