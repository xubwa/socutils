/*
 * Elementwise grid work of the fixed-grid RKS XC Hessian
 * (socutils.hessian.rhf_fast.xc_hess_explicit), one pass over AOs x grid
 * points, OpenMP over grid sub-blocks:
 *
 *   R[x,k,A,g]  = sum_{mu in A} d/dx (phi_mu ...)_k (D phi)_mu      (unscaled
 *                 d_A rho_k, times -2 in the caller),
 *   diag[x,y,A] += sum_g sum_{mu in A} [d_xy phi_mu e0_mu + sum_j d_xyj phi_mu f_j,mu],
 *   ab[x]       = a_x = 2 w v0 d_x phi + b_x,  ab[3+x] = b_x = 2 sum_j w v_j d_xj phi,
 *
 * with e0 = 2 w v0 (D phi) + 2 sum_j w v_j (D d_j phi),
 *      f_j = 2 w v_j (D phi) + w v_tau (D d_j phi).
 * ao is pyscf's AO derivative stack in memory order (ncomp, nao, ngrid), d the
 * D-contracted AOs (nd, nao, ngrid), wv the weighted vxc (nvar, ngrid).
 * xctype 0 LDA (ncomp >= 10), 1 GGA, 2 MGGA (ncomp >= 20).  y >= x in diag.
 */
#include <stdlib.h>
#include <string.h>

#define BLK 128

static const int D2[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};
static const int D3[3][3][3] = {
        {{10, 11, 12}, {11, 13, 14}, {12, 14, 15}},
        {{11, 13, 14}, {13, 16, 17}, {14, 17, 18}},
        {{12, 14, 15}, {14, 17, 18}, {15, 18, 19}}};

void SOCDFTrks_xc_hess_elem(double *R, double *diag, double *ab,
                            double *ao, double *d, double *wv,
                            int xctype, int nvar, int ngrid, int nao,
                            int *atom_of, int natm)
{
        const size_t G = ngrid;
        const int gga = xctype >= 1;
        const int mgga = xctype == 2;
        const int nblk = (ngrid + BLK - 1) / BLK;
        memset(R, 0, sizeof(double) * 3 * nvar * natm * G);
#pragma omp parallel
{
        double *dg = calloc(9 * natm, sizeof(double));
        double e0[BLK], f[3][BLK];
        int ib, mu, p, x, y, j;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                const int p0 = ib * BLK;
                const int np = (p0 + BLK < ngrid ? p0 + BLK : ngrid) - p0;
                for (mu = 0; mu < nao; mu++) {
                        const int A = atom_of[mu];
#define AO(k) (ao + ((size_t)(k) * nao + mu) * G + p0)
#define DD(k) (d + ((size_t)(k) * nao + mu) * G + p0)
#define WV(k) (wv + (size_t)(k) * G + p0)
#define RR(x, k) (R + (((size_t)(x) * nvar + (k)) * natm + A) * G + p0)
                        const double *d0 = DD(0);
                        const double *w0 = WV(0);
                        for (p = 0; p < np; p++) e0[p] = 2 * w0[p] * d0[p];
                        if (gga) {
                                for (j = 0; j < 3; j++) {
                                        const double *dj = DD(1 + j);
                                        const double *wj = WV(1 + j);
                                        for (p = 0; p < np; p++) {
                                                e0[p] += 2 * wj[p] * dj[p];
                                                f[j][p] = 2 * wj[p] * d0[p];
                                        }
                                        if (mgga) {
                                                const double *wt = WV(4);
                                                for (p = 0; p < np; p++) f[j][p] += wt[p] * dj[p];
                                        }
                                }
                        }
                        for (x = 0; x < 3; x++) {
                                const double *ax = AO(1 + x);
                                double *r = RR(x, 0);
                                for (p = 0; p < np; p++) r[p] += ax[p] * d0[p];
                                if (gga) {
                                        for (j = 0; j < 3; j++) {
                                                const double *a2 = AO(D2[x][j]);
                                                const double *dj = DD(1 + j);
                                                r = RR(x, 1 + j);
                                                for (p = 0; p < np; p++) r[p] += a2[p] * d0[p] + ax[p] * dj[p];
                                        }
                                }
                                if (mgga) {
                                        r = RR(x, 4);
                                        for (j = 0; j < 3; j++) {
                                                const double *a2 = AO(D2[x][j]);
                                                const double *dj = DD(1 + j);
                                                for (p = 0; p < np; p++) r[p] += .5 * a2[p] * dj[p];
                                        }
                                }
                                double *a_out = ab + ((size_t)x * nao + mu) * G + p0;
                                for (p = 0; p < np; p++) a_out[p] = 2 * w0[p] * ax[p];
                                if (gga) {
                                        double *b_out = ab + ((size_t)(3 + x) * nao + mu) * G + p0;
                                        for (p = 0; p < np; p++) b_out[p] = 0;
                                        for (j = 0; j < 3; j++) {
                                                const double *a2 = AO(D2[x][j]);
                                                const double *wj = WV(1 + j);
                                                for (p = 0; p < np; p++) b_out[p] += 2 * wj[p] * a2[p];
                                        }
                                        for (p = 0; p < np; p++) a_out[p] += b_out[p];
                                }
                                for (y = x; y < 3; y++) {
                                        double s = 0;
                                        const double *a2 = AO(D2[x][y]);
                                        for (p = 0; p < np; p++) s += a2[p] * e0[p];
                                        if (gga) {
                                                for (j = 0; j < 3; j++) {
                                                        const double *a3 = AO(D3[x][y][j]);
                                                        for (p = 0; p < np; p++) s += a3[p] * f[j][p];
                                                }
                                        }
                                        dg[(x * 3 + y) * natm + A] += s;
                                }
#undef AO
#undef DD
#undef WV
#undef RR
                        }
                }
        }
#pragma omp critical
        {
                int i;
                for (i = 0; i < 9 * natm; i++) diag[i] += dg[i];
        }
        free(dg);
}
}

/*
 * Only the per-atom grid vectors R[x,k,A,g] = sum_{mu in A} d_x (...)_k (D ...)_mu
 * of SOCDFTrks_xc_hess_elem (unscaled; d_A rho_k = -2 R), for the first-order
 * XC matrices of the CPHF right-hand sides.  Needs the AO derivatives up to 2
 * (GGA) / 1 (LDA) only.  OpenMP over grid sub-blocks.
 */
void SOCDFTrks_dR_rho(double *R, double *ao, double *d, int xctype, int nvar,
                      int ngrid, int nao, int *atom_of, int natm)
{
        const size_t G = ngrid;
        const int gga = xctype >= 1;
        const int nblk = (ngrid + BLK - 1) / BLK;
        memset(R, 0, sizeof(double) * 3 * nvar * natm * G);
#pragma omp parallel
{
        int ib, mu, p, x, j;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                const int p0 = ib * BLK;
                const int np = (p0 + BLK < ngrid ? p0 + BLK : ngrid) - p0;
                for (mu = 0; mu < nao; mu++) {
                        const int A = atom_of[mu];
                        const double *d0 = d + (size_t)mu * G + p0;
                        for (x = 0; x < 3; x++) {
                                const double *ax = ao + ((size_t)(1 + x) * nao + mu) * G + p0;
                                double *r = R + (((size_t)x * nvar) * natm + A) * G + p0;
                                for (p = 0; p < np; p++) r[p] += ax[p] * d0[p];
                                if (gga) {
                                        for (j = 0; j < 3; j++) {
                                                const double *a2 = ao + ((size_t)D2[x][j] * nao + mu) * G + p0;
                                                const double *dj = d + ((size_t)(1 + j) * nao + mu) * G + p0;
                                                r = R + (((size_t)x * nvar + 1 + j) * natm + A) * G + p0;
                                                for (p = 0; p < np; p++) r[p] += a2[p] * d0[p] + ax[p] * dj[p];
                                        }
                                }
                        }
                }
        }
}
}
