/*
 * The f_xc part of a two-component TDA / TDDFT matvec on a closed-shell (Kramers-restricted) reference in the MO
 * form, fused per grid sub-block (SOCDFTfxc2c_mo), #included by vxc_rks.c (eval_select):
 *
 *   psi_i,s(g)  = sum_mu phi_mu(g) Co[s][mu][i]           (occupied spinors, s = alpha, beta; value + nabla)
 *   psi~_xi,s   = sum_mu phi_mu(g) Z[s][mu][x i]          (Z = C_v z_x^T for every trial vector x)
 *   P_st        = sum_i psi~_xi,s psi_i,t^*               (and its gradient)
 *   n1 = P_aa + P_bb, m1 = (P_ab + P_ba, i (P_ab - P_ba), P_aa - P_bb)
 *   v_n = w f_tt n1, v_m = w f_ss m1                      (the isotropic closed-shell kernel)
 *   W_st = v_n 1 + v_m . sigma
 *   R[s][mu][x i] += sum_g phi_mu (sum_t W0_st psi_i,t + sum_c Wc_st d_c psi_i,t) + d_c phi_mu sum_t Wc_st psi_i,t
 *
 * and the caller forms v_xia = sum_s,mu C_v[s][mu][a]^* R[s][mu][x i].  No transition density matrix is built.
 * Complex arrays interleaved (numpy complex128): Co (2, nao, nocc), Z and R (2, nao, K), K = nvec nocc; f_tt,
 * f_ss (nvar, nvar, ngrid).  The sub-blocks accumulate into R under a critical section.  xctype 0 LDA, 1 GGA.
 */
#include <complex.h>

void SOCDFTfxc2c_mo(double *R, double *Co, double *Z, int nocc, int nvec, double *ftt, double *fss,
                    double *coords, double *weights, uint8_t *non0tab, int ngrid, int gs, double cutoff,
                    int xctype, gtoval_t eval, int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                    dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int nao = ao_loc[nbas], ncomp = xctype == 0 ? 1 : 4, nvar = ncomp;
        const int nsub = (ngrid + gs - 1) / gs;
        const int K = nvec * nocc, K2 = 2 * K, O2 = 2 * nocc;
        const double one = 1, zero = 0;
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *full = malloc(sizeof(double) * ncomp * nao * gs);
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *xyz = malloc(sizeof(double) * 3 * gs);
        double *col = malloc(sizeof(double) * 2 * nao * O2);            /* gathered Co rows */
        double *zl = malloc(sizeof(double) * 2 * nao * K2);             /* gathered Z rows */
        double *psi = malloc(sizeof(double) * ncomp * 2 * gs * O2);     /* [c][s] (n, 2 nocc) */
        double *pt = malloc(sizeof(double) * ncomp * 2 * gs * K2);      /* [c][s] (n, 2K): psi~, then A / B */
        double *rl = malloc(sizeof(double) * 2 * nao * K2);             /* [s] (nl, 2K) */
        double complex *P = malloc(sizeof(double complex) * 4 * ncomp * gs * nvec);
        double complex *W = malloc(sizeof(double complex) * 4 * ncomp * gs * nvec);
        int *ix = malloc(sizeof(int) * nao);
        size_t *src = malloc(sizeof(size_t) * 2 * nao);
        int sb;
#pragma omp for schedule(dynamic, 2)
        for (sb = 0; sb < nsub; sb++) {
                const int g0 = sb * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int i, j, g, k, c, s, t, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select(a, ix, src, full, n, xyz, (uint8_t *)msk, ncomp, cutoff, eval,
                                           ao_loc, atm, natm, bas, nbas, env);
                if (nl == 0) continue;
                for (s = 0; s < 2; s++)
                        for (i = 0; i < nl; i++) {
                                memcpy(col + ((size_t)s * nl + i) * O2, Co + ((size_t)s * nao + ix[i]) * O2,
                                       sizeof(double) * O2);
                                memcpy(zl + ((size_t)s * nl + i) * K2, Z + ((size_t)s * nao + ix[i]) * K2,
                                       sizeof(double) * K2);
                        }
                /* psi[c][s] (n, 2nocc) = phi_c^T Co_s ; psi~[c][s] (n, 2K) = phi_c^T Z_s (complex as real pairs) */
                for (c = 0; c < ncomp; c++) {
                        const double *ac = a + (size_t)c * nl * n;
                        for (s = 0; s < 2; s++) {
                                dgemm_("N", "T", &O2, &n, &nl, &one, col + (size_t)s * nl * O2, &O2, ac, &n, &zero,
                                       psi + ((size_t)c * 2 + s) * n * O2, &O2);
                                dgemm_("N", "T", &K2, &n, &nl, &one, zl + (size_t)s * nl * K2, &K2, ac, &n, &zero,
                                       pt + ((size_t)c * 2 + s) * n * K2, &K2);
                        }
                }
                const double complex *ps = (const double complex *)psi;
                double complex *pq = (double complex *)pt;
#define PSI(c, s, g, i) ps[(((size_t)(c) * 2 + (s)) * n + (g)) * nocc + (i)]
#define PST(c, s, g, xi) pq[(((size_t)(c) * 2 + (s)) * n + (g)) * K + (xi)]
#define PP(s, t, c, g, x) P[((((size_t)(s) * 2 + (t)) * ncomp + (c)) * n + (g)) * nvec + (x)]
#define WW(s, t, c, g, x) W[((((size_t)(s) * 2 + (t)) * ncomp + (c)) * n + (g)) * nvec + (x)]
                /* P_st rows: value sum_i psi~_s psi_t^*, gradients d psi~_s psi_t^* + psi~_s d psi_t^* */
                for (s = 0; s < 2; s++)
                        for (t = 0; t < 2; t++)
                                for (g = 0; g < n; g++)
                                        for (x = 0; x < nvec; x++) {
                                                double complex v0 = 0, vc[3] = {0, 0, 0};
                                                const double complex *q0 = &PST(0, s, g, x * nocc);
                                                const double complex *p0 = &PSI(0, t, g, 0);
                                                for (i = 0; i < nocc; i++) v0 += q0[i] * conj(p0[i]);
                                                for (c = 1; c < ncomp; c++) {
                                                        const double complex *qc = &PST(c, s, g, x * nocc);
                                                        const double complex *pc = &PSI(c, t, g, 0);
                                                        double complex acc = 0;
                                                        for (i = 0; i < nocc; i++)
                                                                acc += qc[i] * conj(p0[i]) + q0[i] * conj(pc[i]);
                                                        vc[c - 1] = acc;
                                                }
                                                PP(s, t, 0, g, x) = v0;
                                                for (c = 1; c < ncomp; c++) PP(s, t, c, g, x) = vc[c - 1];
                                        }
                /* the kernel: v_n = w f_tt n1, v_m = w f_ss m1; W_st = v_n 1 + v_m . sigma */
                const double *w = weights + g0;
                for (g = 0; g < n; g++)
                        for (x = 0; x < nvec; x++) {
                                double complex rn[4], rm[3][4], vn[4], vm[3][4];
                                for (c = 0; c < ncomp; c++) {
                                        const double complex paa = PP(0, 0, c, g, x), pab = PP(0, 1, c, g, x);
                                        const double complex pba = PP(1, 0, c, g, x), pbb = PP(1, 1, c, g, x);
                                        rn[c] = paa + pbb;
                                        rm[0][c] = pab + pba;
                                        rm[1][c] = I * (pab - pba);
                                        rm[2][c] = paa - pbb;
                                }
                                for (c = 0; c < nvar; c++) {
                                        double complex an = 0, am0 = 0, am1 = 0, am2 = 0;
                                        for (k = 0; k < nvar; k++) {
                                                const double ft = ftt[((size_t)c * nvar + k) * ngrid + g0 + g];
                                                const double fs = fss[((size_t)c * nvar + k) * ngrid + g0 + g];
                                                an += ft * rn[k];
                                                am0 += fs * rm[0][k];
                                                am1 += fs * rm[1][k];
                                                am2 += fs * rm[2][k];
                                        }
                                        vn[c] = w[g] * an;
                                        vm[0][c] = w[g] * am0;
                                        vm[1][c] = w[g] * am1;
                                        vm[2][c] = w[g] * am2;
                                }
                                for (c = 0; c < ncomp; c++) {
                                        WW(0, 0, c, g, x) = vn[c] + vm[2][c];
                                        WW(0, 1, c, g, x) = vm[0][c] - I * vm[1][c];
                                        WW(1, 0, c, g, x) = vm[0][c] + I * vm[1][c];
                                        WW(1, 1, c, g, x) = vn[c] - vm[2][c];
                                }
                        }
                /* A_s (c = 0 slot of pt) = sum_t W0_st psi_t + sum_c Wc_st d_c psi_t;  B^c_s = sum_t Wc_st psi_t */
                for (s = 0; s < 2; s++)
                        for (g = 0; g < n; g++)
                                for (x = 0; x < nvec; x++) {
                                        double complex *A = &PST(0, s, g, x * nocc);
                                        for (i = 0; i < nocc; i++) {
                                                double complex acc = 0;
                                                for (t = 0; t < 2; t++) {
                                                        acc += WW(s, t, 0, g, x) * PSI(0, t, g, i);
                                                        for (c = 1; c < ncomp; c++)
                                                                acc += WW(s, t, c, g, x) * PSI(c, t, g, i);
                                                }
                                                A[i] = acc;
                                        }
                                        for (c = 1; c < ncomp; c++) {
                                                double complex *B = &PST(c, s, g, x * nocc);
                                                for (i = 0; i < nocc; i++)
                                                        B[i] = WW(s, 0, c, g, x) * PSI(0, 0, g, i)
                                                             + WW(s, 1, c, g, x) * PSI(0, 1, g, i);
                                        }
                                }
#undef PSI
#undef PST
#undef PP
#undef WW
                /* R_s (nl, 2K) = sum_c phi_c Y_c,s  (colmajor: Y (2K x n) . phi_c (n x nl)) */
                for (s = 0; s < 2; s++) {
                        double *r = rl + (size_t)s * nl * K2;
                        for (c = 0; c < ncomp; c++) {
                                const double beta = c == 0 ? 0. : 1.;
                                dgemm_("N", "N", &K2, &nl, &n, &one, pt + ((size_t)c * 2 + s) * n * K2, &K2,
                                       a + (size_t)c * nl * n, &n, &beta, r, &K2);
                        }
                }
#pragma omp critical(socdft_fxc2c_mo)
                {
                        for (s = 0; s < 2; s++)
                                for (i = 0; i < nl; i++) {
                                        double *dst = R + ((size_t)s * nao + ix[i]) * K2;
                                        const double *sr = rl + ((size_t)s * nl + i) * K2;
                                        for (j = 0; j < K2; j++) dst[j] += sr[j];
                                }
                }
        }
        free(full); free(a); free(xyz); free(col); free(zl); free(psi); free(pt); free(rl); free(P); free(W);
        free(ix); free(src);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}
