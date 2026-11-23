/*
 * Point terms of the grid response of the RKS XC Hessian (socutils.hessian.rks_grid_response), on the AOs of
 * one sub-block (nl local AOs, sorted by atom: seg[A] .. seg[A + 1] the AOs of local atom A):
 *
 *     R[x, k, g, A] = sum_{mu in A} sum_t rcoef_t ao[ra_t][g, mu] phid[rd_t][g, mu]      (t with x_t = x, k_t = k)
 *     Q[A, x, y]   += sum_g sum_{mu in A} sum_t V[v_t][g] ao[qa_t][g, mu] phid[qd_t][g, mu]
 *
 * ao (nao_rows, ng, nl), phid (nd_rows, ng, nl) C order (phid = ao_r . D, the D phi rows); V (nv, ng) the
 * per-point weights (w v^ab).  rtab (nr, 4): x, k, ra, rd; qtab (nq, 5): x, y, qa, qd, v.  R (3, nvar, ng, na)
 * is overwritten, Q (na, 3, 3) accumulated.  One atom per task: no write conflicts.
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void SOCDFTgr_point_terms(double *R, double *Q, const double *ao, const double *phid, const double *V,
                          int ng, int nl, const int *seg, int na, const int *rtab, const double *rcoef, int nr,
                          int nvar, const int *qtab, int nq)
{
        const size_t gl = (size_t)ng * nl;
        memset(R, 0, sizeof(double) * 3 * nvar * (size_t)ng * na);
        int A;
#pragma omp parallel for schedule(dynamic, 1)
        for (A = 0; A < na; A++) {
                const int m0 = seg[A], m1 = seg[A + 1];
                int t, g, m;
                for (t = 0; t < nr; t++) {
                        const int x = rtab[t * 4], k = rtab[t * 4 + 1];
                        const double *a = ao + (size_t)rtab[t * 4 + 2] * gl;
                        const double *d = phid + (size_t)rtab[t * 4 + 3] * gl;
                        const double c = rcoef[t];
                        double *out = R + ((size_t)(x * nvar + k) * ng) * na + A;
                        for (g = 0; g < ng; g++) {
                                const double *ag = a + (size_t)g * nl, *dg = d + (size_t)g * nl;
                                double s = 0;
                                for (m = m0; m < m1; m++) s += ag[m] * dg[m];
                                out[(size_t)g * na] += c * s;
                        }
                }
                double q[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                for (t = 0; t < nq; t++) {
                        const int x = qtab[t * 5], y = qtab[t * 5 + 1];
                        const double *a = ao + (size_t)qtab[t * 5 + 2] * gl;
                        const double *d = phid + (size_t)qtab[t * 5 + 3] * gl;
                        const double *v = V + (size_t)qtab[t * 5 + 4] * ng;
                        double s = 0;
                        for (g = 0; g < ng; g++) {
                                const double *ag = a + (size_t)g * nl, *dg = d + (size_t)g * nl;
                                double p = 0;
                                for (m = m0; m < m1; m++) p += ag[m] * dg[m];
                                s += v[g] * p;
                        }
                        q[x * 3 + y] += s;
                }
                for (t = 0; t < 9; t++) Q[(size_t)A * 9 + t] += q[t];
        }
}

/*
 * The Q terms of SOCDFTgr_point_terms in the AO-major layout of rhf_fast's local passes, with the points' owners:
 *
 *     Q[A, w, x, y] += sum_{g: own[g] = w} sum_{mu in A} sum_t V[v_t][g] ao[qa_t][mu, g] phid[qd_t][mu, g]
 *
 * ao (nrow, nl, ng), phid (nd, nl, ng) (rows strided by nl ng); seg (na + 1) the local AO ranges of the local
 * atoms; own (ng) the local owner index of each point, nown owners; Q (na, nown, 3, 3) accumulated.
 */
void SOCDFTgr_q_owner(double *Q, const double *ao, const double *phid, const double *V, int ng, int nl,
                      const int *seg, int na, const int *own, int nown, const int *qtab, int nq)
{
        const size_t gl = (size_t)ng * nl;
        int A;
#pragma omp parallel
{
        double *acc = malloc(sizeof(double) * ng);
#pragma omp for schedule(dynamic, 1)
        for (A = 0; A < na; A++) {
                const int m0 = seg[A], m1 = seg[A + 1];
                int t, g, m;
                for (t = 0; t < nq; t++) {
                        const int x = qtab[t * 5], y = qtab[t * 5 + 1];
                        const double *a = ao + (size_t)qtab[t * 5 + 2] * gl;
                        const double *d = phid + (size_t)qtab[t * 5 + 3] * gl;
                        const double *v = V + (size_t)qtab[t * 5 + 4] * ng;
                        for (g = 0; g < ng; g++) acc[g] = 0;
                        for (m = m0; m < m1; m++) {
                                const double *am = a + (size_t)m * ng, *dm = d + (size_t)m * ng;
                                for (g = 0; g < ng; g++) acc[g] += am[g] * dm[g];
                        }
                        for (g = 0; g < ng; g++)
                                Q[(((size_t)A * nown + own[g]) * 3 + x) * 3 + y] += v[g] * acc[g];
                }
        }
        free(acc);
}
}
