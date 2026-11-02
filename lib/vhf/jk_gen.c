/*
 * Shell quartets of l <= JKV_LMAX for the J / K builds of jk_multi.c by the
 * Head-Gordon-Pople scheme, one generated kernel per angular class
 * (jk_vrr_gen.c, from gen_jk_vrr.py: the vertical recursions per primitive
 * quartet, the contractions, the horizontal recursions, the d shells'
 * Cartesian-to-spherical transform and the output at the role strides, all
 * unrolled); here the pair tables, the F_m tables and the dispatch.
 *
 * A = the bra shell of the higher l (the pair's shells swapped when l_i <
 * l_j), likewise C, so that the recursions build on the higher l.  The
 * primitive pairs of every pair of shells with l <= JKV_LMAX are tabulated
 * once per build (p, P, K_ab / p, P - R_i, P - R_j, the coefficient products
 * of the contraction pairs; K_ab < GEN_KCUT dropped).  Normalisation as
 * libcint: radial factors in env's coefficients, the common factors of s and
 * p.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

/* primitive pairs with exp(-ab/p R^2) below are dropped (0: none, for tests) */
static double GEN_KCUT = 1e-24;
void SOCVHFgen_set_kcut(double c) { GEN_KCUT = c; }
#define GT_STEP 0.05
#define GT_TMAX 40.
#define GT_N ((int)(GT_TMAX / GT_STEP) + 2)
#define GT_ORD 7                        /* Taylor terms */
#define GT_TN (4 * 2 + GT_ORD + 1)      /* F_0 .. F_(4 lmax + GT_ORD) per grid point */
#define TWO_PI52 34.986836655249725
#define G_MAXCTR 4

typedef struct {
        int n, nc, nci, ncj, ld, li, lj;
        double *p, *px, *py, *pz, *kp;  /* p, P, K_ab / p */
        double *rix, *riy, *riz;        /* P - R_i */
        double *rjx, *rjy, *rjz;        /* P - R_j */
        double *bnd;                    /* bound of the primitive pair (with the 1 / sqrt(p) of the other side) */
        double *beta;                   /* the same without 1 / sqrt(p); the primitives sorted by it, descending */
        double *c;                      /* c[cb * ld + t], cb = c_j nc_i + c_i */
        double bmax, rij[3];            /* max bound without 1 / sqrt(p); R_i - R_j */
} GPair;

static inline double taylor(const double *f, double dt)
{
        double s = f[6] * (dt / 6);
        s = (f[5] + s) * (dt / 5);
        s = (f[4] + s) * (dt / 4);
        s = (f[3] + s) * (dt / 3);
        s = (f[2] + s) * (dt / 2);
        s = (f[1] + s) * dt;
        return f[0] + s;
}

/* F[m] = F_m(t), m <= M */
static inline void boys_n(const double *tab, double t, int M, double *F)
{
        int m;
        if (t < GT_TMAX) {
                const int k = (int)(t / GT_STEP + .5);
                const double dt = k * GT_STEP - t, *f = tab + k * GT_TN;
                for (m = 0; m <= M; m++) F[m] = taylor(f + m, dt);
        } else {
                const double h = .5 / t;
                F[0] = .886226925452758014 / sqrt(t);
                for (m = 0; m < M; m++) F[m + 1] = F[m] * (2 * m + 1) * h;
        }
}

/* the n primitive pairs of a pair table in the order of beta, descending:
 * every array of arrs (length n) and the nc columns of c (stride ld) */
static void sort_prims(double **arrs, int narr, double *c, int nc, int ld, int n, const double *beta)
{
        int *idx = malloc(sizeof(int) * (n > 0 ? n : 1));
        double *tmp = malloc(sizeof(double) * (n > 0 ? n : 1));
        int i, j, a;
        for (i = 0; i < n; i++) idx[i] = i;
        for (i = 1; i < n; i++) {               /* insertion sort, n small */
                const int v = idx[i];
                for (j = i - 1; j >= 0 && beta[idx[j]] < beta[v]; j--) idx[j + 1] = idx[j];
                idx[j + 1] = v;
        }
        for (a = 0; a < narr; a++) {
                double *x = arrs[a];
                for (i = 0; i < n; i++) tmp[i] = x[idx[i]];
                memcpy(x, tmp, sizeof(double) * n);
        }
        for (a = 0; a < nc; a++) {
                double *x = c + (size_t)a * ld;
                for (i = 0; i < n; i++) tmp[i] = x[idx[i]];
                memcpy(x, tmp, sizeof(double) * n);
        }
        free(idx);
        free(tmp);
}

/* the leading bra primitives b with beta_b >= thr (beta descending) */
static inline int n_above(const double *beta, int n, double thr)
{
        while (n > 0 && beta[n - 1] < thr) n--;
        return n;
}

#include "jk_vrr_gen.c"

typedef struct GenCtx {
        int nbas;
        GPair *pair;                    /* per shell pair i >= j, l <= JKV_LMAX (n = 0 otherwise) */
        double *tab;
} GenCtx;

#define N3(d) (((d) + 1) * ((d) + 2) * ((d) + 3) / 6)
#define N2(d) (((d) + 1) * ((d) + 2) / 2)

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

static void f_series(double *f, double t, int m)
{
        const double et = exp(-t);
        double term = 1. / (2 * m + 1), sum = term;
        int k, n;
        for (k = 1; k < 400; k++) {
                term *= 2 * t / (2 * m + 2 * k + 1);
                sum += term;
                if (term < 1e-18 * sum) break;
        }
        f[m] = et * sum;
        for (n = m; n > 0; n--) f[n - 1] = (2 * t * f[n] + et) / (2 * n - 1);
}

static double common_fac(int l)
{
        return l == 0 ? 0.282094791773878143 : (l == 1 ? 0.488602511902919921 : 1.);
}

GenCtx *SOCVHFgen_ctx_build(int *atm, int natm, int *bas, int nbas, double *env)
{
        GenCtx *cx = malloc(sizeof(GenCtx));
        const int npair = nbas * (nbas + 1) / 2;
        int k, pq;
        cx->nbas = nbas;
        cx->tab = malloc(sizeof(double) * GT_N * GT_TN);
        for (k = 0; k < GT_N; k++) f_series(cx->tab + k * GT_TN, k * GT_STEP, GT_TN - 1);
        cx->pair = calloc(npair, sizeof(GPair));
#pragma omp parallel for schedule(dynamic, 16)
        for (pq = 0; pq < npair; pq++) {
                int i, j;
                shell_pair(pq, &i, &j);
                const int *bi = bas + i * BAS_SLOTS, *bj = bas + j * BAS_SLOTS;
                if (bi[ANG_OF] > JKV_LMAX || bj[ANG_OF] > JKV_LMAX) continue;
                const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF];
                const int nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
                const double *ei = env + bi[PTR_EXP], *ej = env + bj[PTR_EXP];
                const double *ci = env + bi[PTR_COEFF], *cj = env + bj[PTR_COEFF];
                const double *Ri = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double *Rj = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double r2 = (Ri[0] - Rj[0]) * (Ri[0] - Rj[0]) + (Ri[1] - Rj[1]) * (Ri[1] - Rj[1])
                                + (Ri[2] - Rj[2]) * (Ri[2] - Rj[2]);
                GPair *P = cx->pair + pq;
                const int nmax = npi * npj, nc = nci * ncj;
                P->nc = nc; P->nci = nci; P->ncj = ncj; P->ld = nmax;
                P->li = bi[ANG_OF]; P->lj = bj[ANG_OF];
                for (k = 0; k < 3; k++) P->rij[k] = Ri[k] - Rj[k];
                P->p = malloc(sizeof(double) * nmax * (13 + nc));
                P->px = P->p + nmax; P->py = P->px + nmax; P->pz = P->py + nmax;
                P->kp = P->pz + nmax; P->bnd = P->kp + nmax;
                P->rix = P->bnd + nmax; P->riy = P->rix + nmax; P->riz = P->riy + nmax;
                P->rjx = P->riz + nmax; P->rjy = P->rjx + nmax; P->rjz = P->rjy + nmax;
                P->beta = P->rjz + nmax;
                P->c = P->beta + nmax;
                int pi, pj, a, b, t = 0, x;
                double bmax = 0;
                for (pj = 0; pj < npj; pj++)
                for (pi = 0; pi < npi; pi++) {
                        const double ai = ei[pi], aj = ej[pj], p = ai + aj, ip = 1. / p;
                        const double e = exp(-ai * aj * ip * r2);
                        if (e < GEN_KCUT) continue;
                        double cm = 0;
                        for (b = 0; b < ncj; b++)
                                for (a = 0; a < nci; a++)
                                        cm = fmax(cm, fabs(ci[a * npi + pi] * cj[b * npj + pj]));
                        double P_[3], ni = 0, nj = 0;
                        for (x = 0; x < 3; x++) {
                                P_[x] = (ai * Ri[x] + aj * Rj[x]) * ip;
                                ni += (P_[x] - Ri[x]) * (P_[x] - Ri[x]);
                                nj += (P_[x] - Rj[x]) * (P_[x] - Rj[x]);
                        }
                        P->p[t] = p;
                        P->px[t] = P_[0]; P->py[t] = P_[1]; P->pz[t] = P_[2];
                        P->rix[t] = P_[0] - Ri[0]; P->riy[t] = P_[1] - Ri[1]; P->riz[t] = P_[2] - Ri[2];
                        P->rjx[t] = P_[0] - Rj[0]; P->rjy[t] = P_[1] - Rj[1]; P->rjz[t] = P_[2] - Rj[2];
                        P->kp[t] = e * ip;
                        /* a generous bound of the Obara-Saika factors: (|PA| + 1 / sqrt(p) + 1)^l per shell */
                        const double fi = sqrt(ni) + sqrt(ip) + 1, fj = sqrt(nj) + sqrt(ip) + 1;
                        const double pf = pow(fi, P->li) * pow(fj, P->lj);
                        P->bnd[t] = e * ip * cm * pf / sqrt(p);
                        P->beta[t] = e * ip * cm * pf;
                        bmax = fmax(bmax, e * ip * cm * pf);
                        for (b = 0; b < ncj; b++)
                                for (a = 0; a < nci; a++)
                                        P->c[(b * nci + a) * nmax + t] = ci[a * npi + pi] * cj[b * npj + pj];
                        t++;
                }
                P->n = t;
                P->bmax = bmax;
                double *arrs[13] = {P->p, P->px, P->py, P->pz, P->kp, P->bnd, P->rix, P->riy, P->riz,
                                    P->rjx, P->rjy, P->rjz, P->beta};
                double *bcopy = malloc(sizeof(double) * (t > 0 ? t : 1));
                memcpy(bcopy, P->beta, sizeof(double) * t);
                sort_prims(arrs, 13, P->c, nc, nmax, t, bcopy);
                free(bcopy);
        }
        return cx;
}

void SOCVHFgen_ctx_free(GenCtx *cx)
{
        if (!cx) return;
        const int npair = cx->nbas * (cx->nbas + 1) / 2;
        int pq;
        for (pq = 0; pq < npair; pq++) free(cx->pair[pq].p);
        free(cx->pair);
        free(cx->tab);
        free(cx);
}

/* 1 if SOCVHFgen_int2e takes the quartet (every l <= JKV_LMAX, contractions <= G_MAXCTR) */
int SOCVHFgen_takes(const int *shls, const int *bas)
{
        int s;
        for (s = 0; s < 4; s++) {
                const int *b = bas + shls[s] * BAS_SLOTS;
                if (b[ANG_OF] > JKV_LMAX || b[NCTR_OF] > G_MAXCTR) return 0;
        }
        return 1;
}

/* doubles of workspace for SOCVHFgen_int2e: v, w, SK of the generated kernels */
size_t SOCVHFgen_wksize(int *bas, int nbas)
{
        int i, np = 1;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + ANG_OF] <= JKV_LMAX && bas[i * BAS_SLOTS + NPRIM_OF] > np)
                        np = bas[i * BAS_SLOTS + NPRIM_OF];
        const size_t ne = N3(2 * JKV_LMAX) - (JKV_LMAX > 0 ? N3(JKV_LMAX - 1) : 0);
        const size_t NT = ne * ne;
        return NT * np * np + (size_t)G_MAXCTR * G_MAXCTR * NT
             + (size_t)G_MAXCTR * G_MAXCTR * G_MAXCTR * G_MAXCTR * NT;
}

/*
 * out of the shell quartet shls (i >= j, k >= l; SOCVHFgen_takes true) in
 * int2e_sph's layout; returns 0 when every ket primitive pair was screened
 * out (out not set).
 */
int SOCVHFgen_int2e(double *out, const int *shls, const GenCtx *cx, double dmax, double prim_tol,
                    double *wk, const int *bas)
{
        const int ij = shls[0] * (shls[0] + 1) / 2 + shls[1];
        const int kl = shls[2] * (shls[2] + 1) / 2 + shls[3];
        const GPair *B = cx->pair + ij, *K = cx->pair + kl;
        if (B->n == 0 || K->n == 0) return 0;
        const int bs = B->li < B->lj, ks = K->li < K->lj;
        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
        const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
        const double fac = TWO_PI52 * common_fac(la) * common_fac(lb) * common_fac(lc) * common_fac(ld);
        const double cut = prim_tol / (fac * B->bmax * dmax * B->nc * K->nc);
        /* int2e_sph's layout ((l DK + k) DJ + j) DI + i; role (A, B, C, D) -> position */
        const size_t DI = (size_t)B->nci * (2 * B->li + 1), DJ = (size_t)B->ncj * (2 * B->lj + 1);
        const size_t DK = (size_t)K->nci * (2 * K->li + 1);
        const size_t stp[4] = {1, DI, DI * DJ, DI * DJ * DK};
        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
        const size_t st[4] = {stp[pos[0]], stp[pos[1]], stp[pos[2]], stp[pos[3]]};
        (void)bas;
        return jkg_table[la][lb][lc][ld](out, st, pos, B, bs, K, ks, cut, fac, cx->tab, wk);
}
