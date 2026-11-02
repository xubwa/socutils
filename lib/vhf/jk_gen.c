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
#include "jk_gen.h"

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
                P->p = malloc(sizeof(double) * nmax * (15 + nc));
                P->px = P->p + nmax; P->py = P->px + nmax; P->pz = P->py + nmax;
                P->kp = P->pz + nmax; P->bnd = P->kp + nmax;
                P->rix = P->bnd + nmax; P->riy = P->rix + nmax; P->riz = P->riy + nmax;
                P->rjx = P->riz + nmax; P->rjy = P->rjx + nmax; P->rjz = P->rjy + nmax;
                P->beta = P->rjz + nmax;
                P->ei = P->beta + nmax; P->ej = P->ei + nmax;
                P->c = P->ej + nmax;
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
                        P->ei[t] = ai;
                        P->ej[t] = aj;
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
                double *arrs[15] = {P->p, P->px, P->py, P->pz, P->kp, P->bnd, P->rix, P->riy, P->riz,
                                    P->rjx, P->rjy, P->rjz, P->beta, P->ei, P->ej};
                double *bcopy = malloc(sizeof(double) * (t > 0 ? t : 1));
                memcpy(bcopy, P->beta, sizeof(double) * t);
                sort_prims(arrs, 15, P->c, nc, nmax, t, bcopy);
                double am = 0;
                for (k = 0; k < npi; k++) am = fmax(am, ei[k]);
                for (k = 0; k < npj; k++) am = fmax(am, ej[k]);
                P->amax = am;
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
             + (size_t)G_MAXCTR * G_MAXCTR * G_MAXCTR * G_MAXCTR * NT
             + (size_t)(4 * JKV_LMAX + 1) * np * np;           /* F_m of the bra primitives */
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

/* ------------------------------------------------------------------------
 * First derivatives of E_g = jfac_g E_J + kfac_g E_K at a fixed symmetric
 * density (the conventions of hess2e_cart.c / hgp.c), Cartesian basis, by the
 * generated kernels of grad_vrr_gen_*.c (gen_grad_vrr.py) for the quartets of
 * l <= GGD_LMAX whose total l is in [gen_lmin, gen_lmax]: the unique shell
 * quartets with q_ij q_kl times the J / K density bounds >= tol, the
 * symmetrized density weights dd of the quartet (as hgp.c), the derivatives
 * of the roles A, B, C from the kernel, D by translational invariance.
 * grad (ngroup, natm, 3) +=.  Same signature as SOCVHFhgp_grad2e (mode 0).
 * ------------------------------------------------------------------------ */
#include "grad_vrr_gen.h"
#define GG_MAXGROUP 4
static int gen_grad_lmin = 0, gen_grad_lmax = 1000;
void SOCVHFgen_grad_set_lrange(int lo, int hi) { gen_grad_lmin = lo; gen_grad_lmax = hi; }

int SOCVHFgen_grad_takes(const int *bas, int nbas)
{
        int i;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + ANG_OF] > GGD_LMAX || bas[i * BAS_SLOTS + NCTR_OF] > G_MAXCTR) return 0;
        return 1;
}

static inline double dmax2g(double a, double b) { return a > b ? a : b; }

void SOCVHFgen_grad2e(double *grad, int mode, int ngroup, double *jfac, double *kfac,
                      double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                      int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (mode != 0 || ngroup > GG_MAXGROUP || !SOCVHFgen_grad_takes(bas, nbas)) return;
        const size_t nao = ao_loc[nbas];
        const size_t nh1 = (size_t)natm * 3, nh = nh1 * ngroup;
        const int npair = nbas * (nbas + 1) / 2;
        GenCtx *cx = SOCVHFgen_ctx_build(atm, natm, bas, nbas, env);
        double jw = 0, kw = 0;
        int g, ib, np = 1, maxf = 0;
        for (g = 0; g < ngroup; g++) {
                jw = dmax2g(jw, fabs(jfac[g]) * 4);
                kw = dmax2g(kw, fabs(kfac[g]) * 2);
        }
        for (ib = 0; ib < nbas; ib++) {
                if (bas[ib * BAS_SLOTS + NPRIM_OF] > np) np = bas[ib * BAS_SLOTS + NPRIM_OF];
                const int nf = ao_loc[ib + 1] - ao_loc[ib];
                if (nf > maxf) maxf = nf;
        }
        const size_t maxf4 = (size_t)maxf * maxf * maxf * maxf;
        const size_t c2 = (size_t)G_MAXCTR * G_MAXCTR;
        const size_t nwk = (size_t)GGD_NT * np * np + c2 * GGD_NW + c2 * c2 * GGD_NW
                         + (size_t)(4 * GGD_LMAX + 2) * np * np;
#pragma omp parallel
{
        double *hloc = calloc(nh, sizeof(double));
        double *dd = malloc(sizeof(double) * maxf4 * ngroup);
        double *wk = malloc(sizeof(double) * nwk);
        double sum[GG_MAXGROUP * 9];
        int ij, kl, ish, jsh, ksh, lsh;
#pragma omp for schedule(dynamic, 1)
        for (ij = npair - 1; ij >= 0; ij--) {
                shell_pair(ij, &ish, &jsh);
                shell_pair(0, &ksh, &lsh);
                for (kl = 0; kl <= ij; kl++, (lsh == ksh ? (ksh++, lsh = 0) : lsh++)) {
                        const double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        const double dbound = jw * dcond[ish * nbas + jsh] * dcond[ksh * nbas + lsh]
                                            + kw * dmax2g(dcond[ish * nbas + ksh] * dcond[jsh * nbas + lsh],
                                                          dcond[ish * nbas + lsh] * dcond[jsh * nbas + ksh]);
                        if (q4 * dbound < tol) continue;
                        const int sh[4] = {ish, jsh, ksh, lsh};
                        int l[4], nf[4], nctr[4], at[4], s;
                        double fac = 1;
                        for (s = 0; s < 4; s++) {
                                const int *b = bas + sh[s] * BAS_SLOTS;
                                l[s] = b[ANG_OF];
                                nf[s] = (l[s] + 1) * (l[s] + 2) / 2;
                                nctr[s] = b[NCTR_OF];
                                at[s] = b[ATOM_OF];
                                fac *= common_fac(l[s]);
                        }
                        const int L = l[0] + l[1] + l[2] + l[3];
                        if (L < gen_grad_lmin || L > gen_grad_lmax || L > GGD_LSUM_MAX) continue;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        const GPair *B = cx->pair + ij, *K = cx->pair + kl;
                        if (B->n == 0 || K->n == 0) continue;
                        const int NI = nctr[0] * nf[0], NJ = nctr[1] * nf[1], NK = nctr[2] * nf[2], NL = nctr[3] * nf[3];
                        const size_t nd = (size_t)NI * NJ * NK * NL;
                        const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh], c0 = ao_loc[ksh], e0 = ao_loc[lsh];
                        int a, b, c, d;
                        double ddmax = 0;
                        for (g = 0; g < ngroup; g++) {
                                double *ddg = dd + g * nd;
                                const double jf = 4 * jfac[g] * fac, kf = kfac[g] * fac;
                                for (a = 0; a < NI; a++)
                                for (b = 0; b < NJ; b++) {
                                        const double dab = jf * dm[(a0 + a) * nao + b0 + b];
                                        const double *dA = dm + (a0 + a) * nao, *dB = dm + (b0 + b) * nao;
                                        for (c = 0; c < NK; c++) {
                                                const double dbc = dB[c0 + c], dac = dA[c0 + c];
                                                const double *dC = dm + (c0 + c) * nao;
                                                double *o = ddg + ((a * NJ + b) * NK + c) * NL;
                                                for (d = 0; d < NL; d++) {
                                                        o[d] = dab * dC[e0 + d] + kf * (dac * dB[e0 + d] + dA[e0 + d] * dbc);
                                                        ddmax = dmax2g(ddmax, fabs(o[d]));
                                                }
                                        }
                                }
                        }
                        if (ddmax == 0) continue;
                        const int bs = B->li < B->lj, ks = K->li < K->lj;
                        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
                        const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
                        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
                        const size_t stp[4] = {(size_t)NJ * NK * NL, (size_t)NK * NL, (size_t)NL, 1};
                        const size_t ds[4] = {stp[pos[0]], stp[pos[1]], stp[pos[2]], stp[pos[3]]};
                        /* derivative factors: 2 alpha and the extra power of the raised shell */
                        const double der = (1 + 2 * B->amax) * (1 + 2 * K->amax);
                        const double cut = prim_tol / (TWO_PI52 * B->bmax * ddmax * B->nc * K->nc * der);
                        memset(sum, 0, sizeof(double) * ngroup * 9);
                        const ggd_fn kern = ggd_table[la][lb][lc][ld];
                        if (kern == NULL || !kern(sum, dd, ngroup, nd, ds, pos, B, bs, K, ks, cut, cx->tab, wk))
                                continue;
                        for (g = 0; g < ngroup; g++) {
                                double *hg = hloc + g * nh1;
                                int x;
                                for (x = 0; x < 3; x++) {
                                        const double gA = TWO_PI52 * sum[g * 9 + x * 3];
                                        const double gB = TWO_PI52 * sum[g * 9 + x * 3 + 1];
                                        const double gC = TWO_PI52 * sum[g * 9 + x * 3 + 2];
                                        hg[at[pos[0]] * 3 + x] += gA;
                                        hg[at[pos[1]] * 3 + x] += gB;
                                        hg[at[pos[2]] * 3 + x] += gC;
                                        hg[at[pos[3]] * 3 + x] -= gA + gB + gC;
                                }
                        }
                }
        }
#pragma omp critical
        {
                size_t n;
                for (n = 0; n < nh; n++) grad[n] += hloc[n];
        }
        free(hloc);
        free(dd);
        free(wk);
}
        SOCVHFgen_ctx_free(cx);
}
