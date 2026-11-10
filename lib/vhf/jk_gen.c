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
#include "jk_lane_gen.c"

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
        cx->tab = malloc(sizeof(double) * (GT_N * GT_TN + GT_N));     /* F_m rows, then e^-T_k */
        for (k = 0; k < GT_N; k++) f_series(cx->tab + k * GT_TN, k * GT_STEP, GT_TN - 1);
        for (k = 0; k < GT_N; k++) cx->tab[GT_N * GT_TN + k] = exp(-k * GT_STEP);
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

/* 1 if SOCVHFgen_int2e takes the quartet: every l <= JKV_LMAX, contractions <= G_MAXCTR, the class
 * generated (gen_jk_vrr.py: all of l <= 2, those with f shells up to a total l) */
int SOCVHFgen_takes(const int *shls, const int *bas)
{
        int s, l[4];
        for (s = 0; s < 4; s++) {
                const int *b = bas + shls[s] * BAS_SLOTS;
                if (b[ANG_OF] > JKV_LMAX || b[NCTR_OF] > G_MAXCTR) return 0;
                l[s] = b[ANG_OF];
        }
        const int la = l[0] > l[1] ? l[0] : l[1], lb = l[0] > l[1] ? l[1] : l[0];
        const int lc = l[2] > l[3] ? l[2] : l[3], ld = l[2] > l[3] ? l[3] : l[2];
        return jkg_table[la][lb][lc][ld] != NULL;
}

/* doubles of workspace for SOCVHFgen_int2e: v, w, SK of the generated kernels */
size_t SOCVHFgen_wksize(int *bas, int nbas)
{
        int i, np = 1;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + ANG_OF] <= JKV_LMAX && bas[i * BAS_SLOTS + NPRIM_OF] > np)
                        np = bas[i * BAS_SLOTS + NPRIM_OF];
        const size_t NT = JKG_NT;       /* the most VRR targets of a generated class */
        return NT * np * np + (size_t)G_MAXCTR * G_MAXCTR * NT
             + (size_t)G_MAXCTR * G_MAXCTR * G_MAXCTR * G_MAXCTR * NT
             + (size_t)(JKG_LSUM + 1) * np * np;               /* F_m of the bra primitives */
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
        /* int2e_sph's layout ((l DK + k) DJ + j) DI + i */
        const size_t DI = (size_t)B->nci * (2 * B->li + 1), DJ = (size_t)B->ncj * (2 * B->lj + 1);
        const size_t DK = (size_t)K->nci * (2 * K->li + 1);
        const size_t stp[4] = {1, DI, DI * DJ, DI * DJ * DK};
        /* the kernels vectorise over the bra primitive pairs: the pair of more
         * primitives goes to the bra, (ij|kl) = (kl|ij) (a d-containing pair of
         * cc-pVDZ has 1 - 3 primitive pairs, an s pair up to 64) */
        const int swap = B->n < K->n;
        if (swap) { const GPair *t = B; B = K; K = t; }
        const int bs = B->li < B->lj, ks = K->li < K->lj;
        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
        const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
        const double fac = TWO_PI52 * common_fac(la) * common_fac(lb) * common_fac(lc) * common_fac(ld);
        const double cut = prim_tol / (fac * B->bmax * dmax * B->nc * K->nc);
        /* role (A, B, C, D) -> the kernel's slot (bra i, bra j, ket i, ket j) -> position in out */
        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
        const int sw = swap ? 2 : 0;
        const size_t st[4] = {stp[(pos[0] + sw) & 3], stp[(pos[1] + sw) & 3],
                              stp[(pos[2] + sw) & 3], stp[(pos[3] + sw) & 3]};
        (void)bas;
        if (jkg_table[la][lb][lc][ld] == NULL) return 0;        /* not generated: callers check SOCVHFgen_takes */
        return jkg_table[la][lb][lc][ld](out, st, pos, B, bs, K, ks, cut, fac, cx->tab, wk);
}

/*
 * J only of the quartet shls (as SOCVHFgen_int2e): jb (DJ, DI) += (ij|kl) dk,
 * jk (DL, DK) += (ij|kl) db, the blocks of the pairs (i, k fastest; DI = the
 * functions of shell i, ...), the density contracted before the horizontal
 * recursions (jjg_* of jk_vrr_gen.c).  Returns 0 when everything was screened
 * out (jb, jk untouched).
 */
int SOCVHFgen_j2e(double *jb, double *jk, const double *db, const double *dk, const int *shls,
                  const GenCtx *cx, double dmax, double prim_tol, double *wk)
{
        const int ij = shls[0] * (shls[0] + 1) / 2 + shls[1];
        const int kl = shls[2] * (shls[2] + 1) / 2 + shls[3];
        const GPair *B = cx->pair + ij, *K = cx->pair + kl;
        if (B->n == 0 || K->n == 0) return 0;
        size_t DI = (size_t)B->nci * (2 * B->li + 1), DK = (size_t)K->nci * (2 * K->li + 1);
        if (B->n < K->n) {      /* the pair of more primitive pairs as the bra (vectorised over) */
                const GPair *t = B; B = K; K = t;
                double *o = jb; jb = jk; jk = o;
                const double *d = db; db = dk; dk = d;
                const size_t s = DI; DI = DK; DK = s;
        }
        const int bs = B->li < B->lj, ks = K->li < K->lj;
        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
        const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
        const double fac = TWO_PI52 * common_fac(la) * common_fac(lb) * common_fac(lc) * common_fac(ld);
        const double cut = prim_tol / (fac * B->bmax * dmax * B->nc * K->nc);
        if (jjg_table[la][lb][lc][ld] == NULL) return -1;       /* no J form (f shells) */
        return jjg_table[la][lb][lc][ld](jb, jk, db, dk, DI, DK, B, bs, K, ks, cut, fac, cx->tab, wk);
}

/* ------------------------------------------------------------------------
 * Lanes: GL_NL quartets (ij | k_l l_l) sharing the bra pair ij and the ket
 * shell types (jk_lane_gen.c, gen_jk_lanes.py).
 * ------------------------------------------------------------------------ */

/* type[i]: shells of one type have equal l, primitives, contractions, exponents and coefficients;
 * returns the number of types */
int SOCVHFgen_shell_types(int *type, const int *bas, int nbas)
{
        int i, j, nt = 0;
        int *rep = malloc(sizeof(int) * (nbas > 0 ? nbas : 1));
        for (i = 0; i < nbas; i++) {
                const int *b = bas + i * BAS_SLOTS;
                type[i] = -1;
                for (j = 0; j < nt; j++) {
                        const int *r = bas + rep[j] * BAS_SLOTS;
                        if (r[ANG_OF] == b[ANG_OF] && r[NPRIM_OF] == b[NPRIM_OF] && r[NCTR_OF] == b[NCTR_OF]
                            && r[PTR_EXP] == b[PTR_EXP] && r[PTR_COEFF] == b[PTR_COEFF]) {
                                type[i] = j;
                                break;
                        }
                }
                if (type[i] < 0) {
                        rep[nt] = i;
                        type[i] = nt++;
                }
        }
        free(rep);
        return nt;
}

/* 1 if the ket pair (k, l) can be a lane (l <= JKV_LMAX, contractions <= G_MAXCTR, primitive pairs <= GL_MAXP) */
int SOCVHFgen_lane_ok(int ksh, int lsh, const int *bas)
{
        const int *bk = bas + ksh * BAS_SLOTS, *bl = bas + lsh * BAS_SLOTS;
        return bk[ANG_OF] <= JKL_LMAX && bl[ANG_OF] <= JKL_LMAX && bk[NCTR_OF] <= G_MAXCTR
            && bl[NCTR_OF] <= G_MAXCTR && bk[NPRIM_OF] * bl[NPRIM_OF] <= GL_MAXP;
}

/* doubles of workspace of SOCVHFgen_int2e_lanes */
size_t SOCVHFgen_lane_wksize(const int *bas, int nbas, int nbuf)
{
        int i, mc = 1;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + NCTR_OF] > mc) mc = bas[i * BAS_SLOTS + NCTR_OF];
        if (mc > G_MAXCTR) mc = G_MAXCTR;
        return sizeof(LaneK) / sizeof(double) + 1 + (size_t)nbuf * GL_NL
             + (size_t)JKL_NT * GL_NL * (1 + (size_t)mc * mc * mc * mc);
}

/*
 * The quartets (ish jsh | ksh[l] lsh[l]), l < nl <= GL_NL, of one ket type
 * (SOCVHFgen_shell_types) and SOCVHFgen_lane_ok kets, ish, jsh of l <=
 * JKV_LMAX: quartet l into out + l * ostride in int2e_sph's layout.  dmax:
 * the largest density bound of the quartets.  Returns 0 when every primitive
 * was screened out (out not set).
 */
int SOCVHFgen_int2e_lanes(double *out, size_t ostride, int ish, int jsh, const int *ksh, const int *lsh, int nl,
                          const GenCtx *cx, double dmax, double prim_tol, double *wk,
                          const int *atm, const int *bas, const double *env)
{
        const int ij = ish * (ish + 1) / 2 + jsh;
        const GPair *B = cx->pair + ij;
        if (B->n == 0 || nl <= 0) return 0;
        LaneK *K = (LaneK *)wk;
        double *lout = wk + sizeof(LaneK) / sizeof(double) + 1;
        const int *bk = bas + ksh[0] * BAS_SLOTS, *bl = bas + lsh[0] * BAS_SLOTS;
        const int lk = bk[ANG_OF], ll = bl[ANG_OF];
        const int npk = bk[NPRIM_OF], npl = bl[NPRIM_OF], nck_ = bk[NCTR_OF], ncl = bl[NCTR_OF];
        const double *ek = env + bk[PTR_EXP], *el = env + bl[PTR_EXP];
        const double *cK = env + bk[PTR_COEFF], *cL = env + bl[PTR_COEFF];
        const int ks = lk < ll;
        int t, l, pk, pl, a, b, x;
        K->nk = npk * npl;
        K->nci = nck_; K->ncj = ncl; K->nck = nck_ * ncl;
        double Rk[GL_NL][3], Rl[GL_NL][3];
        for (l = 0; l < GL_NL; l++) {
                const int s = l < nl ? l : 0;
                const double *rk = env + atm[bas[ksh[s] * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double *rl = env + atm[bas[lsh[s] * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
                for (x = 0; x < 3; x++) {
                        Rk[l][x] = rk[x];
                        Rl[l][x] = rl[x];
                        K->cd[x][l] = ks ? rl[x] - rk[x] : rk[x] - rl[x];
                }
        }
        for (pl = 0; pl < npl; pl++)
        for (pk = 0; pk < npk; pk++) {
                t = pl * npk + pk;
                const double ak = ek[pk], al = el[pl], q = ak + al, iq = 1. / q;
                double cm = 0;
                for (b = 0; b < ncl; b++)
                        for (a = 0; a < nck_; a++) {
                                const double c = cK[a * npk + pk] * cL[b * npl + pl];
                                K->c[(b * nck_ + a) * GL_MAXP + t] = c;
                                cm = fmax(cm, fabs(c));
                        }
                K->q[t] = q;
                double bmx = 0;
                for (l = 0; l < GL_NL; l++) {
                        double P[3], r2 = 0, nk2 = 0, nl2 = 0;
                        for (x = 0; x < 3; x++) {
                                const double d = Rk[l][x] - Rl[l][x];
                                r2 += d * d;
                                P[x] = (ak * Rk[l][x] + al * Rl[l][x]) * iq;
                                nk2 += (P[x] - Rk[l][x]) * (P[x] - Rk[l][x]);
                                nl2 += (P[x] - Rl[l][x]) * (P[x] - Rl[l][x]);
                        }
                        const double e = l < nl ? exp(-ak * al * iq * r2) : 0;
                        K->qx[t][l] = P[0]; K->qy[t][l] = P[1]; K->qz[t][l] = P[2];
                        K->qcx[t][l] = P[0] - (ks ? Rl[l][0] : Rk[l][0]);
                        K->qcy[t][l] = P[1] - (ks ? Rl[l][1] : Rk[l][1]);
                        K->qcz[t][l] = P[2] - (ks ? Rl[l][2] : Rk[l][2]);
                        K->kp[t][l] = e * iq;
                        const double fi = sqrt(nk2) + sqrt(iq) + 1, fj = sqrt(nl2) + sqrt(iq) + 1;
                        const double bnd = e * iq * cm * pow(fi, lk) * pow(fj, ll) / sqrt(q);
                        if (bnd > bmx) bmx = bnd;
                }
                K->bmx[t] = bmx;
        }
        const int bs = B->li < B->lj;
        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
        const int lc = ks ? ll : lk, ld = ks ? lk : ll;
        const double fac = TWO_PI52 * common_fac(la) * common_fac(lb) * common_fac(lc) * common_fac(ld);
        const double cut = prim_tol / (fac * B->bmax * dmax * B->nc * K->nck);
        const size_t DI = (size_t)B->nci * (2 * B->li + 1), DJ = (size_t)B->ncj * (2 * B->lj + 1);
        const size_t DK = (size_t)nck_ * (2 * lk + 1), DL = (size_t)ncl * (2 * ll + 1);
        const size_t stp[4] = {1, DI, DI * DJ, DI * DJ * DK};
        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
        const size_t st[4] = {stp[pos[0]], stp[pos[1]], stp[pos[2]], stp[pos[3]]};
        const size_t nf = DI * DJ * DK * DL;
        double *kwk = lout + nf * GL_NL;
        if (!jkl_table[la][lb][lc][ld](lout, st, pos, B, bs, K, cut, fac, cx->tab, kwk)) return 0;
        size_t e;
        for (l = 0; l < nl; l++) {
                double *o = out + l * ostride;
                for (e = 0; e < nf; e++) o[e] = lout[e * GL_NL + l];
        }
        return 1;
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
        /* quartets with a shell of l > GGD_LMAX or more than G_MAXCTR contractions are skipped
         * (the Rys kernel's, hess2e_cart.c SOCVHFcart_set_skip_gen) */
        if (mode != 0 || ngroup > GG_MAXGROUP) return;
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
                        if (l[0] > GGD_LMAX || l[1] > GGD_LMAX || l[2] > GGD_LMAX || l[3] > GGD_LMAX
                            || nctr[0] > G_MAXCTR || nctr[1] > G_MAXCTR || nctr[2] > G_MAXCTR
                            || nctr[3] > G_MAXCTR) continue;
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
                        /* the pair of the higher l as the bra: its two derivative streams (A, B) share
                         * the raised bra recursion, the ket's one (C) is then the cheap one (VRR of
                         * (21|00) 575 operations against 938 for (00|21)) */
                        const int swap = K->li + K->lj > B->li + B->lj;
                        if (swap) { const GPair *t_ = B; B = K; K = t_; }
                        const int sw = swap ? 2 : 0;
                        const int bs = B->li < B->lj, ks = K->li < K->lj;
                        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
                        const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
                        /* role -> the kernel's slot (bra i, bra j, ket i, ket j); slot -> position (slot + sw) & 3 */
                        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
                        const size_t stp[4] = {(size_t)NJ * NK * NL, (size_t)NK * NL, (size_t)NL, 1};
                        const size_t ds[4] = {stp[(pos[0] + sw) & 3], stp[(pos[1] + sw) & 3],
                                              stp[(pos[2] + sw) & 3], stp[(pos[3] + sw) & 3]};
                        const int ra[4] = {at[(pos[0] + sw) & 3], at[(pos[1] + sw) & 3],
                                           at[(pos[2] + sw) & 3], at[(pos[3] + sw) & 3]};
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
                                        hg[ra[0] * 3 + x] += gA;
                                        hg[ra[1] * 3 + x] += gB;
                                        hg[ra[2] * 3 + x] += gC;
                                        hg[ra[3] * 3 + x] -= gA + gB + gC;
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

/* ------------------------------------------------------------------------
 * Second derivatives of E_g = jfac_g E_J + kfac_g E_K, the order-2 analogue of
 * SOCVHFgen_grad2e by the generated kernels of hess_vrr_gen_*.c
 * (gen_hess_vrr.py): the blocks AA, AB, AC, BB, BC, CC of the roles from the
 * kernel, those with D by translational invariance.  hess (ngroup, natm,
 * natm, 3, 3) +=.  Same signature as SOCVHFhess2e_jk_cart (mode 0).
 * ------------------------------------------------------------------------ */
#include "hess_vrr_gen.h"
static int gen_hess_lmin = 0, gen_hess_lmax = 1000;
void SOCVHFgen_hess_set_lrange(int lo, int hi) { gen_hess_lmin = lo; gen_hess_lmax = hi; }
int SOCVHFgen_hess_lsum_max() { return GHD_LSUM_MAX; }

void SOCVHFgen_hess2e(double *hess, int mode, int ngroup, double *jfac, double *kfac,
                      double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                      int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (mode != 0 || ngroup > GG_MAXGROUP) return;
        const size_t nao = ao_loc[nbas];
        const size_t nh1 = (size_t)natm * natm * 9, nh = nh1 * ngroup;
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
        const size_t nwk = (size_t)GHD_NT * np * np + c2 * GHD_NY + c2 * c2 * GHD_NW
                         + (size_t)GHD_NF * np * np;
#pragma omp parallel
{
        double *hloc = calloc(nh, sizeof(double));
        double *dd = malloc(sizeof(double) * maxf4 * ngroup);
        double *wk = malloc(sizeof(double) * nwk);
        double sum[GG_MAXGROUP * 54];
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
                        if (L < gen_hess_lmin || L > gen_hess_lmax || L > GHD_LSUM_MAX) continue;
                        if (l[0] > GHD_LMAX || l[1] > GHD_LMAX || l[2] > GHD_LMAX || l[3] > GHD_LMAX
                            || nctr[0] > G_MAXCTR || nctr[1] > G_MAXCTR || nctr[2] > G_MAXCTR
                            || nctr[3] > G_MAXCTR) continue;
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
                        const int swap = K->li + K->lj > B->li + B->lj;
                        if (swap) { const GPair *t_ = B; B = K; K = t_; }
                        const int sw = swap ? 2 : 0;
                        const int bs = B->li < B->lj, ks = K->li < K->lj;
                        const int la = bs ? B->lj : B->li, lb = bs ? B->li : B->lj;
                        const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
                        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
                        const size_t stp[4] = {(size_t)NJ * NK * NL, (size_t)NK * NL, (size_t)NL, 1};
                        const size_t ds[4] = {stp[(pos[0] + sw) & 3], stp[(pos[1] + sw) & 3],
                                              stp[(pos[2] + sw) & 3], stp[(pos[3] + sw) & 3]};
                        const int ra[4] = {at[(pos[0] + sw) & 3], at[(pos[1] + sw) & 3],
                                           at[(pos[2] + sw) & 3], at[(pos[3] + sw) & 3]};
                        const double der = (1 + 2 * B->amax) * (1 + 2 * B->amax) * (1 + 2 * K->amax) * (1 + 2 * K->amax);
                        const double cut = prim_tol / (TWO_PI52 * B->bmax * ddmax * B->nc * K->nc * der);
                        memset(sum, 0, sizeof(double) * ngroup * 54);
                        const ggd_fn kern = ghd_table[la][lb][lc][ld];
                        if (kern == NULL || !kern(sum, dd, ngroup, nd, ds, pos, B, bs, K, ks, cut, cx->tab, wk))
                                continue;
                        /* the blocks in the slots AA AB AC BB BC CC */
                        static const int blk[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
                        for (g = 0; g < ngroup; g++) {
                                const double *sm = sum + g * 54;
                                double Gf[4][4][3][3];
                                int s1, t1, x, y;
                                for (s1 = 0; s1 < 3; s1++)
                                for (t1 = 0; t1 < 3; t1++) {
                                        const double *hb = sm + blk[s1][t1] * 9;
                                        for (x = 0; x < 3; x++)
                                        for (y = 0; y < 3; y++)
                                                Gf[s1][t1][x][y] = TWO_PI52 * (s1 <= t1 ? hb[x * 3 + y] : hb[y * 3 + x]);
                                }
                                for (x = 0; x < 3; x++)
                                for (y = 0; y < 3; y++) {
                                        double dd3 = 0;
                                        for (s1 = 0; s1 < 3; s1++) {
                                                double r3 = 0, c3 = 0;
                                                for (t1 = 0; t1 < 3; t1++) {
                                                        r3 += Gf[s1][t1][x][y];
                                                        c3 += Gf[t1][s1][x][y];
                                                }
                                                Gf[s1][3][x][y] = -r3;
                                                Gf[3][s1][x][y] = -c3;
                                                dd3 += r3;
                                        }
                                        Gf[3][3][x][y] = dd3;
                                }
                                double *hg = hloc + g * nh1;
                                for (s1 = 0; s1 < 4; s1++)
                                for (t1 = 0; t1 < 4; t1++) {
                                        double *hp = hg + ((size_t)ra[s1] * natm + ra[t1]) * 9;
                                        for (x = 0; x < 3; x++)
                                                for (y = 0; y < 3; y++) hp[x * 3 + y] += Gf[s1][t1][x][y];
                                }
                        }
                }
        }
#pragma omp critical
        {
                size_t n;
                for (n = 0; n < nh; n++) hess[n] += hloc[n];
        }
        free(hloc);
        free(dd);
        free(wk);
}
        SOCVHFgen_ctx_free(cx);
}
#include <omp.h>

/* ------------------------------------------------------------------------
 * First-order J / K of all nuclear displacements (fock1.c SOCVHFfock1_jk) in
 * the Cartesian basis: the derivative integrals of the three roles A, B, C of
 * a quartet from one generated HGP pass (ggi_* of gen_grad_vrr.py raw), the
 * fourth by translational invariance; the classes without a generated
 * kernel (l > GGD_LMAX, more than G_MAXCTR contractions, total l above
 * GGD_LSUM_MAX) by three libcint int2e_ip1_cart calls as fock1.c.  The
 * digestion as fock1.c (W += ..., V = W + W^T).  vj, vk (atm1 - atm0, 3, nao,
 * nao) of the Cartesian AOs; opt for int2e_ip1_cart.
 * ------------------------------------------------------------------------ */
int int2e_ip1_cart(double *out, int *dims, int *shls, int *atm, int natm,
                   int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);
static int fock1_gen_on = 1;
void SOCVHFgen_fock1_set(int on) { fock1_gen_on = on; }

static void f1_reduce(double *out, double **bufs, int nbuf, size_t n)
{
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
        const size_t chunk = (n + nt - 1) / nt;
        const size_t s0 = chunk * t < n ? chunk * t : n;
        const size_t s1 = s0 + chunk < n ? s0 + chunk : n;
        size_t i;
        int k;
        for (i = s0; i < s1; i++) out[i] = 0;
        for (k = 0; k < nbuf; k++) {
                const double *b = bufs[k];
                for (i = s0; i < s1; i++) out[i] += b[i];
        }
}

static inline void f1_add_block(double *W, size_t nao, size_t r0, size_t c0, const double *L, int dr, int dc)
{
        int r, c;
        for (r = 0; r < dr; r++) {
                double *w = W + (r0 + r) * nao + c0;
                const double *l = L + r * dc;
                for (c = 0; c < dc; c++) w[c] += l[c];
        }
}

void SOCVHFgen_fock1_jk2(double *vj, double *vk, double *dm, int with_j, int with_k,
                         double *qcond, double *dcond, double tol, double prim_tol, CINTOpt *opt,
                         int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                         int atm0, int atm1);

void SOCVHFgen_fock1_jk(double *vj, double *vk, double *dm, int with_k,
                        double *qcond, double *dcond, double tol, double prim_tol, CINTOpt *opt,
                        int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                        int atm0, int atm1)
{
        SOCVHFgen_fock1_jk2(vj, vk, dm, 1, with_k, qcond, dcond, tol, prim_tol, opt, ao_loc, atm, natm,
                            bas, nbas, env, atm0, atm1);
}

/* with_j 0 (density-fitted J elsewhere): K only, the quartets screened by the K density pairs alone,
 * vj untouched (may be NULL) */
void SOCVHFgen_fock1_jk2(double *vj, double *vk, double *dm, int with_j, int with_k,
                         double *qcond, double *dcond, double tol, double prim_tol, CINTOpt *opt,
                         int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                         int atm0, int atm1)
{
        const size_t nao = ao_loc[nbas];
        const size_t nn = nao * nao;
        const int nbat = atm1 - atm0;
        const size_t nv = (size_t)nbat * 3 * nn;
        int dmx = 0, i, np = 1;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmx = d > dmx ? d : dmx;
                if (bas[i * BAS_SLOTS + NPRIM_OF] > np) np = bas[i * BAS_SLOTS + NPRIM_OF];
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        const size_t cache_size = GTOmax_cache_size(int2e_ip1_cart, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nf4 = (size_t)dmx * dmx * dmx * dmx;
        const size_t nblk = (size_t)dmx * dmx;
        const size_t c2 = (size_t)G_MAXCTR * G_MAXCTR;
        const size_t nwk = (size_t)GGD_NT * np * np + c2 * GGD_NW + c2 * c2 * GGD_NW
                         + (size_t)(4 * GGD_LMAX + 2) * np * np;
        const int npair = nbas * (nbas + 1) / 2;
        GenCtx *cx = SOCVHFgen_ctx_build(atm, natm, bas, nbas, env);
        const int nthmax = omp_get_max_threads();
        double **jbufs = calloc(nthmax, sizeof(double *));
        double **kbufs = calloc(nthmax, sizeof(double *));
        int nth_used = 1;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
#pragma omp single
        nth_used = omp_get_num_threads();
        double *wj = with_j ? calloc(nv, sizeof(double)) : NULL;
        double *wk = with_k ? calloc(nv, sizeof(double)) : NULL;
        jbufs[tid] = wj;
        kbufs[tid] = wk;
        /* G[pos][x][((i NJ + j) NK + k) NL + l]: d/dR of the position's centre */
        double *G = malloc(sizeof(double) * 12 * nf4);
        double *lb = malloc(sizeof(double) * (9 * nf4 + cache_size));
        double *gwk = malloc(sizeof(double) * nwk);
        double *loc = malloc(sizeof(double) * 12 * nblk * 6);
        double *Jij = loc, *Jkl = loc + 12 * nblk;
        double *Kil = loc + 24 * nblk, *Kjl = loc + 36 * nblk;
        double *Kik = loc + 48 * nblk, *Kjk = loc + 60 * nblk;
        int ij, kl, ish, jsh, ksh, lsh, x, pos;
#pragma omp for schedule(dynamic, 2)
        for (ij = npair - 1; ij >= 0; ij--) {
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        const double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        const double dkq = dmax2g(dmax2g(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                                  dmax2g(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh]));
                        const double djq = with_j ? dmax2g(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]) : 0;
                        const double dq = dmax2g(djq, with_k ? dkq : 0);
                        if (q4 * dq < tol) continue;
                        const int sh[4] = {ish, jsh, ksh, lsh};
                        int at[4], l[4], nctr[4], s, in_range = 0;
                        double nfac = 1;
                        for (s = 0; s < 4; s++) {
                                const int *b = bas + sh[s] * BAS_SLOTS;
                                at[s] = b[ATOM_OF];
                                l[s] = b[ANG_OF];
                                nctr[s] = b[NCTR_OF];
                                nfac *= common_fac(l[s]);
                                in_range |= at[s] >= atm0 && at[s] < atm1;
                        }
                        if (!in_range) continue;
                        const int di = ao_loc[ish + 1] - ao_loc[ish], dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        const int dk = ao_loc[ksh + 1] - ao_loc[ksh], dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        const size_t nf = (size_t)di * dj * dk * dl;
                        const int L = l[0] + l[1] + l[2] + l[3];
                        int done = 0;
                        if (fock1_gen_on && L <= GGD_LSUM_MAX && l[0] <= GGD_LMAX && l[1] <= GGD_LMAX
                            && l[2] <= GGD_LMAX && l[3] <= GGD_LMAX && nctr[0] <= G_MAXCTR && nctr[1] <= G_MAXCTR
                            && nctr[2] <= G_MAXCTR && nctr[3] <= G_MAXCTR) {
                                const GPair *B = cx->pair + ij, *K = cx->pair + kl;
                                if (B->n == 0 || K->n == 0) continue;
                                const int swap = K->li + K->lj > B->li + B->lj;
                                if (swap) { const GPair *t_ = B; B = K; K = t_; }
                                const int sw = swap ? 2 : 0;
                                const int bs = B->li < B->lj, ks = K->li < K->lj;
                                const int la = bs ? B->lj : B->li, lb_ = bs ? B->li : B->lj;
                                const int lc = ks ? K->lj : K->li, ld = ks ? K->li : K->lj;
                                const int ps[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
                                const size_t stp[4] = {(size_t)dj * dk * dl, (size_t)dk * dl, (size_t)dl, 1};
                                const size_t ds[4] = {stp[(ps[0] + sw) & 3], stp[(ps[1] + sw) & 3],
                                                      stp[(ps[2] + sw) & 3], stp[(ps[3] + sw) & 3]};
                                const double der = (1 + 2 * B->amax) * (1 + 2 * K->amax);
                                const double cut = prim_tol / (TWO_PI52 * nfac * B->bmax * dmax2g(dq, 1e-300)
                                                               * B->nc * K->nc * der);
                                const ggi_fn kern = ggi_table[la][lb_][lc][ld];
                                if (kern != NULL) {
                                        if (!kern(lb, nf, ds, ps, B, bs, K, ks, cut, cx->tab, gwk)) continue;
                                        /* kernel slot s -> position (ps[s] + sw) & 3 */
                                        const double sc = TWO_PI52 * nfac;
                                        const int p0 = (ps[0] + sw) & 3, p1 = (ps[1] + sw) & 3, p2 = (ps[2] + sw) & 3;
                                        const int p3 = 6 - p0 - p1 - p2;
                                        size_t n;
                                        for (x = 0; x < 3; x++) {
                                                double *g0 = G + (p0 * 3 + x) * nf, *g1 = G + (p1 * 3 + x) * nf;
                                                double *g2 = G + (p2 * 3 + x) * nf, *g3 = G + (p3 * 3 + x) * nf;
                                                const double *r0 = lb + x * nf, *r1 = lb + (3 + x) * nf, *r2 = lb + (6 + x) * nf;
                                                for (n = 0; n < nf; n++) {
                                                        g0[n] = sc * r0[n];
                                                        g1[n] = sc * r1[n];
                                                        g2[n] = sc * r2[n];
                                                        g3[n] = -(g0[n] + g1[n] + g2[n]);
                                                }
                                        }
                                        done = 1;
                                }
                        }
                        if (!done) {
                                /* libcint: (ij|kl), (ji|kl), (kl|ij), -nabla of the first centre */
                                double *b0 = lb, *b1 = lb + 3 * nf, *b2 = lb + 6 * nf, *cache = lb + 9 * nf4;
                                int shls[4] = {ish, jsh, ksh, lsh}, ok = 0;
                                if (int2e_ip1_cart(b0, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) ok = 1;
                                else memset(b0, 0, sizeof(double) * 3 * nf);
                                shls[0] = jsh; shls[1] = ish;
                                if (int2e_ip1_cart(b1, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) ok = 1;
                                else memset(b1, 0, sizeof(double) * 3 * nf);
                                shls[0] = ksh; shls[1] = lsh; shls[2] = ish; shls[3] = jsh;
                                if (int2e_ip1_cart(b2, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) ok = 1;
                                else memset(b2, 0, sizeof(double) * 3 * nf);
                                if (!ok) continue;
                                int i0, j0, k0, l0;
                                for (x = 0; x < 3; x++)
                                for (i0 = 0; i0 < di; i0++)
                                for (j0 = 0; j0 < dj; j0++)
                                for (k0 = 0; k0 < dk; k0++)
                                for (l0 = 0; l0 < dl; l0++) {
                                        const size_t o = ((i0 * dj + j0) * dk + k0) * dl + l0;
                                        const double g0 = -b0[x * nf + ((l0 * dk + k0) * dj + j0) * di + i0];
                                        const double g1 = -b1[x * nf + ((l0 * dk + k0) * di + i0) * dj + j0];
                                        const double g2 = -b2[x * nf + ((j0 * di + i0) * dl + l0) * dk + k0];
                                        G[x * nf + o] = g0;
                                        G[(3 + x) * nf + o] = g1;
                                        G[(6 + x) * nf + o] = g2;
                                        G[(9 + x) * nf + o] = -(g0 + g1 + g2);
                                }
                        }
                        double fac = 1.;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        memset(loc, 0, sizeof(double) * 12 * nblk * 6);
                        const size_t i0s = ao_loc[ish], j0s = ao_loc[jsh], k0s = ao_loc[ksh], l0s = ao_loc[lsh];
                        int i0, j0, k0, l0;
                        for (i0 = 0; i0 < di; i0++)
                        for (j0 = 0; j0 < dj; j0++) {
                                const size_t mu = i0s + i0, nu = j0s + j0;
                                const double Dmn = dm[mu * nao + nu];
                                for (k0 = 0; k0 < dk; k0++) {
                                        const size_t la_ = k0s + k0;
                                        const double Dnl = dm[nu * nao + la_], Dml = dm[mu * nao + la_];
                                        for (l0 = 0; l0 < dl; l0++) {
                                                const size_t si = l0s + l0;
                                                const size_t o = ((i0 * dj + j0) * dk + k0) * dl + l0;
                                                const double Dls = dm[la_ * nao + si];
                                                const double Dns = dm[nu * nao + si], Dms = dm[mu * nao + si];
                                                for (pos = 0; pos < 4; pos++)
                                                for (x = 0; x < 3; x++) {
                                                        const size_t base = (size_t)(pos * 3 + x) * nblk;
                                                        const double gg = G[(pos * 3 + x) * nf + o] * fac;
                                                        if (with_j) {
                                                                Jij[base + i0 * dj + j0] += 2 * gg * Dls;
                                                                Jkl[base + k0 * dl + l0] += 2 * gg * Dmn;
                                                        }
                                                        if (with_k) {
                                                                Kil[base + i0 * dl + l0] += gg * Dnl;
                                                                Kjl[base + j0 * dl + l0] += gg * Dml;
                                                                Kik[base + i0 * dk + k0] += gg * Dns;
                                                                Kjk[base + j0 * dk + k0] += gg * Dms;
                                                        }
                                                }
                                        }
                                }
                        }
                        for (pos = 0; pos < 4; pos++) {
                                if (at[pos] < atm0 || at[pos] >= atm1) continue;
                                for (x = 0; x < 3; x++) {
                                        const size_t base = (size_t)(pos * 3 + x) * nblk;
                                        if (with_j) {
                                                double *WJ = wj + ((size_t)(at[pos] - atm0) * 3 + x) * nn;
                                                f1_add_block(WJ, nao, i0s, j0s, Jij + base, di, dj);
                                                f1_add_block(WJ, nao, k0s, l0s, Jkl + base, dk, dl);
                                        }
                                        if (with_k) {
                                                double *WK = wk + ((size_t)(at[pos] - atm0) * 3 + x) * nn;
                                                f1_add_block(WK, nao, i0s, l0s, Kil + base, di, dl);
                                                f1_add_block(WK, nao, j0s, l0s, Kjl + base, dj, dl);
                                                f1_add_block(WK, nao, i0s, k0s, Kik + base, di, dk);
                                                f1_add_block(WK, nao, j0s, k0s, Kjk + base, dj, dk);
                                        }
                                }
                        }
                }
        }
#pragma omp barrier
        if (with_j) f1_reduce(vj, jbufs, nth_used, nv);
        if (with_k) f1_reduce(vk, kbufs, nth_used, nv);
#pragma omp barrier
        free(wj);
        free(wk);
        free(G);
        free(lb);
        free(gwk);
        free(loc);
}
        free(jbufs);
        free(kbufs);
        SOCVHFgen_ctx_free(cx);
        size_t a, p, q;
#pragma omp parallel for private(p, q)
        for (a = 0; a < (size_t)nbat * 3; a++) {
                double *v;
                if (with_j) {
                        v = vj + a * nn;
                        for (p = 0; p < nao; p++) {
                                for (q = 0; q < p; q++) {
                                        const double s = v[p * nao + q] + v[q * nao + p];
                                        v[p * nao + q] = v[q * nao + p] = s;
                                }
                                v[p * nao + p] *= 2;
                        }
                }
                if (with_k) {
                        v = vk + a * nn;
                        for (p = 0; p < nao; p++) {
                                for (q = 0; q < p; q++) {
                                        const double s = v[p * nao + q] + v[q * nao + p];
                                        v[p * nao + q] = v[q * nao + p] = s;
                                }
                                v[p * nao + p] *= 2;
                        }
                }
        }
}
