/*
 * Shell quartets of l <= JKV_LMAX for the J / K builds of jk_multi.c by the
 * Head-Gordon-Pople scheme with generated primitive kernels (jk_vrr_gen.c,
 * from gen_jk_vrr.py):
 *
 *   per primitive quartet (generated, straight-line): F_m(T), the vertical
 *   recursions to [e0|f0]^(m=0) for deg e in la .. la + lb, deg f in lc ..
 *   lc + ld, the bra then the ket contraction;
 *   per contracted quartet (here): the horizontal recursions (hrr, as hgp.c),
 *   the Cartesian-to-spherical transform of the d shells (libcint's
 *   coefficients), the output in int2e_sph's layout.
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

#define GEN_KCUT 1e-24
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

#include "jk_vrr_gen.c"

typedef struct GenCtx {
        int nbas;
        GPair *pair;                    /* per shell pair i >= j, l <= JKV_LMAX (n = 0 otherwise) */
        double *tab;
} GenCtx;

/* ---- Cartesian tables and the horizontal recursion (as hgp.c) ---- */
#define EMAX (2 * JKV_LMAX + 1)
#define N3(d) (((d) + 1) * ((d) + 2) * ((d) + 3) / 6)
#define N2(d) (((d) + 1) * ((d) + 2) / 2)
#define NE3 N3(EMAX)
static int c_ready = 0;
static int c_pow[NE3 + N2(EMAX + 1)][3];
static int c_dir[NE3], c_dec1[NE3];
static int c_inc[NE3][3];
static int c_idx[EMAX + 2][EMAX + 2][EMAX + 2];

static void tables_init(void)
{
        int d, x, y, n = 0, k;
        for (d = 0; d <= EMAX + 1; d++)
                for (x = d; x >= 0; x--)
                        for (y = d - x; y >= 0; y--) {
                                const int z = d - x - y;
                                c_idx[x][y][z] = n;
                                c_pow[n][0] = x; c_pow[n][1] = y; c_pow[n][2] = z;
                                n++;
                        }
        for (n = 0; n < NE3; n++) {
                for (k = 0; k < 3; k++) {
                        int p[3] = {c_pow[n][0], c_pow[n][1], c_pow[n][2]};
                        p[k] += 1;
                        c_inc[n][k] = c_idx[p[0]][p[1]][p[2]];
                }
                if (n == 0) continue;
                k = c_pow[n][0] ? 0 : (c_pow[n][1] ? 1 : 2);
                c_dir[n] = k;
                int p[3] = {c_pow[n][0], c_pow[n][1], c_pow[n][2]};
                p[k] -= 1;
                c_dec1[n] = c_idx[p[0]][p[1]][p[2]];
        }
        c_ready = 1;
}

/* src[e * ncol + col], e of degree l1 .. l1 + l2 (global) -> dst[(a * nb + b) * ncol + col] */
static void hrr(double *dst, const double *src, int ncol, int l1, int l2, const double *R, double *wk)
{
        const int nb2 = N2(l2);
        int j, a, b, col;
        if (l2 == 0) {
                memcpy(dst, src + (size_t)(l1 > 0 ? N3(l1 - 1) : 0) * ncol, sizeof(double) * N2(l1) * ncol);
                return;
        }
        const int a0 = l1 > 0 ? N3(l1 - 1) : 0;
        const int na = N3(l1 + l2) - a0;
        double *prev = wk, *cur = wk + (size_t)na * nb2 * ncol;
        memcpy(prev, src + (size_t)a0 * ncol, sizeof(double) * na * ncol);
        int nbp = 1;
        for (j = 1; j <= l2; j++) {
                const int nbj = N2(j), amax = N3(l1 + l2 - j) - a0;
                const int b0 = N3(j - 1), bp0 = j > 1 ? N3(j - 2) : 0;
                for (a = 0; a < amax; a++) {
                        const int ag = a0 + a;
                        for (b = 0; b < nbj; b++) {
                                const int bg = b0 + b, k = c_dir[bg], b1 = c_dec1[bg] - bp0;
                                const int a1 = c_inc[ag][k] - a0;
                                const double *u = prev + ((size_t)a1 * nbp + b1) * ncol;
                                const double *v = prev + ((size_t)a * nbp + b1) * ncol;
                                double *o = cur + ((size_t)a * nbj + b) * ncol;
                                const double r = R[k];
                                for (col = 0; col < ncol; col++) o[col] = u[col] + r * v[col];
                        }
                }
                double *t = prev; prev = cur; cur = t;
                nbp = nbj;
        }
        memcpy(dst, prev, sizeof(double) * N2(l1) * nb2 * ncol);
}

/* (la lb | lc ld) Cartesian from S[e * NF + f] (global e, f): out[((c * nd + d) * na + a) * nb + b] */
static void stream_integrals(double *out, const double *S, int NF, int la, int lb, int lc, int ld,
                             const double *AB, const double *CD, double *wk)
{
        const int f0 = lc > 0 ? N3(lc - 1) : 0, f1 = N3(lc + ld);
        const int nf = f1 - f0, nab = N2(la) * N2(lb);
        double *bra = wk, *t = bra + (size_t)nab * NF, *w2 = t + (size_t)NF * nab;
        int e, f, ab;
        double *src = w2;
        const int e0 = la > 0 ? N3(la - 1) : 0, e1 = N3(la + lb);
        for (e = e0; e < e1; e++)
                for (f = 0; f < nf; f++) src[(size_t)e * nf + f] = S[(size_t)e * NF + f0 + f];
        hrr(bra, src, nf, la, lb, AB, w2 + (size_t)e1 * nf);
        for (ab = 0; ab < nab; ab++)
                for (f = 0; f < nf; f++) t[(size_t)(f0 + f) * nab + ab] = bra[(size_t)ab * nf + f];
        hrr(out, t, nab, lc, ld, CD, w2);
}

/* libcint's Cartesian-to-spherical coefficients of d (xx xy xz yy yz zz -> xy yz z2 xz x2-y2) */
static const double C2S_D[5][6] = {
        {0, 1.092548430592079070, 0, 0, 0, 0},
        {0, 0, 0, 0, 1.092548430592079070, 0},
        {-0.315391565252520002, 0, 0, -0.315391565252520002, 0, 0.630783130505040012},
        {0, 0, 1.092548430592079070, 0, 0, 0},
        {0.546274215296039535, 0, 0, -0.546274215296039535, 0, 0}};

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
        if (!c_ready) tables_init();
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
                P->p = malloc(sizeof(double) * nmax * (12 + nc));
                P->px = P->p + nmax; P->py = P->px + nmax; P->pz = P->py + nmax;
                P->kp = P->pz + nmax; P->bnd = P->kp + nmax;
                P->rix = P->bnd + nmax; P->riy = P->rix + nmax; P->riz = P->riy + nmax;
                P->rjx = P->riz + nmax; P->rjy = P->rjx + nmax; P->rjz = P->rjy + nmax;
                P->c = P->rjz + nmax;
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
                        bmax = fmax(bmax, e * ip * cm * pf);
                        for (b = 0; b < ncj; b++)
                                for (a = 0; a < nci; a++)
                                        P->c[(b * nci + a) * nmax + t] = ci[a * npi + pi] * cj[b * npj + pj];
                        t++;
                }
                P->n = t;
                P->bmax = bmax;
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

/* doubles of workspace for SOCVHFgen_int2e */
size_t SOCVHFgen_wksize(int *bas, int nbas)
{
        int i, np = 1;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + ANG_OF] <= JKV_LMAX && bas[i * BAS_SLOTS + NPRIM_OF] > np)
                        np = bas[i * BAS_SLOTS + NPRIM_OF];
        const size_t NT = (size_t)(N3(2 * JKV_LMAX) - N3(JKV_LMAX - 1)) * (N3(2 * JKV_LMAX) - N3(JKV_LMAX - 1));
        const size_t c4 = (size_t)G_MAXCTR * G_MAXCTR * G_MAXCTR * G_MAXCTR;
        const size_t NE = N3(2 * JKV_LMAX + 1), nab = (size_t)N2(JKV_LMAX + 1) * N2(JKV_LMAX);
        const size_t nhrr = 2 * nab * NE + NE * NE + 2 * NE * N2(JKV_LMAX + 1) * (NE > nab ? NE : nab);
        const size_t n6 = (size_t)N2(JKV_LMAX) * N2(JKV_LMAX) * N2(JKV_LMAX) * N2(JKV_LMAX);
        return NT * np * np                     /* v */
             + G_MAXCTR * G_MAXCTR * NT         /* w */
             + c4 * NT                          /* SK */
             + NE * NE                          /* S */
             + 2 * n6                           /* Cartesian, spherical integrals */
             + nhrr;
}

/* one axis of a (n_out, d_axis, n_in) block from Cartesian d (6) to spherical (5) */
static void c2s_axis(double *dst, const double *src, int nout, int nin)
{
        int o, m, c, i;
        for (o = 0; o < nout; o++)
                for (m = 0; m < 5; m++) {
                        double *d = dst + ((size_t)o * 5 + m) * nin;
                        for (i = 0; i < nin; i++) d[i] = 0;
                        for (c = 0; c < 6; c++) {
                                const double f = C2S_D[m][c];
                                if (f == 0) continue;
                                const double *s = src + ((size_t)o * 6 + c) * nin;
                                for (i = 0; i < nin; i++) d[i] += f * s[i];
                        }
                }
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
        const int ncb = B->nc, nck = K->nc;
        const double fac = TWO_PI52 * common_fac(la) * common_fac(lb) * common_fac(lc) * common_fac(ld);
        const double cut = prim_tol / (fac * B->bmax * dmax * ncb * nck);
        const int nt = jkv_nt[la][lb][lc][ld];
        const size_t NT = (size_t)(N3(2 * JKV_LMAX) - N3(JKV_LMAX - 1)) * (N3(2 * JKV_LMAX) - N3(JKV_LMAX - 1));
        double *SK = wk + NT * B->n + (size_t)G_MAXCTR * G_MAXCTR * NT;
        double *S = SK + (size_t)G_MAXCTR * G_MAXCTR * G_MAXCTR * G_MAXCTR * NT;
        const size_t NEsq = (size_t)N3(2 * JKV_LMAX + 1) * N3(2 * JKV_LMAX + 1);
        const size_t n6 = (size_t)N2(JKV_LMAX) * N2(JKV_LMAX) * N2(JKV_LMAX) * N2(JKV_LMAX);
        double *Ic = S + NEsq, *Is = Ic + n6, *hw = Is + n6;
        if (!jkv_table[la][lb][lc][ld](SK, B, bs, K, ks, cut, cx->tab, wk)) return 0;
        double AB[3], CD[3];
        int x;
        for (x = 0; x < 3; x++) {
                AB[x] = bs ? -B->rij[x] : B->rij[x];
                CD[x] = ks ? -K->rij[x] : K->rij[x];
        }
        const int NF = N3(lc + ld), e0 = la > 0 ? N3(la - 1) : 0, f0 = lc > 0 ? N3(lc - 1) : 0;
        const int NFT = NF - f0, NET = N3(la + lb) - e0;
        const int na = N2(la), nb_ = N2(lb), nc_ = N2(lc), nd = N2(ld);
        /* spherical sizes and the shells' positions: role r (A, B, C, D) -> position */
        const int lr[4] = {la, lb, lc, ld};
        int nsph[4], r;
        for (r = 0; r < 4; r++) nsph[r] = 2 * lr[r] + 1;
        const int pos[4] = {bs ? 1 : 0, bs ? 0 : 1, ks ? 3 : 2, ks ? 2 : 3};
        int nfp[4], ncp[4];                     /* per position: spherical functions, contractions */
        ncp[0] = B->nci; ncp[1] = B->ncj; ncp[2] = K->nci; ncp[3] = K->ncj;
        for (r = 0; r < 4; r++) nfp[pos[r]] = nsph[r];
        const size_t DI = (size_t)ncp[0] * nfp[0], DJ = (size_t)ncp[1] * nfp[1], DK = (size_t)ncp[2] * nfp[2];
        int ci, cj, ck, cl, e, f;
        for (cl = 0; cl < K->ncj; cl++)
        for (ck = 0; ck < K->nci; ck++)
        for (cj = 0; cj < B->ncj; cj++)
        for (ci = 0; ci < B->nci; ci++) {
                const int cb = cj * B->nci + ci, kc = cl * K->nci + ck;
                const double *sk = SK + ((size_t)kc * ncb + cb) * nt;
                for (e = 0; e < NET; e++)
                        for (f = 0; f < NFT; f++) S[(size_t)(e0 + e) * NF + f0 + f] = fac * sk[e * NFT + f];
                stream_integrals(Ic, S, NF, la, lb, lc, ld, AB, CD, hw);
                /* Ic[((c * nd + d) * na + a) * nb + b] -> spherical along the d shells, in place order (C, D, A, B) */
                const double *cur = Ic;
                double *nxt = Is;
                int dims[4] = {nc_, nd, na, nb_};       /* axis order C, D, A, B */
                const int lax[4] = {lc, ld, la, lb};
                int ax;
                for (ax = 0; ax < 4; ax++) {
                        if (lax[ax] != 2) continue;
                        int nout = 1, nin = 1, q;
                        for (q = 0; q < ax; q++) nout *= dims[q];
                        for (q = ax + 1; q < 4; q++) nin *= dims[q];
                        c2s_axis(nxt, cur, nout, nin);
                        dims[ax] = 5;
                        const double *tmp = cur;
                        cur = nxt;
                        nxt = (double *)tmp;
                }
                /* place: role (A, B, C, D) component m -> position pos[r], contraction of that position */
                const int cpos[4] = {ci, cj, ck, cl};
                int mc, md, ma, mb;
                for (mc = 0; mc < dims[0]; mc++)
                for (md = 0; md < dims[1]; md++)
                for (ma = 0; ma < dims[2]; ma++)
                for (mb = 0; mb < dims[3]; mb++) {
                        int idx[4];
                        idx[pos[0]] = cpos[pos[0]] * nfp[pos[0]] + ma;
                        idx[pos[1]] = cpos[pos[1]] * nfp[pos[1]] + mb;
                        idx[pos[2]] = cpos[pos[2]] * nfp[pos[2]] + mc;
                        idx[pos[3]] = cpos[pos[3]] * nfp[pos[3]] + md;
                        out[(((size_t)idx[3] * DK + idx[2]) * DJ + idx[1]) * DI + idx[0]]
                                = cur[((mc * dims[1] + md) * dims[2] + ma) * dims[3] + mb];
                }
        }
        return 1;
}
