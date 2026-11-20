/*
 * ONX-style significant-partner lists for the exchange loops (Schwegler,
 * Challacombe, Head-Gordon, J. Chem. Phys. 106, 9708 (1997)): instead of
 * testing every ket pair kl <= ij of a bra pair (i j), the ket pairs whose
 * exchange bound
 *
 *     q_ij q_kl max(D_ik, D_il, D_jk, D_jl) >= tol
 *
 * can hold are generated from two sorted lists,
 *
 *     the density partners a of i and of j (D_ia, D_ja descending), and
 *     the Schwarz partners b of a (q_ab descending),
 *
 * both walked only while the bound can still reach the threshold.  The bound
 * is symmetric in k <-> l, so every significant {k, l} is reached from the
 * member a in {k, l} carrying the largest density weight.  The candidate set
 * is a superset of the quartets the full loop keeps (the thresholds are
 * loosened by a few ulps); the caller applies its own test unchanged, so the
 * quartets evaluated and the results are those of the full loop.  The work
 * per bra pair follows the significant quartets instead of npair: in large
 * molecules and for the small density changes of incremental SCF builds the
 * O(npair^2) pair loop is gone.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

typedef struct ONXLists {
        int nbas;
        double qmax;
        int *qn, *qp;           /* per shell a: qn[a] partners b, q[a b] descending */
        double *qv;
        int *dn, *dp;           /* per shell s: dn[s] density partners a, dcond[s a] descending */
        double *dv;
} ONXLists;

typedef struct { double v; int i; } VI;

static int cmp_vi_desc(const void *x, const void *y)
{
        const double a = ((const VI *)x)->v, b = ((const VI *)y)->v;
        if (a > b) return -1;
        if (a < b) return 1;
        return ((const VI *)x)->i - ((const VI *)y)->i;
}

static void sorted_rows(const double *m, int nbas, int *cnt, int *idx, double *val)
{
        int a;
#pragma omp parallel
{
        VI *row = malloc(sizeof(VI) * (nbas > 0 ? nbas : 1));
#pragma omp for schedule(dynamic, 4)
        for (a = 0; a < nbas; a++) {
                int b, n = 0;
                for (b = 0; b < nbas; b++) {
                        const double v = m[(size_t)a * nbas + b];
                        if (v > 0) {
                                row[n].v = v;
                                row[n].i = b;
                                n++;
                        }
                }
                qsort(row, n, sizeof(VI), cmp_vi_desc);
                for (b = 0; b < n; b++) {
                        idx[(size_t)a * nbas + b] = row[b].i;
                        val[(size_t)a * nbas + b] = row[b].v;
                }
                cnt[a] = n;
        }
        free(row);
}
}

ONXLists *SOCVHFonx_build(const double *q, const double *dcond, int nbas)
{
        ONXLists *L = malloc(sizeof(ONXLists));
        const size_t n2 = (size_t)nbas * nbas;
        L->nbas = nbas;
        L->qn = malloc(sizeof(int) * (nbas + 1));
        L->qp = malloc(sizeof(int) * (n2 + 1));
        L->qv = malloc(sizeof(double) * (n2 + 1));
        L->dn = malloc(sizeof(int) * (nbas + 1));
        L->dp = malloc(sizeof(int) * (n2 + 1));
        L->dv = malloc(sizeof(double) * (n2 + 1));
        sorted_rows(q, nbas, L->qn, L->qp, L->qv);
        sorted_rows(dcond, nbas, L->dn, L->dp, L->dv);
        double qm = 0;
        int a;
        for (a = 0; a < nbas; a++)
                if (L->qn[a] > 0 && L->qv[(size_t)a * nbas] > qm) qm = L->qv[(size_t)a * nbas];
        L->qmax = qm;
        return L;
}

void SOCVHFonx_free(ONXLists *L)
{
        if (L == NULL) return;
        free(L->qn);
        free(L->qp);
        free(L->qv);
        free(L->dn);
        free(L->dp);
        free(L->dv);
        free(L);
}

/* per-thread scratch of SOCVHFonx_kpairs */
typedef struct ONXScratch {
        uint64_t *bits;         /* one bit per ket pair, (npair + 63) / 64 words, zero between calls */
        unsigned char *smark;   /* nbas */
        int *stmp;              /* nbas */
} ONXScratch;

ONXScratch *SOCVHFonx_scratch(int nbas)
{
        const size_t npair = (size_t)nbas * (nbas + 1) / 2;
        ONXScratch *s = malloc(sizeof(ONXScratch));
        s->bits = calloc(npair / 64 + 1, sizeof(uint64_t));
        s->smark = calloc(nbas + 1, 1);
        s->stmp = malloc(sizeof(int) * (nbas + 1));
        return s;
}

void SOCVHFonx_scratch_free(ONXScratch *s)
{
        if (s == NULL) return;
        free(s->bits);
        free(s->smark);
        free(s->stmp);
        free(s);
}

#define ONX_LOOSEN (1 - 1e-10)

/*
 * The ket pairs kl <= ij (k >= l, kl = k (k + 1) / 2 + l) of the bra pair
 * (ish jsh) with q_kl max(D_ik, D_il, D_jk, D_jl) >= thr (thr = tol / q_ij),
 * ascending (a bit set over the pairs, read back word by word), into out
 * (room for ij + 1); returns their number.
 */
int SOCVHFonx_kpairs(const ONXLists *L, ONXScratch *S, int ish, int jsh, int ij, double thr,
                     const double *dcond, int *out)
{
        const int nbas = L->nbas;
        const double t = thr * ONX_LOOSEN;
        uint64_t *bits = S->bits;
        unsigned char *smark = S->smark;
        int *stmp = S->stmp;
        const double *Di = dcond + (size_t)ish * nbas, *Dj = dcond + (size_t)jsh * nbas;
        int ns = 0, r, u, v, wmin = ij / 64 + 1, wmax = -1;
        for (r = 0; r < (ish == jsh ? 1 : 2); r++) {
                const int s = r == 0 ? ish : jsh;
                const int *P = L->dp + (size_t)s * nbas;
                const double *V = L->dv + (size_t)s * nbas;
                const int np = L->dn[s];
                for (u = 0; u < np; u++) {
                        if (V[u] * L->qmax < t) break;
                        const int a = P[u];
                        if (a > ish || smark[a]) continue;     /* k >= a > i: kl > ij */
                        smark[a] = 1;
                        stmp[ns++] = a;
                        const double d = Di[a] > Dj[a] ? Di[a] : Dj[a];
                        const int nq = L->qn[a];
                        if (nq == 0 || L->qv[(size_t)a * nbas] * d < t) continue;
                        const int *Q = L->qp + (size_t)a * nbas;
                        const double *W = L->qv + (size_t)a * nbas;
                        const int a2 = a * (a + 1) / 2;
                        for (v = 0; v < nq; v++) {
                                if (W[v] * d < t) break;
                                const int b = Q[v];
                                if (b > ish) continue;
                                const int p = a >= b ? a2 + b : b * (b + 1) / 2 + a;
                                if (p > ij) continue;
                                const int w = p >> 6;
                                bits[w] |= (uint64_t)1 << (p & 63);
                                if (w < wmin) wmin = w;
                                if (w > wmax) wmax = w;
                        }
                }
        }
        for (u = 0; u < ns; u++) smark[stmp[u]] = 0;
        int n = 0, w;
        for (w = wmin; w <= wmax; w++) {
                uint64_t x = bits[w];
                if (!x) continue;
                bits[w] = 0;
                while (x) {
                        out[n++] = (w << 6) + __builtin_ctzll(x);
                        x &= x - 1;
                }
        }
        return n;
}
