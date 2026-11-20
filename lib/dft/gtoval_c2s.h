/*
 * libcint's cartesian -> spherical coefficients for any l <= C2S_LMAX, computed once at run time from the
 * closed form of the real solid harmonics (Helgaker, Jorgensen, Olsen eq. 6.4.47) normalised as libcint
 * (pyscf.gto.cart2sph(l): Y_lm on the unit sphere for l >= 2; s and p carry their factor separately,
 * common_fac_sp).  Rows m = -l .. l (pyscf's order), columns the cartesians x^i y^j z^k in pyscf's order
 * (i descending, then j descending).  Dense rows (c2s_row) and their nonzero entries (c2s_nz / c2s_ix /
 * c2s_cf).  Static per translation unit; c2s_init() before use (idempotent, thread-safe).
 */
#ifndef SOCDFT_GTOVAL_C2S_H
#define SOCDFT_GTOVAL_C2S_H
#include <math.h>
#include <string.h>

#define C2S_LMAX 7
#define C2S_NCART(l) (((l) + 1) * ((l) + 2) / 2)
#define C2S_NCMAX C2S_NCART(C2S_LMAX)

static double c2s_tab[C2S_LMAX + 1][2 * C2S_LMAX + 1][C2S_NCMAX];
static int c2s_nz[C2S_LMAX + 1][2 * C2S_LMAX + 1];
static int c2s_ix[C2S_LMAX + 1][2 * C2S_LMAX + 1][C2S_NCMAX];
static double c2s_cf[C2S_LMAX + 1][2 * C2S_LMAX + 1][C2S_NCMAX];
static int c2s_ready = 0;

static double c2s_fact(int n)
{
        double f = 1;
        while (n > 1) f *= n--;
        return f;
}

static double c2s_binom(int n, int k)
{
        if (k < 0 || k > n) return 0;
        return c2s_fact(n) / (c2s_fact(k) * c2s_fact(n - k));
}

static int c2s_cart_index(int l, int x, int y)
{
        /* position of x^x y^y z^(l-x-y) in pyscf's order */
        const int a = l - x;            /* cartesians with a larger x power come first: a (a + 1) / 2 of them */
        return a * (a + 1) / 2 + (a - y);
}

static void c2s_init(void)
{
        int ready;
#pragma omp atomic read
        ready = c2s_ready;
        if (ready) return;
#pragma omp critical(socdft_c2s_init)
{
        if (!c2s_ready) {
                int l, k, i;
                memset(c2s_tab, 0, sizeof(c2s_tab));
                c2s_tab[0][0][0] = 1;
                for (k = 0; k < 3; k++) c2s_tab[1][k][k] = 1;
                for (l = 2; l <= C2S_LMAX; l++) {
                        for (k = 0; k < 2 * l + 1; k++) {
                                const int m = k - l, am = m < 0 ? -m : m, wm = m < 0;
                                double n = 1. / (pow(2., am) * c2s_fact(l))
                                         * sqrt(2 * c2s_fact(l + am) * c2s_fact(l - am) / (m == 0 ? 2. : 1.));
                                n *= sqrt((2 * l + 1) / (4 * M_PI));
                                int t, u, w;
                                for (t = 0; t <= (l - am) / 2; t++)
                                for (u = 0; u <= t; u++)
                                for (w = wm; w <= am; w += 2) {         /* w = 2 v */
                                        const int sg = (t + (w - wm) / 2) % 2 ? -1 : 1;
                                        const double c = sg * pow(.25, t) * c2s_binom(l, t) * c2s_binom(l - t, am + t)
                                                       * c2s_binom(t, u) * c2s_binom(am, w);
                                        const int ex = 2 * t + am - 2 * u - w, ey = 2 * u + w;
                                        c2s_tab[l][k][c2s_cart_index(l, ex, ey)] += n * c;
                                }
                        }
                }
                for (l = 0; l <= C2S_LMAX; l++)
                        for (k = 0; k < 2 * l + 1; k++) {
                                int nz = 0;
                                for (i = 0; i < C2S_NCART(l); i++) {
                                        if (fabs(c2s_tab[l][k][i]) < 1e-15) {
                                                c2s_tab[l][k][i] = 0;
                                                continue;
                                        }
                                        c2s_ix[l][k][nz] = i;
                                        c2s_cf[l][k][nz] = c2s_tab[l][k][i];
                                        nz++;
                                }
                                c2s_nz[l][k] = nz;
                        }
#pragma omp atomic write
                c2s_ready = 1;
        }
}
}

/* row m (0 .. 2l) of the dense table, ncart its length */
static inline const double *c2s_row(int l, int m, int *ncart)
{
        *ncart = C2S_NCART(l);
        return c2s_tab[l][m];
}
#endif
