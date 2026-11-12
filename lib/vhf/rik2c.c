/*
 * Half transformation of a block of a real three-index factor B^P_{mu nu}
 * (packed lower triangle, pyscf pack_tril order) with two-component orbitals,
 * for socutils.scf.rik_2c: every orbital as four real AO vectors (Re / Im of
 * the alpha and beta components), the coefficients one column-major
 * (nao, 4 ncol) matrix c, columns (i, x), x = 0..3.  Per P and tile of
 * orbitals:
 *
 *     X^P_{(i, x), mu} = sum_nu c_{nu, (i, x)} B^P_{nu mu}        (one dgemm)
 *
 * and the output is not X itself but ncomb two-term combinations of its
 * components, written as ncomb row-major (nb ncol, nao) matrices
 *
 *     out[r][(P, i), mu] = X^P_{(i, ia[r]), mu} + sb[r] X^P_{(i, ib[r]), mu}
 *
 * (ib[r] < 0: the single component ia[r]).  The combinations are formed from
 * the dgemm tile while it is in cache; rik_2c's Kramers exchange needs eight
 * of them (four GEMMs instead of a 4 nao x 4 nao Gram matrix, half the flops).
 * Parallel over P when the block has at least as many P as threads (each
 * thread unpacks its B^P once), else over (P, tile).  Optionally the Coulomb
 * matrices of ndj densities from the same rows (vj += sum_P rho_P B^P), with
 * no extra pass over the block.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *, const int *,
            const double *, double *, const int *);

static void unpack_tril(double *out, const double *tril, int n)
{
        size_t i, j, ij = 0;
        for (i = 0; i < (size_t)n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        out[i * n + j] = tril[ij];
                        out[j * n + i] = tril[ij];
                }
        }
}

/* o[mu] = xa[mu] + s xb[mu] (xb NULL: a copy), streamed past the caches: every output row is
 * written once here and read back only by the GEMMs of the next phase */
static void comb_row(double *o, const double *xa, const double *xb, double s, int n)
{
        int mu = 0;
#ifdef __AVX512F__
        while (mu < n && ((uintptr_t)(o + mu) & 63)) {
                o[mu] = xb ? xa[mu] + s * xb[mu] : xa[mu];
                mu++;
        }
        if (xb) {
                const __m512d vs = _mm512_set1_pd(s);
                for (; mu + 8 <= n; mu += 8) {
                        _mm512_stream_pd(o + mu, _mm512_fmadd_pd(vs, _mm512_loadu_pd(xb + mu),
                                                                 _mm512_loadu_pd(xa + mu)));
                }
        } else {
                for (; mu + 8 <= n; mu += 8) {
                        _mm512_stream_pd(o + mu, _mm512_loadu_pd(xa + mu));
                }
        }
#endif
        for (; mu < n; mu++) {
                o[mu] = xb ? xa[mu] + s * xb[mu] : xa[mu];
        }
}

static void tile_comb(double **out, const double *x, int nao, int ncol, int p, int i0, int ni,
                      int ncomb, const int *ia, const int *ib, const double *sb)
{
        int r, i;
        /* orbital-major: the four component rows of an orbital stay in L1/L2 for all combinations */
        for (i = 0; i < ni; i++) {
                const double *xi = x + (size_t)i * 4 * nao;
                for (r = 0; r < ncomb; r++) {
                        double *o = out[r] + ((size_t)p * ncol + i0 + i) * nao;
                        comb_row(o, xi + (size_t)ia[r] * nao, ib[r] < 0 ? NULL : xi + (size_t)ib[r] * nao,
                                 sb[r], nao);
                }
        }
#ifdef __AVX512F__
        _mm_sfence();
#endif
}

/* J of ndj densities from the same rows: rho_d = sum_ij B^P_ij dtril[d, ij] (dtril: the packed
 * D + D^T with the diagonal once), vj[d] += rho_d B^P, while B^P is in cache */
static void j_row(double *vjt, const double *e, const double *dtril, int ndj, size_t npair)
{
        int d;
        size_t ij;
        for (d = 0; d < ndj; d++) {
                const double *dt = dtril + d * npair;
                double *v = vjt + d * npair;
                double rho = 0;
#pragma omp simd reduction(+:rho)
                for (ij = 0; ij < npair; ij++) {
                        rho += e[ij] * dt[ij];
                }
#pragma omp simd
                for (ij = 0; ij < npair; ij++) {
                        v[ij] += rho * e[ij];
                }
        }
}

void SOCVHFrik2c_half(double *out, const double *eri, const double *c, int nb, int nao, int ncol,
                      int ncomb, const int *ia, const int *ib, const double *sb, int tile,
                      const double *dtril, int ndj, double *vj)
{
        const size_t npair = (size_t)nao * (nao + 1) / 2;
        const size_t nrow = (size_t)nb * ncol * nao;
        double *outs[16];
        int r;
        if (ncomb > 16 || ncol == 0 || nb == 0) {
                return;
        }
        for (r = 0; r < ncomb; r++) {
                outs[r] = out + r * nrow;
        }
        if (tile <= 0 || tile > ncol) {
                tile = ncol;
        }
        const int ntile = (ncol + tile - 1) / tile;
        const int by_p = nb >= omp_get_max_threads();
        const double D1 = 1., D0 = 0.;
        const int nthreads = omp_get_max_threads();
        double **vjts = calloc(nthreads, sizeof(double *));
#pragma omp parallel
{
        double *b = malloc(sizeof(double) * nao * nao);
        double *x = malloc(sizeof(double) * nao * 4 * tile);
        double *vjt = NULL;
        int last = -1;
        int k;
        if (ndj > 0) {
                vjt = calloc((size_t)ndj * npair, sizeof(double));
                vjts[omp_get_thread_num()] = vjt;
        }
        if (by_p) {
                int p;
#pragma omp for schedule(dynamic, 1)
                for (p = 0; p < nb; p++) {
                        unpack_tril(b, eri + p * npair, nao);
                        if (ndj > 0) {
                                j_row(vjt, eri + p * npair, dtril, ndj, npair);
                        }
                        int t;
                        for (t = 0; t < ntile; t++) {
                                const int i0 = t * tile;
                                const int ni = (ncol - i0 < tile) ? ncol - i0 : tile;
                                const int m4 = ni * 4;
                                /* x (col-major nao x 4ni) = B c_tile: row-major (4ni, nao) X^P */
                                dgemm_("N", "N", &nao, &m4, &nao, &D1, b, &nao,
                                       c + (size_t)i0 * 4 * nao, &nao, &D0, x, &nao);
                                tile_comb(outs, x, nao, ncol, p, i0, ni, ncomb, ia, ib, sb);
                        }
                }
        } else {
#pragma omp for schedule(dynamic, 1)
                for (k = 0; k < nb * ntile; k++) {
                        const int p = k / ntile, t = k % ntile;
                        if (p != last) {
                                unpack_tril(b, eri + p * npair, nao);
                                last = p;
                        }
                        if (ndj > 0 && t == 0) {
                                j_row(vjt, eri + p * npair, dtril, ndj, npair);
                        }
                        const int i0 = t * tile;
                        const int ni = (ncol - i0 < tile) ? ncol - i0 : tile;
                        const int m4 = ni * 4;
                        dgemm_("N", "N", &nao, &m4, &nao, &D1, b, &nao,
                               c + (size_t)i0 * 4 * nao, &nao, &D0, x, &nao);
                        tile_comb(outs, x, nao, ncol, p, i0, ni, ncomb, ia, ib, sb);
                }
        }
        free(x);
        free(b);
        if (ndj > 0) {
#pragma omp barrier
                size_t ij, n = (size_t)ndj * npair;
                int it;
#pragma omp for schedule(static)
                for (ij = 0; ij < n; ij++) {
                        double v = 0;
                        for (it = 0; it < nthreads; it++) {
                                if (vjts[it] != NULL) {
                                        v += vjts[it][ij];
                                }
                        }
                        vj[ij] += v;
                }
                free(vjt);
        }
}
        free(vjts);
}
