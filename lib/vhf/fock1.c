/*
 * First-order Coulomb / exchange matrices of all nuclear displacements at a
 * fixed symmetric density D (the 2e part of the CPHF right-hand sides,
 * pyscf.hessian.rhf.make_h1):
 *
 *     VJ^{A,x}_ab = sum_cd d/dR_{A,x} (ab|cd) D_cd,
 *     VK^{A,x}_ad = sum_bc d/dR_{A,x} (ab|cd) D_bc,
 *
 * in one pass over the unique shell quartets.  By translational invariance
 * three int2e_ip1 evaluations per quartet ((ij|kl), (ji|kl), (kl|ij)) give the
 * derivatives of all four centres.  With D symmetric the outputs are
 * symmetric, so the eight permutations collapse to two J and four K updates
 * per integral (W += ...; V = W + W^T at the end), accumulated in
 * quartet-local blocks and added to the matrices of the four atoms.
 * Outputs (natm, 3, nao, nao).  Spherical AOs.
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cint.h"

int int2e_ip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                  int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);

/* out[0:n] = sum_t bufs[t][0:n], each thread of the enclosing parallel region
 * summing its own slice */
static void reduce_thread_buffers(double *out, double **bufs, int nbuf, size_t n)
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

static inline double dmax2(double a, double b) { return a > b ? a : b; }

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((__builtin_sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

/* add a local block L[di x dj] into W[r0.., c0..] (row-major nao) */
static inline void add_block(double *W, size_t nao, size_t r0, size_t c0,
                             double *L, int dr, int dc)
{
        int r, c;
        for (r = 0; r < dr; r++) {
                double *w = W + (r0 + r) * nao + c0;
                double *l = L + r * dc;
                for (c = 0; c < dc; c++) w[c] += l[c];
        }
}

void SOCVHFfock1_jk(double *vj, double *vk, double *dm, int with_k,
                    double *qcond, double *dcond, double tol, CINTOpt *opt,
                    int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t nn = nao * nao;
        const size_t nv = (size_t)natm * 3 * nn;
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        size_t cache_size = GTOmax_cache_size(int2e_ip1_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)3 * dmax * dmax * dmax * dmax;
        const size_t nblk = (size_t)dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        const int nthmax = omp_get_max_threads();
        double **jbufs = calloc(nthmax, sizeof(double *));
        double **kbufs = calloc(nthmax, sizeof(double *));
        int nth_used = 1;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
#pragma omp single
        nth_used = omp_get_num_threads();
        double *wj = calloc(nv, sizeof(double));
        double *wk = with_k ? calloc(nv, sizeof(double)) : NULL;
        jbufs[tid] = wj;
        kbufs[tid] = wk;
        double *buf = malloc(sizeof(double) * (nbuf * 3 + cache_size));
        double *b0 = buf, *b1 = buf + nbuf, *b2 = buf + nbuf * 2;
        double *cache = buf + nbuf * 3;
        /* local blocks [4 positions][3][d x d] for J (ij, kl) and K (il, jl, ik, jk) */
        double *loc = malloc(sizeof(double) * 12 * nblk * 6);
        double *Jij = loc, *Jkl = loc + 12 * nblk;
        double *Kil = loc + 24 * nblk, *Kjl = loc + 36 * nblk;
        double *Kik = loc + 48 * nblk, *Kjk = loc + 60 * nblk;
        int ij, kl, ish, jsh, ksh, lsh, x, pos;
        int shls[4];
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        double d = dmax2(dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]),
                                         dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                               dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh])));
                        if (q4 * d < tol) {
                                continue;
                        }
                        int di = ao_loc[ish + 1] - ao_loc[ish];
                        int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                        int dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        int nf = di * dj * dk * dl;
                        int ok = 0;
#define CALL(b) do { \
                        if (int2e_ip1_sph(b, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) ok = 1; \
                        else memset(b, 0, sizeof(double) * 3 * nf); } while (0)
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        CALL(b0);
                        shls[0] = jsh; shls[1] = ish;
                        CALL(b1);
                        shls[0] = ksh; shls[1] = lsh; shls[2] = ish; shls[3] = jsh;
                        CALL(b2);
#undef CALL
                        if (!ok) {
                                continue;
                        }
                        double fac = 1.;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        memset(loc, 0, sizeof(double) * 12 * nblk * 6);
                        size_t i0s = ao_loc[ish], j0s = ao_loc[jsh];
                        size_t k0s = ao_loc[ksh], l0s = ao_loc[lsh];
                        int i0, j0, k0, l0;
                        for (l0 = 0; l0 < dl; l0++) {
                        for (k0 = 0; k0 < dk; k0++) {
                        for (j0 = 0; j0 < dj; j0++) {
                        for (i0 = 0; i0 < di; i0++) {
                                size_t o0 = ((l0 * dk + k0) * dj + j0) * di + i0;   /* (i j|k l) */
                                size_t o1 = ((l0 * dk + k0) * di + i0) * dj + j0;   /* (j i|k l) */
                                size_t o2 = ((j0 * di + i0) * dl + l0) * dk + k0;   /* (k l|i j) */
                                size_t mu = i0s + i0, nu = j0s + j0, la = k0s + k0, si = l0s + l0;
                                double Dls = dm[la * nao + si], Dmn = dm[mu * nao + nu];
                                double Dnl = dm[nu * nao + la], Dml = dm[mu * nao + la];
                                double Dns = dm[nu * nao + si], Dms = dm[mu * nao + si];
                                for (x = 0; x < 3; x++) {
                                        /* nuclear derivatives (-nabla) of the four positions */
                                        double g0 = -b0[x * nf + o0] * fac;
                                        double g1 = -b1[x * nf + o1] * fac;
                                        double g2 = -b2[x * nf + o2] * fac;
                                        double g[4] = {g0, g1, g2, -(g0 + g1 + g2)};
                                        for (pos = 0; pos < 4; pos++) {
                                                size_t base = (size_t)(pos * 3 + x) * nblk;
                                                double gg = g[pos];
                                                Jij[base + i0 * dj + j0] += 2 * gg * Dls;
                                                Jkl[base + k0 * dl + l0] += 2 * gg * Dmn;
                                                if (with_k) {
                                                        Kil[base + i0 * dl + l0] += gg * Dnl;
                                                        Kjl[base + j0 * dl + l0] += gg * Dml;
                                                        Kik[base + i0 * dk + k0] += gg * Dns;
                                                        Kjk[base + j0 * dk + k0] += gg * Dms;
                                                }
                                        }
                                }
                        } } } }
                        int at[4] = {bas[ATOM_OF + ish * BAS_SLOTS], bas[ATOM_OF + jsh * BAS_SLOTS],
                                     bas[ATOM_OF + ksh * BAS_SLOTS], bas[ATOM_OF + lsh * BAS_SLOTS]};
                        for (pos = 0; pos < 4; pos++) {
                        for (x = 0; x < 3; x++) {
                                size_t base = (size_t)(pos * 3 + x) * nblk;
                                double *WJ = wj + ((size_t)at[pos] * 3 + x) * nn;
                                add_block(WJ, nao, i0s, j0s, Jij + base, di, dj);
                                add_block(WJ, nao, k0s, l0s, Jkl + base, dk, dl);
                                if (with_k) {
                                        double *WK = wk + ((size_t)at[pos] * 3 + x) * nn;
                                        add_block(WK, nao, i0s, l0s, Kil + base, di, dl);
                                        add_block(WK, nao, j0s, l0s, Kjl + base, dj, dl);
                                        add_block(WK, nao, i0s, k0s, Kik + base, di, dk);
                                        add_block(WK, nao, j0s, k0s, Kjk + base, dj, dk);
                                }
                        } }
                }
        }
#pragma omp barrier
        reduce_thread_buffers(vj, jbufs, nth_used, nv);
        if (with_k) reduce_thread_buffers(vk, kbufs, nth_used, nv);
#pragma omp barrier
        free(wj);
        free(wk);
        free(buf);
        free(loc);
}
        free(jbufs);
        free(kbufs);
        /* V = W + W^T */
        size_t a, p, q;
#pragma omp parallel for private(p, q)
        for (a = 0; a < (size_t)natm * 3; a++) {
                double *v = vj + a * nn;
                for (p = 0; p < nao; p++) {
                        for (q = 0; q < p; q++) {
                                double s = v[p * nao + q] + v[q * nao + p];
                                v[p * nao + q] = v[q * nao + p] = s;
                        }
                        v[p * nao + p] *= 2;
                }
                if (with_k) {
                        v = vk + a * nn;
                        for (p = 0; p < nao; p++) {
                                for (q = 0; q < p; q++) {
                                        double s = v[p * nao + q] + v[q * nao + p];
                                        v[p * nao + q] = v[q * nao + p] = s;
                                }
                                v[p * nao + p] *= 2;
                        }
                }
        }
}
