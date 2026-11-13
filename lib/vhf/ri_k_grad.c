/*
 * Three-centre derivative contraction of a density-fitted energy with a
 * non-separable three-index weight (the RI-K gradient, socutils.grad.rik_grad):
 *
 *     g[C', x] = sum_{P in block} sum_ij Gamma^P_ij d/dR_{C',x} (ij|P)
 *
 * for a symmetric Gamma^P (Gamma^P_ij = Gamma^P_ji), over one block of
 * auxiliary shells [ksh0, ksh1).  As in ri_j_deriv.c only the bra derivative
 * of the i function is evaluated (int3c2e_ip1 = (nabla i j|P) = -d/dR_A of
 * the i function on atom A); summed over the ordered pairs the j derivatives
 * equal the i ones (Gamma symmetric), and the auxiliary centre C follows from
 * translational invariance:
 *
 *     g[A] -= 2 s,   g[natm + C] += 2 s,   s = sum Gamma^P_ij (nabla i j|P).
 *
 * gout is (natm + nauxatm, 3), the auxiliary atoms (pyscf gto.conc_env of mol
 * and auxmol) after the AO ones; per-thread buffers reduced at the end.
 *
 * gamma: Gamma^p_ij at gamma[p sp + i si + j] for the auxiliary functions p of
 * the block (p from 0 at ao_loc[ksh0]; (np, nao, nao): sp = nao^2, si = nao;
 * (nao, np, nao): sp = nao, si = np nao); gcond: (ksh1 - ksh0, nbas, nbas) the
 * largest |Gamma^P_ij| over the functions of each (P shell, i shell, j shell).
 * A triplet is skipped when qij[i,j] qP[k] gcond[k,i,j] < tol (Schwarz:
 * |(ab|P)| <= sqrt((ab|ab)) sqrt((P|P)), qij the shell-pair maximum over the
 * derivative types).  Spherical AOs.
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cint.h"

int int3c2e_ip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                    int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);

void SOCVHFri_k_grad(double *gout, double *gamma, size_t sp, size_t si, double *gcond, int ksh0, int ksh1,
                     double *qij, double *qP, double tol, CINTOpt *opt_ip1,
                     int *ao_loc, int *atm, int natm, int nauxatm, int *bas, int nbas,
                     int nauxbas, double *env)
{
        const int nbtot = nbas + nauxbas;
        const int natm_tot = natm + nauxatm;
        const size_t nao = ao_loc[nbas];
        const int nk = ksh1 - ksh0;
        const int pbase = ao_loc[ksh0];
        const size_t ng = (size_t)natm_tot * 3;
        int dmax = 0, i;
        for (i = 0; i < nbtot; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                if (d > dmax) dmax = d;
        }
        double qPmax = 0;
        for (i = ksh0; i < ksh1; i++)
                if (qP[i - nbas] > qPmax) qPmax = qP[i - nbas];
        int shls_slice[6] = {0, nbas, 0, nbas, ksh0, ksh1};
        size_t cache_size = GTOmax_cache_size(int3c2e_ip1_sph, shls_slice, 3,
                                              atm, natm_tot, bas, nbtot, env);
        const int nthreads = omp_get_max_threads();
        double *gbufs = calloc(ng * nthreads, sizeof(double));
        const size_t nwork = (size_t)nbas * nbas;

#pragma omp parallel
{
        const int t = omp_get_thread_num();
        double *gloc = gbufs + ng * t;
        double *buf = malloc(sizeof(double) * (size_t)dmax * dmax * dmax * 3);
        double *cache = malloc(sizeof(double) * cache_size);
        size_t ij;
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < nwork; ij++) {
                const int ish = ij / nbas, jsh = ij % nbas;
                const double q2 = qij[ish * nbas + jsh];
                if (q2 * qPmax == 0) continue;
                const int i0 = ao_loc[ish], di = ao_loc[ish + 1] - i0;
                const int j0 = ao_loc[jsh], dj = ao_loc[jsh + 1] - j0;
                const int A = bas[ish * BAS_SLOTS + ATOM_OF];
                const int dij = di * dj;
                double sA[3] = {0, 0, 0};
                int kq;
                for (kq = 0; kq < nk; kq++) {
                        const int ksh = ksh0 + kq;
                        if (q2 * qP[ksh - nbas] * gcond[((size_t)kq * nbas + ish) * nbas + jsh] < tol)
                                continue;
                        int shls[3] = {ish, jsh, ksh};
                        if (!int3c2e_ip1_sph(buf, NULL, shls, atm, natm_tot, bas, nbtot, env,
                                             opt_ip1, cache))
                                continue;
                        const int p0 = ao_loc[ksh] - pbase, dp = ao_loc[ksh + 1] - ao_loc[ksh];
                        const int C = bas[ksh * BAS_SLOTS + ATOM_OF];
                        const size_t nf = (size_t)dij * dp;
                        int x, a, b, k;
                        for (x = 0; x < 3; x++) {
                                const double *g = buf + x * nf;
                                double s = 0;
                                for (k = 0; k < dp; k++) {
                                        const double *gk = g + (size_t)k * dij;
                                        const double *gm = gamma + (size_t)(p0 + k) * sp + (size_t)i0 * si + j0;
                                        for (a = 0; a < di; a++) {
                                                const double *gr = gm + (size_t)a * si;
                                                for (b = 0; b < dj; b++)
                                                        s += gk[b * di + a] * gr[b];
                                        }
                                }
                                sA[x] += s;
                                gloc[C * 3 + x] += 2 * s;
                        }
                }
                gloc[A * 3 + 0] -= 2 * sA[0];
                gloc[A * 3 + 1] -= 2 * sA[1];
                gloc[A * 3 + 2] -= 2 * sA[2];
        }
        free(buf);
        free(cache);
}
        size_t e;
        int k;
        for (e = 0; e < ng; e++) {
                double s = 0;
                for (k = 0; k < nthreads; k++) s += gbufs[ng * k + e];
                gout[e] += s;
        }
        free(gbufs);
}

/* gcond[kq, ish, jsh] = max |gamma[p, i, j]| over the functions of P shell
 * ksh0 + kq, i shell ish, j shell jsh */
void SOCVHFri_k_gcond(double *gcond, double *gamma, size_t sp, size_t si, int ksh0, int ksh1,
                      int *ao_loc, int nbas)
{
        const int pbase = ao_loc[ksh0];
        int kq;
#pragma omp parallel for schedule(dynamic)
        for (kq = 0; kq < ksh1 - ksh0; kq++) {
                const int ksh = ksh0 + kq;
                const int p0 = ao_loc[ksh] - pbase, p1 = ao_loc[ksh + 1] - pbase;
                double *gc = gcond + (size_t)kq * nbas * nbas;
                int ish, jsh, p, a, b;
                for (ish = 0; ish < nbas; ish++)
                for (jsh = 0; jsh < nbas; jsh++) {
                        double m = 0;
                        for (p = p0; p < p1; p++)
                        for (a = ao_loc[ish]; a < ao_loc[ish + 1]; a++) {
                                const double *r = gamma + p * sp + a * si;
                                for (b = ao_loc[jsh]; b < ao_loc[jsh + 1]; b++) {
                                        double v = r[b] < 0 ? -r[b] : r[b];
                                        if (v > m) m = v;
                                }
                        }
                        gc[ish * nbas + jsh] = m;
                }
        }
}
