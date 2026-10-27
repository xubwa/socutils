/*
 * Three-centre derivative contractions of the density-fitted (RI-J) Coulomb
 * energy E_J = 1/2 rho^T M^{-1} rho, rho_P = sum_ij (ij|P) D_ij, d = M^{-1} rho,
 * for its nuclear Hessian and first-order Coulomb matrices
 * (socutils.hessian.rhf_fast):
 *
 *     Y[C,x,P]    = d/dR_{C,x} rho_P                            (rho^xi)
 *     W[C,x,i,j]  : V^{C,x} = W + W^T, V^{C,x}_ij = sum_P d/dR_{C,x} (ij|P) d_P
 *     hess[A,B,x,y] = sum_ijP d^2/dR_{A,x} dR_{B,y} (ij|P) D_ij d_P
 *
 * in one pass over the shell triplets (ish, jsh, P), ish and jsh both over all
 * AO shells.  Only bra derivatives are evaluated (int3c2e_ip1, _ipip1,
 * _ipvip1, all on the i function), and the auxiliary centre follows from
 * translational invariance, d/dR_C = -(d/dR_A + d/dR_B).  With D symmetric,
 * summed over the ordered pairs the j derivatives equal the i derivatives, so
 * Y and hess take the i terms twice.  Per triplet (i on A, j on B, P on C),
 * with aa = (i''j|P) D d and ab = (i'j'|P) D d summed over the functions and
 * X = aa + ab:
 *
 *     hess[A,A] += 2 aa,  hess[A,B] += 2 ab,  hess[C,C] += 2 X,
 *     hess[A,C] -= 2 X,   hess[C,A] -= 2 X^T.
 *
 * W collects only the i derivative of element (i, j) (V = W + W^T adds the j
 * one), so the (ish, jsh) blocks of W are written by one thread each; Y and
 * hess are per thread and reduced.
 *
 * Screening: |(ab|P)| <= sqrt((ab|ab)) sqrt((P|P)) for any bra functions a, b
 * (Schwarz), with qij the shell-pair maximum over the derivative types and qP
 * sqrt((P|P)) per auxiliary shell; a triplet is skipped when the bound times
 * its largest weight (|D| |d| for hess, |D| for Y, |d| for W) is below tol.
 *
 * Combined basis (pyscf gto.conc_env(mol, auxmol)): AO shells [0, nbas), then
 * the auxiliary shells; the atoms of the auxiliary shells are those of the AO
 * molecule shifted by natm.  Spherical AOs.  Any of hess, Y, W may be NULL.
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cint.h"

int int3c2e_ip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                    int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
int int3c2e_ipip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                      int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
int int3c2e_ipvip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                       int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);

static inline double dmax2(double a, double b) { return a > b ? a : b; }

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

void SOCVHFri_j_deriv(double *hess, double *Y, double *W, double *dm, double *dfit,
                      double *qij, double *qP, double *dcond, double *dPcond, double tol,
                      CINTOpt *opt_ip1, CINTOpt *opt_ipip1, CINTOpt *opt_ipvip1,
                      int *ao_loc, int *atm, int natm, int *bas, int nbas, int nauxbas,
                      double *env)
{
        const int nbtot = nbas + nauxbas;
        const int natm_tot = natm * 2;
        const size_t nao = ao_loc[nbas];
        const size_t naux = ao_loc[nbtot] - nao;
        const size_t nn = nao * nao;
        const size_t nh = (size_t)natm * natm * 9;
        const size_t ny = (size_t)natm * 3 * naux;
        const int want_d1 = Y != NULL || W != NULL;
        int dmax = 0, i;
        for (i = 0; i < nbtot; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                if (d > dmax) dmax = d;
        }
        double qPmax = 0, dPmax = 0, dijmax = 0;
        for (i = 0; i < nauxbas; i++) {
                qPmax = dmax2(qPmax, qP[i]);
                dPmax = dmax2(dPmax, dPcond[i]);
        }
        for (i = 0; i < nbas * nbas; i++) dijmax = dmax2(dijmax, dcond[i]);
        int shls_slice[6] = {0, nbas, 0, nbas, nbas, nbtot};
        size_t cache_size = GTOmax_cache_size(int3c2e_ipvip1_sph, shls_slice, 3,
                                              atm, natm_tot, bas, nbtot, env);
        size_t c2 = GTOmax_cache_size(int3c2e_ipip1_sph, shls_slice, 3,
                                      atm, natm_tot, bas, nbtot, env);
        if (c2 > cache_size) cache_size = c2;
        const int nthreads = omp_get_max_threads();
        double **hbufs = malloc(sizeof(double *) * nthreads);
        double **ybufs = malloc(sizeof(double *) * nthreads);
        if (W != NULL) memset(W, 0, sizeof(double) * natm * 3 * nn);

#pragma omp parallel
{
        const int t = omp_get_thread_num();
        double *hloc = hess != NULL ? calloc(nh, sizeof(double)) : NULL;
        double *yloc = Y != NULL ? calloc(ny, sizeof(double)) : NULL;
        hbufs[t] = hloc;
        ybufs[t] = yloc;
        const size_t nf3 = (size_t)dmax * dmax * dmax;
        double *buf = malloc(sizeof(double) * nf3 * 9);
        double *cache = malloc(sizeof(double) * cache_size);
        int ij;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas * nbas; ij++) {
                const int ish = ij / nbas, jsh = ij % nbas;
                const double q2 = qij[ish * nbas + jsh];
                const double dij = dcond[ish * nbas + jsh];
                /* largest weight of this pair over all auxiliary shells */
                double wmax = 0;
                if (hess != NULL) wmax = dmax2(wmax, dij * dPmax);
                if (Y != NULL) wmax = dmax2(wmax, dij);
                if (W != NULL) wmax = dmax2(wmax, dPmax);
                if (q2 * qPmax * wmax < tol) continue;
                const int i0 = ao_loc[ish], di = ao_loc[ish + 1] - i0;
                const int j0 = ao_loc[jsh], dj = ao_loc[jsh + 1] - j0;
                const int A = bas[ish * BAS_SLOTS + ATOM_OF];
                const int B = bas[jsh * BAS_SLOTS + ATOM_OF];
                const int dij_n = di * dj;
                int ksh;
                for (ksh = nbas; ksh < nbtot; ksh++) {
                        const int kq = ksh - nbas;
                        const double q3 = q2 * qP[kq];
                        const double dk = dPcond[kq];
                        const int do_h = hess != NULL && q3 * dij * dk >= tol;
                        const int do_y = Y != NULL && q3 * dij >= tol;
                        const int do_w = W != NULL && q3 * dk >= tol;
                        if (!(do_h || do_y || do_w)) continue;
                        const int p0 = ao_loc[ksh] - nao, dp = ao_loc[ksh + 1] - ao_loc[ksh];
                        const int C = bas[ksh * BAS_SLOTS + ATOM_OF] - natm;
                        const size_t nf = (size_t)dij_n * dp;
                        int shls[3] = {ish, jsh, ksh};
                        int x, y, a, b, k;
                        if ((do_y || do_w) && want_d1 &&
                            int3c2e_ip1_sph(buf, NULL, shls, atm, natm_tot, bas, nbtot, env,
                                            opt_ip1, cache)) {
                                for (x = 0; x < 3; x++) {
                                        const double *g = buf + x * nf;
                                        if (do_y) {
                                                double *ya = yloc + ((size_t)A * 3 + x) * naux + p0;
                                                double *yc = yloc + ((size_t)C * 3 + x) * naux + p0;
                                                for (k = 0; k < dp; k++) {
                                                        const double *gk = g + (size_t)k * dij_n;
                                                        double s = 0;
                                                        for (b = 0; b < dj; b++) {
                                                                const double *dr = dm + j0 + b;
                                                                for (a = 0; a < di; a++)
                                                                        s += gk[b * di + a] * dr[(i0 + a) * nao];
                                                        }
                                                        /* d/dR_A of the i function: -ip1 */
                                                        ya[k] -= 2 * s;      /* x2: the j derivatives */
                                                        yc[k] += 2 * s;
                                                }
                                        }
                                        if (do_w) {
                                                double *wa = W + ((size_t)A * 3 + x) * nn;
                                                double *wc = W + ((size_t)C * 3 + x) * nn;
                                                for (b = 0; b < dj; b++) {
                                                        for (a = 0; a < di; a++) {
                                                                double s = 0;
                                                                for (k = 0; k < dp; k++)
                                                                        s += g[(size_t)k * dij_n + b * di + a] * dfit[p0 + k];
                                                                const size_t e = (i0 + a) * nao + j0 + b;
                                                                wa[e] -= s;
                                                                wc[e] += s;
                                                        }
                                                }
                                        }
                                }
                        }
                        if (!do_h) continue;
                        double aa[9], ab[9];
                        int which;
                        for (which = 0; which < 2; which++) {
                                double *o = which == 0 ? aa : ab;
                                int nz = which == 0
                                        ? int3c2e_ipip1_sph(buf, NULL, shls, atm, natm_tot, bas, nbtot, env,
                                                            opt_ipip1, cache)
                                        : int3c2e_ipvip1_sph(buf, NULL, shls, atm, natm_tot, bas, nbtot, env,
                                                             opt_ipvip1, cache);
                                for (x = 0; x < 9; x++) {
                                        double s = 0;
                                        if (nz) {
                                                const double *g = buf + x * nf;
                                                for (k = 0; k < dp; k++) {
                                                        const double *gk = g + (size_t)k * dij_n;
                                                        double sk = 0;
                                                        for (b = 0; b < dj; b++) {
                                                                const double *dr = dm + j0 + b;
                                                                for (a = 0; a < di; a++)
                                                                        sk += gk[b * di + a] * dr[(i0 + a) * nao];
                                                        }
                                                        s += sk * dfit[p0 + k];
                                                }
                                        }
                                        o[x] = 2 * s;     /* x2: the j derivatives */
                                }
                        }
                        double *hAA = hloc + ((size_t)A * natm + A) * 9;
                        double *hAB = hloc + ((size_t)A * natm + B) * 9;
                        double *hCC = hloc + ((size_t)C * natm + C) * 9;
                        double *hAC = hloc + ((size_t)A * natm + C) * 9;
                        double *hCA = hloc + ((size_t)C * natm + A) * 9;
                        for (x = 0; x < 3; x++) {
                                for (y = 0; y < 3; y++) {
                                        const double X = aa[x * 3 + y] + ab[x * 3 + y];
                                        hAA[x * 3 + y] += aa[x * 3 + y];
                                        hAB[x * 3 + y] += ab[x * 3 + y];
                                        hCC[x * 3 + y] += X;
                                        hAC[x * 3 + y] -= X;
                                        hCA[y * 3 + x] -= X;
                                }
                        }
                }
        }
        free(buf);
        free(cache);
#pragma omp barrier
        if (hess != NULL) reduce_thread_buffers(hess, hbufs, nthreads, nh);
        if (Y != NULL) reduce_thread_buffers(Y, ybufs, nthreads, ny);
#pragma omp barrier
        free(hloc);
        free(yloc);
}
        free(hbufs);
        free(ybufs);
}
