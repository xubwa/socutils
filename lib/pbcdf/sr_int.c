/*
 * Short-range (erfc) Coulomb integrals of Gaussian functions summed over
 * explicitly enumerated lattice images, folded to the Gamma point:
 *
 *   B[P, mu, nu] = sum_{j image of nu, k image of P} (mu_0 j | k)_SR
 *   M[P, Q]      = sum_{k image of Q} (P_0 | k)_SR
 *
 * The caller builds one libcint environment holding, in this order,
 *   AO image shells   [0, nao_sh)       (the first nbas0 are cell 0)
 *   aux image shells  [aux_sh0, ...)    (the first naux_sh0 are cell 0)
 * with env[PTR_RANGE_OMEGA] = -omega (short-range attenuation), plus the fold
 * maps (image shell -> AO / aux offset in cell 0) and per-shell screening
 * parameters (most diffuse exponent and largest |normalised coefficient|).
 *
 * Screening.  A shell product is bounded by the s-type envelope of its most
 * diffuse primitives:  |chi_i chi_j| <~ c_i c_j exp(-mu R_ij^2) exp(-p |r-P|^2),
 * p = a+b, mu = ab/p.  The SR interaction of two normalised s Gaussians
 * (exponents p, q) at distance d is
 *     g(d) = [erf(gamma d) - erf(beta d)]/d,
 *     gamma = sqrt(pq/(p+q)),  1/beta^2 = 1/omega^2 + 1/p + 1/q,
 * so a triple is skipped when
 *     c_i c_j exp(-mu R^2) (pi/p)^1.5 * c_k (pi/q)^1.5 * g(d) < cutoff.
 * Angular factors are not included; the bound is validated numerically
 * (see socutils/pbc/test) and the caller's cutoff carries a safety margin.
 * Auxiliary image shells are binned on a uniform grid so that each pair only
 * visits the bins inside its interaction sphere.
 *
 * Work is distributed over cell-0 AO shells (rows mu of B), so threads never
 * write the same element.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"
#include "cint_funcs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

extern CINTIntegralFunction int3c2e_sph;
extern CINTIntegralFunction int2c2e_sph;
extern CINTOptimizerFunction int3c2e_optimizer;
extern CINTOptimizerFunction int2c2e_optimizer;

static inline const double *shell_center(int sh, int *atm, int *bas, double *env)
{
        return env + atm[bas[sh * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
}

static inline int shell_dim(int sh, int *bas)
{
        return (2 * bas[sh * BAS_SLOTS + ANG_OF] + 1) * bas[sh * BAS_SLOTS + NCTR_OF];
}

/* SR interaction of two normalised s Gaussians at distance d */
static inline double sr_gauss(double p, double q, double omega, double d)
{
        double gamma = sqrt(p * q / (p + q));
        double beta = 1. / sqrt(1. / (omega * omega) + 1. / p + 1. / q);
        if (d < 1e-8) {
                return 2. * (gamma - beta) / sqrt(M_PI);
        }
        return (erf(gamma * d) - erf(beta * d)) / d;
}

/* radius beyond which erfc(beta d)/d * fac < cutoff (upper bound of g) */
static double sr_radius(double beta, double fac, double cutoff)
{
        if (fac <= cutoff) {
                return 0.;
        }
        double lo = 0., hi = 1.;
        while (erfc(beta * hi) / hi * fac > cutoff && hi < 1e4) {
                lo = hi;
                hi *= 2.;
        }
        int it;
        for (it = 0; it < 60; it++) {
                double mid = .5 * (lo + hi);
                if (erfc(beta * mid) / mid * fac > cutoff) {
                        lo = mid;
                } else {
                        hi = mid;
                }
        }
        return hi;
}

typedef struct {
        double origin[3];
        double spacing;
        int dims[3];
        int *bin_start;   /* CSR over bins, length nbins+1 */
        int *bin_shells;  /* aux shell indices sorted by bin */
} AuxGrid;

static size_t max_cache(CINTIntegralFunction *intor, int sh0, int sh1,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        size_t cache = 0;
        int shls[3];
        int i;
        for (i = sh0; i < sh1; i++) {
                shls[0] = i; shls[1] = i; shls[2] = i;
                size_t n = (size_t)(*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache = MAX(cache, n);
        }
        return cache;
}

#define NPBIN_MAX 24
#define PBIN_RATIO 4.

/* envelope of a shell pair, binned in the product exponent p = a + b */
typedef struct {
        int nbin;
        double p[NPBIN_MAX];      /* smallest p in the bin */
        double k[NPBIN_MAX];      /* largest |c_a c_b| exp(-mu R^2) (pi/p)^1.5 in the bin */
        double shift[NPBIN_MAX];  /* largest distance of a product centre from the midpoint */
} PairEnvelope;

/* largest |coefficient| of primitive ip of shell sh over its contractions */
static inline double prim_cmax(int sh, int ip, int *bas, double *env)
{
        const int nprim = bas[sh * BAS_SLOTS + NPRIM_OF];
        const int nctr = bas[sh * BAS_SLOTS + NCTR_OF];
        const double *c = env + bas[sh * BAS_SLOTS + PTR_COEFF];
        double m = 0.;
        int ic;
        for (ic = 0; ic < nctr; ic++) {
                m = MAX(m, fabs(c[ip + nprim * ic]));
        }
        return m;
}

/* Keep primitive pairs that can exceed the cutoff with some aux function:
 * k * max(c_k (pi/q)^1.5) * max_q g(0; p, q) >= cutoff (g(0) is monotone in q,
 * so its extremes are at the smallest/largest aux exponents). */
static void pair_envelope(PairEnvelope *pe, int ish, int jsh, double r2, double rab,
                          double cutoff, double cqhi, double qlo, double qhi, double omega,
                          int *bas, double *env)
{
        const int ni = bas[ish * BAS_SLOTS + NPRIM_OF];
        const int nj = bas[jsh * BAS_SLOTS + NPRIM_OF];
        const double *ai = env + bas[ish * BAS_SLOTS + PTR_EXP];
        const double *aj = env + bas[jsh * BAS_SLOTS + PTR_EXP];
        pe->nbin = 0;
        int ip, jp, n;
        for (ip = 0; ip < ni; ip++) {
                double ca = prim_cmax(ish, ip, bas, env);
                for (jp = 0; jp < nj; jp++) {
                        double a = ai[ip], b = aj[jp];
                        double p = a + b;
                        double k = ca * prim_cmax(jsh, jp, bas, env) * exp(-a * b / p * r2) *
                                   pow(M_PI / p, 1.5);
                        double g0 = MAX(sr_gauss(p, qlo, omega, 0.), sr_gauss(p, qhi, omega, 0.));
                        if (k * cqhi * g0 < cutoff) {
                                continue;
                        }
                        /* product centre A + b/p (B - A): offset from the midpoint */
                        double shift = fabs(b / p - .5) * rab;
                        for (n = 0; n < pe->nbin; n++) {
                                if (p < pe->p[n] * PBIN_RATIO && p * PBIN_RATIO > pe->p[n]) {
                                        break;
                                }
                        }
                        if (n == pe->nbin) {
                                if (n == NPBIN_MAX) {
                                        n = NPBIN_MAX - 1;  /* merge into the last bin */
                                } else {
                                        pe->nbin++;
                                        pe->p[n] = p;
                                        pe->k[n] = 0.;
                                        pe->shift[n] = 0.;
                                }
                        }
                        pe->p[n] = MIN(pe->p[n], p);
                        pe->k[n] = MAX(pe->k[n], k);
                        pe->shift[n] = MAX(pe->shift[n], shift);
                }
        }
}

/* bound of (pair | aux shell)_SR for an aux envelope (q, kq) at distance d from the midpoint */
static inline double pair_aux_bound(PairEnvelope *pe, double q, double kq, double d, double omega)
{
        double est = 0.;
        int n;
        for (n = 0; n < pe->nbin; n++) {
                double dd = MAX(0., d - pe->shift[n]);
                est = MAX(est, pe->k[n] * kq * sr_gauss(pe->p[n], q, omega, dd));
        }
        return est;
}

/*
 * out:       (naux0, nao0, nao0), accumulated
 * nbas0:     number of cell-0 AO shells (rows)
 * nao_sh:    number of AO image shells (cell 0 first)
 * aux_sh0, naux_sh: range of aux image shells
 * ao_fold:   [nao_sh] AO offset in cell 0 of each AO image shell
 * sh_cell:   [nao_sh] cell-0 shell of each AO image shell; only pairs with
 *            sh_cell[j] >= i are evaluated (the caller adds the transpose of
 *            the off-diagonal shell blocks: (mu_0 nu_T|.) = (nu_0 mu_-T|.) )
 * aux_fold:  [natot] aux offset in cell 0 of each aux image shell (indexed by shell id)
 * qmin, cqmax: [natot] screening parameters of aux shells (indexed by shell id)
 * grid:      aux bins (origin, spacing, dims, bin_start, bin_shells)
 * stats:     [3] out: number of pairs kept, triples evaluated, triples screened
 */
void SOCpbc_sr_int3c(double *out, int nao0, int naux0,
                     int nbas0, int nao_sh, int aux_sh0, int naux_sh,
                     int *ao_fold, int *sh_cell, int *aux_fold,
                     double *qmin, double *cqmax,
                     double *grid_origin, double grid_spacing, int *grid_dims,
                     int *bin_start, int *bin_shells,
                     double omega, double cutoff,
                     int *atm, int natm, int *bas, int nbas, double *env,
                     long *stats)
{
        CINTOpt *opt = NULL;
        int3c2e_optimizer(&opt, atm, natm, bas, nbas, env);
        size_t cache = max_cache(int3c2e_sph, 0, nbas, atm, natm, bas, nbas, env);
        int dimax = 0;
        int i;
        for (i = 0; i < nbas; i++) {
                dimax = MAX(dimax, shell_dim(i, bas));
        }
        double qlo = 1e300, qhi = 0., cqhi = 0.;
        for (i = aux_sh0; i < aux_sh0 + naux_sh; i++) {
                qlo = MIN(qlo, qmin[i]);
                qhi = MAX(qhi, qmin[i]);
                cqhi = MAX(cqhi, cqmax[i] * pow(M_PI / qmin[i], 1.5));
        }
        const size_t nn = (size_t)nao0 * nao0;
        long npair = 0, ntrip = 0, nskip = 0;

#pragma omp parallel reduction(+:npair, ntrip, nskip)
{
        double *buf = malloc(sizeof(double) * ((size_t)dimax * dimax * dimax + cache));
        double *cbuf = buf + (size_t)dimax * dimax * dimax;
        PairEnvelope pe;
        int shls[3];
        int ish, jsh;
#pragma omp for schedule(dynamic)
        for (ish = 0; ish < nbas0; ish++) {
                const double *ri = shell_center(ish, atm, bas, env);
                const int di = shell_dim(ish, bas);
                const int mi = ao_fold[ish];
                for (jsh = 0; jsh < nao_sh; jsh++) {
                        if (sh_cell[jsh] < ish) {
                                continue;
                        }
                        const double *rj = shell_center(jsh, atm, bas, env);
                        double rij[3] = {ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]};
                        double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                        double rab = sqrt(r2);
                        /* primitive pairs that can matter for any aux function */
                        pair_envelope(&pe, ish, jsh, r2, rab, cutoff, cqhi, qlo, qhi, omega, bas, env);
                        if (pe.nbin == 0) {
                                continue;
                        }
                        double rmax = 0.;
                        int n, x;
                        for (n = 0; n < pe.nbin; n++) {
                                double beta = 1. / sqrt(1. / (omega * omega) + 1. / pe.p[n] + 1. / qlo);
                                double g0 = MAX(sr_gauss(pe.p[n], qlo, omega, 0.),
                                                sr_gauss(pe.p[n], qhi, omega, 0.));
                                if (pe.k[n] * cqhi * g0 < cutoff) {
                                        continue;
                                }
                                rmax = MAX(rmax, sr_radius(beta, pe.k[n] * cqhi, cutoff) + pe.shift[n]);
                        }
                        if (rmax <= 0.) {
                                continue;
                        }
                        npair++;
                        double pc[3];
                        for (x = 0; x < 3; x++) {
                                pc[x] = .5 * (ri[x] + rj[x]);
                        }
                        int lo[3], hi[3];
                        for (x = 0; x < 3; x++) {
                                lo[x] = (int)floor((pc[x] - rmax - grid_origin[x]) / grid_spacing);
                                hi[x] = (int)floor((pc[x] + rmax - grid_origin[x]) / grid_spacing);
                                lo[x] = MAX(lo[x], 0);
                                hi[x] = MIN(hi[x], grid_dims[x] - 1);
                        }
                        const int dj = shell_dim(jsh, bas);
                        const int nj = ao_fold[jsh];
                        int bx, by, bz;
                        for (bx = lo[0]; bx <= hi[0]; bx++) {
                        for (by = lo[1]; by <= hi[1]; by++) {
                        for (bz = lo[2]; bz <= hi[2]; bz++) {
                                int bin = (bx * grid_dims[1] + by) * grid_dims[2] + bz;
                                for (n = bin_start[bin]; n < bin_start[bin + 1]; n++) {
                                        int ksh = bin_shells[n];
                                        const double *rk = shell_center(ksh, atm, bas, env);
                                        double d = sqrt((pc[0] - rk[0]) * (pc[0] - rk[0]) +
                                                        (pc[1] - rk[1]) * (pc[1] - rk[1]) +
                                                        (pc[2] - rk[2]) * (pc[2] - rk[2]));
                                        if (d > rmax) {
                                                continue;
                                        }
                                        double q = qmin[ksh];
                                        double est = pair_aux_bound(&pe, q, cqmax[ksh] * pow(M_PI / q, 1.5),
                                                                    d, omega);
                                        if (est < cutoff) {
                                                nskip++;
                                                continue;
                                        }
                                        ntrip++;
                                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh;
                                        if (!int3c2e_sph(buf, NULL, shls, atm, natm, bas, nbas,
                                                         env, opt, cbuf)) {
                                                continue;
                                        }
                                        const int dk = shell_dim(ksh, bas);
                                        const int pk = aux_fold[ksh];
                                        int ii, jj, kk;
                                        for (kk = 0; kk < dk; kk++) {
                                                double *o = out + (size_t)(pk + kk) * nn;
                                                for (jj = 0; jj < dj; jj++) {
                                                for (ii = 0; ii < di; ii++) {
                                                        o[(size_t)(mi + ii) * nao0 + nj + jj] +=
                                                                buf[ii + di * (jj + dj * kk)];
                                                } }
                                        }
                                }
                        } } }
                }
        }
        free(buf);
}
        CINTdel_optimizer(&opt);
        stats[0] = npair;
        stats[1] = ntrip;
        stats[2] = nskip;
}

/*
 * out: (naux0, naux0), accumulated.  Rows: the first naux_sh0 aux image shells
 * (cell 0, starting at aux_sh0); columns: all aux image shells, folded.
 */
void SOCpbc_sr_int2c(double *out, int naux0, int aux_sh0, int naux_sh0, int naux_sh,
                     int *aux_fold, double *qmin, double *cqmax,
                     double omega, double cutoff,
                     int *atm, int natm, int *bas, int nbas, double *env,
                     long *stats)
{
        CINTOpt *opt = NULL;
        int2c2e_optimizer(&opt, atm, natm, bas, nbas, env);
        size_t cache = max_cache(int2c2e_sph, aux_sh0, aux_sh0 + naux_sh,
                                 atm, natm, bas, nbas, env);
        int dimax = 0;
        int i;
        for (i = aux_sh0; i < aux_sh0 + naux_sh; i++) {
                dimax = MAX(dimax, shell_dim(i, bas));
        }
        long npair = 0, nskip = 0;
#pragma omp parallel reduction(+:npair, nskip)
{
        double *buf = malloc(sizeof(double) * ((size_t)dimax * dimax + cache));
        double *cbuf = buf + (size_t)dimax * dimax;
        int shls[2];
        int psh, qsh;
#pragma omp for schedule(dynamic)
        for (psh = aux_sh0; psh < aux_sh0 + naux_sh0; psh++) {
                const double *rp = shell_center(psh, atm, bas, env);
                const int dp = shell_dim(psh, bas);
                const int pp = aux_fold[psh];
                const double kp = cqmax[psh] * pow(M_PI / qmin[psh], 1.5);
                for (qsh = aux_sh0; qsh < aux_sh0 + naux_sh; qsh++) {
                        const double *rq = shell_center(qsh, atm, bas, env);
                        double d = sqrt((rp[0] - rq[0]) * (rp[0] - rq[0]) +
                                        (rp[1] - rq[1]) * (rp[1] - rq[1]) +
                                        (rp[2] - rq[2]) * (rp[2] - rq[2]));
                        double est = kp * cqmax[qsh] * pow(M_PI / qmin[qsh], 1.5) *
                                     sr_gauss(qmin[psh], qmin[qsh], omega, d);
                        if (est < cutoff) {
                                nskip++;
                                continue;
                        }
                        npair++;
                        shls[0] = psh; shls[1] = qsh;
                        if (!int2c2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cbuf)) {
                                continue;
                        }
                        const int dq = shell_dim(qsh, bas);
                        const int qq0 = aux_fold[qsh];
                        int a, b;
                        for (b = 0; b < dq; b++) {
                        for (a = 0; a < dp; a++) {
                                out[(size_t)(pp + a) * naux0 + qq0 + b] += buf[a + dp * b];
                        } }
                }
        }
        free(buf);
}
        CINTdel_optimizer(&opt);
        stats[0] = npair;
        stats[1] = nskip;
}
