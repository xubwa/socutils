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
 * Screening (rigorous envelopes).  Every primitive of angular momentum l is
 * bounded pointwise by an s Gaussian,
 *     |c Y_lm r^l exp(-a r^2)| <= c Ymax_l (l/(a e))^(l/2) exp(-a r^2/2),
 * Ymax_l = sqrt((2l+1)/4pi) (l = 0: exponent a, factor Ymax_0).  With a
 * positive kernel the interaction of the envelopes bounds the integral, and
 * the SR interaction of two unit-charge s Gaussians (exponents p, q) at
 * distance d is
 *     g(d) = [erf(gamma d) - erf(beta d)]/d,
 *     gamma = sqrt(pq/(p+q)),  1/beta^2 = 1/omega^2 + 1/p + 1/q.
 * Products of AO primitive envelopes are binned in p (ratio PBIN_RATIO; the
 * charges in a bin are summed, g is taken at both ends of the bin).  Each aux
 * shell comes with an envelope exponent and envelope charge computed by the
 * caller (in the units in which the aux functions are used).
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
#ifndef M_E
#define M_E 2.71828182845904523536
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
        double p[NPBIN_MAX];      /* smallest envelope exponent p in the bin */
        double pmax[NPBIN_MAX];   /* largest p in the bin */
        double k[NPBIN_MAX];      /* summed envelope charges in the bin */
        double shift[NPBIN_MAX];  /* largest distance of a product centre from the midpoint */
} PairEnvelope;

/* Envelopes of a primitive (exponent a, angular l).  For 0 < t < 1,
 *   r^l exp(-a r^2) <= (l / (2 (1-t) a e))^(l/2) exp(-t a r^2),
 * so each t gives a pointwise s-Gaussian bound (l = 0: exact, t = 1).  A larger
 * t decays faster but has a larger amplitude; the screening uses the smallest
 * of the bounds over a few t (NENV).  ang[l] is the largest |angular part| of a
 * libcint function of momentum l per unit coefficient, measured by the caller
 * (it absorbs libcint's conventions). */
#define NENV 3
static const double ENV_T[NENV] = {.5, .75, .9};

static inline void prim_envelope(double a, int l, double t, const double *ang, double *ae, double *fac)
{
        double ymax = ang[l];
        if (l == 0) {
                *ae = a;
                *fac = ymax;
        } else {
                *ae = t * a;
                *fac = ymax * pow(l / (2. * (1. - t) * a * M_E), .5 * l);
        }
}

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

typedef struct {
        double w, p, k, shift;
} PrimPair;

static int cmp_primpair(const void *x, const void *y)
{
        double a = ((const PrimPair *)x)->w, b = ((const PrimPair *)y)->w;
        return (a > b) - (a < b);
}

/* Envelope of the primitive pairs that can matter for some aux function.
 * w = k * max_k Q_k * max_q g(0; p, q) bounds the contribution of a primitive
 * pair to any triple (g(0) is monotone in q, so its extremes are at the
 * smallest/largest aux exponents).  The smallest are dropped while their sum
 * stays below cutoff/2, which bounds the error of the dropped pairs together.
 * pp: work space of ni*nj PrimPair. */
static void pair_envelope(PairEnvelope *pe, int ish, int jsh, double r2, double rab,
                          double cutoff, double cqhi, double qlo, double qhi, double omega,
                          double t, const double *ang, int *bas, double *env, PrimPair *pp)
{
        const int ni = bas[ish * BAS_SLOTS + NPRIM_OF];
        const int nj = bas[jsh * BAS_SLOTS + NPRIM_OF];
        const int li = bas[ish * BAS_SLOTS + ANG_OF];
        const int lj = bas[jsh * BAS_SLOTS + ANG_OF];
        const double *ai = env + bas[ish * BAS_SLOTS + PTR_EXP];
        const double *aj = env + bas[jsh * BAS_SLOTS + PTR_EXP];
        int ip, jp, n, m = 0;
        double wsum = 0.;
        for (ip = 0; ip < ni; ip++) {
                double a, fa;
                prim_envelope(ai[ip], li, t, ang, &a, &fa);
                double ca = prim_cmax(ish, ip, bas, env) * fa;
                for (jp = 0; jp < nj; jp++) {
                        double b, fb;
                        prim_envelope(aj[jp], lj, t, ang, &b, &fb);
                        double p = a + b;
                        double k = ca * prim_cmax(jsh, jp, bas, env) * fb * exp(-a * b / p * r2) *
                                   pow(M_PI / p, 1.5);
                        double g0 = MAX(sr_gauss(p, qlo, omega, 0.), sr_gauss(p, qhi, omega, 0.));
                        pp[m].w = k * cqhi * g0;
                        pp[m].p = p;
                        pp[m].k = k;
                        /* product centre A + b/p (B - A): offset from the midpoint */
                        pp[m].shift = fabs(b / p - .5) * rab;
                        wsum += pp[m].w;
                        m++;
                }
        }
        pe->nbin = 0;
        if (wsum < .5 * cutoff) {
                return;
        }
        qsort(pp, m, sizeof(PrimPair), cmp_primpair);
        int i0 = 0;
        double dropped = 0.;
        while (i0 < m && dropped + pp[i0].w < .5 * cutoff) {
                dropped += pp[i0].w;
                i0++;
        }
        for (; i0 < m; i0++) {
                double p = pp[i0].p;
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
                                pe->pmax[n] = p;
                                pe->k[n] = 0.;
                                pe->shift[n] = 0.;
                        }
                }
                pe->p[n] = MIN(pe->p[n], p);
                pe->pmax[n] = MAX(pe->pmax[n], p);
                pe->k[n] += pp[i0].k;
                pe->shift[n] = MAX(pe->shift[n], pp[i0].shift);
        }
}

/* sum over bins of the long-range tail bound k Q erfc(beta d')/d', d' = d - shift */
static double pair_tail(PairEnvelope *pe, const double *beta, double kq, double d)
{
        double t = 0.;
        int n;
        for (n = 0; n < pe->nbin; n++) {
                double dd = d - pe->shift[n];
                if (dd <= 0.) {
                        return 1e300;
                }
                t += pe->k[n] * kq * erfc(beta[n] * dd) / dd;
        }
        return t;
}

/* radius beyond which no aux function can reach the cutoff (0 if none can at all) */
static double pair_radius(PairEnvelope *pe, double cqhi, double qlo, double qhi,
                          double omega, double cutoff)
{
        double beta[NPBIN_MAX];
        double g0 = 0.;
        int n;
        for (n = 0; n < pe->nbin; n++) {
                beta[n] = 1. / sqrt(1. / (omega * omega) + 1. / pe->p[n] + 1. / qlo);
                g0 += pe->k[n] * cqhi * MAX(MAX(sr_gauss(pe->p[n], qlo, omega, 0.),
                                                sr_gauss(pe->p[n], qhi, omega, 0.)),
                                            MAX(sr_gauss(pe->pmax[n], qlo, omega, 0.),
                                                sr_gauss(pe->pmax[n], qhi, omega, 0.)));
        }
        if (g0 < cutoff) {
                return 0.;
        }
        double lo = 0., hi = 1.;
        while (pair_tail(pe, beta, cqhi, hi) > cutoff && hi < 1e4) {
                lo = hi;
                hi *= 2.;
        }
        int it;
        for (it = 0; it < 50; it++) {
                double mid = .5 * (lo + hi);
                if (pair_tail(pe, beta, cqhi, mid) > cutoff) {
                        lo = mid;
                } else {
                        hi = mid;
                }
        }
        return hi;
}

static inline double sr_gauss0(double p, double q, double omega)
{
        double gamma = sqrt(p * q / (p + q));
        double beta = 1. / sqrt(1. / (omega * omega) + 1. / p + 1. / q);
        return 2. * (gamma - beta) / sqrt(M_PI);
}

/* cheap upper bound of pair_aux_bound (no erf): g(d) <= g(0) and
 * g(d) <= erfc(beta d)/d <= exp(-beta^2 d^2)/d, beta decreasing towards small p */
static inline double pair_aux_bound_cheap(PairEnvelope *pe, double q, double kq, double d, double omega)
{
        double est = 0.;
        int n;
        for (n = 0; n < pe->nbin; n++) {
                double dd = MAX(0., d - pe->shift[n]);
                double g = MAX(sr_gauss0(pe->p[n], q, omega), sr_gauss0(pe->pmax[n], q, omega));
                if (dd > 0.) {
                        double b2 = 1. / (1. / (omega * omega) + 1. / pe->p[n] + 1. / q);
                        g = MIN(g, exp(-b2 * dd * dd) / dd);
                }
                est += pe->k[n] * kq * g;
        }
        return est;
}

/* bound of (pair | aux shell)_SR for an aux envelope (q, charge kq) at distance d
 * from the midpoint: sum over bins, g at both ends of each bin */
static inline double pair_aux_bound(PairEnvelope *pe, double q, double kq, double d, double omega)
{
        double est = 0.;
        int n;
        for (n = 0; n < pe->nbin; n++) {
                double dd = MAX(0., d - pe->shift[n]);
                double g = MAX(sr_gauss(pe->p[n], q, omega, dd), sr_gauss(pe->pmax[n], q, omega, dd));
                est += pe->k[n] * kq * g;
        }
        return est;
}

/*
 * out:       (naux0, nrow, nao0), accumulated: rows are the AOs of the cell-0
 *            shells [ish0, ish1), starting at AO row0
 * nbas0:     number of cell-0 AO shells
 * nao_sh:    number of AO image shells (cell 0 first)
 * aux_sh0, naux_sh: range of aux image shells
 * ao_fold:   [nao_sh] AO offset in cell 0 of each AO image shell
 * sh_cell:   [nao_sh] cell-0 shell of each AO image shell (image shells are
 *            translation-major, nao_sh = nT nbas0); only pairs with
 *            sh_cell[j] <= i are evaluated, i.e. the lower shell triangle of
 *            the folded (symmetric) matrices; diagonal shell blocks are complete
 * aux_fold:  [natot] aux offset in cell 0 of each aux image shell (indexed by shell id)
 * qmin, cqmax: [NENV, nbas] envelope exponent and charge of each aux shell for
 *            each envelope parameter ENV_T (rows indexed by shell id)
 * grid:      aux bins (origin, spacing, dims, bin_start, bin_shells)
 * ang:       [lmax+1] largest |angular part| per unit coefficient (see prim_envelope)
 * stats:     [3] out: number of pairs kept, triples evaluated, triples screened
 */
void SOCpbc_sr_int3c(double *out, int nao0, int naux0, int ish0, int ish1, int row0, int nrow,
                     int nbas0, int nao_sh, int aux_sh0, int naux_sh,
                     int *ao_fold, int *sh_cell, int *aux_fold,
                     double *qmin, double *cqmax,
                     double *grid_origin, double grid_spacing, int *grid_dims,
                     int *bin_start, int *bin_shells,
                     double omega, double cutoff, double *ang,
                     int *atm, int natm, int *bas, int nbas, double *env,
                     long *stats)
{
        CINTOpt *opt = NULL;
        int3c2e_optimizer(&opt, atm, natm, bas, nbas, env);
        size_t cache = max_cache(int3c2e_sph, 0, nbas, atm, natm, bas, nbas, env);
        int dimax = 0, npmax = 1;
        int i;
        for (i = 0; i < nbas; i++) {
                dimax = MAX(dimax, shell_dim(i, bas));
        }
        for (i = 0; i < nao_sh; i++) {
                npmax = MAX(npmax, bas[i * BAS_SLOTS + NPRIM_OF]);
        }
        double qlo[NENV], qhi[NENV], cqhi[NENV];
        int ie;
        for (ie = 0; ie < NENV; ie++) {
                qlo[ie] = 1e300; qhi[ie] = 0.; cqhi[ie] = 0.;
                for (i = aux_sh0; i < aux_sh0 + naux_sh; i++) {
                        qlo[ie] = MIN(qlo[ie], qmin[ie * nbas + i]);
                        qhi[ie] = MAX(qhi[ie], qmin[ie * nbas + i]);
                        cqhi[ie] = MAX(cqhi[ie], cqmax[ie * nbas + i]);
                }
        }
        const size_t nn = (size_t)nrow * nao0;
        long npair = 0, ntrip = 0, nskip = 0;

#pragma omp parallel reduction(+:npair, ntrip, nskip)
{
        double *buf = malloc(sizeof(double) * ((size_t)dimax * dimax * dimax + cache));
        double *cbuf = buf + (size_t)dimax * dimax * dimax;
        PrimPair *pp = malloc(sizeof(PrimPair) * (size_t)npmax * npmax);
        PairEnvelope pe[NENV];
        int shls[3];
        int ish, jsh;
        /* tasks: (row shell ish, cell-0 column shell jc <= ish) with all images
         * of jc (image shells are translation-major: jsh = t nbas0 + jc), so
         * tasks write disjoint output blocks and there are many of them */
        int task;
#pragma omp for schedule(dynamic)
        for (task = 0; task < (ish1 - ish0) * nbas0; task++) {
                const int jc = task % nbas0;
                ish = ish0 + task / nbas0;
                if (jc > ish) {
                        continue;
                }
                const double *ri = shell_center(ish, atm, bas, env);
                const int di = shell_dim(ish, bas);
                const int mi = ao_fold[ish] - row0;
                for (jsh = jc; jsh < nao_sh; jsh += nbas0) {
                        const double *rj = shell_center(jsh, atm, bas, env);
                        double rij[3] = {ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]};
                        double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
                        double rab = sqrt(r2);
                        /* primitive pairs that can matter for any aux function */
                        /* every envelope bounds the pair; the primitive pairs
                         * each drops sum to < cutoff/2, so a triple can be
                         * skipped when one of them bounds the rest by cutoff/2 */
                        int n, x, e;
                        double rmax = 1e300;
                        for (e = 0; e < NENV; e++) {
                                pair_envelope(pe + e, ish, jsh, r2, rab, cutoff, cqhi[e], qlo[e], qhi[e],
                                              omega, ENV_T[e], ang, bas, env, pp);
                                rmax = MIN(rmax, pe[e].nbin == 0 ? 0. :
                                           pair_radius(pe + e, cqhi[e], qlo[e], qhi[e], omega, .5 * cutoff));
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
                                        int skip = 0;
                                        for (e = 0; e < NENV && !skip; e++) {
                                                skip = pair_aux_bound_cheap(pe + e, qmin[e * nbas + ksh],
                                                                            cqmax[e * nbas + ksh], d, omega) < .5 * cutoff;
                                        }
                                        for (e = 0; e < NENV && !skip; e++) {
                                                skip = pair_aux_bound(pe + e, qmin[e * nbas + ksh],
                                                                      cqmax[e * nbas + ksh], d, omega) < .5 * cutoff;
                                        }
                                        if (skip) {
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
        free(pp);
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
 * env_full, env_lr: env with range omega 0 and +omega.  The SR integrals are
 * evaluated as full - LR: libcint's erfc-attenuated 2c integrals lose ~1e-10
 * relative accuracy for diffuse high-l pairs (Y f shells: 3.7e-10 at omega
 * 0.5, checked against Fourier quadrature), enough to make the fitting metric
 * indefinite.  For the kept (short-range) pairs the difference is benign.
 */
void SOCpbc_sr_int2c(double *out, int naux0, int aux_sh0, int naux_sh0, int naux_sh,
                     int *aux_fold, double *qmin, double *cqmax,
                     double omega, double cutoff,
                     int *atm, int natm, int *bas, int nbas, double *env_full, double *env_lr,
                     long *stats)
{
        CINTOpt *opt = NULL;
        CINTOpt *opt_lr = NULL;
        int2c2e_optimizer(&opt, atm, natm, bas, nbas, env_full);
        int2c2e_optimizer(&opt_lr, atm, natm, bas, nbas, env_lr);
        size_t cache = MAX(max_cache(int2c2e_sph, aux_sh0, aux_sh0 + naux_sh, atm, natm, bas, nbas, env_full),
                           max_cache(int2c2e_sph, aux_sh0, aux_sh0 + naux_sh, atm, natm, bas, nbas, env_lr));
        int dimax = 0;
        int i;
        for (i = aux_sh0; i < aux_sh0 + naux_sh; i++) {
                dimax = MAX(dimax, shell_dim(i, bas));
        }
        long npair = 0, nskip = 0;
#pragma omp parallel reduction(+:npair, nskip)
{
        double *buf = malloc(sizeof(double) * ((size_t)2 * dimax * dimax + cache));
        double *buf_lr = buf + (size_t)dimax * dimax;
        double *cbuf = buf_lr + (size_t)dimax * dimax;
        int shls[2];
        int psh, qsh;
#pragma omp for schedule(dynamic)
        for (psh = aux_sh0; psh < aux_sh0 + naux_sh0; psh++) {
                const double *rp = shell_center(psh, atm, bas, env_full);
                const int dp = shell_dim(psh, bas);
                const int pp = aux_fold[psh];

                for (qsh = aux_sh0; qsh < aux_sh0 + naux_sh; qsh++) {
                        const double *rq = shell_center(qsh, atm, bas, env_full);
                        double d = sqrt((rp[0] - rq[0]) * (rp[0] - rq[0]) +
                                        (rp[1] - rq[1]) * (rp[1] - rq[1]) +
                                        (rp[2] - rq[2]) * (rp[2] - rq[2]));
                        double est = 1e300;
                        int e;
                        for (e = 0; e < NENV; e++) {
                                est = MIN(est, cqmax[e * nbas + psh] * cqmax[e * nbas + qsh] *
                                               sr_gauss(qmin[e * nbas + psh], qmin[e * nbas + qsh], omega, d));
                        }
                        if (est < cutoff) {
                                nskip++;
                                continue;
                        }
                        npair++;
                        shls[0] = psh; shls[1] = qsh;
                        int nz = int2c2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env_full, opt, cbuf);
                        int nz_lr = int2c2e_sph(buf_lr, NULL, shls, atm, natm, bas, nbas, env_lr, opt_lr, cbuf);
                        if (!nz && !nz_lr) {
                                continue;
                        }
                        const int dq = shell_dim(qsh, bas);
                        const int qq0 = aux_fold[qsh];
                        int a, b;
                        for (b = 0; b < dq; b++) {
                        for (a = 0; a < dp; a++) {
                                out[(size_t)(pp + a) * naux0 + qq0 + b] +=
                                        (nz ? buf[a + dp * b] : 0.) - (nz_lr ? buf_lr[a + dp * b] : 0.);
                        } }
                }
        }
        free(buf);
}
        CINTdel_optimizer(&opt);
        CINTdel_optimizer(&opt_lr);
        stats[0] = npair;
        stats[1] = nskip;
}

int SOCpbc_nenv(double *t)
{
        int e;
        for (e = 0; e < NENV; e++) {
                t[e] = ENV_T[e];
        }
        return NENV;
}
