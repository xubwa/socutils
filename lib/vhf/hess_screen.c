/*
 * Prescreen function for the contractions of two-electron derivative
 * integrals with arbitrary density pairs (socutils.hessian.ghf: integral
 * derivatives of the GHF / GKS energy at fixed densities, any script).
 *
 * pyscf's hessian prescreen functions (lib/vhf/hessian_screen.c) test the
 * density on the index pairs of the RHF scripts only.  Here the integral
 * estimate is q_cond[0][ij] * q_cond[1][kl] (the derivative-integral
 * estimates of pyscf's CVHFnr_int2e_pp(pp)_q_cond for the pair carrying the
 * derivatives, the plain Schwarz estimate otherwise), and a quartet is kept
 * if any of the six index pairs of the (symmetrized) dm_cond can make the
 * contribution exceed direct_scf_cutoff.
 */
#include <stddef.h>
#include "cint.h"
#include "vhf/optimizer.h"

static inline double dmax2(double a, double b)
{
        return a > b ? a : b;
}

int SOCVHFderiv_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        double *q_bra = opt->q_cond;
        double *q_ket = opt->q_cond + n * n;
        double qijkl = q_bra[i*n+j] * q_ket[k*n+l];
        if (qijkl < opt->direct_scf_cutoff) {
                return 0;
        }
        double dmin = opt->direct_scf_cutoff / qijkl;
        double *dm = opt->dm_cond;
        double d = dmax2(dmax2(dm[i*n+j], dm[k*n+l]),
                         dmax2(dmax2(dm[i*n+k], dm[i*n+l]),
                               dmax2(dm[j*n+k], dm[j*n+l])));
        return d > dmin;
}
