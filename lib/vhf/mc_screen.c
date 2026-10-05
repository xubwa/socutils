/*
 * Multi-centre prescreen functions for the relativistic direct-SCF drivers of
 * pyscf (lib/vhf/r_direct_o1.c).
 *
 * The X2C model potential contracts a superposition of atomic densities with
 * the molecular two-electron integrals.  For such an atom-block-diagonal
 * density the shell quartets with all four shells on the same atom reproduce
 * the two-electron Fock matrix of the atomic calculation, which the atomic
 * code (x2camf) has already built.  Each function below rejects those
 * one-centre quartets and defers every other quartet to the corresponding
 * pyscf prescreen function, so the driver returns the multi-centre part of J
 * and K only.
 */
#include <stddef.h>
#include "cint.h"
#include "vhf/optimizer.h"

int CVHFrkbllll_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env);
int CVHFrkbssll_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env);
int CVHFrkb_gaunt_lsls_prescreen(int *shls, CVHFOpt *opt,
                                 int *atm, int *bas, double *env);
int CVHFrkb_gaunt_lssl_prescreen(int *shls, CVHFOpt *opt,
                                 int *atm, int *bas, double *env);

static inline int one_centre(int *shls, int *bas)
{
        int ia = bas[ATOM_OF + shls[0] * BAS_SLOTS];
        return ia == bas[ATOM_OF + shls[1] * BAS_SLOTS]
            && ia == bas[ATOM_OF + shls[2] * BAS_SLOTS]
            && ia == bas[ATOM_OF + shls[3] * BAS_SLOTS];
}

/* (LL|LL), and (SS|SS) which pyscf screens with the same function */
int SOCrkbllll_mc_prescreen(int *shls, CVHFOpt *opt,
                            int *atm, int *bas, double *env)
{
        if (one_centre(shls, bas)) {
                return 0;
        }
        return CVHFrkbllll_prescreen(shls, opt, atm, bas, env);
}

int SOCrkbssll_mc_prescreen(int *shls, CVHFOpt *opt,
                            int *atm, int *bas, double *env)
{
        if (one_centre(shls, bas)) {
                return 0;
        }
        return CVHFrkbssll_prescreen(shls, opt, atm, bas, env);
}

int SOCrkb_gaunt_lsls_mc_prescreen(int *shls, CVHFOpt *opt,
                                   int *atm, int *bas, double *env)
{
        if (one_centre(shls, bas)) {
                return 0;
        }
        return CVHFrkb_gaunt_lsls_prescreen(shls, opt, atm, bas, env);
}

int SOCrkb_gaunt_lssl_mc_prescreen(int *shls, CVHFOpt *opt,
                                   int *atm, int *bas, double *env)
{
        if (one_centre(shls, bas)) {
                return 0;
        }
        return CVHFrkb_gaunt_lssl_prescreen(shls, opt, atm, bas, env);
}

/*
 * Exchange-only prescreen for the Gaunt/Breit integrals.  The direct (J-type)
 * Gaunt/Breit term is the interaction with the current density, which is
 * zero for a superposition of spherically averaged, Kramers-symmetric atomic
 * densities.  Only the quartets that contribute to K are kept:
 *
 *   (LS|LS), K_il = (ij|kl) D^SL_jk.  dm_cond is that of the one density
 *   matrix D^SL (CVHFrkb_dm_cond).
 *
 *   (LS|SL), K^SS_kj = (ij|kl) D^LL_li and K^LL_il = (ij|kl) D^SS_jk.
 *   dm_cond is that of the two density matrices (D^LL, D^SS): the block 0
 *   is the maximum over the two, the blocks 1 and 2 are D^LL and D^SS.
 *   K^SS and K^LL are Hermitian, so only the quartets for their lower
 *   triangles (k >= j, i >= l) are kept; the upper triangles returned by the
 *   driver are incomplete.
 */
int SOCrkb_gaunt_lsls_mck_prescreen(int *shls, CVHFOpt *opt,
                                    int *atm, int *bas, double *env)
{
        if (one_centre(shls, bas)) {
                return 0;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            && opt->dm_cond[j*n+k] > dmin;
}

int SOCrkb_gaunt_lssl_mck_prescreen(int *shls, CVHFOpt *opt,
                                    int *atm, int *bas, double *env)
{
        if (one_centre(shls, bas)) {
                return 0;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        double *dmll = opt->dm_cond + n*n;
        double *dmss = opt->dm_cond + n*n*2;
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((k >= j && dmll[l*n+i] > dmin)
            || (i >= l && dmss[j*n+k] > dmin));
}
