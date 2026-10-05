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
