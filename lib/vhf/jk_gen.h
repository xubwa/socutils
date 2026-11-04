/*
 * Definitions shared by jk_gen.c and the generated first-derivative kernels
 * (grad_vrr_gen_*.c, gen_grad_vrr.py): the primitive-pair table, the F_m
 * Taylor tables' layout, the screening helper.
 */
#ifndef SOCVHF_JK_GEN_H
#define SOCVHF_JK_GEN_H
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

#define GT_STEP 0.05
#define GT_TMAX 40.
#define GT_N ((int)(GT_TMAX / GT_STEP) + 2)
#define GT_ORD 7                        /* Taylor terms */
#define GT_TN (4 * 2 + GT_ORD + 1)      /* F_0 .. F_(4 lmax + GT_ORD) per grid point */
#define TWO_PI52 34.986836655249725
#define G_MAXCTR 4

typedef struct {
        int n, nc, nci, ncj, ld, li, lj;
        double *p, *px, *py, *pz, *kp;  /* p, P, K_ab / p */
        double *rix, *riy, *riz;        /* P - R_i */
        double *rjx, *rjy, *rjz;        /* P - R_j */
        double *bnd;                    /* bound of the primitive pair (with the 1 / sqrt(p) of the other side) */
        double *beta;                   /* the same without 1 / sqrt(p); the primitives sorted by it, descending */
        double *c;                      /* c[cb * ld + t], cb = c_j nc_i + c_i */
        double *ei, *ej;                /* the primitive exponents of shells i, j */
        double bmax, rij[3];            /* max bound without 1 / sqrt(p); R_i - R_j */
        double amax;                    /* the largest exponent of the pair (derivative bounds) */
} GPair;

/* the kets of GL_NL shell quartets of one ket type (jk_lane_gen.c): per ket
 * primitive pair t (canonical order) the scalars and the lane vectors */
#ifndef GL_NL
#define GL_NL 8                 /* quartets per lane call (jk_multi.c JK_NL, jk_bench.c BL_NL the same) */
#endif
#define GL_MAXP 64
typedef struct {
        int nk, nck, nci, ncj;
        double q[GL_MAXP], bmx[GL_MAXP];        /* q; max over the lanes of the bound */
        double c[G_MAXCTR * G_MAXCTR * GL_MAXP];  /* c[kc * GL_MAXP + t] */
        double qx[GL_MAXP][GL_NL], qy[GL_MAXP][GL_NL], qz[GL_MAXP][GL_NL];
        double qcx[GL_MAXP][GL_NL], qcy[GL_MAXP][GL_NL], qcz[GL_MAXP][GL_NL];
        double kp[GL_MAXP][GL_NL];              /* K_cd / q */
        double cd[3][GL_NL];                    /* R_C - R_D */
} LaneK;

static inline double taylor(const double *f, double dt)
{
        double s = f[6] * (dt / 6);
        s = (f[5] + s) * (dt / 5);
        s = (f[4] + s) * (dt / 4);
        s = (f[3] + s) * (dt / 3);
        s = (f[2] + s) * (dt / 2);
        s = (f[1] + s) * dt;
        return f[0] + s;
}

/* the leading bra primitives b with beta_b >= thr (beta descending) */
static inline int n_above(const double *beta, int n, double thr)
{
        while (n > 0 && beta[n - 1] < thr) n--;
        return n;
}

#endif
