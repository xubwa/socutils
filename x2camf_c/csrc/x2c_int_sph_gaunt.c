/*
 * x2c_int_sph_gaunt.c -- C translation of src/int_sph_gaunt.cpp (X2CAMF):
 * Gaunt angular helpers + compact/direct two-electron drivers.
 * The unused INT_SPH::get_h2e_JK_gaunt (non-compact) is intentionally
 * omitted.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "x2c_int_sph.h"

/*
    evaluate angular part in 2e gaunt integrals
*/
static inline double factor_p1(int l, int m)
{
    /* return sqrt((l-m)*(l-m-1.0)); */
    return sqrt((l - m) * (l - m - 1.0) / (2.0 * l + 1.0) / (2.0 * l - 1.0));
}
static inline double factor_p2(int l, int m)
{
    /* return -sqrt((l+m+1.0)*(l+m+2.0)); */
    return -sqrt((l + m + 1.0) * (l + m + 2.0) / (2.0 * l + 1.0) /
                 (2.0 * l + 3.0));
}
static inline double factor_m1(int l, int m)
{
    /* return -sqrt((l+m)*(l+m-1.0)); */
    return -sqrt((l + m) * (l + m - 1.0) / (2.0 * l + 1.0) / (2.0 * l - 1.0));
}
static inline double factor_m2(int l, int m)
{
    /* return sqrt((l-m+1.0)*(l-m+2.0)); */
    return sqrt((l - m + 1.0) * (l - m + 2.0) / (2.0 * l + 1.0) /
                (2.0 * l + 3.0));
}
static inline double factor_z1(int l, int m)
{
    /* return sqrt((l+m)*(l-m)); */
    return sqrt((l + m) * (l - m) / (2.0 * l + 1.0) / (2.0 * l - 1.0));
}
static inline double factor_z2(int l, int m)
{
    /* return sqrt((l+m+1.0)*(l-m+1.0)); */
    return sqrt((l + m + 1.0) * (l - m + 1.0) / (2.0 * l + 1.0) /
                (2.0 * l + 3.0));
}

double int_sph_int2e_get_angular_gaunt_LSLS(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL)
{
    double angular = 0.0;
    int l2p = l2 + s2, l4p = l4 + s4;
    for (int MM = -LL; MM <= LL; MM++)
    {
        double tmp1 = s1 * sqrt(l1 + 0.5 + s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 + s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 + 1) / 2, -MM,
                                       (two_m2 + 1) / 2);
        double tmp2 = s2 * sqrt(l1 + 0.5 - s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 - s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 - 1) / 2, -MM,
                                       (two_m2 - 1) / 2);
        double tmp3 = s3 * sqrt(l3 + 0.5 + s3 * two_m3 / 2.0) *
                      sqrt(l4p + 0.5 + s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3, LL, l4p, (-two_m3 + 1) / 2, MM,
                                       (two_m4 + 1) / 2);
        double tmp4 = s4 * sqrt(l3 + 0.5 - s3 * two_m3 / 2.0) *
                      sqrt(l4p + 0.5 - s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3, LL, l4p, (-two_m3 - 1) / 2, MM,
                                       (two_m4 - 1) / 2);
        double tmp5 = s1 * s2 * sqrt(l1 + 0.5 + s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 - s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 + 1) / 2, -MM,
                                       (two_m2 - 1) / 2);
        double tmp6 = sqrt(l1 + 0.5 - s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 + s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 - 1) / 2, -MM,
                                       (two_m2 + 1) / 2);
        double tmp7 = s3 * s4 * sqrt(l3 + 0.5 + s3 * two_m3 / 2.0) *
                      sqrt(l4p + 0.5 - s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3, LL, l4p, (-two_m3 + 1) / 2, MM,
                                       (two_m4 - 1) / 2);
        double tmp8 = sqrt(l3 + 0.5 - s3 * two_m3 / 2.0) *
                      sqrt(l4p + 0.5 + s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3, LL, l4p, (-two_m3 - 1) / 2, MM,
                                       (two_m4 + 1) / 2);
        angular += pow(-1, MM) * (2.0 * tmp1 * tmp4 + 2.0 * tmp2 * tmp3 +
                                  (-tmp5 + tmp6) * (-tmp7 + tmp8));
    }
    return angular * pow(-1, (two_m1 + two_m3) / 2 + 1) *
           CG_wigner_3j_zeroM(l1, LL, l2p) * CG_wigner_3j_zeroM(l3, LL, l4p);
}
double int_sph_int2e_get_angular_gaunt_LSSL(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL)
{
    double angular = 0.0;
    int l2p = l2 + s2, l3p = l3 + s3;
    for (int MM = -LL; MM <= LL; MM++)
    {
        double tmp1 = s1 * sqrt(l1 + 0.5 + s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 + s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 + 1) / 2, -MM,
                                       (two_m2 + 1) / 2);
        double tmp2 = s2 * sqrt(l1 + 0.5 - s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 - s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 - 1) / 2, -MM,
                                       (two_m2 - 1) / 2);
        double tmp3 = s3 * sqrt(l3p + 0.5 - s3 * two_m3 / 2.0) *
                      sqrt(l4 + 0.5 - s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3p, LL, l4, (-two_m3 + 1) / 2, MM,
                                       (two_m4 + 1) / 2);
        double tmp4 = s4 * sqrt(l3p + 0.5 + s3 * two_m3 / 2.0) *
                      sqrt(l4 + 0.5 + s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3p, LL, l4, (-two_m3 - 1) / 2, MM,
                                       (two_m4 - 1) / 2);
        double tmp5 = s1 * s2 * sqrt(l1 + 0.5 + s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 - s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 + 1) / 2, -MM,
                                       (two_m2 - 1) / 2);
        double tmp6 = sqrt(l1 + 0.5 - s1 * two_m1 / 2.0) *
                      sqrt(l2p + 0.5 + s2 * two_m2 / 2.0) *
                      CG_wigner_3j_int(l1, LL, l2p, (-two_m1 - 1) / 2, -MM,
                                       (two_m2 + 1) / 2);
        double tmp7 = s3 * s4 * sqrt(l3p + 0.5 - s3 * two_m3 / 2.0) *
                      sqrt(l4 + 0.5 + s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3p, LL, l4, (-two_m3 + 1) / 2, MM,
                                       (two_m4 - 1) / 2);
        double tmp8 = sqrt(l3p + 0.5 + s3 * two_m3 / 2.0) *
                      sqrt(l4 + 0.5 - s4 * two_m4 / 2.0) *
                      CG_wigner_3j_int(l3p, LL, l4, (-two_m3 - 1) / 2, MM,
                                       (two_m4 + 1) / 2);
        angular += pow(-1, MM) * (2.0 * tmp1 * tmp4 + 2.0 * tmp2 * tmp3 +
                                  (-tmp5 + tmp6) * (tmp7 - tmp8));
    }
    return angular * pow(-1, (two_m1 + two_m3) / 2) *
           CG_wigner_3j_zeroM(l1, LL, l2p) * CG_wigner_3j_zeroM(l3p, LL, l4);
}
/*
    Another implementation using wigner-9j symbol
    A little bit slower than the direct expansion but much simpler in
    formulations
*/
double int_sph_int2e_get_angular_gaunt_LSLS_9j(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int l3, int two_m3, int s3,
                                               int l4, int two_m4, int s4,
                                               int LL)
{
    double angular = 0.0;
    int l2p = l2 + s2, l4p = l4 + s4;
    int two_j1 = 2 * l1 + s1, two_j2 = 2 * l2 + s2, two_j3 = 2 * l3 + s3,
        two_j4 = 2 * l4 + s4, vv = LL;
    double threeJ1 = CG_wigner_3j_zeroM(l2p, vv, l1),
           threeJ2 = CG_wigner_3j_zeroM(l3, vv, l4p), tmp;
    for (int LLL = abs(LL - 1); LLL <= LL + 1; LLL++)
    {
        tmp = 0.0;
        double rme1 =
            int_sph_int2e_get_angularX_RME(two_j2, l2p, two_j1, l1, LLL, vv,
                                           threeJ1);
        double rme2 =
            int_sph_int2e_get_angularX_RME(two_j3, l3, two_j4, l4p, LLL, vv,
                                           threeJ2);
        for (int MMM = -LLL; MMM <= LLL; MMM++)
        {
            tmp += CG_wigner_3j(two_j2, 2 * LLL, two_j1, -two_m2, 2 * MMM,
                                two_m1) *
                   CG_wigner_3j(two_j3, 2 * LLL, two_j4, -two_m3, 2 * MMM,
                                two_m4);
        }
        angular += tmp * rme1 * rme2;
    }

    return angular * pow(-1, (two_j2 + two_j3 - two_m2 - two_m3) / 2);
}
double int_sph_int2e_get_angular_gaunt_LSSL_9j(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int l3, int two_m3, int s3,
                                               int l4, int two_m4, int s4,
                                               int LL)
{
    double angular = 0.0;
    int l2p = l2 + s2, l3p = l3 + s3;
    int two_j1 = 2 * l1 + s1, two_j2 = 2 * l2 + s2, two_j3 = 2 * l3 + s3,
        two_j4 = 2 * l4 + s4, vv = LL;
    double threeJ1 = CG_wigner_3j_zeroM(l2p, vv, l1),
           threeJ2 = CG_wigner_3j_zeroM(l3p, vv, l4), tmp;
    for (int LLL = abs(LL - 1); LLL <= LL + 1; LLL++)
    {
        tmp = 0.0;
        double rme1 =
            int_sph_int2e_get_angularX_RME(two_j2, l2p, two_j1, l1, LLL, vv,
                                           threeJ1);
        double rme2 =
            int_sph_int2e_get_angularX_RME(two_j3, l3p, two_j4, l4, LLL, vv,
                                           threeJ2);
        for (int MMM = -LLL; MMM <= LLL; MMM++)
        {
            tmp += CG_wigner_3j(two_j2, 2 * LLL, two_j1, -two_m2, 2 * MMM,
                                two_m1) *
                   CG_wigner_3j(two_j3, 2 * LLL, two_j4, -two_m3, 2 * MMM,
                                two_m4);
        }
        angular += tmp * rme1 * rme2;
    }

    return angular * pow(-1, (two_j2 + two_j3 - two_m2 - two_m3) / 2);
}

/*
    Evaluate intermediates for angular part of spin-free Gaunt operator
*/
double int_sph_int2e_get_angular_gauntSF_p1_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1, LL, l2 - 1);
    double tmp = 0.0;
    if (abs((two_m2 + 1) / 2) <= l2 - 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_p1(l2, (two_m2 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 - 1) / 2, LL, MM, l2 - 1,
                                         (two_m2 + 1) / 2, threeJ);
    if (abs((two_m2 + 3) / 2) <= l2 - 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_p1(l2, (two_m2 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 + 1) / 2, LL, MM, l2 - 1,
                                         (two_m2 + 3) / 2, threeJ);
    /* return tmp/(2.0*l2+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_p2_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1, LL, l2 + 1);
    double tmp = 0.0;
    if (abs((two_m2 + 1) / 2) <= l2 + 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_p2(l2, (two_m2 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 - 1) / 2, LL, MM, l2 + 1,
                                         (two_m2 + 1) / 2, threeJ);
    if (abs((two_m2 + 3) / 2) <= l2 + 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_p2(l2, (two_m2 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 + 1) / 2, LL, MM, l2 + 1,
                                         (two_m2 + 3) / 2, threeJ);
    /* return tmp/(2.0*l2+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_m1_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1, LL, l2 - 1);
    double tmp = 0.0;
    if (abs((two_m2 - 3) / 2) <= l2 - 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_m1(l2, (two_m2 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 - 1) / 2, LL, MM, l2 - 1,
                                         (two_m2 - 3) / 2, threeJ);
    if (abs((two_m2 - 1) / 2) <= l2 - 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_m1(l2, (two_m2 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 + 1) / 2, LL, MM, l2 - 1,
                                         (two_m2 - 1) / 2, threeJ);
    /* return tmp/(2.0*l2+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_m2_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1, LL, l2 + 1);
    double tmp = 0.0;
    if (abs((two_m2 - 3) / 2) <= l2 + 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_m2(l2, (two_m2 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 - 1) / 2, LL, MM, l2 + 1,
                                         (two_m2 - 3) / 2, threeJ);
    if (abs((two_m2 - 1) / 2) <= l2 + 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_m2(l2, (two_m2 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 + 1) / 2, LL, MM, l2 + 1,
                                         (two_m2 - 1) / 2, threeJ);
    /* return tmp/(2.0*l2+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_z1_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1, LL, l2 - 1);
    double tmp = 0.0;
    if (abs((two_m2 - 1) / 2) <= l2 - 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_z1(l2, (two_m2 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 - 1) / 2, LL, MM, l2 - 1,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m2 + 1) / 2) <= l2 - 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_z1(l2, (two_m2 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 + 1) / 2, LL, MM, l2 - 1,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l2+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_z2_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1, LL, l2 + 1);
    double tmp = 0.0;
    if (abs((two_m2 - 1) / 2) <= l2 + 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_z2(l2, (two_m2 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 - 1) / 2, LL, MM, l2 + 1,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m2 + 1) / 2) <= l2 + 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_z2(l2, (two_m2 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1, (two_m1 + 1) / 2, LL, MM, l2 + 1,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l2+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}

double int_sph_int2e_get_angular_gauntSF_p1_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1 - 1, LL, l2);
    double tmp = 0.0;
    if (abs((two_m1 + 1) / 2) <= l1 - 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_p1(l1, (two_m1 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1 - 1, (two_m1 + 1) / 2, LL, MM, l2,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m1 + 3) / 2) <= l1 - 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_p1(l1, (two_m1 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1 - 1, (two_m1 + 3) / 2, LL, MM, l2,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l1+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_p2_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1 + 1, LL, l2);
    double tmp = 0.0;
    if (abs((two_m1 + 1) / 2) <= l1 + 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_p2(l1, (two_m1 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1 + 1, (two_m1 + 1) / 2, LL, MM, l2,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m1 + 3) / 2) <= l1 + 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_p2(l1, (two_m1 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1 + 1, (two_m1 + 3) / 2, LL, MM, l2,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l1+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_m1_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1 - 1, LL, l2);
    double tmp = 0.0;
    if (abs((two_m1 - 3) / 2) <= l1 - 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_m1(l1, (two_m1 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1 - 1, (two_m1 - 3) / 2, LL, MM, l2,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m1 - 1) / 2) <= l1 - 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_m1(l1, (two_m1 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1 - 1, (two_m1 - 1) / 2, LL, MM, l2,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l1+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_m2_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1 + 1, LL, l2);
    double tmp = 0.0;
    if (abs((two_m1 - 3) / 2) <= l1 + 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_m2(l1, (two_m1 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1 + 1, (two_m1 - 3) / 2, LL, MM, l2,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m1 - 1) / 2) <= l1 + 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_m2(l1, (two_m1 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1 + 1, (two_m1 - 1) / 2, LL, MM, l2,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l1+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_z1_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1 - 1, LL, l2);
    double tmp = 0.0;
    if (abs((two_m1 - 1) / 2) <= l1 - 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_z1(l1, (two_m1 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1 - 1, (two_m1 - 1) / 2, LL, MM, l2,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m1 + 1) / 2) <= l1 - 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_z1(l1, (two_m1 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1 - 1, (two_m1 + 1) / 2, LL, MM, l2,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l1+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}
double int_sph_int2e_get_angular_gauntSF_z2_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM)
{
    double threeJ = CG_wigner_3j_zeroM(l1 + 1, LL, l2);
    double tmp = 0.0;
    if (abs((two_m1 - 1) / 2) <= l1 + 1)
        tmp += s1 * s2 *
               sqrt((l1 + 0.5 + s1 * two_m1 / 2.0) *
                    (l2 + 0.5 + s2 * two_m2 / 2.0)) *
               factor_z2(l1, (two_m1 - 1) / 2) *
               int_sph_int2e_get_threeSH(l1 + 1, (two_m1 - 1) / 2, LL, MM, l2,
                                         (two_m2 - 1) / 2, threeJ);
    if (abs((two_m1 + 1) / 2) <= l1 + 1)
        tmp += sqrt((l1 + 0.5 - s1 * two_m1 / 2.0) *
                    (l2 + 0.5 - s2 * two_m2 / 2.0)) *
               factor_z2(l1, (two_m1 + 1) / 2) *
               int_sph_int2e_get_threeSH(l1 + 1, (two_m1 + 1) / 2, LL, MM, l2,
                                         (two_m2 + 1) / 2, threeJ);
    /* return tmp/(2.0*l1+1.0); */
    return tmp / sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0));
}

/*
    Evaluate angular part of spin-free Gaunt operator together in one
    subroutine
*/
void int_sph_int2e_get_angular_gauntSF_LSLS(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL,
                                            double* lsls11, double* lsls12,
                                            double* lsls21, double* lsls22)
{
    for (int mm = -LL; mm <= LL; mm++)
    {
        double p1_12 = int_sph_int2e_get_angular_gauntSF_p1_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double p1_34 = int_sph_int2e_get_angular_gauntSF_p1_LS(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double m1_12 = int_sph_int2e_get_angular_gauntSF_m1_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double m1_34 = int_sph_int2e_get_angular_gauntSF_m1_LS(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double z1_12 = int_sph_int2e_get_angular_gauntSF_z1_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double z1_34 = int_sph_int2e_get_angular_gauntSF_z1_LS(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double p2_12 = int_sph_int2e_get_angular_gauntSF_p2_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double p2_34 = int_sph_int2e_get_angular_gauntSF_p2_LS(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double m2_12 = int_sph_int2e_get_angular_gauntSF_m2_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double m2_34 = int_sph_int2e_get_angular_gauntSF_m2_LS(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double z2_12 = int_sph_int2e_get_angular_gauntSF_z2_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double z2_34 = int_sph_int2e_get_angular_gauntSF_z2_LS(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        *lsls11 +=
            pow(-1, mm) * (0.5 * (p1_12 * m1_34 + m1_12 * p1_34) +
                           z1_12 * z1_34);
        *lsls12 +=
            pow(-1, mm) * (0.5 * (p1_12 * m2_34 + m1_12 * p2_34) +
                           z1_12 * z2_34);
        *lsls21 +=
            pow(-1, mm) * (0.5 * (p2_12 * m1_34 + m2_12 * p1_34) +
                           z2_12 * z1_34);
        *lsls22 +=
            pow(-1, mm) * (0.5 * (p2_12 * m2_34 + m2_12 * p2_34) +
                           z2_12 * z2_34);
    }
    return;
}
void int_sph_int2e_get_angular_gauntSF_LSSL(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL,
                                            double* lssl11, double* lssl12,
                                            double* lssl21, double* lssl22)
{
    for (int mm = -LL; mm <= LL; mm++)
    {
        double p1_12 = int_sph_int2e_get_angular_gauntSF_p1_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double m1_12 = int_sph_int2e_get_angular_gauntSF_m1_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double z1_12 = int_sph_int2e_get_angular_gauntSF_z1_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double p2_12 = int_sph_int2e_get_angular_gauntSF_p2_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double m2_12 = int_sph_int2e_get_angular_gauntSF_m2_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double z2_12 = int_sph_int2e_get_angular_gauntSF_z2_LS(
            l1, two_m1, s1, l2, two_m2, s2, LL, -mm);
        double p1_34 = int_sph_int2e_get_angular_gauntSF_p1_SL(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double p2_34 = int_sph_int2e_get_angular_gauntSF_p2_SL(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double m1_34 = int_sph_int2e_get_angular_gauntSF_m1_SL(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double m2_34 = int_sph_int2e_get_angular_gauntSF_m2_SL(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double z1_34 = int_sph_int2e_get_angular_gauntSF_z1_SL(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        double z2_34 = int_sph_int2e_get_angular_gauntSF_z2_SL(
            l3, two_m3, s3, l4, two_m4, s4, LL, mm);
        *lssl11 +=
            pow(-1, mm) * (0.5 * (p1_12 * p1_34 + m1_12 * m1_34) +
                           z1_12 * z1_34);
        *lssl12 +=
            pow(-1, mm) * (0.5 * (p1_12 * p2_34 + m1_12 * m2_34) +
                           z1_12 * z2_34);
        *lssl21 +=
            pow(-1, mm) * (0.5 * (p2_12 * p1_34 + m2_12 * m1_34) +
                           z2_12 * z1_34);
        *lssl22 +=
            pow(-1, mm) * (0.5 * (p2_12 * p2_34 + m2_12 * m2_34) +
                           z2_12 * z2_34);
    }
    return;
}

/*
    Evaluate different two-electron Coulomb and Exchange integral in
    2-spinor basis
*/
int2eJK int_sph_get_h2e_JK_gaunt_compact(const INT_SPH* self,
                                         const char* intType, int occMaxL)
{
    int occMaxShell = 0, Nirrep_compact = 0;
    if (occMaxL == -1)
        occMaxShell = self->size_shell;
    else
    {
        for (int ii = 0; ii < self->size_shell; ii++)
        {
            if (self->shell_list[ii].l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    for (int ii = 0; ii < occMaxShell; ii++)
    {
        if (self->shell_list[ii].l == 0)
            Nirrep_compact += 1;
        else
            Nirrep_compact += 2;
    }

    int2eJK int_2e_JK;
    int_2e_JK.J =
        (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    int_2e_JK.K =
        (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    for (int ii = 0; ii < Nirrep_compact; ii++)
    {
        int_2e_JK.J[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        int_2e_JK.K[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
    }

    int int_tmp1_p = 0;
    for (int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = self->shell_list[pshell].l, int_tmp1_q = 0;
    for (int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = self->shell_list[qshell].l, l_max = MAX(l_p, l_q), LmaxJ = 1,
            LmaxK = l_p + l_q + 1;
        (void)l_max;
        int size_gtos_p = self->shell_list[pshell].coeff.rows,
            size_gtos_q = self->shell_list[qshell].coeff.rows;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_angular_J[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K[LmaxK + 1][size_tmp_p][size_tmp_q];
        double (*radial_2e_list_J)[LmaxJ + 1][4] =
            malloc(sizeof(double[LmaxJ + 1][4]) *
                   (size_t)(size_gtos_p * size_gtos_p * size_gtos_q *
                            size_gtos_q));
        double (*radial_2e_list_K)[LmaxK + 1][4] =
            malloc(sizeof(double[LmaxK + 1][4]) *
                   (size_t)(size_gtos_p * size_gtos_p * size_gtos_q *
                            size_gtos_q));

        #pragma omp parallel  for
        for (int tt = 0;
             tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q; tt++)
        {
            int e1J = tt / (size_gtos_q * size_gtos_q);
            int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
            int ii = e1J / size_gtos_p, jj = e1J - ii * size_gtos_p;
            int kk = e2J / size_gtos_q, ll = e2J - kk * size_gtos_q;
            int e1K = ii * size_gtos_q + ll, e2K = kk * size_gtos_p + jj;
            (void)e1K;
            (void)e2K;
            double a_i_J = self->shell_list[pshell].exp_a.d[ii],
                   a_j_J = self->shell_list[pshell].exp_a.d[jj],
                   a_k_J = self->shell_list[qshell].exp_a.d[kk],
                   a_l_J = self->shell_list[qshell].exp_a.d[ll];
            double a_i_K = self->shell_list[pshell].exp_a.d[ii],
                   a_j_K = self->shell_list[qshell].exp_a.d[ll],
                   a_k_K = self->shell_list[qshell].exp_a.d[kk],
                   a_l_K = self->shell_list[pshell].exp_a.d[jj];

            if (strncmp(intType, "LSLS", 4) == 0)
            {
                for (int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J,
                        LL);
                    if (l_p != 0)
                        radial_2e_list_J[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q + 1,
                            a_l_J, LL);
                    if (l_q != 0)
                        radial_2e_list_J[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q - 1,
                            a_l_J, LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_J[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q - 1,
                            a_l_J, LL);
                }
                for (int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K,
                        LL);
                    if (l_q != 0)
                        radial_2e_list_K[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p + 1,
                            a_l_K, LL);
                    if (l_p != 0)
                        radial_2e_list_K[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p - 1,
                            a_l_K, LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_K[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p - 1,
                            a_l_K, LL);
                }
            }
            else if (strncmp(intType, "LSSL", 4) == 0)
            {
                for (int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J,
                        LL);
                    if (l_p != 0)
                        radial_2e_list_J[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J, l_q,
                            a_l_J, LL);
                    if (l_q != 0)
                        radial_2e_list_J[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J, l_q,
                            a_l_J, LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_J[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J, l_q,
                            a_l_J, LL);
                }
                for (int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K,
                        LL);
                    if (l_q != 0)
                    {
                        radial_2e_list_K[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K, l_p,
                            a_l_K, LL);
                        radial_2e_list_K[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K, l_p,
                            a_l_K, LL);
                        radial_2e_list_K[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K, l_p,
                            a_l_K, LL);
                    }
                }
            }
            else
            {
                printf("ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                exit(99);
            }
        }

        for (int twojj_p = abs(2 * l_p - 1); twojj_p <= 2 * l_p + 1;
             twojj_p = twojj_p + 2)
        for (int twojj_q = abs(2 * l_q - 1); twojj_q <= 2 * l_q + 1;
             twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2 * l_p, sym_aq = twojj_q - 2 * l_q;
            int int_tmp2_p = (twojj_p - abs(2 * l_p - 1)) / 2,
                int_tmp2_q = (twojj_q - abs(2 * l_q - 1)) / 2;
            int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_p) *
                                 sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_p; iii++)
                int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [iii] = (double*)malloc(
                               (size_t)(size_gtos_q * size_gtos_q) *
                               sizeof(double));
            int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_q) *
                                 sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_q; iii++)
                int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [iii] = (double*)malloc(
                               (size_t)(size_gtos_p * size_gtos_q) *
                               sizeof(double));

            /* Angular */
            for (int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp_d += int_sph_int2e_get_angular_gaunt_LSLS(
                            l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q,
                            2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q,
                            sym_aq, tmp);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp_d += int_sph_int2e_get_angular_gaunt_LSSL(
                            l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q,
                            2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q,
                            sym_aq, tmp);
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp_d += int_sph_int2e_get_angular_gaunt_LSLS(
                            l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q,
                            sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p,
                            twojj_p, sym_ap, tmp);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp_d += int_sph_int2e_get_angular_gaunt_LSSL(
                            l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q,
                            sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p,
                            twojj_p, sym_ap, tmp);
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }

            /* Radial */
            double k_p = -(twojj_p + 1.0) * sym_ap / 2.0,
                   k_q = -(twojj_q + 1.0) * sym_aq / 2.0;
            #pragma omp parallel  for
            for (int tt = 0;
                 tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q;
                 tt++)
            {
                double radial_J, radial_K;
                int e1J = tt / (size_gtos_q * size_gtos_q);
                int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
                int ii = e1J / size_gtos_p, jj = e1J - ii * size_gtos_p;
                int kk = e2J / size_gtos_q, ll = e2J - kk * size_gtos_q;
                int e1K = ii * size_gtos_q + ll, e2K = kk * size_gtos_p + jj;
                double norm_J = self->shell_list[pshell].norm.d[ii] *
                                self->shell_list[pshell].norm.d[jj] *
                                self->shell_list[qshell].norm.d[kk] *
                                self->shell_list[qshell].norm.d[ll],
                       norm_K = self->shell_list[pshell].norm.d[ii] *
                                self->shell_list[qshell].norm.d[ll] *
                                self->shell_list[qshell].norm.d[kk] *
                                self->shell_list[pshell].norm.d[jj];
                double lk1 = 1 + l_p + k_p, lk2 = 1 + l_p + k_p,
                       lk3 = 1 + l_q + k_q, lk4 = 1 + l_q + k_q,
                       a1 = self->shell_list[pshell].exp_a.d[ii],
                       a2 = self->shell_list[pshell].exp_a.d[jj],
                       a3 = self->shell_list[qshell].exp_a.d[kk],
                       a4 = self->shell_list[qshell].exp_a.d[ll];
                int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1K][e2K] = 0.0;

                for (int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial_J =
                            int_sph_get_radial_LSLS_J(
                                l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2, lk3,
                                lk4, radial_2e_list_J[tt][tmp], false) /
                            norm_J / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial_J =
                            -int_sph_get_radial_LSSL_J(
                                l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2, lk3,
                                lk4, radial_2e_list_J[tt][tmp], false) /
                            norm_J / 4.0 / pow(speedOfLight, 2);
                    }
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                    int_2e_JK.J[int_tmp1_p + int_tmp2_p]
                               [int_tmp1_q + int_tmp2_q][e1J][e2J] +=
                        radial_J * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1 + l_q + k_q;
                lk4 = 1 + l_p + k_p;
                a2 = self->shell_list[qshell].exp_a.d[ll];
                a4 = self->shell_list[pshell].exp_a.d[jj];
                for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial_K =
                            int_sph_get_radial_LSLS_K(
                                l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2, lk3,
                                lk4, radial_2e_list_K[tt][tmp], false) /
                            norm_K / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial_K =
                            -int_sph_get_radial_LSSL_K(
                                l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2, lk3,
                                lk4, radial_2e_list_K[tt][tmp], false) /
                            norm_K / 4.0 / pow(speedOfLight, 2);
                    }
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p]
                               [int_tmp1_q + int_tmp2_q][e1K][e2K] +=
                        radial_K * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        free(radial_2e_list_J);
        free(radial_2e_list_K);
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return int_2e_JK;
}
int2eJK int_sph_get_h2e_JK_gauntSF_compact(INT_SPH* self, const char* intType,
                                           int occMaxL)
{
    int occMaxShell = 0, Nirrep_compact = 0;
    if (occMaxL == -1)
        occMaxShell = self->size_shell;
    else
    {
        for (int ii = 0; ii < self->size_shell; ii++)
        {
            if (self->shell_list[ii].l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    for (int ii = 0; ii < occMaxShell; ii++)
    {
        if (self->shell_list[ii].l == 0)
            Nirrep_compact += 1;
        else
            Nirrep_compact += 2;
    }

    int2eJK int_2e_JK;
    int_2e_JK.J =
        (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    int_2e_JK.K =
        (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    for (int ii = 0; ii < Nirrep_compact; ii++)
    {
        int_2e_JK.J[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        int_2e_JK.K[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
    }

    int int_tmp1_p = 0;
    for (int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = self->shell_list[pshell].l, int_tmp1_q = 0;
    for (int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = self->shell_list[qshell].l, l_max = MAX(l_p, l_q), LmaxJ = 1,
            LmaxK = l_p + l_q + 1;
        (void)l_max;
        int size_gtos_p = self->shell_list[pshell].coeff.rows,
            size_gtos_q = self->shell_list[qshell].coeff.rows;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        const size_t nJ = (size_t)(size_gtos_p * size_gtos_p) *
                          (size_t)(size_gtos_q * size_gtos_q) *
                          (size_t)size_tmp_p * (size_t)size_tmp_q;
        const size_t nK = (size_t)(size_gtos_p * size_gtos_q) *
                          (size_t)(size_gtos_q * size_gtos_p) *
                          (size_t)size_tmp_p * (size_t)size_tmp_q;
        typedef double arrJ_t[size_gtos_p * size_gtos_p]
                             [size_gtos_q * size_gtos_q][size_tmp_p]
                             [size_tmp_q];
        typedef double arrK_t[size_gtos_p * size_gtos_q]
                             [size_gtos_q * size_gtos_p][size_tmp_p]
                             [size_tmp_q];
        arrJ_t* array_radial_J11 =
            (arrJ_t*)malloc((size_t)(LmaxJ + 1) * nJ * sizeof(double));
        arrK_t* array_radial_K11 =
            (arrK_t*)malloc((size_t)(LmaxK + 1) * nK * sizeof(double));
        arrJ_t* array_radial_J12 =
            (arrJ_t*)malloc((size_t)(LmaxJ + 1) * nJ * sizeof(double));
        arrK_t* array_radial_K12 =
            (arrK_t*)malloc((size_t)(LmaxK + 1) * nK * sizeof(double));
        arrJ_t* array_radial_J21 =
            (arrJ_t*)malloc((size_t)(LmaxJ + 1) * nJ * sizeof(double));
        arrK_t* array_radial_K21 =
            (arrK_t*)malloc((size_t)(LmaxK + 1) * nK * sizeof(double));
        arrJ_t* array_radial_J22 =
            (arrJ_t*)malloc((size_t)(LmaxJ + 1) * nJ * sizeof(double));
        arrK_t* array_radial_K22 =
            (arrK_t*)malloc((size_t)(LmaxK + 1) * nK * sizeof(double));
        double array_angular_J11[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K11[LmaxK + 1][size_tmp_p][size_tmp_q];
        double array_angular_J12[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K12[LmaxK + 1][size_tmp_p][size_tmp_q];
        double array_angular_J21[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K21[LmaxK + 1][size_tmp_p][size_tmp_q];
        double array_angular_J22[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K22[LmaxK + 1][size_tmp_p][size_tmp_q];
        double (*radial_2e_list_J)[LmaxJ + 1][4] =
            malloc(sizeof(double[LmaxJ + 1][4]) *
                   (size_t)(size_gtos_p * size_gtos_p * size_gtos_q *
                            size_gtos_q));
        double (*radial_2e_list_K)[LmaxK + 1][4] =
            malloc(sizeof(double[LmaxK + 1][4]) *
                   (size_t)(size_gtos_p * size_gtos_p * size_gtos_q *
                            size_gtos_q));

        #pragma omp parallel  for
        for (int tt = 0;
             tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q; tt++)
        {
            int e1J = tt / (size_gtos_q * size_gtos_q);
            int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
            int ii = e1J / size_gtos_p, jj = e1J - ii * size_gtos_p;
            int kk = e2J / size_gtos_q, ll = e2J - kk * size_gtos_q;
            int e1K = ii * size_gtos_q + ll, e2K = kk * size_gtos_p + jj;
            (void)e1K;
            (void)e2K;
            double a_i_J = self->shell_list[pshell].exp_a.d[ii],
                   a_j_J = self->shell_list[pshell].exp_a.d[jj],
                   a_k_J = self->shell_list[qshell].exp_a.d[kk],
                   a_l_J = self->shell_list[qshell].exp_a.d[ll];
            double a_i_K = self->shell_list[pshell].exp_a.d[ii],
                   a_j_K = self->shell_list[qshell].exp_a.d[ll],
                   a_k_K = self->shell_list[qshell].exp_a.d[kk],
                   a_l_K = self->shell_list[pshell].exp_a.d[jj];

            if (strncmp(intType, "LSLS", 4) == 0)
            {
                for (int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J,
                        LL);
                    if (l_p != 0)
                        radial_2e_list_J[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q + 1,
                            a_l_J, LL);
                    if (l_q != 0)
                        radial_2e_list_J[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q - 1,
                            a_l_J, LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_J[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q - 1,
                            a_l_J, LL);
                }
                for (int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K,
                        LL);
                    if (l_q != 0)
                        radial_2e_list_K[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p + 1,
                            a_l_K, LL);
                    if (l_p != 0)
                        radial_2e_list_K[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p - 1,
                            a_l_K, LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_K[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p - 1,
                            a_l_K, LL);
                }
            }
            else if (strncmp(intType, "LSSL", 4) == 0)
            {
                for (int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J,
                        LL);
                    if (l_p != 0)
                        radial_2e_list_J[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J, l_q,
                            a_l_J, LL);
                    if (l_q != 0)
                        radial_2e_list_J[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J, l_q,
                            a_l_J, LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_J[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J, l_q,
                            a_l_J, LL);
                }
                for (int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][0] = int_sph_int2e_get_radial(
                        l_p, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K,
                        LL);
                    if (l_q != 0)
                    {
                        radial_2e_list_K[tt][LL][1] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K, l_p,
                            a_l_K, LL);
                        radial_2e_list_K[tt][LL][2] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K, l_p,
                            a_l_K, LL);
                        radial_2e_list_K[tt][LL][3] = int_sph_int2e_get_radial(
                            l_p, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K, l_p,
                            a_l_K, LL);
                    }
                }
            }
            else
            {
                printf("ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                exit(99);
            }
        }

        #pragma omp parallel  for
        for (int twojj_p = abs(2 * l_p - 1); twojj_p <= 2 * l_p + 1;
             twojj_p = twojj_p + 2)
        for (int twojj_q = abs(2 * l_q - 1); twojj_q <= 2 * l_q + 1;
             twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2 * l_p, sym_aq = twojj_q - 2 * l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2 * l_p + 1 - twojj_p) / 2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2 * l_q + 1 - twojj_q) / 2 : 0;

            for (int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d11 = 0.0, tmp_d12 = 0.0, tmp_d21 = 0.0,
                       tmp_d22 = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        int_sph_int2e_get_angular_gauntSF_LSLS(
                            l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q,
                            2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q,
                            sym_aq, tmp, &tmp_d11, &tmp_d12, &tmp_d21,
                            &tmp_d22);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        int_sph_int2e_get_angular_gauntSF_LSSL(
                            l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q,
                            2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q,
                            sym_aq, tmp, &tmp_d11, &tmp_d12, &tmp_d21,
                            &tmp_d22);
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                }
                tmp_d11 /= (twojj_q + 1);
                array_angular_J11[tmp][index_tmp_p][index_tmp_q] = tmp_d11;
                tmp_d12 /= (twojj_q + 1);
                array_angular_J12[tmp][index_tmp_p][index_tmp_q] = tmp_d12;
                tmp_d21 /= (twojj_q + 1);
                array_angular_J21[tmp][index_tmp_p][index_tmp_q] = tmp_d21;
                tmp_d22 /= (twojj_q + 1);
                array_angular_J22[tmp][index_tmp_p][index_tmp_q] = tmp_d22;
            }
            for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d11 = 0.0, tmp_d12 = 0.0, tmp_d21 = 0.0,
                       tmp_d22 = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        int_sph_int2e_get_angular_gauntSF_LSLS(
                            l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q,
                            sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p,
                            twojj_p, sym_ap, tmp, &tmp_d11, &tmp_d12,
                            &tmp_d21, &tmp_d22);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        int_sph_int2e_get_angular_gauntSF_LSSL(
                            l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q,
                            sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p,
                            twojj_p, sym_ap, tmp, &tmp_d11, &tmp_d12,
                            &tmp_d21, &tmp_d22);
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                }
                tmp_d11 /= (twojj_q + 1);
                array_angular_K11[tmp][index_tmp_p][index_tmp_q] = tmp_d11;
                tmp_d12 /= (twojj_q + 1);
                array_angular_K12[tmp][index_tmp_p][index_tmp_q] = tmp_d12;
                tmp_d21 /= (twojj_q + 1);
                array_angular_K21[tmp][index_tmp_p][index_tmp_q] = tmp_d21;
                tmp_d22 /= (twojj_q + 1);
                array_angular_K22[tmp][index_tmp_p][index_tmp_q] = tmp_d22;
            }
        }

        #pragma omp parallel  for
        for (int tt = 0;
             tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q; tt++)
        {
            int e1J = tt / (size_gtos_q * size_gtos_q);
            int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
            int ii = e1J / size_gtos_p, jj = e1J - ii * size_gtos_p;
            int kk = e2J / size_gtos_q, ll = e2J - kk * size_gtos_q;
            int e1K = ii * size_gtos_q + ll, e2K = kk * size_gtos_p + jj;
            double a_i_J = self->shell_list[pshell].exp_a.d[ii],
                   a_j_J = self->shell_list[pshell].exp_a.d[jj],
                   a_k_J = self->shell_list[qshell].exp_a.d[kk],
                   a_l_J = self->shell_list[qshell].exp_a.d[ll];
            double a_i_K = self->shell_list[pshell].exp_a.d[ii],
                   a_j_K = self->shell_list[qshell].exp_a.d[ll],
                   a_k_K = self->shell_list[qshell].exp_a.d[kk],
                   a_l_K = self->shell_list[pshell].exp_a.d[jj];
            (void)a_i_J; (void)a_j_J; (void)a_k_J; (void)a_l_J;
            (void)a_i_K; (void)a_j_K; (void)a_k_K; (void)a_l_K;

            for (int twojj_p = abs(2 * l_p - 1); twojj_p <= 2 * l_p + 1;
                 twojj_p = twojj_p + 2)
            for (int twojj_q = abs(2 * l_q - 1); twojj_q <= 2 * l_q + 1;
                 twojj_q = twojj_q + 2)
            {
                int index_tmp_p =
                    (l_p > 0) ? 1 - (2 * l_p + 1 - twojj_p) / 2 : 0;
                int index_tmp_q =
                    (l_q > 0) ? 1 - (2 * l_q + 1 - twojj_q) / 2 : 0;
                int sym_ap = twojj_p - 2 * l_p, sym_aq = twojj_q - 2 * l_q;
                double k_p = -(twojj_p + 1.0) * sym_ap / 2.0,
                       k_q = -(twojj_q + 1.0) * sym_aq / 2.0;
                (void)k_p;
                (void)k_q;
                double norm_J = self->shell_list[pshell].norm.d[ii] *
                                self->shell_list[pshell].norm.d[jj] *
                                self->shell_list[qshell].norm.d[kk] *
                                self->shell_list[qshell].norm.d[ll],
                       norm_K = self->shell_list[pshell].norm.d[ii] *
                                self->shell_list[qshell].norm.d[ll] *
                                self->shell_list[qshell].norm.d[kk] *
                                self->shell_list[pshell].norm.d[jj];
                double lk1 = 1 + l_p * 2.0, lk2 = 1 + l_p * 2.0,
                       lk3 = 1 + l_q * 2.0, lk4 = 1 + l_q * 2.0,
                       a1 = self->shell_list[pshell].exp_a.d[ii],
                       a2 = self->shell_list[pshell].exp_a.d[jj],
                       a3 = self->shell_list[qshell].exp_a.d[kk],
                       a4 = self->shell_list[qshell].exp_a.d[ll];
                (void)lk1;
                (void)a1;

                for (int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_J[tt][tmp][0];
                        array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_J[tt][tmp][0];
                        array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_J[tt][tmp][0];
                        array_radial_J22[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_J[tt][tmp][0];
                        if (l_p != 0 && l_q != 0)
                        {
                            array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] +=
                                lk2 * lk4 * radial_2e_list_J[tt][tmp][3] -
                                2.0 * a4 * lk2 * radial_2e_list_J[tt][tmp][1] -
                                2.0 * a2 * lk4 * radial_2e_list_J[tt][tmp][2];
                            array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a4 * lk2 * radial_2e_list_J[tt][tmp][1];
                            array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk4 * radial_2e_list_J[tt][tmp][2];
                        }
                        else if (l_p != 0 && l_q == 0)
                        {
                            array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a4 * lk2 * radial_2e_list_J[tt][tmp][1];
                            array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a4 * lk2 * radial_2e_list_J[tt][tmp][1];
                        }

                        else if (l_p == 0 && l_q != 0)
                        {
                            array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk4 * radial_2e_list_J[tt][tmp][2];
                            array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk4 * radial_2e_list_J[tt][tmp][2];
                        }
                        array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_J * 4.0 * pow(speedOfLight, 2);
                        array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_J * 4.0 * pow(speedOfLight, 2);
                        array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_J * 4.0 * pow(speedOfLight, 2);
                        array_radial_J22[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_J[tt][tmp][0];
                        array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_J[tt][tmp][0];
                        array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_J[tt][tmp][0];
                        array_radial_J22[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_J[tt][tmp][0];
                        if (l_p != 0 && l_q != 0)
                        {
                            array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] +=
                                lk2 * lk3 * radial_2e_list_J[tt][tmp][3] -
                                2.0 * a3 * lk2 * radial_2e_list_J[tt][tmp][1] -
                                2.0 * a2 * lk3 * radial_2e_list_J[tt][tmp][2];
                            array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a3 * lk2 * radial_2e_list_J[tt][tmp][1];
                            array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk3 * radial_2e_list_J[tt][tmp][2];
                        }
                        else if (l_p != 0 && l_q == 0)
                        {
                            array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a3 * lk2 * radial_2e_list_J[tt][tmp][1];
                            array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a3 * lk2 * radial_2e_list_J[tt][tmp][1];
                        }
                        else if (l_p == 0 && l_q != 0)
                        {
                            array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk3 * radial_2e_list_J[tt][tmp][2];
                            array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk3 * radial_2e_list_J[tt][tmp][2];
                        }
                        array_radial_J11[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                        array_radial_J12[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                        array_radial_J21[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                        array_radial_J22[tmp][e1J][e2J][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                }
                lk2 = 1 + l_q * 2.0;
                lk4 = 1 + l_p * 2.0;
                a2 = self->shell_list[qshell].exp_a.d[ll];
                a4 = self->shell_list[pshell].exp_a.d[jj];
                for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_K[tt][tmp][0];
                        array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_K[tt][tmp][0];
                        array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_K[tt][tmp][0];
                        array_radial_K22[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a4 * radial_2e_list_K[tt][tmp][0];
                        if (l_p != 0 && l_q != 0)
                        {
                            array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] +=
                                lk2 * lk4 * radial_2e_list_K[tt][tmp][3] -
                                2.0 * a4 * lk2 * radial_2e_list_K[tt][tmp][1] -
                                2.0 * a2 * lk4 * radial_2e_list_K[tt][tmp][2];
                            array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a4 * lk2 * radial_2e_list_K[tt][tmp][1];
                            array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk4 * radial_2e_list_K[tt][tmp][2];
                        }
                        else if (l_p == 0 && l_q != 0)
                        {
                            array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a4 * lk2 * radial_2e_list_K[tt][tmp][1];
                            array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a4 * lk2 * radial_2e_list_K[tt][tmp][1];
                        }
                        else if (l_p != 0 && l_q == 0)
                        {
                            array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk4 * radial_2e_list_K[tt][tmp][2];
                            array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk4 * radial_2e_list_K[tt][tmp][2];
                        }
                        array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_K * 4.0 * pow(speedOfLight, 2);
                        array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_K * 4.0 * pow(speedOfLight, 2);
                        array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_K * 4.0 * pow(speedOfLight, 2);
                        array_radial_K22[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_K[tt][tmp][0];
                        array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_K[tt][tmp][0];
                        array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_K[tt][tmp][0];
                        array_radial_K22[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] =
                            4.0 * a2 * a3 * radial_2e_list_K[tt][tmp][0];
                        if (l_q != 0)
                        {
                            array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] +=
                                lk2 * lk3 * radial_2e_list_K[tt][tmp][3] -
                                2.0 * a3 * lk2 * radial_2e_list_K[tt][tmp][1] -
                                2.0 * a2 * lk3 * radial_2e_list_K[tt][tmp][2];
                            array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a3 * lk2 * radial_2e_list_K[tt][tmp][1];
                            array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                            [index_tmp_q] -=
                                2.0 * a2 * lk3 * radial_2e_list_K[tt][tmp][2];
                        }
                        array_radial_K11[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                        array_radial_K12[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                        array_radial_K21[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                        array_radial_K22[tmp][e1K][e2K][index_tmp_p]
                                        [index_tmp_q] /=
                            -1.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    else
                    {
                        printf(
                            "ERROR: Unkonwn intType in get_h2e_JK_gaunt.\n");
                        exit(99);
                    }
                }
            }
        }

        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for (int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for (int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_p) *
                                 sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_p; iii++)
                int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [iii] = (double*)malloc(
                               (size_t)(size_gtos_q * size_gtos_q) *
                               sizeof(double));
            int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_q) *
                                 sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_q; iii++)
                int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [iii] = (double*)malloc(
                               (size_t)(size_gtos_p * size_gtos_q) *
                               sizeof(double));
            #pragma omp parallel  for
            for (int tt = 0;
                 tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q;
                 tt++)
            {
                int e1J = tt / (size_gtos_q * size_gtos_q);
                int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
                int e1K = tt / (size_gtos_p * size_gtos_q);
                int e2K = tt - e1K * (size_gtos_p * size_gtos_q);
                int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1K][e2K] = 0.0;
                for (int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                    int_2e_JK.J[int_tmp1_p + int_tmp2_p]
                               [int_tmp1_q + int_tmp2_q][e1J][e2J] +=
                        array_radial_J11[tmp][e1J][e2J][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_J11[tmp][int_tmp2_p][int_tmp2_q] +
                        array_radial_J12[tmp][e1J][e2J][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_J12[tmp][int_tmp2_p][int_tmp2_q] +
                        array_radial_J21[tmp][e1J][e2J][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_J21[tmp][int_tmp2_p][int_tmp2_q] +
                        array_radial_J22[tmp][e1J][e2J][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_J22[tmp][int_tmp2_p][int_tmp2_q];
                for (int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p]
                               [int_tmp1_q + int_tmp2_q][e1K][e2K] +=
                        array_radial_K11[tmp][e1K][e2K][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_K11[tmp][int_tmp2_p][int_tmp2_q] +
                        array_radial_K12[tmp][e1K][e2K][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_K12[tmp][int_tmp2_p][int_tmp2_q] +
                        array_radial_K21[tmp][e1K][e2K][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_K21[tmp][int_tmp2_p][int_tmp2_q] +
                        array_radial_K22[tmp][e1K][e2K][int_tmp2_p]
                                        [int_tmp2_q] *
                            array_angular_K22[tmp][int_tmp2_p][int_tmp2_q];
            }
        }
        free(array_radial_J11);
        free(array_radial_K11);
        free(array_radial_J12);
        free(array_radial_K12);
        free(array_radial_J21);
        free(array_radial_K21);
        free(array_radial_J22);
        free(array_radial_K22);
        free(radial_2e_list_J);
        free(radial_2e_list_K);
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return int_2e_JK;
}

void int_sph_get_h2e_JK_gaunt_direct(INT_SPH* self, int2eJK* LSLS,
                                     int2eJK* LSSL, int occMaxL, bool spinFree)
{
    if (spinFree)
    {
        *LSLS = int_sph_get_h2e_JK_gauntSF_compact(self, "LSLS", occMaxL);
        *LSSL = int_sph_get_h2e_JK_gauntSF_compact(self, "LSSL", occMaxL);
    }
    else
    {
        *LSLS = int_sph_get_h2e_JK_gaunt_compact(self, "LSLS", occMaxL);
        *LSSL = int_sph_get_h2e_JK_gaunt_compact(self, "LSSL", occMaxL);
    }

    return;
}
