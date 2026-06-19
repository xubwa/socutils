/*
 * x2c_int_sph_gauge.c -- C translation of src/int_sph_gauge.cpp (X2CAMF):
 * gauge-term radial/angular helpers and the compact/direct two-electron
 * drivers.  The unused INT_SPH::get_h2e_JK_gauge (non-compact) and the
 * get_N_coeff helper (only referenced from commented-out code) are omitted.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "x2c_int_sph.h"

/*
    evaluate radial part and angular part in 2e integrals
*/
double int_sph_int2e_get_radial_gauge(int l1, double a1, int l2, double a2,
                                      int l3, double a3, int l4, double a4,
                                      int LL, int v1, int v2)
{
    if (LL + v1 < 0 || LL + v2 < 0 || l1 + l2 + 2 < LL || l3 + l4 + 2 < LL)
        return 0.0;
    double radial = 0.0, fac = 0.0;
    if (v1 == v2)
    {
        int vv = LL + v1;
        fac = -2.0 * (2.0 * LL + 1.0) / (2.0 * vv + 1.0);
        radial =
            int_sph_int2e_get_radial(l1, a1, l2, a2, l3, a3, l4, a4, vv);
    }
    else if (v1 == -1 && v2 == +1)
    {
        fac = -(2.0 * LL + 1.0);
        if ((l1 + l2 + 3 + LL) % 2)
        {
            radial = int_sph_auxiliary_2e_0_r(l1 + l2 + 3 + LL, l3 + l4 - LL,
                                              a1 + a2, a3 + a4) -
                     int_sph_auxiliary_2e_0_r(l1 + l2 + 1 + LL,
                                              l3 + l4 + 2 - LL, a1 + a2,
                                              a3 + a4);
        }
        else
        {
            radial = int_sph_auxiliary_2e_r_inf(l3 + l4 - LL, l1 + l2 + 3 + LL,
                                                a3 + a4, a1 + a2) -
                     int_sph_auxiliary_2e_r_inf(l3 + l4 + 2 - LL,
                                                l1 + l2 + 1 + LL, a3 + a4,
                                                a1 + a2);
        }
    }
    else if (v1 == +1 && v2 == -1)
    {
        fac = -(2.0 * LL + 1.0);
        if ((l3 + l4 + 3 + LL) % 2)
        {
            radial = int_sph_auxiliary_2e_0_r(l3 + l4 + 3 + LL, l1 + l2 - LL,
                                              a3 + a4, a1 + a2) -
                     int_sph_auxiliary_2e_0_r(l3 + l4 + 1 + LL,
                                              l1 + l2 + 2 - LL, a3 + a4,
                                              a1 + a2);
        }
        else
        {
            radial = int_sph_auxiliary_2e_r_inf(l1 + l2 - LL, l3 + l4 + 3 + LL,
                                                a1 + a2, a3 + a4) -
                     int_sph_auxiliary_2e_r_inf(l1 + l2 + 2 - LL,
                                                l3 + l4 + 1 + LL, a1 + a2,
                                                a3 + a4);
        }
    }
    else
    {
        printf("WARNING: gauge radial input v1 and v2 are out of domain.\n");
    }

    return radial * fac;
}

double int_sph_int2e_get_angular_gauge_LSLS(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int ll, int v1,
                                            int v2)
{
    if (ll + v1 < 0 || ll + v2 < 0) return 0.0;
    double angular = 0.0;
    int l2p = l2 + s2, l4p = l4 + s4;
    int two_j1 = 2 * l1 + s1, two_j2 = 2 * l2 + s2, two_j3 = 2 * l3 + s3,
        two_j4 = 2 * l4 + s4;
    double threeJ1 = CG_wigner_3j(2 * l1, 2 * ll + 2 * v1, 2 * l2p, 0, 0, 0),
           threeJ2 = CG_wigner_3j(2 * l3, 2 * ll + 2 * v2, 2 * l4p, 0, 0, 0),
           tmp;
    tmp = 0.0;
    double rme1 = int_sph_int2e_get_angularX_RME(two_j1, l1, two_j2, l2p, ll,
                                                 ll + v1, threeJ1);
    double rme2 = int_sph_int2e_get_angularX_RME(two_j3, l3, two_j4, l4p, ll,
                                                 ll + v2, threeJ2);
    for (int MMM = -ll; MMM <= ll; MMM++)
    {
        tmp += pow(-1, MMM) *
               CG_wigner_3j(two_j1, 2 * ll, two_j2, -two_m1, -2 * MMM,
                            two_m2) *
               CG_wigner_3j(two_j3, 2 * ll, two_j4, -two_m3, 2 * MMM, two_m4);
    }
    angular += tmp * rme1 * rme2;

    return angular * pow(-1, (two_j1 + two_j3 - two_m1 - two_m3) / 2) *
           (2.0 * (ll + v1) + 1.0) * (2.0 * (ll + v2) + 1.0) / (2.0 * ll + 1) *
           CG_wigner_3j(2, 2 * ll + 2 * v1, 2 * ll, 0, 0, 0) *
           CG_wigner_3j(2, 2 * ll + 2 * v2, 2 * ll, 0, 0, 0);
}
double int_sph_int2e_get_angular_gauge_LSSL(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int ll, int v1,
                                            int v2)
{
    if (ll + v1 < 0 || ll + v2 < 0) return 0.0;
    double angular = 0.0;
    int l2p = l2 + s2, l3p = l3 + s3;
    int two_j1 = 2 * l1 + s1, two_j2 = 2 * l2 + s2, two_j3 = 2 * l3 + s3,
        two_j4 = 2 * l4 + s4;
    double threeJ1 = CG_wigner_3j(2 * l1, 2 * ll + 2 * v1, 2 * l2p, 0, 0, 0),
           threeJ2 = CG_wigner_3j(2 * l3p, 2 * ll + 2 * v2, 2 * l4, 0, 0, 0),
           tmp;
    tmp = 0.0;
    double rme1 = int_sph_int2e_get_angularX_RME(two_j1, l1, two_j2, l2p, ll,
                                                 ll + v1, threeJ1);
    double rme2 = int_sph_int2e_get_angularX_RME(two_j3, l3p, two_j4, l4, ll,
                                                 ll + v2, threeJ2);
    for (int MMM = -ll; MMM <= ll; MMM++)
    {
        tmp += pow(-1, MMM) *
               CG_wigner_3j(two_j1, 2 * ll, two_j2, -two_m1, -2 * MMM,
                            two_m2) *
               CG_wigner_3j(two_j3, 2 * ll, two_j4, -two_m3, 2 * MMM, two_m4);
    }
    angular += tmp * rme1 * rme2;

    return angular * pow(-1, (two_j1 + two_j3 - two_m1 - two_m3) / 2) *
           (2.0 * (ll + v1) + 1.0) * (2.0 * (ll + v2) + 1.0) / (2.0 * ll + 1) *
           CG_wigner_3j(2, 2 * ll + 2 * v1, 2 * ll, 0, 0, 0) *
           CG_wigner_3j(2, 2 * ll + 2 * v2, 2 * ll, 0, 0, 0);
}

int2eJK int_sph_get_h2e_JK_gauge_compact(const INT_SPH* self,
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
    int_2e_JK.J = (double****)malloc(Nirrep_compact * sizeof(double***));
    int_2e_JK.K = (double****)malloc(Nirrep_compact * sizeof(double***));
    for (int ii = 0; ii < Nirrep_compact; ii++)
    {
        int_2e_JK.J[ii] = (double***)malloc(Nirrep_compact * sizeof(double**));
        int_2e_JK.K[ii] = (double***)malloc(Nirrep_compact * sizeof(double**));
    }

    int int_tmp1_p = 0;
    for (int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = self->shell_list[pshell].l, int_tmp1_q = 0;
    for (int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = self->shell_list[qshell].l;
        int LmaxJ[4], LminJ[4], LmaxK[4], LminK[4];
        LmaxK[0] = l_p + l_q + 2; LminJ[0] = 1; LminK[0] = 1;
        LmaxK[1] = l_p + l_q;     LminJ[1] = 1; LminK[1] = 1;
        LmaxK[2] = l_p + l_q;     LminJ[2] = 1; LminK[2] = 1;
        LmaxK[3] = l_p + l_q;     LminJ[3] = 0; LminK[3] = 0;
        LmaxJ[0] = 1;
        LmaxJ[1] = 1;
        LmaxJ[2] = 1;
        LmaxJ[3] = 0;
        int size_gtos_p = self->shell_list[pshell].coeff.rows,
            size_gtos_q = self->shell_list[qshell].coeff.rows;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_angular_Jmm[LmaxJ[0] - LminJ[0] + 1][size_tmp_p][size_tmp_q],
               array_angular_Kmm[LmaxK[0] - LminK[0] + 1][size_tmp_p][size_tmp_q];
        double array_angular_Jmp[LmaxJ[1] - LminJ[1] + 1][size_tmp_p][size_tmp_q],
               array_angular_Kmp[LmaxK[1] - LminK[1] + 1][size_tmp_p][size_tmp_q];
        double array_angular_Jpm[LmaxJ[2] - LminJ[2] + 1][size_tmp_p][size_tmp_q],
               array_angular_Kpm[LmaxK[2] - LminK[2] + 1][size_tmp_p][size_tmp_q];
        double array_angular_Jpp[LmaxJ[3] - LminJ[3] + 1][size_tmp_p][size_tmp_q],
               array_angular_Kpp[LmaxK[3] - LminK[3] + 1][size_tmp_p][size_tmp_q];
        double radial_2e_list_Jmm[LmaxJ[0] - LminJ[0] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Kmm[LmaxK[0] - LminK[0] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Jmp[LmaxJ[1] - LminJ[1] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Kmp[LmaxK[1] - LminK[1] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Jpm[LmaxJ[2] - LminJ[2] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Kpm[LmaxK[2] - LminK[2] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Jpp[LmaxJ[3] - LminJ[3] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];
        double radial_2e_list_Kpp[LmaxK[3] - LminK[3] + 1][size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q][4];

        #pragma omp parallel  for
        for (int tt = 0; tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q; tt++)
        {
            int e1J = tt / (size_gtos_q * size_gtos_q);
            int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
            int ii = e1J / size_gtos_p, jj = e1J - ii * size_gtos_p;
            int kk = e2J / size_gtos_q, ll = e2J - kk * size_gtos_q;
            int e1K = ii * size_gtos_q + ll, e2K = kk * size_gtos_p + jj;
            (void)e1K; (void)e2K;
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
                for (int LL = LmaxJ[0] - LminJ[0]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jmm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[0], -1, -1);
                    if (l_p != 0)
                        radial_2e_list_Jmm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[0], -1, -1);
                    if (l_q != 0)
                        radial_2e_list_Jmm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[0], -1, -1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jmm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[0], -1, -1);
                }
                for (int LL = LmaxK[0] - LminK[0]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kmm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[0], -1, -1);
                    if (l_q != 0)
                        radial_2e_list_Kmm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[0], -1, -1);
                    if (l_p != 0)
                        radial_2e_list_Kmm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[0], -1, -1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Kmm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[0], -1, -1);
                }
                for (int LL = LmaxJ[1] - LminJ[1]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jmp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[1], -1, 1);
                    if (l_p != 0)
                        radial_2e_list_Jmp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[1], -1, 1);
                    if (l_q != 0)
                        radial_2e_list_Jmp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[1], -1, 1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jmp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[1], -1, 1);
                }
                for (int LL = LmaxK[1] - LminK[1]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kmp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[1], -1, 1);
                    if (l_q != 0)
                        radial_2e_list_Kmp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[1], -1, 1);
                    if (l_p != 0)
                        radial_2e_list_Kmp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[1], -1, 1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Kmp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[1], -1, 1);
                }
                for (int LL = LmaxJ[2] - LminJ[2]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jpm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[2], 1, -1);
                    if (l_p != 0)
                        radial_2e_list_Jpm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[2], 1, -1);
                    if (l_q != 0)
                        radial_2e_list_Jpm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[2], 1, -1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jpm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[2], 1, -1);
                }
                for (int LL = LmaxK[2] - LminK[2]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kpm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[2], 1, -1);
                    if (l_q != 0)
                        radial_2e_list_Kpm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[2], 1, -1);
                    if (l_p != 0)
                        radial_2e_list_Kpm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[2], 1, -1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Kpm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[2], 1, -1);
                }
                for (int LL = LmaxJ[3] - LminJ[3]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jpp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[3], 1, 1);
                    if (l_p != 0)
                        radial_2e_list_Jpp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q + 1, a_l_J, LL + LminJ[3], 1, 1);
                    if (l_q != 0)
                        radial_2e_list_Jpp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[3], 1, 1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jpp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q - 1, a_l_J, LL + LminJ[3], 1, 1);
                }
                for (int LL = LmaxK[3] - LminK[3]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kpp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[3], 1, 1);
                    if (l_q != 0)
                        radial_2e_list_Kpp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p + 1, a_l_K, LL + LminK[3], 1, 1);
                    if (l_p != 0)
                        radial_2e_list_Kpp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[3], 1, 1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Kpp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p - 1, a_l_K, LL + LminK[3], 1, 1);
                }
            }
            else if (strncmp(intType, "LSSL", 4) == 0)
            {
                for (int LL = LmaxJ[0] - LminJ[0]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jmm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[0], -1, -1);
                    if (l_p != 0)
                        radial_2e_list_Jmm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[0], -1, -1);
                    if (l_q != 0)
                        radial_2e_list_Jmm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[0], -1, -1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jmm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[0], -1, -1);
                }
                for (int LL = LmaxK[0] - LminK[0]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kmm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[0], -1, -1);
                    if (l_q != 0)
                    {
                        radial_2e_list_Kmm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[0], -1, -1);
                        radial_2e_list_Kmm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[0], -1, -1);
                        radial_2e_list_Kmm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[0], -1, -1);
                    }
                }
                for (int LL = LmaxJ[1] - LminJ[1]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jmp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[1], -1, 1);
                    if (l_p != 0)
                        radial_2e_list_Jmp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[1], -1, 1);
                    if (l_q != 0)
                        radial_2e_list_Jmp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[1], -1, 1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jmp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[1], -1, 1);
                }
                for (int LL = LmaxK[1] - LminK[1]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kmp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[1], -1, 1);
                    if (l_q != 0)
                    {
                        radial_2e_list_Kmp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[1], -1, 1);
                        radial_2e_list_Kmp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[1], -1, 1);
                        radial_2e_list_Kmp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[1], -1, 1);
                    }
                }
                for (int LL = LmaxJ[2] - LminJ[2]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jpm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[2], 1, -1);
                    if (l_p != 0)
                        radial_2e_list_Jpm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[2], 1, -1);
                    if (l_q != 0)
                        radial_2e_list_Jpm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[2], 1, -1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jpm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[2], 1, -1);
                }
                for (int LL = LmaxK[2] - LminK[2]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kpm[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[2], 1, -1);
                    if (l_q != 0)
                    {
                        radial_2e_list_Kpm[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[2], 1, -1);
                        radial_2e_list_Kpm[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[2], 1, -1);
                        radial_2e_list_Kpm[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[2], 1, -1);
                    }
                }
                for (int LL = LmaxJ[3] - LminJ[3]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Jpp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[3], 1, 1);
                    if (l_p != 0)
                        radial_2e_list_Jpp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J, l_q, a_l_J, LL + LminJ[3], 1, 1);
                    if (l_q != 0)
                        radial_2e_list_Jpp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[3], 1, 1);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_Jpp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J, l_q, a_l_J, LL + LminJ[3], 1, 1);
                }
                for (int LL = LmaxK[3] - LminK[3]; LL >= 0; LL -= 2)
                {
                    radial_2e_list_Kpp[LL][tt][0] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[3], 1, 1);
                    if (l_q != 0)
                    {
                        radial_2e_list_Kpp[LL][tt][1] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K, l_p, a_l_K, LL + LminK[3], 1, 1);
                        radial_2e_list_Kpp[LL][tt][2] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[3], 1, 1);
                        radial_2e_list_Kpp[LL][tt][3] = int_sph_int2e_get_radial_gauge(l_p, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K, l_p, a_l_K, LL + LminK[3], 1, 1);
                    }
                }
            }
            else
            {
                printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                exit(99);
            }
        }

        for (int twojj_p = abs(2 * l_p - 1); twojj_p <= 2 * l_p + 1; twojj_p = twojj_p + 2)
        for (int twojj_q = abs(2 * l_q - 1); twojj_q <= 2 * l_q + 1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2 * l_p, sym_aq = twojj_q - 2 * l_q;
            int int_tmp2_p = (twojj_p - abs(2 * l_p - 1)) / 2,
                int_tmp2_q = (twojj_q - abs(2 * l_q - 1)) / 2;
            int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc(size_gtos_p * size_gtos_p * sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_p; iii++)
                int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][iii] =
                    (double*)malloc(size_gtos_q * size_gtos_q * sizeof(double));
            int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc(size_gtos_p * size_gtos_q * sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_q; iii++)
                int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][iii] =
                    (double*)malloc(size_gtos_p * size_gtos_q * sizeof(double));

            // Angular
            for (int LL = LmaxJ[0] - LminJ[0]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[0], -1, -1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[0], -1, -1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxK[0] - LminK[0]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[0], -1, -1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[0], -1, -1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxJ[1] - LminJ[1]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[1], -1, 1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[1], -1, 1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxK[1] - LminK[1]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[1], -1, 1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[1], -1, 1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxJ[2] - LminJ[2]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[2], 1, -1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[2], 1, -1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxK[2] - LminK[2]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[2], 1, -1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[2], 1, -1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxJ[3] - LminJ[3]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[3], 1, 1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, LL + LminJ[3], 1, 1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }
            for (int LL = LmaxK[3] - LminK[3]; LL >= 0; LL -= 2)
            {
                double tmp = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if (strncmp(intType, "LSLS", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[3], 1, 1);
                    else if (strncmp(intType, "LSSL", 4) == 0)
                        tmp += int_sph_int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq, l_q, 2 * mq - twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL + LminK[3], 1, 1);
                    else
                    {
                        printf("ERROR: Unkonwn intType in get_h2e_JK_gauge.\n");
                        exit(99);
                    }
                }
                array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q + 1.0);
            }

            // Radial
            double k_p = -(twojj_p + 1.0) * sym_ap / 2.0,
                   k_q = -(twojj_q + 1.0) * sym_aq / 2.0;
            #pragma omp parallel  for
            for (int tt = 0; tt < size_gtos_p * size_gtos_p * size_gtos_q * size_gtos_q; tt++)
            {
                double radial = 0.0;
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

                int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K][e2K] = 0.0;

                for (int LL = LmaxJ[0] - LminJ[0]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_J(l_p, l_q, LL + LminJ[0], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jmm[LL][tt], false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_J(l_p, l_q, LL + LminJ[0], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jmm[LL][tt], false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J][e2J] += radial * array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q];
                }
                for (int LL = LmaxJ[1] - LminJ[1]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_J(l_p, l_q, LL + LminJ[1], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jmp[LL][tt], false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_J(l_p, l_q, LL + LminJ[1], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jmp[LL][tt], false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J][e2J] += radial * array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q];
                }
                for (int LL = LmaxJ[2] - LminJ[2]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_J(l_p, l_q, LL + LminJ[2], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jpm[LL][tt], false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_J(l_p, l_q, LL + LminJ[2], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jpm[LL][tt], false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J][e2J] += radial * array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q];
                }
                for (int LL = LmaxJ[3] - LminJ[3]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_J(l_p, l_q, LL + LminJ[3], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jpp[LL][tt], false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_J(l_p, l_q, LL + LminJ[3], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Jpp[LL][tt], false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J][e2J] += radial * array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1 + l_q + k_q; lk4 = 1 + l_p + k_p;
                a2 = self->shell_list[qshell].exp_a.d[ll];
                a4 = self->shell_list[pshell].exp_a.d[jj];
                for (int LL = LmaxK[0] - LminK[0]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_K(l_p, l_q, LL + LminK[0], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kmm[LL][tt], false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_K(l_p, l_q, LL + LminK[0], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kmm[LL][tt], false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K][e2K] += radial * array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q];
                }
                for (int LL = LmaxK[1] - LminK[1]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_K(l_p, l_q, LL + LminK[1], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kmp[LL][tt], false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_K(l_p, l_q, LL + LminK[1], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kmp[LL][tt], false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K][e2K] += radial * array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q];
                }
                for (int LL = LmaxK[2] - LminK[2]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_K(l_p, l_q, LL + LminK[2], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kpm[LL][tt], false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_K(l_p, l_q, LL + LminK[2], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kpm[LL][tt], false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K][e2K] += radial * array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q];
                }
                for (int LL = LmaxK[3] - LminK[3]; LL >= 0; LL -= 2)
                {
                    if (strcmp(intType, "LSLS") == 0)
                    {
                        radial = int_sph_get_radial_LSLS_K(l_p, l_q, LL + LminK[3], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kpp[LL][tt], false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "LSSL") == 0)
                    {
                        radial = int_sph_get_radial_LSSL_K(l_p, l_q, LL + LminK[3], a1, a2, a3, a4, lk1, lk2, lk3, lk4, radial_2e_list_Kpp[LL][tt], false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight, 2);
                    }
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K][e2K] += radial * array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return int_2e_JK;
}

void int_sph_get_h2e_JK_gauge_direct(INT_SPH* self, int2eJK* LSLS,
                                     int2eJK* LSSL, int occMaxL, bool spinFree)
{
    (void)spinFree;
    *LSLS = int_sph_get_h2e_JK_gauge_compact(self, "LSLS", occMaxL);
    *LSSL = int_sph_get_h2e_JK_gauge_compact(self, "LSSL", occMaxL);

    return;
}
