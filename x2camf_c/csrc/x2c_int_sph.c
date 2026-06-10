/*
 * x2c_int_sph.c -- C translation of src/int_sph.cpp (X2CAMF):
 * INT_SPH::get_h1e, get_h2e_JK_compact, get_h2e_JK_direct and
 * get_h2eSD_JK_direct.  (The unused generic get_h2e_JK is not translated.)
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>
#include <string.h>

#include "x2c_int_sph.h"

/*
    Evaluate different one-electron integral in 2-spinor basis
*/
vmat_t int_sph_get_h1e(const INT_SPH* self, const char* intType)
{
    vmat_t int_1e = vmat_new(self->Nirrep);
    /* isotope mass for finite nuclear calculation */
    static const double mass_tmp[118] = {
        1,   4,   7,   9,   11,  12,  14,  16,  19,  20,  23,  24,  27,  28,
        31,  32,  35,  40,  39,  40,  45,  48,  51,  52,  55,  56,  59,  58,
        63,  64,  69,  74,  75,  80,  79,  84,  85,  88,  89,  90,  93,  98,
        98,  102, 103, 106, 107, 114, 115, 120, 121, 130, 127, 132, 133, 138,
        139, 140, 141, 144, 145, 152, 153, 158, 159, 162, 162, 168, 169, 174,
        175, 180, 181, 184, 187, 192, 193, 195, 197, 202, 205, 208, 209, 209,
        210, 222, 223, 226, 227, 232, 231, 238, 237, 244, 243, 247, 247, 251,
        252, 257, 258, 259, 266, 267, 268, 269, 270, 277, 278, 281, 282, 285,
        286, 289, 290, 293, 294, 294};
    int int_tmp = 0;
    for (int irrep = 0; irrep < self->Nirrep; irrep++)
    {
        mat_assign(&int_1e.m[irrep], mat_new(self->irrep_list[irrep].size,
                                             self->irrep_list[irrep].size));
    }
    for (int ishell = 0; ishell < self->size_shell; ishell++)
    {
        int ll = self->shell_list[ishell].l;
        int size_gtos = self->shell_list[ishell].coeff.rows;
        vmat_t h1e_single_shell;
        if (ll == 0)
            h1e_single_shell = vmat_new(1);
        else
            h1e_single_shell = vmat_new(2);
        for (int ii = 0; ii < h1e_single_shell.n; ii++)
            mat_assign(&h1e_single_shell.m[ii], mat_new(size_gtos, size_gtos));

        for (int ii = 0; ii < size_gtos; ii++)
        for (int jj = 0; jj < size_gtos; jj++)
        {
            double a1 = self->shell_list[ishell].exp_a.d[ii],
                   a2 = self->shell_list[ishell].exp_a.d[jj];
            double auxiliary_1e_list[6];
            for (int mm = 0; mm <= 4; mm++)
                auxiliary_1e_list[mm] =
                    int_sph_auxiliary_1e(2 * ll + mm, a1 + a2);
            if (ll != 0)
            {
                auxiliary_1e_list[5] =
                    int_sph_auxiliary_1e(2 * ll - 1, a1 + a2);
            }
            else
            {
                auxiliary_1e_list[5] = 0.0;
            }
            for (int twojj = abs(2 * ll - 1); twojj <= 2 * ll + 1;
                 twojj = twojj + 2)
            {
                double kappa = (twojj + 1.0) * (ll - twojj / 2.0);
                int index_tmp = 1 - (2 * ll + 1 - twojj) / 2;
                if (ll == 0) index_tmp = 0;
                mat_t h1e_mat = h1e_single_shell.m[index_tmp];

                if (strcmp(intType, "s_p_nuc_s_p") == 0)
                {
                    M(h1e_mat, ii, jj) = 4 * a1 * a2 * auxiliary_1e_list[3];
                    if (ll != 0)
                        M(h1e_mat, ii, jj) +=
                            pow(ll + kappa + 1.0, 2) * auxiliary_1e_list[5] -
                            2.0 * (ll + kappa + 1.0) * (a1 + a2) *
                                auxiliary_1e_list[1];
                    M(h1e_mat, ii, jj) *= -self->atomNumber;
                }
                else if (strcmp(intType, "s_p_nuc_s_p_sf") == 0)
                {
                    M(h1e_mat, ii, jj) = 4 * a1 * a2 * auxiliary_1e_list[3];
                    if (ll != 0)
                        M(h1e_mat, ii, jj) +=
                            (2 * ll * ll + ll) * auxiliary_1e_list[5] -
                            2.0 * ll * (a1 + a2) * auxiliary_1e_list[1];
                    M(h1e_mat, ii, jj) *= -self->atomNumber;
                }
                else if (strcmp(intType, "s_p_nuc_s_p_sd") == 0)
                {
                    M(h1e_mat, ii, jj) = 0.0;
                    if (ll != 0)
                        M(h1e_mat, ii, jj) +=
                            (kappa + 1.0) * auxiliary_1e_list[5];
                    M(h1e_mat, ii, jj) *= -self->atomNumber;
                }
                else if (strcmp(intType, "s_p_s_p") == 0)
                {
                    M(h1e_mat, ii, jj) = 4 * a1 * a2 * auxiliary_1e_list[4];
                    if (ll != 0)
                        M(h1e_mat, ii, jj) +=
                            pow(ll + kappa + 1.0, 2) * auxiliary_1e_list[0] -
                            2.0 * (ll + kappa + 1.0) * (a1 + a2) *
                                auxiliary_1e_list[2];
                }
                else if (strcmp(intType, "overlap") == 0)
                    M(h1e_mat, ii, jj) = auxiliary_1e_list[2];
                else if (strcmp(intType, "nuc_attra") == 0)
                    M(h1e_mat, ii, jj) =
                        -self->atomNumber * auxiliary_1e_list[1];
                else if (strcmp(intType, "nucGau_attra") == 0)
                {
                    double a_13 =
                        pow(mass_tmp[self->atomNumber - 1], 1.0 / 3.0);
                    double rnuc = (0.836 * a_13 + 0.570) / 52917.7249,
                           xi = 3.0 / 2.0 / rnuc / rnuc;
                    double norm = -self->atomNumber * pow(xi / M_PI, 1.5);
                    M(h1e_mat, ii, jj) =
                        norm *
                        int_sph_int2e_get_radial(0, 0.0, 0, xi, ll, a1, ll, a2,
                                                 0) *
                        4.0 * M_PI;
                }
                else if (strcmp(intType, "s_p_nucGau_s_p") == 0)
                {
                    double a_13 =
                        pow(mass_tmp[self->atomNumber - 1], 1.0 / 3.0);
                    double rnuc = (0.836 * a_13 + 0.570) / 52917.7249,
                           xi = 3.0 / 2.0 / rnuc / rnuc;
                    double norm = -self->atomNumber * pow(xi / M_PI, 1.5);
                    double tmp = 4.0 * a1 * a2 *
                                 int_sph_int2e_get_radial(0, 0.0, 0, xi,
                                                          ll + 1, a1, ll + 1,
                                                          a2, 0);
                    if (ll != 0)
                        tmp += (1.0 + ll + kappa) * (1.0 + ll + kappa) *
                                   int_sph_int2e_get_radial(0, 0.0, 0, xi,
                                                            ll - 1, a1, ll - 1,
                                                            a2, 0) -
                               2.0 * (a1 + a2) * (1 + ll + kappa) *
                                   int_sph_int2e_get_radial(0, 0.0, 0, xi,
                                                            ll + 1, a1, ll - 1,
                                                            a2, 0);
                    M(h1e_mat, ii, jj) = norm * tmp * 4.0 * M_PI;
                }
                else if (strcmp(intType, "s_p_nucGau_s_p_sf") == 0)
                {
                    double a_13 =
                        pow(mass_tmp[self->atomNumber - 1], 1.0 / 3.0);
                    double rnuc = (0.836 * a_13 + 0.570) / 52917.7249,
                           xi = 3.0 / 2.0 / rnuc / rnuc;
                    double norm = -self->atomNumber * pow(xi / M_PI, 1.5);
                    double tmp = 4.0 * a1 * a2 *
                                 int_sph_int2e_get_radial(0, 0.0, 0, xi,
                                                          ll + 1, a1, ll + 1,
                                                          a2, 0);
                    if (ll != 0)
                        tmp += (2.0 * ll * ll + ll) *
                                   int_sph_int2e_get_radial(0, 0.0, 0, xi,
                                                            ll - 1, a1, ll - 1,
                                                            a2, 0) -
                               2.0 * (a1 + a2) * (ll) *
                                   int_sph_int2e_get_radial(0, 0.0, 0, xi,
                                                            ll + 1, a1, ll - 1,
                                                            a2, 0);
                    M(h1e_mat, ii, jj) = norm * tmp * 4.0 * M_PI;
                }
                else if (strcmp(intType, "kinetic") == 0)
                {
                    M(h1e_mat, ii, jj) = 4 * a1 * a2 * auxiliary_1e_list[4];
                    if (ll != 0)
                        M(h1e_mat, ii, jj) +=
                            pow(ll + kappa + 1.0, 2) * auxiliary_1e_list[0] -
                            2.0 * (ll + kappa + 1.0) * (a1 + a2) *
                                auxiliary_1e_list[2];
                    M(h1e_mat, ii, jj) /= 2.0;
                }
                else
                {
                    printf("ERROR: get_h1e is called for undefined type of "
                           "integrals!\n");
                    exit(99);
                }
                M(h1e_mat, ii, jj) = M(h1e_mat, ii, jj) /
                                     self->shell_list[ishell].norm.d[ii] /
                                     self->shell_list[ishell].norm.d[jj];
            }
        }
        for (int ii = 0; ii < self->irrep_list[int_tmp].two_j + 1; ii++)
            mat_assign(&int_1e.m[int_tmp + ii],
                       mat_clone(h1e_single_shell.m[0]));
        int_tmp += self->irrep_list[int_tmp].two_j + 1;
        if (ll != 0)
        {
            for (int ii = 0; ii < self->irrep_list[int_tmp].two_j + 1; ii++)
                mat_assign(&int_1e.m[int_tmp + ii],
                           mat_clone(h1e_single_shell.m[1]));
            int_tmp += self->irrep_list[int_tmp].two_j + 1;
        }
        vmat_free(&h1e_single_shell);
    }

    return int_1e;
}

/*
    Evaluate different two-electron Coulomb and Exchange integral in
    2-spinor basis using the compact irrep layout
*/
int2eJK int_sph_get_h2e_JK_compact(const INT_SPH* self, const char* intType,
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
        int l_q = self->shell_list[qshell].l, l_max = MAX(l_p, l_q),
            LmaxJ = MIN(l_p + l_p, l_q + l_q), LmaxK = l_p + l_q;
        (void)l_max;
        int size_gtos_p = self->shell_list[pshell].coeff.rows,
            size_gtos_q = self->shell_list[qshell].coeff.rows;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_angular_J[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K[LmaxK + 1][size_tmp_p][size_tmp_q];
        double radial_2e_list_J[size_gtos_p * size_gtos_p * size_gtos_q *
                                size_gtos_q][LmaxJ + 1][3][3];
        double radial_2e_list_K[size_gtos_p * size_gtos_p * size_gtos_q *
                                size_gtos_q][LmaxK + 1][3][3];

        #pragma omp parallel for
        for (int tt = 0; tt < size_gtos_p * size_gtos_p * size_gtos_q *
                              size_gtos_q;
             tt++)
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

            for (int LL = LmaxJ; LL >= 0; LL -= 2)
            {
                radial_2e_list_J[tt][LL][0][0] = int_sph_int2e_get_radial(
                    l_p, a_i_J, l_p, a_j_J, l_q, a_k_J, l_q, a_l_J, LL);
            }
            for (int LL = LmaxK; LL >= 0; LL -= 2)
            {
                radial_2e_list_K[tt][LL][0][0] = int_sph_int2e_get_radial(
                    l_p, a_i_K, l_q, a_j_K, l_q, a_k_K, l_p, a_l_K, LL);
            }
            if (strncmp(intType, "SS", 2) == 0)
            {
                for (int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][1][0] = int_sph_int2e_get_radial(
                        l_p + 1, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q, a_l_J,
                        LL);
                    if (l_p != 0)
                        radial_2e_list_J[tt][LL][2][0] =
                            int_sph_int2e_get_radial(l_p - 1, a_i_J, l_p - 1,
                                                     a_j_J, l_q, a_k_J, l_q,
                                                     a_l_J, LL);
                }
                for (int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][1][0] = int_sph_int2e_get_radial(
                        l_p + 1, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p, a_l_K,
                        LL);
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_K[tt][LL][2][0] =
                            int_sph_int2e_get_radial(l_p - 1, a_i_K, l_q - 1,
                                                     a_j_K, l_q, a_k_K, l_p,
                                                     a_l_K, LL);
                }
            }
            if (strncmp(intType, "SSSS", 4) == 0)
            {
                for (int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][0][1] = int_sph_int2e_get_radial(
                        l_p, a_i_J, l_p, a_j_J, l_q + 1, a_k_J, l_q + 1, a_l_J,
                        LL);
                    radial_2e_list_J[tt][LL][1][1] = int_sph_int2e_get_radial(
                        l_p + 1, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J,
                        l_q + 1, a_l_J, LL);
                    if (l_p != 0)
                    {
                        radial_2e_list_J[tt][LL][2][1] =
                            int_sph_int2e_get_radial(l_p - 1, a_i_J, l_p - 1,
                                                     a_j_J, l_q + 1, a_k_J,
                                                     l_q + 1, a_l_J, LL);
                    }
                    if (l_q != 0)
                    {
                        radial_2e_list_J[tt][LL][0][2] =
                            int_sph_int2e_get_radial(l_p, a_i_J, l_p, a_j_J,
                                                     l_q - 1, a_k_J, l_q - 1,
                                                     a_l_J, LL);
                        radial_2e_list_J[tt][LL][1][2] =
                            int_sph_int2e_get_radial(l_p + 1, a_i_J, l_p + 1,
                                                     a_j_J, l_q - 1, a_k_J,
                                                     l_q - 1, a_l_J, LL);
                    }
                    if (l_p != 0 && l_q != 0)
                        radial_2e_list_J[tt][LL][2][2] =
                            int_sph_int2e_get_radial(l_p - 1, a_i_J, l_p - 1,
                                                     a_j_J, l_q - 1, a_k_J,
                                                     l_q - 1, a_l_J, LL);
                }
                for (int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][0][1] = int_sph_int2e_get_radial(
                        l_p, a_i_K, l_q, a_j_K, l_q + 1, a_k_K, l_p + 1, a_l_K,
                        LL);
                    radial_2e_list_K[tt][LL][1][1] = int_sph_int2e_get_radial(
                        l_p + 1, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K,
                        l_p + 1, a_l_K, LL);
                    if (l_p != 0 && l_q != 0)
                    {
                        radial_2e_list_K[tt][LL][2][1] =
                            int_sph_int2e_get_radial(l_p - 1, a_i_K, l_q - 1,
                                                     a_j_K, l_q + 1, a_k_K,
                                                     l_p + 1, a_l_K, LL);
                        radial_2e_list_K[tt][LL][0][2] =
                            int_sph_int2e_get_radial(l_p, a_i_K, l_q, a_j_K,
                                                     l_q - 1, a_k_K, l_p - 1,
                                                     a_l_K, LL);
                        radial_2e_list_K[tt][LL][1][2] =
                            int_sph_int2e_get_radial(l_p + 1, a_i_K, l_q + 1,
                                                     a_j_K, l_q - 1, a_k_K,
                                                     l_p - 1, a_l_K, LL);
                        radial_2e_list_K[tt][LL][2][2] =
                            int_sph_int2e_get_radial(l_p - 1, a_i_K, l_q - 1,
                                                     a_j_K, l_q - 1, a_k_K,
                                                     l_p - 1, a_l_K, LL);
                    }
                }
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
                    tmp_d += int_sph_int2e_get_angular_J(
                        l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq,
                        tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int_sph_int2e_get_angular_K(
                        l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq,
                        tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            /* Radial */
            double k_p = -(twojj_p + 1.0) * sym_ap / 2.0,
                   k_q = -(twojj_q + 1.0) * sym_aq / 2.0;
            #pragma omp parallel for
            for (int tt = 0; tt < size_gtos_p * size_gtos_p * size_gtos_q *
                                  size_gtos_q;
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
                    if (strcmp(intType, "LLLL") == 0)
                    {
                        radial_J = int_sph_get_radial_LLLL_J(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_J[tt][tmp],
                                       false) /
                                   norm_J;
                    }
                    else if (strcmp(intType, "SSLL") == 0)
                    {
                        radial_J = int_sph_get_radial_SSLL_J(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_J[tt][tmp],
                                       false) /
                                   norm_J / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "SSSS") == 0)
                    {
                        radial_J = int_sph_get_radial_SSSS_J(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_J[tt][tmp],
                                       false) /
                                   norm_J / 16.0 / pow(speedOfLight, 4);
                    }
                    else if (strcmp(intType, "SSLL_SF") == 0)
                    {
                        radial_J = int_sph_get_radial_SSLL_J(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_J[tt][tmp],
                                       true) /
                                   norm_J / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "SSSS_SF") == 0)
                    {
                        radial_J = int_sph_get_radial_SSSS_J(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_J[tt][tmp],
                                       true) /
                                   norm_J / 16.0 / pow(speedOfLight, 4);
                    }
                    else if (strcmp(intType, "SSLL_SD") == 0)
                    {
                        radial_J = (int_sph_get_radial_SSLL_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], false) -
                                    int_sph_get_radial_SSLL_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], true)) /
                                   norm_J / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "SSSS_SD") == 0)
                    {
                        radial_J = (int_sph_get_radial_SSSS_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], false) -
                                    int_sph_get_radial_SSSS_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], true)) /
                                   norm_J / 16.0 / pow(speedOfLight, 4);
                    }
                    else
                    {
                        printf("ERROR: Unknown integralTYPE in get_h2e:\n");
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
                    if (strcmp(intType, "LLLL") == 0)
                    {
                        radial_K = int_sph_get_radial_LLLL_K(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_K[tt][tmp],
                                       false) /
                                   norm_K;
                    }
                    else if (strcmp(intType, "SSLL") == 0)
                    {
                        radial_K = int_sph_get_radial_SSLL_K(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_K[tt][tmp],
                                       false) /
                                   norm_K / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "SSSS") == 0)
                    {
                        radial_K = int_sph_get_radial_SSSS_K(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_K[tt][tmp],
                                       false) /
                                   norm_K / 16.0 / pow(speedOfLight, 4);
                    }
                    else if (strcmp(intType, "SSLL_SF") == 0)
                    {
                        radial_K = int_sph_get_radial_SSLL_K(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_K[tt][tmp],
                                       true) /
                                   norm_K / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "SSSS_SF") == 0)
                    {
                        radial_K = int_sph_get_radial_SSSS_K(
                                       l_p, l_q, tmp, a1, a2, a3, a4, lk1, lk2,
                                       lk3, lk4, radial_2e_list_K[tt][tmp],
                                       true) /
                                   norm_K / 16.0 / pow(speedOfLight, 4);
                    }
                    else if (strcmp(intType, "SSLL_SD") == 0)
                    {
                        radial_K = (int_sph_get_radial_SSLL_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], false) -
                                    int_sph_get_radial_SSLL_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], true)) /
                                   norm_K / 4.0 / pow(speedOfLight, 2);
                    }
                    else if (strcmp(intType, "SSSS_SD") == 0)
                    {
                        radial_K = (int_sph_get_radial_SSSS_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], false) -
                                    int_sph_get_radial_SSSS_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], true)) /
                                   norm_K / 16.0 / pow(speedOfLight, 4);
                    }
                    else
                    {
                        printf("ERROR: Unknown integralTYPE in get_h2e:\n");
                        exit(99);
                    }
                    int_2e_JK.K[int_tmp1_p + int_tmp2_p]
                               [int_tmp1_q + int_tmp2_q][e1K][e2K] +=
                        radial_K * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return int_2e_JK;
}

/*
    Evaluate all compact 2e integral together for DHF calculations
*/
void int_sph_get_h2e_JK_direct(INT_SPH* self, int2eJK* LLLL, int2eJK* SSLL,
                               int2eJK* SSSS, int occMaxL, bool spinFree)
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

    LLLL->J = (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    LLLL->K = (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    SSLL->J = (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    SSLL->K = (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    SSSS->J = (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    SSSS->K = (double****)malloc((size_t)Nirrep_compact * sizeof(double***));
    for (int ii = 0; ii < Nirrep_compact; ii++)
    {
        LLLL->J[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        LLLL->K[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        SSLL->J[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        SSLL->K[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        SSSS->J[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
        SSSS->K[ii] =
            (double***)malloc((size_t)Nirrep_compact * sizeof(double**));
    }

    int int_tmp1_p = 0;
    for (int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = self->shell_list[pshell].l, int_tmp1_q = 0;
    for (int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = self->shell_list[qshell].l, l_max = MAX(l_p, l_q),
            LmaxJ = MIN(l_p + l_p, l_q + l_q), LmaxK = l_p + l_q;
        (void)l_max;
        /* This is correct for J (and not K) but the author did not
           understand.  Same for Gaunt and gauge term.  A very limited
           acceleration. */
        /* LmaxJ = 0; */
        int size_gtos_p = self->shell_list[pshell].coeff.rows,
            size_gtos_q = self->shell_list[qshell].coeff.rows;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_angular_J[LmaxJ + 1][size_tmp_p][size_tmp_q],
            array_angular_K[LmaxK + 1][size_tmp_p][size_tmp_q];
        double radial_2e_list_J[size_gtos_p * size_gtos_p * size_gtos_q *
                                size_gtos_q][LmaxJ + 1][3][3];
        double radial_2e_list_K[size_gtos_p * size_gtos_p * size_gtos_q *
                                size_gtos_q][LmaxK + 1][3][3];

        #pragma omp parallel for
        for (int tt = 0; tt < size_gtos_p * size_gtos_p * size_gtos_q *
                              size_gtos_q;
             tt++)
        {
            int e1J = tt / (size_gtos_q * size_gtos_q);
            int e2J = tt - e1J * (size_gtos_q * size_gtos_q);
            int ii = e1J / size_gtos_p, jj = e1J - ii * size_gtos_p;
            int kk = e2J / size_gtos_q, ll = e2J - kk * size_gtos_q;
            double a_i_J = self->shell_list[pshell].exp_a.d[ii],
                   a_j_J = self->shell_list[pshell].exp_a.d[jj],
                   a_k_J = self->shell_list[qshell].exp_a.d[kk],
                   a_l_J = self->shell_list[qshell].exp_a.d[ll];
            double a_i_K = self->shell_list[pshell].exp_a.d[ii],
                   a_j_K = self->shell_list[qshell].exp_a.d[ll],
                   a_k_K = self->shell_list[qshell].exp_a.d[kk],
                   a_l_K = self->shell_list[pshell].exp_a.d[jj];

            for (int LL = LmaxJ; LL >= 0; LL -= 2)
            {
                radial_2e_list_J[tt][LL][0][0] = int_sph_int2e_get_radial(
                    l_p, a_i_J, l_p, a_j_J, l_q, a_k_J, l_q, a_l_J, LL);
                radial_2e_list_J[tt][LL][1][0] = int_sph_int2e_get_radial(
                    l_p + 1, a_i_J, l_p + 1, a_j_J, l_q, a_k_J, l_q, a_l_J,
                    LL);
                radial_2e_list_J[tt][LL][0][1] = int_sph_int2e_get_radial(
                    l_p, a_i_J, l_p, a_j_J, l_q + 1, a_k_J, l_q + 1, a_l_J,
                    LL);
                radial_2e_list_J[tt][LL][1][1] = int_sph_int2e_get_radial(
                    l_p + 1, a_i_J, l_p + 1, a_j_J, l_q + 1, a_k_J, l_q + 1,
                    a_l_J, LL);
                if (l_p != 0)
                {
                    radial_2e_list_J[tt][LL][2][0] = int_sph_int2e_get_radial(
                        l_p - 1, a_i_J, l_p - 1, a_j_J, l_q, a_k_J, l_q, a_l_J,
                        LL);
                    radial_2e_list_J[tt][LL][2][1] = int_sph_int2e_get_radial(
                        l_p - 1, a_i_J, l_p - 1, a_j_J, l_q + 1, a_k_J,
                        l_q + 1, a_l_J, LL);
                }
                if (l_q != 0)
                {
                    radial_2e_list_J[tt][LL][0][2] = int_sph_int2e_get_radial(
                        l_p, a_i_J, l_p, a_j_J, l_q - 1, a_k_J, l_q - 1, a_l_J,
                        LL);
                    radial_2e_list_J[tt][LL][1][2] = int_sph_int2e_get_radial(
                        l_p + 1, a_i_J, l_p + 1, a_j_J, l_q - 1, a_k_J,
                        l_q - 1, a_l_J, LL);
                }
                if (l_p != 0 && l_q != 0)
                    radial_2e_list_J[tt][LL][2][2] = int_sph_int2e_get_radial(
                        l_p - 1, a_i_J, l_p - 1, a_j_J, l_q - 1, a_k_J,
                        l_q - 1, a_l_J, LL);
            }
            for (int LL = LmaxK; LL >= 0; LL -= 2)
            {
                radial_2e_list_K[tt][LL][0][0] = int_sph_int2e_get_radial(
                    l_p, a_i_K, l_q, a_j_K, l_q, a_k_K, l_p, a_l_K, LL);
                radial_2e_list_K[tt][LL][1][0] = int_sph_int2e_get_radial(
                    l_p + 1, a_i_K, l_q + 1, a_j_K, l_q, a_k_K, l_p, a_l_K,
                    LL);
                radial_2e_list_K[tt][LL][0][1] = int_sph_int2e_get_radial(
                    l_p, a_i_K, l_q, a_j_K, l_q + 1, a_k_K, l_p + 1, a_l_K,
                    LL);
                radial_2e_list_K[tt][LL][1][1] = int_sph_int2e_get_radial(
                    l_p + 1, a_i_K, l_q + 1, a_j_K, l_q + 1, a_k_K, l_p + 1,
                    a_l_K, LL);
                if (l_p != 0 && l_q != 0)
                {
                    radial_2e_list_K[tt][LL][2][0] = int_sph_int2e_get_radial(
                        l_p - 1, a_i_K, l_q - 1, a_j_K, l_q, a_k_K, l_p, a_l_K,
                        LL);
                    radial_2e_list_K[tt][LL][2][1] = int_sph_int2e_get_radial(
                        l_p - 1, a_i_K, l_q - 1, a_j_K, l_q + 1, a_k_K,
                        l_p + 1, a_l_K, LL);
                    radial_2e_list_K[tt][LL][0][2] = int_sph_int2e_get_radial(
                        l_p, a_i_K, l_q, a_j_K, l_q - 1, a_k_K, l_p - 1, a_l_K,
                        LL);
                    radial_2e_list_K[tt][LL][1][2] = int_sph_int2e_get_radial(
                        l_p + 1, a_i_K, l_q + 1, a_j_K, l_q - 1, a_k_K,
                        l_p - 1, a_l_K, LL);
                    radial_2e_list_K[tt][LL][2][2] = int_sph_int2e_get_radial(
                        l_p - 1, a_i_K, l_q - 1, a_j_K, l_q - 1, a_k_K,
                        l_p - 1, a_l_K, LL);
                }
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

            LLLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_p) *
                                 sizeof(double*));
            LLLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_q) *
                                 sizeof(double*));
            SSLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_p) *
                                 sizeof(double*));
            SSLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_q) *
                                 sizeof(double*));
            SSSS->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_p) *
                                 sizeof(double*));
            SSSS->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q] =
                (double**)malloc((size_t)(size_gtos_p * size_gtos_q) *
                                 sizeof(double*));
            for (int iii = 0; iii < size_gtos_p * size_gtos_p; iii++)
            {
                LLLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                       [iii] = (double*)malloc(
                           (size_t)(size_gtos_q * size_gtos_q) *
                           sizeof(double));
                SSLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                       [iii] = (double*)malloc(
                           (size_t)(size_gtos_q * size_gtos_q) *
                           sizeof(double));
                SSSS->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                       [iii] = (double*)malloc(
                           (size_t)(size_gtos_q * size_gtos_q) *
                           sizeof(double));
            }
            for (int iii = 0; iii < size_gtos_p * size_gtos_q; iii++)
            {
                LLLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                       [iii] = (double*)malloc(
                           (size_t)(size_gtos_q * size_gtos_p) *
                           sizeof(double));
                SSLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                       [iii] = (double*)malloc(
                           (size_t)(size_gtos_q * size_gtos_p) *
                           sizeof(double));
                SSSS->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                       [iii] = (double*)malloc(
                           (size_t)(size_gtos_q * size_gtos_p) *
                           sizeof(double));
            }

            /* Angular */
            for (int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int_sph_int2e_get_angular_J(
                        l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq,
                        tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for (int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int_sph_int2e_get_angular_K(
                        l_p, twojj_p, sym_ap, l_q, 2 * mq - twojj_q, sym_aq,
                        tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }

            /* Radial */
            double k_p = -(twojj_p + 1.0) * sym_ap / 2.0,
                   k_q = -(twojj_q + 1.0) * sym_aq / 2.0;
            #pragma omp parallel for
            for (int tt = 0; tt < size_gtos_p * size_gtos_p * size_gtos_q *
                                  size_gtos_q;
                 tt++)
            {
                double radial_J_LLLL, radial_K_LLLL, radial_J_SSLL,
                    radial_K_SSLL, radial_J_SSSS, radial_K_SSSS;
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
                LLLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J]
                       [e2J] = 0.0;
                LLLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K]
                       [e2K] = 0.0;
                SSLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J]
                       [e2J] = 0.0;
                SSLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K]
                       [e2K] = 0.0;
                SSSS->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1J]
                       [e2J] = 0.0;
                SSSS->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q][e1K]
                       [e2K] = 0.0;

                for (int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    radial_J_LLLL = int_sph_get_radial_LLLL_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], spinFree) /
                                    norm_J;
                    radial_J_SSLL = int_sph_get_radial_SSLL_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], spinFree) /
                                    norm_J / 4.0 / pow(speedOfLight, 2);
                    radial_J_SSSS = int_sph_get_radial_SSSS_J(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_J[tt][tmp], spinFree) /
                                    norm_J / 16.0 / pow(speedOfLight, 4);
                    LLLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1J][e2J] +=
                        radial_J_LLLL *
                        array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                    SSLL->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1J][e2J] +=
                        radial_J_SSLL *
                        array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                    SSSS->J[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1J][e2J] +=
                        radial_J_SSSS *
                        array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1 + l_q + k_q;
                lk4 = 1 + l_p + k_p;
                a2 = self->shell_list[qshell].exp_a.d[ll];
                a4 = self->shell_list[pshell].exp_a.d[jj];
                for (int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    radial_K_LLLL = int_sph_get_radial_LLLL_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], spinFree) /
                                    norm_K;
                    radial_K_SSLL = int_sph_get_radial_SSLL_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], spinFree) /
                                    norm_K / 4.0 / pow(speedOfLight, 2);
                    radial_K_SSSS = int_sph_get_radial_SSSS_K(
                                        l_p, l_q, tmp, a1, a2, a3, a4, lk1,
                                        lk2, lk3, lk4,
                                        radial_2e_list_K[tt][tmp], spinFree) /
                                    norm_K / 16.0 / pow(speedOfLight, 4);
                    LLLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1K][e2K] +=
                        radial_K_LLLL *
                        array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                    SSLL->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1K][e2K] +=
                        radial_K_SSLL *
                        array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                    SSSS->K[int_tmp1_p + int_tmp2_p][int_tmp1_q + int_tmp2_q]
                           [e1K][e2K] +=
                        radial_K_SSSS *
                        array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return;
}

void int_sph_get_h2eSD_JK_direct(INT_SPH* self, int2eJK* SSLL, int2eJK* SSSS,
                                 int occMaxL)
{
    *SSLL = int_sph_get_h2e_JK_compact(self, "SSLL_SD", occMaxL);
    *SSSS = int_sph_get_h2e_JK_compact(self, "SSSS_SD", occMaxL);
}
