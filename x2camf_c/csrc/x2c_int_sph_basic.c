/*
 * x2c_int_sph_basic.c -- C translation of src/int_sph_basic.cpp (X2CAMF):
 * INT_SPH constructor and the auxiliary radial / angular integral helpers.
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

INT_SPH* int_sph_new(int atom_number, int nshell, int nbas, const int* shell,
                     const double* exp_a)
{
    INT_SPH* self = (INT_SPH*)calloc(1, sizeof(INT_SPH));
    self->atomNumber = atom_number;
    self->size_gtoc = nbas;
    self->size_gtou = nbas;
    self->size_shell = nshell;
    self->atomName = elem_list[atom_number];

    /* orbitalInfo(0,ii) = l, orbitalInfo(1,ii) = orbitalInfo(2,ii) =
       number of primitives with that l */
    int* shell_info = (int*)calloc(10, sizeof(int));
    int* accumu = (int*)calloc(10, sizeof(int));
    self->shell_list = (intShell*)calloc((size_t)nshell, sizeof(intShell));
    for (int ibas = 0; ibas < nbas; ibas++)
    {
        shell_info[shell[ibas]] += 1;
    }
    for (int ii = 0; ii < nshell; ii++)
    {
        if (ii == 0) continue;
        accumu[ii] = accumu[ii - 1] + shell_info[ii - 1];
    }

    self->Nirrep = 0;
    for (int ii = 0; ii < nshell; ii++)
    {
        self->Nirrep += 2 * (2 * ii + 1);
    }
    self->irrep_list =
        (irrep_jm*)calloc((size_t)self->Nirrep, sizeof(irrep_jm));
    int tmp_i = 0;
    for (int ii = 0; ii < nshell; ii++)
    {
        int two_jj = 2 * ii + 1;
        if (ii != 0)
        {
            int two_jj_m = 2 * ii - 1;
            for (int two_mj = -two_jj_m; two_mj <= two_jj_m; two_mj += 2)
            {
                self->irrep_list[tmp_i].l = ii;
                self->irrep_list[tmp_i].size = shell_info[ii];
                self->irrep_list[tmp_i].two_j = two_jj_m;
                self->irrep_list[tmp_i].two_mj = two_mj;
                tmp_i++;
            }
        }
        for (int two_mj = -two_jj; two_mj <= two_jj; two_mj += 2)
        {
            self->irrep_list[tmp_i].l = ii;
            self->irrep_list[tmp_i].size = shell_info[ii];
            self->irrep_list[tmp_i].two_j = two_jj;
            self->irrep_list[tmp_i].two_mj = two_mj;
            tmp_i++;
        }
    }

    self->size_gtou = 0;
    self->size_gtoc = 0;
    for (int ishell = 0; ishell < nshell; ishell++)
    {
        const int nprim = shell_info[ishell];
        self->size_gtou += (2 * ishell + 1) * nprim;
        self->size_gtoc += (2 * ishell + 1) * nprim;
        intShell* sh = &self->shell_list[ishell];
        sh->l = ishell;
        sh->coeff = mat_new(nprim, nprim);
        sh->exp_a = vecd_new(nprim);
        sh->norm = vecd_new(nprim);
        int offset = accumu[ishell];
        for (int ii = 0; ii < nprim; ii++)
        {
            sh->exp_a.d[ii] = exp_a[ii + offset];
            M(sh->coeff, ii, ii) = 1.0; /* assumes only uncontracted basis */
            sh->norm.d[ii] =
                sqrt(int_sph_auxiliary_1e(2 * sh->l + 2, 2 * sh->exp_a.d[ii]));
        }
    }
    self->size_gtoc_spinor = 2 * self->size_gtoc;
    self->size_gtou_spinor = 2 * self->size_gtou;

    free(shell_info);
    free(accumu);
    return self;
}

void int_sph_free(INT_SPH* self)
{
    if (self == NULL) return;
    for (int i = 0; i < self->size_shell; i++)
    {
        vecd_free(&self->shell_list[i].exp_a);
        vecd_free(&self->shell_list[i].norm);
        mat_free(&self->shell_list[i].coeff);
    }
    free(self->shell_list);
    free(self->irrep_list);
    free(self);
}

/*
    auxiliary_1e is to evaluate \int_0^inf x^l exp(-ax^2) dx
*/
double int_sph_auxiliary_1e(int l, double a)
{
    int n = l / 2;
    if (l < 0)
    {
        /* l = -2 is special case in gauge term.
           It will not contribute to the final integral. */
        if (l == -2) return 0.0;
        printf("ERROR: l = %d for auxiliary 1e integral.\n", l);
        exit(99);
    }
    else if (l == 0)
        return 0.5 * sqrt(M_PI / a);
    else if (n * 2 == l)
        return double_factorial(2 * n - 1) / pow(a, n) / pow(2.0, n + 1) *
               sqrt(M_PI / a);
    else
        return factorial(n) / 2.0 / pow(a, n + 1);
}

/*
    auxiliary_2e_0_r is to evaluate
    \int_0^inf \int_0^r2 r1^l1 r2^l2 exp(-a1 r1^2) exp(-a2 r2^2) dr1dr2
*/
double int_sph_auxiliary_2e_0_r(int l1, int l2, double a1, double a2)
{
    int n1 = l1 / 2;
    if (l1 < 0 || l2 < 0)
    {
        return 0.0;
    }
    if (n1 * 2 == l1)
    {
        printf("ERROR: When auxiliary_2e_0_r is called, l1 must be set to an "
               "odd number!\n");
        exit(99);
    }
    else
    {
        double tmp = 0.5 / pow(a1, n1 + 1) * int_sph_auxiliary_1e(l2, a2);
        for (int kk = 0; kk <= n1; kk++)
        {
            tmp -= 0.5 / factorial(kk) / pow(a1, n1 - kk + 1) *
                   int_sph_auxiliary_1e(l2 + 2 * kk, a1 + a2);
        }
        return tmp * factorial(n1);
    }
}

/*
    auxiliary_2e_r_inf is to evaluate
    \int_0^inf \int_r2^inf r1^l1 r2^l2 exp(-a1 r1^2) exp(-a2 r2^2) dr1dr2
*/
double int_sph_auxiliary_2e_r_inf(int l1, int l2, double a1, double a2)
{
    if (l1 < 0 || l2 < 0)
    {
        return 0.0;
    }
    int n1 = l1 / 2;
    if (n1 * 2 == l1)
    {
        printf("ERROR: When auxiliary_2e_r_inf is called, l1 must be set to "
               "an odd number!\n");
        exit(99);
    }
    else
    {
        double tmp = 0.0;
        for (int kk = 0; kk <= n1; kk++)
        {
            tmp += 0.5 / factorial(kk) / pow(a1, n1 - kk + 1) *
                   int_sph_auxiliary_1e(l2 + 2 * kk, a1 + a2);
        }
        return tmp * factorial(n1);
    }
}

/*
    evaluate radial part and angular part in 2e integrals
*/
double int_sph_int2e_get_radial(int l1, double a1, int l2, double a2, int l3,
                                double a3, int l4, double a4, int LL)
{
    if ((l1 + l2 + l3 + l4) % 2) return 0.0;
    double radial = 0.0;
    if ((l1 + l2 + 2 + LL) % 2)
    {
        radial = int_sph_auxiliary_2e_0_r(l1 + l2 + 2 + LL, l3 + l4 + 1 - LL,
                                          a1 + a2, a3 + a4) +
                 int_sph_auxiliary_2e_0_r(l3 + l4 + 2 + LL, l1 + l2 + 1 - LL,
                                          a3 + a4, a1 + a2);
    }
    else
    {
        radial = int_sph_auxiliary_2e_r_inf(l3 + l4 + 1 - LL, l1 + l2 + 2 + LL,
                                            a3 + a4, a1 + a2) +
                 int_sph_auxiliary_2e_r_inf(l1 + l2 + 1 - LL, l3 + l4 + 2 + LL,
                                            a1 + a2, a3 + a4);
    }

    return radial;
}
double int_sph_int2e_get_angular(int l1, int two_m1, int s1, int l2,
                                 int two_m2, int s2, int l3, int two_m3,
                                 int s3, int l4, int two_m4, int s4, int LL)
{
    if ((l1 + l2 + LL) % 2 || (l3 + l4 + LL) % 2) return 0.0;

    int two_j1 = 2 * l1 + s1;
    int two_j2 = 2 * l2 + s2;
    int two_j3 = 2 * l3 + s3;
    int two_j4 = 2 * l4 + s4;
    double angular = 0.0;
    for (int mm = -LL; mm <= LL; mm++)
    {
        if (two_m2 - two_m1 - 2 * mm != 0 || two_m4 - two_m3 + 2 * mm != 0)
            continue;
        else
        {
            angular += pow(-1, mm) *
                       CG_wigner_3j(two_j1, 2 * LL, two_j2, -two_m1, -2 * mm,
                                    two_m2) *
                       CG_wigner_3j(two_j3, 2 * LL, two_j4, -two_m3, 2 * mm,
                                    two_m4);
        }
    }

    return pow(-1.0, two_j1 + two_j3 - (two_m1 + two_m3) / 2 - 1) * angular *
           sqrt((two_j1 + 1.0) * (two_j2 + 1.0) * (two_j3 + 1.0) *
                (two_j4 + 1.0)) *
           CG_wigner_3j(two_j1, 2 * LL, two_j2, 1, 0, -1) *
           CG_wigner_3j(two_j3, 2 * LL, two_j4, 1, 0, -1);
}
double int_sph_int2e_get_angular_J(int l1, int two_m1, int s1, int l2,
                                   int two_m2, int s2, int LL)
{
    return int_sph_int2e_get_angular(l1, two_m1, s1, l1, two_m1, s1, l2,
                                     two_m2, s2, l2, two_m2, s2, LL);
}
double int_sph_int2e_get_angular_K(int l1, int two_m1, int s1, int l2,
                                   int two_m2, int s2, int LL)
{
    return int_sph_int2e_get_angular(l1, two_m1, s1, l2, two_m2, s2, l2,
                                     two_m2, s2, l1, two_m1, s1, LL);
}

double int_sph_get_radial_LLLL_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree)
{
    (void)lp; (void)lq; (void)LL; (void)a1; (void)a2; (void)a3; (void)a4;
    (void)lk1; (void)lk2; (void)lk3; (void)lk4; (void)spinFree;
    return radial_list[0][0];
}
double int_sph_get_radial_LLLL_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree)
{
    (void)lp; (void)lq; (void)LL; (void)a1; (void)a2; (void)a3; (void)a4;
    (void)lk1; (void)lk2; (void)lk3; (void)lk4; (void)spinFree;
    return radial_list[0][0];
}
double int_sph_get_radial_SSLL_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree)
{
    (void)a3; (void)a4; (void)lk3; (void)lk4; (void)lq;
    double tmp = 4.0 * a1 * a2 * radial_list[1][0];
    if (spinFree)
    {
        double l12 = lp * lp + lp * (lp + 1) / 2 + lp * (lp + 1) / 2 -
                     LL * (LL + 1) / 2;
        if (lp != 0)
            tmp += l12 * radial_list[2][0] -
                   (2.0 * a1 * lp + 2.0 * a2 * lp) * radial_list[0][0];
    }
    else
    {
        if (lp != 0)
            tmp += lk1 * lk2 * radial_list[2][0] -
                   (2.0 * a1 * lk2 + 2.0 * a2 * lk1) * radial_list[0][0];
    }

    return tmp;
}
double int_sph_get_radial_SSLL_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree)
{
    (void)a3; (void)a4; (void)lk3; (void)lk4;
    double tmp = 4.0 * a1 * a2 * radial_list[1][0];
    if (spinFree)
    {
        double l12 = lp * lq + lp * (lp + 1) / 2 + lq * (lq + 1) / 2 -
                     LL * (LL + 1) / 2;
        if (lp != 0 && lq != 0)
            tmp += l12 * radial_list[2][0] -
                   (2.0 * a1 * lq + 2.0 * a2 * lp) * radial_list[0][0];
        else if (lp != 0 || lq != 0)
            tmp += -(2.0 * a1 * lq + 2.0 * a2 * lp) * radial_list[0][0];
    }
    else
    {
        if (lp != 0 && lq != 0)
            tmp += lk1 * lk2 * radial_list[2][0] -
                   (2.0 * a1 * lk2 + 2.0 * a2 * lk1) * radial_list[0][0];
        else if (lp != 0 || lq != 0)
            tmp += -(2.0 * a1 * lk2 + 2.0 * a2 * lk1) * radial_list[0][0];
    }

    return tmp;
}
double int_sph_get_radial_SSSS_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree)
{
    double tmp = 4 * a1 * a2 * 4 * a3 * a4 * radial_list[1][1];
    if (spinFree)
    {
        double l12 = lp * lp + lp * (lp + 1) / 2 + lp * (lp + 1) / 2 -
                     LL * (LL + 1) / 2,
               l34 = lq * lq + lq * (lq + 1) / 2 + lq * (lq + 1) / 2 -
                     LL * (LL + 1) / 2;
        if (lp != 0)
        {
            if (lq != 0)
                tmp += l12 * l34 * radial_list[2][2] -
                       (2 * a1 * lp + 2 * a2 * lp) * l34 * radial_list[0][2] +
                       4 * a1 * a2 * l34 * radial_list[1][2] -
                       l12 * (2 * a3 * lq + 2 * a4 * lq) * radial_list[2][0] +
                       (2 * a1 * lp + 2 * a2 * lp) * (2 * a3 * lq + 2 * a4 * lq) *
                           radial_list[0][0] -
                       4 * a1 * a2 * (2 * a3 * lq + 2 * a4 * lq) *
                           radial_list[1][0] +
                       l12 * 4 * a3 * a4 * radial_list[2][1] -
                       (2 * a1 * lp + 2 * a2 * lp) * 4 * a3 * a4 *
                           radial_list[0][1];
            else
                tmp += l12 * 4 * a3 * a4 * radial_list[2][1] -
                       (2 * a1 * lp + 2 * a2 * lp) * 4 * a3 * a4 *
                           radial_list[0][1];
        }
        else
        {
            if (lq != 0)
                tmp += 4 * a1 * a2 * l34 * radial_list[1][2] -
                       4 * a1 * a2 * (2 * a3 * lq + 2 * a4 * lq) *
                           radial_list[1][0];
        }
    }
    else
    {
        if (lp != 0)
        {
            if (lq != 0)
                tmp += lk1 * lk2 * lk3 * lk4 * radial_list[2][2] -
                       (2 * a1 * lk2 + 2 * a2 * lk1) * lk3 * lk4 *
                           radial_list[0][2] +
                       4 * a1 * a2 * lk3 * lk4 * radial_list[1][2] -
                       lk1 * lk2 * (2 * a3 * lk4 + 2 * a4 * lk3) *
                           radial_list[2][0] +
                       (2 * a1 * lk2 + 2 * a2 * lk1) *
                           (2 * a3 * lk4 + 2 * a4 * lk3) * radial_list[0][0] -
                       4 * a1 * a2 * (2 * a3 * lk4 + 2 * a4 * lk3) *
                           radial_list[1][0] +
                       lk1 * lk2 * 4 * a3 * a4 * radial_list[2][1] -
                       (2 * a1 * lk2 + 2 * a2 * lk1) * 4 * a3 * a4 *
                           radial_list[0][1];
            else
                tmp += lk1 * lk2 * 4 * a3 * a4 * radial_list[2][1] -
                       (2 * a1 * lk2 + 2 * a2 * lk1) * 4 * a3 * a4 *
                           radial_list[0][1];
        }
        else
        {
            if (lq != 0)
                tmp += 4 * a1 * a2 * lk3 * lk4 * radial_list[1][2] -
                       4 * a1 * a2 * (2 * a3 * lk4 + 2 * a4 * lk3) *
                           radial_list[1][0];
        }
    }

    return tmp;
}
double int_sph_get_radial_SSSS_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree)
{
    double tmp = 4 * a1 * a2 * 4 * a3 * a4 * radial_list[1][1];
    if (spinFree)
    {
        double l12 = lp * lq + lp * (lp + 1) / 2 + lq * (lq + 1) / 2 -
                     LL * (LL + 1) / 2,
               l34 = lq * lp + lq * (lq + 1) / 2 + lp * (lp + 1) / 2 -
                     LL * (LL + 1) / 2;
        if (lp != 0 && lq != 0)
            tmp += l12 * l34 * radial_list[2][2] -
                   (2 * a1 * lq + 2 * a2 * lp) * l34 * radial_list[0][2] +
                   4 * a1 * a2 * l34 * radial_list[1][2] -
                   l12 * (2 * a3 * lp + 2 * a4 * lq) * radial_list[2][0] +
                   (2 * a1 * lq + 2 * a2 * lp) * (2 * a3 * lp + 2 * a4 * lq) *
                       radial_list[0][0] -
                   4 * a1 * a2 * (2 * a3 * lp + 2 * a4 * lq) *
                       radial_list[1][0] +
                   l12 * 4 * a3 * a4 * radial_list[2][1] -
                   (2 * a1 * lq + 2 * a2 * lp) * 4 * a3 * a4 *
                       radial_list[0][1];
        else if (lp != 0 || lq != 0)
            tmp += (2 * a1 * lq + 2 * a2 * lp) * (2 * a3 * lp + 2 * a4 * lq) *
                       radial_list[0][0] -
                   4 * a1 * a2 * (2 * a3 * lp + 2 * a4 * lq) *
                       radial_list[1][0] -
                   (2 * a1 * lq + 2 * a2 * lp) * 4 * a3 * a4 *
                       radial_list[0][1];
    }
    else
    {
        if (lp != 0 && lq != 0)
            tmp += lk1 * lk2 * lk3 * lk4 * radial_list[2][2] -
                   (2 * a1 * lk2 + 2 * a2 * lk1) * lk3 * lk4 *
                       radial_list[0][2] +
                   4 * a1 * a2 * lk3 * lk4 * radial_list[1][2] -
                   lk1 * lk2 * (2 * a3 * lk4 + 2 * a4 * lk3) *
                       radial_list[2][0] +
                   (2 * a1 * lk2 + 2 * a2 * lk1) *
                       (2 * a3 * lk4 + 2 * a4 * lk3) * radial_list[0][0] -
                   4 * a1 * a2 * (2 * a3 * lk4 + 2 * a4 * lk3) *
                       radial_list[1][0] +
                   lk1 * lk2 * 4 * a3 * a4 * radial_list[2][1] -
                   (2 * a1 * lk2 + 2 * a2 * lk1) * 4 * a3 * a4 *
                       radial_list[0][1];
        else if (lp != 0 || lq != 0)
            tmp += (2 * a1 * lk2 + 2 * a2 * lk1) *
                       (2 * a3 * lk4 + 2 * a4 * lk3) * radial_list[0][0] -
                   4 * a1 * a2 * (2 * a3 * lk4 + 2 * a4 * lk3) *
                       radial_list[1][0] -
                   (2 * a1 * lk2 + 2 * a2 * lk1) * 4 * a3 * a4 *
                       radial_list[0][1];
    }

    return tmp;
}

double int_sph_get_radial_LSLS_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree)
{
    (void)LL; (void)a1; (void)a3; (void)lk1; (void)lk3;
    double tmp = 4.0 * a2 * a4 * radial_list[0];
    if (spinFree)
    {
    }
    else
    {
        if (lp != 0 && lq != 0)
            tmp += lk2 * lk4 * radial_list[3] - 2.0 * a4 * lk2 * radial_list[1] -
                   2.0 * a2 * lk4 * radial_list[2];
        else if (lp != 0 && lq == 0)
            tmp -= 2.0 * a4 * lk2 * radial_list[1];
        else if (lp == 0 && lq != 0)
            tmp -= 2.0 * a2 * lk4 * radial_list[2];
    }

    return tmp;
}
double int_sph_get_radial_LSLS_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree)
{
    (void)LL; (void)a1; (void)a3; (void)lk1; (void)lk3;
    double tmp = 4.0 * a2 * a4 * radial_list[0];
    if (spinFree)
    {
    }
    else
    {
        if (lp != 0 && lq != 0)
            tmp += lk2 * lk4 * radial_list[3] - 2.0 * a4 * lk2 * radial_list[1] -
                   2.0 * a2 * lk4 * radial_list[2];
        else if (lp == 0 && lq != 0)
            tmp -= 2.0 * a4 * lk2 * radial_list[1];
        else if (lp != 0 && lq == 0)
            tmp -= 2.0 * a2 * lk4 * radial_list[2];
    }

    return tmp;
}
double int_sph_get_radial_LSSL_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree)
{
    (void)LL; (void)a1; (void)a4; (void)lk1; (void)lk4;
    double tmp = 4.0 * a2 * a3 * radial_list[0];
    if (spinFree)
    {
    }
    else
    {
        if (lp != 0 && lq != 0)
            tmp += lk2 * lk3 * radial_list[3] - 2.0 * a3 * lk2 * radial_list[1] -
                   2.0 * a2 * lk3 * radial_list[2];
        else if (lp != 0 && lq == 0)
            tmp -= 2.0 * a3 * lk2 * radial_list[1];
        else if (lp == 0 && lq != 0)
            tmp -= 2.0 * a2 * lk3 * radial_list[2];
    }

    return tmp;
}
double int_sph_get_radial_LSSL_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree)
{
    (void)lp; (void)LL; (void)a1; (void)a4; (void)lk1; (void)lk4;
    double tmp = 4.0 * a2 * a3 * radial_list[0];
    if (spinFree)
    {
    }
    else
    {
        if (lq != 0)
            tmp += lk2 * lk3 * radial_list[3] - 2.0 * a3 * lk2 * radial_list[1] -
                   2.0 * a2 * lk3 * radial_list[2];
    }

    return tmp;
}

double int_sph_int2e_get_threeSH(int l1, int m1, int l2, int m2, int l3,
                                 int m3, double threeJ)
{
    return pow(-1, m1) * threeJ * CG_wigner_3j_int(l1, l2, l3, -m1, m2, m3) *
           sqrt((2.0 * l1 + 1.0) * (2.0 * l3 + 1.0));
}
double int_sph_int2e_get_angularX_RME(int two_j1, int l1, int two_j2, int l2,
                                      int LL, int vv, double threeJ)
{
    return sqrt(6.0 * (two_j1 + 1.0) * (two_j2 + 1.0) * (2 * LL + 1.0) *
                (2 * l1 + 1.0) * (2 * l2 + 1.0)) *
           threeJ *
           CG_wigner_9j(2 * l1, 2 * l2, 2 * vv, 1, 1, 2, two_j1, two_j2,
                        2 * LL) *
           pow(-1, l1);
}
