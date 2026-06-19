/*
 * x2c_int_sph.h
 *
 * C translation of the INT_SPH class (include/int_sph.h): spherical atomic
 * integrals in the 2-spinor basis.  Only the functions reachable from the
 * amfi / atm_integrals / pcc_K entry points are translated; the GENBAS
 * file reader, the generic get_h2e_JK and the unused gaunt/gauge
 * non-compact drivers are intentionally omitted.
 *
 * The implementation is split like the C++ sources:
 *   x2c_int_sph_basic.c  - constructor, auxiliary radial integrals,
 *                          generic radial/angular helpers
 *   x2c_int_sph.c        - get_h1e, get_h2e_JK_compact, get_h2e_JK_direct,
 *                          get_h2eSD_JK_direct
 *   x2c_int_sph_gaunt.c  - Gaunt angular helpers + compact/direct drivers
 *   x2c_int_sph_gauge.c  - gauge radial/angular helpers + compact/direct
 */

#ifndef X2C_INT_SPH_H
#define X2C_INT_SPH_H

#include <stdbool.h>

#include "x2c_general.h"
#include "x2c_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    intShell* shell_list; /* length size_shell */
    irrep_jm* irrep_list; /* length Nirrep */
    int atomNumber, Nirrep;
    int size_gtoc, size_gtou, size_shell;
    int size_gtoc_spinor, size_gtou_spinor;
    const char* atomName; /* points into elem_list */
} INT_SPH;

/* Programmatic constructor (uncontracted basis), mirrors
   INT_SPH::INT_SPH(atom_number, nshell, nbas, shell, exp_a). */
INT_SPH* int_sph_new(int atom_number, int nshell, int nbas, const int* shell,
                     const double* exp_a);
void int_sph_free(INT_SPH* self);

/* ---- one-electron integrals -------------------------------------- */
/* intType: "overlap", "kinetic", "nuc_attra", "s_p_nuc_s_p",
   "s_p_nuc_s_p_sf", "nucGau_attra", "s_p_nucGau_s_p",
   "s_p_nucGau_s_p_sf", ...  Returns Nirrep matrices. */
vmat_t int_sph_get_h1e(const INT_SPH* self, const char* intType);

/* ---- two-electron integrals --------------------------------------- */
int2eJK int_sph_get_h2e_JK_compact(const INT_SPH* self, const char* intType,
                                   int occMaxL);
void int_sph_get_h2e_JK_direct(INT_SPH* self, int2eJK* LLLL, int2eJK* SSLL,
                               int2eJK* SSSS, int occMaxL, bool spinFree);
void int_sph_get_h2eSD_JK_direct(INT_SPH* self, int2eJK* SSLL, int2eJK* SSSS,
                                 int occMaxL);
int2eJK int_sph_get_h2e_JK_gaunt_compact(const INT_SPH* self,
                                         const char* intType, int occMaxL);
int2eJK int_sph_get_h2e_JK_gauntSF_compact(INT_SPH* self, const char* intType,
                                           int occMaxL);
void int_sph_get_h2e_JK_gaunt_direct(INT_SPH* self, int2eJK* LSLS,
                                     int2eJK* LSSL, int occMaxL,
                                     bool spinFree);
int2eJK int_sph_get_h2e_JK_gauge_compact(const INT_SPH* self,
                                         const char* intType, int occMaxL);
void int_sph_get_h2e_JK_gauge_direct(INT_SPH* self, int2eJK* LSLS,
                                     int2eJK* LSSL, int occMaxL,
                                     bool spinFree);

/* ---- internal helpers shared between the translation units -------- */
/* \int_0^inf r^l exp(-a r^2) dr */
double int_sph_auxiliary_1e(int l, double a);
double int_sph_auxiliary_2e_0_r(int l1, int l2, double a1, double a2);
double int_sph_auxiliary_2e_r_inf(int l1, int l2, double a1, double a2);
double int_sph_int2e_get_radial(int l1, double a1, int l2, double a2, int l3,
                                double a3, int l4, double a4, int LL);

double int_sph_get_radial_LLLL_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree);
double int_sph_get_radial_LLLL_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree);
double int_sph_get_radial_SSLL_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree);
double int_sph_get_radial_SSLL_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree);
double int_sph_get_radial_SSSS_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree);
double int_sph_get_radial_SSSS_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4,
                                 double radial_list[3][3], bool spinFree);
double int_sph_get_radial_LSLS_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree);
double int_sph_get_radial_LSLS_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree);
double int_sph_get_radial_LSSL_J(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree);
double int_sph_get_radial_LSSL_K(int lp, int lq, int LL, double a1, double a2,
                                 double a3, double a4, double lk1, double lk2,
                                 double lk3, double lk4, double radial_list[4],
                                 bool spinFree);

double int_sph_int2e_get_radial_gauge(int l1, double a1, int l2, double a2,
                                      int l3, double a3, int l4, double a4,
                                      int LL, int v1, int v2);

double int_sph_int2e_get_angular(int l1, int two_m1, int s1, int l2,
                                 int two_m2, int s2, int l3, int two_m3,
                                 int s3, int l4, int two_m4, int s4, int LL);
double int_sph_int2e_get_angular_J(int l1, int two_m1, int s1, int l2,
                                   int two_m2, int s2, int LL);
double int_sph_int2e_get_angular_K(int l1, int two_m1, int s1, int l2,
                                   int two_m2, int s2, int LL);

double int_sph_int2e_get_angular_gaunt_LSLS(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL);
double int_sph_int2e_get_angular_gaunt_LSSL(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL);
double int_sph_int2e_get_angular_gaunt_LSLS_9j(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int l3, int two_m3, int s3,
                                               int l4, int two_m4, int s4,
                                               int LL);
double int_sph_int2e_get_angular_gaunt_LSSL_9j(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int l3, int two_m3, int s3,
                                               int l4, int two_m4, int s4,
                                               int LL);
double int_sph_int2e_get_angular_gauge_LSLS(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int ll, int v1,
                                            int v2);
double int_sph_int2e_get_angular_gauge_LSSL(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int ll, int v1,
                                            int v2);
void int_sph_int2e_get_angular_gauntSF_LSLS(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL,
                                            double* lsls11, double* lsls12,
                                            double* lsls21, double* lsls22);
void int_sph_int2e_get_angular_gauntSF_LSSL(int l1, int two_m1, int s1, int l2,
                                            int two_m2, int s2, int l3,
                                            int two_m3, int s3, int l4,
                                            int two_m4, int s4, int LL,
                                            double* lssl11, double* lssl12,
                                            double* lssl21, double* lssl22);
double int_sph_int2e_get_angular_gauntSF_p1_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_p2_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_m1_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_m2_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_z1_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_z2_LS(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_p1_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_p2_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_m1_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_m2_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_z1_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);
double int_sph_int2e_get_angular_gauntSF_z2_SL(int l1, int two_m1, int s1,
                                               int l2, int two_m2, int s2,
                                               int LL, int MM);

double int_sph_int2e_get_threeSH(int l1, int m1, int l2, int m2, int l3,
                                 int m3, double threeJ);
double int_sph_int2e_get_angularX_RME(int two_j1, int l1, int two_j2, int l2,
                                      int LL, int vv, double threeJ);

#ifdef __cplusplus
}
#endif

#endif /* X2C_INT_SPH_H */
