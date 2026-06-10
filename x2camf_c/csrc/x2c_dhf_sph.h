/*
 * x2c_dhf_sph.h
 *
 * C translation of the DHF_SPH / DHF_SPH_CA classes (include/dhf_sph.h,
 * include/dhf_sph_ca.h): atomic Dirac-Hartree-Fock in the j-adapted
 * spinor basis, X2C transformations and the AMFI / PCC integrals.
 *
 * Translation layout (mirrors the C++ sources):
 *   x2c_dhf_sph.c     - constructor, SCF, Fock builds, amfi evaluation
 *   x2c_dhf_sph_ca.c  - average-of-configuration open-shell variant
 *   x2c_dhf_sph_pcc.c - x2c2ePCC / x2c2ePCC_K / h_x2c2e and the
 *                       evaluateFock_2e / _J / _K helpers
 *
 * The basisGenerator, radialDensity, coreIonization and get_coeff_bs
 * members are not used by the library entry points and are not
 * translated.
 */

#ifndef X2C_DHF_SPH_H
#define X2C_DHF_SPH_H

#include <stdbool.h>

#include "x2c_general.h"
#include "x2c_int_sph.h"
#include "x2c_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* extension block carrying the DHF_SPH_CA private members; ca != NULL on a
   handle created by dhf_sph_ca_new() selects the CA overrides */
typedef struct
{
    int openShell, NOpenShells;
    /* std::vector<double> NN_list, MM_list, f_list (length NOpenShells) */
    int n_shells; /* == NOpenShells */
    double *NN_list, *MM_list, *f_list;
    /* vector<vVectorXd> occNumberShells (length NOpenShells + 2) */
    int n_occShells;
    vvecd_t* occNumberShells;
    /* Matrix<vMatrixXd,-1,1> densityShells (length NOpenShells + 1) */
    int n_denShells;
    vmat_t* densityShells;
} DHF_CA_EXT;

typedef struct
{
    /* index maps between full and compact irrep lists */
    int *all2compact, *compact2all; /* lengths Nirrep / Nirrep_compact */
    vmat_t overlap, kinetic, WWW, Vnuc;
    int2eJK h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK, gauntLSLS_JK, gauntLSSL_JK;
    vmat_t density, fock_4c, h1e_4c, overlap_4c, overlap_half_i_4c, x2cXXX,
        x2cRRR;
    vvecd_t norm_s;
    vvecd_t occNumber;
    double d_density, nelec;
    bool converged, renormalizedSmall, with_gaunt, with_gauge, X_calculated;

    /* public members */
    intShell* shell_list; /* borrowed from INT_SPH */
    int size_shell;
    int size_basis_spinor, Nirrep, Nirrep_compact, occMax_irrep,
        occMax_irrep_compact;
    irrep_jm* irrep_list; /* borrowed from INT_SPH */
    int maxIter, size_DIIS, printLevel;
    double convControl, ene_scf;
    vmat_t coeff;
    vvecd_t ene_orb;

    DHF_CA_EXT* ca; /* NULL unless constructed with dhf_sph_ca_new */
} DHF_SPH;

/* ---- construction / destruction ----------------------------------- */
DHF_SPH* dhf_sph_new(INT_SPH* int_sph, const char* filename, int printLevel,
                     bool spinFree, bool twoC, bool with_gaunt,
                     bool with_gauge, bool allInt, bool gaussian_nuc);
DHF_SPH* dhf_sph_ca_new(INT_SPH* int_sph, const char* filename,
                        int printLevel, bool spinFree, bool twoC,
                        bool with_gaunt, bool with_gauge, bool allInt,
                        bool gaussian_nuc);
void dhf_sph_free(DHF_SPH* self);

/* ---- SCF ----------------------------------------------------------- */
/* Dispatches to the CA override when self->ca != NULL. */
void dhf_sph_run_scf(DHF_SPH* self, bool twoC, bool renormSmall);
void dhf_sph_run_scf_base(DHF_SPH* self, bool twoC, bool renormSmall);
void dhf_sph_ca_run_scf(DHF_SPH* self, bool twoC, bool renormSmall);

/* ---- amfi SOC integrals -------------------------------------------- */
vmat_t dhf_sph_get_amfi_unc(DHF_SPH* self, INT_SPH* int_sph, bool twoC,
                            const char* Xmethod, bool amfi_with_gaunt,
                            bool amfi_with_gauge, bool amfi4c, bool sd_gaunt);
vmat_t dhf_sph_get_amfi_unc_int2e(DHF_SPH* self, const int2eJK* h2eSSLL_SD,
                                  const int2eJK* h2eSSSS_SD,
                                  const int2eJK* gauntLSLS_SD,
                                  const int2eJK* gauntLSSL_SD,
                                  const vmat_t* density_, const char* Xmethod,
                                  bool amfi_with_gaunt, bool amfi4c);
vmat_t dhf_sph_get_amfi_unc_2c(DHF_SPH* self, const int2eJK* h2eSSLL_SD,
                               const int2eJK* h2eSSSS_SD,
                               bool amfi_with_gaunt, bool amfi4c);

/* ---- x2c2e picture change (x2c_dhf_sph_pcc.c) ----------------------- */
vmat_t dhf_sph_x2c2ePCC(DHF_SPH* self, bool amfi4c, vmat_t* coeff2c);
vmat_t dhf_sph_x2c2ePCC_K(DHF_SPH* self, bool amfi4c, vmat_t* coeff2c);
vmat_t dhf_sph_h_x2c2e(DHF_SPH* self, vmat_t* coeff2c);

/* ---- getters (deep copies, caller frees) ---------------------------- */
vmat_t dhf_sph_get_fock_4c(DHF_SPH* self);
vmat_t dhf_sph_get_fock_fw(DHF_SPH* self);
vmat_t dhf_sph_get_fock_4c_K(DHF_SPH* self, const vmat_t* den, bool twoC);
vmat_t dhf_sph_get_fock_4c_2ePart(DHF_SPH* self, const vmat_t* den,
                                  bool twoC);
vmat_t dhf_sph_get_h1e_4c(DHF_SPH* self);
vmat_t dhf_sph_get_overlap_4c(DHF_SPH* self);
vmat_t dhf_sph_get_density(DHF_SPH* self);
vmat_t dhf_sph_get_density_fw(DHF_SPH* self);
vvecd_t dhf_sph_get_occNumber(DHF_SPH* self);
vmat_t dhf_sph_get_X(DHF_SPH* self);
vmat_t dhf_sph_get_R(DHF_SPH* self);
void dhf_sph_set_h1e_4c(DHF_SPH* self, const vmat_t* inputM, bool addto);

/* ---- internal helpers shared between the translation units ---------- */
void dhf_sph_setOCC(DHF_SPH* self, const char* filename,
                    const char* atomName);
void dhf_sph_renormalize_small(DHF_SPH* self);
void dhf_sph_renormalize_h2e(DHF_SPH* self, int2eJK* h2e,
                             const char* intType);
void dhf_sph_symmetrize_h2e(DHF_SPH* self, bool twoC);
void dhf_sph_symmetrize_JK(DHF_SPH* self, int2eJK* h2e, int Ncompact);
void dhf_sph_symmetrize_JK_gaunt(DHF_SPH* self, int2eJK* h2e, int Ncompact);
/* evaluate the Fock matrix for irrep Iirrep into *fock (replaced) */
void dhf_sph_evaluateFock(DHF_SPH* self, mat_t* fock, bool twoC,
                          const vmat_t* den, int size, int Iirrep);
void dhf_sph_evaluateFock_2e(DHF_SPH* self, mat_t* fock, bool twoC,
                             const vmat_t* den, int size, int Iirrep);
void dhf_sph_evaluateFock_J(DHF_SPH* self, mat_t* fock, bool twoC,
                            const vmat_t* den, int size, int Iirrep);
void dhf_sph_evaluateFock_K(DHF_SPH* self, mat_t* fock, bool twoC,
                            const vmat_t* den, int size, int Iirrep);
mat_t dhf_sph_evaluateDensity_spinor(mat_t coeff_, vecd_t occNumber_,
                                     bool twoC);
vmat_t dhf_sph_evaluateDensity_spinor_irrep(DHF_SPH* self, bool twoC);
mat_t dhf_sph_evaluateErrorDIIS_fds(mat_t fock_, mat_t overlap_,
                                    mat_t density_);
mat_t dhf_sph_evaluateErrorDIIS_den(mat_t den_old, mat_t den_new);
void dhf_sph_eigensolverG_irrep(DHF_SPH* self, const vmat_t* inputM,
                                const vmat_t* s_h_i, vvecd_t* values,
                                vmat_t* vectors);
double dhf_sph_evaluateChange_irrep(DHF_SPH* self, const vmat_t* M1,
                                    const vmat_t* M2);

#ifdef __cplusplus
}
#endif

#endif /* X2C_DHF_SPH_H */
