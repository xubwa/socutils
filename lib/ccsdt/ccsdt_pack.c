/*
 * ccsdt_pack.c -- C backend for socutils spinor CCSDT.
 *
 * Signed pack/unpack between the full antisymmetric amplitudes and their unique
 * blocks (the foundation of the antisymmetry-exploiting CCSDT, cf. pyscf
 * rccsdt's tri2block kernels).  Operating on complex128 (interleaved re,im,
 * == C99 double complex).
 *
 * t3[i,j,k,a,b,c] fully antisymmetric in (ijk) and (abc); unique block is
 * i<j<k, a<b<c, enumerated in the same nested order as the numpy reference
 * (for i: for j>i: for k>j ...).  unpack scatters each unique value to all
 * 36 signed permutations; pack gathers the unique values.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <complex.h>
#include <string.h>

/* the 6 permutations of (0,1,2) and their parities */
static const int PERM3[6][3] = {{0,1,2},{1,2,0},{2,0,1},{1,0,2},{0,2,1},{2,1,0}};
static const int SGN3[6]     = {  1,      1,      1,     -1,     -1,     -1   };
/* the 2 permutations of (0,1) */
static const int PERM2[2][2] = {{0,1},{1,0}};
static const int SGN2[2]     = {  1,    -1  };

static long c3(long n) { return n<3 ? 0 : n*(n-1)*(n-2)/6; }
static long c2(long n) { return n<2 ? 0 : n*(n-1)/2; }

/* ---- T3 ---- */

void ccsdt_unpack_t3(double complex *full, const double complex *packed,
                     const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir;
    const long nvt = c3(nv);
    memset(full, 0, (size_t)(no*no*no*nv*nv*nv) * sizeof(double complex));
    long po = 0;
    for (int i = 0; i < nocc; i++)
    for (int j = i+1; j < nocc; j++)
    for (int k = j+1; k < nocc; k++) {
        const int oo[3] = {i, j, k};
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++)
        for (int c = b+1; c < nvir; c++) {
            const int vv[3] = {a, b, c};
            const double complex val = packed[po*nvt + pv];
            for (int p = 0; p < 6; p++) {
                const int pi = oo[PERM3[p][0]], pj = oo[PERM3[p][1]], pk = oo[PERM3[p][2]];
                const long ooff = ((long)pi*no + pj)*no + pk;
                for (int q = 0; q < 6; q++) {
                    const int pa = vv[PERM3[q][0]], pb = vv[PERM3[q][1]], pc = vv[PERM3[q][2]];
                    const long idx = ((ooff*nv + pa)*nv + pb)*nv + pc;
                    full[idx] = (double)(SGN3[p]*SGN3[q]) * val;
                }
            }
            pv++;
        }
        po++;
    }
}

void ccsdt_pack_t3(const double complex *full, double complex *packed,
                   const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir;
    const long nvt = c3(nv);
    long po = 0;
    for (int i = 0; i < nocc; i++)
    for (int j = i+1; j < nocc; j++)
    for (int k = j+1; k < nocc; k++) {
        const long ooff = ((long)i*no + j)*no + k;
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++)
        for (int c = b+1; c < nvir; c++) {
            packed[po*nvt + pv] = full[((ooff*nv + a)*nv + b)*nv + c];
            pv++;
        }
        po++;
    }
}

/* ---- T2 ---- */

void ccsdt_unpack_t2(double complex *full, const double complex *packed,
                     const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir;
    const long nvp = c2(nv);
    memset(full, 0, (size_t)(no*no*nv*nv) * sizeof(double complex));
    long po = 0;
    for (int i = 0; i < nocc; i++)
    for (int j = i+1; j < nocc; j++) {
        const int oo[2] = {i, j};
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++) {
            const int vv[2] = {a, b};
            const double complex val = packed[po*nvp + pv];
            for (int p = 0; p < 2; p++) {
                const int pi = oo[PERM2[p][0]], pj = oo[PERM2[p][1]];
                for (int q = 0; q < 2; q++) {
                    const int pa = vv[PERM2[q][0]], pb = vv[PERM2[q][1]];
                    full[(((long)pi*no + pj)*nv + pa)*nv + pb] = (double)(SGN2[p]*SGN2[q]) * val;
                }
            }
            pv++;
        }
        po++;
    }
}

void ccsdt_pack_t2(const double complex *full, double complex *packed,
                   const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir;
    const long nvp = c2(nv);
    long po = 0;
    for (int i = 0; i < nocc; i++)
    for (int j = i+1; j < nocc; j++) {
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++) {
            packed[po*nvp + pv] = full[(((long)i*no + j)*nv + a)*nv + b];
            pv++;
        }
        po++;
    }
}
