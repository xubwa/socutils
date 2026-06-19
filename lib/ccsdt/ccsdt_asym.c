/*
 * ccsdt_asym.c -- C backend for the CCSDT T3-residual permutation
 * (anti)symmetrizers.
 *
 * The pure-numpy fullasym / P(...) operators each allocate several full
 * O(o^3 v^3) transposed temporaries; these single-pass gather kernels write
 * the result directly (36 / 9 / 3 signed reads per output element) and are
 * OpenMP-parallel over the leading occupied index.  Each kernel reproduces the
 * matching numpy routine in socutils.cc.zccsdt element-wise.
 *
 * out and in are complex128 (C99 double complex), shape (no,no,no,nv,nv,nv),
 * C-contiguous; out and in must not alias.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <complex.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* the 6 permutations of (0,1,2) and their parities (identity + two 3-cycles
 * with +1, three transpositions with -1) */
static const int P3[6][3] = {{0,1,2},{1,2,0},{2,0,1},{1,0,2},{0,2,1},{2,1,0}};
static const int S3[6]     = {  1,      1,      1,     -1,     -1,     -1   };

/* Full P(i/jk) P(a/bc): out[ijkabc] = sum_{p,q} S3[p]S3[q]
 *                                       in[ (ijk)[P3[p]], (abc)[P3[q]] ]. */
void ccsdt_fullasym(double complex *out, const double complex *in,
                    const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir;
    const long sc = 1, sb = nv, sa = nv*nv;       /* strides of c,b,a */
    const long sk = nv*nv*nv, sj = sk*no, si = sj*no;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++)
        for (int k = 0; k < nocc; k++) {
            const int o3[3] = {i, j, k};
            for (int a = 0; a < nvir; a++)
            for (int b = 0; b < nvir; b++)
            for (int c = 0; c < nvir; c++) {
                const int v3[3] = {a, b, c};
                double complex acc = 0.0;
                for (int p = 0; p < 6; p++) {
                    const long obase = o3[P3[p][0]]*si + o3[P3[p][1]]*sj
                                     + o3[P3[p][2]]*sk;
                    const int sgp = S3[p];
                    for (int q = 0; q < 6; q++) {
                        const long idx = obase + v3[P3[q][0]]*sa
                                       + v3[P3[q][1]]*sb + v3[P3[q][2]]*sc;
                        acc += (double)(sgp*S3[q]) * in[idx];
                    }
                }
                out[((( (long)i*no + j)*no + k)*nv + a)*nv*nv + b*nv + c] = acc;
            }
        }
    }
}

/* helper: generic 3-term operator on the vir (last three) axes.
 * out[..abc] = in[..abc] + s1*in[.. perm1(abc)] + s2*in[.. perm2(abc)] */
static void vir3(double complex *out, const double complex *in,
                 int no, int nv, const int perm1[3], int s1,
                 const int perm2[3], int s2)
{
    const long sc = 1, sb = nv, sa = (long)nv*nv;
#pragma omp parallel for schedule(static)
    for (int ijk = 0; ijk < no*no*no; ijk++) {
        const double complex *blk = in + (long)ijk*nv*nv*nv;
        double complex *ob = out + (long)ijk*nv*nv*nv;
        for (int a = 0; a < nv; a++)
        for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++) {
            const int v[3] = {a, b, c};
            ob[a*sa + b*sb + c] =
                blk[a*sa + b*sb + c]
              + (double)s1*blk[v[perm1[0]]*sa + v[perm1[1]]*sb + v[perm1[2]]*sc]
              + (double)s2*blk[v[perm2[0]]*sa + v[perm2[1]]*sb + v[perm2[2]]*sc];
        }
    }
}

/* helper: generic 3-term operator on the occ (first three) axes. */
static void occ3(double complex *out, const double complex *in,
                 int no, int nv, const int perm1[3], int s1,
                 const int perm2[3], int s2)
{
    const long blk = (long)nv*nv*nv;
    const long sk = blk, sj = blk*no, si = blk*no*no;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < no; i++)
    for (int j = 0; j < no; j++)
    for (int k = 0; k < no; k++) {
        const int o[3] = {i, j, k};
        const long base = (long)i*si + (long)j*sj + (long)k*sk;
        const long b1 = (long)o[perm1[0]]*si + (long)o[perm1[1]]*sj + (long)o[perm1[2]]*sk;
        const long b2 = (long)o[perm2[0]]*si + (long)o[perm2[1]]*sj + (long)o[perm2[2]]*sk;
        for (long e = 0; e < blk; e++)
            out[base + e] = in[base + e] + (double)s1*in[b1 + e] + (double)s2*in[b2 + e];
    }
}

/* P(a/bc): out[abc] = in[abc] - in[bac] - in[cba] */
void ccsdt_Pabc(double complex *out, const double complex *in, int no, int nv)
{   static const int p1[3]={1,0,2}, p2[3]={2,1,0};  vir3(out,in,no,nv,p1,-1,p2,-1); }

/* P(c/ab): out[abc] = in[abc] - in[cba] - in[acb] */
void ccsdt_Pc_ab(double complex *out, const double complex *in, int no, int nv)
{   static const int p1[3]={2,1,0}, p2[3]={0,2,1};  vir3(out,in,no,nv,p1,-1,p2,-1); }

/* P(i/jk): out[ijk] = in[ijk] - in[jik] - in[kji] */
void ccsdt_Pijk(double complex *out, const double complex *in, int no, int nv)
{   static const int p1[3]={1,0,2}, p2[3]={2,1,0};  occ3(out,in,no,nv,p1,-1,p2,-1); }

/* P(k/ij): out[ijk] = in[ijk] - in[kji] - in[ikj] */
void ccsdt_Pk_ij(double complex *out, const double complex *in, int no, int nv)
{   static const int p1[3]={2,1,0}, p2[3]={0,2,1};  occ3(out,in,no,nv,p1,-1,p2,-1); }

/* P(i/jk) P(a/bc): compose Pijk after Pabc via a scratch buffer. */
void ccsdt_Pijk_Pabc(double complex *out, const double complex *in,
                     int no, int nv)
{
    const size_t n = (size_t)no*no*no*nv*nv*nv;
    double complex *tmp = (double complex*)malloc(n*sizeof(double complex));
    if (!tmp) { memset(out, 0, n*sizeof(double complex)); return; }
    ccsdt_Pabc(tmp, in, no, nv);
    ccsdt_Pijk(out, tmp, no, nv);
    free(tmp);
}
