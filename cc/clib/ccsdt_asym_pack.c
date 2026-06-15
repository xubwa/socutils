/*
 * ccsdt_asym_pack.c -- antisymmetrize-into-packed kernels for the CCSDT T3
 * residual.
 *
 * Each residual term is OP(X) with OP an (anti)symmetrizer and X a contraction
 * whose result already inherits t3's antisymmetry, so OP(X) is fully
 * antisymmetric and is fully determined by its unique block (i<j<k, a<b<c).
 * These kernels accumulate scale*OP(X) directly onto the packed residual Rp
 * (shape (notri, nvtri), same nested enumeration as ccsdt_pack_t3), so the full
 * O(o^3 v^3) output is never materialized.
 *
 * Three input layouts, matching how each contraction is produced:
 *   pvir : X occ-restricted, shape (notri, nv, nv, nv); OP acts on vir only
 *          (op=0 P(a/bc): X[abc]-X[bac]-X[cba];  op=1 P(c/ab): X[abc]-X[cba]-X[acb])
 *   pocc : X vir-restricted, shape (no, no, no, nvtri); OP acts on occ only
 *          (op=0 P(i/jk): X[ijk]-X[jik]-X[kji];  op=1 P(k/ij): X[ijk]-X[kji]-X[ikj])
 *   full : X full (no,no,no,nv,nv,nv); op=0 fullasym (36 terms),
 *          op=1 P(i/jk)P(a/bc) (9 terms)
 *
 * Rp is accumulated (+=); zero it before the first call.  scale is real.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const int QP3[6][3] = {{0,1,2},{1,2,0},{2,0,1},{1,0,2},{0,2,1},{2,1,0}};
static const int QS3[6]     = {  1,      1,      1,     -1,     -1,     -1   };

static long ntri(long n) { return n<3 ? 0 : n*(n-1)*(n-2)/6; }

/* ---- pvir: X is (notri, nv, nv, nv), antisymmetrize vir into packed ---- */
void ccsdt_pvir_to_pack(double complex *Rp, const double complex *X,
                        const double scale, const int op,
                        const int nocc, const int nvir)
{
    const long nv = nvir, nvt = ntri(nv);
    const long sI = nv*nv*nv;                /* stride of one occ-triple block */
#pragma omp parallel for schedule(static)
    for (long po = 0; po < ntri(nocc); po++) {
        const double complex *XI = X + po*sI;
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++)
        for (int c = b+1; c < nvir; c++) {
            double complex val;
            const double complex abc = XI[(a*nv + b)*nv + c];
            if (op == 0)        /* P(a/bc): abc - bac - cba */
                val = abc - XI[(b*nv + a)*nv + c] - XI[(c*nv + b)*nv + a];
            else                /* P(c/ab): abc - cba - acb */
                val = abc - XI[(c*nv + b)*nv + a] - XI[(a*nv + c)*nv + b];
            Rp[po*nvt + pv] += scale * val;
            pv++;
        }
    }
}

/* ---- pocc: X is (no, no, no, nvtri), antisymmetrize occ into packed ---- */
void ccsdt_pocc_to_pack(double complex *Rp, const double complex *X,
                        const double scale, const int op,
                        const int nocc, const int nvir)
{
    const long no = nocc, nvt = ntri(nvir);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nocc; i++)
    for (int j = i+1; j < nocc; j++)
    for (int k = j+1; k < nocc; k++) {
        /* packed occ index po for (i<j<k) */
        long po = 0; int done = 0;
        for (int ii = 0; ii < nocc && !done; ii++)
        for (int jj = ii+1; jj < nocc && !done; jj++)
        for (int kk = jj+1; kk < nocc && !done; kk++) {
            if (ii==i && jj==j && kk==k) done = 1; else po++;
        }
        const double complex *Xijk = X + (((long)i*no + j)*no + k)*nvt;
        const double complex *Xjik = X + (((long)j*no + i)*no + k)*nvt;
        const double complex *Xkji = X + (((long)k*no + j)*no + i)*nvt;
        const double complex *Xikj = X + (((long)i*no + k)*no + j)*nvt;
        for (long pv = 0; pv < nvt; pv++) {
            double complex val;
            if (op == 0)        /* P(i/jk): ijk - jik - kji */
                val = Xijk[pv] - Xjik[pv] - Xkji[pv];
            else                /* P(k/ij): ijk - kji - ikj */
                val = Xijk[pv] - Xkji[pv] - Xikj[pv];
            Rp[po*nvt + pv] += scale * val;
        }
    }
}

/* ---- full: X is (no,no,no,nv,nv,nv) ---- */
void ccsdt_full_to_pack(double complex *Rp, const double complex *X,
                        const double scale, const int op,
                        const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir, nvt = ntri(nv);
    const long sk = nv*nv*nv, sj = sk*no, si = sj*no;
    const long sa = nv*nv, sb = nv;
    /* fullasym: all 6 perms; P(i/jk)P(a/bc): identity + the two transpositions
     * {0,3,5} = (012),(102),(210) of QP3 (signs +,-,-). */
    const int idx6[6] = {0,1,2,3,4,5};
    const int idx3[3] = {0,3,5};
    const int nperm = (op == 0) ? 6 : 3;
    const int *pl = (op == 0) ? idx6 : idx3;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nocc; i++)
    for (int j = i+1; j < nocc; j++)
    for (int k = j+1; k < nocc; k++) {
        long po = 0; int done = 0;
        for (int ii = 0; ii < nocc && !done; ii++)
        for (int jj = ii+1; jj < nocc && !done; jj++)
        for (int kk = jj+1; kk < nocc && !done; kk++) {
            if (ii==i && jj==j && kk==k) done = 1; else po++;
        }
        const int o3[3] = {i, j, k};
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++)
        for (int c = b+1; c < nvir; c++) {
            const int v3[3] = {a, b, c};
            double complex acc = 0.0;
            for (int pp = 0; pp < nperm; pp++) {
                const int p = pl[pp];
                const long ob = o3[QP3[p][0]]*si + o3[QP3[p][1]]*sj + o3[QP3[p][2]]*sk;
                for (int qq = 0; qq < nperm; qq++) {
                    const int q = pl[qq];
                    const long idx = ob + v3[QP3[q][0]]*sa + v3[QP3[q][1]]*sb + v3[QP3[q][2]];
                    acc += (double)(QS3[p]*QS3[q]) * X[idx];
                }
            }
            Rp[po*nvt + pv] += scale * acc;
            pv++;
        }
    }
}
