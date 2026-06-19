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
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const int QP3[6][3] = {{0,1,2},{1,2,0},{2,0,1},{1,0,2},{0,2,1},{2,1,0}};
static const int QS3[6]     = {  1,      1,      1,     -1,     -1,     -1   };

static long ntri(long n) { return n<3 ? 0 : n*(n-1)*(n-2)/6; }

/* number of ordered triples (a<b<c) whose first element is < i */
static long base_tri(long i, long n)
{
    long s = 0;
    for (long a = 0; a < i; a++) s += (n-1-a)*(n-2-a)/2;
    return s;
}

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

/* ---- ring: X reduced to (no, nv, opair, vpair) = (i, a, [j<k], [b<c]) ----
 * accumulates scale * P(i/jk)P(a/bc)(X_full) onto the packed block, where
 * X_full[i,j,k,a,b,c] = Xr[i,a,pair(j,k),pair(b,c)] * sgn(j,k) * sgn(b,c).
 * The 9-term projector reads Xr at its (first, second<third)-pair layout. */
void ccsdt_ring_to_pack(double complex *Rp, const double complex *Xr,
                        const double scale, const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir, nvt = ntri(nv);
    const long opair = no*(no-1)/2, vpair = nv*(nv-1)/2;
    const long sQ = 1, sP = vpair, sa = opair*vpair, si = nv*sa;
    /* pair-index lookup tables (symmetric, sorted enumeration) */
    int *opi = (int*)malloc((size_t)no*no*sizeof(int));
    int *vpi = (int*)malloc((size_t)nv*nv*sizeof(int));
    { int p = 0; for (int x = 0; x < no; x++) for (int y = x+1; y < no; y++)
                     { opi[x*no+y] = p; opi[y*no+x] = p; p++; } }
    { int p = 0; for (int x = 0; x < nv; x++) for (int y = x+1; y < nv; y++)
                     { vpi[x*nv+y] = p; vpi[y*nv+x] = p; p++; } }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nocc; i++) {
        long po = base_tri(i, no);
        for (int j = i+1; j < nocc; j++)
        for (int k = j+1; k < nocc; k++, po++) {
            /* occ: P(i/jk) -> (first, pair, coeff) */
            const int fo[3] = {i, j, k};
            const long Po[3] = {opi[j*no+k], opi[i*no+k], opi[i*no+j]};
            const double co[3] = {1.0, -1.0, 1.0};
            long pv = 0;
            for (int a = 0; a < nvir; a++)
            for (int b = a+1; b < nvir; b++)
            for (int c = b+1; c < nvir; c++) {
                const int fg[3] = {a, b, c};
                const long Qo[3] = {vpi[b*nv+c], vpi[a*nv+c], vpi[a*nv+b]};
                const double cv[3] = {1.0, -1.0, 1.0};
                double complex acc = 0.0;
                for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    acc += co[p]*cv[q] *
                           Xr[fo[p]*si + fg[q]*sa + Po[p]*sP + Qo[q]*sQ];
                Rp[po*nvt + pv] += scale * acc;
                pv++;
            }
        }
    }
    free(opi); free(vpi);
}

/* ---- drive ----
 * X reduced to canonical layout (no, opair, vpair, nv) = (f, [<], [<], g),
 * representing a tensor antisymmetric in the occ-pair and vir-pair slots with
 * one free occ index (f, slot 0) and one free vir index (g, slot 5):
 *   X_full[d0..d5] = sgn(d1,d2) sgn(d3,d4) Xr[d0, opair(d1,d2), vpair(d3,d4), d5].
 * Accumulates scale * fullasym(X_full) onto the packed block (36 terms).  Both
 * drive sub-terms map onto this single layout and are summed before the call,
 * so the full O(o^3 v^3) intermediate is never formed. */
void ccsdt_drive_to_pack(double complex *Rp, const double complex *Xr,
                         const double scale, const int nocc, const int nvir)
{
    const long no = nocc, nv = nvir, nvt = ntri(nv);
    const long opair = no*(no-1)/2, vpair = nv*(nv-1)/2;
    const long sQ = nv, sP = vpair*nv, sf = opair*vpair*nv;
    int *opi = (int*)malloc((size_t)no*no*sizeof(int));
    int *vpi = (int*)malloc((size_t)nv*nv*sizeof(int));
    { int p = 0; for (int x = 0; x < no; x++) for (int y = x+1; y < no; y++)
                     { opi[x*no+y] = p; opi[y*no+x] = p; p++; } }
    { int p = 0; for (int x = 0; x < nv; x++) for (int y = x+1; y < nv; y++)
                     { vpi[x*nv+y] = p; vpi[y*nv+x] = p; p++; } }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nocc; i++) {
        long po = base_tri(i, no);
        for (int j = i+1; j < nocc; j++)
        for (int k = j+1; k < nocc; k++, po++) {
            const int o3[3] = {i, j, k};
            long pv = 0;
            for (int a = 0; a < nvir; a++)
            for (int b = a+1; b < nvir; b++)
            for (int c = b+1; c < nvir; c++) {
                const int v3[3] = {a, b, c};
                double complex acc = 0.0;
                for (int p = 0; p < 6; p++) {
                    const int d0 = o3[QP3[p][0]], d1 = o3[QP3[p][1]], d2 = o3[QP3[p][2]];
                    const double so = QS3[p] * (d1 < d2 ? 1.0 : -1.0);
                    const long base = d0*sf + (long)opi[d1*no+d2]*sP;
                    for (int q = 0; q < 6; q++) {
                        const int d3 = v3[QP3[q][0]], d4 = v3[QP3[q][1]], d5 = v3[QP3[q][2]];
                        const double sv = QS3[q] * (d3 < d4 ? 1.0 : -1.0);
                        acc += so*sv * Xr[base + (long)vpi[d3*nv+d4]*sQ + d5];
                    }
                }
                Rp[po*nvt + pv] += scale * acc;
                pv++;
            }
        }
    }
    free(opi); free(vpi);
}

/* ---- pp ladder: Xpp (notri, vpair, nv) = (I, [a<b], c); P(c/ab) into packed.
 *   P(c/ab)(X)[abc] = X[abc]-X[cba]-X[acb], X antisym in (a,b), so for a<b<c
 *   = Xpp[I,vp(a,b),c] + Xpp[I,vp(b,c),a] - Xpp[I,vp(a,c),b]. */
void ccsdt_pp_to_pack(double complex *Rp, const double complex *Xpp,
                      const double scale, const int nocc, const int nvir)
{
    const long nv = nvir, nvt = ntri(nv), vpair = nv*(nv-1)/2;
    const long sc = 1, sQ = nv, sI = vpair*nv;
    int *vpi = (int*)malloc((size_t)nv*nv*sizeof(int));
    { int p = 0; for (int x = 0; x < nv; x++) for (int y = x+1; y < nv; y++)
                     { vpi[x*nv+y] = p; vpi[y*nv+x] = p; p++; } }
#pragma omp parallel for schedule(static)
    for (long po = 0; po < ntri(nocc); po++) {
        const double complex *XI = Xpp + po*sI;
        long pv = 0;
        for (int a = 0; a < nvir; a++)
        for (int b = a+1; b < nvir; b++)
        for (int c = b+1; c < nvir; c++) {
            const double complex val = XI[(long)vpi[a*nv+b]*sQ + c]
                                     + XI[(long)vpi[b*nv+c]*sQ + a]
                                     - XI[(long)vpi[a*nv+c]*sQ + b];
            Rp[po*nvt + pv] += scale * val;
            pv++;
        }
    }
    free(vpi);
}

/* ---- hh ladder: Xhh (opair, no, nvtri) = ([i<j], k, A); P(k/ij) into packed.
 *   P(k/ij)(X)[ijk] = X[ijk]-X[kji]-X[ikj], X antisym in (i,j), so for i<j<k
 *   = Xhh[op(i,j),k,A] + Xhh[op(j,k),i,A] - Xhh[op(i,k),j,A]. */
void ccsdt_hh_to_pack(double complex *Rp, const double complex *Xhh,
                      const double scale, const int nocc, const int nvir)
{
    const long no = nocc, nvt = ntri(nvir);
    const long sk = nvt, sP = no*nvt;
    int *opi = (int*)malloc((size_t)no*no*sizeof(int));
    { int p = 0; for (int x = 0; x < no; x++) for (int y = x+1; y < no; y++)
                     { opi[x*no+y] = p; opi[y*no+x] = p; p++; } }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nocc; i++) {
        long po = base_tri(i, no);
        for (int j = i+1; j < nocc; j++)
        for (int k = j+1; k < nocc; k++, po++) {
            const double complex *Xij = Xhh + (long)opi[i*no+j]*sP + (long)k*sk;
            const double complex *Xjk = Xhh + (long)opi[j*no+k]*sP + (long)i*sk;
            const double complex *Xik = Xhh + (long)opi[i*no+k]*sP + (long)j*sk;
            for (long pv = 0; pv < nvt; pv++)
                Rp[po*nvt + pv] += scale * (Xij[pv] + Xjk[pv] - Xik[pv]);
        }
    }
    free(opi);
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
