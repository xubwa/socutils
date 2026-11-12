/*
 * COSX grid response (included at the end of cosx.c, its static helpers in scope; socutils.grad.grid_response,
 * socutils.scf.cosx.COSX.grad(grid_response=True)).
 *
 * The COSX exchange energy of SOCDFTcosx_grad, E = sum_g w_g e_g, plus the grid-overlap term of the fit,
 *
 *     e_g = sum_ab H_a(g) A_ab(g) F_b(g) + 1/2 phi(g)^T Yg phi(g),   H = M^T phi, F = D^T phi,
 *
 * point by point: eg[g] = e_g (for the partition-weight derivatives) and the weighted point derivative
 *
 *     pg[x][g] = w_g d e_g / d r_g,x
 *              = w_g sum_i d_x phi_i Q_i  -  sum_pairs sum_t (sum_ab E_ab,t W_ab(g)) R_{t+1x}(g),
 *
 * Q = M G^F + D G^H + Yg phi (G^F = A F, G^H = A H) as the term (i) of SOCDFTcosx_grad, W_ab = w (H_a F_b +
 * H_b F_a) and d A_ab / d r = -sum_t E_ab,t R_{t+1x} (the point derivative of the Hermite R) as its term
 * (ii).  The point moves with its owner atom, so the gradient gains + sum_{g of A} pg[g] on A (and the
 * weight term).  By translational invariance sum_g pg[g] = -(the sum over the atoms of the fixed-grid
 * gradient), a check.  Same screening as SOCDFTcosx_grad.
 *
 * Entry points: SOCDFTcosx_grad_gr (cosx_grad_gr.h, generated from SOCDFTcosx_grad by gen_cosx_grad_gr.py):
 * the fixed-grid gradient and, from the same pass, eg / pg without the Yg term; SOCDFTcosx_gram_gr adds the
 * Yg term (Yg is known only after that pass: it needs K_plain[D]); SOCDFTcosx_grid_resp: all of it in a pass of
 * its own (reference).  On trpn_1_5 / def2-SVP, GRIDX3, 8 threads: fixed-grid gradient 11.5 s, with the fused
 * point terms 16.6-17.2 s, separate pass 28 s.
 */
/* V (nh, n) = E^T W row by row, then p -= V . R_{t+1} */
static void gr_point2(double *px, double *py, double *pz, const double *E, const double *R, const double *W,
                      double *V, int nab, int nh, int n, const int *sh, int nhl)
{
        int ab, t, g;
        memset(V, 0, sizeof(double) * nh * n);
        for (ab = 0; ab < nab; ab++) {
                const double *e = E + (size_t)ab * nh, *wr = W + (size_t)ab * n;
                for (t = 0; t < nh; t++) {
                        const double ee = e[t];
                        double *v = V + (size_t)t * n;
#pragma omp simd
                        for (g = 0; g < n; g++) v[g] += ee * wr[g];
                }
        }
        for (t = 0; t < nh; t++) {
                const double *v = V + (size_t)t * n;
                const double *rx = R + (size_t)sh[t] * n, *ry = R + (size_t)sh[nhl + t] * n;
                const double *rz = R + (size_t)sh[2 * nhl + t] * n;
#pragma omp simd
                for (g = 0; g < n; g++) {
                        px[g] -= v[g] * rx[g];
                        py[g] -= v[g] * ry[g];
                        pz[g] -= v[g] * rz[g];
                }
        }
}

/* gr_point2 with V = W^T E by dgemm for the larger pairs */
static void gr_point4(double *px, double *py, double *pz, const double *E, const double *R, const double *W,
                      double *V, int nab, int nh, int n, const int *sh, int nhl, dgemm_t dgemm_)
{
        if (nab * nh < 32) {
                gr_point2(px, py, pz, E, R, W, V, nab, nh, n, sh, nhl);
                return;
        }
        const double one = 1, zero = 0;
        int t, g;
        dgemm_("N", "T", &n, &nh, &nab, &one, W, &n, E, &nh, &zero, V, &n);
        for (t = 0; t < nh; t++) {
                const double *v = V + (size_t)t * n;
                const double *rx = R + (size_t)sh[t] * n, *ry = R + (size_t)sh[nhl + t] * n;
                const double *rz = R + (size_t)sh[2 * nhl + t] * n;
#pragma omp simd
                for (g = 0; g < n; g++) {
                        px[g] -= v[g] * rx[g];
                        py[g] -= v[g] * ry[g];
                        pz[g] -= v[g] * rz[g];
                }
        }
}

static int cx_gr_mode = 4;      /* the point contraction: 4 V by dgemm (2 for the small pairs), 2 row loops,
                                 * 0 off (timing only) */
void SOCDFTcosx_set_gr_mode(int v) { cx_gr_mode = v; }

int SOCDFTcosx_grid_resp(double *pg, double *eg, double *dm, double *dmm, double *yg, void *optp, double *coords,
                         double *weights, uint8_t *non0tab, int ngrid, int blk, double cutoff, double tol,
                         double tcut, int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                         dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        CXOpt *opt = optp;
        cx_tables();
        const int nao = ao_loc[nbas];
        const int nblk = (ngrid + blk - 1) / blk;
        const int Lmax = opt->lmax + 1, nhmax = NH(Lmax);
        int nab_max = 1, i0, err = 0;
        for (i0 = 0; i0 < opt->npair; i0++)
                if (opt->pairs[i0].nI * opt->pairs[i0].nJ > nab_max) nab_max = opt->pairs[i0].nI * opt->pairs[i0].nJ;
        int *shl_of = malloc(sizeof(int) * nao);
        for (i0 = 0; i0 < nbas; i0++) {
                int mu;
                for (mu = ao_loc[i0]; mu < ao_loc[i0 + 1]; mu++) shl_of[mu] = i0;
        }
        memset(pg, 0, sizeof(double) * 3 * (size_t)ngrid);
        memset(eg, 0, sizeof(double) * (size_t)ngrid);
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        const double one = 1, zero = 0;
        double *a = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *full = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *rmx = malloc(sizeof(double) * nao);
        int *ix = malloc(sizeof(int) * nao);
        int *runs = malloc(sizeof(int) * (2 * (nbas + 1) + nao));
        double *dr = malloc(sizeof(double) * (size_t)nao * nao);
        double *dr2 = malloc(sizeof(double) * (size_t)nao * nao);
        const int nhl = NH(opt->lmax);
        int *sh = malloc(sizeof(int) * 3 * nhl);
        double *F = malloc(sizeof(double) * (size_t)nao * blk);
        double *H = malloc(sizeof(double) * (size_t)nao * blk);
        double *GF = malloc(sizeof(double) * (size_t)nao * blk);
        double *GH = malloc(sizeof(double) * (size_t)nao * blk);
        double *Q = malloc(sizeof(double) * (size_t)nao * blk);
        double *YP = malloc(sizeof(double) * (size_t)nao * blk);
        double *fm = malloc(sizeof(double) * nbas), *ws = malloc(sizeof(double) * nbas);
        double *A = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *W = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *V = malloc(sizeof(double) * (size_t)nhmax * blk);
        double *bk = malloc(sizeof(double) * 4096);
        double *R = malloc(sizeof(double) * (size_t)nhmax * blk);
        double *Rwk = malloc(sizeof(double) * 2 * (size_t)nhmax * blk);
        double *S = malloc(sizeof(double) * (size_t)(Lmax + 1) * blk);
        double *X = malloc(sizeof(double) * 4 * blk), *Y = X + blk, *Z = Y + blk, *T = Z + blk;
        double *xyz = malloc(sizeof(double) * 3 * blk);
        double **farR = calloc(natm, sizeof(double *));
        int *stamp = malloc(sizeof(int) * natm);
        double *armin = malloc(sizeof(double) * natm);
        int at, i, oom = !a || !full || !rmx || !ix || !runs || !dr || !dr2 || !sh || !F || !H || !GF || !GH
                         || !Q || !YP || !fm || !ws || !A || !W || !V || !bk || !R || !Rwk || !S || !X || !xyz
                         || !farR || !stamp || !armin;
        if (!oom) {
                int t_;
                for (t_ = 0; t_ < nhl; t_++) {
                        sh[t_] = h_idx[h_t[t_] + 1][h_u[t_]][h_v[t_]];
                        sh[nhl + t_] = h_idx[h_t[t_]][h_u[t_] + 1][h_v[t_]];
                        sh[2 * nhl + t_] = h_idx[h_t[t_]][h_u[t_]][h_v[t_] + 1];
                }
                for (at = 0; at < natm; at++) {
                        stamp[at] = -1;
                        if (opt->atm_lmax[at] >= 0) {
                                farR[at] = malloc(sizeof(double) * NH(opt->atm_lmax[at] + 1) * blk);
                                if (!farR[at]) oom = 1;
                        }
                }
        }
        int b;
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < nblk; b++) {
                if (oom) continue;
                const int g0 = b * blk, n = g0 + blk <= ngrid ? blk : ngrid - g0;
                uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int k, g, ip, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select1(a, ix, runs, n, xyz, msk, cutoff, ao_loc, atm, bas, nbas, env, rmx, full);
                if (nl == 0) continue;
                const double *w = weights + g0;
                double *px = pg + g0, *py = pg + (size_t)ngrid + g0, *pz = pg + 2 * (size_t)ngrid + g0;
                double *pp[3] = {px, py, pz};
                for (i = 0; i < nl; i++) memcpy(dr + (size_t)i * nao, dm + (size_t)ix[i] * nao, sizeof(double) * nao);
                dgemm_("N", "T", &n, &nao, &nl, &one, a, &n, dr, &nao, &zero, F, &n);
                if (dmm == dm) {
                        memcpy(H, F, sizeof(double) * nao * n);
                } else {
                        for (i = 0; i < nl; i++) memcpy(dr2 + (size_t)i * nao, dmm + (size_t)ix[i] * nao, sizeof(double) * nao);
                        dgemm_("N", "T", &n, &nao, &nl, &one, a, &n, dr2, &nao, &zero, H, &n);
                }
                for (i = 0; i < nbas; i++) fm[i] = ws[i] = 0;
                {
                        int mu;
                        for (mu = 0; mu < nao; mu++) {
                                const double *f = F + (size_t)mu * n, *h = H + (size_t)mu * n;
                                double m = 0, s = 0;
                                for (g = 0; g < n; g++) {
                                        const double v = fabs(f[g]) + fabs(h[g]);
                                        m = v > m ? v : m;
                                        s += w[g] * v;
                                }
                                const int sh_ = shl_of[mu];
                                if (m > fm[sh_]) fm[sh_] = m;
                                if (s > ws[sh_]) ws[sh_] = s;
                        }
                }
                memset(GF, 0, sizeof(double) * nao * n);
                memset(GH, 0, sizeof(double) * nao * n);
                double cb[3] = {0, 0, 0}, rb = 0;
                for (g = 0; g < n; g++) {
                        cb[0] += xyz[g]; cb[1] += xyz[n + g]; cb[2] += xyz[2 * n + g];
                }
                cb[0] /= n; cb[1] /= n; cb[2] /= n;
                for (g = 0; g < n; g++) {
                        const double dx = xyz[g] - cb[0], dy = xyz[n + g] - cb[1], dz = xyz[2 * n + g] - cb[2];
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > rb) rb = r2;
                }
                rb = sqrt(rb);
                for (ip = 0; ip < opt->npair; ip++) {
                        const CXPair *pr = opt->pairs + ip;
                        const int I = pr->I, J = pr->J, both = I != J;
                        const double wgt = fm[J] * ws[I] + (both ? fm[I] * ws[J] : 0);
                        if (pr->bound == 0 || pr->bound * wgt < tol) continue;
                        const int L = pr->L, nh = NH(L), nI = pr->nI, nJ = pr->nJ, nab = nI * nJ;
                        double bsum = 0;
                        for (k = 0; k < pr->np; k++) {
                                const double *P = opt->pp_P + 3 * (pr->po + k);
                                const double dx = P[0] - cb[0], dy = P[1] - cb[1], dz = P[2] - cb[2];
                                const double dmin = sqrt(dx * dx + dy * dy + dz * dz) - rb;
                                double f = 1;
                                if (dmin > 0) {
                                        const double T0 = opt->pp_p[pr->po + k] * dmin * dmin;
                                        if (T0 > 1) {
                                                const double ir = 1 / sqrt(T0);
                                                double rp = ir, mx = 0;
                                                int dd;
                                                for (dd = 0; dd <= L + 1; dd++) {
                                                        const double v = cx_damp[dd] * rp;
                                                        if (v > mx) mx = v;
                                                        rp *= ir;
                                                }
                                                f = mx < 1 ? mx : 1;
                                        }
                                }
                                bk[k] = f * opt->pp_b[pr->po + k] * sqrt(2 * opt->pp_p[pr->po + k]);
                                bsum += bk[k];
                        }
                        if (bsum * wgt < tol) continue;
                        const size_t sz = (size_t)nab * nh;
                        const int I0 = ao_loc[I], J0 = ao_loc[J];
                        {
                                int ia, jb;
                                for (ia = 0; ia < nI; ia++)
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *wr = W + (size_t)(ia * nJ + jb) * n;
                                                const double *ha = H + (size_t)(I0 + ia) * n, *fb = F + (size_t)(J0 + jb) * n;
                                                const double *hb = H + (size_t)(J0 + jb) * n, *fa = F + (size_t)(I0 + ia) * n;
                                                if (both) {
#pragma omp simd
                                                        for (g = 0; g < n; g++) wr[g] = w[g] * (ha[g] * fb[g] + hb[g] * fa[g]);
                                                } else {
#pragma omp simd
                                                        for (g = 0; g < n; g++) wr[g] = w[g] * ha[g] * fb[g];
                                                }
                                        }
                        }
                        memset(A, 0, sizeof(double) * nab * n);
                        int k0 = 0;
                        /* the point derivative of one E R term: pg[x] -= sum_t (E^T W)_t R_{t+1x} */
#define GR_POINT(E_, R_) \
        if (cx_gr_mode == 4) gr_point4(px, py, pz, (E_), (R_), W, V, nab, nh, n, sh, nhl, dgemm_); \
        else if (cx_gr_mode) gr_point2(px, py, pz, (E_), (R_), W, V, nab, nh, n, sh, nhl);
                        if (pr->onec) {
                                at = pr->atom;
                                const double *Ra = env + atm[at * ATM_SLOTS + PTR_COORD];
                                const int La = opt->atm_lmax[at] + 1;
                                if (stamp[at] != b) {
                                        double rmin = 1e300;
                                        for (g = 0; g < n; g++) {
                                                X[g] = Ra[0] - xyz[g];
                                                Y[g] = Ra[1] - xyz[n + g];
                                                Z[g] = Ra[2] - xyz[2 * n + g];
                                                T[g] = X[g] * X[g] + Y[g] * Y[g] + Z[g] * Z[g];
                                                if (T[g] < rmin) rmin = T[g];
                                        }
                                        armin[at] = rmin;
                                        stamp[at] = b;
                                        if (rmin > 1e-12) {
                                                int m;
                                                rsqrt_vec(S + (size_t)La * n, T, n);
                                                for (g = 0; g < n; g++) {
                                                        const double ir = S[(size_t)La * n + g], ir2 = ir * ir;
                                                        double s = ir;
                                                        for (m = 0; m <= La; m++) {
                                                                S[m * n + g] = s;
                                                                s *= -(2 * m + 1) * ir2;
                                                        }
                                                }
                                                hermite_R_vec(farR[at], S, X, Y, Z, La, n, Rwk);
                                        } else {
                                                armin[at] = 0;
                                        }
                                }
                                const double rmin = armin[at];
                                if (rmin > 0) {
                                        const double *pv = opt->pp_p + pr->po;
                                        while (k0 < pr->np && pv[k0] * rmin >= tcut) k0++;
                                }
                                if (k0 > 0) {
                                        const double *Mo = opt->mdata + pr->mo + sz * (k0 - 1);
                                        e_times_r(A, Mo, farR[at], nab, nh, n);
                                        GR_POINT(Mo, farR[at])
                                }
                        }
                        for (k = k0; k < pr->np; k++) {
                                if (bk[k] * wgt < tol) continue;
                                const double p = opt->pp_p[pr->po + k];
                                const double *P = opt->pp_P + 3 * (pr->po + k);
                                const double *E = opt->edata + pr->eo + sz * k;
#ifdef CX_FUSED
                                cxs_table[L](A, nab, E, p, P, xyz, n, R);
#else
                                double tmin = 1e300, tmax = 0;
                                for (g = 0; g < n; g++) {
                                        X[g] = P[0] - xyz[g];
                                        Y[g] = P[1] - xyz[n + g];
                                        Z[g] = P[2] - xyz[2 * n + g];
                                        T[g] = p * (X[g] * X[g] + Y[g] * Y[g] + Z[g] * Z[g]);
                                        tmin = T[g] < tmin ? T[g] : tmin;
                                        tmax = T[g] > tmax ? T[g] : tmax;
                                }
                                boys_vec(S, n, T, L + 1, n, tmin, tmax);
                                double sc = 1;
                                int m;
                                for (m = 1; m <= L + 1; m++) {
                                        sc *= -2 * p;
                                        double *sm = S + (size_t)m * n;
                                        for (g = 0; g < n; g++) sm[g] *= sc;
                                }
                                hermite_R_vec(R, S, X, Y, Z, L + 1, n, Rwk);
                                e_times_r(A, E, R, nab, nh, n);
#endif
                                GR_POINT(E, R)
                        }
#undef GR_POINT
                        /* G^F = A F, G^H = A H (both directions) */
                        {
                                int ia, jb;
                                for (ia = 0; ia < nI; ia++) {
                                        double *gfi = GF + (size_t)(I0 + ia) * n, *ghi = GH + (size_t)(I0 + ia) * n;
                                        for (jb = 0; jb < nJ; jb++) {
                                                const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                const double *fj = F + (size_t)(J0 + jb) * n, *hj = H + (size_t)(J0 + jb) * n;
#pragma omp simd
                                                for (g = 0; g < n; g++) {
                                                        gfi[g] += ar[g] * fj[g];
                                                        ghi[g] += ar[g] * hj[g];
                                                }
                                        }
                                }
                                if (both) {
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *gfj = GF + (size_t)(J0 + jb) * n, *ghj = GH + (size_t)(J0 + jb) * n;
                                                for (ia = 0; ia < nI; ia++) {
                                                        const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                        const double *fi = F + (size_t)(I0 + ia) * n, *hi = H + (size_t)(I0 + ia) * n;
#pragma omp simd
                                                        for (g = 0; g < n; g++) {
                                                                gfj[g] += ar[g] * fi[g];
                                                                ghj[g] += ar[g] * hi[g];
                                                        }
                                                }
                                        }
                                }
                        }
                }
                /* e_g = sum_a H_a G^F_a (+ 1/2 phi^T Yg phi) */
                double *e = eg + g0;
                {
                        int mu;
                        for (mu = 0; mu < nao; mu++) {
                                const double *h = H + (size_t)mu * n, *gf = GF + (size_t)mu * n;
#pragma omp simd
                                for (g = 0; g < n; g++) e[g] += h[g] * gf[g];
                        }
                }
                /* Q (nl, n) = M[ix, :] G^F + D[ix, :] G^H (+ Yg[ix, ix] phi) */
                dgemm_("N", "N", &n, &nl, &nao, &one, GF, &n, dmm == dm ? dr : dr2, &nao, &zero, Q, &n);
                dgemm_("N", "N", &n, &nl, &nao, &one, GH, &n, dr, &nao, &one, Q, &n);
                if (yg) {
                        int j;
                        for (i = 0; i < nl; i++)
                                for (j = 0; j < nl; j++) dr[(size_t)i * nl + j] = yg[(size_t)ix[i] * nao + ix[j]];
                        dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, dr, &nl, &zero, YP, &n);
                        for (i = 0; i < nl; i++) {
                                const double *ph = a + (size_t)i * n, *yp = YP + (size_t)i * n;
                                double *q = Q + (size_t)i * n;
#pragma omp simd
                                for (g = 0; g < n; g++) {
                                        e[g] += .5 * ph[g] * yp[g];
                                        q[g] += yp[g];
                                }
                        }
                }
                for (i = 0; i < nl; i++) {
                        const double *q = Q + (size_t)i * n;
                        for (x = 0; x < 3; x++) {
                                const double *d = a + ((size_t)(1 + x) * nl + i) * n;
                                double *o = pp[x];
#pragma omp simd
                                for (g = 0; g < n; g++) o[g] += w[g] * d[g] * q[g];
                        }
                }
        }
        if (oom) {
#pragma omp atomic write
                err = 1;
        }
        free(a); free(full); free(rmx); free(ix); free(runs); free(dr); free(dr2); free(sh); free(F); free(H);
        free(GF); free(GH); free(Q); free(YP); free(fm); free(ws); free(A); free(W); free(V); free(bk); free(R);
        free(Rwk); free(S); free(X); free(xyz);
        if (farR)
                for (at = 0; at < natm; at++) free(farR[at]);
        free(farR); free(stamp); free(armin);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
        free(shl_of);
        return err;
}

#include "cosx_grad_gr.h"

/*
 * The grid-overlap term of the fit point by point (added): eg[g] += 1/2 phi^T Yg phi,
 * pg[x][g] += w_g d_x phi^T Yg phi (Yg symmetric), with the AOs of each block; no integrals.
 */
int SOCDFTcosx_gram_gr(double *pg, double *eg, double *yg, double *coords, double *weights, uint8_t *non0tab,
                       int ngrid, int blk, double cutoff, int *ao_loc, int *atm, int natm, int *bas, int nbas,
                       double *env, dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int nao = ao_loc[nbas];
        const int nblk = (ngrid + blk - 1) / blk;
        int err = 0;
        (void)natm;
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        const double one = 1, zero = 0;
        double *a = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *full = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *rmx = malloc(sizeof(double) * nao);
        int *ix = malloc(sizeof(int) * nao);
        int *runs = malloc(sizeof(int) * (2 * (nbas + 1) + nao));
        double *yr = malloc(sizeof(double) * (size_t)nao * nao);
        double *YP = malloc(sizeof(double) * (size_t)nao * blk);
        double *xyz = malloc(sizeof(double) * 3 * blk);
        const int oom = !a || !full || !rmx || !ix || !runs || !yr || !YP || !xyz;
        int b;
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < nblk; b++) {
                if (oom) continue;
                const int g0 = b * blk, n = g0 + blk <= ngrid ? blk : ngrid - g0;
                uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int i, j, g, k, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select1(a, ix, runs, n, xyz, msk, cutoff, ao_loc, atm, bas, nbas, env, rmx, full);
                if (nl == 0) continue;
                const double *w = weights + g0;
                for (i = 0; i < nl; i++)
                        for (j = 0; j < nl; j++) yr[(size_t)i * nl + j] = yg[(size_t)ix[i] * nao + ix[j]];
                dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, yr, &nl, &zero, YP, &n);
                double *e = eg + g0;
                for (i = 0; i < nl; i++) {
                        const double *ph = a + (size_t)i * n, *yp = YP + (size_t)i * n;
                        for (g = 0; g < n; g++) e[g] += .5 * ph[g] * yp[g];
                        for (x = 0; x < 3; x++) {
                                const double *d = a + ((size_t)(1 + x) * nl + i) * n;
                                double *o = pg + (size_t)x * ngrid + g0;
#pragma omp simd
                                for (g = 0; g < n; g++) o[g] += w[g] * d[g] * yp[g];
                        }
                }
        }
        if (oom) {
#pragma omp atomic write
                err = 1;
        }
        free(a); free(full); free(rmx); free(ix); free(runs); free(yr); free(YP); free(xyz);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
        return err;
}
