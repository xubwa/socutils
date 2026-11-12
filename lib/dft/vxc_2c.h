/*
 * Two-component (GKS / spinor) Vxc in one pass over the grid (SOCDFTvxc_fused_2c), #included by vxc_rks.c
 * (eval_select, gtoval_t, libxc_t, dgemm_t): SOCDFTvxc_fused for nd real symmetric densities on the same AOs
 * of each sub-block,
 *
 *   scheme 0 (collinear):  dms = (D_a, D_b), libxc spin-polarised on (rho_a, rho_b) (pyscf's nr_uks);
 *   scheme 1 (locally collinear: socutils.dft.noncollinear 'ncol', and 'mcol' for LDA):
 *                          dms = (D_n, D_mx, D_my, D_mz), the real symmetric matrices of the density and of
 *                          the magnetisation; s = |m|, the gradient rows projected on n = m / |m| (the
 *                          unpolarised limit where s <= s_thrd rho), libxc on ((t + s) / 2, (t - s) / 2) and the
 *                          chain rule to v_n, v_m (noncollinear._ncol_core with deriv 1);
 *   scheme 2 (multi-collinear GGA, noncollinear.eval_xc_mcol): the same densities; points with s >= col_thrd
 *                          rho on the local principal axis (_mcol_paxis), the others averaged over the nu
 *                          half-sphere directions u / weights wu by lib/dft/mcol.c (SOCmcol_project, libxc
 *                          spin-polarised to second derivatives, SOCmcol_sphere; xcmap the xcfun order of the
 *                          xclen raw libxc rows), per sub-block.
 *
 * Per sub-block of gs points: the AOs once (eval_select), for every density c = phi D_sub and its rows, the
 * functional, wv_k = w dE/d(rows of density k) and phi_0 aow_k^T into the thread's V_k (2 nd GEMMs).  vbuf
 * (nthreads, nd, nao, nao) (GGA: the caller adds V + V^T), ex / ne (nthreads).  The caller assembles the spin
 * blocks.  xctype 0 LDA, 1 GGA.
 */
void SOCmcol_project(double *rho_ud, const double *rho_tm, const double *u, int nvar, int ng, int nu);
void SOCmcol_sphere(double *exc, double *vxc, double *fxc, const double *rho_tm, const double *u,
                    const double *wu, const double *xc, const int *xcmap, int nvar, int ng, int nu, int deriv);

/* the rows of (t, s) for libxc as (rho_a, rho_b) = ((t + s) / 2, (t - s) / 2): s = |m|, the gradient rows
 * projected on n = m / |m| (0 where s <= thr t); nh (3) the axis */
static void ts_rows(double *rud, size_t ld, int ncomp, const double *r, size_t lr, double thr, double *nh)
{
        /* r[(c * ncomp + v) * lr], c = 0 density, 1..3 magnetisation */
        const double mx = r[ncomp * lr], my = r[2 * ncomp * lr], mz = r[3 * ncomp * lr];
        const double sm = sqrt(mx * mx + my * my + mz * mz);
        const double r0 = r[0] > 1e-300 ? r[0] : 1e-300;
        const int pol = sm > thr * r0;
        const double is = pol ? 1. / sm : 0.;
        int v;
        nh[0] = mx * is;
        nh[1] = my * is;
        nh[2] = mz * is;
        for (v = 0; v < ncomp; v++) {
                const double t = r[v * lr];
                double sv = nh[0] * r[(ncomp + v) * lr] + nh[1] * r[(2 * ncomp + v) * lr]
                          + nh[2] * r[(3 * ncomp + v) * lr];
                if (v == 0) sv = pol ? sm : 0.;
                rud[v * ld] = .5 * (t + sv);
                rud[(ncomp + v) * ld] = .5 * (t - sv);
        }
}

/* dE/d (t, s rows) from libxc's spin-polarised output (outlen rows of ld): vt, vs (ncomp) */
static void ts_derivs(double *vt, double *vs, const double *out, size_t ld, const double *rud, size_t lrd,
                      int ncomp)
{
        vt[0] = .5 * (out[ld] + out[2 * ld]);
        vs[0] = .5 * (out[ld] - out[2 * ld]);
        int k;
        for (k = 1; k < ncomp; k++) {
                const double ga = rud[k * lrd], gb = rud[(ncomp + k) * lrd];
                const double saa = out[3 * ld], sab = out[4 * ld], sbb = out[5 * ld];
                const double da = 2 * saa * ga + sab * gb, db = 2 * sbb * gb + sab * ga;
                vt[k] = .5 * (da + db);
                vs[k] = .5 * (da - db);
        }
}

/* per-thread profile of SOCDFTvxc_fused_2c (wall seconds, SOCDFTvxc2c_stats): 0 AOs, 1 density rows (GEMMs),
 * 2 the functional on the local axis (collinear / ncol / mcol principal-axis points), 3 mcol projection, 4 libxc
 * on the sampled directions, 5 sphere contraction, 6 V (GEMMs, scatter); counts: 0 points, 1 sphere points,
 * 2 axis points, 3 libxc evaluations on the sampled directions (points x directions) */
#define VX2C_MAXT 256
static double vx2c_t[VX2C_MAXT][8];
static double vx2c_n[VX2C_MAXT][4];

void SOCDFTvxc2c_stats(double *out, int reset)
{
        int t, k;
        for (k = 0; k < 12; k++) out[k] = 0;
        for (t = 0; t < VX2C_MAXT; t++) {
                for (k = 0; k < 8; k++) out[k] += vx2c_t[t][k];
                for (k = 0; k < 4; k++) out[8 + k] += vx2c_n[t][k];
        }
        if (reset) {
                memset(vx2c_t, 0, sizeof(vx2c_t));
                memset(vx2c_n, 0, sizeof(vx2c_n));
        }
}

void SOCDFTvxc_fused_2c(double *vbuf, double *ex, double *ne, double *dms, int nd, int scheme, double s_thrd,
                        double col_thrd, double *u, double *wu, int nu, int *xcmap, int xclen,
                        double *coords, double *weights, uint8_t *non0tab, int ngrid, int gs, double cutoff,
                        int xctype, gtoval_t eval, int *ao_loc, int *atm, int natm, int *bas, int nbas,
                        double *env, libxc_t xcf, int nfn, void *xc_arr, double *facs,
                        dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int nao = ao_loc[nbas], ncomp = xctype == 0 ? 1 : 4;
        const int nsub = (ngrid + gs - 1) / gs, outlen = xctype == 0 ? 3 : 6;
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
        const int nus = scheme == 2 ? nu : 1;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *V = vbuf + (size_t)tid * nd * nn;
        double *full = malloc(sizeof(double) * ncomp * nao * gs);
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *c = malloc(sizeof(double) * nao * gs);
        double *d = malloc(sizeof(double) * nn);
        double *xyz = malloc(sizeof(double) * 3 * gs);
        double *rho = malloc(sizeof(double) * nd * ncomp * gs);        /* (nd, ncomp, n) */
        double *vf = malloc(sizeof(double) * nd * ncomp * gs);         /* dE/d rows, then w dE/d rows */
        double *eg = malloc(sizeof(double) * gs);                      /* exc per point */
        double *rud = malloc(sizeof(double) * 2 * ncomp * gs * nus);   /* libxc input (2, ncomp, np) */
        double *out = malloc(sizeof(double) * (scheme == 2 ? xclen : outlen) * gs * nus);
        double *tm = scheme == 2 ? malloc(sizeof(double) * 4 * ncomp * gs) : NULL;     /* compact rho_tm */
        double *vt2 = scheme == 2 ? malloc(sizeof(double) * 4 * ncomp * gs) : NULL;
        double *et2 = scheme == 2 ? malloc(sizeof(double) * gs) : NULL;
        double *nhs = malloc(sizeof(double) * 3 * gs);
        int *pts = malloc(sizeof(int) * gs);
        int *ix = malloc(sizeof(int) * nao);
        size_t *src = malloc(sizeof(size_t) * 2 * nao);
        double exs = 0, nes = 0;
        double *tp = vx2c_t[tid < VX2C_MAXT ? tid : VX2C_MAXT - 1];
        double *cp = vx2c_n[tid < VX2C_MAXT ? tid : VX2C_MAXT - 1];
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                const size_t rn = (size_t)ncomp * n;
                int i, j, g, k, x, v;
                double tw = omp_get_wtime(), tw1;
                cp[0] += n;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select(a, ix, src, full, n, xyz, (uint8_t *)msk, ncomp, cutoff, eval,
                                           ao_loc, atm, natm, bas, nbas, env);
                tw1 = omp_get_wtime(); tp[0] += tw1 - tw; tw = tw1;
                if (nl == 0) continue;
                /* the rows of every density: x = sum phi D phi, d x = 2 sum d phi D phi */
                for (x = 0; x < nd; x++) {
                        const double *dmx = dms + (size_t)x * nn;
                        double *rx = rho + (size_t)x * rn;
                        for (i = 0; i < nl; i++)
                                for (j = 0; j < nl; j++) d[i * nl + j] = dmx[(size_t)ix[i] * nao + ix[j]];
                        dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, d, &nl, &zero, c, &n);
                        memset(rx, 0, sizeof(double) * rn);
                        for (i = 0; i < nl; i++) {
                                const double *p = a + (size_t)i * n, *q = c + (size_t)i * n;
                                for (g = 0; g < n; g++) rx[g] += p[g] * q[g];
                        }
                        for (k = 1; k < ncomp; k++) {
                                double *rk = rx + (size_t)k * n;
                                for (i = 0; i < nl; i++) {
                                        const double *p = a + ((size_t)k * nl + i) * n, *q = c + (size_t)i * n;
                                        for (g = 0; g < n; g++) rk[g] += 2 * p[g] * q[g];
                                }
                        }
                }
                tw1 = omp_get_wtime(); tp[1] += tw1 - tw; tw = tw1;
                const double *w = weights + g0;
                if (scheme == 0) {
                        memset(out, 0, sizeof(double) * outlen * n);
                        xcf(nfn, xc_arr, facs, 1, 1, ncomp, n, outlen, rho, out);
                        for (g = 0; g < n; g++) {
                                eg[g] = out[g];
                                vf[g] = out[n + g];
                                vf[rn + g] = out[2 * n + g];
                        }
                        if (ncomp == 4)
                                for (k = 1; k < 4; k++)
                                        for (g = 0; g < n; g++) {
                                                const double ga = rho[k * n + g], gb = rho[rn + k * n + g];
                                                vf[k * n + g] = 2 * out[3 * n + g] * ga + out[4 * n + g] * gb;
                                                vf[rn + k * n + g] = 2 * out[5 * n + g] * gb + out[4 * n + g] * ga;
                                        }
                } else {
                        /* the points on the local axis: all (scheme 1), s >= col_thrd rho (scheme 2) */
                        int np = 0, nsph = 0;
                        for (g = 0; g < n; g++) {
                                int on = 1;
                                if (scheme == 2) {
                                        const double mx = rho[rn + g], my = rho[2 * rn + g], mz = rho[3 * rn + g];
                                        on = sqrt(mx * mx + my * my + mz * mz) >= col_thrd * rho[g];
                                }
                                if (on) pts[np++] = g;
                                else pts[n - 1 - nsph++] = g;
                        }
                        if (np > 0) {
                                const double thr = scheme == 1 ? s_thrd : 0.;
                                for (j = 0; j < np; j++)
                                        ts_rows(rud + j, np, ncomp, rho + pts[j], n, thr, nhs + 3 * j);
                                memset(out, 0, sizeof(double) * outlen * np);
                                xcf(nfn, xc_arr, facs, 1, 1, ncomp, np, outlen, rud, out);
                                for (j = 0; j < np; j++) {
                                        g = pts[j];
                                        const double *nh = nhs + 3 * j;
                                        double vt[4], vs[4];
                                        ts_derivs(vt, vs, out + j, np, rud + j, np, ncomp);
                                        eg[g] = out[j];
                                        for (v = 0; v < ncomp; v++) vf[v * n + g] = vt[v];
                                        for (x = 0; x < 3; x++) {
                                                double v0 = nh[x] * vs[0];
                                                for (v = 1; v < ncomp; v++) {
                                                        vf[(size_t)(1 + x) * rn + v * n + g] = nh[x] * vs[v];
                                                        if (scheme == 1) {
                                                                /* ncol: + sum_v>0 vs_v (m_x,v - n_x s_v) / s */
                                                                const double mx = rho[rn + g], my = rho[2 * rn + g],
                                                                             mz = rho[3 * rn + g];
                                                                const double sm = sqrt(mx * mx + my * my + mz * mz);
                                                                const double r0 = rho[g] > 1e-300 ? rho[g] : 1e-300;
                                                                if (sm > s_thrd * r0) {
                                                                        const double sv = rud[v * np + j]
                                                                                        - rud[(ncomp + v) * np + j];
                                                                        const double mxv = rho[(size_t)(1 + x) * rn
                                                                                               + v * n + g];
                                                                        v0 += vs[v] * (mxv - nh[x] * sv) / sm;
                                                                }
                                                        }
                                                }
                                                vf[(size_t)(1 + x) * rn + g] = v0;
                                        }
                                }
                        }
                        cp[1] += nsph;
                        cp[2] += np;
                        tw1 = omp_get_wtime(); tp[2] += tw1 - tw; tw = tw1;
                        if (nsph > 0) {
                                /* the sphere average (mcol.c) on the compact points */
                                const int *ps = pts + n - nsph;
                                for (x = 0; x < 4; x++)
                                        for (v = 0; v < ncomp; v++)
                                                for (j = 0; j < nsph; j++)
                                                        tm[((size_t)x * ncomp + v) * nsph + j] =
                                                                rho[(size_t)x * rn + v * n + ps[j]];
                                SOCmcol_project(rud, tm, u, ncomp, nsph, nu);
                                const size_t npt = (size_t)nsph * nu;
                                tw1 = omp_get_wtime(); tp[3] += tw1 - tw; tw = tw1;
                                memset(out, 0, sizeof(double) * xclen * npt);
                                xcf(nfn, xc_arr, facs, 1, 2, ncomp, (int)npt, xclen, rud, out);
                                cp[3] += npt;
                                tw1 = omp_get_wtime(); tp[4] += tw1 - tw; tw = tw1;
                                SOCmcol_sphere(et2, vt2, NULL, tm, u, wu, out, xcmap, ncomp, nsph, nu, 1);
                                for (j = 0; j < nsph; j++) {
                                        g = ps[j];
                                        eg[g] = et2[j];
                                        for (x = 0; x < 4; x++)
                                                for (v = 0; v < ncomp; v++)
                                                        vf[(size_t)x * rn + v * n + g] =
                                                                vt2[((size_t)x * ncomp + v) * nsph + j];
                                }
                        }
                }
                for (g = 0; g < n; g++) {
                        /* the density: rho_a + rho_b (scheme 0) or t */
                        const double r = scheme == 0 ? rho[g] + rho[rn + g] : rho[g];
                        const double den = w[g] * r;
                        nes += den;
                        exs += den * eg[g];
                }
                tw1 = omp_get_wtime(); tp[5] += tw1 - tw; tw = tw1;
                /* V_k += phi_0 (sum_c w vf_k,c d_c phi)^T (GGA: the value row halved for V + V^T) */
                for (x = 0; x < nd; x++) {
                        double *wx = vf + (size_t)x * rn;
                        for (k = 0; k < ncomp; k++)
                                for (g = 0; g < n; g++) wx[k * n + g] *= w[g];
                        if (ncomp == 4)
                                for (g = 0; g < n; g++) wx[g] *= .5;
                        for (i = 0; i < nl; i++) {
                                double *o = c + (size_t)i * n;
                                const double *p = a + (size_t)i * n;
                                for (g = 0; g < n; g++) o[g] = wx[g] * p[g];
                                for (k = 1; k < ncomp; k++) {
                                        const double *pk = a + ((size_t)k * nl + i) * n, *wk = wx + (size_t)k * n;
                                        for (g = 0; g < n; g++) o[g] += wk[g] * pk[g];
                                }
                        }
                        dgemm_("T", "N", &nl, &nl, &n, &one, a, &n, c, &n, &zero, d, &nl);
                        double *Vx = V + (size_t)x * nn;
                        for (i = 0; i < nl; i++) {
                                double *row = Vx + (size_t)ix[i] * nao;
                                for (j = 0; j < nl; j++) row[ix[j]] += d[i * nl + j];
                        }
                }
                tp[6] += omp_get_wtime() - tw;
        }
        ex[tid] = exs;
        ne[tid] = nes;
        free(full); free(a); free(c); free(d); free(xyz); free(rho); free(vf); free(eg); free(rud); free(out);
        free(tm); free(vt2); free(et2); free(nhs); free(pts); free(ix); free(src);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}
