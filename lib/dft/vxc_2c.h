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
 *                          chain rule to v_n, v_m (noncollinear._ncol_core with deriv 1).
 *
 * Per sub-block of gs points: the AOs once (eval_select), for every density c = phi D_sub and its rows, the
 * functional, wv_k and phi_0 aow_k^T into the thread's V_k (2 nd GEMMs).  vbuf (nthreads, nd, nao, nao)
 * (GGA: the caller adds V + V^T), ex / ne (nthreads).  The caller assembles the spin blocks.  xctype 0 LDA,
 * 1 GGA.
 */
void SOCDFTvxc_fused_2c(double *vbuf, double *ex, double *ne, double *dms, int nd, int scheme, double s_thrd,
                        double *coords, double *weights, uint8_t *non0tab, int ngrid, int gs, double cutoff,
                        int xctype, gtoval_t eval, int *ao_loc, int *atm, int natm, int *bas, int nbas,
                        double *env, libxc_t xcf, int nfn, void *xc_arr, double *facs,
                        dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int nao = ao_loc[nbas], ncomp = xctype == 0 ? 1 : 4;
        const int nsub = (ngrid + gs - 1) / gs, outlen = xctype == 0 ? 3 : 6;
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
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
        double *rud = malloc(sizeof(double) * 2 * ncomp * gs);         /* libxc input (2, ncomp, n) */
        double *out = malloc(sizeof(double) * outlen * gs);
        double *wv = malloc(sizeof(double) * nd * ncomp * gs);         /* (nd, ncomp, n) */
        int *ix = malloc(sizeof(int) * nao);
        size_t *src = malloc(sizeof(size_t) * 2 * nao);
        double exs = 0, nes = 0;
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                const size_t rn = (size_t)ncomp * n;
                int i, j, g, k, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select(a, ix, src, full, n, xyz, (uint8_t *)msk, ncomp, cutoff, eval,
                                           ao_loc, atm, natm, bas, nbas, env);
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
                const double *w = weights + g0;
                if (scheme == 0) {
                        memcpy(rud, rho, sizeof(double) * 2 * rn);
                } else {
                        /* t = the density rows, s = |m| and the magnetisation rows projected on the local axis */
                        for (g = 0; g < n; g++) {
                                const double mx = rho[rn + g], my = rho[2 * rn + g], mz = rho[3 * rn + g];
                                const double sm = sqrt(mx * mx + my * my + mz * mz);
                                const double r0 = rho[g] > 1e-300 ? rho[g] : 1e-300;
                                const int pol = sm > s_thrd * r0;
                                const double is = pol ? 1. / sm : 0.;
                                const double nx = mx * is, ny = my * is, nz = mz * is;
                                for (k = 0; k < ncomp; k++) {
                                        const double t = rho[k * n + g];
                                        double sv = nx * rho[rn + k * n + g] + ny * rho[2 * rn + k * n + g]
                                                  + nz * rho[3 * rn + k * n + g];
                                        if (k == 0) sv = pol ? sm : 0.;
                                        rud[k * n + g] = .5 * (t + sv);
                                        rud[rn + k * n + g] = .5 * (t - sv);
                                }
                        }
                }
                /* out (outlen, n): exc, v_a, v_b (, vs_aa, vs_ab, vs_bb) */
                memset(out, 0, sizeof(double) * outlen * n);
                xcf(nfn, xc_arr, facs, 1, 1, ncomp, n, outlen, rud, out);
                const double *va = out + n, *vb = out + 2 * n;
                const double *saa = out + 3 * n, *sab = out + 4 * n, *sbb = out + 5 * n;
                for (g = 0; g < n; g++) {
                        const double den = w[g] * (rud[g] + rud[rn + g]);
                        nes += den;
                        exs += den * out[g];
                }
                if (scheme == 0) {
                        /* dE / d (rho_a, rho_b) rows */
                        for (g = 0; g < n; g++) {
                                wv[g] = w[g] * va[g];
                                wv[rn + g] = w[g] * vb[g];
                        }
                        if (ncomp == 4)
                                for (k = 1; k < 4; k++)
                                        for (g = 0; g < n; g++) {
                                                const double ga = rud[k * n + g], gb = rud[rn + k * n + g];
                                                wv[k * n + g] = w[g] * (2 * saa[g] * ga + sab[g] * gb);
                                                wv[rn + k * n + g] = w[g] * (2 * sbb[g] * gb + sab[g] * ga);
                                        }
                } else {
                        for (g = 0; g < n; g++) {
                                const double mx = rho[rn + g], my = rho[2 * rn + g], mz = rho[3 * rn + g];
                                const double sm = sqrt(mx * mx + my * my + mz * mz);
                                const double r0 = rho[g] > 1e-300 ? rho[g] : 1e-300;
                                const int pol = sm > s_thrd * r0;
                                const double is = pol ? 1. / sm : 0.;
                                const double nh[3] = {mx * is, my * is, mz * is};
                                double vt[4], vs[4], sv[4];
                                vt[0] = .5 * (va[g] + vb[g]);
                                vs[0] = .5 * (va[g] - vb[g]);
                                for (k = 1; k < ncomp; k++) {
                                        const double ga = rud[k * n + g], gb = rud[rn + k * n + g];
                                        const double da = 2 * saa[g] * ga + sab[g] * gb;
                                        const double db = 2 * sbb[g] * gb + sab[g] * ga;
                                        vt[k] = .5 * (da + db);
                                        vs[k] = .5 * (da - db);
                                        sv[k] = ga - gb;                /* the projected rows */
                                }
                                for (k = 0; k < ncomp; k++) wv[k * n + g] = w[g] * vt[k];
                                for (x = 0; x < 3; x++) {
                                        /* v_m_x: rows n_x vs_v; value row n_x vs_0 + sum_v>0 vs_v P_x,v,
                                         * P_x,v = (m_x,v - n_x s_v) / s */
                                        double v0 = nh[x] * vs[0];
                                        for (k = 1; k < ncomp; k++) {
                                                const double mxv = rho[(size_t)(1 + x) * rn + k * n + g];
                                                v0 += vs[k] * (mxv - nh[x] * sv[k]) * is;
                                                wv[(size_t)(1 + x) * rn + k * n + g] = w[g] * nh[x] * vs[k];
                                        }
                                        wv[(size_t)(1 + x) * rn + g] = w[g] * v0;
                                }
                        }
                }
                /* V_k += phi_0 (sum_c wv_k,c d_c phi)^T (GGA: the value row halved for V + V^T) */
                for (x = 0; x < nd; x++) {
                        double *wx = wv + (size_t)x * rn;
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
        }
        ex[tid] = exs;
        ne[tid] = nes;
        free(full); free(a); free(c); free(d); free(xyz); free(rho); free(rud); free(out); free(wv); free(ix);
        free(src);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}
