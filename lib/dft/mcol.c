/*
 * Author Xubo Wang <wangxubo0201@outlook.com>
 *
 * Multi-collinear XC kernel (GGA / meta-GGA): sphere average of the collinear
 * functional, accumulated directly in the (rho, mx, my, mz) representation.
 *
 * For every grid point g and spin-projection direction u (weight w) the
 * collinear functional is evaluated by libxc at
 *     rho_a = (rho + s_u)/2,  rho_b = (rho - s_u)/2,   s_u = u . m
 * (every derivative row projected), and the multi-collinear quantities are
 *     rho e_mc = sum_u w [ rho e + s_u . v_s ]
 *     v_mc     = sum_u w C_u [ v + (v_s) + f . s_u ]
 *     f_mc     = sum_u w C_u C_u [ c f + k . s_u ],   c = 1 (tt), 2 (ts), 3 (ss)
 * with C_u mapping the (t, s) channels to (rho, u m).  This is the same
 * algebra as mcfun.eval_xc_eff, but the functional derivatives are taken from
 * the raw libxc output (derivatives to rho_a, rho_b, sigma_aa, sigma_ab,
 * sigma_bb [, tau_a, tau_b]) and the chain rule to gradient rows is done per
 * point in small local buffers: the third derivative is only ever needed
 * contracted with s_u, so the (2 nvar)^3 kernel tensor is never formed.
 *
 * xc layout: raw libxc output of pyscf.dft.libxc._eval_xc, shape (ncomp, npts),
 * point index p = g * nu + o.  xcmap[c] is the row of component c in xcfun
 * order (the unique index combinations i <= j <= k in lexicographic order),
 * i.e. the reordering eval_xc1 applies -- read through the map, not copied.
 */

#include <stdlib.h>
#include <string.h>

#define MAXQ 7      /* rho_a rho_b s_aa s_ab s_bb tau_a tau_b */
#define MAXV 5      /* rho dx dy dz tau */
#define MAXX 10     /* 2 * MAXV */

typedef struct {
        int nq, nvar, nx;
        int off1, off2, off3;
        int i2[MAXQ][MAXQ];
        int i3[MAXQ][MAXQ][MAXQ];
} Layout;

static void make_layout(Layout *L, int nvar)
{
        int i, j, k, n;
        L->nvar = nvar;
        L->nx = 2 * nvar;
        L->nq = (nvar == 5) ? 7 : 5;
        int nq = L->nq;
        L->off1 = 1;
        L->off2 = 1 + nq;
        L->off3 = L->off2 + nq * (nq + 1) / 2;
        n = 0;
        for (i = 0; i < nq; i++)
        for (j = i; j < nq; j++) {
                L->i2[i][j] = L->i2[j][i] = n++;
        }
        n = 0;
        for (i = 0; i < nq; i++)
        for (j = i; j < nq; j++)
        for (k = j; k < nq; k++) {
                L->i3[i][j][k] = L->i3[i][k][j] = L->i3[j][i][k] = n;
                L->i3[j][k][i] = L->i3[k][i][j] = L->i3[k][j][i] = n;
                n++;
        }
}

/* G[q][x] = d q / d x, x = (a rows, b rows).  Without the constant part it is
 * the directional derivative dG = sum_y d2q/dx dy delta_y (sigma is quadratic). */
static void build_G(const Layout *L, const double *a, const double *b,
                    double G[MAXQ][MAXX], int with_const)
{
        int nvar = L->nvar;
        int i;
        memset(G, 0, sizeof(double) * MAXQ * MAXX);
        if (with_const) {
                G[0][0] = 1;
                G[1][nvar] = 1;
                if (nvar == 5) {
                        G[5][4] = 1;
                        G[6][nvar + 4] = 1;
                }
        }
        for (i = 1; i < 4; i++) {
                G[2][i] = 2 * a[i];
                G[3][i] = b[i];
                G[3][nvar + i] = a[i];
                G[4][nvar + i] = 2 * b[i];
        }
}

/* M += sum_q c_q d2 q / dx dy */
static void add_hq(const Layout *L, const double *c, double M[MAXX][MAXX])
{
        int nvar = L->nvar;
        int i;
        for (i = 1; i < 4; i++) {
                M[i][i] += 2 * c[2];
                M[i][nvar + i] += c[3];
                M[nvar + i][i] += c[3];
                M[nvar + i][nvar + i] += 2 * c[4];
        }
}

/* M += A^T F B   (A, B: nq x nx; F: nq x nq) */
static void add_atfb(const Layout *L, double A[MAXQ][MAXX], double F[MAXQ][MAXQ],
                     double B[MAXQ][MAXX], double M[MAXX][MAXX])
{
        int nq = L->nq;
        int nx = L->nx;
        int q, r, x, y;
        double FB[MAXQ][MAXX];
        for (q = 0; q < nq; q++)
        for (y = 0; y < nx; y++) {
                double acc = 0;
                for (r = 0; r < nq; r++) {
                        acc += F[q][r] * B[r][y];
                }
                FB[q][y] = acc;
        }
        for (x = 0; x < nx; x++)
        for (q = 0; q < nq; q++) {
                double ax = A[q][x];
                if (ax == 0) continue;
                for (y = 0; y < nx; y++) {
                        M[x][y] += ax * FB[q][y];
                }
        }
}

/*
 * rho_tm: (4, nvar, ng)      u: (nu, 3)      wu: (nu)
 * xc:     (ncomp, ng * nu)   from eval_xc1(spin=1, deriv=deriv+1)
 * exc:    (ng)               vxc: (4, nvar, ng)     fxc: (4, nvar, 4, nvar, ng)
 * deriv = 1 (exc, vxc) or 2 (exc, vxc, fxc).  Outputs are overwritten.
 */
/*
 * Spin densities projected on the sampled directions, for libxc:
 * rho_ud[sigma, v, g*nu+o] = (rho[v,g] +/- u_o . m[:,v,g]) / 2
 */
void SOCmcol_project(double *rho_ud, const double *rho_tm, const double *u,
                     int nvar, int ng, int nu)
{
        const size_t Ng = ng;
        const size_t npts = (size_t)ng * nu;
#pragma omp parallel for schedule(static)
        for (int g = 0; g < ng; g++) {
                for (int v = 0; v < nvar; v++) {
                        const double r = rho_tm[v * Ng + g];
                        const double mx = rho_tm[(1 * nvar + v) * Ng + g];
                        const double my = rho_tm[(2 * nvar + v) * Ng + g];
                        const double mz = rho_tm[(3 * nvar + v) * Ng + g];
                        double *ra = rho_ud + v * npts + (size_t)g * nu;
                        double *rb = rho_ud + (nvar + v) * npts + (size_t)g * nu;
                        for (int o = 0; o < nu; o++) {
                                const double s = u[o*3] * mx + u[o*3+1] * my + u[o*3+2] * mz;
                                ra[o] = .5 * (r + s);
                                rb[o] = .5 * (r - s);
                        }
                }
        }
}

void SOCmcol_sphere(double *exc, double *vxc, double *fxc,
                    const double *rho_tm, const double *u, const double *wu,
                    const double *xc, const int *xcmap, int nvar, int ng, int nu,
                    int deriv)
{
        Layout L;
        make_layout(&L, nvar);
        const size_t npts = (size_t)ng * nu;
        const size_t Ng = ng;

#pragma omp parallel for schedule(dynamic, 8)
        for (int g = 0; g < ng; g++) {
                const int nq = L.nq;
                const int nx = L.nx;
                double rho[MAXV], m[3][MAXV];
                double s[MAXV], a[MAXV], b[MAXV], da[MAXV], db[MAXV], dlt[MAXX];
                double vq[MAXQ], F[MAXQ][MAXQ], Kd[MAXQ][MAXQ], dq[MAXQ], Fdq[MAXQ];
                double G[MAXQ][MAXX], dG[MAXQ][MAXX];
                double fud[MAXX][MAXX], kud[MAXX][MAXX];
                double vud[MAXX], fdl[MAXX];
                double erho = 0;
                double gt[MAXV] = {0}, gs[3][MAXV] = {{0}};
                double Htt[MAXV][MAXV] = {{0}};
                double Hts[3][MAXV][MAXV] = {{{0}}};
                double Hss[3][3][MAXV][MAXV] = {{{{0}}}};
                int v, w, k, l, q, r, t, x, y, o;

                for (v = 0; v < nvar; v++) {
                        rho[v] = rho_tm[v * Ng + g];
                        for (k = 0; k < 3; k++) {
                                m[k][v] = rho_tm[((k + 1) * nvar + v) * Ng + g];
                        }
                }

                for (o = 0; o < nu; o++) {
                        const size_t p = (size_t)g * nu + o;
                        const double wo = wu[o];
                        const double *uo = u + o * 3;
                        for (v = 0; v < nvar; v++) {
                                s[v] = uo[0] * m[0][v] + uo[1] * m[1][v] + uo[2] * m[2][v];
                                a[v] = .5 * (rho[v] + s[v]);
                                b[v] = .5 * (rho[v] - s[v]);
                                da[v] = .5 * s[v];
                                db[v] = -.5 * s[v];
                                dlt[v] = da[v];
                                dlt[nvar + v] = db[v];
                        }
                        const double e = xc[(size_t)xcmap[0] * npts + p];
                        for (q = 0; q < nq; q++) {
                                vq[q] = xc[(size_t)xcmap[L.off1 + q] * npts + p];
                                for (r = 0; r < nq; r++) {
                                        F[q][r] = xc[(size_t)xcmap[L.off2 + L.i2[q][r]] * npts + p];
                                }
                        }

                        build_G(&L, a, b, G, 1);
                        for (x = 0; x < nx; x++) {
                                double acc = 0;
                                for (q = 0; q < nq; q++) acc += vq[q] * G[q][x];
                                vud[x] = acc;
                        }
                        memset(fud, 0, sizeof(fud));
                        add_atfb(&L, G, F, G, fud);
                        add_hq(&L, vq, fud);
                        for (y = 0; y < nx; y++) {
                                double acc = 0;
                                for (x = 0; x < nx; x++) acc += dlt[x] * fud[x][y];
                                fdl[y] = acc;
                        }

                        /* energy: rho e + s . v_s,  v_s = (v_a - v_b)/2 */
                        double es = e * rho[0];
                        for (v = 0; v < nvar; v++) {
                                es += .5 * (vud[v] - vud[nvar + v]) * s[v];
                        }
                        erho += wo * es;

                        /* potential: g_t = v_t + (f.s)_t,  g_s = 2 v_s + (f.s)_s */
                        for (v = 0; v < nvar; v++) {
                                double vt = .5 * (vud[v] + vud[nvar + v]);
                                double vs = .5 * (vud[v] - vud[nvar + v]);
                                double ft = .5 * (fdl[v] + fdl[nvar + v]);
                                double fs = .5 * (fdl[v] - fdl[nvar + v]);
                                double gsv = 2 * vs + fs;
                                gt[v] += wo * (vt + ft);
                                for (k = 0; k < 3; k++) {
                                        gs[k][v] += wo * uo[k] * gsv;
                                }
                        }
                        if (deriv < 2) continue;

                        /* kernel: c f + (k . s), all in u/d first */
                        for (q = 0; q < nq; q++) {
                                double acc = 0;
                                for (x = 0; x < nx; x++) acc += G[q][x] * dlt[x];
                                dq[q] = acc;
                        }
                        for (q = 0; q < nq; q++) {
                                double acc2 = 0;
                                for (r = 0; r < nq; r++) {
                                        double acc = 0;
                                        for (t = 0; t < nq; t++) {
                                                acc += xc[(size_t)xcmap[L.off3 + L.i3[q][r][t]] * npts + p] * dq[t];
                                        }
                                        Kd[q][r] = acc;
                                        acc2 += F[q][r] * dq[r];
                                }
                                Fdq[q] = acc2;
                        }
                        build_G(&L, da, db, dG, 0);
                        memset(kud, 0, sizeof(kud));
                        add_atfb(&L, G, Kd, G, kud);
                        add_atfb(&L, G, F, dG, kud);
                        add_atfb(&L, dG, F, G, kud);
                        add_hq(&L, Fdq, kud);

                        for (v = 0; v < nvar; v++)
                        for (w = 0; w < nvar; w++) {
                                const int va = v, vb = nvar + v;
                                const int wa = w, wb = nvar + w;
                                /* ud -> ts on both indices */
                                double ftt = .25 * (fud[va][wa] + fud[va][wb] + fud[vb][wa] + fud[vb][wb]);
                                double fts = .25 * (fud[va][wa] - fud[va][wb] + fud[vb][wa] - fud[vb][wb]);
                                double fss = .25 * (fud[va][wa] - fud[va][wb] - fud[vb][wa] + fud[vb][wb]);
                                double ktt = .25 * (kud[va][wa] + kud[va][wb] + kud[vb][wa] + kud[vb][wb]);
                                double kts = .25 * (kud[va][wa] - kud[va][wb] + kud[vb][wa] - kud[vb][wb]);
                                double kss = .25 * (kud[va][wa] - kud[va][wb] - kud[vb][wa] + kud[vb][wb]);
                                double htt = ftt + ktt;
                                double hts = 2 * fts + kts;
                                double hss = 3 * fss + kss;
                                Htt[v][w] += wo * htt;
                                for (k = 0; k < 3; k++) {
                                        Hts[k][v][w] += wo * uo[k] * hts;
                                        for (l = k; l < 3; l++) {
                                                Hss[k][l][v][w] += wo * uo[k] * uo[l] * hss;
                                        }
                                }
                        }
                }

                exc[g] = (rho[0] != 0) ? erho / rho[0] : 0;
                for (v = 0; v < nvar; v++) {
                        vxc[v * Ng + g] = gt[v];
                        for (k = 0; k < 3; k++) {
                                vxc[((k + 1) * nvar + v) * Ng + g] = gs[k][v];
                        }
                }
                if (deriv < 2) continue;
#define FXC(c, x, c2, y) fxc[((((size_t)(c) * nvar + (x)) * 4 + (c2)) * nvar + (y)) * Ng + g]
                for (v = 0; v < nvar; v++)
                for (w = 0; w < nvar; w++) {
                        FXC(0, v, 0, w) = Htt[v][w];
                        for (k = 0; k < 3; k++) {
                                /* Hts[k][v][w] ~ h(t_v, s_w);  h(s_v, t_w) = h(t_w, s_v) */
                                FXC(0, v, k + 1, w) = Hts[k][v][w];
                                FXC(k + 1, v, 0, w) = Hts[k][w][v];
                                for (l = 0; l < 3; l++) {
                                        FXC(k + 1, v, l + 1, w) = (k <= l) ? Hss[k][l][v][w]
                                                                           : Hss[l][k][v][w];
                                }
                        }
                }
#undef FXC
        }
}

/*
 * Kernel contraction of the TDDFT response (complex density, real kernel):
 *   v[m, g] = sum_k f[k, m, g] * rho1[k, g]      k, m < n = 4 nvar
 * rho1, v: complex (interleaved re/im), shape (n, ng);  f: (n, n, ng).
 */
#define FXC_BLK 128
void SOCfxc_contract(double *v, const double *f, const double *rho1,
                     int n, int ng)
{
        const size_t Ng = ng;
        int nblk = (ng + FXC_BLK - 1) / FXC_BLK;
#pragma omp parallel for schedule(static)
        for (int ib = 0; ib < nblk; ib++) {
                int g0 = ib * FXC_BLK;
                int g1 = g0 + FXC_BLK < ng ? g0 + FXC_BLK : ng;
                int g, k, m;
                for (m = 0; m < n; m++) {
                        double *vm = v + 2 * (m * Ng);
                        for (g = g0; g < g1; g++) {
                                vm[2*g] = 0;
                                vm[2*g+1] = 0;
                        }
                        for (k = 0; k < n; k++) {
                                const double *fkm = f + ((size_t)k * n + m) * Ng;
                                const double *rk = rho1 + 2 * (k * Ng);
                                for (g = g0; g < g1; g++) {
                                        vm[2*g]   += fkm[g] * rk[2*g];
                                        vm[2*g+1] += fkm[g] * rk[2*g+1];
                                }
                        }
                }
        }
}
#undef FXC_BLK
