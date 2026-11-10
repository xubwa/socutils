/*
 * COSX Hessian pieces (included at the end of cosx.c, its static helpers in scope).
 *
 * SOCDFTcosx_fock1: the first-order plain K of all atoms at a fixed closed-shell density
 * D = 2 C C^T, in the occupied form of the Hessian's Fock1 path,
 *
 *     V[A, x]  = d K_plain[D] / dR_{A,x}   C     (natm, 3, nao, nocc)
 *     VT[A, x] = (d K_plain[D] / dR_{A,x})^T Y   (Y: e.g. P^T C for the fitted K)
 *
 * on the fixed grid (no grid response).  K_plain_mu nu = sum_g w phi_mu G_nu, G = A F,
 * F = D^T phi = 2 C psi (psi = C^T phi), and with Aoc(g) = A(g) C (the integrals applied to the
 * occupied orbitals, the multi-density contraction with constant "densities" C), the occupied
 * form needs no (nao, nao) derivative matrix:
 *
 *   U_i      = sum_nu C_nu i G_nu = 2 (C^T Aoc) psi
 *   (f1) V[A, x][mu in A]  -= sum_g w d_x phi_mu U
 *   (f2) U^{A,x}           += 2 (C^T Aoc) psi^{A,x},  psi^{A,x} = -sum_{k in A} C_k d_x phi_k
 *   (f3) U^{Y,y}           += C^T (d_Y A) F           (per pair, the centre and point derivatives)
 *        V[Y, y][mu]       += sum_g w phi_mu U^{Y,y}
 *   (t1) VT[A, x]          += sum_g G (w eta^{A,x})^T, eta = Y^T phi, eta^{A,x} its derivative
 *   (t2) VT[A, x]          += sum_g (A Fd^{A,x}) (w eta)^T,  A Fd = 2 Aoc psi^{A,x}
 *   (t3) VT[Y, y][a]       += sum_g ((d_Y A) F)_a (w eta)^T   (per pair)
 */

/* G[g][nu][i] += sum_lam A_nu lam(g) C[lam][i]: multi_contract with the same "density" C at every
 * point (nocc a multiple of 8) */
static void multi_contract_const(double *Ga, const double *Cm, const double *Abuf, const CXRec *rec, int nrec,
                                 int nao, int n, int nset)
{
        const size_t NQ = (size_t)nao * nset;
        int g, r, r1, rr;
        for (g = 0; g < n; g++) {
                const double *Fg = Cm;
                double *Gg = Ga + g * NQ;
                for (r = 0; r < nrec; r = r1) {
                        const int I0 = rec[r].I0, nI = rec[r].nI;
                        int anyI = 0;
                        for (r1 = r; r1 < nrec && rec[r1].I0 == I0; r1++) anyI |= rec[r1].cI;
                        const double *FI = Fg + (size_t)I0 * nset;
                        double *GI = Gg + (size_t)I0 * nset;
                        int ia, jb, c;
                        if (anyI) {
                                for (c = 0; c < nset; c += 8) {
                                        for (ia = 0; ia < nI; ia++) {
                                                double *ga = GI + (size_t)ia * nset + c;
                                                V8 acc = V8LOAD(ga);
                                                for (rr = r; rr < r1; rr++) {
                                                        const CXRec *rc = rec + rr;
                                                        if (!rc->cI) continue;
                                                        const int nJ = rc->nJ;
                                                        const double *ar = Abuf + rc->off + (size_t)g * nI * nJ + ia * nJ;
                                                        const double *FJ = Fg + (size_t)rc->J0 * nset + c;
                                                        for (jb = 0; jb < nJ; jb++)
                                                                acc = V8FMA(V8SET1(ar[jb]), V8LOAD(FJ + (size_t)jb * nset), acc);
                                                }
                                                V8STORE(ga, acc);
                                        }
                                }
                        }
                        for (rr = r; rr < r1; rr++) {
                                const CXRec *rc = rec + rr;
                                if (!rc->cJ) continue;
                                const int nJ = rc->nJ;
                                const double *Ar = Abuf + rc->off + (size_t)g * nI * nJ;
                                double *GJ = Gg + (size_t)rc->J0 * nset;
                                for (c = 0; c < nset; c += 8) {
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *gb = GJ + (size_t)jb * nset + c;
                                                V8 acc = V8LOAD(gb);
                                                for (ia = 0; ia < nI; ia++)
                                                        acc = V8FMA(V8SET1(Ar[ia * nJ + jb]), V8LOAD(FI + (size_t)ia * nset + c), acc);
                                                V8STORE(gb, acc);
                                        }
                                }
                        }
                }
        }
}

/* dAC[x] (nab, n) -= sum_t E_t R_{t+1x}: the derivative of A with respect to the grid point */
static void point_deriv(double *dAC, const double *E, const double *R, int nab, int nh, int n)
{
        int ab, t, x, g;
        for (x = 0; x < 3; x++) {
                double *d = dAC + (size_t)x * nab * n;
                for (ab = 0; ab < nab; ab++) {
                        double *o = d + (size_t)ab * n;
                        const double *e = E + (size_t)ab * nh;
                        for (t = 0; t < nh; t++) {
                                const double v = e[t];
                                if (v == 0) continue;
                                const int t1 = h_idx[h_t[t] + (x == 0)][h_u[t] + (x == 1)][h_v[t] + (x == 2)];
                                const double *r = R + (size_t)t1 * n;
#pragma omp simd
                                for (g = 0; g < n; g++) o[g] -= v * r[g];
                        }
                }
        }
}

/* r_a = sum_b dA_ab F_b (nI, n); s_b = sum_a dA_ab F_a (nJ, n) */
static void pair_rs(double *r, double *s, const double *dA, const double *F, int I0, int J0, int nI, int nJ,
                    int n, int both)
{
        int ia, jb, g;
        memset(r, 0, sizeof(double) * nI * n);
        if (both) memset(s, 0, sizeof(double) * nJ * n);
        for (ia = 0; ia < nI; ia++) {
                double *ra = r + (size_t)ia * n;
                const double *fa = F + (size_t)(I0 + ia) * n;
                for (jb = 0; jb < nJ; jb++) {
                        const double *d = dA + (size_t)(ia * nJ + jb) * n, *fb = F + (size_t)(J0 + jb) * n;
                        if (both) {
                                double *sb = s + (size_t)jb * n;
#pragma omp simd
                                for (g = 0; g < n; g++) {
                                        ra[g] += d[g] * fb[g];
                                        sb[g] += d[g] * fa[g];
                                }
                        } else {
#pragma omp simd
                                for (g = 0; g < n; g++) ra[g] += d[g] * fb[g];
                        }
                }
        }
}

/*
 * V, VT (natm, 3, nao, nocc) = see the header (overwritten); co, yo (nao, nocc) row-major, nocc a
 * multiple of 8 (zero columns padded).  Returns 0, or 1 when a buffer could not be allocated.
 */
int SOCDFTcosx_fock1(double *V, double *VT, double *co, double *yo, int nocc, void *optp, double *coords,
                     double *weights, uint8_t *non0tab, int ngrid, int blk, double cutoff, double tol,
                     double tcut, int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                     dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        CXOpt *opt = optp;
        cx_tables();
        const int nao = ao_loc[nbas];
        const int nblk = (ngrid + blk - 1) / blk;
        const int Lmax = opt->lmax + 1, nhmax = NH(Lmax);
        const size_t nvo = (size_t)natm * 3 * nao * nocc;
        int nab_max = 1, nI_max = 1, i0;
        for (i0 = 0; i0 < opt->npair; i0++) {
                if (opt->pairs[i0].nI * opt->pairs[i0].nJ > nab_max) nab_max = opt->pairs[i0].nI * opt->pairs[i0].nJ;
                if (opt->pairs[i0].nI > nI_max) nI_max = opt->pairs[i0].nI;
                if (opt->pairs[i0].nJ > nI_max) nI_max = opt->pairs[i0].nJ;
        }
        int *shl_of = malloc(sizeof(int) * nao);
        for (i0 = 0; i0 < nbas; i0++) {
                int mu;
                for (mu = ao_loc[i0]; mu < ao_loc[i0 + 1]; mu++) shl_of[mu] = i0;
        }
        int *atm_of_ao = malloc(sizeof(int) * nao);
        for (i0 = 0; i0 < nao; i0++) atm_of_ao[i0] = bas[shl_of[i0] * BAS_SLOTS + ATOM_OF];
        const int nthreads = omp_get_max_threads();
        double **vb = calloc(nthreads, sizeof(double *)), **vtb = calloc(nthreads, sizeof(double *));
        int err = 0;
        memset(V, 0, sizeof(double) * nvo);
        memset(VT, 0, sizeof(double) * nvo);
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        const double one = 1, zero = 0, two = 2, mone = -1;
        const size_t NQ = (size_t)nao * nocc;
        double *Vt = vb[tid] = calloc(nvo, sizeof(double));
        double *VTt = vtb[tid] = calloc(nvo, sizeof(double));
        double *a = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *full = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *rmx = malloc(sizeof(double) * nao);
        int *ix = malloc(sizeof(int) * nao);
        int *runs = malloc(sizeof(int) * (2 * (nbas + 1) + nao));
        double *cs = malloc(sizeof(double) * NQ), *ys = malloc(sizeof(double) * NQ);
        double *F = malloc(sizeof(double) * (size_t)nao * blk);
        double *GFb = malloc(sizeof(double) * (size_t)nao * blk);
        double *psi = malloc(sizeof(double) * (size_t)nocc * blk), *weta = malloc(sizeof(double) * (size_t)nocc * blk);
        double *Ub = malloc(sizeof(double) * (size_t)nocc * blk);
        /* local atoms of a block: psi^{A,x}, eta^{A,x} (3 nla, nocc, n) */
        double *psid = malloc(sizeof(double) * (size_t)3 * natm * nocc * blk);
        double *wetad = malloc(sizeof(double) * (size_t)3 * natm * nocc * blk);
        double *Bd = malloc(sizeof(double) * (size_t)nao * blk);
        int *la = malloc(sizeof(int) * (natm + 1) * 3);                /* atom, ix range */
        double *U = malloc(sizeof(double) * (size_t)natm * 3 * nocc * blk);
        char *touched = malloc(natm);
        double *fmax = malloc(sizeof(double) * nbas);
        double *A = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *dA = malloc(sizeof(double) * (size_t)6 * nab_max * blk);
        double *rs = malloc(sizeof(double) * (size_t)12 * nI_max * blk);
        double *bk = malloc(sizeof(double) * 4096);
        double *R = malloc(sizeof(double) * (size_t)nhmax * blk);
        double *Rwk = malloc(sizeof(double) * 2 * (size_t)nhmax * blk);
        double *S = malloc(sizeof(double) * (size_t)(Lmax + 1) * blk);
        double *X = malloc(sizeof(double) * 4 * blk), *Yv = X + blk, *Z = Yv + blk, *T = Z + blk;
        double *xyz = malloc(sizeof(double) * 3 * blk);
        double *Abuf = malloc(sizeof(double) * CX_ABUF);
        CXRec *rec = malloc(sizeof(CXRec) * (opt->npair + 1));
        double *Aoc = malloc(sizeof(double) * NQ * blk);
        double *Aoo = malloc(sizeof(double) * (size_t)nocc * nocc);
        double *Psi = malloc(sizeof(double) * (size_t)nocc * (3 * natm + 1));
        double *Out = malloc(sizeof(double) * (size_t)nao * (3 * natm + 1));
        double *Out2 = malloc(sizeof(double) * (size_t)nocc * (3 * natm + 1));
        double *tmp = malloc(sizeof(double) * (size_t)nao * nocc);
        double **farR = calloc(natm, sizeof(double *));
        int *stamp = malloc(sizeof(int) * natm);
        double *armin = malloc(sizeof(double) * natm);
        int at, i;
        for (at = 0; at < natm; at++) {
                stamp[at] = -1;
                if (opt->atm_lmax[at] >= 0) farR[at] = malloc(sizeof(double) * NH(opt->atm_lmax[at] + 1) * blk);
        }
        const int oom = !Vt || !VTt || !a || !full || !psid || !wetad || !U || !Abuf || !rec || !Aoc || !Out
                        || !Bd || !dA || !rs || !cs || !ys;
        if (oom) {
#pragma omp atomic write
                err = 1;
        }
        int b;
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < nblk; b++) {
                if (oom) continue;
                const int g0 = b * blk, n = g0 + blk <= ngrid ? blk : ngrid - g0;
                uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int k, g, ip, x, d;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select1(a, ix, runs, n, xyz, msk, cutoff, ao_loc, atm, bas, nbas, env, rmx, full);
                if (nl == 0) continue;
                const double *w = weights + g0;
                double xsum = 0;
                for (i = 0; i < nl; i++) {
                        double s = 0;
                        for (g = 0; g < n; g++) s += fabs(w[g] * a[(size_t)i * n + g]);
                        if (s > xsum) xsum = s;
                }
                /* psi = C[ix]^T phi, eta = Y[ix]^T phi (nocc, n); F = 2 C psi (nao, n) */
                for (i = 0; i < nl; i++) {
                        memcpy(cs + (size_t)i * nocc, co + (size_t)ix[i] * nocc, sizeof(double) * nocc);
                        memcpy(ys + (size_t)i * nocc, yo + (size_t)ix[i] * nocc, sizeof(double) * nocc);
                }
                dgemm_("N", "T", &n, &nocc, &nl, &one, a, &n, cs, &nocc, &zero, psi, &n);
                dgemm_("N", "T", &n, &nocc, &nl, &one, a, &n, ys, &nocc, &zero, weta, &n);
                for (i = 0; i < nocc; i++)
                        for (g = 0; g < n; g++) weta[(size_t)i * n + g] *= w[g];
                dgemm_("N", "N", &n, &nao, &nocc, &two, psi, &n, co, &nocc, &zero, F, &n);
                for (i = 0; i < nbas; i++) fmax[i] = 0;
                {
                        int mu;
                        for (mu = 0; mu < nao; mu++) {
                                double m = 0;
                                const double *f = F + (size_t)mu * n;
                                for (g = 0; g < n; g++) m = fabs(f[g]) > m ? fabs(f[g]) : m;
                                if (m > fmax[shl_of[mu]]) fmax[shl_of[mu]] = m;
                        }
                }
                /* the local atoms (ix sorted: each atom one range) and psi^{A,x}, w eta^{A,x} */
                int nla = 0;
                for (i = 0; i < nl; i++) {
                        const int aa = atm_of_ao[ix[i]];
                        if (nla == 0 || la[3 * (nla - 1)] != aa) {
                                la[3 * nla] = aa;
                                la[3 * nla + 1] = i;
                                nla++;
                        }
                        la[3 * (nla - 1) + 2] = i + 1;
                }
                const int nd = 3 * nla;
                for (k = 0; k < nla; k++) {
                        const int r0 = la[3 * k + 1], nr = la[3 * k + 2] - r0;
                        for (x = 0; x < 3; x++) {
                                double *pd = psid + (size_t)(3 * k + x) * nocc * n, *ed = wetad + (size_t)(3 * k + x) * nocc * n;
                                const double *ax = a + ((size_t)(1 + x) * nl + r0) * n;
                                dgemm_("N", "T", &n, &nocc, &nr, &mone, ax, &n, cs + (size_t)r0 * nocc, &nocc, &zero, pd, &n);
                                dgemm_("N", "T", &n, &nocc, &nr, &mone, ax, &n, ys + (size_t)r0 * nocc, &nocc, &zero, ed, &n);
                                for (i = 0; i < nocc; i++)
                                        for (g = 0; g < n; g++) ed[(size_t)i * n + g] *= w[g];
                        }
                }
                memset(touched, 0, natm);
                memset(Aoc, 0, sizeof(double) * NQ * n);
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
                int nrec = 0;
                size_t aoff = 0;
                for (ip = 0; ip < opt->npair; ip++) {
                        const CXPair *pp = opt->pairs + ip;
                        if (pp->bound == 0) continue;
                        const int I = pp->I, J = pp->J, both = I != J;
                        const double wgt = xsum * (fmax[I] + fmax[J]);
                        if (pp->bound * wgt < tol) continue;
                        const int L = pp->L, nh = NH(L), nh1 = NH(L + 1), nI = pp->nI, nJ = pp->nJ, nab = nI * nJ;
                        double bsum = 0;
                        for (k = 0; k < pp->np; k++) {
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                const double dx = P[0] - cb[0], dy = P[1] - cb[1], dz = P[2] - cb[2];
                                const double dmin = sqrt(dx * dx + dy * dy + dz * dz) - rb;
                                double f = 1;
                                if (dmin > 0) {
                                        const double T0 = opt->pp_p[pp->po + k] * dmin * dmin;
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
                                bk[k] = f * opt->pp_b[pp->po + k] * (1 + sqrt(2 * opt->pp_p[pp->po + k]));
                                bsum += bk[k];
                        }
                        if (bsum * wgt < tol) continue;
                        const size_t sz = (size_t)nab * nh, szd = (size_t)3 * nab * nh1;
                        const int I0 = ao_loc[I], J0 = ao_loc[J];
                        memset(A, 0, sizeof(double) * nab * n);
                        memset(dA, 0, sizeof(double) * 6 * nab * n);
                        double *dAI = dA, *dAC = dA + (size_t)3 * nab * n;
                        int k0 = 0;
                        if (pp->onec) {
                                at = pp->atom;
                                const double *Ra = env + atm[at * ATM_SLOTS + PTR_COORD];
                                const int La = opt->atm_lmax[at] + 1;
                                if (stamp[at] != b) {
                                        double rmin = 1e300;
                                        for (g = 0; g < n; g++) {
                                                X[g] = Ra[0] - xyz[g];
                                                Yv[g] = Ra[1] - xyz[n + g];
                                                Z[g] = Ra[2] - xyz[2 * n + g];
                                                T[g] = X[g] * X[g] + Yv[g] * Yv[g] + Z[g] * Z[g];
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
                                                hermite_R_vec(farR[at], S, X, Yv, Z, La, n, Rwk);
                                        } else {
                                                armin[at] = 0;
                                        }
                                }
                                const double rmin = armin[at];
                                if (rmin > 0) {
                                        const double *pv = opt->pp_p + pp->po;
                                        while (k0 < pp->np && pv[k0] * rmin >= tcut) k0++;
                                }
                                if (k0 > 0) {
                                        const double *Mo = opt->mdata + pp->mo + sz * (k0 - 1);
                                        e_times_r(A, Mo, farR[at], nab, nh, n);
                                        point_deriv(dAC, Mo, farR[at], nab, nh, n);
                                }
                        }
                        for (k = k0; k < pp->np; k++) {
                                if (bk[k] * wgt < tol) continue;
                                const double p = opt->pp_p[pp->po + k];
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                const double *E = opt->edata + pp->eo + sz * k;
#ifdef CX_FUSED
                                cxs_table[L](A, nab, E, p, P, xyz, n, R);
#else
                                double tmin = 1e300, tmax = 0;
                                for (g = 0; g < n; g++) {
                                        X[g] = P[0] - xyz[g];
                                        Yv[g] = P[1] - xyz[n + g];
                                        Z[g] = P[2] - xyz[2 * n + g];
                                        T[g] = p * (X[g] * X[g] + Yv[g] * Yv[g] + Z[g] * Z[g]);
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
                                hermite_R_vec(R, S, X, Yv, Z, L + 1, n, Rwk);
                                e_times_r(A, E, R, nab, nh, n);
#endif
                                point_deriv(dAC, E, R, nab, nh, n);
                                if (!pp->onec) {
                                        const double *Ed = opt->eddata + opt->edo[ip] + szd * k;
                                        for (x = 0; x < 3; x++)
                                                e_times_r(dAI + (size_t)x * nab * n, Ed + (size_t)x * nab * nh1, R, nab, nh1, n);
                                }
                        }
                        /* the integrals, point-major, for Aoc */
                        if (aoff + (size_t)nab * n > CX_ABUF) {
                                multi_contract_const(Aoc, co, Abuf, rec, nrec, nao, n, nocc);
                                nrec = 0;
                                aoff = 0;
                        }
                        {
                                double *atp = Abuf + aoff;
                                int ab;
                                for (ab = 0; ab < nab; ab++) {
                                        const double *ar = A + (size_t)ab * n;
                                        for (g = 0; g < n; g++) atp[(size_t)g * nab + ab] = ar[g];
                                }
                                rec[nrec].I0 = I0; rec[nrec].J0 = J0; rec[nrec].nI = nI; rec[nrec].nJ = nJ;
                                rec[nrec].cI = 1; rec[nrec].cJ = both; rec[nrec].off = aoff;
                                nrec++;
                                aoff += (size_t)nab * n;
                        }
                        /* (f3), (t3): r, s of the centre / point derivatives, per atom */
                        const int aI = bas[I * BAS_SLOTS + ATOM_OF], aJ = bas[J * BAS_SLOTS + ATOM_OF];
                        for (x = 0; x < 3; x++) {
                                double *rI = rs, *sI = rI + (size_t)nI_max * n, *rC = sI + (size_t)nI_max * n;
                                double *sC = rC + (size_t)nI_max * n;
                                pair_rs(rC, sC, dAC + (size_t)x * nab * n, F, I0, J0, nI, nJ, n, both);
                                int nat = 0, ats[2];
                                double *rr[2], *ss[2];
                                if (pp->onec) {
                                        /* the atom's derivative: minus the point derivative */
                                        int q;
                                        for (q = 0; q < nI * n; q++) rC[q] = -rC[q];
                                        if (both) for (q = 0; q < nJ * n; q++) sC[q] = -sC[q];
                                        ats[0] = aI; rr[0] = rC; ss[0] = sC; nat = 1;
                                } else {
                                        pair_rs(rI, sI, dAI + (size_t)x * nab * n, F, I0, J0, nI, nJ, n, 1);
                                        int q;
                                        /* J: -(I + C) */
                                        for (q = 0; q < nI * n; q++) rC[q] = -(rI[q] + rC[q]);
                                        for (q = 0; q < nJ * n; q++) sC[q] = -(sI[q] + sC[q]);
                                        ats[0] = aI; rr[0] = rI; ss[0] = sI;
                                        ats[1] = aJ; rr[1] = rC; ss[1] = sC;
                                        nat = 2;
                                }
                                int q;
                                for (q = 0; q < nat; q++) {
                                        const int ya = ats[q];
                                        double *Uy = U + ((size_t)ya * 3 + x) * nocc * n;
                                        if (!touched[ya]) {
                                                memset(U + (size_t)ya * 3 * nocc * n, 0, sizeof(double) * 3 * nocc * n);
                                                touched[ya] = 1;
                                        }
                                        /* U^{y}(n, nocc cm) += r^T C_I + s^T C_J */
                                        dgemm_("N", "T", &n, &nocc, &nI, &one, rr[q], &n, co + (size_t)I0 * nocc, &nocc, &one, Uy, &n);
                                        if (both)
                                                dgemm_("N", "T", &n, &nocc, &nJ, &one, ss[q], &n, co + (size_t)J0 * nocc, &nocc, &one, Uy, &n);
                                        /* VT rows a of atom ya: += r (w eta)^T */
                                        double *vt = VTt + (((size_t)ya * 3 + x) * nao) * nocc;
                                        dgemm_("T", "N", &nocc, &nI, &n, &one, weta, &n, rr[q], &n, &one, vt + (size_t)I0 * nocc, &nocc);
                                        if (both)
                                                dgemm_("T", "N", &nocc, &nJ, &n, &one, weta, &n, ss[q], &n, &one, vt + (size_t)J0 * nocc, &nocc);
                                }
                        }
                }
                if (nrec) multi_contract_const(Aoc, co, Abuf, rec, nrec, nao, n, nocc);
                /* per point: G^F/2 = Aoc psi, Bd/2 = Aoc psi^{d}; U/2 = Aoo psi, U^{d}/2 = Aoo psi^{d} */
                for (k = 0; k < nla; k++) {
                        const int aa = la[3 * k];
                        if (!touched[aa]) {
                                memset(U + (size_t)aa * 3 * nocc * n, 0, sizeof(double) * 3 * nocc * n);
                                touched[aa] = 1;
                        }
                }
                const int ncol = 1 + nd;
                for (g = 0; g < n; g++) {
                        const double *Ag = Aoc + g * NQ;            /* (nao, nocc) row-major */
                        for (i = 0; i < nocc; i++) {
                                Psi[i * ncol] = psi[(size_t)i * n + g];
                                for (d = 0; d < nd; d++) Psi[i * ncol + 1 + d] = psid[((size_t)d * nocc + i) * n + g];
                        }
                        /* Out (nao, ncol) = 2 Ag Psi */
                        dgemm_("N", "N", &ncol, &nao, &nocc, &two, Psi, &ncol, Ag, &nocc, &zero, Out, &ncol);
                        /* Aoo (nocc, nocc) = C^T Ag */
                        dgemm_("N", "T", &nocc, &nocc, &nao, &one, Ag, &nocc, co, &nocc, &zero, Aoo, &nocc);
                        /* Out2 (nocc, ncol) = 2 Aoo Psi */
                        dgemm_("N", "N", &ncol, &nocc, &nocc, &two, Psi, &ncol, Aoo, &nocc, &zero, Out2, &ncol);
                        int nu;
                        for (nu = 0; nu < nao; nu++) GFb[(size_t)nu * n + g] = Out[(size_t)nu * ncol];
                        for (i = 0; i < nocc; i++) Ub[(size_t)i * n + g] = Out2[(size_t)i * ncol];
                        for (k = 0; k < nla; k++)
                                for (x = 0; x < 3; x++) {
                                        double *Uy = U + ((size_t)la[3 * k] * 3 + x) * nocc * n;
                                        for (i = 0; i < nocc; i++) Uy[(size_t)i * n + g] += Out2[(size_t)i * ncol + 1 + 3 * k + x];
                                }
                        /* Bd stored transiently: (t2) needs sum_g Bd (w eta)^T; accumulate per point */
                        for (d = 0; d < nd; d++) {
                                const int ya = la[3 * (d / 3)], y = d % 3;
                                double *vt = VTt + (((size_t)ya * 3 + y) * nao) * nocc;
                                for (nu = 0; nu < nao; nu++) {
                                        const double bdv = Out[(size_t)nu * ncol + 1 + d];
                                        if (bdv == 0) continue;
                                        double *row = vt + (size_t)nu * nocc;
                                        for (i = 0; i < nocc; i++) row[i] += bdv * weta[(size_t)i * n + g];
                                }
                        }
                }
                /* (t1): VT[A, x] += G^F (w eta^{A,x})^T */
                for (d = 0; d < nd; d++) {
                        const int ya = la[3 * (d / 3)], y = d % 3;
                        double *vt = VTt + (((size_t)ya * 3 + y) * nao) * nocc;
                        dgemm_("T", "N", &nocc, &nao, &n, &one, wetad + (size_t)d * nocc * n, &n, GFb, &n, &one, vt, &nocc);
                }
                /* (f1): V[A, x][mu in A] -= sum_g w d_x phi_mu U */
                for (i = 0; i < nocc; i++)
                        for (g = 0; g < n; g++) Ub[(size_t)i * n + g] *= w[g];
                for (k = 0; k < nla; k++) {
                        const int r0 = la[3 * k + 1], nr = la[3 * k + 2] - r0, aa = la[3 * k];
                        for (x = 0; x < 3; x++) {
                                dgemm_("T", "N", &nocc, &nr, &n, &one, Ub, &n, a + ((size_t)(1 + x) * nl + r0) * n, &n, &zero, tmp, &nocc);
                                double *v = Vt + (((size_t)aa * 3 + x) * nao) * nocc;
                                int r_;
                                for (r_ = 0; r_ < nr; r_++) {
                                        double *row = v + (size_t)ix[r0 + r_] * nocc;
                                        for (i = 0; i < nocc; i++) row[i] -= tmp[(size_t)r_ * nocc + i];
                                }
                        }
                }
                /* V[Y, y][mu in ix] += sum_g w phi_mu U^{Y,y} */
                {
                        double *wphi = full;            /* (nl, n) */
                        for (i = 0; i < nl; i++)
                                for (g = 0; g < n; g++) wphi[(size_t)i * n + g] = w[g] * a[(size_t)i * n + g];
                        for (at = 0; at < natm; at++) {
                                if (!touched[at]) continue;
                                for (x = 0; x < 3; x++) {
                                        const double *Uy = U + ((size_t)at * 3 + x) * nocc * n;
                                        dgemm_("T", "N", &nocc, &nl, &n, &one, Uy, &n, wphi, &n, &zero, tmp, &nocc);
                                        double *v = Vt + (((size_t)at * 3 + x) * nao) * nocc;
                                        for (i = 0; i < nl; i++) {
                                                double *row = v + (size_t)ix[i] * nocc;
                                                const double *src = tmp + (size_t)i * nocc;
                                                int j;
                                                for (j = 0; j < nocc; j++) row[j] += src[j];
                                        }
                                }
                        }
                }
        }
#pragma omp barrier
        {
                const size_t chunk = (nvo + nthreads - 1) / nthreads;
                const size_t s0 = chunk * tid < nvo ? chunk * tid : nvo, s1 = s0 + chunk < nvo ? s0 + chunk : nvo;
                size_t q;
                int t;
                for (t = 0; t < nthreads; t++) {
                        if (!vb[t] || !vtb[t]) continue;
                        const double *p1 = vb[t], *p2 = vtb[t];
                        for (q = s0; q < s1; q++) {
                                V[q] += p1[q];
                                VT[q] += p2[q];
                        }
                }
        }
#pragma omp barrier
        free(Vt); free(VTt); free(a); free(full); free(rmx); free(ix); free(runs); free(cs); free(ys); free(F);
        free(GFb); free(psi); free(weta); free(Ub); free(psid); free(wetad); free(Bd); free(la); free(U); free(touched);
        free(fmax); free(A); free(dA); free(rs); free(bk); free(R); free(Rwk); free(S); free(X); free(xyz);
        free(Abuf); free(rec); free(Aoc); free(Aoo); free(Psi); free(Out); free(Out2); free(tmp);
        for (at = 0; at < natm; at++) free(farR[at]);
        free(farR); free(stamp); free(armin);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
        free(vb); free(vtb); free(shl_of); free(atm_of_ao);
        return err;
}
