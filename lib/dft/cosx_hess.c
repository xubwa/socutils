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

/* ---------------------------------------------------------------- K second derivatives */

/* the second derivative of the spherical E of primitive pair (ip, jp) with respect to the centre of
 * I: out[c][ab][t] (6 = xx xy xz yy yz zz, nI nJ, NH(L + 2)); d^2 / dA_x^2 of x^i e^{-a x^2}:
 * 4 a^2 (i+2) - 2 a (2i + 1) (i) + i (i - 1) (i-2); mixed: the first derivatives of two directions */
static void prim_Edd(double *out, int I, int J, int ip, int jp, int *atm, int *bas, double *env, double *wk)
{
        const int *bi = bas + I * BAS_SLOTS, *bj = bas + J * BAS_SLOTS;
        const int li = bi[ANG_OF], lj = bj[ANG_OF], L = li + lj, nh2 = NH(L + 2);
        const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF], nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
        const int di = 2 * li + 1, dj = 2 * lj + 1, nJ = ncj * dj, nI = nci * di;
        const int nfi = (li + 1) * (li + 2) / 2, nfj = (lj + 1) * (lj + 2) / 2;
        const double ai = env[bi[PTR_EXP] + ip], aj = env[bj[PTR_EXP] + jp];
        const double *Ra = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double *Rb = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double p = ai + aj;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        const double e0 = 2 * M_PI / p * exp(-ai * aj / p * r2) * common_fac_sp(li) * common_fac_sp(lj);
        const int Li = li + 2, s2 = Li + lj + 1, s1 = (lj + 1) * s2, sz = (Li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        double *Ec = Ez + sz;                                   /* (6, nfi, nfj, nh2) */
        double *Eh = Ec + (size_t)6 * nfi * nfj * nh2;          /* (6, di, nfj, nh2) */
        double *Es = Eh + (size_t)6 * di * nfj * nh2;           /* (6, di, dj, nh2) */
        static const int cxy[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        hermite_E(Ex, Li, lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, Li, lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, Li, lj, ai, aj, Ra[2], Rb[2], e0);
        const double *Es3[3] = {Ex, Ey, Ez};
        int ix, iy, jx, jy, n, fa = 0, c;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                const int iz = li - ix - iy, ia[3] = {ix, iy, iz};
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        const int jz = lj - jx - jy, ja[3] = {jx, jy, jz};
                        for (c = 0; c < 6; c++) {
                                const int x = cxy[c][0], y = cxy[c][1];
                                double *e = Ec + (((size_t)c * nfi + fa) * nfj + fb) * nh2;
                                for (n = 0; n < nh2; n++) {
                                        const int tt[3] = {h_t[n], h_u[n], h_v[n]};
                                        double val = 1;
                                        int k;
                                        for (k = 0; k < 3; k++) {
                                                const double *E = Es3[k];
                                                const int i = ia[k], j = ja[k], t = tt[k];
#define EE(ii) ((ii) >= 0 ? E[(ii) * s1 + j * s2 + t] : 0.)
                                                double f;
                                                if (k == x && k == y)
                                                        f = 4 * ai * ai * EE(i + 2) - 2 * ai * (2 * i + 1) * EE(i)
                                                            + i * (i - 1) * EE(i - 2);
                                                else if (k == x || k == y)
                                                        f = 2 * ai * EE(i + 1) - i * EE(i - 1);
                                                else
                                                        f = EE(i);
#undef EE
                                                val *= f;
                                        }
                                        e[n] = val;
                                }
                        }
                }
        }
        int m, f, k, nc;
        for (c = 0; c < 6; c++) {
                const double *Ecx = Ec + (size_t)c * nfi * nfj * nh2;
                double *Ehx = Eh + (size_t)c * di * nfj * nh2, *Esx = Es + (size_t)c * di * dj * nh2;
                if (li <= 1) {
                        memcpy(Ehx, Ecx, sizeof(double) * nfi * nfj * nh2);
                } else {
                        for (m = 0; m < di; m++) {
                                const double *cc = c2s_row(li, m, &nc);
                                double *o = Ehx + (size_t)m * nfj * nh2;
                                memset(o, 0, sizeof(double) * nfj * nh2);
                                for (f = 0; f < nfi; f++) {
                                        if (cc[f] == 0) continue;
                                        const double *s_ = Ecx + (size_t)f * nfj * nh2;
                                        for (k = 0; k < nfj * nh2; k++) o[k] += cc[f] * s_[k];
                                }
                        }
                }
                if (lj <= 1) {
                        memcpy(Esx, Ehx, sizeof(double) * di * nfj * nh2);
                } else {
                        for (fa = 0; fa < di; fa++)
                                for (m = 0; m < dj; m++) {
                                        const double *cc = c2s_row(lj, m, &nc);
                                        double *o = Esx + ((size_t)fa * dj + m) * nh2;
                                        memset(o, 0, sizeof(double) * nh2);
                                        for (f = 0; f < nfj; f++) {
                                                if (cc[f] == 0) continue;
                                                const double *s_ = Ehx + ((size_t)fa * nfj + f) * nh2;
                                                for (k = 0; k < nh2; k++) o[k] += cc[f] * s_[k];
                                        }
                                }
                }
        }
        const double *cfi = env + bi[PTR_COEFF], *cfj = env + bj[PTR_COEFF];
        int ci, cj, mi, mj;
        for (c = 0; c < 6; c++)
        for (ci = 0; ci < nci; ci++)
        for (mi = 0; mi < di; mi++)
        for (cj = 0; cj < ncj; cj++)
        for (mj = 0; mj < dj; mj++) {
                const double cc = cfi[ci * npi + ip] * cfj[cj * npj + jp];
                const double *s_ = Es + (((size_t)c * di + mi) * dj + mj) * nh2;
                double *o = out + (((size_t)c * nI + ci * di + mi) * nJ + cj * dj + mj) * nh2;
                for (k = 0; k < nh2; k++) o[k] = cc * s_[k];
        }
}

static size_t prim_wk_size_dd(void)
{
        const int sE = (CX_LMAX + 3) * (CX_LMAX + 1) * (2 * CX_LMAX + 3);
        const int nc = (CX_LMAX + 1) * (CX_LMAX + 2) / 2, nh = NH(2 * CX_LMAX + 2);
        return 3 * sE + 18 * (size_t)nc * nc * nh;
}

int SOCDFTcosx_build_hess(void *p, int *atm, int *bas, double *env)
{
        CXOpt *opt = p;
        if (!SOCDFTcosx_build_grad(p, atm, bas, env)) return 0;
        if (opt->eedata) return 1;
        size_t eo = 0;
        int ip_;
        opt->eeo = malloc(sizeof(size_t) * (opt->npair + 1));
        for (ip_ = 0; ip_ < opt->npair; ip_++) {
                const CXPair *pp = opt->pairs + ip_;
                opt->eeo[ip_] = eo;
                if (!pp->onec && pp->bound > 0) eo += (size_t)6 * pp->nI * pp->nJ * NH(pp->L + 2) * pp->np;
        }
        opt->eedata = malloc(sizeof(double) * (eo + 1));
        if (!opt->eedata) return 0;
#pragma omp parallel
{
        double *wk = malloc(sizeof(double) * prim_wk_size_dd());
#pragma omp for schedule(dynamic, 8)
        for (ip_ = 0; ip_ < opt->npair; ip_++) {
                const CXPair *pp = opt->pairs + ip_;
                if (pp->onec || pp->bound == 0) continue;
                const size_t sz = (size_t)6 * pp->nI * pp->nJ * NH(pp->L + 2);
                const int npj = bas[pp->J * BAS_SLOTS + NPRIM_OF];
                int k;
                for (k = 0; k < pp->np; k++) {
                        const int id = opt->pp_id[pp->po + k];
                        prim_Edd(opt->eedata + opt->eeo[ip_] + sz * k, pp->I, pp->J, id / npj, id % npj, atm, bas, env, wk);
                }
        }
        free(wk);
}
        return 1;
}

/* AO values and derivatives (ncomp = 1, 4, 10 for deriv 0, 1, 2) of the significant AOs of a block */
static int eval_selectd(int deriv, double *a, int *ix, int *runs, int n, double *xyz, uint8_t *msk, double cutoff,
                        int *ao_loc, int *atm, int *bas, int nbas, double *env, double *rmx, double *full)
{
        const int ncomp = deriv == 2 ? 10 : (deriv == 1 ? 4 : 1);
        const int nrow = (n + MASK_BLK - 1) / MASK_BLK;
        int sh = 0, nrun = 0, r, i, c;
        size_t ntot = 0;
        while (sh < nbas) {
                int on = 0;
                for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                if (!on) { sh++; continue; }
                const int s0 = sh;
                for (sh++; sh < nbas; sh++) {
                        on = 0;
                        for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                        if (!on) break;
                }
                runs[2 * nrun] = s0;
                runs[2 * nrun + 1] = sh;
                nrun++;
                ntot += ao_loc[sh] - ao_loc[s0];
        }
        if (ntot == 0) return 0;
        size_t row = 0;
        memset(rmx, 0, sizeof(double) * ntot);
        for (i = 0; i < nrun; i++) {
                SOCDFTgtoval_ld(deriv, n, runs[2 * i], runs[2 * i + 1], ao_loc, full, ntot, row, xyz, msk, atm, bas,
                                nbas, env, rmx);
                row += ao_loc[runs[2 * i + 1]] - ao_loc[runs[2 * i]];
        }
        int nl = 0;
        row = 0;
        int *src = runs + 2 * nbas;
        for (i = 0; i < nrun; i++) {
                int mu;
                for (mu = ao_loc[runs[2 * i]]; mu < ao_loc[runs[2 * i + 1]]; mu++, row++) {
                        if (rmx[row] > cutoff) {
                                src[nl] = (int)row;
                                ix[nl++] = mu;
                        }
                }
        }
        for (c = 0; c < ncomp; c++)
                for (i = 0; i < nl; i++)
                        memcpy(a + ((size_t)c * nl + i) * n, full + ((size_t)c * ntot + src[i]) * n, sizeof(double) * n);
        return nl;
}

/*
 * hs (3 natm, 3 natm) = d^2 / dR dR' of E = sum_g w H^T A F at fixed M = 2 Y C^T, D = 2 C C^T
 * (H = M^T phi = 2 C eta, eta = Y^T phi; F = 2 C psi), on the fixed grid; with V, VT != NULL also
 * the Fock1 of SOCDFTcosx_fock1 (same pass).  In the occupied form (Aoc = A C, Aoo = C^T A C per
 * point, U = 2 Aoo psi, U' = 2 Aoo eta, U3 / U3' the per-pair C^T (d_Y A) F / H):
 *   (1ab) 2 sum_g w (eta^{R xy} U + psi^{R xy} U')        the AO second derivatives, atom blocks
 *   (1c)  4 sum_g w eta^{d1} Aoo psi^{d2} + transpose      AO first derivatives on both sides
 *   (2)   2 sum_g w (eta^{d} U3^{Y} + psi^{d} U3'^{Y}) + transpose   AO x integral derivatives
 *   (3)   sum_pairs <d^2 A, W>, W_ab = w (H_a F_b + H_b F_a): centre-centre (the d^2 E tables), centre-
 *         point, point-point; the second centre by translational invariance.
 */
int SOCDFTcosx_hess(double *hs, double *V, double *VT, double *co, double *yo, int nocc, void *optp,
                    double *coords, double *weights, uint8_t *non0tab, int ngrid, int blk, double cutoff, double tol,
                    double tcut, int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                    dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        CXOpt *opt = optp;
        cx_tables();
        const int do_f1 = V != NULL;
        const int nao = ao_loc[nbas];
        const int nblk = (ngrid + blk - 1) / blk;
        const int Lmax = opt->lmax + 2, nhmax = NH(Lmax);
        const size_t nvo = do_f1 ? (size_t)natm * 3 * nao * nocc : 0;
        const int n3 = 3 * natm;
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
        double **hb = calloc(nthreads, sizeof(double *));
        int err = 0;
        if (do_f1) {
                memset(V, 0, sizeof(double) * nvo);
                memset(VT, 0, sizeof(double) * nvo);
        }
        memset(hs, 0, sizeof(double) * n3 * n3);
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        const double one = 1, zero = 0, two = 2, mone = -1;
        const size_t NQ = (size_t)nao * nocc;
        double *Vt = vb[tid] = do_f1 ? calloc(nvo, sizeof(double)) : NULL;
        double *VTt = vtb[tid] = do_f1 ? calloc(nvo, sizeof(double)) : NULL;
        double *Ht = hb[tid] = calloc((size_t)n3 * n3, sizeof(double));
        double *a = malloc(sizeof(double) * 10 * (size_t)nao * blk);
        double *full = malloc(sizeof(double) * 10 * (size_t)nao * blk);
        double *rmx = malloc(sizeof(double) * nao);
        int *ix = malloc(sizeof(int) * nao);
        int *runs = malloc(sizeof(int) * (2 * (nbas + 1) + nao));
        double *cs = malloc(sizeof(double) * NQ), *ys = malloc(sizeof(double) * NQ);
        double *F = malloc(sizeof(double) * (size_t)nao * blk), *H = malloc(sizeof(double) * (size_t)nao * blk);
        double *GFb = malloc(sizeof(double) * (size_t)nao * blk);
        double *psi = malloc(sizeof(double) * (size_t)nocc * blk), *eta = malloc(sizeof(double) * (size_t)nocc * blk);
        double *weta = malloc(sizeof(double) * (size_t)nocc * blk);
        double *Ub = malloc(sizeof(double) * (size_t)nocc * blk), *Upb = malloc(sizeof(double) * (size_t)nocc * blk);
        double *psid = malloc(sizeof(double) * (size_t)3 * natm * nocc * blk);
        double *wpsid = malloc(sizeof(double) * (size_t)3 * natm * nocc * blk);
        double *wetad = malloc(sizeof(double) * (size_t)3 * natm * nocc * blk);
        int *la = malloc(sizeof(int) * (natm + 1) * 3);
        double *U = malloc(sizeof(double) * (size_t)natm * 3 * nocc * blk);
        double *Up = malloc(sizeof(double) * (size_t)natm * 3 * nocc * blk);
        char *touched = malloc(natm);
        double *fmax = malloc(sizeof(double) * nbas);
        double *A = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *W = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *dA = malloc(sizeof(double) * (size_t)6 * nab_max * blk);
        double *rs = malloc(sizeof(double) * (size_t)8 * nI_max * blk);
        double *Mm = malloc(sizeof(double) * (size_t)nab_max * nhmax);
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
        double *Psi = malloc(sizeof(double) * (size_t)nocc * (3 * natm + 2));
        double *Out = malloc(sizeof(double) * (size_t)nao * (3 * natm + 2));
        double *Out2 = malloc(sizeof(double) * (size_t)nocc * (3 * natm + 2));
        double *tmp = malloc(sizeof(double) * (size_t)nao * (nocc > 3 * natm ? nocc : 3 * natm));
        double *Qm = malloc(sizeof(double) * (size_t)nao * blk);
        double **farR = calloc(natm, sizeof(double *));
        int *stamp = malloc(sizeof(int) * natm);
        double *armin = malloc(sizeof(double) * natm);
        int at, i;
        for (at = 0; at < natm; at++) {
                stamp[at] = -1;
                if (opt->atm_lmax[at] >= 0) farR[at] = malloc(sizeof(double) * NH(opt->atm_lmax[at] + 2) * blk);
        }
        const int oom = (do_f1 && (!Vt || !VTt)) || !Ht || !a || !full || !psid || !wpsid || !wetad || !U || !Up
                        || !Abuf || !rec || !Aoc || !Out || !dA || !rs || !cs || !ys || !W || !Mm;
        if (oom) {
#pragma omp atomic write
                err = 1;
        }
        static const int cxy[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
#define HS(i_, j_) Ht[(size_t)(i_) * n3 + (j_)]
        int b;
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < nblk; b++) {
                if (oom) continue;
                const int g0 = b * blk, n = g0 + blk <= ngrid ? blk : ngrid - g0;
                uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int k, g, ip, x, y, d, c;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_selectd(2, a, ix, runs, n, xyz, msk, cutoff, ao_loc, atm, bas, nbas, env, rmx, full);
                if (nl == 0) continue;
                const double *w = weights + g0;
                double xsum = 0;
                for (i = 0; i < nl; i++) {
                        double s = 0;
                        for (g = 0; g < n; g++) s += fabs(w[g] * a[(size_t)i * n + g]);
                        if (s > xsum) xsum = s;
                }
                for (i = 0; i < nl; i++) {
                        memcpy(cs + (size_t)i * nocc, co + (size_t)ix[i] * nocc, sizeof(double) * nocc);
                        memcpy(ys + (size_t)i * nocc, yo + (size_t)ix[i] * nocc, sizeof(double) * nocc);
                }
                dgemm_("N", "T", &n, &nocc, &nl, &one, a, &n, cs, &nocc, &zero, psi, &n);
                dgemm_("N", "T", &n, &nocc, &nl, &one, a, &n, ys, &nocc, &zero, eta, &n);
                for (i = 0; i < nocc; i++)
                        for (g = 0; g < n; g++) weta[(size_t)i * n + g] = w[g] * eta[(size_t)i * n + g];
                dgemm_("N", "N", &n, &nao, &nocc, &two, psi, &n, co, &nocc, &zero, F, &n);
                dgemm_("N", "N", &n, &nao, &nocc, &two, eta, &n, co, &nocc, &zero, H, &n);
                for (i = 0; i < nbas; i++) fmax[i] = 0;
                {
                        int mu;
                        for (mu = 0; mu < nao; mu++) {
                                double m = 0;
                                const double *f = F + (size_t)mu * n, *h = H + (size_t)mu * n;
                                for (g = 0; g < n; g++) {
                                        const double v = fabs(f[g]) + fabs(h[g]);
                                        m = v > m ? v : m;
                                }
                                if (m > fmax[shl_of[mu]]) fmax[shl_of[mu]] = m;
                        }
                }
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
                                double *wpd = wpsid + (size_t)(3 * k + x) * nocc * n;
                                const double *ax = a + ((size_t)(1 + x) * nl + r0) * n;
                                dgemm_("N", "T", &n, &nocc, &nr, &mone, ax, &n, cs + (size_t)r0 * nocc, &nocc, &zero, pd, &n);
                                dgemm_("N", "T", &n, &nocc, &nr, &mone, ax, &n, ys + (size_t)r0 * nocc, &nocc, &zero, ed, &n);
                                for (i = 0; i < nocc; i++)
                                        for (g = 0; g < n; g++) {
                                                ed[(size_t)i * n + g] *= w[g];
                                                wpd[(size_t)i * n + g] = w[g] * pd[(size_t)i * n + g];
                                        }
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
                        const int L = pp->L, nh = NH(L), nh1 = NH(L + 1), nh2 = NH(L + 2);
                        const int nI = pp->nI, nJ = pp->nJ, nab = nI * nJ;
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
                                                for (dd = 0; dd <= L + 2; dd++) {
                                                        const double v = cx_damp[dd] * rp;
                                                        if (v > mx) mx = v;
                                                        rp *= ir;
                                                }
                                                f = mx < 1 ? mx : 1;
                                        }
                                }
                                const double sp = sqrt(2 * opt->pp_p[pp->po + k]);
                                bk[k] = f * opt->pp_b[pp->po + k] * (1 + sp + 2 * sp * sp);
                                bsum += bk[k];
                        }
                        if (bsum * wgt < tol) continue;
                        const size_t sz = (size_t)nab * nh, szd = (size_t)3 * nab * nh1, szdd = (size_t)6 * nab * nh2;
                        const int I0 = ao_loc[I], J0 = ao_loc[J];
                        /* W_ab = w (H_a F_b + H_b F_a) */
                        {
                                int ia, jb;
                                for (ia = 0; ia < nI; ia++)
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *wr = W + (size_t)(ia * nJ + jb) * n;
                                                const double *ha = H + (size_t)(I0 + ia) * n, *fb = F + (size_t)(J0 + jb) * n;
                                                const double *hb = H + (size_t)(J0 + jb) * n, *fa = F + (size_t)(I0 + ia) * n;
                                                if (both) {
                                                        for (g = 0; g < n; g++) wr[g] = w[g] * (ha[g] * fb[g] + hb[g] * fa[g]);
                                                } else {
                                                        for (g = 0; g < n; g++) wr[g] = w[g] * ha[g] * fb[g];
                                                }
                                        }
                        }
                        double SAA[3][3] = {{0}}, SAC[3][3] = {{0}}, SCC[3][3] = {{0}};
                        memset(A, 0, sizeof(double) * nab * n);
                        memset(dA, 0, sizeof(double) * 6 * nab * n);
                        double *dAI = dA, *dAC = dA + (size_t)3 * nab * n;
                        int k0 = 0;
                        if (pp->onec) {
                                at = pp->atom;
                                const double *Ra = env + atm[at * ATM_SLOTS + PTR_COORD];
                                const int La = opt->atm_lmax[at] + 2;
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
                                        w_dot_r(Mm, W, farR[at], nab, nh2, n, dgemm_);
                                        int ab, t;
                                        for (ab = 0; ab < nab; ab++)
                                                for (t = 0; t < nh; t++) {
                                                        const double e = Mo[(size_t)ab * nh + t];
                                                        if (e == 0) continue;
                                                        for (c = 0; c < 6; c++) {
                                                                const int cx_ = cxy[c][0], cy_ = cxy[c][1];
                                                                const int t2 = h_idx[h_t[t] + (cx_ == 0) + (cy_ == 0)][h_u[t] + (cx_ == 1) + (cy_ == 1)]
                                                                                    [h_v[t] + (cx_ == 2) + (cy_ == 2)];
                                                                SCC[cx_][cy_] += e * Mm[(size_t)ab * nh2 + t2];
                                                        }
                                                }
                                }
                        }
                        for (k = k0; k < pp->np; k++) {
                                if (bk[k] * wgt < tol) continue;
                                const double p = opt->pp_p[pp->po + k];
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                const double *E = opt->edata + pp->eo + sz * k;
                                double tmin = 1e300, tmax = 0;
                                for (g = 0; g < n; g++) {
                                        X[g] = P[0] - xyz[g];
                                        Yv[g] = P[1] - xyz[n + g];
                                        Z[g] = P[2] - xyz[2 * n + g];
                                        T[g] = p * (X[g] * X[g] + Yv[g] * Yv[g] + Z[g] * Z[g]);
                                        tmin = T[g] < tmin ? T[g] : tmin;
                                        tmax = T[g] > tmax ? T[g] : tmax;
                                }
                                boys_vec(S, n, T, L + 2, n, tmin, tmax);
                                double sc = 1;
                                int m;
                                for (m = 1; m <= L + 2; m++) {
                                        sc *= -2 * p;
                                        double *sm = S + (size_t)m * n;
                                        for (g = 0; g < n; g++) sm[g] *= sc;
                                }
                                hermite_R_vec(R, S, X, Yv, Z, L + 2, n, Rwk);
                                e_times_r(A, E, R, nab, nh, n);
                                point_deriv(dAC, E, R, nab, nh, n);
                                w_dot_r(Mm, W, R, nab, nh2, n, dgemm_);
                                int ab, t;
                                /* point-point: sum_t E_t Mm[t + ex + ey] */
                                for (ab = 0; ab < nab; ab++)
                                        for (t = 0; t < nh; t++) {
                                                const double e = E[(size_t)ab * nh + t];
                                                if (e == 0) continue;
                                                for (c = 0; c < 6; c++) {
                                                        const int cx_ = cxy[c][0], cy_ = cxy[c][1];
                                                        const int t2 = h_idx[h_t[t] + (cx_ == 0) + (cy_ == 0)][h_u[t] + (cx_ == 1) + (cy_ == 1)]
                                                                            [h_v[t] + (cx_ == 2) + (cy_ == 2)];
                                                        SCC[cx_][cy_] += e * Mm[(size_t)ab * nh2 + t2];
                                                }
                                        }
                                if (!pp->onec) {
                                        const double *Ed = opt->eddata + opt->edo[ip] + szd * k;
                                        const double *Edd = opt->eedata + opt->eeo[ip] + szdd * k;
                                        for (x = 0; x < 3; x++)
                                                e_times_r(dAI + (size_t)x * nab * n, Ed + (size_t)x * nab * nh1, R, nab, nh1, n);
                                        /* centre-point: -sum_t Ed^x_t Mm[t + ey]; centre-centre: Edd . Mm */
                                        for (x = 0; x < 3; x++) {
                                                const double *ed = Ed + (size_t)x * nab * nh1;
                                                for (ab = 0; ab < nab; ab++)
                                                        for (t = 0; t < nh1; t++) {
                                                                const double e = ed[(size_t)ab * nh1 + t];
                                                                if (e == 0) continue;
                                                                for (y = 0; y < 3; y++) {
                                                                        const int t1 = h_idx[h_t[t] + (y == 0)][h_u[t] + (y == 1)][h_v[t] + (y == 2)];
                                                                        SAC[x][y] -= e * Mm[(size_t)ab * nh2 + t1];
                                                                }
                                                        }
                                        }
                                        for (c = 0; c < 6; c++) {
                                                const double *edd = Edd + (size_t)c * nab * nh2;
                                                double s = 0;
                                                size_t q;
                                                for (q = 0; q < (size_t)nab * nh2; q++) s += edd[q] * Mm[q];
                                                SAA[cxy[c][0]][cxy[c][1]] += s;
                                        }
                                }
                        }
                        /* (3) */
                        const int aI = bas[I * BAS_SLOTS + ATOM_OF], aJ = bas[J * BAS_SLOTS + ATOM_OF];
                        for (x = 0; x < 3; x++)
                                for (y = x + 1; y < 3; y++) {
                                        SAA[y][x] = SAA[x][y];
                                        SCC[y][x] = SCC[x][y];
                                }
                        for (x = 0; x < 3; x++)
                                for (y = 0; y < 3; y++) {
                                        if (pp->onec) {
                                                HS(3 * aI + x, 3 * aI + y) += SCC[x][y];
                                        } else {
                                                const double ab_ = -(SAA[x][y] + SAC[x][y]);
                                                HS(3 * aI + x, 3 * aI + y) += SAA[x][y];
                                                HS(3 * aI + x, 3 * aJ + y) += ab_;
                                                HS(3 * aJ + y, 3 * aI + x) += ab_;
                                                HS(3 * aJ + x, 3 * aJ + y) += SAA[x][y] + SAC[x][y] + SAC[y][x] + SCC[x][y];
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
                        /* U3 (F) and U3' (H) of the centre / point derivatives, per atom; VT (t3) */
                        for (x = 0; x < 3; x++) {
                                double *rI = rs, *sI = rI + (size_t)nI_max * n, *rC = sI + (size_t)nI_max * n;
                                double *sC = rC + (size_t)nI_max * n;
                                double *qI = sC + (size_t)nI_max * n, *tI = qI + (size_t)nI_max * n;
                                double *qC = tI + (size_t)nI_max * n, *tC = qC + (size_t)nI_max * n;
                                int pass;
                                for (pass = 0; pass < 2; pass++) {
                                        const double *Fx = pass == 0 ? F : H;
                                        double *r_I = pass == 0 ? rI : qI, *s_I = pass == 0 ? sI : tI;
                                        double *r_C = pass == 0 ? rC : qC, *s_C = pass == 0 ? sC : tC;
                                        pair_rs(r_C, s_C, dAC + (size_t)x * nab * n, Fx, I0, J0, nI, nJ, n, both);
                                        int q;
                                        if (pp->onec) {
                                                for (q = 0; q < nI * n; q++) r_C[q] = -r_C[q];
                                                if (both) for (q = 0; q < nJ * n; q++) s_C[q] = -s_C[q];
                                        } else {
                                                pair_rs(r_I, s_I, dAI + (size_t)x * nab * n, Fx, I0, J0, nI, nJ, n, 1);
                                                for (q = 0; q < nI * n; q++) r_C[q] = -(r_I[q] + r_C[q]);
                                                for (q = 0; q < nJ * n; q++) s_C[q] = -(s_I[q] + s_C[q]);
                                        }
                                }
                                int nat, ats[2];
                                double *rr[2], *ss[2], *rq[2], *sq[2];
                                if (pp->onec) {
                                        ats[0] = aI; rr[0] = rC; ss[0] = sC; rq[0] = qC; sq[0] = tC; nat = 1;
                                } else {
                                        ats[0] = aI; rr[0] = rI; ss[0] = sI; rq[0] = qI; sq[0] = tI;
                                        ats[1] = aJ; rr[1] = rC; ss[1] = sC; rq[1] = qC; sq[1] = tC;
                                        nat = 2;
                                }
                                int q;
                                for (q = 0; q < nat; q++) {
                                        const int ya = ats[q];
                                        if (!touched[ya]) {
                                                memset(U + (size_t)ya * 3 * nocc * n, 0, sizeof(double) * 3 * nocc * n);
                                                memset(Up + (size_t)ya * 3 * nocc * n, 0, sizeof(double) * 3 * nocc * n);
                                                touched[ya] = 1;
                                        }
                                        double *Uy = U + ((size_t)ya * 3 + x) * nocc * n, *Upy = Up + ((size_t)ya * 3 + x) * nocc * n;
                                        dgemm_("N", "T", &n, &nocc, &nI, &one, rr[q], &n, co + (size_t)I0 * nocc, &nocc, &one, Uy, &n);
                                        dgemm_("N", "T", &n, &nocc, &nI, &one, rq[q], &n, co + (size_t)I0 * nocc, &nocc, &one, Upy, &n);
                                        if (both) {
                                                dgemm_("N", "T", &n, &nocc, &nJ, &one, ss[q], &n, co + (size_t)J0 * nocc, &nocc, &one, Uy, &n);
                                                dgemm_("N", "T", &n, &nocc, &nJ, &one, sq[q], &n, co + (size_t)J0 * nocc, &nocc, &one, Upy, &n);
                                        }
                                        if (do_f1) {
                                                double *vt = VTt + (((size_t)ya * 3 + x) * nao) * nocc;
                                                dgemm_("T", "N", &nocc, &nI, &n, &one, weta, &n, rr[q], &n, &one, vt + (size_t)I0 * nocc, &nocc);
                                                if (both)
                                                        dgemm_("T", "N", &nocc, &nJ, &n, &one, weta, &n, ss[q], &n, &one, vt + (size_t)J0 * nocc, &nocc);
                                        }
                                }
                        }
                }
                if (nrec) multi_contract_const(Aoc, co, Abuf, rec, nrec, nao, n, nocc);
                for (k = 0; k < nla; k++) {
                        const int aa = la[3 * k];
                        if (!touched[aa]) {
                                memset(U + (size_t)aa * 3 * nocc * n, 0, sizeof(double) * 3 * nocc * n);
                                memset(Up + (size_t)aa * 3 * nocc * n, 0, sizeof(double) * 3 * nocc * n);
                                touched[aa] = 1;
                        }
                }
                /* (2): 2 sum_g (w eta^d U3^{Y} + w psi^d U3'^{Y}), both orders (before the (f2) part enters U) */
                for (at = 0; at < natm; at++) {
                        if (!touched[at]) continue;
                        const int ny = 3, nk = nocc * n, nd_ = nd;
                        /* tmp (3 at, nd) = U3^T wetad + U3'^T wpsid over (i, g) */
                        dgemm_("T", "N", &ny, &nd_, &nk, &two, U + (size_t)at * 3 * nk, &nk, wetad, &nk, &zero, tmp, &ny);
                        dgemm_("T", "N", &ny, &nd_, &nk, &two, Up + (size_t)at * 3 * nk, &nk, wpsid, &nk, &one, tmp, &ny);
                        for (d = 0; d < nd; d++) {
                                const int ra = la[3 * (d / 3)], rx = d % 3;
                                for (y = 0; y < 3; y++) {
                                        const double v = tmp[(size_t)d * 3 + y];
                                        HS(3 * ra + rx, 3 * at + y) += v;
                                        HS(3 * at + y, 3 * ra + rx) += v;
                                }
                        }
                }
                /* per point: Out = 2 Aoc [psi, eta, psi^d]; Out2 = 2 Aoo [psi, eta, psi^d] */
                const int ncol = 2 + nd;
                for (g = 0; g < n; g++) {
                        const double *Ag = Aoc + g * NQ;
                        for (i = 0; i < nocc; i++) {
                                Psi[i * ncol] = psi[(size_t)i * n + g];
                                Psi[i * ncol + 1] = eta[(size_t)i * n + g];
                                for (d = 0; d < nd; d++) Psi[i * ncol + 2 + d] = psid[((size_t)d * nocc + i) * n + g];
                        }
                        dgemm_("N", "N", &ncol, &nao, &nocc, &two, Psi, &ncol, Ag, &nocc, &zero, Out, &ncol);
                        dgemm_("N", "T", &nocc, &nocc, &nao, &one, Ag, &nocc, co, &nocc, &zero, Aoo, &nocc);
                        dgemm_("N", "N", &ncol, &nocc, &nocc, &two, Psi, &ncol, Aoo, &nocc, &zero, Out2, &ncol);
                        int nu;
                        for (nu = 0; nu < nao; nu++) GFb[(size_t)nu * n + g] = Out[(size_t)nu * ncol];
                        for (i = 0; i < nocc; i++) {
                                Ub[(size_t)i * n + g] = Out2[(size_t)i * ncol];
                                Upb[(size_t)i * n + g] = Out2[(size_t)i * ncol + 1];
                        }
                        /* (1c): 4 w eta^{d1} Aoo psi^{d2} = 2 (w eta^{d1}) . Out2[:, 2 + d2], plus the transpose */
                        int d2;
                        for (d = 0; d < nd; d++)
                                for (d2 = 0; d2 < nd; d2++) {
                                        double s = 0;
                                        for (i = 0; i < nocc; i++)
                                                s += wetad[((size_t)d * nocc + i) * n + g] * Out2[(size_t)i * ncol + 2 + d2];
                                        s *= 2;
                                        const int r1 = 3 * la[3 * (d / 3)] + d % 3, r2 = 3 * la[3 * (d2 / 3)] + d2 % 3;
                                        HS(r1, r2) += s;
                                        HS(r2, r1) += s;
                                }
                        if (do_f1) {
                                for (k = 0; k < nla; k++)
                                        for (x = 0; x < 3; x++) {
                                                double *Uy = U + ((size_t)la[3 * k] * 3 + x) * nocc * n;
                                                for (i = 0; i < nocc; i++) Uy[(size_t)i * n + g] += Out2[(size_t)i * ncol + 2 + 3 * k + x];
                                        }
                                for (d = 0; d < nd; d++) {
                                        const int ya = la[3 * (d / 3)], yy = d % 3;
                                        double *vt = VTt + (((size_t)ya * 3 + yy) * nao) * nocc;
                                        for (nu = 0; nu < nao; nu++) {
                                                const double bdv = Out[(size_t)nu * ncol + 2 + d];
                                                if (bdv == 0) continue;
                                                double *row = vt + (size_t)nu * nocc;
                                                for (i = 0; i < nocc; i++) row[i] += bdv * weta[(size_t)i * n + g];
                                        }
                                }
                        }
                }
                /* (1ab): Q_rho = 2 (Y_rho U + C_rho U'), sum_g w d_x d_y phi_rho Q_rho per atom */
                dgemm_("N", "N", &n, &nl, &nocc, &two, Ub, &n, ys, &nocc, &zero, Qm, &n);
                dgemm_("N", "N", &n, &nl, &nocc, &two, Upb, &n, cs, &nocc, &one, Qm, &n);
                for (k = 0; k < nla; k++) {
                        const int aa = la[3 * k], r0 = la[3 * k + 1], r1 = la[3 * k + 2];
                        for (c = 0; c < 6; c++) {
                                double s = 0;
                                int r_;
                                for (r_ = r0; r_ < r1; r_++) {
                                        const double *d2p = a + ((size_t)(4 + c) * nl + r_) * n, *qq = Qm + (size_t)r_ * n;
                                        for (g = 0; g < n; g++) s += w[g] * d2p[g] * qq[g];
                                }
                                const int cx_ = cxy[c][0], cy_ = cxy[c][1];
                                HS(3 * aa + cx_, 3 * aa + cy_) += s;
                                if (cx_ != cy_) HS(3 * aa + cy_, 3 * aa + cx_) += s;
                        }
                }
                if (!do_f1) continue;
                for (d = 0; d < nd; d++) {
                        const int ya = la[3 * (d / 3)], yy = d % 3;
                        double *vt = VTt + (((size_t)ya * 3 + yy) * nao) * nocc;
                        dgemm_("T", "N", &nocc, &nao, &n, &one, wetad + (size_t)d * nocc * n, &n, GFb, &n, &one, vt, &nocc);
                }
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
                {
                        double *wphi = full;
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
#undef HS
#pragma omp barrier
        {
                size_t chunk = (nvo + nthreads - 1) / nthreads;
                size_t s0 = chunk * tid < nvo ? chunk * tid : nvo, s1 = s0 + chunk < nvo ? s0 + chunk : nvo;
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
                const size_t nh_ = (size_t)n3 * n3;
                chunk = (nh_ + nthreads - 1) / nthreads;
                s0 = chunk * tid < nh_ ? chunk * tid : nh_;
                s1 = s0 + chunk < nh_ ? s0 + chunk : nh_;
                for (t = 0; t < nthreads; t++) {
                        if (!hb[t]) continue;
                        for (q = s0; q < s1; q++) hs[q] += hb[t][q];
                }
        }
#pragma omp barrier
        free(Vt); free(VTt); free(Ht); free(a); free(full); free(rmx); free(ix); free(runs); free(cs); free(ys);
        free(F); free(H); free(GFb); free(psi); free(eta); free(weta); free(Ub); free(Upb); free(psid); free(wpsid);
        free(wetad); free(la); free(U); free(Up); free(touched); free(fmax); free(A); free(W); free(dA); free(rs);
        free(Mm); free(bk); free(R); free(Rwk); free(S); free(X); free(xyz); free(Abuf); free(rec); free(Aoc);
        free(Aoo); free(Psi); free(Out); free(Out2); free(tmp); free(Qm);
        for (at = 0; at < natm; at++) free(farR[at]);
        free(farR); free(stamp); free(armin);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
        free(vb); free(vtb); free(hb); free(shl_of); free(atm_of_ao);
        return err;
}
