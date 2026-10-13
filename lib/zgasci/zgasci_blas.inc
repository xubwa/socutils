/* The BLAS kernel, compiled once for each scalar type by zgasci.c (see
 * zgasci_kernel.inc for the macros; ZG_GEMM and ZG_AXPY are the BLAS
 * routines of the scalar type).
 *
 * Design.  As in the block kernel, each of the nine kinds of terms
 * v[P,R] a+_p a+_q a_s a_r (P, R within A, across, or within B) is
 *     sum_{x,y} W[x][y] O_A(x) (x) O_B(y)
 * with a sign that depends on the A string and on the B string only.  One
 * side is taken as the "row" side and the other as the "column" side, and
 * the sum is done in the way of Knowles and Handy / Olsen:
 *
 *   for a target row string r' and a run of source column strings c:
 *     gather  D[x][c] = sign * C[row(r', x)][c]   for the operators x that
 *                                                 give a row of the space,
 *     multiply F[y][c] = sum_x W[y][x] D[x][c]    (GEMM),
 *     scatter sigma[r'][O(y) c] += sign * F[y][c] for the y that act on c.
 *
 * The gather copies whole stretches of rows of C, the multiplication is
 * dense, and only the scatter is indirect (one addition per connected
 * column string, where the other kernels do one per connected determinant).
 * The price: F is worked out for every y that acts on some string of the
 * run, though each string takes only some of them; small products are
 * therefore done as dot products over x for the pairs (y, c) that are
 * needed.
 *
 * Electrons pass between the sides (spinors), so the row side must be chosen
 * per kind.  The operators on the row side cost nothing extra (only those
 * valid on r' are taken), those on the column side are dense, so the side
 * with more operators is the row side, and for a+ a+ (x) a a the side of the
 * creation operators (fewer electrons than holes):
 *
 *   half 0, rows = A strings:  (AA,AA) (AA,AB) (AB,AA) (AB,AB) (AA,BB)
 *   half 1, rows = B strings:  (BB,BB) (BB,AB) (AB,BB) (BB,AA)
 *
 * Half 1 is half 0 with the sides exchanged, on the transposed vector: C
 * is transposed block by block into workspace, its result is transposed
 * back into sigma, and half 0 adds to that (two vectors of workspace).
 * Kinds without operators on the column side ((AA,AA), (BB,BB)) are AXPYs of
 * whole rows.
 *
 * Blocks.  A piece of work is a range of rows of a bucket of the row side,
 * with every block these rows run through; its thread alone writes these
 * rows of the result, so there are no atomic additions (within a half; the
 * halves run one after the other).  For the rows of a piece the operators x
 * are enumerated once per kind and sorted by the bucket of the source row;
 * for each source column bucket, in runs of `cols` strings, the operators y
 * are enumerated once for the whole piece (their cost is shared by its
 * rows), and those whose target is not a block of the piece are dropped, as
 * are the rows of D whose source block does not exist.  With symmetry labels
 * this cuts x and y to the operators of matching label automatically.
 * Nothing is stored at the level of strings beyond a piece of work.
 *
 * BLAS runs sequentially inside the parallel region (see zg_blas_single in
 * zgasci.c). */

#ifndef ZG_BLAS_GROW
#define ZG_BLAS_GROW(ptr, cap, need)                         \
  do {                                                       \
    if ((int64_t)(need) > (cap)) {                           \
      (cap) = 2 * (int64_t)(need) + 64;                      \
      (ptr) = realloc((ptr), (size_t)(cap) * sizeof(*(ptr))); \
    }                                                        \
  } while (0)
#endif

/* out = the transposed vector of in (to_native 0), or back. */
static void ZG_NAME(blas_transpose)(const zg_plan *p, const ZG_SCALAR *in,
                                    ZG_SCALAR *out, int to_native) {
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
#pragma omp parallel for schedule(dynamic, 8)
  for (int64_t it = 0; it < p->ntpiece; ++it) {
    const zg_tpiece *piece = &p->tpieces[it];
    const int ba = (int)(piece->k / sb->nclass);
    const int bb = p->match[piece->k];
    const int64_t na = sa->bucket_size[ba], nb = sb->bucket_size[bb];
    const int64_t on = p->blockoff[piece->k];
    const int64_t ot = p->blockoff_t[(size_t)bb * sa->nclass + sa->bucket_class[ba]];
    for (int64_t j0 = 0; j0 < nb; j0 += 64) {
      const int64_t j1 = j0 + 64 < nb ? j0 + 64 : nb;
      for (int64_t i = piece->i0; i < piece->i1; ++i) {
        if (to_native) {
          for (int64_t j = j0; j < j1; ++j) out[on + i * nb + j] = in[ot + j * na + i];
        } else {
          for (int64_t j = j0; j < j1; ++j) out[ot + j * na + i] = in[on + i * nb + j];
        }
      }
    }
  }
}

/* One half: sv += (the kinds of half h) cv, both seen with the strings of
 * side h as rows.  w[k], live[k]: the matrices of the nine kinds. */
static void ZG_NAME(blas_half)(const zg_plan *p, int h, ZG_SCALAR *const *w,
                               const int *live, const ZG_SCALAR *cv,
                               ZG_SCALAR *sv, int64_t cols, double direct_ratio,
                               double direct_small, double *stat) {
  const int rs = h, cs = 1 - h;
  const zg_side *sr = &p->s[rs], *sc = &p->s[cs];
  const int ncc = sc->nclass;
  const int32_t *match = h ? p->match_t : p->match;
  const int64_t *off = h ? p->blockoff_t : p->blockoff;
  const int nk = p->nbkind[h];
  const zg_bkind *kinds = p->bkinds[h];
  const int nsr = p->nside[rs], nsc = p->nside[cs];
  /* Bounds of the operator lists of a row and of a column string. */
  int64_t capx = 1, capy = 1, maxcidx = 1;
  for (int k = 0; k < nk; ++k) {
    int64_t bx = 1, by = 1;
    for (int i = 0; i < kinds[k].kr; ++i) bx *= nsr;
    for (int i = 0; i + 1 < kinds[k].kr; ++i) bx /= kinds[k].rel_r[i] ? 2 : 1;
    for (int i = 0; i < kinds[k].kc; ++i) by *= nsc;
    if (bx + 1 > capx) capx = bx + 1;
    if (by + 1 > capy) capy = by + 1;
    if (kinds[k].ncidx > maxcidx) maxcidx = kinds[k].ncidx;
  }
  double st_gemm = 0, st_direct = 0, st_axpy = 0, st_scatter = 0;
  /* Seconds (ZGASCI_STATS): row operators, axpy, candidates, column
   * operators, gathers of D and of W, products, dot products, scatter. */
  const int prof = getenv("ZGASCI_STATS") != NULL;
  double tm[9] = {0};
#define ZG_TICK(i)                         \
  do {                                     \
    if (prof) {                            \
      const double now_ = omp_get_wtime(); \
      tm[i] += now_ - last;                \
      last = now_;                         \
    }                                      \
  } while (0)
#pragma omp parallel reduction(+ : st_gemm, st_direct, st_axpy, st_scatter, tm[:9])
  {
    double last = prof ? omp_get_wtime() : 0.0;
    const int blas_before = zg_blas_single();
    const int one = 1;
    const ZG_SCALAR sone = 1, szero = 0;
    /* Operators of a row, as enumerated. */
    int64_t *xs = malloc((size_t)capx * sizeof(int64_t));
    uint64_t *xsrc = malloc((size_t)capx * sizeof(uint64_t));
    int8_t *xsign = malloc((size_t)capx);
    int32_t *xrun = malloc((size_t)capx * sizeof(int32_t));
    int32_t *xpos = malloc((size_t)capx * sizeof(int32_t));
    /* Runs of the operators of a row by source bucket. */
    int32_t *rstamp = calloc((size_t)sr->nbucket + 1, sizeof(int32_t));
    int32_t *rlocal = malloc((size_t)(sr->nbucket + 1) * sizeof(int32_t));
    int64_t *lfill = malloc((size_t)(sr->nbucket + 1) * sizeof(int64_t));
    int32_t *lbucket = malloc((size_t)(sr->nbucket + 1) * sizeof(int32_t));
    int32_t stamp = 0;
    /* The operators of the rows of a piece: entries (index in W, place of
     * the source row, sign) in runs of one source bucket; rk_run[r * nk + k]
     * is the first run of row r and kind k. */
    int64_t ecap = 0, rcap = 0;
    int64_t *ent_idx = NULL;
    int32_t *ent_pos = NULL;
    int8_t *ent_sign = NULL;
    int32_t *run_bucket = NULL;
    int64_t *run_start = NULL;
    int64_t *rk_run = malloc((size_t)(ZG_BLAS_ROWS * nk + 1) * sizeof(int64_t));
    /* Target blocks of a piece. */
    int32_t *tb_class = malloc((size_t)(ncc + 1) * sizeof(int32_t));
    int32_t *tb_bucket = malloc((size_t)(ncc + 1) * sizeof(int32_t));
    /* Candidate source column buckets. */
    int32_t *cstamp = calloc((size_t)sc->nbucket + 1, sizeof(int32_t));
    int32_t *rseen = calloc((size_t)sr->nbucket + 1, sizeof(int32_t));
    int32_t *cand = malloc((size_t)(sc->nbucket + 1) * sizeof(int32_t));
    int32_t cmark = 0;
    /* Column operators of a run of source strings: (y, source column, place
     * of the target in a row of the piece, columns of that block, sign). */
    int64_t *yx = malloc((size_t)capy * sizeof(int64_t));
    uint64_t *ytgt = malloc((size_t)capy * sizeof(uint64_t));
    int8_t *ysign = malloc((size_t)capy);
    int32_t *ymap = malloc((size_t)maxcidx * sizeof(int32_t));
    int64_t *ylist = malloc((size_t)maxcidx * sizeof(int64_t));
    for (int64_t i = 0; i < maxcidx; ++i) ymap[i] = -1;
    int64_t tcap = 0;
    int32_t *t_y = NULL, *t_col = NULL;
    int64_t *t_base = NULL, *t_ncol = NULL;
    int8_t *t_sign = NULL;
    /* Matrices. */
    int64_t dcap = 0, wcap = 0, fcap = 0, xwcap = 0;
    ZG_SCALAR *dmat = NULL, *wmat = NULL, *fmat = NULL;
    int64_t *xw = NULL;

#pragma omp for schedule(dynamic, 1)
    for (int64_t it = 0; it < p->nrtask[h]; ++it) {
      const zg_rtask *task = &p->rtasks[h][it];
      const int bt = task->bucket;
      const int64_t r0 = task->r0;
      if (prof) last = omp_get_wtime();
      const int m = (int)(task->r1 - task->r0);
      int ntb = 0;
      for (int ct = 0; ct < ncc; ++ct) {
        const int bb = match[(size_t)bt * ncc + ct];
        if (bb < 0) continue;
        tb_class[ntb] = ct;
        tb_bucket[ntb++] = bb;
      }
      /* The row operators of the piece. */
      int64_t nent = 0, nrun = 0;
      for (int r = 0; r < m; ++r) {
        const uint64_t wrow =
            zg_side_word(p, rs, sr->bucket_sid[sr->bucket_start[bt] + r0 + r]);
        for (int k = 0; k < nk; ++k) {
          const zg_bkind *kd = &kinds[k];
          rk_run[r * nk + k] = nrun;
          if (!live[kd->term]) continue;
          const int n = zg_enum_ops(kd->kr, kd->kind_r, kd->rel_r, nsr, wrow, xs,
                                    xsrc, xsign, (int)capx);
          ++stamp;
          int nl = 0;
          int64_t kept = 0;
          const int64_t runs_most = nrun + (n < sr->nbucket ? n : sr->nbucket) + 2;
          if (runs_most > rcap) {
            rcap = 2 * runs_most + 64;
            run_bucket = realloc(run_bucket, (size_t)rcap * sizeof(int32_t));
            run_start = realloc(run_start, (size_t)rcap * sizeof(int64_t));
          }
          for (int e = 0; e < n; ++e) {
            const int64_t sid = zg_word_sid(p, rs, xsrc[e]);
            if (sid < 0) {
              xrun[e] = -1;
              continue;
            }
            const int b = sr->bucket[sid];
            if (rstamp[b] != stamp) {
              rstamp[b] = stamp;
              rlocal[b] = nl;
              lfill[nl] = 0;
              lbucket[nl++] = b;
            }
            xrun[e] = rlocal[b];
            xpos[e] = (int32_t)sr->pos[sid];
            lfill[rlocal[b]]++;
            ++kept;
          }
          if (nent + kept + 1 > ecap) {
            ecap = 2 * (nent + kept + 1) + 64;
            ent_idx = realloc(ent_idx, (size_t)ecap * sizeof(int64_t));
            ent_pos = realloc(ent_pos, (size_t)ecap * sizeof(int32_t));
            ent_sign = realloc(ent_sign, (size_t)ecap);
          }
          int64_t at = nent;
          for (int l = 0; l < nl; ++l) {
            const int64_t count = lfill[l];
            run_bucket[nrun + l] = lbucket[l];
            run_start[nrun + l] = at;
            lfill[l] = at;
            at += count;
          }
          for (int e = 0; e < n; ++e) {
            if (xrun[e] < 0) continue;
            const int64_t q = lfill[xrun[e]]++;
            int8_t sg = xsign[e];
            if (kd->par_r && (__builtin_popcountll(xsrc[e]) & 1)) sg = (int8_t)-sg;
            ent_idx[q] = xs[e];
            ent_pos[q] = xpos[e];
            ent_sign[q] = sg;
          }
          nrun += nl;
          nent += kept;
        }
      }
      rk_run[m * nk] = nrun;
      if (nrun + 1 > rcap) {
        rcap = nrun + 64;
        run_bucket = realloc(run_bucket, (size_t)rcap * sizeof(int32_t));
        run_start = realloc(run_start, (size_t)rcap * sizeof(int64_t));
      }
      run_start[nrun] = nent;
      ZG_TICK(0);

      for (int k = 0; k < nk; ++k) {
        const zg_bkind *kd = &kinds[k];
        if (!live[kd->term]) continue;
        const ZG_SCALAR *wk = w[kd->term];
        if (kd->kc == 0) {
          /* Whole rows. */
          for (int r = 0; r < m; ++r) {
            for (int64_t run = rk_run[r * nk + k]; run < rk_run[r * nk + k + 1]; ++run) {
              const int bs = run_bucket[run];
              for (int t = 0; t < ntb; ++t) {
                const int ct = tb_class[t], bb = tb_bucket[t];
                if (match[(size_t)bs * ncc + ct] != bb) continue;
                const int64_t nb = sc->bucket_size[bb];
                const int nbi = (int)nb;
                ZG_SCALAR *dst = sv + off[(size_t)bt * ncc + ct] + (r0 + r) * nb;
                const ZG_SCALAR *src = cv + off[(size_t)bs * ncc + ct];
                for (int64_t q = run_start[run]; q < run_start[run + 1]; ++q) {
                  ZG_SCALAR f = wk[ent_idx[q] * kd->wr];
                  if (f == 0) continue;
                  if (ent_sign[q] < 0) f = -f;
                  const ZG_SCALAR *from = src + ent_pos[q] * nb;
                  if (nb < 16) {
                    for (int64_t j = 0; j < nb; ++j) dst[j] += f * from[j];
                  } else {
                    ZG_AXPY(&nbi, &f, from, &one, dst, &one);
                  }
                }
                st_axpy += (double)nb * (run_start[run + 1] - run_start[run]);
              }
            }
          }
          ZG_TICK(1);
          continue;
        }
        /* The source column buckets that pair with a source row bucket. */
        ++cmark;
        int ncand = 0;
        int64_t maxx = 0;
        for (int r = 0; r < m; ++r) {
          const int64_t ra = rk_run[r * nk + k], rb = rk_run[r * nk + k + 1];
          if (run_start[rb] - run_start[ra] > maxx) maxx = run_start[rb] - run_start[ra];
          for (int64_t run = ra; run < rb; ++run) {
            const int bs = run_bucket[run];
            if (rseen[bs] == cmark) continue;
            rseen[bs] = cmark;
            for (int cc = 0; cc < ncc; ++cc) {
              const int bb = match[(size_t)bs * ncc + cc];
              if (bb < 0 || cstamp[bb] == cmark) continue;
              cstamp[bb] = cmark;
              cand[ncand++] = bb;
            }
          }
        }
        ZG_TICK(2);
        if (maxx == 0) continue;
        ZG_BLAS_GROW(dmat, dcap, maxx * cols);
        ZG_BLAS_GROW(xw, xwcap, maxx);
        for (int ic = 0; ic < ncand; ++ic) {
          const int bbs = cand[ic], cls = sc->bucket_class[bbs];
          if (!zg_blas_reach(p, h, kd, cls, bt)) continue;
          const int64_t nbs = sc->bucket_size[bbs];
          for (int64_t j0 = 0; j0 < nbs; j0 += cols) {
            const int64_t nc = j0 + cols < nbs ? cols : nbs - j0;
            /* The column operators of this run of strings. */
            int64_t ne = 0;
            int nyc = 0;
            for (int64_t j = 0; j < nc; ++j) {
              const uint64_t wc =
                  zg_side_word(p, cs, sc->bucket_sid[sc->bucket_start[bbs] + j0 + j]);
              const int n = zg_enum_ops(kd->kc, kd->kind_c, kd->rel_c, nsc, wc, yx,
                                        ytgt, ysign, (int)capy);
              const int flip = kd->par_c && (__builtin_popcountll(wc) & 1);
              if (ne + n > tcap) {
                tcap = 2 * (ne + n) + 64;
                t_y = realloc(t_y, (size_t)tcap * sizeof(int32_t));
                t_col = realloc(t_col, (size_t)tcap * sizeof(int32_t));
                t_base = realloc(t_base, (size_t)tcap * sizeof(int64_t));
                t_ncol = realloc(t_ncol, (size_t)tcap * sizeof(int64_t));
                t_sign = realloc(t_sign, (size_t)tcap);
              }
              for (int e = 0; e < n; ++e) {
                const int64_t sid = zg_word_sid(p, cs, ytgt[e]);
                if (sid < 0) continue;
                const int tb = sc->bucket[sid];
                const size_t kk = (size_t)bt * ncc + sc->bucket_class[tb];
                if (match[kk] != tb) continue;
                const int64_t idx =
                    kd->kc == 2 ? yx[e] / nsc + (yx[e] % nsc) * nsc : yx[e];
                if (ymap[idx] < 0) {
                  ymap[idx] = nyc;
                  ylist[nyc++] = idx;
                }
                t_y[ne] = ymap[idx];
                t_col[ne] = (int32_t)j;
                t_base[ne] = off[kk] + sc->pos[sid];
                t_ncol[ne] = sc->bucket_size[tb];
                t_sign[ne] = (int8_t)(flip ? -ysign[e] : ysign[e]);
                ++ne;
              }
            }
            ZG_TICK(3);
            if (ne == 0) continue;
            ZG_BLAS_GROW(wmat, wcap, maxx * nyc);
            ZG_BLAS_GROW(fmat, fcap, (int64_t)nyc * nc);
            for (int r = 0; r < m; ++r) {
              /* D: the source rows that have a block with this bucket. */
              int64_t nx = 0;
              for (int64_t run = rk_run[r * nk + k]; run < rk_run[r * nk + k + 1]; ++run) {
                const size_t kk = (size_t)run_bucket[run] * ncc + cls;
                if (match[kk] != bbs) continue;
                const ZG_SCALAR *src = cv + off[kk] + j0;
                for (int64_t q = run_start[run]; q < run_start[run + 1]; ++q) {
                  const ZG_SCALAR *from = src + ent_pos[q] * nbs;
                  ZG_SCALAR *to = dmat + nx * nc;
                  if (ent_sign[q] > 0) {
                    for (int64_t j = 0; j < nc; ++j) to[j] = from[j];
                  } else {
                    for (int64_t j = 0; j < nc; ++j) to[j] = -from[j];
                  }
                  xw[nx++] = ent_idx[q] * kd->wr;
                }
              }
              ZG_TICK(4);
              if (nx == 0) continue;
              /* W[y][x] of these x and y, as a matrix (nyc, nx). */
              for (int64_t x = 0; x < nx; ++x) {
                const ZG_SCALAR *from = wk + xw[x];
                ZG_SCALAR *to = wmat + x * nyc;
                for (int y = 0; y < nyc; ++y) to[y] = from[ylist[y] * kd->wc];
              }
              ZG_TICK(5);
              ZG_SCALAR *row = sv;
              const int64_t rr = r0 + r;
              if ((double)nyc * nc > direct_ratio * (double)ne ||
                  (double)nx * nyc * nc < direct_small) {
                /* Few operators: only the (y, column) that are needed. */
                for (int64_t e = 0; e < ne; ++e) {
                  const ZG_SCALAR *wy = wmat + t_y[e];
                  const ZG_SCALAR *dc = dmat + t_col[e];
                  ZG_SCALAR sum = 0;
                  for (int64_t x = 0; x < nx; ++x) sum += wy[x * nyc] * dc[x * nc];
                  if (t_sign[e] < 0) sum = -sum;
                  row[t_base[e] + rr * t_ncol[e]] += sum;
                }
                st_direct += (double)ne * nx;
                ZG_TICK(7);
              } else {
                const int mi = nyc, ni = (int)nc, ki = (int)nx;
                ZG_GEMM("N", "T", &mi, &ni, &ki, &sone, wmat, &mi, dmat, &ni, &szero,
                        fmat, &mi);
                ZG_TICK(6);
                for (int64_t e = 0; e < ne; ++e) {
                  const ZG_SCALAR f = fmat[t_y[e] + (int64_t)t_col[e] * nyc];
                  if (t_sign[e] > 0) {
                    row[t_base[e] + rr * t_ncol[e]] += f;
                  } else {
                    row[t_base[e] + rr * t_ncol[e]] -= f;
                  }
                }
                st_gemm += (double)nx * nyc * nc;
                ZG_TICK(8);
              }
              st_scatter += (double)ne;
            }
            for (int y = 0; y < nyc; ++y) ymap[ylist[y]] = -1;
          }
        }
      }
    }
    free(xs);
    free(xsrc);
    free(xsign);
    free(xrun);
    free(xpos);
    free(rstamp);
    free(rlocal);
    free(lfill);
    free(lbucket);
    free(ent_idx);
    free(ent_pos);
    free(ent_sign);
    free(run_bucket);
    free(run_start);
    free(rk_run);
    free(tb_class);
    free(tb_bucket);
    free(cstamp);
    free(rseen);
    free(cand);
    free(yx);
    free(ytgt);
    free(ysign);
    free(ymap);
    free(ylist);
    free(t_y);
    free(t_col);
    free(t_base);
    free(t_ncol);
    free(t_sign);
    free(dmat);
    free(wmat);
    free(fmat);
    free(xw);
    zg_blas_restore(blas_before);
  }
  stat[0] += st_gemm;
  stat[1] += st_direct;
  stat[2] += st_axpy;
  stat[3] += st_scatter;
  if (prof) {
    fprintf(stderr,
            "zgasci blas half %d, thread seconds: row operators %.2f, axpy %.2f, "
            "candidates %.2f, column operators %.2f, gather D %.2f, gather W %.2f, "
            "products %.2f, dot products %.2f, scatter %.2f\n",
            h, tm[0], tm[1], tm[2], tm[3], tm[4], tm[5], tm[6], tm[7], tm[8]);
  }
#undef ZG_TICK
}

void ZG_NAME(zg_sigma_blas)(const zg_plan *p, const ZG_SCALAR *v,
                            const ZG_SCALAR *c, ZG_SCALAR *sigma) {
  if (!p->block_ok) {
    ZG_NAME(zg_sigma)(p, v, c, sigma);
    return;
  }
  ZG_SCALAR *w[9];
  int live[9];
  const char *only = getenv("ZGASCI_TERMS");
  const int mask = only ? atoi(only) : 511;
  for (int k = 0; k < 9; ++k) {
    w[k] = ZG_NAME(term_matrix)(p, &p->terms[k], v, &live[k]);
    live[k] = live[k] && (mask >> k & 1);
  }
  /* ZGASCI_BLAS_COLS: source columns of a product.  A product is done as
   * dot products when it would do more than ZGASCI_BLAS_DIRECT times their
   * multiply-adds, or fewer than ZGASCI_BLAS_SMALL multiply-adds. */
  int64_t cols = 256;
  double direct_ratio = ZG_BLAS_DIRECT, direct_small = 512.0;
  const char *text = getenv("ZGASCI_BLAS_COLS");
  if (text && atol(text) > 0) cols = atol(text);
  text = getenv("ZGASCI_BLAS_DIRECT");
  if (text) direct_ratio = atof(text);
  text = getenv("ZGASCI_BLAS_SMALL");
  if (text) direct_small = atof(text);
  double stat[4] = {0, 0, 0, 0};
  const double t_start = omp_get_wtime();
  double t_half1 = t_start;
  int any1 = 0;
  for (int k = 0; k < p->nbkind[1]; ++k) any1 = any1 || live[p->bkinds[1][k].term];
  if (any1) {
    ZG_SCALAR *ct = malloc(((size_t)p->dim + 1) * sizeof(ZG_SCALAR));
    ZG_SCALAR *st = malloc(((size_t)p->dim + 1) * sizeof(ZG_SCALAR));
    ZG_NAME(blas_transpose)(p, c, ct, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < p->dim; ++i) st[i] = 0;
    ZG_NAME(blas_half)(p, 1, w, live, ct, st, cols, direct_ratio, direct_small, stat);
    ZG_NAME(blas_transpose)(p, st, sigma, 1);
    free(ct);
    free(st);
    t_half1 = omp_get_wtime();
  } else {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < p->dim; ++i) sigma[i] = 0;
  }
  ZG_NAME(blas_half)(p, 0, w, live, c, sigma, cols, direct_ratio, direct_small, stat);
  if (getenv("ZGASCI_STATS")) {
    fprintf(stderr,
            "zgasci blas: multiply-adds %.3e in products, %.3e in dot products, "
            "%.3e in axpy; %.3e scattered; %.2f s for the half of the transposed "
            "vector, %.2f s for the other\n",
            stat[0], stat[1], stat[2], stat[3], t_half1 - t_start,
            omp_get_wtime() - t_half1);
  }
  for (int k = 0; k < 9; ++k) free(w[k]);
}
