/* The BLAS kernel, compiled once for each scalar type by zgasci.c (see
 * zgasci_kernel.inc for the macros; ZG_GEMM and ZG_AXPY are the BLAS
 * routines of the scalar type, ZG_BLAS_DIRECT a threshold, see below).
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
 * The gather copies stretches of rows of C, the multiplication is dense, and
 * only the scatter is indirect: one addition per connected column string,
 * where the other kernels do one multiply-add per connected determinant.
 * The price: F is worked out for every y that acts on some string of the
 * run, though each string takes only some of them.  When that is more than
 * ZG_BLAS_DIRECT times what is needed (dilute spaces, small blocks), or the
 * product is tiny, the needed elements F[y][c] are instead worked out one
 * by one as dot products over x, reading C in place.
 *
 * Electrons pass between the sides (spinors), so the row side is chosen per
 * kind.  The operators on the row side cost nothing extra (only those valid
 * on r' are taken), those on the column side are dense, so the side with
 * more operators is the row side, and for a+ a+ (x) a a the side of the
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
 * Blocks.  A piece of work is a range of rows of a bucket of the row side
 * (at most ZG_BLAS_ROWS), with every block these rows run through; its
 * thread alone writes these rows of the result, so there are no atomic
 * additions (within a half; the halves run one after the other).  For a
 * piece and a kind:
 *   - the operators are placed in the sub-groups in every way that leads to
 *     a class of source strings that can matter (zg_asg_list: the source
 *     row class must pair with a column class from which the column
 *     operators reach a target block of the piece), so that constraints are
 *     used before any string is touched;
 *   - the row operators of each row are enumerated placement by placement,
 *     the numbers of the source strings from the ranks of their sub-strings
 *     (zg_asg_enum), with a filter on the labels where the column side can
 *     make up for few changes (zg_bkind), and sorted by source bucket;
 *   - the column operators are enumerated from the target columns of the
 *     piece (so their cost is shared by its rows) and sorted by source
 *     bucket and by run of `cols` source columns; those whose source bucket
 *     has no block with a source row bucket are dropped;
 *   - for each run of source columns and each row: gather, product or dot
 *     products, scatter.
 * With symmetry labels this cuts x and y to the operators of matching label
 * automatically.  Nothing is stored at the level of strings beyond a piece
 * of work.
 *
 * BLAS runs sequentially inside the parallel region (see zg_blas_single in
 * zgasci.c).  ZGASCI_STATS prints the multiply-adds and the thread seconds
 * of each step. */

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
 * side h as rows.  w[k], live[k]: the matrices of the nine kinds.
 * Density mode (gacc not NULL; w and sv unused): the same loops add
 * conj(bv[target]) s_x s_y cv[source] to G[x][y] of the kind, gacc[k] in the
 * layout of w[k] (zg_rdm2_blas), by dot products. */
static void ZG_NAME(blas_half)(const zg_plan *p, int h, ZG_SCALAR *const *w,
                               const int *live, const ZG_SCALAR *cv,
                               ZG_SCALAR *sv, int64_t cols, double direct_ratio,
                               double direct_small, double *stat,
                               const ZG_SCALAR *bv, ZG_SCALAR *const *gacc) {
  const int rdm = gacc != NULL;
  const int rs = h, cs = 1 - h;
  const zg_side *sr = &p->s[rs], *sc = &p->s[cs];
  const int ncc = sc->nclass, ncr = sr->nclass;
  const int32_t *match = h ? p->match_t : p->match;
  const int64_t *off = h ? p->blockoff_t : p->blockoff;
  const int64_t *pair_ptr = p->pair_ptr[h];
  const int32_t *pair_cls = p->pair_cls[h];
  const int nk = p->nbkind[h];
  const zg_bkind *kinds = p->bkinds[h];
  const int nsr = p->nside[rs], nsc = p->nside[cs];
  const uint64_t *bpr = p->bp[rs], *bpc = p->bp[cs];
  /* Whether a row class and a column class pair. */
#define ZG_PAIR(rc, cc) \
  (h ? p->cpair[(size_t)(cc) * ncr + (rc)] : p->cpair[(size_t)(rc) * ncc + (cc)])
  /* Bounds: operators of a row and of a column string, placements. */
  int64_t capx = 1, capy = 1, maxcidx = 1, cap_ar = 1, cap_ac = 1;
  for (int k = 0; k < nk; ++k) {
    int64_t bx = 1, by = 1, ar = 1, ac = 1;
    for (int i = 0; i < kinds[k].kr; ++i) bx *= nsr, ar *= sr->nsub;
    for (int i = 0; i + 1 < kinds[k].kr; ++i) bx /= kinds[k].rel_r[i] ? 2 : 1;
    for (int i = 0; i < kinds[k].kc; ++i) by *= nsc, ac *= sc->nsub;
    if (bx + 1 > capx) capx = bx + 1;
    if (by + 1 > capy) capy = by + 1;
    if (ar > cap_ar) cap_ar = ar;
    if (ac > cap_ac) cap_ac = ac;
    if (kinds[k].ncidx > maxcidx) maxcidx = kinds[k].ncidx;
  }
  /* Column operators are worked through when this many have been found. */
  const int64_t flush_at = (int64_t)1 << 20;
  double st_gemm = 0, st_direct = 0, st_axpy = 0, st_scatter = 0;
  /* Seconds (ZGASCI_STATS): row operators, axpy, candidates, column
   * operators, gathers of D and of W, products, dot products, scatter. */
  const int prof = getenv("ZGASCI_STATS") != NULL;
  const int filter = getenv("ZGASCI_BLAS_NOFILTER") == NULL;
  double tm[11] = {0};
  double cnt[4] = {0}; /* rows x kinds, placements tried, row operators, column operators */
#define ZG_TICK(i)                         \
  do {                                     \
    if (prof) {                            \
      const double now_ = omp_get_wtime(); \
      tm[i] += now_ - last;                \
      last = now_;                         \
    }                                      \
  } while (0)
#pragma omp parallel reduction(+ : st_gemm, st_direct, st_axpy, st_scatter, tm[:11], cnt[:4])
  {
    double last = prof ? omp_get_wtime() : 0.0;
    const int blas_before = zg_blas_single();
    const int one = 1;
    const ZG_SCALAR sone = 1, szero = 0;
    /* Placements of the operators of the kind at hand. */
    zg_asg *asg_r = malloc((size_t)cap_ar * sizeof(zg_asg));
    zg_asg *asg_c = malloc((size_t)cap_ac * sizeof(zg_asg));
    uint8_t *okr = malloc((size_t)ncr + 1);
    uint8_t *reach = calloc((size_t)ncc + 1, 1);
    int32_t *reach_list = malloc((size_t)(ncc + 1) * sizeof(int32_t));
    uint8_t *okc = calloc((size_t)ncc + 1, 1);
    int64_t rk[ZG_MAX_SUB];
    uint32_t sm[ZG_MAX_SUB]; /* a string as its sub-strings */
    /* Operators of a string, as enumerated. */
    int64_t *xs = malloc((size_t)(capx > capy ? capx : capy) * sizeof(int64_t));
    int64_t *xsid = malloc((size_t)(capx > capy ? capx : capy) * sizeof(int64_t));
    int8_t *xsign = malloc((size_t)(capx > capy ? capx : capy));
    int32_t *xrun = malloc((size_t)capx * sizeof(int32_t));
    /* Runs of the operators of a row by source bucket. */
    int32_t *rstamp = calloc((size_t)sr->nbucket + 1, sizeof(int32_t));
    int32_t *rlocal = malloc((size_t)(sr->nbucket + 1) * sizeof(int32_t));
    int64_t *lfill = malloc((size_t)(sr->nbucket + 1) * sizeof(int64_t));
    int32_t *lbucket = malloc((size_t)(sr->nbucket + 1) * sizeof(int32_t));
    int32_t stamp = 0;
    /* The operators of the rows of a piece for one kind: entries (index in
     * W, place of the source row, sign) in runs of one source bucket;
     * row_run[r] is the first run of row r. */
    int64_t ecap = 0, rcap = 0;
    int64_t *ent_idx = NULL;
    int32_t *ent_pos = NULL;
    int8_t *ent_sign = NULL;
    int32_t *run_bucket = NULL;
    int64_t *run_start = NULL;
    int64_t row_run[ZG_BLAS_ROWS + 1];
    /* Target blocks of a piece. */
    int32_t *tb_class = malloc((size_t)(ncc + 1) * sizeof(int32_t));
    int32_t *tb_bucket = malloc((size_t)(ncc + 1) * sizeof(int32_t));
    int64_t *tb_off = malloc((size_t)(ncc + 1) * sizeof(int64_t));
    int64_t *tb_ncol = malloc((size_t)(ncc + 1) * sizeof(int64_t));
    /* Source column buckets (candidates): those that have a block with a
     * source row bucket; for each, the (row, run) that pair with it. */
    int32_t *cstamp = calloc((size_t)sc->nbucket + 1, sizeof(int32_t));
    int32_t *clocal = malloc((size_t)(sc->nbucket + 1) * sizeof(int32_t));
    int32_t *cand = malloc((size_t)(sc->nbucket + 1) * sizeof(int32_t));
    int64_t *lr_start = malloc((size_t)(sc->nbucket + 2) * sizeof(int64_t));
    int64_t *koff = malloc((size_t)(sc->nbucket + 2) * sizeof(int64_t));
    int32_t cmark = 0;
    int64_t lcap = 0;
    int32_t *lr_lb = NULL, *lr_row = NULL, *sl_row = NULL;
    int64_t *lr_runid = NULL, *sl_runid = NULL;
    /* Column operators: (index in W, source bucket and place, target block
     * and column, sign), as found and sorted by run of source strings. */
    int32_t *ymap = malloc((size_t)maxcidx * sizeof(int32_t));
    int64_t *ylist = malloc((size_t)maxcidx * sizeof(int64_t));
    for (int64_t i = 0; i < maxcidx; ++i) ymap[i] = -1;
    int64_t tcap = 0;
    int64_t *u_idx = NULL, *u_key = NULL;
    int32_t *u_pos = NULL, *u_tb = NULL, *u_tcol = NULL;
    int8_t *u_sign = NULL;
    int32_t *t_y = NULL, *t_col = NULL, *t_tb = NULL, *t_tcol = NULL;
    int8_t *t_sign = NULL;
    int64_t kcap = 0;
    int64_t *kstart = NULL;
    /* Matrices. */
    int64_t dcap = 0, wcap = 0, fcap = 0, xwcap = 0;
    ZG_SCALAR *dmat = NULL, *wmat = NULL, *fmat = NULL;
    int64_t *xw = NULL;
    const ZG_SCALAR **rowptr = NULL;
    int8_t *rowsign = NULL;
    /* Density mode: G of each kind for this thread, made when first used. */
    ZG_SCALAR *gl[9] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

#pragma omp for schedule(dynamic, 1)
    for (int64_t it = 0; it < p->nrtask[h]; ++it) {
      const zg_rtask *task = &p->rtasks[h][it];
      const int bt = task->bucket;
      const int64_t r0 = task->r0;
      const int m = (int)(task->r1 - task->r0);
      if (prof) last = omp_get_wtime();
      int ntb = 0;
      for (int64_t i = pair_ptr[bt]; i < pair_ptr[bt + 1]; ++i) {
        tb_class[ntb] = pair_cls[i];
        tb_bucket[ntb] = match[(size_t)bt * ncc + pair_cls[i]];
        tb_off[ntb] = off[(size_t)bt * ncc + pair_cls[i]];
        tb_ncol[ntb] = sc->bucket_size[tb_bucket[ntb]];
        ++ntb;
      }
      for (int k = 0; k < nk; ++k) {
        const zg_bkind *kd = &kinds[k];
        if (!live[kd->term]) continue;
        /* Product plans: the kinds without column operators are done by
         * blas_rows (tiles over the columns too). */
        if (p->product && !rdm && kd->kc == 0) continue;
        const ZG_SCALAR *wk = rdm ? NULL : w[kd->term];
        ZG_SCALAR *gk = NULL;
        /* The classes of source rows that can matter, the placements. */
        int nreach = 0;
        if (kd->kc == 0) {
          for (int c = 0; c < ncr; ++c) {
            okr[c] = 0;
            for (int t = 0; t < ntb && !okr[c]; ++t) okr[c] = ZG_PAIR(c, tb_class[t]);
          }
        } else {
          for (int t = 0; t < ntb; ++t) {
            const int na = zg_asg_list(p, cs, kd->kc, kd->kind_c, kd->rel_c,
                                       tb_class[t], NULL, asg_c, (int)cap_ac);
            for (int i = 0; i < na; ++i) {
              if (!reach[asg_c[i].cls]) {
                reach[asg_c[i].cls] = 1;
                reach_list[nreach++] = asg_c[i].cls;
              }
            }
          }
          for (int c = 0; c < ncr; ++c) {
            okr[c] = 0;
            for (int i = 0; i < nreach && !okr[c]; ++i) okr[c] = ZG_PAIR(c, reach_list[i]);
          }
        }
        const int nar = zg_asg_list(p, rs, kd->kr, kd->kind_r, kd->rel_r,
                                    sr->bucket_class[bt], okr, asg_r, (int)cap_ar);
        ZG_TICK(0);
        /* The row operators. */
        int64_t nent = 0, nrun = 0, maxx = 0;
        for (int r = 0; r < m && nar > 0; ++r) {
          const int64_t rsid = sr->bucket_sid[sr->bucket_start[bt] + r0 + r];
          row_run[r] = nrun;
          int n = 0, erow;
          if (p->product) {
            erow = sr->tuple_elec[sr->bucket_class[bt]];
            n = zg_prod_ops(p, rs, kd->pat_r, rsid, xs, xsid, xsign, 0);
          } else {
            erow = zg_side_masks(p, rs, rsid, sm);
            zg_string_ranks(p, rs, sm, rk);
            for (int i = 0; i < nar; ++i) {
              n = zg_asg_enum(p, rs, kd->kr, kd->kind_r, kd->rel_r, &asg_r[i], sm, rk,
                              filter ? kd : NULL, xs, xsid, xsign, n);
            }
          }
          const int flip = kd->par_r && ((erow + kd->add_r) & 1);
          ZG_TICK(9);
          cnt[0] += 1;
          cnt[1] += nar;
          cnt[2] += n;
          ++stamp;
          int nl = 0;
          const int64_t runs_most = nrun + (n < sr->nbucket ? n : sr->nbucket) + 2;
          if (runs_most > rcap) {
            rcap = 2 * runs_most + 64;
            run_bucket = realloc(run_bucket, (size_t)rcap * sizeof(int32_t));
            run_start = realloc(run_start, (size_t)rcap * sizeof(int64_t));
          }
          for (int e = 0; e < n; ++e) {
            const uint64_t where = bpr[xsid[e]];
            const int b = (int)(where >> 32);
            xsid[e] = (int64_t)(where & 0xffffffffu); /* now the place */
            if (rstamp[b] != stamp) {
              rstamp[b] = stamp;
              rlocal[b] = nl;
              lfill[nl] = 0;
              lbucket[nl++] = b;
            }
            xrun[e] = rlocal[b];
            lfill[rlocal[b]]++;
          }
          if (nent + n + 1 > ecap) {
            ecap = 2 * (nent + n + 1) + 64;
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
            const int64_t q = lfill[xrun[e]]++;
            ent_idx[q] = xs[e] * kd->wr;
            ent_pos[q] = (int32_t)xsid[e];
            ent_sign[q] = (int8_t)(flip ? -xsign[e] : xsign[e]);
          }
          nrun += nl;
          nent += n;
          if (n > maxx) maxx = n;
          ZG_TICK(10);
        }
        if (nent == 0) {
          for (int i = 0; i < nreach; ++i) reach[reach_list[i]] = 0;
          continue;
        }
        row_run[m] = nrun;
        run_start[nrun] = nent;
        if (rdm) {
          const zg_term *tm_ = &p->terms[kd->term];
          if (!gl[kd->term]) {
            gl[kd->term] = calloc((size_t)(tm_->nx * tm_->ny) + 1, sizeof(ZG_SCALAR));
          }
          gk = gl[kd->term];
        }

        if (kd->kc == 0) {
          /* Whole rows. */
          for (int r = 0; r < m; ++r) {
            for (int64_t run = row_run[r]; run < row_run[r + 1]; ++run) {
              const int bs = run_bucket[run];
              for (int t = 0; t < ntb; ++t) {
                const int ct = tb_class[t], bb = tb_bucket[t];
                if (match[(size_t)bs * ncc + ct] != bb) continue;
                const int64_t nb = sc->bucket_size[bb];
                const int nbi = (int)nb;
                const ZG_SCALAR *src = cv + off[(size_t)bs * ncc + ct];
                if (rdm) {
                  const ZG_SCALAR *bra = bv + off[(size_t)bt * ncc + ct] + (r0 + r) * nb;
                  for (int64_t q = run_start[run]; q < run_start[run + 1]; ++q) {
                    const ZG_SCALAR *from = src + ent_pos[q] * nb;
                    ZG_SCALAR dot = 0;
                    for (int64_t j = 0; j < nb; ++j) dot += ZG_CONJ(bra[j]) * from[j];
                    gk[ent_idx[q]] += ent_sign[q] < 0 ? -dot : dot;
                  }
                  st_axpy += (double)nb * (run_start[run + 1] - run_start[run]);
                  continue;
                }
                ZG_SCALAR *dst = sv + off[(size_t)bt * ncc + ct] + (r0 + r) * nb;
                for (int64_t q = run_start[run]; q < run_start[run + 1]; ++q) {
                  ZG_SCALAR f = wk[ent_idx[q]];
                  if (f == 0) continue;
                  if (ent_sign[q] < 0) f = -f;
                  const ZG_SCALAR *from = src + ent_pos[q] * nb;
                  if (q + 4 < run_start[run + 1]) {
                    __builtin_prefetch(src + ent_pos[q + 4] * nb);
                  }
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

        /* The source column buckets that have a block with a source row
         * bucket, and the (row, run) of each. */
        ++cmark;
        int ncand = 0;
        int64_t nlr = 0;
        for (int r = 0; r < m; ++r) {
          for (int64_t run = row_run[r]; run < row_run[r + 1]; ++run) {
            const int bs = run_bucket[run];
            const int64_t most = nlr + pair_ptr[bs + 1] - pair_ptr[bs];
            if (most > lcap) {
              lcap = 2 * most + 64;
              lr_lb = realloc(lr_lb, (size_t)lcap * sizeof(int32_t));
              lr_row = realloc(lr_row, (size_t)lcap * sizeof(int32_t));
              lr_runid = realloc(lr_runid, (size_t)lcap * sizeof(int64_t));
              sl_row = realloc(sl_row, (size_t)lcap * sizeof(int32_t));
              sl_runid = realloc(sl_runid, (size_t)lcap * sizeof(int64_t));
            }
            for (int64_t i = pair_ptr[bs]; i < pair_ptr[bs + 1]; ++i) {
              const int cc = pair_cls[i];
              if (!reach[cc]) continue;
              const int bb = match[(size_t)bs * ncc + cc];
              if (cstamp[bb] != cmark) {
                cstamp[bb] = cmark;
                clocal[bb] = ncand;
                cand[ncand++] = bb;
                okc[cc] = 1;
              }
              lr_lb[nlr] = clocal[bb];
              lr_row[nlr] = r;
              lr_runid[nlr++] = run;
            }
          }
        }
        for (int i = 0; i < nreach; ++i) reach[reach_list[i]] = 0;
        /* Sorted by candidate (rows stay in order); runs of source columns
         * of each candidate numbered through (keys). */
        for (int i = 0; i <= ncand; ++i) lr_start[i] = 0;
        for (int64_t i = 0; i < nlr; ++i) lr_start[lr_lb[i] + 1]++;
        for (int i = 0; i < ncand; ++i) lr_start[i + 1] += lr_start[i];
        for (int64_t i = 0; i < nlr; ++i) {
          const int64_t q = lr_start[lr_lb[i]]++;
          sl_row[q] = lr_row[i];
          sl_runid[q] = lr_runid[i];
        }
        for (int i = ncand; i > 0; --i) lr_start[i] = lr_start[i - 1];
        lr_start[0] = 0;
        koff[0] = 0;
        for (int i = 0; i < ncand; ++i) {
          koff[i + 1] = koff[i] + (sc->bucket_size[cand[i]] + cols - 1) / cols;
        }
        const int64_t nkey = koff[ncand];
        if (nkey + 2 > kcap) {
          kcap = 2 * nkey + 64;
          kstart = realloc(kstart, (size_t)kcap * sizeof(int64_t));
        }
        ZG_BLAS_GROW(dmat, dcap, maxx * cols);
        if (maxx > xwcap) {
          xwcap = 2 * maxx + 64;
          xw = realloc(xw, (size_t)xwcap * sizeof(int64_t));
          rowptr = realloc(rowptr, (size_t)xwcap * sizeof(ZG_SCALAR *));
          rowsign = realloc(rowsign, (size_t)xwcap);
        }
        ZG_TICK(2);

        /* The column operators, from the target columns of the piece; they
         * are worked through whenever flush_at have been found. */
        int tnow = 0;
        int64_t jnow = 0;
        while (tnow < ntb && ncand > 0) {
          int64_t ne = 0;
          for (; tnow < ntb && ne < flush_at; ++tnow, jnow = 0) {
            const int bb = tb_bucket[tnow];
            const int nac = zg_asg_list(p, cs, kd->kc, kd->kind_c, kd->rel_c,
                                        tb_class[tnow], okc, asg_c, (int)cap_ac);
            if (nac == 0) continue;
            const int64_t ncol = sc->bucket_size[bb];
            for (; jnow < ncol && ne < flush_at; ++jnow) {
              const int64_t csid = sc->bucket_sid[sc->bucket_start[bb] + jnow];
              int n = 0, ecol;
              if (p->product) {
                ecol = sc->tuple_elec[tb_class[tnow]];
                n = zg_prod_ops(p, cs, kd->pat_c, csid, xs, xsid, xsign, 0);
              } else {
                ecol = zg_side_masks(p, cs, csid, sm);
                zg_string_ranks(p, cs, sm, rk);
                for (int i = 0; i < nac; ++i) {
                  n = zg_asg_enum(p, cs, kd->kc, kd->kind_c, kd->rel_c, &asg_c[i], sm,
                                  rk, NULL, xs, xsid, xsign, n);
                }
              }
              const int flip = kd->par_c && ((ecol + kd->add_c) & 1);
              cnt[3] += n;
              if (ne + n > tcap) {
                tcap = 2 * (ne + n) + 256;
                u_idx = realloc(u_idx, (size_t)tcap * sizeof(int64_t));
                u_key = realloc(u_key, (size_t)tcap * sizeof(int64_t));
                u_pos = realloc(u_pos, (size_t)tcap * sizeof(int32_t));
                u_tb = realloc(u_tb, (size_t)tcap * sizeof(int32_t));
                u_tcol = realloc(u_tcol, (size_t)tcap * sizeof(int32_t));
                u_sign = realloc(u_sign, (size_t)tcap);
                t_y = realloc(t_y, (size_t)tcap * sizeof(int32_t));
                t_col = realloc(t_col, (size_t)tcap * sizeof(int32_t));
                t_tb = realloc(t_tb, (size_t)tcap * sizeof(int32_t));
                t_tcol = realloc(t_tcol, (size_t)tcap * sizeof(int32_t));
                t_sign = realloc(t_sign, (size_t)tcap);
              }
              for (int e = 0; e < n; ++e) {
                const uint64_t where = bpc[xsid[e]];
                const int sb_ = (int)(where >> 32);
                if (cstamp[sb_] != cmark) continue;
                const int64_t pos = (int64_t)(where & 0xffffffffu);
                u_idx[ne] = xs[e];
                u_key[ne] = koff[clocal[sb_]] + pos / cols;
                u_pos[ne] = (int32_t)(pos % cols);
                u_tb[ne] = tnow;
                u_tcol[ne] = (int32_t)jnow;
                u_sign[ne] = (int8_t)(flip ? -xsign[e] : xsign[e]);
                ++ne;
              }
            }
            if (jnow < ncol) break;
          }
          if (ne == 0) continue;
          /* Sorted by run of source columns. */
          for (int64_t i = 0; i <= nkey; ++i) kstart[i] = 0;
          for (int64_t e = 0; e < ne; ++e) kstart[u_key[e] + 1]++;
          for (int64_t i = 0; i < nkey; ++i) kstart[i + 1] += kstart[i];
          for (int64_t e = 0; e < ne; ++e) {
            const int64_t q = kstart[u_key[e]]++;
            t_y[q] = (int32_t)e; /* for now: where it came from */
          }
          for (int64_t i = nkey; i > 0; --i) kstart[i] = kstart[i - 1];
          kstart[0] = 0;
          for (int64_t q = 0; q < ne; ++q) {
            const int64_t e = t_y[q];
            t_col[q] = u_pos[e];
            t_tb[q] = u_tb[e];
            t_tcol[q] = u_tcol[e];
            t_sign[q] = u_sign[e];
            u_key[q] = u_idx[e]; /* u_key is free now: the index in W */
          }
          ZG_TICK(3);
          for (int lb = 0; lb < ncand; ++lb) {
            const int bbs = cand[lb], cls = sc->bucket_class[bbs];
            const int64_t nbs = sc->bucket_size[bbs];
            for (int64_t key = koff[lb]; key < koff[lb + 1]; ++key) {
              const int64_t q0 = kstart[key], nq = kstart[key + 1] - kstart[key];
              if (nq == 0) continue;
              const int64_t j0 = (key - koff[lb]) * cols;
              const int64_t nc = j0 + cols < nbs ? cols : nbs - j0;
              int nyc = 0;
              for (int64_t q = q0; q < q0 + nq; ++q) {
                const int64_t idx = u_key[q];
                if (ymap[idx] < 0) {
                  ymap[idx] = nyc;
                  ylist[nyc++] = idx * kd->wc;
                }
                t_y[q] = ymap[idx];
              }
              for (int64_t q = q0; q < q0 + nq; ++q) ymap[u_key[q]] = -1;
              ZG_BLAS_GROW(wmat, wcap, maxx * nyc);
              ZG_BLAS_GROW(fmat, fcap, (int64_t)nyc * nc);
              ZG_TICK(3);
#define ZG_TARGET(q) sv[tb_off[t_tb[q]] + rr * tb_ncol[t_tb[q]] + t_tcol[q]]
              int64_t i = lr_start[lb];
              while (i < lr_start[lb + 1]) {
                /* D: the source rows of this row that have a block with the
                 * bucket. */
                const int r = sl_row[i];
                const int64_t rr = r0 + r;
                const int64_t i0 = i;
                int64_t nx = 0;
                for (; i < lr_start[lb + 1] && sl_row[i] == r; ++i) {
                  nx += run_start[sl_runid[i] + 1] - run_start[sl_runid[i]];
                }
                const int direct = rdm || !ZG_HAVE_BLAS ||
                                   (double)nyc * nc > direct_ratio * (double)nq ||
                                   (double)nx * nyc * nc < direct_small;
                nx = 0;
                for (int64_t ii = i0; ii < i; ++ii) {
                  const int64_t run = sl_runid[ii];
                  const ZG_SCALAR *src =
                      cv + off[(size_t)run_bucket[run] * ncc + cls] + j0;
                  for (int64_t q = run_start[run]; q < run_start[run + 1]; ++q) {
                    rowptr[nx] = src + ent_pos[q] * nbs;
                    __builtin_prefetch(rowptr[nx]);
                    if (nc > 8) __builtin_prefetch(rowptr[nx] + 8);
                    rowsign[nx] = ent_sign[q];
                    xw[nx++] = ent_idx[q];
                  }
                }
                ZG_TICK(4);
                if (rdm) {
                  /* G[x][y] += conj(bra[target]) s_y s_x C[source]. */
                  for (int64_t q = q0; q < q0 + nq; ++q) {
                    const ZG_SCALAR b = t_sign[q] < 0 ? -ZG_CONJ(bv[tb_off[t_tb[q]] + rr * tb_ncol[t_tb[q]] + t_tcol[q]])
                                                      : ZG_CONJ(bv[tb_off[t_tb[q]] + rr * tb_ncol[t_tb[q]] + t_tcol[q]]);
                    const int64_t col = t_col[q], yi = ylist[t_y[q]];
                    for (int64_t x = 0; x < nx; ++x) {
                      const ZG_SCALAR f = b * rowptr[x][col];
                      if (rowsign[x] > 0) {
                        gk[xw[x] + yi] += f;
                      } else {
                        gk[xw[x] + yi] -= f;
                      }
                    }
                  }
                  st_direct += (double)nq * nx;
                  ZG_TICK(7);
                } else if (direct) {
                  /* Only the (y, column) that are needed: W[y][x] with the
                   * signs of x, as rows over x; C read in place. */
                  for (int64_t x = 0; x < nx; ++x) {
                    const ZG_SCALAR *from = wk + xw[x];
                    if (rowsign[x] > 0) {
                      for (int y = 0; y < nyc; ++y) wmat[(int64_t)y * nx + x] = from[ylist[y]];
                    } else {
                      for (int y = 0; y < nyc; ++y) wmat[(int64_t)y * nx + x] = -from[ylist[y]];
                    }
                  }
                  ZG_TICK(5);
                  for (int64_t q = q0; q < q0 + nq; ++q) {
                    const ZG_SCALAR *wy = wmat + (int64_t)t_y[q] * nx;
                    const int64_t col = t_col[q];
                    ZG_SCALAR sum = 0;
                    for (int64_t x = 0; x < nx; ++x) sum += wy[x] * rowptr[x][col];
                    if (t_sign[q] < 0) sum = -sum;
                    ZG_TARGET(q) += sum;
                  }
                  st_direct += (double)nq * nx;
                  ZG_TICK(7);
                } else {
                  /* W[y][x] of these x and y, as a matrix (nyc, nx). */
                  for (int64_t x = 0; x < nx; ++x) {
                    const ZG_SCALAR *from = wk + xw[x];
                    ZG_SCALAR *to = wmat + x * nyc;
                    for (int y = 0; y < nyc; ++y) to[y] = from[ylist[y]];
                  }
                  ZG_TICK(5);
                  for (int64_t x = 0; x < nx; ++x) {
                    const ZG_SCALAR *from = rowptr[x];
                    ZG_SCALAR *to = dmat + x * nc;
                    if (rowsign[x] > 0) {
                      for (int64_t j = 0; j < nc; ++j) to[j] = from[j];
                    } else {
                      for (int64_t j = 0; j < nc; ++j) to[j] = -from[j];
                    }
                  }
                  ZG_TICK(4);
                  const int mi = nyc, ni = (int)nc, ki = (int)nx;
                  ZG_GEMM("N", "T", &mi, &ni, &ki, &sone, wmat, &mi, dmat, &ni,
                          &szero, fmat, &mi);
                  ZG_TICK(6);
                  for (int64_t q = q0; q < q0 + nq; ++q) {
                    const ZG_SCALAR f = fmat[t_y[q] + (int64_t)t_col[q] * nyc];
                    if (t_sign[q] > 0) {
                      ZG_TARGET(q) += f;
                    } else {
                      ZG_TARGET(q) -= f;
                    }
                  }
                  st_gemm += (double)nx * nyc * nc;
                  ZG_TICK(8);
                }
#undef ZG_TARGET
                st_scatter += (double)nq;
              }
            }
          }
        }
        for (int i = 0; i < ncand; ++i) okc[sc->bucket_class[cand[i]]] = 0;
      }
    }
    free(asg_r);
    free(asg_c);
    free(okr);
    free(reach);
    free(reach_list);
    free(okc);
    free(xs);
    free(xsid);
    free(xsign);
    free(xrun);
    free(rstamp);
    free(rlocal);
    free(lfill);
    free(lbucket);
    free(ent_idx);
    free(ent_pos);
    free(ent_sign);
    free(run_bucket);
    free(run_start);
    free(tb_class);
    free(tb_bucket);
    free(tb_off);
    free(tb_ncol);
    free(cstamp);
    free(clocal);
    free(cand);
    free(lr_start);
    free(koff);
    free(lr_lb);
    free(lr_row);
    free(sl_row);
    free(lr_runid);
    free(sl_runid);
    free(ymap);
    free(ylist);
    free(u_idx);
    free(u_key);
    free(u_pos);
    free(u_tb);
    free(u_tcol);
    free(u_sign);
    free(t_y);
    free(t_col);
    free(t_tb);
    free(t_tcol);
    free(t_sign);
    free(kstart);
    free(dmat);
    free(wmat);
    free(fmat);
    free(xw);
    free(rowptr);
    free(rowsign);
    for (int k = 0; k < 9; ++k) {
      if (!gl[k]) continue;
      const int64_t size = p->terms[k].nx * p->terms[k].ny;
#pragma omp critical(zg_rdm2_blas_reduce)
      for (int64_t i = 0; i < size; ++i) gacc[k][i] += gl[k][i];
      free(gl[k]);
    }
    zg_blas_restore(blas_before);
  }
  stat[0] += st_gemm;
  stat[1] += st_direct;
  stat[2] += st_axpy;
  stat[3] += st_scatter;
  if (prof) {
    fprintf(stderr,
            "zgasci blas half %d, thread seconds: placements %.2f, row operators "
            "%.2f, their sorting %.2f, axpy %.2f, candidates %.2f, column operators "
            "%.2f, gather D %.2f, gather W %.2f, products %.2f, dot products %.2f, "
            "scatter %.2f\n",
            h, tm[0], tm[9], tm[10], tm[1], tm[2], tm[3], tm[4], tm[5], tm[6], tm[7],
            tm[8]);
    fprintf(stderr,
            "zgasci blas half %d: %.3e rows x kinds, %.3e placements, %.3e row "
            "operators, %.3e column operators found\n",
            h, cnt[0], cnt[1], cnt[2], cnt[3]);
  }
#undef ZG_TICK
#undef ZG_PAIR
}

/* The kinds without column operators ((AA,AA) of half 0, (BB,BB) of half 1)
 * of a product plan: sigma[r'][c] += sum_r H(r', r) C[r][c], H the matrix
 * of the kind between the strings of the row side (summed over the
 * operators that connect the same two strings).  Tiles of at most
 * ZG_ROWS_R rows of a row bucket x ZG_ROWS_C columns of one of their
 * blocks, each written by one thread.  For each source bucket the entries
 * of the tile's rows are either put into a dense slab H(rows, source
 * strings) and multiplied by the source rows (GEMM), when the slab has at
 * least one entry in ZGASCI_ROWS_DENSE (default 16) of its elements, or
 * applied as one AXPY per distinct (row, source row). */
#define ZG_ROWS_R 128
#define ZG_ROWS_C 256
#define ZG_ROWS_SLAB (1 << 21)
static void ZG_NAME(blas_rows)(const zg_plan *p, int h, const zg_bkind *kd,
                               const ZG_SCALAR *wk, const ZG_SCALAR *cv,
                               ZG_SCALAR *sv, double *stat) {
  const int rs = h, cs = 1 - h;
  const zg_side *sr = &p->s[rs], *sc = &p->s[cs];
  const int ncc = sc->nclass;
  const int32_t *match = h ? p->match_t : p->match;
  const int64_t *off = h ? p->blockoff_t : p->blockoff;
  const uint64_t *bpr = p->bp[rs];
  double dense_ratio = 16.0;
  const char *text = getenv("ZGASCI_ROWS_DENSE");
  if (text) dense_ratio = atof(text);
  if (!ZG_HAVE_BLAS) dense_ratio = 0.0;
  /* The tiles: (row bucket, first row, block in the pair list, first
   * column). */
  int64_t ntile = 0, cap = 256;
  int64_t(*tile)[4] = malloc((size_t)cap * sizeof(*tile));
  int64_t maxb = 1;
  for (int bt = 0; bt < sr->nbucket; ++bt) {
    if (sr->bucket_size[bt] > maxb) maxb = sr->bucket_size[bt];
    for (int64_t r0 = 0; r0 < sr->bucket_size[bt]; r0 += ZG_ROWS_R) {
      for (int64_t i = p->pair_ptr[h][bt]; i < p->pair_ptr[h][bt + 1]; ++i) {
        const int bb = match[(size_t)bt * ncc + p->pair_cls[h][i]];
        for (int64_t j0 = 0; j0 < sc->bucket_size[bb]; j0 += ZG_ROWS_C) {
          if (ntile == cap) {
            cap *= 2;
            tile = realloc(tile, (size_t)cap * sizeof(*tile));
          }
          tile[ntile][0] = bt;
          tile[ntile][1] = r0;
          tile[ntile][2] = i;
          tile[ntile][3] = j0;
          ++ntile;
        }
      }
    }
  }
  int64_t capx = 1;
  for (int i = 0; i < kd->kr; ++i) capx *= p->nside[rs];
  capx = capx + 1;
  double st_gemm = 0, st_axpy = 0, t_gemm = 0, t_axpy = 0, t_ops = 0;
  const int prof = getenv("ZGASCI_STATS") != NULL;
#pragma omp parallel reduction(+ : st_gemm, st_axpy, t_gemm, t_axpy, t_ops)
  {
    const int blas_before = zg_blas_single();
    const int one = 1;
    const ZG_SCALAR sone = 1;
    int64_t *xs = malloc((size_t)capx * sizeof(int64_t));
    int64_t *xsid = malloc((size_t)capx * sizeof(int64_t));
    int8_t *xsign = malloc((size_t)capx);
    int64_t ecap = 0;
    int32_t *e_row = NULL, *e_pos = NULL, *e_bucket = NULL;
    ZG_SCALAR *e_val = NULL;
    int32_t *e_order = NULL;
    int32_t *bstamp = calloc((size_t)sr->nbucket + 1, sizeof(int32_t));
    int32_t *blocal = malloc((size_t)(sr->nbucket + 1) * sizeof(int32_t));
    int64_t *bcount = malloc((size_t)(sr->nbucket + 2) * sizeof(int64_t));
    int32_t *blist = malloc((size_t)(sr->nbucket + 1) * sizeof(int32_t));
    int32_t stamp = 0;
    int32_t *pmark = malloc((size_t)maxb * sizeof(int32_t));
    for (int64_t i = 0; i < maxb; ++i) pmark[i] = -1;
    int32_t *dpos = NULL;
    ZG_SCALAR *dval = NULL;
    int64_t dcap = 0;
    ZG_SCALAR *slab = NULL;
    int64_t slab_cap = 0;
#pragma omp for schedule(dynamic, 1)
    for (int64_t it = 0; it < ntile; ++it) {
      const int bt = (int)tile[it][0];
      const int64_t r0 = tile[it][1], j0 = tile[it][3];
      const int ct = p->pair_cls[h][tile[it][2]];
      const int bb = match[(size_t)bt * ncc + ct];
      const int64_t nb = sc->bucket_size[bb];
      const int64_t w = nb - j0 < ZG_ROWS_C ? nb - j0 : ZG_ROWS_C;
      const int m = (int)(sr->bucket_size[bt] - r0 < ZG_ROWS_R ? sr->bucket_size[bt] - r0
                                                                 : ZG_ROWS_R);
      ZG_SCALAR *dst = sv + off[(size_t)bt * ncc + ct] + r0 * nb + j0;
      double last = prof ? omp_get_wtime() : 0.0;
      /* The entries of the rows whose source bucket has a block with the
       * same column bucket: (row, source place, value), by source bucket. */
      ++stamp;
      int nl = 0;
      int64_t ne = 0;
      for (int r = 0; r < m; ++r) {
        const int64_t rsid = sr->bucket_sid[sr->bucket_start[bt] + r0 + r];
        const int n = zg_prod_ops(p, rs, kd->pat_r, rsid, xs, xsid, xsign, 0);
        if (ne + n > ecap) {
          ecap = 2 * (ne + n) + 256;
          e_row = realloc(e_row, (size_t)ecap * sizeof(int32_t));
          e_pos = realloc(e_pos, (size_t)ecap * sizeof(int32_t));
          e_bucket = realloc(e_bucket, (size_t)ecap * sizeof(int32_t));
          e_val = realloc(e_val, (size_t)ecap * sizeof(ZG_SCALAR));
          e_order = realloc(e_order, (size_t)ecap * sizeof(int32_t));
        }
        for (int e = 0; e < n; ++e) {
          const ZG_SCALAR f = wk[xs[e] * kd->wr];
          if (f == 0) continue;
          const uint64_t where = bpr[xsid[e]];
          const int bs = (int)(where >> 32);
          if (match[(size_t)bs * ncc + ct] != bb) continue;
          if (bstamp[bs] != stamp) {
            bstamp[bs] = stamp;
            blocal[bs] = nl;
            blist[nl++] = bs;
          }
          e_row[ne] = r;
          e_pos[ne] = (int32_t)(where & 0xffffffffu);
          e_bucket[ne] = blocal[bs];
          e_val[ne] = xsign[e] > 0 ? f : -f;
          ++ne;
        }
      }
      /* By source bucket, rows in order. */
      for (int l = 0; l <= nl; ++l) bcount[l] = 0;
      for (int64_t e = 0; e < ne; ++e) bcount[e_bucket[e] + 1]++;
      for (int l = 0; l < nl; ++l) bcount[l + 1] += bcount[l];
      for (int64_t e = 0; e < ne; ++e) e_order[bcount[e_bucket[e]]++] = (int32_t)e;
      for (int l = nl; l > 0; --l) bcount[l] = bcount[l - 1];
      bcount[0] = 0;
      if (prof) {
        const double now = omp_get_wtime();
        t_ops += now - last;
        last = now;
      }
      for (int l = 0; l < nl; ++l) {
        const int bs = blist[l];
        const int64_t nbs = sr->bucket_size[bs];
        const int64_t e0 = bcount[l], e1 = bcount[l + 1];
        const ZG_SCALAR *src = cv + off[(size_t)bs * ncc + ct] + j0;
        const double elems = (double)m * nbs;
        if (elems <= dense_ratio * (double)(e1 - e0) && elems <= ZG_ROWS_SLAB) {
          ZG_BLAS_GROW(slab, slab_cap, m * nbs);
          memset(slab, 0, (size_t)(m * nbs) * sizeof(ZG_SCALAR));
          for (int64_t q = e0; q < e1; ++q) {
            const int32_t e = e_order[q];
            slab[(int64_t)e_row[e] * nbs + e_pos[e]] += e_val[e];
          }
          /* Column-major: dst^T (w x m) += src^T (w x nbs) slab^T (nbs x m). */
          const int wi = (int)w, mi = m, ki = (int)nbs, lc = (int)nb;
          ZG_GEMM("N", "N", &wi, &mi, &ki, &sone, src, &lc, slab, &ki, &sone, dst, &lc);
          (void)lc;
          st_gemm += (double)m * nbs * w;
          if (prof) {
            const double now = omp_get_wtime();
            t_gemm += now - last;
            last = now;
          }
          continue;
        }
        /* Distinct source rows of each row, then one AXPY each. */
        if (e1 - e0 > dcap) {
          dcap = 2 * (e1 - e0) + 64;
          dpos = realloc(dpos, (size_t)dcap * sizeof(int32_t));
          dval = realloc(dval, (size_t)dcap * sizeof(ZG_SCALAR));
        }
        int64_t q = e0;
        while (q < e1) {
          const int r = e_row[e_order[q]];
          int nd = 0;
          for (; q < e1 && e_row[e_order[q]] == r; ++q) {
            const int32_t e = e_order[q];
            if (pmark[e_pos[e]] < 0) {
              pmark[e_pos[e]] = nd;
              dpos[nd] = e_pos[e];
              dval[nd++] = e_val[e];
            } else {
              dval[pmark[e_pos[e]]] += e_val[e];
            }
          }
          ZG_SCALAR *to = dst + (int64_t)r * nb;
          const int wi = (int)w;
          for (int d = 0; d < nd; ++d) {
            pmark[dpos[d]] = -1;
            const ZG_SCALAR f = dval[d];
            const ZG_SCALAR *from = src + (int64_t)dpos[d] * nb;
            if (w < 16) {
              for (int64_t j = 0; j < w; ++j) to[j] += f * from[j];
            } else {
              ZG_AXPY(&wi, &f, from, &one, to, &one);
            }
          }
          st_axpy += (double)nd * w;
        }
        if (prof) {
          const double now = omp_get_wtime();
          t_axpy += now - last;
          last = now;
        }
      }
    }
    free(xs);
    free(xsid);
    free(xsign);
    free(e_row);
    free(e_pos);
    free(e_bucket);
    free(e_val);
    free(e_order);
    free(bstamp);
    free(blocal);
    free(bcount);
    free(blist);
    free(pmark);
    free(dpos);
    free(dval);
    free(slab);
    zg_blas_restore(blas_before);
  }
  free(tile);
  stat[0] += st_gemm;
  stat[2] += st_axpy;
  if (prof) {
    fprintf(stderr,
            "zgasci blas rows half %d: %lld tiles; thread seconds: operators %.2f, slabs "
            "and products %.2f (%.3e multiply-adds), axpy %.2f (%.3e)\n",
            h, (long long)ntile, t_ops, t_gemm, st_gemm, t_axpy, st_axpy);
  }
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
  const double t_begin = omp_get_wtime();
  for (int k = 0; k < 9; ++k) {
    w[k] = ZG_NAME(term_matrix)(p, &p->terms[k], v, &live[k]);
    live[k] = live[k] && (mask >> k & 1);
  }
  /* The matrices of the kinds of half 1 with the B index first. */
  for (int i = 0; i < p->nbkind[1]; ++i) {
    const zg_term *t = &p->terms[p->bkinds[1][i].term];
    const int64_t nx = t->nx, ny = t->ny;
    if (nx == 1 || ny == 1) continue;
    ZG_SCALAR *old = w[p->bkinds[1][i].term];
    ZG_SCALAR *new = malloc((size_t)(nx * ny + 1) * sizeof(ZG_SCALAR));
#pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < ny; ++y) {
      for (int64_t x = 0; x < nx; ++x) new[y * nx + x] = old[x * ny + y];
    }
    free(old);
    w[p->bkinds[1][i].term] = new;
  }
  /* ZGASCI_BLAS_COLS: source columns of a product.  A product is done as
   * dot products when it would do more than ZGASCI_BLAS_DIRECT times their
   * multiply-adds, or fewer than ZGASCI_BLAS_SMALL multiply-adds. */
  int64_t cols = 256;
  /* Product plans: 16, from measurements (README). */
  double direct_ratio = p->product ? 16.0 : ZG_BLAS_DIRECT, direct_small = 512.0;
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
    ZG_NAME(blas_half)(p, 1, w, live, ct, st, cols, direct_ratio, direct_small, stat,
                       NULL, NULL);
    for (int k = 0; p->product && k < p->nbkind[1]; ++k) {
      const zg_bkind *kd = &p->bkinds[1][k];
      if (kd->kc == 0 && live[kd->term]) ZG_NAME(blas_rows)(p, 1, kd, w[kd->term], ct, st, stat);
    }
    ZG_NAME(blas_transpose)(p, st, sigma, 1);
    free(ct);
    free(st);
    t_half1 = omp_get_wtime();
  } else {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < p->dim; ++i) sigma[i] = 0;
  }
  ZG_NAME(blas_half)(p, 0, w, live, c, sigma, cols, direct_ratio, direct_small, stat,
                     NULL, NULL);
  for (int k = 0; p->product && k < p->nbkind[0]; ++k) {
    const zg_bkind *kd = &p->bkinds[0][k];
    if (kd->kc == 0 && live[kd->term]) ZG_NAME(blas_rows)(p, 0, kd, w[kd->term], c, sigma, stat);
  }
  if (getenv("ZGASCI_STATS")) {
    fprintf(stderr,
            "zgasci blas: multiply-adds %.3e in products, %.3e in dot products, "
            "%.3e in axpy; %.3e scattered; %.2f s for the matrices of the kinds, "
            "%.2f s for the half of the transposed vector, %.2f s for the other\n",
            stat[0], stat[1], stat[2], stat[3], t_start - t_begin, t_half1 - t_start,
            omp_get_wtime() - t_half1);
  }
  for (int k = 0; k < 9; ++k) free(w[k]);
}

int ZG_NAME(zg_sigma_flip)(const zg_plan *p, const ZG_SCALAR *v, const ZG_SCALAR *c,
                           ZG_SCALAR *sigma, int parity) {
  if (!p->product || !p->mirror || (parity != 1 && parity != -1)) return 0;
  ZG_SCALAR *w[9];
  int live[9];
  for (int k = 0; k < 9; ++k) w[k] = ZG_NAME(term_matrix)(p, &p->terms[k], v, &live[k]);
  /* Only (AA,AA) = 0, (AB,AB) = 4, (BB,BB) = 8, and W of (BB,BB) that of
   * (AA,AA). */
  int good = p->terms[0].nx == p->terms[8].ny;
  for (int k = 0; k < 9 && good; ++k) good = !live[k] || k == 0 || k == 4 || k == 8;
  for (int64_t i = 0; good && i < p->terms[0].nx; ++i) good = w[0][i] == w[8][i];
  if (good) {
    double stat[4] = {0, 0, 0, 0};
    const double t0 = omp_get_wtime();
    ZG_SCALAR *x = malloc(((size_t)p->dim + 1) * sizeof(ZG_SCALAR));
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < p->dim; ++i) x[i] = 0;
    for (int k = 0; k < p->nbkind[0]; ++k) {
      const zg_bkind *kd = &p->bkinds[0][k];
      if (kd->kc == 0 && live[kd->term]) ZG_NAME(blas_rows)(p, 0, kd, w[kd->term], c, x, stat);
    }
    /* The transposed layout of a mirror plan is the native one of X^T. */
    ZG_NAME(blas_transpose)(p, x, sigma, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < p->dim; ++i) sigma[i] = x[i] + (parity > 0 ? sigma[i] : -sigma[i]);
    free(x);
    const double t1 = omp_get_wtime();
    int only[9] = {0, 0, 0, 0, live[4], 0, 0, 0, 0};
    int64_t cols = 256;
    double direct_ratio = 16.0, direct_small = 512.0;
    const char *text = getenv("ZGASCI_BLAS_COLS");
    if (text && atol(text) > 0) cols = atol(text);
    text = getenv("ZGASCI_BLAS_DIRECT");
    if (text) direct_ratio = atof(text);
    text = getenv("ZGASCI_BLAS_SMALL");
    if (text) direct_small = atof(text);
    if (only[4]) {
      ZG_NAME(blas_half)(p, 0, w, only, c, sigma, cols, direct_ratio, direct_small, stat,
                         NULL, NULL);
    }
    if (getenv("ZGASCI_STATS")) {
      fprintf(stderr, "zgasci flip: %.2f s for X + X^T, %.2f s for (AB,AB)\n", t1 - t0,
              omp_get_wtime() - t1);
    }
  }
  for (int k = 0; k < 9; ++k) free(w[k]);
  return good;
}

/* g[P,R] = <bra| A+_P A_R |ket> by the loops of blas_half in density mode:
 * G[x][y] = <bra| O_A(x) O_B(y) (with the sign of the kind) |ket> for each
 * kind, then g[P,R] = sign * G[x][y] as term_matrix maps them. */
void ZG_NAME(zg_rdm2_blas)(const zg_plan *p, const ZG_SCALAR *bra,
                           const ZG_SCALAR *ket, ZG_SCALAR *out) {
  const size_t npair = (size_t)p->npair;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < npair * npair; ++i) out[i] = 0;
  if (!p->block_ok) {
    ZG_NAME(zg_rdm2)(p, bra, ket, out);
    return;
  }
  ZG_SCALAR *g[9];
  int live[9];
  for (int k = 0; k < 9; ++k) {
    g[k] = calloc((size_t)(p->terms[k].nx * p->terms[k].ny) + 1, sizeof(ZG_SCALAR));
    live[k] = 1;
  }
  double stat[4] = {0, 0, 0, 0};
  {
    ZG_SCALAR *kt = malloc(((size_t)p->dim + 1) * sizeof(ZG_SCALAR));
    ZG_SCALAR *bt = malloc(((size_t)p->dim + 1) * sizeof(ZG_SCALAR));
    ZG_NAME(blas_transpose)(p, ket, kt, 0);
    ZG_NAME(blas_transpose)(p, bra, bt, 0);
    ZG_NAME(blas_half)(p, 1, g, live, kt, NULL, 256, 0.0, 0.0, stat, bt, g);
    free(kt);
    free(bt);
  }
  ZG_NAME(blas_half)(p, 0, g, live, ket, NULL, 256, 0.0, 0.0, stat, bra, g);
  /* The kinds of half 1 back to the A index first. */
  for (int i = 0; i < p->nbkind[1]; ++i) {
    const zg_term *t = &p->terms[p->bkinds[1][i].term];
    const int64_t nx = t->nx, ny = t->ny;
    if (nx == 1 || ny == 1) continue;
    ZG_SCALAR *old = g[p->bkinds[1][i].term];
    ZG_SCALAR *new = malloc((size_t)(nx * ny + 1) * sizeof(ZG_SCALAR));
#pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < nx; ++x) {
      for (int64_t y = 0; y < ny; ++y) new[x * ny + y] = old[y * nx + x];
    }
    free(old);
    g[p->bkinds[1][i].term] = new;
  }
  for (int k = 0; k < 9; ++k) {
    const zg_term *t = &p->terms[k];
    const int64_t nx = t->nx, ny = t->ny;
    const ZG_SCALAR *gk = g[k];
#pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < nx; ++x) {
      int idx[4];
      int64_t rest = x;
      for (int i = 0; i < t->ka; ++i) {
        idx[t->role_a[i]] = (int)(rest % p->nside[0]);
        rest /= p->nside[0];
      }
      for (int64_t y = 0; y < ny; ++y) {
        int64_t yr = y;
        for (int i = 0; i < t->kb; ++i) {
          idx[t->role_b[i]] = p->nside[0] + (int)(yr % p->nside[1]);
          yr /= p->nside[1];
        }
        const int pp = idx[0], qq = idx[1], ss = idx[2], rr = idx[3];
        if (pp >= qq || rr >= ss) continue;
        const size_t P = (size_t)qq * (qq - 1) / 2 + pp;
        const size_t R = (size_t)ss * (ss - 1) / 2 + rr;
        out[P * npair + R] = t->sign > 0 ? gk[x * ny + y] : -gk[x * ny + y];
      }
    }
  }
  for (int k = 0; k < 9; ++k) free(g[k]);
}
