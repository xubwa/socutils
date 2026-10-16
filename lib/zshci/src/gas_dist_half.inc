/* Derived from blas_half and term_matrix of src/zgasci (zgasci_blas.inc,
 * zgasci_block.inc at zgasci 85686dc), generated with few changes, each
 * marked "dist:", so that the kernel can be re-derived when zgasci changes;
 * compiled twice by gas_dist.c. */

#ifndef ZG_BLAS_GROW
#define ZG_BLAS_GROW(ptr, cap, need)                         \
  do {                                                       \
    if ((int64_t)(need) > (cap)) {                           \
      (cap) = 2 * (int64_t)(need) + 64;                      \
      (ptr) = realloc((ptr), (size_t)(cap) * sizeof(*(ptr))); \
    }                                                        \
  } while (0)
#endif

/* W[x][y] = sign * v[P,R] of a kind of term, and whether it is all zero. */
static ZG_SCALAR *ZG_NAME(term_matrix)(const zg_plan *p, const zg_term *t,
                                       const ZG_SCALAR *v, int *nonzero) {
  const int64_t nx = t->nx, ny = t->ny;
  const size_t npair = (size_t)p->npair;
  ZG_SCALAR *w = calloc((size_t)(nx * ny) + 1, sizeof(ZG_SCALAR));
  int any = 0;
#pragma omp parallel for reduction(|| : any) schedule(static)
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
      const ZG_SCALAR value = v[P * npair + R];
      if (value != 0) {
        w[x * ny + y] = t->sign > 0 ? value : -value;
        any = 1;
      }
    }
  }
  *nonzero = any;
  return w;
}

/* blas_half of zgasci_blas.inc for the target rows of the row buckets t0 ..
 * t1 - 1 and the source rows of the row buckets s0 .. s1 - 1 only: sv holds
 * the elements of the (native or transposed) vector from sbase on, cv those
 * from cbase on.  Changes against blas_half are marked "dist:". */
static void ZG_NAME(dist_half)(const zg_plan *p, int h, ZG_SCALAR *const *w,
                               const int *live, const ZG_SCALAR *cv,
                               ZG_SCALAR *sv, int64_t cols, double direct_ratio,
                               double direct_small, double *stat, int t0, int t1,
                               int s0, int s1, int64_t cbase, int64_t sbase,
                               double *bucket_seconds, void (*progress)(void)) {
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
  /* dist: the pieces of the target buckets, and the row classes with a
   * bucket among the sources. */
  int64_t it0 = 0, it1 = p->nrtask[h];
  while (it0 < it1 && p->rtasks[h][it0].bucket < t0) ++it0;
  while (it1 > it0 && p->rtasks[h][it1 - 1].bucket >= t1) --it1;
  uint8_t *in_slice = malloc((size_t)ncr + 1);
  for (int c = 0; c < ncr; ++c) {
    in_slice[c] = sr->class_bucket[c] < s1 && sr->class_bucket[c + 1] > s0;
  }
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

#pragma omp for schedule(dynamic, 1)
    for (int64_t it = it0; it < it1; ++it) {
      /* dist: the seconds of each target bucket, and a call that lets
       * one-sided communication go on. */
      const double task_start = bucket_seconds ? omp_get_wtime() : 0.0;
      if (progress && omp_get_thread_num() == 0) progress();
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
        const ZG_SCALAR *wk = w[kd->term];
        /* The classes of source rows that can matter, the placements. */
        int nreach = 0;
        if (kd->kc == 0) {
          for (int c = 0; c < ncr; ++c) {
            okr[c] = 0;
            for (int t = 0; t < ntb && !okr[c] && in_slice[c]; ++t) okr[c] = ZG_PAIR(c, tb_class[t]);
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
            for (int i = 0; i < nreach && !okr[c] && in_slice[c]; ++i) okr[c] = ZG_PAIR(c, reach_list[i]);
          }
        }
        const int nar = zg_asg_list(p, rs, kd->kr, kd->kind_r, kd->rel_r,
                                    sr->bucket_class[bt], okr, asg_r, (int)cap_ar);
        ZG_TICK(0);
        /* The row operators. */
        int64_t nent = 0, nrun = 0, maxx = 0;
        for (int r = 0; r < m && nar > 0; ++r) {
          const int erow =
              zg_side_masks(p, rs, sr->bucket_sid[sr->bucket_start[bt] + r0 + r], sm);
          zg_string_ranks(p, rs, sm, rk);
          row_run[r] = nrun;
          int n = 0;
          for (int i = 0; i < nar; ++i) {
            n = zg_asg_enum(p, rs, kd->kr, kd->kind_r, kd->rel_r, &asg_r[i], sm, rk,
                            filter ? kd : NULL, xs, xsid, xsign, n);
          }
          const int flip = kd->par_r && ((erow + kd->add_r) & 1);
          /* dist: only the source rows of the slice. */
          if (s0 > 0 || s1 < sr->nbucket) {
            int kept = 0;
            for (int e = 0; e < n; ++e) {
              const int b = (int)(bpr[xsid[e]] >> 32);
              if (b < s0 || b >= s1) continue;
              xs[kept] = xs[e];
              xsid[kept] = xsid[e];
              xsign[kept++] = xsign[e];
            }
            n = kept;
          }
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
                ZG_SCALAR *dst = sv + (off[(size_t)bt * ncc + ct] - sbase) + (r0 + r) * nb;
                const ZG_SCALAR *src = cv + (off[(size_t)bs * ncc + ct] - cbase);
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
              const int ecol =
                  zg_side_masks(p, cs, sc->bucket_sid[sc->bucket_start[bb] + jnow], sm);
              zg_string_ranks(p, cs, sm, rk);
              int n = 0;
              for (int i = 0; i < nac; ++i) {
                n = zg_asg_enum(p, cs, kd->kc, kd->kind_c, kd->rel_c, &asg_c[i], sm,
                                rk, NULL, xs, xsid, xsign, n);
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
#define ZG_TARGET(q) sv[tb_off[t_tb[q]] - sbase + rr * tb_ncol[t_tb[q]] + t_tcol[q]]
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
                const int direct = !ZG_HAVE_BLAS ||
                                   (double)nyc * nc > direct_ratio * (double)nq ||
                                   (double)nx * nyc * nc < direct_small;
                nx = 0;
                for (int64_t ii = i0; ii < i; ++ii) {
                  const int64_t run = sl_runid[ii];
                  const ZG_SCALAR *src =
                      cv + (off[(size_t)run_bucket[run] * ncc + cls] - cbase) + j0;
                  for (int64_t q = run_start[run]; q < run_start[run + 1]; ++q) {
                    rowptr[nx] = src + ent_pos[q] * nbs;
                    __builtin_prefetch(rowptr[nx]);
                    if (nc > 8) __builtin_prefetch(rowptr[nx] + 8);
                    rowsign[nx] = ent_sign[q];
                    xw[nx++] = ent_idx[q];
                  }
                }
                ZG_TICK(4);
                if (direct) {
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
      if (bucket_seconds) {
        const double spent = omp_get_wtime() - task_start;
#pragma omp atomic
        bucket_seconds[bt] += spent;
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
    zg_blas_restore(blas_before);
  }
  free(in_slice);
  stat[0] += st_gemm;
  stat[1] += st_direct;
  stat[2] += st_axpy;
  stat[3] += st_scatter;
  if (prof) {
    fprintf(stderr,
            "zgasci dist half %d, thread seconds: placements %.2f, row operators "
            "%.2f, their sorting %.2f, axpy %.2f, candidates %.2f, column operators "
            "%.2f, gather D %.2f, gather W %.2f, products %.2f, dot products %.2f, "
            "scatter %.2f\n",
            h, tm[0], tm[9], tm[10], tm[1], tm[2], tm[3], tm[4], tm[5], tm[6], tm[7],
            tm[8]);
    fprintf(stderr,
            "zgasci dist half %d: %.3e rows x kinds, %.3e placements, %.3e row "
            "operators, %.3e column operators found\n",
            h, cnt[0], cnt[1], cnt[2], cnt[3]);
  }
#undef ZG_TICK
#undef ZG_PAIR
}
