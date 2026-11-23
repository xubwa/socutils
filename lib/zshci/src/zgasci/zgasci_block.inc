/* The block kernel, compiled once for each scalar type by zgasci.c (see
 * zgasci_kernel.inc for the macros).  sigma is built row by row of its
 * blocks: each row of the target is written by the one thread that owns its
 * piece of work, no atomic additions.  The nine kinds of terms (zg_term) are
 * each a sum over x, y of W[x][y] O_A(x) (x) O_B(y); x and the source rows
 * are found from the target row, y and the source columns from the target
 * column, both by acting on the sub-strings of the strings. */

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


void ZG_NAME(zg_sigma_block)(const zg_plan *p, const ZG_SCALAR *v,
                             const ZG_SCALAR *c, ZG_SCALAR *sigma) {
  if (p->product) {
    ZG_NAME(zg_sigma_blas)(p, v, c, sigma);
    return;
  }
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  const int ncb = sb->nclass;
  ZG_SCALAR *w[9];
  int live[9];
  /* ZGASCI_TERMS (a bit mask of the nine kinds) restricts the terms, for
   * timings and tests; ZGASCI_STATS prints the multiply-adds of each. */
  const char *only = getenv("ZGASCI_TERMS");
  const int mask = only ? atoi(only) : 511;
  for (int k = 0; k < 9; ++k) {
    w[k] = ZG_NAME(term_matrix)(p, &p->terms[k], v, &live[k]);
    live[k] = live[k] && (mask >> k & 1);
  }
  /* Bounds of the operator lists of a row and of a column. */
  int capx = 1, capy = 1;
  for (int k = 0; k < 9; ++k) {
    const zg_term *t = &p->terms[k];
    int64_t bx = t->nx, by = t->ny;
    for (int i = 0; i + 1 < t->ka; ++i) bx /= t->rel_a[i] ? 2 : 1;
    for (int i = 0; i + 1 < t->kb; ++i) by /= t->rel_b[i] ? 2 : 1;
    if (bx + 1 > capx) capx = (int)bx + 1;
    if (by + 1 > capy) capy = (int)by + 1;
  }
  int64_t maxb = 1;
  for (int b = 0; b < sb->nbucket; ++b) {
    if (sb->bucket_size[b] > maxb) maxb = sb->bucket_size[b];
  }
  /* Electrons of a source A string minus those of the target. */
  int add_a[9];
  for (int k = 0; k < 9; ++k) {
    add_a[k] = 0;
    for (int i = 0; i < p->terms[k].ka; ++i) add_a[k] += p->terms[k].kind_a[i] ? -1 : 1;
  }
  const int stats = getenv("ZGASCI_STATS") != NULL;
  double macs[9] = {0};
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < p->dim; ++i) sigma[i] = 0;
#pragma omp parallel reduction(+ : macs[:9])
  {
    int64_t *xs = malloc((size_t)capx * sizeof(int64_t));
    int64_t *xsrc = malloc((size_t)capx * sizeof(int64_t));
    int8_t *xsign = malloc((size_t)capx);
    int64_t *ysrc = malloc((size_t)capy * sizeof(int64_t));
    uint32_t sm[ZG_MAX_SUB];
    int64_t *yx = malloc((size_t)capy * sizeof(int64_t));
    int8_t *ys = malloc((size_t)capy);
    ZG_SCALAR *zero = calloc((size_t)maxb + 1, sizeof(ZG_SCALAR));
    /* Column entries of a tile for each kind of term: y, the source bucket
     * (as a number in the tile's list), its place; per column first the
     * positive, then the negative ones. */
    int64_t ecap = 0;
    int32_t *ey = NULL, *ekey = NULL, *epos = NULL;
    int64_t *estart = malloc((size_t)9 * (2 * ZG_MAX_COLS + 1) * sizeof(int64_t));
    int32_t *ukey = malloc((size_t)(sb->nbucket + 1) * sizeof(int32_t));
    int32_t *ulist = malloc((size_t)(sb->nbucket + 1) * sizeof(int32_t));
    const ZG_SCALAR **uptr = malloc((size_t)(sb->nbucket + 1) * sizeof(ZG_SCALAR *));
    for (int b = 0; b < sb->nbucket; ++b) ukey[b] = -1;
#pragma omp for schedule(dynamic, 1)
    for (int64_t it = 0; it < p->nbtask; ++it) {
      const zg_btask *task = &p->btasks[it];
      const int ba_t = (int)(task->k / ncb), cb_t = (int)(task->k % ncb);
      const int bb_t = p->match[task->k];
      const int64_t ncol = sb->bucket_size[bb_t];
      const int64_t j0 = task->j0, nj = task->j1 - task->j0;
      ZG_SCALAR *const target = sigma + p->blockoff[task->k];
      /* The columns: entries of every kind of term with B operators. */
      int nu = 0;
      int64_t ne = 0;
      for (int k = 0; k < 9; ++k) {
        const zg_term *t = &p->terms[k];
        int64_t *st = estart + (size_t)k * (2 * ZG_MAX_COLS + 1);
        if (!live[k] || t->kb == 0) continue;
        for (int64_t j = 0; j < nj; ++j) {
          zg_side_masks(p, 1, sb->bucket_sid[sb->bucket_start[bb_t] + j0 + j], sm);
          const int n = zg_enum_ops(p, 1, t->kb, t->kind_b, t->rel_b, sm, yx, ysrc, ys,
                                    capy);
          if (ne + n > ecap) {
            ecap = 2 * (ne + n);
            ey = realloc(ey, ecap * sizeof(int32_t));
            ekey = realloc(ekey, ecap * sizeof(int32_t));
            epos = realloc(epos, ecap * sizeof(int32_t));
          }
          for (int pass = 0; pass < 2; ++pass) {
            st[2 * j + pass] = ne;
            for (int e = 0; e < n; ++e) {
              if ((ys[e] < 0) != pass) continue;
              const int64_t sid = ysrc[e];
              const int bb = sb->bucket[sid];
              if (ukey[bb] < 0) {
                ukey[bb] = nu;
                ulist[nu++] = bb;
              }
              ey[ne] = (int32_t)yx[e];
              ekey[ne] = ukey[bb];
              epos[ne] = (int32_t)sb->pos[sid];
              ++ne;
            }
          }
        }
        st[2 * nj] = ne;
      }
      for (int u = 0; u < nu; ++u) ukey[ulist[u]] = -1;
      /* The rows. */
      for (int64_t r = task->r0; r < task->r1; ++r) {
        const int ea = zg_side_masks(p, 0, sa->bucket_sid[sa->bucket_start[ba_t] + r], sm);
        ZG_SCALAR *row = target + r * ncol + j0;
        for (int k = 0; k < 9; ++k) {
          const zg_term *t = &p->terms[k];
          if (!live[k]) continue;
          const int64_t *st = estart + (size_t)k * (2 * ZG_MAX_COLS + 1);
          const int n = zg_enum_ops(p, 0, t->ka, t->kind_a, t->rel_a, sm, xs, xsrc,
                                    xsign, capx);
          for (int e = 0; e < n; ++e) {
            const int64_t sid = xsrc[e];
            const int ba = sa->bucket[sid];
            const int64_t pa = sa->pos[sid];
            int sg = xsign[e];
            if ((t->kb & 1) && ((ea + add_a[k]) & 1)) sg = -sg;
            if (t->kb == 0) {
              const size_t kk = (size_t)ba * ncb + cb_t;
              if (p->match[kk] != bb_t) continue;
              const ZG_SCALAR *src = c + p->blockoff[kk] + pa * ncol + j0;
              ZG_SCALAR f = w[k][xs[e]];
              if (f == 0) continue;
              if (sg < 0) f = -f;
              for (int64_t j = 0; j < nj; ++j) row[j] += f * src[j];
              if (stats) macs[k] += nj;
              continue;
            }
            /* Where the source row lies in each source bucket. */
            for (int u = 0; u < nu; ++u) {
              const int bb = ulist[u];
              const size_t kk = (size_t)ba * ncb + sb->bucket_class[bb];
              uptr[u] = p->match[kk] == bb
                            ? c + p->blockoff[kk] + pa * sb->bucket_size[bb]
                            : zero;
            }
            const ZG_SCALAR *wx = w[k] + xs[e] * t->ny;
            if (stats) macs[k] += st[2 * nj] - st[0];
            for (int64_t j = 0; j < nj; ++j) {
              ZG_SCALAR plus = 0, minus = 0;
              const int64_t e0 = st[2 * j], e1 = st[2 * j + 1], e2 = st[2 * j + 2];
              for (int64_t q = e0; q < e1; ++q) plus += wx[ey[q]] * uptr[ekey[q]][epos[q]];
              for (int64_t q = e1; q < e2; ++q) minus += wx[ey[q]] * uptr[ekey[q]][epos[q]];
              row[j] += sg > 0 ? plus - minus : minus - plus;
            }
          }
        }
      }
    }
    free(xs);
    free(xsrc);
    free(xsign);
    free(ysrc);
    free(yx);
    free(ys);
    free(zero);
    free(ey);
    free(ekey);
    free(epos);
    free(estart);
    free(ukey);
    free(ulist);
    free(uptr);
  }
  if (stats) {
    for (int k = 0; k < 9; ++k) fprintf(stderr, "zgasci term %d: %.3e multiply-adds\n", k, macs[k]);
  }
  for (int k = 0; k < 9; ++k) free(w[k]);
}
