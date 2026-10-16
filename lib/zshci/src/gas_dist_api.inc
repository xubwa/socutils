/* The functions of gas_dist.h for one scalar type (gas_dist.c). */

zgd_matrices *ZG_NAME(zgd_matrices_new)(const zg_plan *p, const ZG_SCALAR *v) {
  zgd_matrices *m = matrices_alloc();
  const char *only = getenv("ZGASCI_TERMS");
  const int mask = only ? atoi(only) : 511;
  for (int k = 0; k < 9; ++k) {
    m->w[k] = ZG_NAME(term_matrix)(p, &p->terms[k], v, &m->live[k]);
    m->live[k] = m->live[k] && (mask >> k & 1);
  }
  /* The matrices of the kinds of half 1 with the B index first, as
   * zg_sigma_blas has them. */
  for (int i = 0; i < p->nbkind[1]; ++i) {
    const zg_term *t = &p->terms[p->bkinds[1][i].term];
    const int64_t nx = t->nx, ny = t->ny;
    if (nx == 1 || ny == 1) continue;
    ZG_SCALAR *old = m->w[p->bkinds[1][i].term];
    ZG_SCALAR *new = malloc((size_t)(nx * ny + 1) * sizeof(ZG_SCALAR));
#pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < ny; ++y) {
      for (int64_t x = 0; x < nx; ++x) new[y * nx + x] = old[x * ny + y];
    }
    free(old);
    m->w[p->bkinds[1][i].term] = new;
  }
  return m;
}

void ZG_NAME(zgd_half)(const zg_plan *p, zgd_matrices *m, int h, int t0,
                       int t1, int s0, int s1, const ZG_SCALAR *cv,
                       int64_t cbase, ZG_SCALAR *sv, int64_t sbase) {
  ZG_NAME(zgd_half_timed)(p, m, h, t0, t1, s0, s1, cv, cbase, sv, sbase, NULL,
                          NULL);
}

void ZG_NAME(zgd_half_timed)(const zg_plan *p, zgd_matrices *m, int h, int t0,
                             int t1, int s0, int s1, const ZG_SCALAR *cv,
                             int64_t cbase, ZG_SCALAR *sv, int64_t sbase,
                             double *bucket_seconds, void (*progress)(void)) {
  int any = 0;
  for (int k = 0; k < p->nbkind[h]; ++k) any = any || m->live[p->bkinds[h][k].term];
  if (!any || t0 >= t1 || s0 >= s1) return;
  ZG_NAME(dist_half)(p, h, (ZG_SCALAR *const *)m->w, m->live, cv, sv, m->cols,
                     m->direct_ratio, m->direct_small, m->stat, t0, t1, s0, s1,
                     cbase, sbase, bucket_seconds, progress);
}

void ZG_NAME(zgd_hdiag)(const zg_plan *p, const ZG_SCALAR *v, int b0, int b1,
                        double *out, int64_t base) {
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  const size_t npair = (size_t)p->npair;
#pragma omp parallel for schedule(dynamic)
  for (int ba = b0; ba < b1; ++ba) {
    int occ[ZG_MAX_SPINOR];
    for (int cb = 0; cb < sb->nclass; ++cb) {
      const size_t k = (size_t)ba * sb->nclass + cb;
      const int bb = p->match[k];
      if (bb < 0) continue;
      const int64_t nb = sb->bucket_size[bb];
      for (int64_t i = 0; i < sa->bucket_size[ba]; ++i) {
        const int na =
            zg_string_occ(p, 0, sa->bucket_sid[sa->bucket_start[ba] + i], occ);
        for (int64_t j = 0; j < nb; ++j) {
          const int n = na + zg_string_occ(
                                 p, 1, sb->bucket_sid[sb->bucket_start[bb] + j],
                                 occ + na);
          double e = 0.0;
          for (int y = 1; y < n; ++y) {
            const size_t pq0 = (size_t)occ[y] * (occ[y] - 1) / 2;
            for (int x = 0; x < y; ++x) {
              const size_t pq = pq0 + occ[x];
              e += ZG_REAL(v[pq * npair + pq]);
            }
          }
          out[p->blockoff[k] - base + i * nb + j] = e;
        }
      }
    }
  }
}
