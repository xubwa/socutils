/* The kernels, compiled once for each scalar type by zgasci.c, which
 * defines ZG_SCALAR, ZG_NAME(x), ZG_CONJ(x), ZG_REAL(x) and
 * ZG_ATOMIC_ADD(target, value).  These are the reference kernels: one loop
 * over the determinants K of N - 2 electrons (see zgasci.h). */

int zg_string_occ(const zg_plan *p, int side, int64_t sid, int *occ);

/* Product plans: H_KK = e_A(a) + e_B(b) + sum_{q in b} u_a[q], e the pairs
 * within a string, u_a[q] = sum_{p in a} v[pq,pq] over the A spinors p. */
static void ZG_NAME(hdiag_product)(const zg_plan *p, const ZG_SCALAR *v, double *out) {
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  const size_t npair = (size_t)p->npair;
  const int na_sp = p->nside[0], nb_sp = p->nside[1];
#define ZG_VDIAG(x, y) ZG_REAL(v[((size_t)(y) * ((y) - 1) / 2 + (x)) * (npair + 1)])
  double *ea = malloc((size_t)(sa->nsid + 1) * sizeof(double));
  double *eb = malloc((size_t)(sb->nsid + 1) * sizeof(double));
  double *u = malloc((size_t)(sa->nsid + 1) * nb_sp * sizeof(double));
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 0; i < sa->nsid; ++i) {
    int occ[ZG_MAX_SPINOR];
    const int n = zg_string_occ(p, 0, i, occ);
    double e = 0.0;
    for (int y = 1; y < n; ++y) {
      for (int x = 0; x < y; ++x) e += ZG_VDIAG(occ[x], occ[y]);
    }
    ea[i] = e;
    double *ui = u + (size_t)i * nb_sp;
    for (int q = 0; q < nb_sp; ++q) {
      double sum = 0.0;
      for (int x = 0; x < n; ++x) sum += ZG_VDIAG(occ[x], na_sp + q);
      ui[q] = sum;
    }
  }
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t j = 0; j < sb->nsid; ++j) {
    int occ[ZG_MAX_SPINOR];
    const int n = zg_string_occ(p, 1, j, occ);
    double e = 0.0;
    for (int y = 1; y < n; ++y) {
      for (int x = 0; x < y; ++x) e += ZG_VDIAG(occ[x], occ[y]);
    }
    eb[j] = e;
  }
#undef ZG_VDIAG
#pragma omp parallel for schedule(dynamic)
  for (int ba = 0; ba < sa->nbucket; ++ba) {
    int occ[ZG_MAX_SPINOR];
    for (int cb = 0; cb < sb->nclass; ++cb) {
      const size_t k = (size_t)ba * sb->nclass + cb;
      const int bb = p->match[k];
      if (bb < 0) continue;
      const int64_t nb = sb->bucket_size[bb], na = sa->bucket_size[ba];
      for (int64_t j = 0; j < nb; ++j) {
        const int64_t jb = sb->bucket_sid[sb->bucket_start[bb] + j];
        const int n = zg_string_occ(p, 1, jb, occ);
        for (int x = 0; x < n; ++x) occ[x] -= na_sp;
        for (int64_t i = 0; i < na; ++i) {
          const int64_t ia = sa->bucket_sid[sa->bucket_start[ba] + i];
          const double *ui = u + (size_t)ia * nb_sp;
          double e = ea[ia] + eb[jb];
          for (int x = 0; x < n; ++x) e += ui[occ[x]];
          out[p->blockoff[k] + i * nb + j] = e;
        }
      }
    }
  }
  free(ea);
  free(eb);
  free(u);
}

void ZG_NAME(zg_hdiag)(const zg_plan *p, const ZG_SCALAR *v, double *out) {
  if (p->product) {
    ZG_NAME(hdiag_product)(p, v, out);
    return;
  }
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  const size_t npair = (size_t)p->npair;
#pragma omp parallel for schedule(dynamic)
  for (int ba = 0; ba < sa->nbucket; ++ba) {
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
            const size_t base = (size_t)occ[y] * (occ[y] - 1) / 2;
            for (int x = 0; x < y; ++x) {
              const size_t pq = base + occ[x];
              e += ZG_REAL(v[pq * npair + pq]);
            }
          }
          out[p->blockoff[k] + i * nb + j] = e;
        }
      }
    }
  }
}

void ZG_NAME(zg_sigma)(const zg_plan *p, const ZG_SCALAR *v,
                       const ZG_SCALAR *c, ZG_SCALAR *sigma) {
  if (p->product) {
    ZG_NAME(zg_sigma_blas)(p, v, c, sigma);
    return;
  }
  const size_t npair = (size_t)p->npair;
  memset(sigma, 0, (size_t)p->dim * sizeof(ZG_SCALAR));
#pragma omp parallel
  {
    int32_t *pair = malloc((npair + 1) * sizeof(int32_t));
    int64_t *addr = malloc((npair + 1) * sizeof(int64_t));
    int8_t *sign = malloc(npair + 1);
    ZG_SCALAR *d = malloc((npair + 1) * sizeof(ZG_SCALAR));
    uint32_t masks[ZG_MAX_SUB];
#pragma omp for schedule(dynamic, 1)
    for (int64_t t = 0; t < p->ntask; ++t) {
      const zg_task *task = &p->tasks[t];
      const int64_t size_b = p->s[1].tuple_size[task->tb];
      for (int64_t a = task->a0; a < task->a1; ++a) {
        for (int64_t b = 0; b < size_b; ++b) {
          zg_task_masks(p, task, a, b, masks);
          const int n = zg_connect(p, masks, pair, addr, sign);
          for (int x = 0; x < n; ++x) {
            d[x] = sign[x] > 0 ? c[addr[x]] : -c[addr[x]];
          }
          for (int x = 0; x < n; ++x) {
            const ZG_SCALAR *row = v + (size_t)pair[x] * npair;
            ZG_SCALAR e = 0;
            for (int y = 0; y < n; ++y) e += row[pair[y]] * d[y];
            if (sign[x] < 0) e = -e;
            ZG_ATOMIC_ADD(sigma[addr[x]], e);
          }
        }
      }
    }
    free(pair);
    free(addr);
    free(sign);
    free(d);
  }
}

void ZG_NAME(zg_rdm2)(const zg_plan *p, const ZG_SCALAR *bra,
                      const ZG_SCALAR *ket, ZG_SCALAR *out) {
  if (p->product) {
    ZG_NAME(zg_rdm2_blas)(p, bra, ket, out);
    return;
  }
  const size_t npair = (size_t)p->npair;
  memset(out, 0, npair * npair * sizeof(ZG_SCALAR));
#pragma omp parallel
  {
    int32_t *pair = malloc((npair + 1) * sizeof(int32_t));
    int64_t *addr = malloc((npair + 1) * sizeof(int64_t));
    int8_t *sign = malloc(npair + 1);
    ZG_SCALAR *db = malloc((npair + 1) * sizeof(ZG_SCALAR));
    ZG_SCALAR *dk = malloc((npair + 1) * sizeof(ZG_SCALAR));
    ZG_SCALAR *mine = calloc(npair * npair + 1, sizeof(ZG_SCALAR));
    uint32_t masks[ZG_MAX_SUB];
#pragma omp for schedule(dynamic, 1)
    for (int64_t t = 0; t < p->ntask; ++t) {
      const zg_task *task = &p->tasks[t];
      const int64_t size_b = p->s[1].tuple_size[task->tb];
      for (int64_t a = task->a0; a < task->a1; ++a) {
        for (int64_t b = 0; b < size_b; ++b) {
          zg_task_masks(p, task, a, b, masks);
          const int n = zg_connect(p, masks, pair, addr, sign);
          for (int x = 0; x < n; ++x) {
            db[x] = ZG_CONJ(bra[addr[x]]);
            dk[x] = ket[addr[x]];
            if (sign[x] < 0) {
              db[x] = -db[x];
              dk[x] = -dk[x];
            }
          }
          for (int x = 0; x < n; ++x) {
            ZG_SCALAR *row = mine + (size_t)pair[x] * npair;
            for (int y = 0; y < n; ++y) row[pair[y]] += db[x] * dk[y];
          }
        }
      }
    }
#pragma omp critical(zg_rdm2_reduce)
    for (size_t x = 0; x < npair * npair; ++x) out[x] += mine[x];
    free(pair);
    free(addr);
    free(sign);
    free(db);
    free(dk);
    free(mine);
  }
}
