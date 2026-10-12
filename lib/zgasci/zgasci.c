/* The addressing scheme: sub-strings, classes, buckets, blocks, tasks; the
 * determinants K + R reached from a determinant K of N - 2 electrons.  The
 * kernels, real and complex, are zgasci_kernel.inc compiled twice. */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zgasci_plan.h"

static char zg_message[256] = "";
const char *zg_error(void) { return zg_message; }

static int64_t binom_table[ZG_MAX_SUBSIZE + 1][ZG_MAX_SUBSIZE + 1];
static void init_binom(void) {
  for (int n = 0; n <= ZG_MAX_SUBSIZE; ++n) {
    for (int k = 0; k <= ZG_MAX_SUBSIZE; ++k) {
      binom_table[n][k] = k > n                ? 0
                          : (k == 0 || k == n) ? 1
                                               : binom_table[n - 1][k - 1] +
                                                     binom_table[n - 1][k];
    }
  }
}
int64_t zg_binom(int n, int k) {
  return k < 0 || k > n ? 0 : binom_table[n][k];
}

/* Whether the electrons in the sub-groups (counts of side A, then of side
 * B) satisfy the constraints. */
static int gas_ok(const zg_plan *p, const int *counts_a, const int *counts_b) {
  for (int c = 0; c < p->ngas; ++c) {
    int sum = 0;
    for (int g = 0; g < p->nsub; ++g) {
      if (!(p->gas_set[c] >> g & 1)) continue;
      sum += g < p->s[0].nsub ? counts_a[g] : counts_b[g - p->s[0].nsub];
    }
    if (sum < p->gas_min[c] || sum > p->gas_max[c]) return 0;
  }
  return 1;
}

static int reduce_label(int value, int mod) {
  if (mod == 0) return value;
  value %= mod;
  return value < 0 ? value + mod : value;
}

/* The strings of every class of a side: their buckets and places. */
static int number_strings(zg_plan *p, int side) {
  zg_side *s = &p->s[side];
  const int nsets = p->nsets;
  s->class_base = malloc((s->nclass + 1) * sizeof(int64_t));
  s->class_bucket = malloc((s->nclass + 1) * sizeof(int));
  s->nsid = 0;
  for (int c = 0; c < s->nclass; ++c) {
    s->class_base[c] = s->nsid;
    s->nsid += s->tuple_size[s->class_tuple[c]];
  }
  s->class_base[s->nclass] = s->nsid;
  if (s->nsid >= INT32_MAX) {
    snprintf(zg_message, sizeof(zg_message),
             "side %c has %lld strings; at most 2^31 are supported",
             "AB"[side], (long long)s->nsid);
    return 0;
  }
  s->bucket = malloc((s->nsid + 1) * sizeof(int32_t));
  s->pos = malloc((s->nsid + 1) * sizeof(uint32_t));
  int capacity = 64;
  s->nbucket = 0;
  s->bucket_size = malloc(capacity * sizeof(int64_t));
  s->bucket_label = malloc(capacity * (nsets + 1) * sizeof(int));
  s->bucket_class = malloc(capacity * sizeof(int));
  for (int c = 0; c < s->nclass; ++c) {
    s->class_bucket[c] = s->nbucket;
    int counts[ZG_MAX_SUB];
    uint32_t masks[ZG_MAX_SUB];
    zg_counts(p, side, s->class_tuple[c], counts);
    const int64_t size = s->tuple_size[s->class_tuple[c]];
    int last = -1;
    for (int64_t address = 0; address < size; ++address) {
      zg_unpack(p, side, counts, address, masks);
      int label[ZG_MAX_SETS];
      for (int set = 0; set < nsets; ++set) {
        int sum = 0;
        for (int g = 0; g < s->nsub; ++g) {
          const int sub = s->first + g;
          sum += p->sublabel[sub][((size_t)set << p->m[sub]) + masks[g]];
        }
        label[set] = reduce_label(sum, p->mods[set]);
      }
      /* Its bucket among those of this class; often the last one used. */
      int found = -1;
      if (last >= 0 && memcmp(s->bucket_label + (size_t)last * nsets, label,
                              nsets * sizeof(int)) == 0) {
        found = last;
      }
      for (int b = s->class_bucket[c]; found < 0 && b < s->nbucket; ++b) {
        if (memcmp(s->bucket_label + (size_t)b * nsets, label,
                   nsets * sizeof(int)) == 0) {
          found = b;
        }
      }
      if (found < 0) {
        if (s->nbucket == capacity) {
          capacity *= 2;
          s->bucket_size = realloc(s->bucket_size, capacity * sizeof(int64_t));
          s->bucket_label =
              realloc(s->bucket_label, capacity * (nsets + 1) * sizeof(int));
          s->bucket_class = realloc(s->bucket_class, capacity * sizeof(int));
        }
        found = s->nbucket++;
        s->bucket_size[found] = 0;
        s->bucket_class[found] = c;
        memcpy(s->bucket_label + (size_t)found * nsets, label,
               nsets * sizeof(int));
      }
      last = found;
      const int64_t sid = s->class_base[c] + address;
      s->bucket[sid] = found;
      s->pos[sid] = (uint32_t)s->bucket_size[found]++;
    }
  }
  s->class_bucket[s->nclass] = s->nbucket;
  s->bucket_start = malloc((s->nbucket + 1) * sizeof(int64_t));
  s->bucket_sid = malloc((s->nsid + 1) * sizeof(int64_t));
  s->bucket_start[0] = 0;
  for (int b = 0; b < s->nbucket; ++b) {
    s->bucket_start[b + 1] = s->bucket_start[b] + s->bucket_size[b];
  }
  for (int64_t sid = 0; sid < s->nsid; ++sid) {
    s->bucket_sid[s->bucket_start[s->bucket[sid]] + s->pos[sid]] = sid;
  }
  return 1;
}

/* Whether some A bucket of class ca pairs with B class cb. */
static int classes_pair(const zg_plan *p, int ca, int cb) {
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  for (int ba = sa->class_bucket[ca]; ba < sa->class_bucket[ca + 1]; ++ba) {
    if (p->match[(size_t)ba * sb->nclass + cb] >= 0) return 1;
  }
  return 0;
}

zg_plan *zg_plan_new(int nspinor, int nelec, int nsub, int nsub_a,
                     const int *sub_size, int ngas, const int *gas_ptr,
                     const int *gas_sub, const int *gas_min,
                     const int *gas_max, int nsets, const int *labels,
                     const int *mods, const int *targets) {
  init_binom();
  zg_message[0] = '\0';
  int total = 0;
  for (int g = 0; g < nsub && g < ZG_MAX_SUB; ++g) total += sub_size[g];
  if (nsub < 2 || nsub > ZG_MAX_SUB || nsub_a < 1 || nsub_a >= nsub ||
      total != nspinor || nelec < 2 || nelec > nspinor || nsets < 0 ||
      nsets > ZG_MAX_SETS || ngas < 0) {
    snprintf(zg_message, sizeof(zg_message),
             "bad arguments: 2 to %d sub-groups that hold all spinors, at "
             "least one on each side, 2 <= electrons <= spinors, at most %d "
             "label sets",
             ZG_MAX_SUB, ZG_MAX_SETS);
    return NULL;
  }
  for (int g = 0; g < nsub; ++g) {
    if (sub_size[g] < 1 || sub_size[g] > ZG_MAX_SUBSIZE) {
      snprintf(zg_message, sizeof(zg_message),
               "a sub-group holds 1 to %d spinors", ZG_MAX_SUBSIZE);
      return NULL;
    }
  }
  for (int c = 0; c < ngas; ++c) {
    for (int k = gas_ptr[c]; k < gas_ptr[c + 1]; ++k) {
      if (gas_sub[k] < 0 || gas_sub[k] >= nsub) {
        snprintf(zg_message, sizeof(zg_message),
                 "constraint %d names sub-group %d of %d", c, gas_sub[k],
                 nsub);
        return NULL;
      }
    }
  }
  zg_plan *p = calloc(1, sizeof(zg_plan));
  p->n = nspinor;
  p->nelec = nelec;
  p->nsub = nsub;
  p->nsets = nsets;
  p->npair = nspinor * (nspinor - 1) / 2;
  for (int set = 0; set < nsets; ++set) {
    p->mods[set] = mods[set];
    p->targets[set] = reduce_label(targets[set], mods[set]);
  }
  p->ngas = ngas;
  p->gas_set = calloc(ngas + 1, sizeof(uint32_t));
  p->gas_min = malloc((ngas + 1) * sizeof(int));
  p->gas_max = malloc((ngas + 1) * sizeof(int));
  for (int c = 0; c < ngas; ++c) {
    for (int k = gas_ptr[c]; k < gas_ptr[c + 1]; ++k) {
      p->gas_set[c] |= 1u << gas_sub[k];
    }
    p->gas_min[c] = gas_min[c];
    p->gas_max[c] = gas_max[c];
  }

  /* Sub-strings. */
  for (int g = 0, first = 0; g < nsub; first += sub_size[g], ++g) {
    const int m = sub_size[g];
    p->m[g] = m;
    p->off[g] = first;
    p->side_of[g] = g >= nsub_a;
    p->rank[g] = malloc(sizeof(uint32_t) << m);
    p->masks[g] = malloc(sizeof(uint32_t) << m);
    p->sublabel[g] = malloc((sizeof(int32_t) * (nsets + 1)) << m);
    p->mstart[g][0] = 0;
    for (int k = 0; k <= m; ++k) {
      p->mstart[g][k + 1] = p->mstart[g][k] + zg_binom(m, k);
    }
    int64_t seen[ZG_MAX_SUBSIZE + 1] = {0};
    for (uint32_t mask = 0; mask < (1u << m); ++mask) {
      const int k = __builtin_popcount(mask);
      p->rank[g][mask] = (uint32_t)seen[k];
      p->masks[g][p->mstart[g][k] + seen[k]++] = mask;
      for (int set = 0; set < nsets; ++set) {
        int sum = 0;
        for (int j = 0; j < m; ++j) {
          if (mask >> j & 1) sum += labels[set * nspinor + first + j];
        }
        p->sublabel[g][((size_t)set << m) + mask] = sum;
      }
    }
  }

  /* Tuples of each side. */
  for (int side = 0; side < 2; ++side) {
    zg_side *s = &p->s[side];
    s->first = side == 0 ? 0 : nsub_a;
    s->nsub = side == 0 ? nsub_a : nsub - nsub_a;
    s->ntuple = 1;
    int nside = 0;
    for (int g = 0; g < s->nsub; ++g) {
      s->stride[g] = s->ntuple;
      s->ntuple *= p->m[s->first + g] + 1;
      nside += p->m[s->first + g];
      if (s->ntuple > 4000000) {
        snprintf(zg_message, sizeof(zg_message),
                 "too many ways to distribute electrons over the sub-groups "
                 "of side %c; use fewer, larger sub-groups",
                 "AB"[side]);
        zg_plan_free(p);
        return NULL;
      }
    }
    s->tuple_elec = malloc(s->ntuple * sizeof(int));
    s->tuple_size = malloc(s->ntuple * sizeof(int64_t));
    s->class_of = malloc(s->ntuple * sizeof(int32_t));
    s->by_elec = malloc(s->ntuple * sizeof(int64_t));
    s->elec_start = calloc(nside + 3, sizeof(int64_t));
    for (int64_t t = 0; t < s->ntuple; ++t) {
      int counts[ZG_MAX_SUB];
      zg_counts(p, side, t, counts);
      int elec = 0;
      double size = 1.0;
      for (int g = 0; g < s->nsub; ++g) {
        elec += counts[g];
        size *= (double)zg_binom(p->m[s->first + g], counts[g]);
      }
      s->tuple_elec[t] = elec;
      s->tuple_size[t] = size < 9e18 ? (int64_t)size : INT64_MAX;
      s->class_of[t] = -1;
      s->elec_start[elec + 2]++;
    }
    for (int k = 0; k <= nside; ++k) s->elec_start[k + 2] += s->elec_start[k + 1];
    for (int64_t t = 0; t < s->ntuple; ++t) {
      s->by_elec[s->elec_start[s->tuple_elec[t] + 1]++] = t;
    }
    /* elec_start[k] .. elec_start[k + 1] now hold the tuples of k. */
  }
  zg_side *sa = &p->s[0], *sb = &p->s[1];
  const int nside_b = p->n - p->off[nsub_a];
#define ZG_TUPLES_B(k, from, to)                    \
  int64_t from = 0, to = 0;                         \
  if ((k) >= 0 && (k) <= nside_b) {                 \
    from = sb->elec_start[k];                       \
    to = sb->elec_start[(k) + 1];                   \
  }

  /* Classes: the tuples that pair up with one of the other side. */
  for (int64_t ta = 0; ta < sa->ntuple; ++ta) {
    int counts_a[ZG_MAX_SUB], counts_b[ZG_MAX_SUB];
    zg_counts(p, 0, ta, counts_a);
    ZG_TUPLES_B(nelec - sa->tuple_elec[ta], from, to);
    for (int64_t i = from; i < to; ++i) {
      const int64_t tb = sb->by_elec[i];
      zg_counts(p, 1, tb, counts_b);
      if (!gas_ok(p, counts_a, counts_b)) continue;
      sa->class_of[ta] = 0;
      sb->class_of[tb] = 0;
    }
  }
  for (int side = 0; side < 2; ++side) {
    zg_side *s = &p->s[side];
    s->nclass = 0;
    for (int64_t t = 0; t < s->ntuple; ++t) {
      if (s->class_of[t] == 0) s->class_of[t] = s->nclass++;
    }
    s->class_tuple = malloc((s->nclass + 1) * sizeof(int64_t));
    for (int64_t t = 0; t < s->ntuple; ++t) {
      if (s->class_of[t] >= 0) s->class_tuple[s->class_of[t]] = t;
    }
    if (!number_strings(p, side)) {
      zg_plan_free(p);
      return NULL;
    }
  }

  /* Blocks. */
  const size_t nmatch = (size_t)sa->nbucket * sb->nclass;
  p->match = malloc((nmatch + 1) * sizeof(int32_t));
  p->blockoff = malloc((nmatch + 1) * sizeof(int64_t));
  p->dim = 0;
  p->nblock = 0;
  for (int ba = 0; ba < sa->nbucket; ++ba) {
    const int ca = sa->bucket_class[ba];
    int counts_a[ZG_MAX_SUB], counts_b[ZG_MAX_SUB];
    zg_counts(p, 0, sa->class_tuple[ca], counts_a);
    for (int cb = 0; cb < sb->nclass; ++cb) {
      const size_t k = (size_t)ba * sb->nclass + cb;
      p->match[k] = -1;
      p->blockoff[k] = 0;
      if (sa->tuple_elec[sa->class_tuple[ca]] +
              sb->tuple_elec[sb->class_tuple[cb]] !=
          nelec) {
        continue;
      }
      zg_counts(p, 1, sb->class_tuple[cb], counts_b);
      if (!gas_ok(p, counts_a, counts_b)) continue;
      for (int bb = sb->class_bucket[cb]; bb < sb->class_bucket[cb + 1]; ++bb) {
        int good = 1;
        for (int set = 0; set < nsets; ++set) {
          const int sum = sa->bucket_label[(size_t)ba * nsets + set] +
                          sb->bucket_label[(size_t)bb * nsets + set];
          good = good && reduce_label(sum, p->mods[set]) == p->targets[set];
        }
        if (!good) continue;
        p->match[k] = bb;
        p->blockoff[k] = p->dim;
        p->dim += sa->bucket_size[ba] * sb->bucket_size[bb];
        p->nblock++;
        break;
      }
    }
  }

  /* Tasks: the pairs of tuples of N - 2 electrons from which two electrons
   * can be added to reach the CI space, the larger ones in pieces. */
  int64_t capacity = 1024;
  p->tasks = malloc(capacity * sizeof(zg_task));
  p->ntask = 0;
  double work = 0.0;
  for (int pass = 0; pass < 2; ++pass) {
    /* Pass 0 adds up the work, pass 1 cuts it. */
    const double piece = work / (64.0 * omp_get_max_threads()) + 1.0;
    for (int64_t ta = 0; ta < sa->ntuple; ++ta) {
      ZG_TUPLES_B(nelec - 2 - sa->tuple_elec[ta], from, to);
      for (int64_t i = from; i < to; ++i) {
        const int64_t tb = sb->by_elec[i];
        int feasible = 0;
        for (int g = 0; g < nsub && !feasible; ++g) {
          for (int h = g; h < nsub && !feasible; ++h) {
            int64_t t[2] = {ta, tb};
            int ok = 1;
            for (int which = 0; which < 2 && ok; ++which) {
              const int sub = which == 0 ? g : h;
              const int side = p->side_of[sub];
              const int local = sub - p->s[side].first;
              const int now =
                  (int)((t[side] / p->s[side].stride[local]) % (p->m[sub] + 1));
              if (now + 1 > p->m[sub]) ok = 0;
              t[side] += p->s[side].stride[local];
            }
            if (!ok) continue;
            const int ca = sa->class_of[t[0]], cb = sb->class_of[t[1]];
            if (ca < 0 || cb < 0) continue;
            feasible = classes_pair(p, ca, cb);
          }
        }
        if (!feasible) continue;
        const double size_b = (double)sb->tuple_size[tb];
        if (pass == 0) {
          work += (double)sa->tuple_size[ta] * size_b;
          continue;
        }
        int64_t step = (int64_t)(piece / size_b);
        if (step < 1) step = 1;
        for (int64_t a0 = 0; a0 < sa->tuple_size[ta]; a0 += step) {
          if (p->ntask == capacity) {
            capacity *= 2;
            p->tasks = realloc(p->tasks, capacity * sizeof(zg_task));
          }
          const int64_t a1 =
              a0 + step < sa->tuple_size[ta] ? a0 + step : sa->tuple_size[ta];
          p->tasks[p->ntask++] = (zg_task){ta, tb, a0, a1};
        }
      }
    }
  }
#undef ZG_TUPLES_B
  return p;
}

void zg_plan_free(zg_plan *p) {
  if (p == NULL) return;
  for (int g = 0; g < ZG_MAX_SUB; ++g) {
    free(p->rank[g]);
    free(p->masks[g]);
    free(p->sublabel[g]);
  }
  free(p->gas_set);
  free(p->gas_min);
  free(p->gas_max);
  for (int side = 0; side < 2; ++side) {
    zg_side *s = &p->s[side];
    free(s->tuple_elec);
    free(s->tuple_size);
    free(s->class_of);
    free(s->by_elec);
    free(s->elec_start);
    free(s->class_tuple);
    free(s->class_base);
    free(s->bucket);
    free(s->pos);
    free(s->bucket_size);
    free(s->bucket_label);
    free(s->bucket_class);
    free(s->bucket_start);
    free(s->bucket_sid);
    free(s->class_bucket);
  }
  free(p->match);
  free(p->blockoff);
  free(p->tasks);
  free(p);
}

int64_t zg_dim(const zg_plan *p) { return p->dim; }
int zg_nwords(const zg_plan *p) { return (p->n + 63) / 64; }

void zg_info(const zg_plan *p, int64_t *out) {
  out[0] = p->s[0].nsid;
  out[1] = p->s[1].nsid;
  out[2] = p->s[0].nclass;
  out[3] = p->s[1].nclass;
  out[4] = p->s[0].nbucket;
  out[5] = p->s[1].nbucket;
  out[6] = p->nblock;
  out[7] = p->ntask;
}

/* The string sid of a side whose sub-strings are masks (with counts), or
 * -1 if its tuple is not a class. */
static inline int64_t side_sid(const zg_plan *p, int side, const int *counts,
                               const uint32_t *masks) {
  const zg_side *s = &p->s[side];
  int64_t t = 0;
  for (int g = 0; g < s->nsub; ++g) t += counts[g] * s->stride[g];
  const int c = s->class_of[t];
  if (c < 0) return -1;
  return s->class_base[c] + zg_address(p, side, counts, masks);
}

/* The place of the determinant of A string sa and B string sb in the CI
 * vector, or -1. */
static inline int64_t locate(const zg_plan *p, int64_t ia, int64_t ib) {
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  const int ba = sa->bucket[ia], bb = sb->bucket[ib];
  const size_t k = (size_t)ba * sb->nclass + sb->bucket_class[bb];
  if (p->match[k] != bb) return -1;
  return p->blockoff[k] + (int64_t)sa->pos[ia] * sb->bucket_size[bb] +
         sb->pos[ib];
}

int zg_connect(const zg_plan *p, const uint32_t *masks, int32_t *pair,
               int64_t *addr, int8_t *sign) {
  const int nsub = p->nsub, na = p->s[0].nsub;
  int counts[ZG_MAX_SUB];
  uint32_t work[ZG_MAX_SUB];
  for (int g = 0; g < nsub; ++g) {
    counts[g] = __builtin_popcount(masks[g]);
    work[g] = masks[g];
  }
  /* The unoccupied spinors, A ones first, and the parity of the number of
   * occupied ones below each. */
  int u[ZG_MAX_SPINOR], ug[ZG_MAX_SPINOR], nu = 0, nua = 0;
  uint8_t par[ZG_MAX_SPINOR];
  for (int g = 0; g < nsub; ++g) {
    for (int j = 0; j < p->m[g]; ++j) {
      if (masks[g] >> j & 1) continue;
      u[nu] = p->off[g] + j;
      ug[nu] = g;
      par[nu] = (uint8_t)((u[nu] - nu) & 1);
      ++nu;
    }
    if (g == na - 1) nua = nu;
  }
  /* Each side as it is, and with one electron added. */
  int64_t sid0[2];
  sid0[0] = side_sid(p, 0, counts, masks);
  sid0[1] = side_sid(p, 1, counts + na, masks + na);
  int64_t sid1[ZG_MAX_SPINOR];
  for (int i = 0; i < nu; ++i) {
    const int g = ug[i], side = g >= na, base = side ? na : 0;
    work[g] |= 1u << (u[i] - p->off[g]);
    counts[g]++;
    sid1[i] = side_sid(p, side, counts + base, work + base);
    work[g] = masks[g];
    counts[g]--;
  }
  int n = 0;
  for (int k = 1; k < nu; ++k) {
    const int gk = ug[k];
    const int32_t base_pair = u[k] * (u[k] - 1) / 2;
    for (int i = 0; i < k; ++i) {
      int64_t ia, ib;
      if (k < nua) {
        /* Both to A. */
        if (sid0[1] < 0) break;
        const int gi = ug[i];
        work[gi] |= 1u << (u[i] - p->off[gi]);
        work[gk] |= 1u << (u[k] - p->off[gk]);
        counts[gi]++;
        counts[gk]++;
        ia = side_sid(p, 0, counts, work);
        work[gi] = masks[gi];
        work[gk] = masks[gk];
        counts[gi]--;
        counts[gk]--;
        ib = sid0[1];
      } else if (i < nua) {
        ia = sid1[i];
        ib = sid1[k];
      } else {
        if (sid0[0] < 0) break;
        const int gi = ug[i];
        work[gi] |= 1u << (u[i] - p->off[gi]);
        work[gk] |= 1u << (u[k] - p->off[gk]);
        counts[gi]++;
        counts[gk]++;
        ib = side_sid(p, 1, counts + na, work + na);
        work[gi] = masks[gi];
        work[gk] = masks[gk];
        counts[gi]--;
        counts[gk]--;
        ia = sid0[0];
      }
      if (ia < 0 || ib < 0) continue;
      const int64_t where = locate(p, ia, ib);
      if (where < 0) continue;
      pair[n] = base_pair + u[i];
      addr[n] = where;
      sign[n] = (par[i] ^ par[k]) ? -1 : 1;
      ++n;
    }
  }
  return n;
}

/* The occupied spinors of string sid of a side; returns their number. */
static int string_occ(const zg_plan *p, int side, int64_t sid, int *occ) {
  const zg_side *s = &p->s[side];
  const int c = s->bucket_class[s->bucket[sid]];
  int counts[ZG_MAX_SUB];
  uint32_t masks[ZG_MAX_SUB];
  zg_counts(p, side, s->class_tuple[c], counts);
  zg_unpack(p, side, counts, sid - s->class_base[c], masks);
  int n = 0;
  for (int g = 0; g < s->nsub; ++g) {
    const int sub = s->first + g;
    for (int j = 0; j < p->m[sub]; ++j) {
      if (masks[g] >> j & 1) occ[n++] = p->off[sub] + j;
    }
  }
  return n;
}

void zg_dets(const zg_plan *p, uint64_t *words) {
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  const int nwords = zg_nwords(p);
  memset(words, 0, (size_t)p->dim * nwords * sizeof(uint64_t));
#pragma omp parallel for schedule(dynamic)
  for (int ba = 0; ba < sa->nbucket; ++ba) {
    int occ[ZG_MAX_SPINOR];
    for (int cb = 0; cb < sb->nclass; ++cb) {
      const size_t k = (size_t)ba * sb->nclass + cb;
      const int bb = p->match[k];
      if (bb < 0) continue;
      const int64_t nb = sb->bucket_size[bb];
      for (int64_t i = 0; i < sa->bucket_size[ba]; ++i) {
        const int na = string_occ(p, 0, sa->bucket_sid[sa->bucket_start[ba] + i], occ);
        for (int64_t j = 0; j < nb; ++j) {
          uint64_t *det = words + (size_t)(p->blockoff[k] + i * nb + j) * nwords;
          for (int x = 0; x < na; ++x) det[occ[x] / 64] |= (uint64_t)1 << (occ[x] % 64);
        }
      }
      for (int64_t j = 0; j < nb; ++j) {
        const int n = string_occ(p, 1, sb->bucket_sid[sb->bucket_start[bb] + j], occ);
        for (int64_t i = 0; i < sa->bucket_size[ba]; ++i) {
          uint64_t *det = words + (size_t)(p->blockoff[k] + i * nb + j) * nwords;
          for (int x = 0; x < n; ++x) det[occ[x] / 64] |= (uint64_t)1 << (occ[x] % 64);
        }
      }
    }
  }
}

/* For the kernels: the occupied spinors of a string. */
int zg_string_occ(const zg_plan *p, int side, int64_t sid, int *occ) {
  return string_occ(p, side, sid, occ);
}

#define ZG_SCALAR double
#define ZG_NAME(x) x##_d
#define ZG_CONJ(x) (x)
#define ZG_REAL(x) (x)
#define ZG_ATOMIC_ADD(target, value) \
  do {                               \
    _Pragma("omp atomic")            \
    (target) += (value);             \
  } while (0)
#include "zgasci_kernel.inc"
#undef ZG_SCALAR
#undef ZG_NAME
#undef ZG_CONJ
#undef ZG_REAL
#undef ZG_ATOMIC_ADD

#include <complex.h>
#define ZG_SCALAR double complex
#define ZG_NAME(x) x##_z
#define ZG_CONJ(x) conj(x)
#define ZG_REAL(x) creal(x)
#define ZG_ATOMIC_ADD(target, value)               \
  do {                                             \
    double *zg_t_ = (double *)&(target);           \
    const double complex zg_v_ = (value);          \
    _Pragma("omp atomic")                          \
    zg_t_[0] += creal(zg_v_);                      \
    _Pragma("omp atomic")                          \
    zg_t_[1] += cimag(zg_v_);                      \
  } while (0)
#include "zgasci_kernel.inc"
