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
    const int *w = p->gas_w + (size_t)c * ZG_MAX_SUB;
    for (int g = 0; g < p->nsub; ++g) {
      if (!w[g]) continue;
      sum += w[g] * (g < p->s[0].nsub ? counts_a[g] : counts_b[g - p->s[0].nsub]);
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

static void setup_block(zg_plan *p);
static void setup_blas(zg_plan *p);
static void build_filter(zg_plan *p, int h, zg_bkind *kd);
static inline int64_t side_sid(const zg_plan *p, int side, const int *counts,
                               const uint32_t *masks);

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
                     const int *gas_sub, const int *gas_weight,
                     const int *gas_min, const int *gas_max, int nsets,
                     const int *labels, const int *mods, const int *targets) {
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
  p->gas_w = calloc((size_t)(ngas + 1) * ZG_MAX_SUB, sizeof(int));
  p->gas_min = malloc((ngas + 1) * sizeof(int));
  p->gas_max = malloc((ngas + 1) * sizeof(int));
  for (int c = 0; c < ngas; ++c) {
    for (int k = gas_ptr[c]; k < gas_ptr[c + 1]; ++k) {
      p->gas_w[(size_t)c * ZG_MAX_SUB + gas_sub[k]] += gas_weight ? gas_weight[k] : 1;
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

  /* Which classes pair at all. */
  p->cpair = calloc((size_t)sa->nclass * sb->nclass + 1, 1);
  for (int ca = 0; ca < sa->nclass; ++ca) {
    for (int cb = 0; cb < sb->nclass; ++cb) {
      p->cpair[(size_t)ca * sb->nclass + cb] = (uint8_t)classes_pair(p, ca, cb);
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
  setup_block(p);
  setup_blas(p);
  return p;
}

void zg_plan_free(zg_plan *p) {
  if (p == NULL) return;
  for (int g = 0; g < ZG_MAX_SUB; ++g) {
    free(p->rank[g]);
    free(p->masks[g]);
    free(p->sublabel[g]);
  }
  free(p->gas_w);
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
  free(p->cpair);
  free(p->tasks);
  free(p->btasks);
  free(p->match_t);
  free(p->blockoff_t);
  free(p->rtasks[0]);
  free(p->rtasks[1]);
  free(p->tpieces);
  for (int h = 0; h < 2; ++h) {
    free(p->pair_ptr[h]);
    free(p->pair_cls[h]);
    free(p->bp[h]);
    for (int k = 0; k < p->nbkind[h]; ++k) {
      free(p->bkinds[h][k].fkey);
      free(p->bkinds[h][k].fmask);
    }
  }
  free(p);
}

int64_t zg_dim(const zg_plan *p) { return p->dim; }
int zg_block_ok(const zg_plan *p) { return p->block_ok; }
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
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  int counts[ZG_MAX_SUB];
  uint32_t work[ZG_MAX_SUB];
  for (int g = 0; g < nsub; ++g) {
    counts[g] = __builtin_popcount(masks[g]);
    work[g] = masks[g];
  }
  /* The unoccupied spinors, A ones first, by sub-group (ustart), and the
   * parity of the number of occupied ones below each. */
  int u[ZG_MAX_SPINOR], nu = 0, ustart[ZG_MAX_SUB + 1];
  uint8_t par[ZG_MAX_SPINOR];
  for (int g = 0; g < nsub; ++g) {
    ustart[g] = nu;
    for (int j = 0; j < p->m[g]; ++j) {
      if (masks[g] >> j & 1) continue;
      u[nu] = p->off[g] + j;
      par[nu] = (uint8_t)((u[nu] - nu) & 1);
      ++nu;
    }
  }
  ustart[nsub] = nu;
  /* The tuples of K, and which pairs of sub-groups can take the two
   * electrons: both resulting tuples classes, and the classes paired. */
  int64_t t[2] = {0, 0};
  for (int g = 0; g < nsub; ++g) {
    const int side = g >= na;
    t[side] += counts[g] * p->s[side].stride[g - (side ? na : 0)];
  }
  /* Each side as it is, and with one electron added to a sub-group. */
  int64_t sid0[2];
  sid0[0] = side_sid(p, 0, counts, masks);
  sid0[1] = side_sid(p, 1, counts + na, masks + na);
  int32_t cls0[2] = {sa->class_of[t[0]], sb->class_of[t[1]]};
  int32_t cls1[ZG_MAX_SUB];
  for (int g = 0; g < nsub; ++g) {
    const int side = g >= na, loc = g - (side ? na : 0);
    cls1[g] = counts[g] < p->m[g] ? p->s[side].class_of[t[side] + p->s[side].stride[loc]] : -1;
  }
  int64_t sid1[ZG_MAX_SPINOR];
  for (int g = 0; g < nsub; ++g) {
    const int side = g >= na, base = side ? na : 0;
    for (int i = ustart[g]; i < ustart[g + 1]; ++i) {
      if (cls1[g] < 0) {
        sid1[i] = -1;
        continue;
      }
      work[g] |= 1u << (u[i] - p->off[g]);
      counts[g]++;
      sid1[i] = side_sid(p, side, counts + base, work + base);
      work[g] = masks[g];
      counts[g]--;
    }
  }
  int n = 0;
  for (int gk = 0; gk < nsub; ++gk) {
    for (int gi = 0; gi <= gk; ++gi) {
      /* Feasibility of the pair of sub-groups. */
      const int side_i = gi >= na, side_k = gk >= na;
      int ca, cb;
      if (side_i != side_k) {
        ca = cls1[gi];
        cb = cls1[gk];
      } else {
        const int side = side_k, base = side ? na : 0;
        const zg_side *s = &p->s[side];
        const int need = gi == gk ? 2 : 1;
        if (counts[gi] + need > p->m[gi] || (gi != gk && counts[gk] + 1 > p->m[gk])) continue;
        const int64_t tt = t[side] + s->stride[gi - base] + s->stride[gk - base];
        const int32_t c2 = s->class_of[tt];
        ca = side ? cls0[0] : c2;
        cb = side ? c2 : cls0[1];
      }
      if (ca < 0 || cb < 0 || !p->cpair[(size_t)ca * sb->nclass + cb]) continue;
      for (int k = ustart[gk]; k < ustart[gk + 1]; ++k) {
        const int32_t base_pair = u[k] * (u[k] - 1) / 2;
        const int i1 = gi == gk ? k : ustart[gi + 1];
        for (int i = ustart[gi]; i < i1; ++i) {
          int64_t ia, ib;
          if (side_i != side_k) {
            ia = sid1[i];
            ib = sid1[k];
          } else {
            const int side = side_k, base = side ? na : 0;
            work[gi] |= 1u << (u[i] - p->off[gi]);
            work[gk] |= 1u << (u[k] - p->off[gk]);
            counts[gi]++;
            counts[gk]++;
            const int64_t s2 = side_sid(p, side, counts + base, work + base);
            work[gi] = masks[gi];
            work[gk] = masks[gk];
            counts[gi]--;
            counts[gk]--;
            ia = side ? sid0[0] : s2;
            ib = side ? s2 : sid0[1];
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
    }
  }
  return n;
}


/* -- the block kernel: terms, pieces of work, strings as words ------------ */

/* The nine kinds of terms; see zg_term. */
static void setup_terms(zg_plan *p) {
  static const int kind[4] = {1, 1, 0, 0}; /* a+_p a+_q a_s a_r */
  for (int tp = 0; tp < 3; ++tp) {
    for (int tr = 0; tr < 3; ++tr) {
      zg_term *t = &p->terms[tp * 3 + tr];
      memset(t, 0, sizeof(*t));
      /* Side of p, q, s, r: P within A (0), across (1), within B (2). */
      int side[4];
      side[0] = tp == 2;
      side[1] = tp >= 1;
      side[3] = tr == 2; /* r */
      side[2] = tr >= 1; /* s */
      int nb_before = 0, sign = 1;
      for (int i = 0; i < 4; ++i) {
        if (side[i]) {
          t->kind_b[t->kb] = kind[i];
          t->role_b[t->kb++] = i;
          ++nb_before;
        } else {
          t->kind_a[t->ka] = kind[i];
          t->role_a[t->ka++] = i;
          if (nb_before % 2) sign = -sign;
        }
      }
      t->sign = sign;
      /* Order within a side: p < q, and s > r (listed s first). */
      for (int which = 0; which < 2; ++which) {
        const int k = which ? t->kb : t->ka;
        const int *role = which ? t->role_b : t->role_a;
        int *rel = which ? t->rel_b : t->rel_a;
        for (int i = 0; i + 1 < k; ++i) {
          if (role[i] == 0 && role[i + 1] == 1) rel[i] = 1;
          if (role[i] == 2 && role[i + 1] == 3) rel[i] = -1;
        }
      }
      t->nx = 1;
      t->ny = 1;
      for (int i = 0; i < t->ka; ++i) t->nx *= p->nside[0];
      for (int i = 0; i < t->kb; ++i) t->ny *= p->nside[1];
    }
  }
}

/* Work of a row of a block, roughly: its columns times the excitations of
 * each side (the same for all rows of a class). */
static double row_weight(const zg_plan *p, int ca, int cb, int64_t ncol) {
  double w[2];
  for (int side = 0; side < 2; ++side) {
    const zg_side *s = &p->s[side];
    const int k = s->tuple_elec[s->class_tuple[side ? cb : ca]];
    const int n = p->nside[side];
    w[side] = 1.0 + (double)k * (n - k) + 0.25 * k * (k - 1) * (n - k) * (n - k - 1);
  }
  return (double)ncol * w[0] * w[1];
}

static void setup_block(zg_plan *p) {
  for (int side = 0; side < 2; ++side) {
    const zg_side *s = &p->s[side];
    p->nside[side] = p->off[s->first + s->nsub - 1] + p->m[s->first + s->nsub - 1] -
                     p->off[s->first];
  }
  p->block_ok = p->nside[0] <= 64 && p->nside[1] <= 64;
  if (!p->block_ok) return;
  setup_terms(p);
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  /* Tile shape; ZGASCI_TILE="rows,columns" overrides it (experiments). */
  int64_t max_rows = 128, max_cols = 32;
  const char *tile = getenv("ZGASCI_TILE");
  if (tile) {
    long r = 0, cc = 0;
    if (sscanf(tile, "%ld,%ld", &r, &cc) == 2 && r > 0 && cc > 0) {
      max_rows = r < ZG_MAX_ROWS ? r : ZG_MAX_ROWS;
      max_cols = cc < ZG_MAX_COLS ? cc : ZG_MAX_COLS;
    }
  }
  double total = 0.0;
  for (int pass = 0; pass < 2; ++pass) {
    const double piece = total / (8.0 * omp_get_max_threads()) + 1.0;
    int64_t capacity = 256;
    if (pass == 1) p->btasks = malloc(capacity * sizeof(zg_btask));
    p->nbtask = 0;
    for (int ba = 0; ba < sa->nbucket; ++ba) {
      for (int cb = 0; cb < sb->nclass; ++cb) {
        const int64_t k = (int64_t)ba * sb->nclass + cb;
        const int bb = p->match[k];
        if (bb < 0) continue;
        const int64_t nrow = sa->bucket_size[ba];
        const double w = row_weight(p, sa->bucket_class[ba], cb, sb->bucket_size[bb]);
        if (pass == 0) {
          total += w * nrow;
          continue;
        }
        /* Tiles of at most ZG_MAX_ROWS x ZG_MAX_COLS, smaller where the
         * work would be too coarse to balance. */
        const int64_t ncol = sb->bucket_size[bb];
        const double per = w / (double)ncol; /* one element */
        int64_t rows = nrow < max_rows ? nrow : max_rows;
        int64_t cols = ncol < max_cols ? ncol : max_cols;
        while (rows * cols * per > piece && (rows > 8 || cols > 8)) {
          if (rows >= cols) rows = (rows + 1) / 2; else cols = (cols + 1) / 2;
        }
        for (int64_t r0 = 0; r0 < nrow; r0 += rows) {
          for (int64_t j0 = 0; j0 < ncol; j0 += cols) {
            if (p->nbtask == capacity) {
              capacity *= 2;
              p->btasks = realloc(p->btasks, capacity * sizeof(zg_btask));
            }
            p->btasks[p->nbtask++] = (zg_btask){
                k, r0, r0 + rows < nrow ? r0 + rows : nrow, j0,
                j0 + cols < ncol ? j0 + cols : ncol};
          }
        }
      }
    }
  }
}

/* -- the BLAS kernel: the transposed view, kinds of terms, pieces of work -- */

/* Multiply-adds of a row of half h, per column, roughly (the matrix
 * products of zgasci_blas.inc for kr electrons of n on the row side). */
static double blas_row_weight(const zg_plan *p, int h, int kr) {
  const double n = p->nside[h], nc = p->nside[1 - h], k = kr, u = n - k;
  const double pairs = 0.5 * k * (k - 1);
  double w = pairs * 0.5 * (u + 2) * (u + 1)      /* two electrons moved */
             + pairs * (u + 2) * nc               /* a+ a+ a (x) a */
             + k * 0.5 * (u + 1) * u * nc         /* a+ a a (x) a+ */
             + pairs * 0.5 * nc * (nc - 1);       /* a+ a+ (x) a a */
  if (h == 0) w += k * (u + 1) * nc * nc;         /* a+ a (x) a+ a */
  return w + 1.0;
}

static void setup_blas(zg_plan *p) {
  if (!p->block_ok) return;
  const zg_side *sa = &p->s[0], *sb = &p->s[1];
  /* The transposed vector. */
  const size_t nt = (size_t)sb->nbucket * sa->nclass;
  p->match_t = malloc((nt + 1) * sizeof(int32_t));
  p->blockoff_t = malloc((nt + 1) * sizeof(int64_t));
  for (size_t i = 0; i < nt; ++i) {
    p->match_t[i] = -1;
    p->blockoff_t[i] = 0;
  }
  for (int ba = 0; ba < sa->nbucket; ++ba) {
    for (int cb = 0; cb < sb->nclass; ++cb) {
      const int bb = p->match[(size_t)ba * sb->nclass + cb];
      if (bb >= 0) p->match_t[(size_t)bb * sa->nclass + sa->bucket_class[ba]] = ba;
    }
  }
  int64_t at = 0;
  for (size_t i = 0; i < nt; ++i) {
    if (p->match_t[i] < 0) continue;
    p->blockoff_t[i] = at;
    at += sb->bucket_size[i / sa->nclass] * sa->bucket_size[p->match_t[i]];
  }
  /* The distinct labels of the spinors of each side. */
  for (int side = 0; side < 2; ++side) {
    const zg_side *s = &p->s[side];
    p->bp[side] = malloc((size_t)(s->nsid + 1) * sizeof(uint64_t));
    for (int64_t sid = 0; sid < s->nsid; ++sid) {
      p->bp[side][sid] = (uint64_t)s->bucket[sid] << 32 | s->pos[sid];
    }
    p->ndlab[side] = 0;
    for (int g = 0; g < s->nsub; ++g) {
      const int sub = s->first + g;
      for (int j = 0; j < p->m[sub]; ++j) {
        int lab[ZG_MAX_SETS] = {0}, found = -1;
        for (int set = 0; set < p->nsets; ++set) {
          lab[set] = reduce_label(
              p->sublabel[sub][((size_t)set << p->m[sub]) + ((size_t)1 << j)],
              p->mods[set]);
        }
        for (int d = 0; d < p->ndlab[side] && found < 0; ++d) {
          if (memcmp(p->dlab[side][d], lab, sizeof(lab)) == 0) found = d;
        }
        if (found < 0) {
          found = p->ndlab[side]++;
          memcpy(p->dlab[side][found], lab, sizeof(lab));
          p->dmask[side][found] = 0;
        }
        p->dmask[side][found] |= (uint64_t)1 << (p->off[sub] + j - p->off[s->first]);
        p->dof[side][p->off[sub] + j - p->off[s->first]] = (uint8_t)found;
      }
    }
  }
  /* The kinds of each half: the side with more operators, or with the
   * creation operators, is the row side. */
  static const int which[2][5] = {{0, 1, 3, 4, 2}, {8, 7, 5, 6, -1}};
  for (int h = 0; h < 2; ++h) {
    p->nbkind[h] = 0;
    for (int i = 0; i < 5 && which[h][i] >= 0; ++i) {
      const zg_term *t = &p->terms[which[h][i]];
      zg_bkind *kd = &p->bkinds[h][p->nbkind[h]++];
      memset(kd, 0, sizeof(*kd));
      kd->term = which[h][i];
      kd->kr = h ? t->kb : t->ka;
      kd->kc = h ? t->ka : t->kb;
      const int *kind_r = h ? t->kind_b : t->kind_a, *rel_r = h ? t->rel_b : t->rel_a;
      const int *kind_c = h ? t->kind_a : t->kind_b, *rel_c = h ? t->rel_a : t->rel_b;
      for (int j = 0; j < kd->kr; ++j) {
        kd->kind_r[j] = kind_r[j];
        kd->rel_r[j] = rel_r[j];
      }
      for (int j = 0; j < kd->kc; ++j) {
        kd->kind_c[j] = kind_c[j];
        kd->rel_c[j] = rel_c[j];
        kd->add_c += kind_c[j] ? -1 : 1;
      }
      for (int j = 0; j < kd->kr; ++j) kd->add_r += kind_r[j] ? -1 : 1;
      kd->par_r = h == 0 && (t->kb & 1);
      kd->par_c = h == 1 && (t->kb & 1);
      /* The matrix of the kind, rows first (zg_sigma_blas transposes those
       * of half 1). */
      kd->wr = h ? t->nx : t->ny;
      kd->wc = 1;
      kd->ncidx = h ? t->nx : t->ny;
      build_filter(p, h, kd);
    }
  }
  /* Pieces of work: rows of the buckets of each half, at most
   * ZG_BLAS_ROWS (ZGASCI_BLAS_ROWS lowers it, for experiments). */
  int64_t max_rows = ZG_BLAS_ROWS;
  const char *text = getenv("ZGASCI_BLAS_ROWS");
  if (text && atol(text) > 0 && atol(text) < max_rows) max_rows = atol(text);
  for (int h = 0; h < 2; ++h) {
    const zg_side *sr = &p->s[h], *sc = &p->s[1 - h];
    const int32_t *match = h ? p->match_t : p->match;
    p->pair_ptr[h] = malloc((size_t)(sr->nbucket + 1) * sizeof(int64_t));
    p->pair_cls[h] = malloc((size_t)(p->nblock + 1) * sizeof(int32_t));
    int64_t npair = 0;
    for (int bt = 0; bt < sr->nbucket; ++bt) {
      p->pair_ptr[h][bt] = npair;
      for (int cc = 0; cc < sc->nclass; ++cc) {
        if (match[(size_t)bt * sc->nclass + cc] >= 0) p->pair_cls[h][npair++] = cc;
      }
    }
    p->pair_ptr[h][sr->nbucket] = npair;
    double total = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
      const double piece = total / (16.0 * omp_get_max_threads()) + 1.0;
      int64_t capacity = 256;
      if (pass == 1) p->rtasks[h] = malloc(capacity * sizeof(zg_rtask));
      p->nrtask[h] = 0;
      for (int bt = 0; bt < sr->nbucket; ++bt) {
        double ncol = 0.0;
        for (int cc = 0; cc < sc->nclass; ++cc) {
          const int bb = match[(size_t)bt * sc->nclass + cc];
          if (bb >= 0) ncol += (double)sc->bucket_size[bb];
        }
        if (ncol == 0.0) continue;
        const int kr = sr->tuple_elec[sr->class_tuple[sr->bucket_class[bt]]];
        const double w = ncol * blas_row_weight(p, h, kr);
        const int64_t nrow = sr->bucket_size[bt];
        if (pass == 0) {
          total += w * nrow;
          continue;
        }
        int64_t rows = (int64_t)(piece / w);
        if (rows < 1) rows = 1;
        if (rows > max_rows) rows = max_rows;
        for (int64_t r0 = 0; r0 < nrow; r0 += rows) {
          if (p->nrtask[h] == capacity) {
            capacity *= 2;
            p->rtasks[h] = realloc(p->rtasks[h], capacity * sizeof(zg_rtask));
          }
          p->rtasks[h][p->nrtask[h]++] =
              (zg_rtask){bt, r0, r0 + rows < nrow ? r0 + rows : nrow};
        }
      }
    }
  }
  /* Pieces of the transposition, of about 2^16 elements. */
  int64_t capacity = 256;
  p->tpieces = malloc(capacity * sizeof(zg_tpiece));
  p->ntpiece = 0;
  for (int ba = 0; ba < sa->nbucket; ++ba) {
    for (int cb = 0; cb < sb->nclass; ++cb) {
      const int64_t k = (int64_t)ba * sb->nclass + cb;
      const int bb = p->match[k];
      if (bb < 0) continue;
      const int64_t na = sa->bucket_size[ba], nb = sb->bucket_size[bb];
      int64_t rows = 65536 / nb;
      if (rows < 1) rows = 1;
      for (int64_t i0 = 0; i0 < na; i0 += rows) {
        if (p->ntpiece == capacity) {
          capacity *= 2;
          p->tpieces = realloc(p->tpieces, capacity * sizeof(zg_tpiece));
        }
        p->tpieces[p->ntpiece++] = (zg_tpiece){k, i0, i0 + rows < na ? i0 + rows : na};
      }
    }
  }
}

int zg_asg_list(const zg_plan *p, int side, int k, const int *kind,
                const int *rel, int from, const uint8_t *ok, zg_asg *out,
                int cap) {
  const zg_side *s = &p->s[side];
  const int64_t ts = s->class_tuple[from];
  int counts[ZG_MAX_SUB];
  zg_counts(p, side, ts, counts);
  int64_t total = 1;
  for (int i = 0; i < k; ++i) total *= s->nsub;
  int n = 0;
  for (int64_t code = 0; code < total; ++code) {
    int g[4] = {0, 0, 0, 0}, good = 1;
    int64_t rest = code;
    for (int i = 0; i < k; ++i) {
      g[i] = (int)(rest % s->nsub);
      rest /= s->nsub;
    }
    for (int i = 0; i + 1 < k; ++i) {
      if ((rel[i] > 0 && g[i] > g[i + 1]) || (rel[i] < 0 && g[i] < g[i + 1])) good = 0;
    }
    if (!good) continue;
    int now[ZG_MAX_SUB];
    int64_t tt = ts;
    memcpy(now, counts, sizeof(int) * s->nsub);
    for (int i = 0; i < k; ++i) {
      /* creation in O: the source has one electron fewer there. */
      const int d = kind[i] ? -1 : 1;
      now[g[i]] += d;
      tt += d * s->stride[g[i]];
    }
    for (int i = 0; i < k; ++i) {
      if (now[g[i]] < 0 || now[g[i]] > p->m[s->first + g[i]]) good = 0;
    }
    if (!good) continue;
    const int cls = s->class_of[tt];
    if (cls < 0 || (ok && !ok[cls])) continue;
    if (n < cap) {
      zg_asg *a = &out[n];
      a->cls = cls;
      a->base = s->class_base[cls];
      a->ntouch = 0;
      for (int i = 0; i < 4; ++i) a->g[i] = i < k ? g[i] : 0;
      for (int i = 0; i < k; ++i) {
        int seen = 0;
        for (int j = 0; j < a->ntouch; ++j) seen = seen || a->touch[j] == g[i];
        if (!seen) a->touch[a->ntouch++] = g[i];
      }
      int64_t radix = 1;
      for (int gg = 0; gg < s->nsub; ++gg) {
        a->radix[gg] = radix;
        radix *= zg_binom(p->m[s->first + gg], now[gg]);
      }
    }
    ++n;
  }
  return n;
}

void zg_string_ranks(const zg_plan *p, int side, uint64_t w, int64_t *rk) {
  const zg_side *s = &p->s[side];
  const int first = p->off[s->first];
  for (int g = 0; g < s->nsub; ++g) {
    const int sub = s->first + g;
    rk[g] = p->rank[sub][(w >> (p->off[sub] - first)) & ((1u << p->m[sub]) - 1)];
  }
}

/* A change of the labels as one number (16 bits a set). */
static inline uint64_t label_key(const int *change, int nsets, const int *mods) {
  uint64_t key = 0;
  for (int s = 0; s < nsets; ++s) {
    key |= (uint64_t)((reduce_label(change[s], mods[s]) + 32768) & 0xffff) << (16 * s);
  }
  return key;
}

static inline uint64_t filter_slot(const zg_bkind *kd, uint64_t key) {
  return (key * 0x9E3779B97F4A7C15ull) >> 40 & (uint64_t)(kd->fsize - 1);
}

/* The spinors the last row operator may take after the change `key`. */
static inline uint64_t filter_mask(const zg_bkind *kd, uint64_t key) {
  for (uint64_t i = filter_slot(kd, key);; i = (i + 1) & (uint64_t)(kd->fsize - 1)) {
    if (kd->fkey[i] == key) return kd->fmask[i];
    if (kd->fkey[i] == UINT64_MAX) return 0;
  }
}

/* Builds the label filter of a kind of half h (see zg_bkind). */
static void build_filter(zg_plan *p, int h, zg_bkind *kd) {
  kd->fsize = 0;
  kd->fkey = kd->fmask = NULL;
  if (p->nsets == 0 || kd->kc > 1 || kd->kr < 1) return;
  const int nd = p->ndlab[h], ndc = p->ndlab[1 - h];
  int64_t combos = 1;
  for (int i = 0; i + 1 < kd->kr; ++i) combos *= nd;
  if (combos > 20000) return;
  int big = 0;
  for (int side = 0; side < 2; ++side) {
    for (int d = 0; d < p->ndlab[side]; ++d) {
      for (int set = 0; set < p->nsets; ++set) {
        if (abs(p->dlab[side][d][set]) > big) big = abs(p->dlab[side][d][set]);
      }
    }
  }
  if (big * 5 > 30000) return;
  /* The changes of the row labels (source minus target) that are allowed:
   * minus the change the column operator makes. */
  int nallow = 0;
  uint64_t allow[65];
  if (kd->kc == 0) {
    int zero[ZG_MAX_SETS] = {0};
    allow[nallow++] = label_key(zero, p->nsets, p->mods);
  } else {
    const int csign = kd->kind_c[0] ? -1 : 1;
    for (int d = 0; d < ndc; ++d) {
      int lab[ZG_MAX_SETS];
      for (int set = 0; set < p->nsets; ++set) lab[set] = -csign * p->dlab[1 - h][d][set];
      allow[nallow++] = label_key(lab, p->nsets, p->mods);
    }
  }
  int size = 16;
  while (size < 2 * combos) size *= 2;
  kd->fsize = size;
  kd->fkey = malloc((size_t)size * sizeof(uint64_t));
  kd->fmask = malloc((size_t)size * sizeof(uint64_t));
  for (int i = 0; i < size; ++i) kd->fkey[i] = UINT64_MAX;
  const int last = kd->kind_r[kd->kr - 1] ? -1 : 1;
  for (int64_t code = 0; code < combos; ++code) {
    int acc[ZG_MAX_SETS] = {0};
    int64_t rest = code;
    for (int i = 0; i + 1 < kd->kr; ++i) {
      const int d = (int)(rest % nd), lsign = kd->kind_r[i] ? -1 : 1;
      rest /= nd;
      for (int set = 0; set < p->nsets; ++set) acc[set] += lsign * p->dlab[h][d][set];
    }
    const uint64_t key = label_key(acc, p->nsets, p->mods);
    uint64_t slot = filter_slot(kd, key);
    while (kd->fkey[slot] != UINT64_MAX && kd->fkey[slot] != key) {
      slot = (slot + 1) & (uint64_t)(size - 1);
    }
    if (kd->fkey[slot] == key) continue;
    uint64_t good = 0;
    for (int d = 0; d < nd; ++d) {
      int total[ZG_MAX_SETS];
      for (int set = 0; set < p->nsets; ++set) total[set] = acc[set] + last * p->dlab[h][d][set];
      const uint64_t tkey = label_key(total, p->nsets, p->mods);
      for (int a = 0; a < nallow; ++a) {
        if (allow[a] == tkey) {
          good |= p->dmask[h][d];
          break;
        }
      }
    }
    kd->fkey[slot] = key;
    kd->fmask[slot] = good;
  }
}

typedef struct {
  int k, nside;
  const int *kind, *rel;
  /* Label filter: the change so far at each slot. */
  const zg_bkind *filter;
  int nsets;
  const int (*dlab)[ZG_MAX_SETS];
  const uint8_t *dof;
  const int *mods;
  int acc[5][ZG_MAX_SETS];
  uint64_t range[4]; /* the spinors of the sub-group of each slot */
  int ntouch, shift[4];
  uint32_t mask[4];
  const uint32_t *rank[4];
  int64_t radix[4];
  int64_t fixed;
  int64_t *x, *sid;
  int8_t *sign;
  int n;
} asg_ctx;

static void asg_rec(asg_ctx *c, int i, uint64_t w, int prev, int8_t sg,
                    int64_t x, int64_t xradix) {
  if (i == c->k) {
    int64_t sid = c->fixed;
    for (int t = 0; t < c->ntouch; ++t) {
      sid += c->radix[t] * c->rank[t][(w >> c->shift[t]) & c->mask[t]];
    }
    c->x[c->n] = x;
    c->sid[c->n] = sid;
    c->sign[c->n] = sg;
    c->n++;
    return;
  }
  if (i > 3) return; /* at most four operators */
  uint64_t bits = (c->kind[i] ? w : ~w) & c->range[i];
  if (i > 0 && c->rel[i - 1] > 0) bits &= ~(((uint64_t)2 << prev) - 1);
  if (i > 0 && c->rel[i - 1] < 0) bits &= ((uint64_t)1 << prev) - 1;
  /* creation in O: the source lacks this spinor. */
  const int lsign = c->kind[i] ? -1 : 1;
  if (c->filter && i == c->k - 1) {
    /* acc is kept reduced, so this is label_key without the divisions. */
    uint64_t key = 0;
    for (int s = 0; s < c->nsets; ++s) {
      key |= (uint64_t)((c->acc[i][s] + 32768) & 0xffff) << (16 * s);
    }
    bits &= filter_mask(c->filter, key);
  }
  while (bits) {
    const int j = __builtin_ctzll(bits);
    bits &= bits - 1;
    if (c->filter && i < c->k - 1) {
      const int *lab = c->dlab[c->dof[j]];
      for (int s = 0; s < c->nsets; ++s) {
        /* lab is reduced: one step brings the sum back into range. */
        int v = c->acc[i][s] + lsign * lab[s];
        const int mod = c->mods[s];
        if (mod) v = v >= mod ? v - mod : v < 0 ? v + mod : v;
        c->acc[i + 1][s] = v;
      }
    }
    const int8_t s =
        (__builtin_popcountll(w & (((uint64_t)1 << j) - 1)) & 1) ? (int8_t)-sg : sg;
    asg_rec(c, i + 1, w ^ ((uint64_t)1 << j), j, s, x + xradix * j,
            xradix * c->nside);
  }
}

int zg_asg_enum(const zg_plan *p, int side, int k, const int *kind,
                const int *rel, const zg_asg *a, uint64_t w, const int64_t *rk,
                const zg_bkind *filter, int64_t *x, int64_t *sid, int8_t *sign,
                int n) {
  const zg_side *s = &p->s[side];
  const int first = p->off[s->first];
  asg_ctx c;
  c.k = k;
  c.nside = p->nside[side];
  c.kind = kind;
  c.rel = rel;
  for (int i = 0; i < k; ++i) {
    const int sub = s->first + a->g[i];
    c.range[i] = (((uint64_t)1 << p->m[sub]) - 1) << (p->off[sub] - first);
  }
  c.ntouch = a->ntouch;
  c.fixed = a->base;
  for (int g = 0; g < s->nsub; ++g) c.fixed += a->radix[g] * rk[g];
  for (int t = 0; t < a->ntouch; ++t) {
    const int g = a->touch[t], sub = s->first + g;
    c.fixed -= a->radix[g] * rk[g];
    c.shift[t] = p->off[sub] - first;
    c.mask[t] = (1u << p->m[sub]) - 1;
    c.rank[t] = p->rank[sub];
    c.radix[t] = a->radix[g];
  }
  c.filter = filter && filter->fsize > 0 ? filter : NULL;
  c.nsets = p->nsets;
  c.dlab = (const int (*)[ZG_MAX_SETS])p->dlab[side];
  c.dof = p->dof[side];
  c.mods = p->mods;
  for (int set = 0; set < ZG_MAX_SETS; ++set) c.acc[0][set] = 0;
  c.x = x;
  c.sid = sid;
  c.sign = sign;
  c.n = n;
  asg_rec(&c, 0, w, -1, 1, 0, 1);
  return c.n;
}

uint64_t zg_side_word(const zg_plan *p, int side, int64_t sid) {
  const zg_side *s = &p->s[side];
  const int c = s->bucket_class[s->bucket[sid]];
  int counts[ZG_MAX_SUB];
  uint32_t masks[ZG_MAX_SUB];
  zg_counts(p, side, s->class_tuple[c], counts);
  zg_unpack(p, side, counts, sid - s->class_base[c], masks);
  uint64_t w = 0;
  const int first = p->off[s->first];
  for (int g = 0; g < s->nsub; ++g) {
    w |= (uint64_t)masks[g] << (p->off[s->first + g] - first);
  }
  return w;
}

int64_t zg_word_sid(const zg_plan *p, int side, uint64_t w) {
  const zg_side *s = &p->s[side];
  int counts[ZG_MAX_SUB];
  uint32_t masks[ZG_MAX_SUB];
  const int first = p->off[s->first];
  for (int g = 0; g < s->nsub; ++g) {
    const int sub = s->first + g;
    masks[g] = (uint32_t)((w >> (p->off[sub] - first)) & ((1u << p->m[sub]) - 1));
    counts[g] = __builtin_popcount(masks[g]);
  }
  return side_sid(p, side, counts, masks);
}

/* Applies O(x)^dagger = o_1^dagger ... slot by slot, o_1^dagger first. */
static int enum_rec(int i, int k, const int *kind, const int *rel, int nside,
                    uint64_t w, int64_t x, int64_t radix, int prev, int8_t sg,
                    int64_t *xs, uint64_t *src, int8_t *sign, int n, int cap) {
  if (i == k) {
    if (n < cap) {
      xs[n] = x;
      src[n] = w;
      sign[n] = sg;
    }
    return n + 1;
  }
  int lo = 0, hi = nside;
  if (i > 0 && rel[i - 1] > 0) lo = prev + 1;
  if (i > 0 && rel[i - 1] < 0) hi = prev;
  /* creation in O: its adjoint removes an electron, and the other way. */
  const uint64_t want = kind[i] ? w : ~w;
  for (int j = lo; j < hi; ++j) {
    if (!(want >> j & 1)) continue;
    const int8_t s =
        (__builtin_popcountll(w & (((uint64_t)1 << j) - 1)) & 1) ? -sg : sg;
    n = enum_rec(i + 1, k, kind, rel, nside, w ^ ((uint64_t)1 << j),
                 x + radix * j, radix * nside, j, s, xs, src, sign, n, cap);
  }
  return n;
}

int zg_enum_ops(int k, const int *kind, const int *rel, int nside, uint64_t w,
                int64_t *x, uint64_t *src, int8_t *sign, int cap) {
  return enum_rec(0, k, kind, rel, nside, w, 0, 1, -1, 1, x, src, sign, 0, cap);
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

/* BLAS (Fortran interface, 32-bit integers as the other socutils
 * libraries).  The kernel calls it from inside its own parallel region, one
 * call per thread at a time, so it must run sequentially there: MKL and
 * OpenBLAS are told so for the calling thread when they are the library in
 * use (weak symbols; another BLAS is assumed to be sequential when called
 * from a parallel region). */
void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);
void zgemm_(const char *, const char *, const int *, const int *, const int *,
            const void *, const void *, const int *, const void *, const int *,
            const void *, void *, const int *);
void daxpy_(const int *, const double *, const double *, const int *, double *,
            const int *);
void zaxpy_(const int *, const void *, const void *, const int *, void *,
            const int *);
extern int MKL_Set_Num_Threads_Local(int) __attribute__((weak));
extern int openblas_set_num_threads_local(int) __attribute__((weak));
static int zg_blas_single(void) {
  if (MKL_Set_Num_Threads_Local) return MKL_Set_Num_Threads_Local(1);
  if (openblas_set_num_threads_local) return openblas_set_num_threads_local(1);
  return 0;
}
static void zg_blas_restore(int before) {
  if (MKL_Set_Num_Threads_Local) {
    MKL_Set_Num_Threads_Local(before);
  } else if (openblas_set_num_threads_local) {
    openblas_set_num_threads_local(before);
  }
}

#define ZG_SCALAR double
#define ZG_GEMM dgemm_
#define ZG_AXPY daxpy_
#define ZG_BLAS_DIRECT 4.0
#define ZG_NAME(x) x##_d
#define ZG_CONJ(x) (x)
#define ZG_REAL(x) (x)
#define ZG_ATOMIC_ADD(target, value) \
  do {                               \
    _Pragma("omp atomic")            \
    (target) += (value);             \
  } while (0)
#include "zgasci_kernel.inc"
#include "zgasci_block.inc"
#include "zgasci_blas.inc"
#undef ZG_SCALAR
#undef ZG_GEMM
#undef ZG_AXPY
#undef ZG_BLAS_DIRECT
#undef ZG_NAME
#undef ZG_CONJ
#undef ZG_REAL
#undef ZG_ATOMIC_ADD

#include <complex.h>
#define ZG_SCALAR double complex
#define ZG_GEMM zgemm_
#define ZG_AXPY zaxpy_
#define ZG_BLAS_DIRECT 4.0
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
#include "zgasci_block.inc"
#include "zgasci_blas.inc"
