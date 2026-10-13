/* The addressing scheme shared by the kernels; see zgasci.h. */
#ifndef ZGASCI_PLAN_H_
#define ZGASCI_PLAN_H_

#include <stdint.h>

#include "zgasci.h"

#define ZG_MAX_SUB 16     /* sub-groups */
#define ZG_MAX_SUBSIZE 16 /* spinors of a sub-group */
#define ZG_MAX_SETS 4     /* label sets */
#define ZG_MAX_SPINOR (ZG_MAX_SUB * ZG_MAX_SUBSIZE)
#define ZG_MAX_ROWS 1024  /* rows of a piece of work of the block kernel */
#define ZG_MAX_COLS 512   /* and its columns */

/* One side.  A "tuple" is a choice of the number of electrons in each of
 * its sub-groups, numbered t = sum_g k_g stride_g; the tuples that occur in
 * the CI space are the classes.  The strings of all classes are numbered
 * through (sid); a bucket is the strings of one class with one label. */
typedef struct {
  int nsub; /* its sub-groups are first .. first + nsub - 1 */
  int first;
  int64_t ntuple;
  int64_t stride[ZG_MAX_SUB];
  int *tuple_elec;      /* [ntuple] electrons */
  int64_t *tuple_size;  /* [ntuple] strings */
  int32_t *class_of;    /* [ntuple] class, or -1 */
  int64_t *by_elec;     /* tuples sorted by their electrons ... */
  int64_t *elec_start;  /* ... those with k from elec_start[k], [n + 2] */
  int nclass;
  int64_t *class_tuple; /* [nclass] */
  int64_t *class_base;  /* [nclass + 1] first sid */
  int64_t nsid;
  int32_t *bucket; /* [nsid] */
  uint32_t *pos;   /* [nsid] place in its bucket */
  int nbucket;
  int64_t *bucket_size;  /* [nbucket] */
  int *bucket_label;     /* [nbucket * nsets] */
  int *bucket_class;     /* [nbucket] */
  int64_t *bucket_start; /* [nbucket + 1] into bucket_sid */
  int64_t *bucket_sid;   /* [nsid] the strings of each bucket in order */
  int *class_bucket;     /* [nclass + 1] first bucket of a class */
} zg_side;

/* A piece of the loop over the determinants K of N - 2 electrons: the A
 * strings a0 .. a1 - 1 of tuple ta with all B strings of tuple tb. */
typedef struct {
  int64_t ta, tb, a0, a1;
} zg_task;

/* One of the nine kinds of terms of the block kernel: v[P,R] a+_p a+_q a_s a_r
 * with each of P, R within A, across (p in A, q in B; r in A, s in B) or
 * within B.  Its operators on each side, in operator order, so that the term
 * is sign * (-1)^(kb N_A(ket)) <a'|O_A(x)|a> <b'|O_B(y)|b>; x, y are the
 * local spinors of the slots in mixed radix (n_A, n_B). */
typedef struct {
  int ka, kb;
  int kind_a[4], kind_b[4]; /* 1: creation, 0: annihilation */
  int rel_a[4], rel_b[4];   /* slot i against slot i + 1: +1 <, -1 >, 0 */
  int role_a[4], role_b[4]; /* 0 p, 1 q, 2 s, 3 r */
  int sign;
  int64_t nx, ny;
} zg_term;

/* A piece of work of the block kernel: rows r0 .. r1 - 1, columns
 * j0 .. j1 - 1 of the block of A bucket ba and B class cb (index
 * k = ba * nclassB + cb).  Its thread alone writes there. */
typedef struct {
  int64_t k, r0, r1, j0, j1;
} zg_btask;

struct zg_plan {
  int n, nelec, nsub, nsets, npair;
  int m[ZG_MAX_SUB];   /* spinors of each sub-group */
  int off[ZG_MAX_SUB]; /* its first spinor */
  int side_of[ZG_MAX_SUB];
  /* sub-strings of a sub-group: rank among those of the same number of
   * electrons (increasing as numbers), and the list by that number. */
  uint32_t *rank[ZG_MAX_SUB];
  uint32_t *masks[ZG_MAX_SUB];
  int64_t mstart[ZG_MAX_SUB][ZG_MAX_SUBSIZE + 2];
  int32_t *sublabel[ZG_MAX_SUB]; /* [nsets << m] label sums of a sub-string */
  int mods[ZG_MAX_SETS], targets[ZG_MAX_SETS];
  int ngas;
  int *gas_w; /* [ngas * ZG_MAX_SUB] weight of each sub-group in a constraint */
  int *gas_min, *gas_max;
  zg_side s[2];
  /* For A bucket ba and B class cb: the B bucket that pairs with it, or -1,
   * and where their block starts.  [nbucketA * nclassB] */
  int32_t *match;
  int64_t *blockoff;
  uint8_t *cpair; /* [nclassA * nclassB] whether two classes pair */
  int64_t nblock;
  int64_t dim;
  zg_task *tasks;
  int64_t ntask;
  /* Block kernel: whether each side fits a 64-bit word, the terms, and the
   * pieces of work. */
  int block_ok;
  int nside[2];
  zg_term terms[9];
  zg_btask *btasks;
  int64_t nbtask;
};

/* Block kernel helpers (zgasci.c). */
uint64_t zg_side_word(const zg_plan *p, int side, int64_t sid);
int64_t zg_word_sid(const zg_plan *p, int side, uint64_t w);
/* The x with O(x)^dagger w a valid string: their numbers, the strings and
 * the signs <w|O(x)|source>.  Returns their number (at most cap). */
int zg_enum_ops(int k, const int *kind, const int *rel, int nside, uint64_t w,
                int64_t *x, uint64_t *src, int8_t *sign, int cap);

int64_t zg_binom(int n, int k);

/* Electrons in each sub-group of a side for tuple t. */
static inline void zg_counts(const zg_plan *p, int side, int64_t t,
                             int *counts) {
  const zg_side *s = &p->s[side];
  for (int g = 0; g < s->nsub; ++g) {
    counts[g] = (int)((t / s->stride[g]) % (p->m[s->first + g] + 1));
  }
}

/* Number of a string among those of its tuple, from its sub-strings. */
static inline int64_t zg_address(const zg_plan *p, int side, const int *counts,
                                 const uint32_t *masks) {
  const zg_side *s = &p->s[side];
  int64_t address = 0, radix = 1;
  for (int g = 0; g < s->nsub; ++g) {
    const int sub = s->first + g;
    address += radix * p->rank[sub][masks[g]];
    radix *= zg_binom(p->m[sub], counts[g]);
  }
  return address;
}

/* The sub-strings of string `address` of a tuple with these counts. */
static inline void zg_unpack(const zg_plan *p, int side, const int *counts,
                             int64_t address, uint32_t *masks) {
  const zg_side *s = &p->s[side];
  for (int g = 0; g < s->nsub; ++g) {
    const int sub = s->first + g;
    const int64_t size = zg_binom(p->m[sub], counts[g]);
    masks[g] = p->masks[sub][p->mstart[sub][counts[g]] + address % size];
    address /= size;
  }
}

/* The determinants K + R of the CI space reached from the determinant K of
 * N - 2 electrons given by its sub-strings masks[nsub] (A then B): their
 * pairs R, their places in the CI vector, and the signs of
 * a+_r a+_s |K>, +1 or -1.  Returns their number, at most npair. */
int zg_connect(const zg_plan *p, const uint32_t *masks, int32_t *pair,
               int64_t *addr, int8_t *sign);

/* The sub-strings of the K of N - 2 electrons of task piece `task`: the A
 * string a of tuple ta, the B string b of tuple tb. */
static inline void zg_task_masks(const zg_plan *p, const zg_task *task,
                                 int64_t a, int64_t b, uint32_t *masks) {
  int counts[ZG_MAX_SUB];
  zg_counts(p, 0, task->ta, counts);
  zg_unpack(p, 0, counts, a, masks);
  zg_counts(p, 1, task->tb, counts);
  zg_unpack(p, 1, counts, b, masks + p->s[0].nsub);
}

#endif
