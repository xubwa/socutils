/* The addressing scheme shared by the kernels; see zgasci.h. */
#ifndef ZGASCI_PLAN_H_
#define ZGASCI_PLAN_H_

#include <stdint.h>

#include "zgasci.h"

#define ZG_MAX_SUB 32     /* sub-groups */
#define ZG_MAX_SUBSIZE 16 /* spinors of a sub-group */
#define ZG_MAX_SETS 4     /* label sets */
#define ZG_MAX_SPINOR (ZG_MAX_SUB * ZG_MAX_SUBSIZE)
#define ZG_MAX_ROWS 1024  /* rows of a piece of work of the block kernel */
#define ZG_MAX_COLS 512   /* and its columns */
#define ZG_BLAS_ROWS 256   /* rows of a piece of work of the BLAS kernel */
#define ZG_MAX_TUPLES 100000000 /* tuples of a side (24 bytes each) */

/* One side.  A "tuple" is a choice of the number of electrons in each of
 * its sub-groups, k_g <= kmax of the sub-group, numbered t = sum_g k_g
 * stride_g; the tuples that occur in the CI space are the classes.  The strings of all classes are numbered
 * through (sid); a bucket is the strings of one class with one label. */
typedef struct {
  int nsub; /* its sub-groups are first .. first + nsub - 1 */
  int first;
  int64_t ntuple;
  int64_t stride[ZG_MAX_SUB];
  int *tuple_elec;      /* [ntuple] electrons */
  int64_t *tuple_size;  /* [ntuple] strings */
  int32_t *class_of;    /* [ntuple] class, or -1 */
  int64_t *by_elec;     /* tuples that the constraints allow on this side
                           alone, sorted by their electrons ... */
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

/* A piece of work of the BLAS kernel: rows r0 .. r1 - 1 of a bucket of the
 * row side, with all the blocks they run through. */
typedef struct {
  int32_t bucket;
  int64_t r0, r1;
} zg_rtask;

/* A kind of term as the BLAS kernel sees it: its operators on the row side
 * and on the column side, as in zg_term. */
typedef struct {
  int term;
  int kr, kc;
  int kind_r[4], rel_r[4];
  int kind_c[4], rel_c[4];
  int par_r, par_c; /* whether (-1)^(N_A of the ket) goes with rows, columns */
  int add_r, add_c; /* electrons of the source string minus the target's */
  int64_t wr, wc;   /* W[row index * wr + column index * wc] */
  int64_t ncidx;    /* column indices */
  /* Label filter of the row operators (kinds with at most one column
   * operator, which can make up for few changes of the labels of the row
   * string): a hash table from the change made by all row operators but the
   * last (packed, zg_label_key) to the spinors the last one may take, as a
   * mask of each sub-group of the row side (fmask[slot * fnsub + g]).
   * fsize 0: no filter. */
  int fsize, fnsub;
  uint64_t *fkey;
  uint32_t *fmask;
  /* Product plans: the tables of the row and of the column operators. */
  int pat_r, pat_c;
} zg_bkind;

#define ZG_MAX_PWORDS 4 /* words of a string of a product plan */
#define ZG_MAX_PAT 8    /* operator patterns of a side */

/* Product plans: the operators of one pattern (k operators of these kinds
 * and orders, as in zg_term) on the strings of a side.  For target string
 * w, entries ptr[w] .. ptr[w + 1]: the source string src with
 * O(x)^dagger |w> = sign |src>, and x (~x if sign is -1). */
typedef struct {
  int k, kind[4], rel[4];
  int64_t *ptr;
  int32_t *src;
  int32_t *x;
} zg_ptable;

/* A way of placing the operators of one side in its sub-groups (g, local
 * to the side), with the class of the source strings O^dagger |target>, and
 * what their numbers need: the first of the class, the radix of each
 * sub-group, the sub-groups the operators act in. */
typedef struct {
  int g[4];
  int ntouch, touch[4];
  int32_t cls;
  int64_t base;
  int64_t radix[ZG_MAX_SUB];
} zg_asg;

/* A piece of the transposition of the CI vector: rows i0 .. i1 - 1 of block
 * k = ba * nclassB + cb. */
typedef struct {
  int64_t k, i0, i1;
} zg_tpiece;

struct zg_plan {
  int n, nelec, nsub, nsets, npair;
  int m[ZG_MAX_SUB];   /* spinors of each sub-group */
  int kmax[ZG_MAX_SUB]; /* the most electrons it can hold in the space */
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
  /* Block kernel: whether it applies (always now), the spinors of each
   * side, the terms, and the pieces of work. */
  int block_ok;
  int nside[2];
  zg_term terms[9];
  zg_btask *btasks;
  int64_t nbtask;
  /* BLAS kernel.  Half 0 has the A strings as rows (the vector as it is),
   * half 1 the B strings (the transposed vector: for B bucket bb and A class
   * ca, match_t gives the A bucket and blockoff_t where the block
   * (B string, A string) starts). */
  int32_t *match_t;
  int64_t *blockoff_t;
  zg_bkind bkinds[2][5];
  int nbkind[2];
  zg_rtask *rtasks[2];
  int64_t nrtask[2];
  zg_tpiece *tpieces;
  int64_t ntpiece;
  /* For each row bucket of a half, the column classes it has a block with:
   * pair_cls[h][pair_ptr[h][bucket] .. pair_ptr[h][bucket + 1]]. */
  int64_t *pair_ptr[2];
  int32_t *pair_cls[2];
  /* The distinct labels of the spinors of each side, and which of them
   * each spinor carries (by its number within the side). */
  int ndlab[2];
  int dlab[2][ZG_MAX_SPINOR][ZG_MAX_SETS];
  uint16_t dof[2][ZG_MAX_SPINOR];
  /* (bucket << 32) + place in the bucket of each string of a side. */
  uint64_t *bp[2];
  /* Product plans (zg_plan_product): the strings of each side in sid order
   * (pwords words each) and the operator tables.  The sub-groups are only
   * pseudo-sub-groups there, and a class is an electron count. */
  int product, mirror;
  int pwords[2];
  uint64_t *pstr[2];
  int npat[2];
  zg_ptable pat[2][ZG_MAX_PAT];
};

/* Product plans: appends from n on the operators of table pat of side
 * `side` on string sid (as zg_asg_enum); returns the new n. */
static inline int zg_prod_ops(const zg_plan *p, int side, int pat, int64_t sid,
                              int64_t *x, int64_t *src, int8_t *sign, int n) {
  if (pat < 0) return n;
  const zg_ptable *t = &p->pat[side][pat];
  for (int64_t e = t->ptr[sid]; e < t->ptr[sid + 1]; ++e) {
    const int32_t v = t->x[e];
    x[n] = v >= 0 ? v : ~v;
    sign[n] = v >= 0 ? 1 : -1;
    src[n++] = t->src[e];
  }
  return n;
}

/* Strings as the kernels act on them: the sub-strings of the sub-groups of
 * their side, masks[g] (bit j: spinor j of sub-group g).  zg_side_masks:
 * those of string sid; returns its electrons.  zg_masks_sid: the string, or
 * -1 if its tuple is not a class. */
int zg_side_masks(const zg_plan *p, int side, int64_t sid, uint32_t *masks);
int64_t zg_masks_sid(const zg_plan *p, int side, const uint32_t *masks);
/* Block kernel helper (zgasci.c): the x with O(x)^dagger |w> a string of
 * the space (w given by its masks): their numbers, the strings and the signs
 * <w|O(x)|source>.  Returns their number (at most cap are stored). */
int zg_enum_ops(const zg_plan *p, int side, int k, const int *kind,
                const int *rel, const uint32_t *masks, int64_t *x,
                int64_t *sid, int8_t *sign, int cap);

/* BLAS kernel helpers (zgasci.c).  zg_asg_list: the placements of k
 * operators (kind, rel as in zg_term) in the sub-groups of a side for which
 * O^dagger takes the strings of class `from` to a class marked in ok (any
 * class of the space if ok is NULL); returns their number (at most cap).
 * zg_asg_enum: for one placement and the target string w (its masks, and
 * rk: the ranks of its sub-strings, zg_string_ranks), appends from n on the
 * operators: their numbers x, the source strings and the signs
 * <w|O(x)|source>; returns the new n.  With a filter (the kind whose row
 * operators these are) only the operators that its label filter lets
 * through. */
int zg_asg_list(const zg_plan *p, int side, int k, const int *kind,
                const int *rel, int from, const uint8_t *ok, zg_asg *out,
                int cap);
void zg_string_ranks(const zg_plan *p, int side, const uint32_t *masks,
                     int64_t *rk);
int zg_asg_enum(const zg_plan *p, int side, int k, const int *kind,
                const int *rel, const zg_asg *a, const uint32_t *masks,
                const int64_t *rk, const zg_bkind *filter, int64_t *x,
                int64_t *sid, int8_t *sign, int n);

int64_t zg_binom(int n, int k);

/* Electrons in each sub-group of a side for tuple t. */
static inline void zg_counts(const zg_plan *p, int side, int64_t t,
                             int *counts) {
  const zg_side *s = &p->s[side];
  for (int g = 0; g < s->nsub; ++g) {
    counts[g] = (int)((t / s->stride[g]) % (p->kmax[s->first + g] + 1));
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
