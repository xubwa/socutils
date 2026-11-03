'''
Blocked tensors and the commutator [O, T]_{0,1,2} of MR-DSRG theory in a spin-orbital /
spinor basis, in the generalized normal order of a multi-determinant (ensemble)
reference.

Storage convention of this module: every operator is held as
X[annihilated indices..., created indices...], i.e. sum_pq X[q, p] p^+ q and
1/4 sum X[r, s, p, q] p^+ q^+ s r; amplitudes are T1[i, a] (a^+ i) and
T2[i, j, a, b].  For a Hermitian operator this is the complex conjugate of the
matrix <p|x|q> or <pq||rs>.  In this layout the expressions of the spin-orbital
MR-DSRG code of forte (MRDSRG_SO::H?_T?_C?, mrdsrg_so.cc, LGPL) hold literally for
complex tensors, with Gamma1[p, q] = <p^+ q>, Eta1 = 1 - Gamma1 and
Lambda2[p, q, r, s] the cumulant of <p^+ q^+ s r>.

Index classes of the contraction strings (as in forte; they differ from the
letters of conventions.md, which this module does not use):
    i j k l  hole (core + active)      a b c d  particle (active + virtual)
    p q r s  general                   m n      core
    u v w x y z  active                e f      virtual
Blocks are labelled by the spaces of their indices, c / a / v.
'''

import collections
import time

import numpy
from pyscf import lib

from itertools import product

from socutils.symm import double_group as dg
from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la

_CLASS = {}
for _letters, _spaces in (("ijkl", "ca"), ("abcd", "av"), ("pqrs", "cav"), ("mn", "c"),
                          ("uvwxyz", "a"), ("ef", "v")):
    for _ch in _letters:
        _CLASS[_ch] = _spaces

PROFILE = None  # set to a dict to time every block contraction
EYE = "identity"  # marker for an identity block (core block of Gamma1, virtual of Eta1)


class Blocks(dict):
    """Blocked tensor: space label -> array.  Missing blocks are zero."""

    def norm2(self):
        return sum(numpy.vdot(v, v).real for v in self.values())


_RANK = {"c": 0, "a": 1, "v": 2}


def canonical(lab):
    """Canonical placement of a two-body label (spaces sorted inside each index pair):
    (label, permutation taking the canonical block to `lab`, sign)."""
    perm, sign = [0, 1, 2, 3], 1.0
    if _RANK[lab[0]] > _RANK[lab[1]]:
        perm[0], perm[1] = 1, 0
        sign = -sign
    if _RANK[lab[2]] > _RANK[lab[3]]:
        perm[2], perm[3] = 3, 2
        sign = -sign
    return "".join(lab[k] for k in perm), tuple(perm), sign


def is_canonical(lab):
    return _RANK[lab[0]] <= _RANK[lab[1]] and _RANK[lab[2]] <= _RANK[lab[3]]


class AntiBlocks(Blocks):
    """Antisymmetric two-body operator holding the canonical placements only; `get`
    returns (view, sign) for any placement."""

    def get(self, lab):
        clab, perm, sign = canonical(lab)
        blk = dict.get(self, clab)
        if blk is None:
            return None
        if clab == lab:
            return blk, 1.0
        return blk.transpose(perm), sign


class HermitianHP(AntiBlocks):
    """Hermitian two-body operator with hole-particle blocks only, holding the canonical
    excitation blocks X[ij, ab]; a de-excitation block is the conjugate transpose,
    X[ab, ij] = X[ij, ab]^*, returned as (view, sign, conjugate)."""

    def get(self, lab):
        if all(x in "ca" for x in lab[:2]) and all(x in "av" for x in lab[2:]):
            return AntiBlocks.get(self, lab)
        x = AntiBlocks.get(self, lab[2:] + lab[:2])
        if x is None:
            return None
        return x[0].transpose(2, 3, 0, 1), x[1], True

    @classmethod
    def from_blocks(cls, C):
        """The hole-particle part of C + C^+ (C: AntiBlocks, emptied)."""
        out = cls()
        labs = set()
        for lab in C:
            for l in (lab, lab[2:] + lab[:2]):
                if (all(x in "ca" for x in l[:2]) and all(x in "av" for x in l[2:])
                        and l != "aaaa"):
                    labs.add(canonical(l)[0])
        for lab in sorted(labs):
            blk = materialize(C.get(lab))
            de = materialize(C.get(lab[2:] + lab[:2]))
            if de is not None:
                de = de.conj().transpose(2, 3, 0, 1)
                blk = de.copy() if blk is None else blk + de
            else:
                blk = numpy.array(blk)
            out[lab] = blk
            # release the consumed blocks of C
            C.pop(lab, None)
            C.pop(canonical(lab[2:] + lab[:2])[0], None)
        return out


class Dims(dict):
    """Dimensions of the spaces c, a, v, and their double-group sectors:
    sym = (doublegroupname, {space: irrep id of every orbital}) or None."""

    def __init__(self, d, sym=None):
        super().__init__(d)
        self.sym = sym
        # plans of the symmetry-blocked products (_pair_symmetric), by index pattern
        self.plans = {}


# Which indices of an operand are created (+1) and annihilated (-1): an element is
# nonzero only if the product of the irreps of its orbitals, the annihilated ones
# conjugated, is totally symmetric.  Operators X[annihilated..., created...] and
# amplitudes are ONE_BODY / TWO_BODY; Gamma1[p, q] = <p^+ q>, Eta1, Lambda2, Lambda3 are
# DENSITY, CUMULANT2, CUMULANT3.  An operand gives its own as the attribute `signs`
# (None: not known, no symmetry is used with it); without it the rank decides.
ONE_BODY = (-1, 1)
TWO_BODY = (-1, -1, 1, 1)
DENSITY = (1, -1)
CUMULANT2 = (1, 1, -1, -1)
CUMULANT3 = (1, 1, 1, -1, -1, -1)
_DEFAULT_SIGNS = {2: ONE_BODY, 4: TWO_BODY}


def _operand_signs(op, rank):
    if hasattr(op, "signs"):
        return op.signs
    return _DEFAULT_SIGNS.get(rank)


def _times(ir, c, group):
    """The irreps ir taken c times (c a nonzero integer; negative: conjugated)."""
    base = ir if c > 0 else numpy.asarray(dg.conj(ir, group))
    if abs(c) == 1:
        return base
    out = numpy.empty_like(base)
    for u in numpy.unique(base):
        v = 0
        for _ in range(abs(c)):
            v = dg.direct_prod(v, int(u), group)
        out[base == u] = v
    return out


# products smaller than this (na * nc * nb) are done dense: the blocking does not pay
# (Br2 cc-pVQZ, Dooh*: the general terms of DSRG-MRPT3 with 1 << 18 took 3x longer)
SYMMETRY_MIN_SIZE = 1 << 28


def _sector_groups(letters, coefs, secs, group):
    """{irreps: flat indices} of the multi-index over letters (C order), grouped by the
    irreps sum_l coefs[k][l] Gamma_l for every coefficient dict of coefs; built from
    the sectors of each letter, without sorting the multi-index."""
    if not letters:
        return {tuple(0 for _ in coefs): numpy.zeros(1, dtype=numpy.int64)}
    per = []
    for l in letters:
        sec = numpy.asarray(secs[l])
        per.append([(int(u), numpy.nonzero(sec == u)[0]) for u in numpy.unique(sec)])
    shape = tuple(numpy.asarray(secs[l]).size for l in letters)
    out = {}
    for combo in product(*per):
        key = []
        for coef in coefs:
            v = 0
            for l, (u, _) in zip(letters, combo):
                c = coef.get(l, 0)
                if c:
                    v = dg.direct_prod(v, int(_times(numpy.asarray([u]), c, group)[0]), group)
            key.append(int(v))
        idx = numpy.ravel_multi_index(numpy.ix_(*(ix for _, ix in combo)), shape).ravel()
        out.setdefault(tuple(key), []).append(idx)
    return {k: numpy.sort(numpy.concatenate(v)) for k, v in out.items()}


def _symmetry_plan(aown, con, bown, sa, sb, secs, group):
    """(rows, cols, right) index arrays of the blocks of at (rows, cols) and bt (cols,
    right) that do not vanish by symmetry, one per pair of irreps of the contracted
    multi-index; the blocked product is the sum of their products."""
    cgroups = _sector_groups(con, (sa, sb), secs, group)
    agroups = _sector_groups(aown, (sa,), secs, group)
    bgroups = _sector_groups(bown, (sb,), secs, group)
    plan = []
    for (s1, t1), cols in cgroups.items():
        rows = agroups.get((dg.conj(s1, group),))
        right = bgroups.get((dg.conj(t1, group),))
        if rows is not None and right is not None:
            plan.append((rows, cols, right))
    return plan


def _pair_symmetric(at, bt, plan):
    """at (na, nc) times bt (nc, nb) as the sum of the products of the blocks of plan
    (gathered and added back with the threaded pyscf.lib.take_2d / takebak_2d)."""
    at = numpy.asarray(at, dtype=numpy.complex128, order="C")
    bt = numpy.asarray(bt, dtype=numpy.complex128, order="C")
    r = numpy.zeros((at.shape[0], bt.shape[1]), dtype=numpy.complex128)
    for rows, cols, right in plan:
        blk = la.gemm(lib.take_2d(at, rows, cols), lib.take_2d(bt, cols, right))
        lib.takebak_2d(r, blk, rows, right)
    return r


def blocked_gemm(a, b, rows, mid, cols):
    """a @ b for a (m, k) nonzero only where rows[i] == mid[j] and b (k, n) only where
    mid[j] == cols[l] (irreps of the rows, the contracted index and the columns): one
    product per irrep.  With rows None the plain product."""
    if rows is None:
        return la.gemm(a, b)
    a = numpy.asarray(a, dtype=numpy.complex128, order="C")
    b = numpy.asarray(b, dtype=numpy.complex128, order="C")
    r = numpy.zeros((a.shape[0], b.shape[1]), dtype=numpy.complex128)
    for ir in numpy.unique(mid):
        i = numpy.nonzero(rows == ir)[0]
        j = numpy.nonzero(mid == ir)[0]
        l = numpy.nonzero(cols == ir)[0]
        if i.size and l.size:
            lib.takebak_2d(r, la.gemm(lib.take_2d(a, i, j), lib.take_2d(b, j, l)), i, l)
    return r


def _combine_signs(sa, sb, con):
    """Signs of the product of two operands with signs sa, sb (dicts letter -> +-1)
    summed over con: the constraints of both, the contracted letters eliminated; None
    if they cannot be eliminated together."""
    if sa is None or sb is None:
        return None
    if not con:
        return None
    c0 = con[0]
    if not sb.get(c0):
        return None
    ratio = -sa.get(c0, 0) / sb[c0]
    if ratio not in (1.0, -1.0):
        return None
    out = {}
    for l in set(sa) | set(sb):
        v = sa.get(l, 0) + ratio * sb.get(l, 0)
        if l in con:
            if v != 0:
                return None
            continue
        if v:
            out[l] = int(v)
    return out


def with_signs(blocks, signs):
    """blocks (a mapping label -> array) as Blocks with the given signs."""
    out = Blocks(blocks)
    out.signs = signs
    return out


def _record_signs(out, signs):
    # the signs of an output are those of all its contributions; None if they differ
    if "signs" not in out.__dict__:
        out.signs = signs
    elif out.signs != signs:
        out.signs = None


def _is_plain(a):
    return a.dtype == numpy.complex128 and a.flags.c_contiguous


def _pair(asub, a, bsub, b, keepers, sa=None, sb=None, secs=None, group=None, plans=None,
          spaces=None):
    """Contraction of two arrays over the letters they share that are not in
    `keepers`: (subscripts, array, signs).  One threaded matrix product between threaded
    transpositions (zmr.lib); numpy.einsum for traces and outer products.  With the
    signs sa, sb of the operands (dicts letter -> coefficient) and the irreps secs of
    the orbitals of every letter, the product skips the blocks that vanish by symmetry
    and the signs of the result are returned (None if not known)."""
    con = [x for x in asub if x in bsub and x not in keepers]
    aown = [x for x in asub if x not in con]
    bown = [x for x in bsub if x not in con]
    if (not con or len(set(asub)) != len(asub) or len(set(bsub)) != len(bsub)
            or set(aown) & set(bown)):
        osub = "".join(dict.fromkeys(x for x in asub + bsub if x in keepers))
        return osub, numpy.einsum(asub + "," + bsub + "->" + osub, a, b, optimize=True), None
    pa = [asub.index(x) for x in aown + con]
    pb = [bsub.index(x) for x in con + bown]
    at = a if (pa == list(range(a.ndim)) and _is_plain(a)) else kernels.transposed(a, pa)
    bt = b if (pb == list(range(b.ndim)) and _is_plain(b)) else kernels.transposed(b, pb)
    na = int(numpy.prod(at.shape[:len(aown)], dtype=numpy.int64))
    nc = int(numpy.prod(at.shape[len(aown):], dtype=numpy.int64))
    nb = int(bt.size // max(nc, 1))
    symmetric = (group is not None and sa is not None and sb is not None
                 and na * nc * nb >= SYMMETRY_MIN_SIZE)
    if at.size == 0 or bt.size == 0:
        r = numpy.zeros(at.shape[:len(aown)] + bt.shape[len(con):], dtype=numpy.complex128)
    elif symmetric:
        key = (tuple(aown), tuple(con), tuple(bown),
               tuple(sorted(sa.items())), tuple(sorted(sb.items())),
               tuple(spaces[x] for x in aown + con + bown))
        plan = plans.get(key)
        if plan is None:
            plan = plans[key] = _symmetry_plan(aown, con, bown, sa, sb, secs, group)
        r = _pair_symmetric(at.reshape(na, nc), bt.reshape(nc, -1), plan)
        r = r.reshape(at.shape[:len(aown)] + bt.shape[len(con):])
    else:
        r = la.gemm(at.reshape(na, nc), bt.reshape(nc, -1))
        r = r.reshape(at.shape[:len(aown)] + bt.shape[len(con):])
    return "".join(aown + bown), r, _combine_signs(sa, sb, con)


def contract(spec, *ops, dims, out=None, scale=1.0, keep=None, fix=""):
    """Blocked einsum.

    `spec` is an einsum string whose letters carry their index class; every operand is a
    mapping label -> array with a `get` method (an array, the marker EYE, (array, sign)
    or (array, sign, conjugate)).  The sum runs over all assignments of orbital spaces
    to the letters for which every operand block and the output block exist.  With
    `out` a mapping, blocks are created on first use, unless `keep(label)` is false; a
    scalar is returned for an empty output string.  The letters in `fix` run over the
    active space only.  The operands are contracted pairwise from left to right.
    """
    lhs, rhs = spec.split("->")
    subs = lhs.split(",")
    letters = sorted(set("".join(subs)))
    total = 0.0
    sym = getattr(dims, "sym", None)
    group, sectors = sym if sym is not None else (None, None)
    for spaces in product(*("a" if ch in fix else _CLASS[ch] for ch in letters)):
        sp = dict(zip(letters, spaces))
        olab = "".join(sp[ch] for ch in rhs)
        if rhs and keep is not None and olab not in out and not keep(olab):
            continue
        arrays, asubs, signs = [], [], []
        ok = True
        fac = scale
        cj = False
        conj_ops = set()
        rename = {}
        for sub, op in zip(subs, ops):
            blk = op.get("".join(sp[ch] for ch in sub))
            if blk is None:
                ok = False
                break
            if isinstance(blk, tuple):
                if len(blk) == 3 and blk[2]:
                    cj = True
                    conj_ops.add(len(arrays))
                blk, sg = blk[0], blk[1]
                fac *= sg
            if blk is EYE:
                x, y = sub
                x, y = rename.get(x, x), rename.get(y, y)
                if x == y:
                    continue
                # delta_xy: drop the letter that is summed over
                gone, stay = (x, y) if x not in rhs else (y, x)
                if gone in rhs:
                    arrays.append(numpy.eye(dims[sp[sub[0]]]))
                    asubs.append(x + y)
                    signs.append(_operand_signs(op, 2))
                    continue
                for k in list(rename):
                    if rename[k] == gone:
                        rename[k] = stay
                rename[gone] = stay
                continue
            arrays.append(blk)
            asubs.append(sub)
            signs.append(_operand_signs(op, len(sub)))
        if not ok:
            continue
        asubs = ["".join(rename.get(ch, ch) for ch in s) for s in asubs]
        secs = None
        if group is not None:
            secs = {ch: sectors[sp[ch]] for ch in letters}
            sdicts = []
            for sub_, sg_ in zip(asubs, signs):
                if sg_ is None:
                    sdicts.append(None)
                    continue
                d = {}
                for ch, c in zip(sub_, sg_):
                    d[ch] = d.get(ch, 0) + c
                sdicts.append(d)
        else:
            sdicts = [None] * len(asubs)
        if PROFILE is not None:
            _t = time.time()
        if cj:
            # einsum(conj(A), B) = conj(einsum(A, conj(B)))
            arrays = [x if k in conj_ops else numpy.conj(x) for k, x in enumerate(arrays)]
        csub, val, csgn = asubs[0], arrays[0], sdicts[0]
        for k in range(1, len(arrays)):
            later = set(rhs).union(*asubs[k + 1:])
            csub, val, csgn = _pair(csub, val, asubs[k], arrays[k], later,
                                    csgn, sdicts[k], secs, group,
                                    getattr(dims, "plans", None), sp)
        if PROFILE is not None:
            key = spec + " " + ",".join("".join(sp[ch] for ch in sub) for sub in subs)
            PROFILE[key] = PROFILE.get(key, 0.0) + time.time() - _t
        if not rhs:
            v = numpy.einsum(csub + "->", val) if csub else val
            total += fac * (numpy.conj(v) if cj else v)
            continue
        if len(set(csub)) != len(csub) or set(csub) != set(rhs):
            val = numpy.einsum(csub + "->" + rhs, val)
            csub = rhs
            csgn = None
        if cj:
            val = numpy.conj(val)
        perm = [csub.index(x) for x in rhs]
        if olab not in out:
            out[olab] = numpy.zeros([val.shape[p] for p in perm], dtype=numpy.complex128)
        if group is not None:
            _record_signs(out, None if csgn is None else tuple(csgn.get(x, 0) for x in rhs))
        kernels.transpose_add(out[olab], numpy.asarray(val, dtype=numpy.complex128)
                              if val.dtype != numpy.complex128 else val, perm, alpha=fac)
    return total


# placements of an antisymmetric Hermitian two-body block C: X = sign * C.transpose(perm),
# complex conjugated where marked (V[rs, pq] = conj(V[pq, rs]))
_PLACEMENTS = (((0, 1, 2, 3), 1.0, False), ((1, 0, 2, 3), -1.0, False),
               ((0, 1, 3, 2), -1.0, False), ((1, 0, 3, 2), 1.0, False),
               ((2, 3, 0, 1), 1.0, True), ((3, 2, 0, 1), -1.0, True),
               ((2, 3, 1, 0), -1.0, True), ((3, 2, 1, 0), 1.0, True))


def hermitian_canonical(lab):
    """(canonical label, perm, sign, conjugate) with block lab = sign * (conj)
    canonical.transpose(perm), the canonical label the largest of the placements (virtual
    indices first: V[vv, vx] is the placement the commutator mostly reads)."""
    orbit = ["".join(lab[k] for k in perm) for perm, _, _ in _PLACEMENTS]
    clab = max(orbit, key=lambda x: [_RANK[c] for c in x])
    for perm, sign, cj in _PLACEMENTS:
        if "".join(clab[k] for k in perm) == lab:
            return clab, perm, sign, cj


def materialize(x):
    """A block as returned by get ((array, sign) or (array, sign, conjugate) included) as
    an array."""
    if x is None or isinstance(x, numpy.ndarray):
        return x
    blk, sign = x[0], x[1]
    if len(x) == 3 and x[2]:
        blk = blk.conj()
    return sign * blk


class ThreeIndexFile:
    """Three-index integrals B[P, p, q] of eris in a temporary HDF5 file, read in chunks
    of P, in the conjugated form of this module: read(pq) = conj(B[pq]).  Only the pairs
    of spaces in the order c, a, v are stored, as they are; B[qp] = conj(B[pq])^T (real
    Cholesky vectors of a symmetric metric), so the other pairs are transposed views."""

    def __init__(self, eris, spaces="cav"):
        self.file = lib.H5TmpFile()
        for x, y in product(spaces, repeat=2):
            if _RANK[x] <= _RANK[y]:
                self.file.create_dataset(x + y, data=eris.cd_block(x + y))
                # the copy kept by eris is not needed
                getattr(eris, "cd", {}).pop(x + y, None)
        self.naux = self.file["cc"].shape[0]

    def shape(self, lab):
        x, y = lab
        n = self.file[x + x].shape[1], self.file[y + y].shape[1]
        return (self.naux,) + n

    def read(self, lab, p0, p1, cols=slice(None)):
        """conj(B[lab])[p0:p1, :, cols]"""
        x, y = lab
        if _RANK[x] <= _RANK[y]:
            return numpy.conj(self.file[lab][p0:p1, :, cols])
        # conj(B[yx]^*T) = B[xy]^T: no conjugation
        return self.file[y + x][p0:p1, cols, :].transpose(0, 2, 1)

    def __getitem__(self, lab):
        return self.read(lab, 0, self.naux)

    def step(self, nbytes_per_p, max_memory):
        return max(1, min(self.naux, int(max_memory * 1e6 / max(nbytes_per_p, 1))))

    def chunks(self, labs, max_memory):
        """{lab: conj(B[lab])[P0:P1]} for consecutive chunks of P within max_memory (MB)."""
        per_p = sum(16.0 * numpy.prod(self.shape(lab)[1:]) for lab in labs)
        step = self.step(per_p, max_memory)
        for p0 in range(0, self.naux, step):
            yield {lab: self.read(lab, p0, p0 + step) for lab in labs}


class BareV:
    """Two-electron integrals in the storage convention, block by block from the
    three-index tensors: V[pq, rs] = (pr|qs) - (ps|qr) with conjugated coefficients.

    B: ThreeIndexFile.  Only one placement of every block is built (the others are views with a sign and a
    complex conjugation, see hermitian_canonical); the most recently used blocks are kept
    up to cache_memory (MB) and the others rebuilt when asked for.  The block with four
    virtual indices is only built when `dense_vvvv` is set (small systems); otherwise it
    is None here and the solver uses the pair-packed rows of `vvvv_rows`.
    """

    def __init__(self, B, dense_vvvv=False, cache_memory=4000, extra=None, read_memory=500):
        self.B = B
        # MB of three-index integrals read at a time
        self.read_memory = read_memory
        self.timings = collections.Counter()
        # blocks added to the factorized integrals (e.g. a correction of 'aaaa'), by
        # canonical label
        self.extra = extra or {}
        self.dense_vvvv = dense_vvvv
        self.cache_memory = cache_memory
        self.cache = collections.OrderedDict()
        self.nbuild = collections.Counter()
        # blocks with three virtual indices left to the caller (vvvx_chunks)
        self.skip_three_virtual = False

    def build(self, lab):
        p, q, r, s = lab
        exchange = not (r == s or p == q)
        labs = {p + r, q + s} | ({p + s, q + r} if exchange else set())
        v = 0.0
        for b in self.B.chunks(labs, self.read_memory):
            v = v + numpy.einsum("Ppr,Pqs->pqrs", b[p + r], b[q + s], optimize=True)
            if exchange:
                v -= numpy.einsum("Pps,Pqr->pqrs", b[p + s], b[q + r], optimize=True)
        if r == s:
            v -= v.swapaxes(2, 3)
        elif p == q:
            v -= v.swapaxes(0, 1)
        if lab in self.extra:
            v += self.extra[lab]
        return v

    def _canonical_block(self, clab):
        if clab in self.cache:
            self.cache.move_to_end(clab)
            return self.cache[clab]
        v = self.build(clab)
        self.nbuild[clab] += 1
        limit = self.cache_memory * 1e6
        used = sum(x.nbytes for x in self.cache.values())
        while self.cache and used + v.nbytes > limit:
            used -= self.cache.popitem(last=False)[1].nbytes
        if v.nbytes <= limit:
            self.cache[clab] = v
        return v

    def get(self, lab):
        if lab == "vvvv" and not self.dense_vvvv:
            return None
        if self.skip_three_virtual and lab.count("v") == 3:
            return None
        clab, perm, sign, cj = hermitian_canonical(lab)
        c = self._canonical_block(clab)
        if perm == (0, 1, 2, 3):
            return c
        # (view, sign, complex conjugate): no copy is made, the contraction applies sign
        # and conjugation to the other factors
        return c.transpose(perm), sign, cj

    def vvvx_chunks(self, x, max_memory, sym=None):
        """V[ab, ek] (a, b, e virtual, k in the space x) in chunks of k within max_memory
        (MB, the chunk and four copies of its size for the caller): yields (k0, k1,
        block[a, b, e, k]).  sym = (doublegroupname, irreps of the virtual orbitals,
        irreps of the orbitals of x): only the products of the transition irreps
        conj(e) a, conj(b) k whose product is totally symmetric are formed."""
        B = self.B
        naux, nv, nx = B.shape("v" + x)
        bvv = B.file["vv"]
        if sym is not None:
            group, irv, irx = (numpy.asarray(y) if k else y for k, y in enumerate(sym))
            # D = conj(<ab|ek>) = conj((ae|bk)) (both factors read conjugated): rows (a, e)
            # of transition irrep conj(a) e, columns (b, k) of conj(b) k
            t_ae = numpy.asarray(dg.direct_prod(dg.conj(irv, group), irv, group))
            t_bk = numpy.asarray(dg.direct_prod(dg.conj(irv, group), irx, group))
        step = max(1, min(nx, int(max_memory * 1e6 / (5 * 16.0 * nv**3))))
        pstep = self.B.step(16.0 * nv * (nv + step), self.read_memory)
        for k0 in range(0, nx, step):
            k1 = min(nx, k0 + step)
            d = numpy.zeros((nv * nv, nv * (k1 - k0)), dtype=complex)
            if sym is not None:
                rl = t_ae.reshape(-1)
                rr = t_bk[:, k0:k1].reshape(-1)
                blocks = []
                for t_ in numpy.unique(rl):
                    cl = numpy.nonzero(rl == t_)[0]
                    cr = numpy.nonzero(rr == dg.conj(int(t_), group))[0]
                    if cl.size and cr.size:
                        blocks.append((cl, cr, numpy.zeros((cl.size, cr.size), dtype=complex)))
            for p0 in range(0, naux, pstep):
                t = time.time()
                left = bvv[p0:p0 + pstep].reshape(-1, nv * nv)            # P, (a e), unconjugated
                right = numpy.ascontiguousarray(B.read("v" + x, p0, p0 + pstep, slice(k0, k1)))
                right = right.reshape(left.shape[0], -1)                  # P, (b k)
                self.timings["read"] += time.time() - t
                t = time.time()
                if sym is None:
                    la.gemm(left, right, opa="C", out=d, beta=1.0)        # conj(B[vv])^T B[vx]
                else:
                    left = numpy.ascontiguousarray(left)
                    prow = numpy.arange(left.shape[0])
                    for cl, cr, blk in blocks:
                        la.gemm(lib.take_2d(left, prow, cl), lib.take_2d(right, prow, cr),
                                opa="C", out=blk, beta=1.0)
                self.timings["vvvx gemm"] += time.time() - t
            if sym is not None:
                for cl, cr, blk in blocks:
                    lib.takebak_2d(d, blk, cl, cr)
                del blocks
            t = time.time()
            d = d.reshape(nv, nv, nv, k1 - k0)                                    # a, e, b, k
            # V[ab, ek] = (ae|bk) - (ak|be) = d[a, e, b, k] - d[b, e, a, k]
            v = kernels.transposed(d, (0, 2, 1, 3))
            kernels.transpose_add(v, d, (2, 0, 1, 3), alpha=-1.0)
            del d
            self.timings["vvvx antisym"] += time.time() - t
            yield k0, k1, v

    def vvvv_rows(self, batch=8, sym=None):
        """V[ab, rs] for a < b, r < s, built from the three-index tensors: yields (first
        row, rows) with the rows of `batch` values of a at a time (pairs in the order of
        numpy.triu_indices).  sym = (doublegroupname, irreps of the virtual orbitals):
        only the products of the transition irreps conj(a) r, conj(b) s whose product is
        totally symmetric are formed."""
        naux, nv, _ = self.B.shape("vv")
        bvv = self.B.file["vv"]                    # unconjugated
        tim = self.timings
        row = 0
        if sym is not None:
            group, irv = sym
            irv = numpy.asarray(irv)
            # transition irreps conj(p) q of the pairs (p, q) of virtual orbitals
            trans = numpy.asarray(dg.direct_prod(dg.conj(irv, group), irv, group))
            # irreps of the pairs p < q
            pirr = numpy.asarray(dg.direct_prod(irv, irv, group))[numpy.triu_indices(nv, 1)]
        else:
            # D of the largest batch, reused (a new array per batch costs its page faults)
            dbuf = numpy.empty(min(batch, nv - 1) * nv * nv * nv, dtype=complex)
        for a0 in range(0, nv - 1, batch):
            a1 = min(a0 + batch, nv - 1)
            nb, nr = a1 - a0, nv - a0
            # D[a, r, b, s] = sum_P B[P, a, r] B[P, b, s] for b >= a0 only (b > a is
            # kept), one pass over B[vv]; V = conj(D - D^T(r, s))
            pstep = self.B.step(16.0 * nr * nv, self.read_memory)
            if sym is None:
                d = dbuf[:nb * nv * nr * nv].reshape(nb * nv, nr * nv)
            else:
                # the blocks of D allowed by symmetry, one after the other in packed:
                # (a, r) of transition irrep t times (b, s) of irrep conj(t)
                t = time.time()
                tl = trans[a0:a1].reshape(-1)       # (a, r)
                tr = trans[a0:].reshape(-1)         # (b, s)
                blocks, rowoff, coloff, size = [], numpy.zeros(tl.size, dtype=numpy.int64), \
                    numpy.zeros(tr.size, dtype=numpy.int64), 0
                for t_ in numpy.unique(tl):
                    cl = numpy.nonzero(tl == t_)[0]
                    cr = numpy.nonzero(tr == dg.conj(int(t_), group))[0]
                    if cl.size and cr.size:
                        rowoff[cl] = size + numpy.arange(cl.size) * cr.size
                        coloff[cr] = numpy.arange(cr.size)
                        blocks.append((cl, cr, size))
                        size += cl.size * cr.size
                packed = numpy.empty(size, dtype=complex)
                tim["vvvv fill"] += time.time() - t
            for p0 in range(0, naux, pstep):
                t = time.time()
                x = bvv[p0:p0 + pstep, a0:]
                tim["read"] += time.time() - t
                t = time.time()
                right = x.reshape(x.shape[0], -1)
                left = numpy.ascontiguousarray(x[:, :nb]).reshape(x.shape[0], -1)
                # the first chunk of P overwrites D, the others add to it
                beta = 0.0 if p0 == 0 else 1.0
                if sym is None:
                    la.gemm(left, right, opa="T", out=d, beta=beta)
                else:
                    rows = numpy.arange(x.shape[0])
                    for cl, cr, off in blocks:
                        la.gemm(lib.take_2d(left, rows, cl), lib.take_2d(right, rows, cr),
                                opa="T", out=packed[off:off + cl.size * cr.size].reshape(
                                    cl.size, cr.size), beta=beta)
                tim["vvvv gemm"] += time.time() - t
            t = time.time()
            if sym is None:
                w = kernels.vvvv_pair_rows(d.reshape(nb, nv, nr, nv), a0)
                del d
            else:
                w = kernels.vvvv_pair_rows_packed(packed, rowoff, coloff, nb, nv, a0, pirr)
                del packed
            tim["vvvv gather"] += time.time() - t
            yield row, w
            row += len(w)

    @staticmethod
    def vvvv_batch(nv, max_memory):
        """Values of a per block of vvvv_rows within max_memory (MB)."""
        per_a = 16.0 * nv**3 * 1.5 / 1e6
        return max(1, min(nv - 1, int(max_memory / per_a)))


class DenseV:
    """Two-electron integrals in the storage convention from the four-index blocks of
    an eris object without three-index integrals (small tests)."""

    dense_vvvv = True

    def __init__(self, eris, extra=None):
        self.eris = eris
        self.extra = extra or {}
        self.cache = {}

    def get(self, lab):
        if lab not in self.cache:
            p, q, r, s = lab
            v = (self.eris.eri_block(p + r + q + s).transpose(0, 2, 1, 3)
                 - self.eris.eri_block(p + s + q + r).transpose(0, 2, 3, 1))
            v = numpy.conj(v)
            if lab in self.extra:
                v = v + self.extra[lab]
            self.cache[lab] = v
        return self.cache[lab]


def _scalar(O1, O2, T1, T2, G1, E1, L2, L3, con):
    """Fully contracted part of [O, T] (MRDSRG_SO::H?_T?_C0)."""
    C0 = 0.0
    tmp = Blocks()
    con("ia,ab,ji->jb", T1, E1, G1, out=tmp)
    C0 += con("jb,bj->", tmp, O1)

    tmp = Blocks()
    con("evxy,ue->uvxy", O2, T1, out=tmp)
    con("uvmy,mx->uvxy", O2, T1, out=tmp, scale=-1.0)
    con("ex,uvey->uvxy", O1, T2, out=tmp)
    con("vm,umxy->uvxy", O1, T2, out=tmp, scale=-1.0)
    C0 += 0.5 * con("uvxy,xyuv->", tmp, L2)

    C0 += 0.25 * con("efmn,mnef->", O2, T2)
    t1 = Blocks()
    con("ijux,ki,lj->klux", T2, G1, G1, out=t1)
    t2 = Blocks()
    con("klux,uv,xy->klvy", t1, E1, E1, out=t2)
    C0 += 0.25 * con("vykl,klvy->", O2, t2)
    t1 = Blocks()
    con("ijue,ki,lj->klue", T2, G1, G1, out=t1)
    t2 = Blocks()
    con("klue,uv->klve", t1, E1, out=t2)
    C0 += 0.5 * con("vekl,klve->", O2, t2)
    t2 = Blocks()
    con("fexu,yvef->yvxu", O2, T2, out=t2, scale=-1.0)
    C0 += 0.25 * con("yvxu,xy,uv->", t2, G1, G1)
    t2 = Blocks()
    con("femu,mvef->vu", O2, T2, out=t2, scale=-1.0)
    C0 += 0.5 * con("vu,uv->", t2, G1)

    t2 = Blocks()
    t1 = Blocks()
    con("uvkl,ki,lj->uvij", O2, G1, G1, out=t1)
    con("uvij,ijxy->uvxy", t1, T2, out=t2, scale=0.125)
    t1 = Blocks()
    con("uvab,ac,bd->uvcd", T2, E1, E1, out=t1)
    con("uvcd,cdxy->uvxy", t1, O2, out=t2, scale=0.125)
    t1 = Blocks()
    con("iuay,ji,ab->juby", T2, G1, E1, out=t1)
    con("vbjx,juby->uvxy", O2, t1, out=t2)
    C0 += con("uvxy,xyuv->", t2, L2)

    if L3 is not None:
        t1 = Blocks()
        con("uviz,iwxy->uvwxyz", O2, T2, out=t1)
        con("waxy,uvaz->uvwxyz", O2, T2, out=t1)
        C0 += 0.25 * con("uvwxyz,xyzuvw->", t1, L3)
    return C0


def commutator(O1, O2, T1, T2, cum, dims, alpha, keep2, ladder=None, timings=None,
               scalar=True, active=False, keep1=None, large=None):
    """alpha * [O, T] -> (C0, C1, C2), all blocks (C2 restricted by keep2).

    Literal transcription of MRDSRG_SO::H?_T?_C? of forte.  `ladder`, if given, adds the
    particle-particle ladder of the four-virtual block of O2 (which is then absent from
    O2 itself).  scalar=False skips C0; active=True builds the active blocks of C1 and C2
    only (intermediates included), as needed for the reference relaxation.  keep1
    restricts the blocks of C1 like keep2 those of C2.  `large(C1, C2, Y, T1, T2, alpha)`,
    if given, adds the terms of the blocks of O2 that it does not return (Y: the
    two-body terms before the antisymmetrization).
    """
    G1, E1, L2, L3 = cum["Gamma1"], cum["Eta1"], cum["Lambda2"], cum["Lambda3"]
    con = lambda spec, *ops, **kw: contract(spec, *ops, dims=dims, **kw)
    C1, C2 = Blocks(), AntiBlocks()
    t0 = time.time()

    def lap(name):
        nonlocal t0
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.time() - t0
        t0 = time.time()

    A = (lambda letters: letters) if active else (lambda letters: "")
    C0 = 0.0
    if scalar:
        C0 = _scalar(O1, O2, T1, T2, G1, E1, L2, L3, con)
    C0 = alpha * C0
    lap("C0")

    # ---- one-body
    con("ap,ia->ip", O1, T1, out=C1, keep=keep1, scale=alpha, fix=A("ip"))
    con("pi,ia->pa", O1, T1, out=C1, keep=keep1, scale=-alpha, fix=A("pa"))

    con("ia,qapj,ji->qp", T1, O2, G1, out=C1, keep=keep1, scale=alpha, fix=A("qp"))
    con("mu,qvpm,uv->qp", T1, O2, G1, out=C1, keep=keep1, scale=-alpha, fix=A("qp"))

    con("ijab,bk,kj->ia", T2, O1, G1, out=C1, keep=keep1, scale=alpha, fix=A("ia"))
    con("ijau,vj,uv->ia", T2, O1, G1, out=C1, keep=keep1, scale=-alpha, fix=A("ia"))
    lap("C1 small")

    t1 = Blocks()
    con("abrk,ijab->ijrk", O2, T2, out=t1, fix=A("ir"))
    con("ijrk,kj->ir", t1, G1, out=C1, keep=keep1, scale=0.5 * alpha, fix=A("ir"))
    t1 = Blocks()
    con("ijux,uv,xy->ijvy", T2, G1, G1, out=t1, fix=A("ir"))
    con("ijvy,vyrj->ir", t1, O2, out=C1, keep=keep1, scale=0.5 * alpha, fix=A("ir"))
    t1 = Blocks()
    con("ijub,kj,uv->ikvb", T2, G1, G1, out=t1, fix=A("ir"))
    con("ikvb,vbrk->ir", t1, O2, out=C1, keep=keep1, scale=-alpha, fix=A("ir"))
    t1 = Blocks()
    con("ijau,uv->ijav", T2, G1, out=t1, fix=A("pa"))
    con("ijav,pvij->pa", t1, O2, out=C1, keep=keep1, scale=-0.5 * alpha, fix=A("pa"))
    t1 = Blocks()
    con("ijab,ki,lj->klab", T2, G1, G1, out=t1, fix=A("pa"))
    con("klab,pbkl->pa", t1, O2, out=C1, keep=keep1, scale=-0.5 * alpha, fix=A("pa"))
    t1 = Blocks()
    con("ijau,uv,kj->ikav", T2, G1, G1, out=t1, fix=A("pa"))
    con("ikav,pvik->pa", t1, O2, out=C1, keep=keep1, scale=alpha, fix=A("pa"))

    t1 = Blocks()
    con("ijxy,xyuv->ijuv", T2, L2, out=t1, fix=A("ir"))
    con("ijuv,uvrj->ir", t1, O2, out=C1, keep=keep1, scale=0.25 * alpha, fix=A("ir"))
    t1 = Blocks()
    con("uvab,xyuv->xyab", T2, L2, out=t1, fix=A("pa"))
    con("xyab,pbxy->pa", t1, O2, out=C1, keep=keep1, scale=-0.25 * alpha, fix=A("pa"))
    t1 = Blocks()
    con("iyav,uvxy->iuax", T2, L2, out=t1)
    con("iuax,axru->ir", t1, O2, out=C1, keep=keep1, scale=alpha, fix=A("ir"))
    con("iuax,pxiu->pa", t1, O2, out=C1, keep=keep1, scale=-alpha, fix=A("pa"))

    t1 = Blocks()
    con("avxy,xyuv->au", O2, L2, out=t1, fix=A("jb"))
    con("au,ujab->jb", t1, T2, out=C1, keep=keep1, scale=0.5 * alpha, fix=A("jb"))
    t1 = Blocks()
    con("xyiv,uvxy->ui", O2, L2, out=t1, fix=A("jb"))
    con("ui,ijub->jb", t1, T2, out=C1, keep=keep1, scale=-0.5 * alpha, fix=A("jb"))
    t1 = Blocks()
    con("uvey,xyuv->xe", T2, L2, out=t1, fix=A("qs"))
    con("xe,eqxs->qs", t1, O2, out=C1, keep=keep1, scale=0.5 * alpha, fix=A("qs"))
    t1 = Blocks()
    con("myuv,uvxy->mx", T2, L2, out=t1, fix=A("qs"))
    con("mx,xqms->qs", t1, O2, out=C1, keep=keep1, scale=-0.5 * alpha, fix=A("qs"))
    lap("C1 [O2,T2]")

    # ---- two-body: canonical placements only
    keepc = lambda lab: is_canonical(lab) and keep2(lab)
    k2 = dict(out=C2, keep=keepc)
    # Y collects terms that still have to be antisymmetrized over both index pairs,
    # C2 += alpha (1 - P01)(1 - P23) Y; a term that is already antisymmetric in one
    # pair enters with a factor 1/2
    Y = Blocks()
    ky = dict(out=Y, keep=keep2)
    con("ijab,ap->ijpb", T2, O1, scale=0.5, **ky, fix=A("ijpb"))
    con("ijab,qi->qjab", T2, O1, scale=-0.5, **ky, fix=A("qjab"))
    con("ia,arpq->irpq", T1, O2, scale=0.5, **ky, fix=A("irpq"))
    con("ia,rsiq->rsaq", T1, O2, scale=-0.5, **ky, fix=A("rsaq"))
    lap("C2 small")

    # particle-particle
    con("abrs,ijab->ijrs", O2, T2, scale=0.5 * alpha, **k2, fix=A("ijrs"))
    if ladder is not None:
        ladder(T2, C2, alpha)
    t1 = Blocks()
    con("ijbx,xy->ijby", T2, G1, out=t1, fix=A("ijrs"))
    con("ijby,byrs->ijrs", t1, O2, scale=-alpha, **k2, fix=A("ijrs"))
    lap("C2 pp")
    # hole-hole
    con("pqij,ijab->pqab", O2, T2, scale=0.5 * alpha, **k2, fix=A("pqab"))
    t1 = Blocks()
    con("yjab,xy->xjab", T2, E1, out=t1, fix=A("pqab"))
    con("xjab,pqxj->pqab", t1, O2, scale=-alpha, **k2, fix=A("pqab"))
    lap("C2 hh")
    # particle-hole
    t1 = Blocks()
    con("ijab,ki->kjab", T2, G1, out=t1, fix=A("qjsb"))
    con("kjab,aqks->qjsb", t1, O2, **ky, fix=A("qjsb"))
    t1 = Blocks()
    con("ijub,uv->ijvb", T2, G1, out=t1, fix=A("qjsb"))
    con("ijvb,vqis->qjsb", t1, O2, scale=-1.0, **ky, fix=A("qjsb"))
    del t1
    lap("C2 ph")
    if large is not None:
        large(C1, C2, Y, T1, T2, alpha)
        lap("C large")
    for lab in list(Y):
        y = Y.pop(lab)
        for perm, sign in (((0, 1, 2, 3), 1.0), ((1, 0, 2, 3), -1.0), ((0, 1, 3, 2), -1.0),
                           ((1, 0, 3, 2), 1.0)):
            olab = "".join(lab[k] for k in perm)
            if not is_canonical(olab):
                continue
            if olab not in C2:
                C2[olab] = numpy.zeros([y.shape[k] for k in perm], dtype=numpy.complex128)
            kernels.transpose_add(C2[olab], y, perm, alpha=sign * alpha)
        del y
    lap("C2 antisym")
    return C0, C1, C2


def add_hc_1(C):
    for lab in list(C):
        if lab[::-1] not in C:
            C[lab[::-1]] = numpy.zeros(C[lab].shape[::-1], dtype=complex)
    done = set()
    for lab in list(C):
        if lab in done:
            continue
        par = lab[::-1]
        C[lab] = C[lab] + C[par].conj().T
        if par != lab:
            C[par] = C[lab].conj().T.copy()
        done.update((lab, par))


def add_hc_2(C):
    for lab in list(C):
        par = lab[2:] + lab[:2]
        if par not in C:
            C[par] = numpy.zeros(tuple(C[lab].shape[k] for k in (2, 3, 0, 1)), dtype=complex)
    done = set()
    for lab in list(C):
        if lab in done:
            continue
        par = lab[2:] + lab[:2]
        t = kernels.transposed(C[par], (2, 3, 0, 1))
        numpy.conj(t, out=t)
        C[lab] += t
        if par != lab:
            kernels.transpose_add(C[par], C[lab], (2, 3, 0, 1), assign=True)
            numpy.conj(C[par], out=C[par])
        del t
        done.update((lab, par))
