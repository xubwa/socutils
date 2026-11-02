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


def _is_plain(a):
    return a.dtype == numpy.complex128 and a.flags.c_contiguous


def _pair(asub, a, bsub, b, keepers):
    """Contraction of two arrays over the letters they share that are not in
    `keepers`: (subscripts, array).  One threaded matrix product between threaded
    transpositions (zmr.lib); numpy.einsum for traces and outer products."""
    con = [x for x in asub if x in bsub and x not in keepers]
    aown = [x for x in asub if x not in con]
    bown = [x for x in bsub if x not in con]
    if (not con or len(set(asub)) != len(asub) or len(set(bsub)) != len(bsub)
            or set(aown) & set(bown)):
        osub = "".join(dict.fromkeys(x for x in asub + bsub if x in keepers))
        return osub, numpy.einsum(asub + "," + bsub + "->" + osub, a, b, optimize=True)
    pa = [asub.index(x) for x in aown + con]
    pb = [bsub.index(x) for x in con + bown]
    at = a if (pa == list(range(a.ndim)) and _is_plain(a)) else kernels.transposed(a, pa)
    bt = b if (pb == list(range(b.ndim)) and _is_plain(b)) else kernels.transposed(b, pb)
    na = int(numpy.prod(at.shape[:len(aown)], dtype=numpy.int64))
    nc = int(numpy.prod(at.shape[len(aown):], dtype=numpy.int64))
    if at.size == 0 or bt.size == 0:
        r = numpy.zeros(at.shape[:len(aown)] + bt.shape[len(con):], dtype=numpy.complex128)
    else:
        r = la.gemm(at.reshape(na, nc), bt.reshape(nc, -1))
        r = r.reshape(at.shape[:len(aown)] + bt.shape[len(con):])
    return "".join(aown + bown), r


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
    for spaces in product(*("a" if ch in fix else _CLASS[ch] for ch in letters)):
        sp = dict(zip(letters, spaces))
        olab = "".join(sp[ch] for ch in rhs)
        if rhs and keep is not None and olab not in out and not keep(olab):
            continue
        arrays, asubs = [], []
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
                    continue
                for k in list(rename):
                    if rename[k] == gone:
                        rename[k] = stay
                rename[gone] = stay
                continue
            arrays.append(blk)
            asubs.append(sub)
        if not ok:
            continue
        asubs = ["".join(rename.get(ch, ch) for ch in s) for s in asubs]
        if PROFILE is not None:
            _t = time.time()
        if cj:
            # einsum(conj(A), B) = conj(einsum(A, conj(B)))
            arrays = [x if k in conj_ops else numpy.conj(x) for k, x in enumerate(arrays)]
        csub, val = asubs[0], arrays[0]
        for k in range(1, len(arrays)):
            later = set(rhs).union(*asubs[k + 1:])
            csub, val = _pair(csub, val, asubs[k], arrays[k], later)
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
        if cj:
            val = numpy.conj(val)
        perm = [csub.index(x) for x in rhs]
        if olab not in out:
            out[olab] = numpy.zeros([val.shape[p] for p in perm], dtype=numpy.complex128)
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

    def vvvx_chunks(self, x, max_memory):
        """V[ab, ek] (a, b, e virtual, k in the space x) in chunks of k within max_memory
        (MB, the chunk and four copies of its size for the caller): yields (k0, k1,
        block[a, b, e, k])."""
        B = self.B
        naux, nv, nx = B.shape("v" + x)
        bvv = B.file["vv"]
        step = max(1, min(nx, int(max_memory * 1e6 / (5 * 16.0 * nv**3))))
        pstep = self.B.step(16.0 * nv * (nv + step), self.read_memory)
        for k0 in range(0, nx, step):
            k1 = min(nx, k0 + step)
            d = numpy.zeros((nv * nv, nv * (k1 - k0)), dtype=complex)
            for p0 in range(0, naux, pstep):
                t = time.time()
                left = bvv[p0:p0 + pstep].reshape(-1, nv * nv)            # P, (a e), unconjugated
                right = numpy.ascontiguousarray(B.read("v" + x, p0, p0 + pstep, slice(k0, k1)))
                right = right.reshape(left.shape[0], -1)                  # P, (b k)
                self.timings["read"] += time.time() - t
                t = time.time()
                la.gemm(left, right, opa="C", out=d, beta=1.0)            # conj(B[vv])^T B[vx]
                self.timings["vvvx gemm"] += time.time() - t
            t = time.time()
            d = d.reshape(nv, nv, nv, k1 - k0)                                    # a, e, b, k
            # V[ab, ek] = (ae|bk) - (ak|be) = d[a, e, b, k] - d[b, e, a, k]
            v = kernels.transposed(d, (0, 2, 1, 3))
            kernels.transpose_add(v, d, (2, 0, 1, 3), alpha=-1.0)
            del d
            self.timings["vvvx antisym"] += time.time() - t
            yield k0, k1, v

    def vvvv_rows(self, batch=8):
        """V[ab, rs] for a < b, r < s, built from the three-index tensors: yields (first
        row, rows) with the rows of `batch` values of a at a time (pairs in the order of
        numpy.triu_indices)."""
        naux, nv, _ = self.B.shape("vv")
        bvv = self.B.file["vv"]                    # unconjugated
        tim = self.timings
        row = 0
        for a0 in range(0, nv - 1, batch):
            a1 = min(a0 + batch, nv - 1)
            nb, nr = a1 - a0, nv - a0
            # D[a, r, b, s] = sum_P B[P, a, r] B[P, b, s] for b >= a0 only (b > a is
            # kept), one pass over B[vv]; V = conj(D - D^T(r, s))
            pstep = self.B.step(16.0 * nr * nv, self.read_memory)
            d = numpy.zeros((nb * nv, nr * nv), dtype=complex)
            for p0 in range(0, naux, pstep):
                t = time.time()
                x = bvv[p0:p0 + pstep, a0:]
                tim["read"] += time.time() - t
                t = time.time()
                la.gemm(x[:, :nb].reshape(x.shape[0], -1), x.reshape(x.shape[0], -1),
                        opa="T", out=d, beta=1.0)
                tim["vvvv gemm"] += time.time() - t
            t = time.time()
            w = kernels.vvvv_pair_rows(d.reshape(nb, nv, nr, nv), a0)
            del d
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
