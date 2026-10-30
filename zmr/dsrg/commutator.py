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

import time

import numpy

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


class BareV:
    """Two-electron integrals in the storage convention, block by block from the
    three-index tensors: V[pq, rs] = (pr|qs) - (ps|qr) with conjugated coefficients.

    Blocks below `cache_size` elements are kept; larger ones are rebuilt when asked for
    (only the most recent one is kept).  The block with four virtual indices is only
    built when `dense_vvvv` is set (small systems); otherwise it is None here and the
    solver uses the packed matrix of `packed_vvvv`.
    """

    def __init__(self, B, dense_vvvv=False, cache_size=2**26, extra=None):
        self.B = B
        # blocks added to the factorized integrals (e.g. a correction of 'aaaa')
        self.extra = extra or {}
        self.dense_vvvv = dense_vvvv
        self.cache_size = cache_size
        self.cache = {}
        self.last = (None, None)

    def build(self, lab):
        p, q, r, s = lab
        B = self.B
        v = numpy.einsum("Ppr,Pqs->pqrs", B[p + r], B[q + s], optimize=True)
        if r == s:
            v -= v.swapaxes(2, 3)
        elif p == q:
            v -= v.swapaxes(0, 1)
        else:
            v -= numpy.einsum("Pps,Pqr->pqrs", B[p + s], B[q + r], optimize=True)
        return v

    def get(self, lab):
        if lab == "vvvv" and not self.dense_vvvv:
            return None
        if lab in self.cache:
            return self.cache[lab]
        if lab in self.extra:
            self.cache[lab] = self.build(lab) + self.extra[lab]
            return self.cache[lab]
        if self.last[0] == lab:
            return self.last[1]
        self.last = (None, None)
        if lab.count("v") == 3:
            # one array per non-virtual space, V[vv, vx]; the other placements follow from
            # antisymmetry and hermiticity (transient copies)
            k = [i for i, x in enumerate(lab) if x != "v"][0]
            canon = "vvv" + lab[k]
            if canon not in self.cache:
                self.cache[canon] = self.build(canon)
            c = self.cache[canon]
            # (view, sign, complex conjugate): no copy of the large array is made, the
            # contraction applies sign and conjugation to the other factors
            if k == 3:
                return c
            if k == 2:
                return c.transpose(0, 1, 3, 2), -1.0, False
            if k == 1:
                return c.transpose(2, 3, 0, 1), 1.0, True
            return c.transpose(3, 2, 0, 1), -1.0, True
        v = self.build(lab)
        if v.size <= self.cache_size or lab == "vvvv":
            self.cache[lab] = v
        else:
            self.last = (lab, v)
        return v

    def packed_vvvv(self, batch=8, single=False):
        """V[ab, rs] for a < b, r < s as a matrix (npair, npair)."""
        Bvv = self.B["vv"]
        naux, nv, _ = Bvv.shape
        iu = numpy.triu_indices(nv, 1)
        npair = iu[0].size
        w = numpy.empty((npair, npair), dtype=numpy.complex64 if single else complex)
        right = numpy.ascontiguousarray(Bvv.reshape(naux, nv * nv))
        left = numpy.ascontiguousarray(Bvv.transpose(1, 2, 0))  # a, r, P
        row = 0
        for a0 in range(0, nv - 1, batch):
            a1 = min(a0 + batch, nv - 1)
            # d[a, r, b, s] = sum_P B[P, a, r] B[P, b, s]
            d = (left[a0:a1].reshape(-1, naux) @ right).reshape(a1 - a0, nv, nv, nv)
            for a in range(a0, a1):
                x = d[a - a0].transpose(1, 0, 2)[a + 1:]  # b, r, s with b > a
                x = x - x.transpose(0, 2, 1)
                n = nv - a - 1
                w[row:row + n] = x[:, iu[0], iu[1]]
                row += n
        assert row == npair
        return w


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
               scalar=True, active=False):
    """alpha * [O, T] -> (C0, C1, C2), all blocks (C2 restricted by keep2).

    Literal transcription of MRDSRG_SO::H?_T?_C? of forte.  `ladder`, if given, adds the
    particle-particle ladder of the four-virtual block of O2 (which is then absent from
    O2 itself).  scalar=False skips C0; active=True builds the active blocks of C1 and C2
    only (intermediates included), as needed for the reference relaxation.
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
    con("ap,ia->ip", O1, T1, out=C1, scale=alpha, fix=A("ip"))
    con("pi,ia->pa", O1, T1, out=C1, scale=-alpha, fix=A("pa"))

    con("ia,qapj,ji->qp", T1, O2, G1, out=C1, scale=alpha, fix=A("qp"))
    con("mu,qvpm,uv->qp", T1, O2, G1, out=C1, scale=-alpha, fix=A("qp"))

    con("ijab,bk,kj->ia", T2, O1, G1, out=C1, scale=alpha, fix=A("ia"))
    con("ijau,vj,uv->ia", T2, O1, G1, out=C1, scale=-alpha, fix=A("ia"))
    lap("C1 small")

    t1 = Blocks()
    con("abrk,ijab->ijrk", O2, T2, out=t1, fix=A("ir"))
    con("ijrk,kj->ir", t1, G1, out=C1, scale=0.5 * alpha, fix=A("ir"))
    t1 = Blocks()
    con("ijux,uv,xy->ijvy", T2, G1, G1, out=t1, fix=A("ir"))
    con("ijvy,vyrj->ir", t1, O2, out=C1, scale=0.5 * alpha, fix=A("ir"))
    t1 = Blocks()
    con("ijub,kj,uv->ikvb", T2, G1, G1, out=t1, fix=A("ir"))
    con("ikvb,vbrk->ir", t1, O2, out=C1, scale=-alpha, fix=A("ir"))
    t1 = Blocks()
    con("ijau,uv->ijav", T2, G1, out=t1, fix=A("pa"))
    con("ijav,pvij->pa", t1, O2, out=C1, scale=-0.5 * alpha, fix=A("pa"))
    t1 = Blocks()
    con("ijab,ki,lj->klab", T2, G1, G1, out=t1, fix=A("pa"))
    con("klab,pbkl->pa", t1, O2, out=C1, scale=-0.5 * alpha, fix=A("pa"))
    t1 = Blocks()
    con("ijau,uv,kj->ikav", T2, G1, G1, out=t1, fix=A("pa"))
    con("ikav,pvik->pa", t1, O2, out=C1, scale=alpha, fix=A("pa"))

    t1 = Blocks()
    con("ijxy,xyuv->ijuv", T2, L2, out=t1, fix=A("ir"))
    con("ijuv,uvrj->ir", t1, O2, out=C1, scale=0.25 * alpha, fix=A("ir"))
    t1 = Blocks()
    con("uvab,xyuv->xyab", T2, L2, out=t1, fix=A("pa"))
    con("xyab,pbxy->pa", t1, O2, out=C1, scale=-0.25 * alpha, fix=A("pa"))
    t1 = Blocks()
    con("iyav,uvxy->iuax", T2, L2, out=t1)
    con("iuax,axru->ir", t1, O2, out=C1, scale=alpha, fix=A("ir"))
    con("iuax,pxiu->pa", t1, O2, out=C1, scale=-alpha, fix=A("pa"))

    t1 = Blocks()
    con("avxy,xyuv->au", O2, L2, out=t1, fix=A("jb"))
    con("au,ujab->jb", t1, T2, out=C1, scale=0.5 * alpha, fix=A("jb"))
    t1 = Blocks()
    con("xyiv,uvxy->ui", O2, L2, out=t1, fix=A("jb"))
    con("ui,ijub->jb", t1, T2, out=C1, scale=-0.5 * alpha, fix=A("jb"))
    t1 = Blocks()
    con("uvey,xyuv->xe", T2, L2, out=t1, fix=A("qs"))
    con("xe,eqxs->qs", t1, O2, out=C1, scale=0.5 * alpha, fix=A("qs"))
    t1 = Blocks()
    con("myuv,uvxy->mx", T2, L2, out=t1, fix=A("qs"))
    con("mx,xqms->qs", t1, O2, out=C1, scale=-0.5 * alpha, fix=A("qs"))
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
