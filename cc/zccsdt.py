#
# Full spin-orbital CCSDT (iterative T3) for socutils two-component spinor CC.
#
# Efficient T1-dressed tensor implementation, consistent with pyscf's rccsdt:
# T1 is absorbed into the antisymmetrized integrals and Fock (x = 1 - t1,
# y = 1 + t1), so the T3 sector is T1-free.  Conventions follow pyscf gccsd /
# socutils zccsd: spin-orbital, antisymmetrized physicist integrals <pq||rs>,
# complex amplitudes, t2/t3 fully antisymmetric.
#
# VALIDATION: every residual (r1, r2, r3) matches the guaranteed-correct
# determinant-space oracle (socutils.cc._ccsdt_bruteforce) element-wise to
# ~1e-15 (random t1/t2/t3, t1=0 and t1!=0), on shapes with both nvir<=nocc and
# nvir>nocc (the latter exercises the bilinear t2*t3 terms), and the converged
# correlation energy matches pyscf rccsdt_highm on H2O/HF/LiH (STO-3G and
# 6-31g, frozen core) to ~1e-9.  See cc/_t3model.py for the standalone
# term-by-term r3 check.
#
# PERFORMANCE: amplitudes stored on their unique antisymmetric blocks; the T3
# residual exploits pair antisymmetry in the ladders/bilinear terms, folds each
# bilinear t2*t3 correction into its linear partner's effective intermediate
# (one contraction + one antisymmetrizer per channel), and offloads the
# permutation (anti)symmetrizers to an optional OpenMP C backend
# (socutils.cc._ccsdt_clib; pure-numpy fallback otherwise).
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from socutils.cc.zccsd import ZCCSD, _PhysicistsERIs
from socutils.cc._ccsdt_bruteforce import build_g
try:
    from socutils.cc import _ccsdt_clib as _clib
except Exception:
    _clib = None

einsum = lib.einsum


def _einsum_opt(*args):
    '''numpy einsum with path optimization -- faster than pyscf lib.einsum for
    the memory-bound single-index F*t3 contractions (which lib.einsum routes
    through extra full-tensor transposes).'''
    return np.einsum(*args, optimize=True)


def _asarray(x):
    return np.asarray(x)


# ---------- antisymmetrizers / partial permutation operators ----------

_USE_CASYM = _clib is not None and getattr(_clib, 'HAVE_ASYM', False)


def fullasym(t):
    '''Full P(i/jk)*P(a/bc) antisymmetrizer (6x6 signed perms, no 1/36).'''
    if _USE_CASYM:
        return _clib.fullasym(t)
    a = (t + t.transpose(1, 2, 0, 3, 4, 5) + t.transpose(2, 0, 1, 3, 4, 5)
         - t.transpose(1, 0, 2, 3, 4, 5) - t.transpose(0, 2, 1, 3, 4, 5)
         - t.transpose(2, 1, 0, 3, 4, 5))
    a = (a + a.transpose(0, 1, 2, 4, 5, 3) + a.transpose(0, 1, 2, 5, 3, 4)
         - a.transpose(0, 1, 2, 4, 3, 5) - a.transpose(0, 1, 2, 3, 5, 4)
         - a.transpose(0, 1, 2, 5, 4, 3))
    return a


def _Pabc(t):
    if _USE_CASYM:
        return _clib.Pabc(t)
    return t - t.transpose(0, 1, 2, 4, 3, 5) - t.transpose(0, 1, 2, 5, 4, 3)


def _Pijk(t):
    if _USE_CASYM:
        return _clib.Pijk(t)
    return t - t.transpose(1, 0, 2, 3, 4, 5) - t.transpose(2, 1, 0, 3, 4, 5)


def _Pc_ab(t):
    if _USE_CASYM:
        return _clib.Pc_ab(t)
    return t - t.transpose(0, 1, 2, 5, 4, 3) - t.transpose(0, 1, 2, 3, 5, 4)


def _Pk_ij(t):
    if _USE_CASYM:
        return _clib.Pk_ij(t)
    return t - t.transpose(2, 1, 0, 3, 4, 5) - t.transpose(0, 2, 1, 3, 4, 5)


def _Pijk_Pabc(t):
    if _USE_CASYM:
        return _clib.Pijk_Pabc(t)
    return _Pijk(_Pabc(t))


# ---------- T2 antisymmetric packing (store only i<j, a<b) ----------

_T2_PAIR_CACHE = {}


def _antisym2(t):
    '''P(ij)P(ab) antisymmetrizer for t2[i,j,a,b] (4 signed perms, no 1/4).'''
    return (t - t.transpose(1, 0, 2, 3) - t.transpose(0, 1, 3, 2)
            + t.transpose(1, 0, 3, 2))


def _t2_pair(nocc, nvir):
    key = (nocc, nvir)
    if key not in _T2_PAIR_CACHE:
        op = np.array([(i, j) for i in range(nocc) for j in range(i + 1, nocc)],
                      dtype=int).reshape(-1, 2)
        vp = np.array([(a, b) for a in range(nvir) for b in range(a + 1, nvir)],
                      dtype=int).reshape(-1, 2)
        _T2_PAIR_CACHE[key] = (op, vp)
    return _T2_PAIR_CACHE[key]


def pack_t2(t2):
    '''Pack antisymmetric t2[i,j,a,b] into its unique (i<j, a<b) components.'''
    nocc, nvir = t2.shape[0], t2.shape[2]
    if _clib is not None and _clib.HAVE_CLIB:
        return _clib.pack_t2(t2, nocc, nvir)
    op, vp = _t2_pair(nocc, nvir)
    if len(op) == 0 or len(vp) == 0:
        return np.zeros(0, dtype=t2.dtype)
    return t2[op[:, 0][:, None], op[:, 1][:, None],
              vp[:, 0][None, :], vp[:, 1][None, :]].ravel()


def unpack_t2(packed, nocc, nvir):
    '''Rebuild the full antisymmetric t2 from its unique components.'''
    if _clib is not None and _clib.HAVE_CLIB:
        return _clib.unpack_t2(packed, nocc, nvir)
    op, vp = _t2_pair(nocc, nvir)
    seed = np.zeros((nocc, nocc, nvir, nvir),
                    dtype=np.result_type(packed, np.complex128))
    if len(op) and len(vp):
        seed[op[:, 0][:, None], op[:, 1][:, None],
             vp[:, 0][None, :], vp[:, 1][None, :]] = packed.reshape(len(op), len(vp))
    return _antisym2(seed)


# ---------- T3 antisymmetric packing (store only i<j<k, a<b<c) ----------

_T3_TRI_CACHE = {}


def _t3_tri(nocc, nvir):
    '''Index arrays of the strictly-ordered occupied (i<j<k) and virtual
    (a<b<c) triples; cached per (nocc, nvir).'''
    key = (nocc, nvir)
    if key not in _T3_TRI_CACHE:
        occ = np.array([(i, j, k) for i in range(nocc)
                        for j in range(i + 1, nocc)
                        for k in range(j + 1, nocc)], dtype=int).reshape(-1, 3)
        vir = np.array([(a, b, c) for a in range(nvir)
                        for b in range(a + 1, nvir)
                        for c in range(b + 1, nvir)], dtype=int).reshape(-1, 3)
        _T3_TRI_CACHE[key] = (occ, vir)
    return _T3_TRI_CACHE[key]


def pack_t3(t3):
    '''Pack a fully-antisymmetric t3[i,j,k,a,b,c] into its unique components
    (i<j<k, a<b<c) -- a flat array ~36x smaller.'''
    nocc, nvir = t3.shape[0], t3.shape[3]
    if _clib is not None and _clib.HAVE_CLIB:
        return _clib.pack_t3(t3, nocc, nvir)
    occ, vir = _t3_tri(nocc, nvir)
    if len(occ) == 0 or len(vir) == 0:
        return np.zeros(0, dtype=t3.dtype)
    return t3[occ[:, 0][:, None], occ[:, 1][:, None], occ[:, 2][:, None],
              vir[:, 0][None, :], vir[:, 1][None, :], vir[:, 2][None, :]].ravel()


def unpack_t3(packed, nocc, nvir):
    '''Rebuild the full antisymmetric t3 from its unique components.  The unique
    entries are scattered into a seed tensor and ``fullasym`` fills the rest
    with the correct signs (no manual sign bookkeeping).'''
    if _clib is not None and _clib.HAVE_CLIB:
        return _clib.unpack_t3(packed, nocc, nvir)
    occ, vir = _t3_tri(nocc, nvir)
    seed = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir),
                    dtype=np.result_type(packed, np.complex128))
    if len(occ) and len(vir):
        seed[occ[:, 0][:, None], occ[:, 1][:, None], occ[:, 2][:, None],
             vir[:, 0][None, :], vir[:, 1][None, :], vir[:, 2][None, :]] = \
            packed.reshape(len(occ), len(vir))
    return fullasym(seed)


# ---------- T1 dressing ----------

def _t1_dress(g, fock, t1, nocc, nmo):
    '''Generalized spin-orbital T1 dressing of the antisymmetrized integrals
    g[p,q,r,s]=<pq||rs> and the Fock matrix (x = 1 - t1, y = 1 + t1).'''
    x = np.eye(nmo, dtype=complex)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(nmo, dtype=complex)
    y[:nocc, nocc:] += t1
    ge = einsum('tvuw,pt->pvuw', g, x)
    ge = einsum('pvuw,rv->pruw', ge, x)
    ge = ge.transpose(2, 3, 0, 1)
    ge = einsum('uwpr,qu->qwpr', ge, y)
    ge = einsum('qwpr,sw->qspr', ge, y)
    ge = ge.transpose(2, 3, 0, 1)
    fdr = fock + einsum('risa,ia->rs', g[:, :nocc, :, nocc:], t1)
    fdr = x @ fdr @ y.T
    return ge, fdr


def _t3_residual(t2, t3, ge, fdr, nocc, nmo):
    '''Spin-orbital CCSDT T3 residual <Phi_ijk^abc|Hbar|0>, T1-free in the
    T1-dressed integrals ge (=<pq||rs> dressed) and Fock fdr.  Verified
    element-wise against the determinant-space oracle to ~1e-15.

    Each bilinear t2*t3 correction has the same index/symmetry structure as its
    linear partner, so it is folded into an *effective* intermediate and the
    antisymmetrizer / O(o^3 v^3) contraction is applied only once per channel:
      F_vv  <- f_vv  - 0.5 dF_vv         (-> P(a/bc))
      F_oo  <- f_oo  + 0.5 dF_oo         (-> P(i/jk))
      W_vvvv<- <ab||de> + 0.5 dW_vvvv    (-> pp ladder, P(c/ab))
      W_oooo<- <mn||ij> + 0.5 dW_oooo    (-> hh ladder, P(k/ij))
      W_ovvo<- <ma||di> + dW_ovvo        (-> ring, P(i/jk)P(a/bc))
      W_vvvo<- W_vvvo + 0.5 dW_vvvo[t3]  (-> fullasym drive)
      W_vooo<- W_vooo + 0.5 dW_vooo[t3]  (-> fullasym drive)
    '''
    o = slice(0, nocc)
    v = slice(nocc, nmo)
    nvir = nmo - nocc
    fvv = fdr[v, v]; foo = fdr[o, o]; fov = fdr[o, v]
    gvvvo = ge[v, v, v, o]; govoo = ge[o, v, o, o]
    vvvv = ge[v, v, v, v]; oooo = ge[o, o, o, o]; ovvo = ge[o, v, v, o]
    goovv = ge[o, o, v, v]; ovvv = ge[o, v, v, v]; ooov = ge[o, o, o, v]
    oovo = ge[o, o, v, o]
    op, vp = _t2_pair(nocc, nvir)        # op = occ pairs m<n, vp = vir pairs d<e/a<b
    md, ne = vp[:, 0], vp[:, 1]
    mm, nn = op[:, 0], op[:, 1]

    # ---- effective intermediates (linear + bilinear folded together) ----
    # drive: W_vvvo / W_vooo carry the linear-in-t2 + quadratic-t2^2 driving;
    # add the t3-dressing (0.5 dW[t3], shared antisym pair packed -> 2x).
    Wvvvo = gvvvo.copy()
    Wvvvo += 0.5 * einsum('mnei,mnab->abei', oovo, t2)              # hole ladder
    tmp = einsum('mbef,miaf->abei', ovvv, t2)
    Wvvvo -= (tmp - tmp.transpose(1, 0, 2, 3))                      # P(ab) particle
    Wvvvo -= einsum('me,miab->abei', fov, t2)                       # dressed-Fock_ov * t2
    Wvvvo += einsum('Pef,iPabf->abei', goovv[mm, nn], t3[:, mm, nn])  # 0.5 dW_vvvo[t3]
    Wvooo = govoo.copy()                                            # slot [m,a,j,i]
    tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
    Wvooo -= (tmp2 - tmp2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    Wvooo -= (0.5 * einsum('mbef,ijef->mbij', ovvv, t2)).transpose(0, 1, 3, 2)
    Wvooo += einsum('mnP,ijnaP->maji', goovv[:, :, md, ne], t3[:, :, :, :, md, ne])  # 0.5 dW_vooo[t3]
    R = -0.25 * fullasym(einsum('abei,jkec->ijkabc', Wvvvo, t2)
                         + einsum('maji,mkbc->ijkabc', Wvooo, t2))

    # F_vv: f_vv - 0.5 dF_vv  (dF_vv mn-pair packed -> 2x)
    Feff_vv = fvv - einsum('Paf,Pdf->ad', t2[mm, nn], goovv[mm, nn])
    R += _Pabc(_einsum_opt('ad,ijkdbc->ijkabc', Feff_vv, t3))
    # F_oo: f_oo + 0.5 dF_oo  (dF_oo ef-pair packed -> 2x)
    Feff_oo = foo + einsum('mnP,inP->mi', goovv[:, :, md, ne], t2[:, :, md, ne])
    R -= _Pijk(_einsum_opt('mi,mjkabc->ijkabc', Feff_oo, t3))
    # W_ovvo (ring): <ma||di> + dW_ovvo
    Weff_ovvo = ovvo + einsum('mnde,inae->madi', goovv, t2)
    R += _Pijk_Pabc(einsum('madi,mjkdbc->ijkabc', Weff_ovvo, t3))

    # pp ladder: W_vvvv = <ab||de> + 0.5 dW_vvvv, fully (ab)(de)-antisymmetric;
    # pack BOTH the summed pair (d<e) and the output pair (a<b) -> 4x.
    Weff_vvvv = vvvv + einsum('Pab,Pde->abde', t2[mm, nn], goovv[mm, nn])
    vvvv_pp = Weff_vvvv[md[:, None], ne[:, None], md[None, :], ne[None, :]]  # (ab<, de<)
    Xpp = einsum('QP,ijkPc->ijkQc', vvvv_pp, t3[:, :, :, md, ne, :])
    Xf = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=R.dtype)
    Xf[:, :, :, md, ne, :] = Xpp
    Xf[:, :, :, ne, md, :] = -Xpp
    R += _Pc_ab(Xf)
    # hh ladder: W_oooo = <mn||ij> + 0.5 dW_oooo; pack the summed pair (m<n) and
    # the output pair (i<j) -> 4x.
    Weff_oooo = oooo + einsum('mnP,ijP->mnij', goovv[:, :, md, ne], t2[:, :, md, ne])
    oooo_pp = Weff_oooo[mm[:, None], nn[:, None], mm[None, :], nn[None, :]]  # (mn<, ij<)
    Xhh = einsum('PQ,Pkabc->Qkabc', oooo_pp, t3[mm, nn])
    Yf = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=R.dtype)
    Yf[mm, nn] = Xhh
    Yf[nn, mm] = -Xhh
    R += _Pk_ij(Yf)
    return R


_USE_CPACK = _clib is not None and getattr(_clib, 'HAVE_ASYM_PACK', False)


def _t3_residual_packed(t2, t3, ge, fdr, nocc, nmo):
    '''Same T3 residual as ``_t3_residual`` but returned on its unique
    antisymmetric block Rp[notri, nvtri] (i<j<k, a<b<c), never materializing the
    full O(o^3 v^3) tensor.

    Because each term inherits t3's antisymmetry, OP(term) is fully
    antisymmetric and the unique block determines it.  Terms whose restricted
    triple is a pure t3 spectator are contracted on the *reduced* t3 (occ-triple
    or vir-triple restricted -> ~6x, and a further 2x for a packed summed pair):
      F_vv, pp-ladder : t3 occ-restricted   (i<j<k)
      F_oo, hh-ladder : t3 vir-restricted   (a<b<c)
    The drive (fullasym) and ring (P(i/jk)P(a/bc)) mix t3 with t2/W on both
    sides, so they form the full contraction X and pack it.  Requires the C
    backend; callers fall back to pack_t3(_t3_residual(...)) otherwise.
    '''
    o = slice(0, nocc); v = slice(nocc, nmo)
    nvir = nmo - nocc
    notri = nocc * (nocc - 1) * (nocc - 2) // 6
    nvtri = nvir * (nvir - 1) * (nvir - 2) // 6
    fvv = fdr[v, v]; foo = fdr[o, o]; fov = fdr[o, v]
    gvvvo = ge[v, v, v, o]; govoo = ge[o, v, o, o]
    vvvv = ge[v, v, v, v]; oooo = ge[o, o, o, o]; ovvo = ge[o, v, v, o]
    goovv = ge[o, o, v, v]; ovvv = ge[o, v, v, v]; ooov = ge[o, o, o, v]
    oovo = ge[o, o, v, o]
    op, vp = _t2_pair(nocc, nvir)
    md, ne = vp[:, 0], vp[:, 1]; mm, nn = op[:, 0], op[:, 1]
    oi, vi = _t3_tri(nocc, nvir)
    oi0, oi1, oi2 = oi[:, 0], oi[:, 1], oi[:, 2]
    vi0, vi1, vi2 = vi[:, 0], vi[:, 1], vi[:, 2]

    Rp = np.zeros(notri * nvtri, dtype=np.complex128)

    # drive (fullasym, full X): W_vvvo / W_vooo with t3-dressing folded in
    Wvvvo = gvvvo.copy()
    Wvvvo += 0.5 * einsum('mnei,mnab->abei', oovo, t2)
    tmp = einsum('mbef,miaf->abei', ovvv, t2)
    Wvvvo -= (tmp - tmp.transpose(1, 0, 2, 3))
    Wvvvo -= einsum('me,miab->abei', fov, t2)
    Wvvvo += einsum('Pef,iPabf->abei', goovv[mm, nn], t3[:, mm, nn])
    Wvooo = govoo.copy()
    tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
    Wvooo -= (tmp2 - tmp2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    Wvooo -= (0.5 * einsum('mbef,ijef->mbij', ovvv, t2)).transpose(0, 1, 3, 2)
    Wvooo += einsum('mnP,ijnaP->maji', goovv[:, :, md, ne], t3[:, :, :, :, md, ne])
    Xdr = (einsum('abei,jkec->ijkabc', Wvvvo, t2)
           + einsum('maji,mkbc->ijkabc', Wvooo, t2))
    _clib.full_to_pack(Rp, Xdr, -0.25, 0, nocc, nvir)

    # F_vv (P(a/bc), occ-restricted t3)
    Feff_vv = fvv - einsum('Paf,Pdf->ad', t2[mm, nn], goovv[mm, nn])
    t3occ = t3[oi0, oi1, oi2]                                   # (notri,v,v,v)
    Xfvv = _einsum_opt('ad,Idbc->Iabc', Feff_vv, t3occ)
    _clib.pvir_to_pack(Rp, Xfvv, 1.0, 0, nocc, nvir)

    # F_oo (-P(i/jk), vir-restricted t3)
    Feff_oo = foo + einsum('mnP,inP->mi', goovv[:, :, md, ne], t2[:, :, md, ne])
    t3vir = t3[:, :, :, vi0, vi1, vi2]                          # (o,o,o,nvtri)
    Xfoo = _einsum_opt('mi,mjkA->ijkA', Feff_oo, t3vir)
    _clib.pocc_to_pack(Rp, Xfoo, -1.0, 0, nocc, nvir)

    # ring (P(i/jk)P(a/bc), full X)
    Weff_ovvo = ovvo + einsum('mnde,inae->madi', goovv, t2)
    Xring = einsum('madi,mjkdbc->ijkabc', Weff_ovvo, t3)
    _clib.full_to_pack(Rp, Xring, 1.0, 1, nocc, nvir)

    # pp ladder (P(c/ab), occ-restricted t3, summed pair d<e packed)
    Weff_vvvv = vvvv + einsum('Pab,Pde->abde', t2[mm, nn], goovv[mm, nn])
    vvvv_pp = Weff_vvvv[md[:, None], ne[:, None], md[None, :], ne[None, :]]
    t3r = t3occ[:, md, ne, :]                                   # (notri, de<, c)
    Xpp = einsum('QP,IPc->IQc', vvvv_pp, t3r)                   # (notri, ab<, c)
    Xocc = np.zeros((notri, nvir, nvir, nvir), dtype=np.complex128)
    Xocc[:, md, ne, :] = Xpp
    Xocc[:, ne, md, :] = -Xpp
    _clib.pvir_to_pack(Rp, Xocc, 1.0, 1, nocc, nvir)

    # hh ladder (P(k/ij), vir-restricted t3, summed pair m<n packed)
    Weff_oooo = oooo + einsum('mnP,ijP->mnij', goovv[:, :, md, ne], t2[:, :, md, ne])
    oooo_pp = Weff_oooo[mm[:, None], nn[:, None], mm[None, :], nn[None, :]]
    t3hr = t3[mm, nn][:, :, vi0, vi1, vi2]                      # (mn<, k, nvtri)
    Xhh = einsum('PQ,PkA->QkA', oooo_pp, t3hr)                  # (ij<, k, nvtri)
    Yvir = np.zeros((nocc, nocc, nocc, nvtri), dtype=np.complex128)
    Yvir[mm, nn] = Xhh
    Yvir[nn, mm] = -Xhh
    _clib.pocc_to_pack(Rp, Yvir, 1.0, 1, nocc, nvir)
    return Rp


def energy(cc, t1, t2, eris):
    '''CCSD-form correlation energy (T3 enters only through t1/t2).'''
    nocc, nvir = t1.shape
    fock = eris.fock
    e = einsum('ia,ia', fock[:nocc, nocc:], t1)
    eris_oovv = _asarray(eris.oovv)
    e += 0.25 * einsum('ijab,ijab', t2, eris_oovv)
    e += 0.5 * einsum('ia,jb,ijab', t1, t1, eris_oovv)
    return e.real


def update_amps(cc, t1, t2, t3, eris):
    '''Next amplitudes (t1new, t2new, t3new).  Efficient tensor residuals:
    the CCSD T1/T2 part from socutils.cc.zccsd (validated), plus the T3->T1/T2
    couplings and the T3 residual in the T1-dressed formalism.

    ``t3`` is the *packed* unique block (i<j<k, a<b<c); it is unpacked once for
    the contractions and ``t3new`` is returned packed, so the iteration never
    holds more than one full O(o^3 v^3) copy at a time.  A full (6-D) ``t3`` is
    also accepted for backwards compatibility.'''
    assert isinstance(eris, _PhysicistsERIs)
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    t3p = t3 if t3.ndim == 1 else pack_t3(t3)
    t3 = t3 if t3.ndim == 6 else unpack_t3(t3p, nocc, nvir)
    o = slice(0, nocc)
    v = slice(nocc, nmo)
    mo_e = eris.mo_energy
    eia = mo_e[:nocc][:, None] - (mo_e[nocc:] + cc.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    # The bare antisymmetrized integral tensor g[p,q,r,s]=<pq||rs> depends only
    # on the (fixed) MO integrals, so build it once and cache it on eris.  Only
    # the T1-dressing below is redone each iteration (t1 changes).
    g = getattr(eris, '_ccsdt_g', None)
    if g is None:
        g = build_g(eris, nocc, nmo)
        eris._ccsdt_g = g
    fock = _asarray(eris.fock)
    ge, fdr = _t1_dress(g, fock, t1, nocc, nmo)

    # CCSD T1/T2 part (socutils, validated == pyscf CCSD)
    t1new, t2new = ZCCSD.update_amps(cc, t1, t2, eris)

    # T3 -> T1
    t1new = t1new + (0.25 * einsum('mnef,imnaef->ia', ge[o, o, v, v], t3)) / eia

    # T3 -> T2 (path-optimized einsum for the two t3-reading contractions)
    r2t3 = einsum('me,ijmabe->ijab', fdr[o, v], t3)
    tmp = 0.5 * _einsum_opt('amef,ijmebf->ijab', ge[v, o, v, v], t3)
    r2t3 = r2t3 + (tmp - tmp.transpose(0, 1, 3, 2))            # P(ab)
    tmp = 0.5 * _einsum_opt('mnej,inmabe->ijab', ge[o, o, v, o], t3)
    r2t3 = r2t3 - (tmp - tmp.transpose(1, 0, 2, 3))            # P(ij)
    t2new = t2new + r2t3 / eijab

    # T3 residual.  With the C backend, compute it on the unique antisymmetric
    # block (packed) -- ~2x faster and no full O(o^3 v^3) intermediate -- and
    # divide by the packed energy denominator before unpacking once.
    if _USE_CPACK:
        r3p = _t3_residual_packed(t2, t3, ge, fdr, nocc, nmo)
        oi, vi = _t3_tri(nocc, nvir)
        mo_o = mo_e[:nocc]; eav = mo_e[nocc:] + cc.level_shift
        eo = mo_o[oi[:, 0]] + mo_o[oi[:, 1]] + mo_o[oi[:, 2]]
        ev = eav[vi[:, 0]] + eav[vi[:, 1]] + eav[vi[:, 2]]
        d = (eo[:, None] - ev[None, :]).ravel()
        t3new = t3p + r3p / d                              # stays packed
    else:
        eijkabc = (eia[:, None, None, :, None, None]
                   + eia[None, :, None, None, :, None]
                   + eia[None, None, :, None, None, :])
        r3 = _t3_residual(t2, t3, ge, fdr, nocc, nmo)
        t3new = pack_t3(t3 + r3 / eijkabc)
    return t1new, t2new, t3new


class ZCCSDT(ZCCSD):
    '''Full spin-orbital CCSDT for two-component spinor CC (T1-dressed tensor).'''

    conv_tol = 1e-9
    conv_tol_normt = 1e-7
    max_cycle = 300

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None,
                 with_mmf=False, erifile=None):
        ZCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ,
                       with_mmf=with_mmf, erifile=erifile)
        self.t3 = None

    update_amps = update_amps
    energy = energy

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e = eris.mo_energy
        eia = mo_e[:nocc, None] - mo_e[None, nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        fov = eris.fock[:nocc, nocc:]
        t1 = fov.conj() / eia
        oovv = _asarray(eris.oovv).conj()
        t2 = oovv / eijab
        t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=t2.dtype)
        emp2 = 0.25 * einsum('ijab,ijab', t2, _asarray(eris.oovv))
        self.emp2 = emp2.real
        return self.emp2, t1, t2, t3

    def kernel(self, t1=None, t2=None, t3=None, eris=None):
        return self.ccsdt(t1, t2, t3, eris)

    def ccsdt(self, t1=None, t2=None, t3=None, eris=None):
        log = logger.new_logger(self)
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo(self.mo_coeff)
            self.eris = eris

        emp2, t1i, t2i, t3i = self.init_amps(eris)
        if t1 is None:
            t1 = t1i
        if t2 is None:
            t2 = t2i
        if t3 is None:
            t3 = t3i

        e_old = self.energy(t1, t2, eris)
        log.info('Init E_corr(CCSDT) = %.15g', e_old)

        # carry t3 packed (unique i<j<k, a<b<c) through the iteration so only a
        # single full O(o^3 v^3) copy ever exists (transiently, inside
        # update_amps); t1/t2 stay full.
        if t3.ndim == 6:
            t3 = pack_t3(t3)

        adiis = lib.diis.DIIS()
        adiis.space = 8

        conv = False
        for icycle in range(self.max_cycle):
            t1new, t2new, t3new = self.update_amps(t1, t2, t3, eris)
            # ||dt3_full|| = 6 ||dt3_packed|| exactly for antisymmetric t3, so
            # the convergence metric matches the full-tensor formulation.
            normt = (np.linalg.norm(t1new - t1)
                     + np.linalg.norm(t2new - t2)
                     + 6.0 * np.linalg.norm(t3new - t3))
            t1, t2, t3 = self.run_diis(t1new, t2new, t3new, adiis)
            e_new = self.energy(t1, t2, eris)
            de = e_new - e_old
            log.info('cycle = %d  E_corr(CCSDT) = %.15g  dE = %.3e  norm(t) = %.3e',
                     icycle + 1, e_new, de, normt)
            e_old = e_new
            if abs(de) < self.conv_tol and normt < self.conv_tol_normt:
                conv = True
                break

        self.converged = conv
        self.e_corr = e_old
        # expose the full t3 on the object for downstream consumers
        self.t1, self.t2, self.t3 = t1, t2, unpack_t3(t3, t1.shape[0], t1.shape[1])
        log.note('E_corr(CCSDT) = %.15g  converged = %s', e_old, conv)
        return self.e_corr, self.t1, self.t2, self.t3

    def amplitudes_to_vector(self, t1, t2, t3):
        # t2, t3 stored packed (unique i<j[<k], a<b[<c]) -> smaller DIIS history.
        # t3 may already be packed (1-D), as it is throughout the iteration.
        t3p = t3 if t3.ndim == 1 else pack_t3(t3)
        return np.hstack((t1.ravel(), pack_t2(t2), t3p))

    def vector_to_amplitudes(self, vec, nocc, nvir):
        n1 = nocc * nvir
        n2 = nocc * (nocc - 1) // 2 * (nvir * (nvir - 1) // 2)
        t1 = vec[:n1].reshape(nocc, nvir)
        t2 = unpack_t2(vec[n1:n1 + n2], nocc, nvir)
        t3 = vec[n1 + n2:].copy()                          # kept packed
        return t1, t2, t3

    def run_diis(self, t1, t2, t3, adiis):
        nocc, nvir = t1.shape
        vec = self.amplitudes_to_vector(t1, t2, t3)
        vec = adiis.update(vec)
        return self.vector_to_amplitudes(vec, nocc, nvir)


if __name__ == '__main__':
    from pyscf import gto
    from socutils.scf import spinor_hf
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='sto-3g', verbose=0)
    mf = spinor_hf.SCF(mol)
    mf.kernel()
    mycc = ZCCSDT(mf, frozen=2)
    print('E_corr(CCSDT) =', mycc.kernel()[0])
