'''
Orbital gradient of the second-order energy of PC-NEVPT2 at fixed CI vectors, class by
class, from the derivatives with respect to the integrals (Python reference, all
amplitudes in memory; real orbitals of the spin-orbital route, zmr.nr.density).

E2 as a function of the MO integrals h[p, q] and B[P, p, q] ((pq|rs) = sum_P B[P, p, q]
B[P, r, s]):

    D[p, q]    = dE2 / dh[p, q]        the unrelaxed density (zmr.uc.density)
    G[P, p, q] = dE2 / dB[P, p, q]     the explicit part (the integrals in v) plus the part
                                        through the Fock matrices (chain rule with D)

For orbitals phi' = phi (1 + K): h' = h + K^T h + h K, B' likewise, so

    dE2 / dK[r, q] = (h (D + D^T) + sum_P B[P] (G[P] + G[P]^T))[r, q]

and in the parameters of PySCF's CASSCF (K[p, q] = x, K[q, p] = -x for p > q)
dE2 / dx = dE2/dK[p, q] - dE2/dK[q, p].
'''

import numpy

from socutils.zmr.uc import density
from socutils.zmr.uc.classes import _slice


def mo_vectors(eris):
    '''B[P, p, q] of all correlated orbitals of a factorized source.'''
    c = eris.mo_coeff
    b = eris.eri_fn.pair(c, c)
    n = c.shape[1]
    return numpy.asarray(b).reshape(-1, n, n)


def fock_chain(bmo, d_f, gamma):
    '''dE/dB[P] through F = h + J[gamma] - K[gamma] with dE/dF = d_f (all orbitals):
    J: d_f tr(gamma B) + gamma tr(d_f B);  K: -(d_f B gamma + gamma B d_f).'''
    tj_g = numpy.einsum('rs,Prs->P', gamma, bmo)
    tj_d = numpy.einsum('rs,Prs->P', d_f, bmo)
    out = (d_f[None] * tj_g[:, None, None] + gamma[None] * tj_d[:, None, None])
    out -= numpy.einsum('pq,Pqr,rs->Pps', d_f, bmo, gamma)
    out -= numpy.einsum('pq,Pqr,rs->Pps', gamma, bmo, d_f)
    return out


def ccvv_explicit(ctx):
    '''dE_ccvv/dB[P, a, i] (vc block): -2 sum_jb t[ij, ab] B[P, b, j], amplitudes of all
    ordered pairs; real integrals.'''
    sp = ctx.sp
    nc, nv = sp.ncore, sp.nvirt
    bvc = numpy.asarray(ctx.eris.cd_block('vc'))
    if numpy.iscomplexobj(bvc):
        if abs(bvc.imag).max() > 1e-12:
            raise NotImplementedError('real orbitals only')
        bvc = bvc.real
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    dab = ev[:, None] + ev[None, :]
    t = numpy.zeros((nc, nc, nv, nv))
    for i in range(nc):
        for j in range(nc):
            if i == j:
                continue
            m = _slice(bvc, i).T @ _slice(bvc, j)              # (ai|bj)
            v = m - m.T
            t[i, j] = v / (dab - ec[i] - ec[j])
    # G[P, a, i] = -2 sum_{j b} t[i, j, a, b] B[P, b, j]
    return -2 * numpy.einsum('ijab,Pbj->Pai', t, bvc)


def ccvv_gradient(pt):
    '''(dE_ccvv/dK in the quasi-canonical orbitals of pt, those orbitals' rotation u from
    the input orbitals): the orbital gradient of ccvv alone.'''
    qc = pt.quasi_canonical
    eris = qc.eris
    sp = eris.spaces
    ctx = density.context(pt)
    _, dcc, dvv = density.ccvv(ctx, kramers=False)
    d = density.assemble(sp, {'cc': dcc, 'vv': dvv}).real
    bmo = mo_vectors(eris)
    if numpy.iscomplexobj(bmo):
        bmo = bmo.real
    g = numpy.zeros_like(bmo)
    g[:, sp.slice('v'), sp.slice('c')] = ccvv_explicit(ctx)
    # the denominators: the generalized Fock matrix of the core and the averaged active
    # density (the one of quasi_canonicalize)
    gamma = numpy.zeros((sp.nmo, sp.nmo))
    gamma[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    gamma[sp.slice('a'), sp.slice('a')] = numpy.asarray(pt.gamma).real
    g += fock_chain(bmo, d, gamma)
    h = eris.h1.real
    grad = h @ (d + d.T) + numpy.einsum('Prs,Psq->rq', bmo, g + g.transpose(0, 2, 1))
    return grad, numpy.asarray(qc.u)


# ----------------------------------------------------------------------
# all classes: the derivative X = dE2/d(pq|rs) collected over the full orbital space

def _w_blocks(pt, ctx, pclass, state=0):
    '''[(block index, W[e, X])] with W = dE/dv[e, X] = -2 sum_k c[e, k] t[k, X] for the
    blocks of pclass.blocks(ctx) (real integrals), and the first-order basis b, the
    Gram matrix g of the amplitudes over the external labels.'''
    b, mu, t, s = density.first_order_space(pt, pclass, state)
    out = []
    g = numpy.zeros((len(mu), len(mu)), dtype=numpy.complex128)
    for n, (delta, v) in enumerate(pclass.blocks(ctx)):
        u = v @ t.T
        c = u / (delta[:, None] + mu[None, :] - s)
        out.append((n, (-2 * c @ t).real))
        g += c.T @ c.conj()
    return out, b, g


def _scatter(name, ctx, blocks):
    '''A = dE/d<pr||qs> per class, as (labels of _antisym, array [p, r, q, s]).'''
    sp = ctx.sp
    nc, na, nv = sp.ncore, sp.nact, sp.nvirt
    ii, jj = numpy.triu_indices(nc, 1)
    aa, bb = numpy.triu_indices(nv, 1)
    tt, uu = numpy.triu_indices(na, 1)
    if name == 'ccvv':
        a = numpy.zeros((nv, nv, nc, nc))
        for n, w in blocks:                           # one pair (i < j) per block
            a[aa, bb, ii[n], jj[n]] += w[:, 0]
        return 'vvcc', a
    if name == 'ccav':
        a = numpy.zeros((nv, na, nc, nc))
        for n, w in blocks:                           # one pair (i < j) per block, v (a, t)
            a[:, :, ii[n], jj[n]] += w
        return 'vacc', a
    if name == 'cavv':
        a = numpy.zeros((nv, nv, nc, na))
        for n, w in blocks:                           # core i, v (ab, t) = -g[aa, bb, i, :]
            a[aa, bb, n, :] -= w
        return 'vvca', a
    if name == 'ccaa':
        a = numpy.zeros((na, na, nc, nc))
        for _, w in blocks:                           # v (ij, tu) = g[tt, uu][:, ii, jj].T
            a[tt[None, :], uu[None, :], ii[:, None], jj[:, None]] += w
        return 'aacc', a
    if name == 'aavv':
        a = numpy.zeros((nv, nv, na, na))
        for _, w in blocks:                           # v (ab, tu) = g[aa, bb][:, tt, uu]
            a[aa[:, None], bb[:, None], tt[None, :], uu[None, :]] += w
        return 'vvaa', a
    if name == 'cava':
        a = numpy.zeros((nv, na, nc, na))
        for n, w in blocks:                           # core i, v[a, 1 + t na + u] = g[a, t, i, u]
            a[:, :, n, :] += w[:, 1:].reshape(nv, na, na)
        return 'vaca', a
    if name == 'caaa':
        a = numpy.zeros((na, na, nc, na))
        for _, w in blocks:                           # v[i, na + tu na + v] = -g[tt, uu, i, v]
            x = w[:, na:].reshape(nc, len(tt), na)
            a[tt, uu] -= x.transpose(1, 0, 2)
        return 'aaca', a
    if name == 'aaav':
        a = numpy.zeros((nv, na, na, na))
        for _, w in blocks:                           # v[a, na + u ntv + tv] = g[a, u, tt, vv]
            x = w[:, na:].reshape(nv, na, len(tt))
            a[:, :, tt, uu] += x
        return 'vaaa', a
    raise ValueError(name)


def _add_antisym(x4, sp, labels, a):
    '''X[p, q, r, s] += A[p, r, q, s], X[p, s, r, q] -= A[p, r, q, s] for
    <pr||qs> = (pq|rs) - (ps|rq), labels (p, r, q, s).'''
    p, r, q, s = (sp.slice(x) for x in labels)
    x4[p, q, r, s] += a.transpose(0, 2, 1, 3)
    x4[p, s, r, q] -= a.transpose(0, 3, 1, 2)


def _active_two_body(pt, pclass, b, g, x4, sp):
    '''dE/d(tu|vw) through H_act = ... + 1/2 sum (tu|vw) t^+ v^+ w u in the first-order
    space and in the zeroth-order energy (as _active_block for the one-body part).'''
    from socutils.zmr.active import rdm as active_rdm
    act = pt.active_space
    ntarget = act.target(pt.nelec, pclass.dn)
    w, y = numpy.linalg.eigh(0.5 * (g + g.conj().T))
    keep = w > 1e-14 * max(1.0, float(abs(w).max()) if len(w) else 1.0)
    na = act.norb
    d2 = numpy.zeros((na,) * 4, dtype=numpy.complex128)
    if keep.any() and ntarget >= 2:
        vecs = numpy.ascontiguousarray(b @ y[:, keep])
        for m, wm in enumerate(w[keep]):
            d2 += wm * active_rdm.make_rdm(act, vecs[:, m], ntarget, 2, bra=0, ket=0)
    if pt.nelec >= 2:
        gam2 = active_rdm.make_rdm(act, pt.civecs[:, [0]], pt.nelec, 2, bra=0, ket=0)
        d2 -= numpy.trace(g) * gam2
    a = sp.slice('a')
    x4[a, a, a, a] += 0.5 * d2.real


def fock_part(eris, d_f, gamma, fock=None, return_w=False):
    '''The orbital gradient of sum_pq d_f[p, q] F[p, q] for F = h + (J - K)[gamma] (MO,
    gamma fixed in the MO basis): F (d_f + d_f^T) + 2 (J - K)[d_f] gamma, with the AO
    Coulomb and exchange builds of the integral source (density fitting): no MO
    three-index integrals.  fock: F itself if known (MO); return_w: also (J - K)[d_f].'''
    c = eris.mo_coeff
    if fock is None:
        dm_gamma = c @ numpy.asarray(gamma).T @ c.conj().T
        fock = c.conj().T @ (eris.hcore_ao + eris.veff_fn(dm_gamma)) @ c
    w = c.conj().T @ eris.veff_fn(c @ numpy.asarray(d_f).T @ c.conj().T) @ c
    grad = (fock @ (d_f + d_f.T) + 2 * w @ gamma).real
    return (grad, w) if return_w else grad


def gradient(pt, state=0, classes_=None, fock='jk'):
    '''(dE2/dK in the quasi-canonical orbitals of pt, u): the orbital gradient of the
    classes named in classes_ (default all eight) at fixed CI vector.'''
    from socutils.zmr.uc import classes as uc_classes
    qc = pt.quasi_canonical
    eris = qc.eris
    sp = eris.spaces
    ctx = density.context(pt)
    names = classes_ or ('ccvv',) + tuple(density._LABELS)
    n = sp.nmo
    x4 = numpy.zeros((n,) * 4)
    d_gen = numpy.zeros((n, n), dtype=numpy.complex128)
    d_core = numpy.zeros((n, n), dtype=numpy.complex128)
    for name in names:
        pclass = uc_classes.CLASS_BY_NAME[name]
        if name == 'ccvv':
            _, dcc, dvv = density.ccvv(ctx, kramers=False)
            d_gen += density.assemble(sp, {'cc': dcc, 'vv': dvv})
            # W of ccvv: v / Delta per pair (no first-order basis: O = 1)
            blocks = [(k, (-2 * v / delta[:, None]).real)
                      for k, (delta, v) in enumerate(pclass.blocks(ctx))]
            _add_antisym(x4, sp, *_scatter(name, ctx, blocks))
            continue
        _, dblk = density.class_reference(pt, ctx, name, state)
        for lab, x in dblk.items():
            target = d_gen if lab in ('cc', 'vv') else d_core
            target += density.assemble(sp, {lab: x})
        blocks, b, g = _w_blocks(pt, ctx, pclass, state)
        _add_antisym(x4, sp, *_scatter(name, ctx, blocks))
        _active_two_body(pt, pclass, b, g, x4, sp)
    bmo = mo_vectors(eris)
    bmo = bmo.real if numpy.iscomplexobj(bmo) else bmo
    gam_gen = numpy.zeros((n, n))
    gam_gen[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    gam_gen[sp.slice('a'), sp.slice('a')] = numpy.asarray(pt.gamma).real
    gam_core = numpy.zeros((n, n))
    gam_core[sp.slice('c'), sp.slice('c')] = numpy.eye(sp.ncore)
    gmat = numpy.einsum('pqrs,Prs->Ppq', x4 + x4.transpose(2, 3, 0, 1), bmo)
    if fock == 'jk':
        # the Fock matrices through AO Coulomb and exchange builds (h included)
        grad = (fock_part(eris, d_gen.real, gam_gen) + fock_part(eris, d_core.real, gam_core))
    else:
        gmat += fock_chain(bmo, d_gen.real, gam_gen) + fock_chain(bmo, d_core.real, gam_core)
        d = (d_gen + d_core).real
        grad = eris.h1.real @ (d + d.T)
    grad = grad + numpy.einsum('Prs,Psq->rq', bmo, gmat + gmat.transpose(0, 2, 1))
    return grad, numpy.asarray(qc.u)
