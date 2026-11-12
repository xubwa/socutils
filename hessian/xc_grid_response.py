#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Exchange-correlation contributions to the GKS Hessian with the full response
of the integration grid (analytic).

For a Kramers closed shell the collinear functional of (Re D_aa, Re D_bb) is
exact; everything here is a functional of those two real symmetric matrices.
Points of atom a sit at r_g = R_a + u_g with Becke weights w_g(R)
(socutils.grad.becke_deriv: exact first and second derivatives).  An AO mu on
atom b is phi_mu(r_g - R_b), so its total derivative is

    d phi / dR_A = c_A(mu) grad phi,   c_A(mu) = delta_Aa - delta_Ab,

and the second derivative c_A c_B grad grad phi: the motion of the point is
built into the AO derivatives.  Densities are quadratic forms of the AO
values; their mixed derivatives along (A, x), (B, y) are extracted exactly
from the displaced AOs  phi + e1 c_A d_x phi + e2 c_B d_y phi + e1 e2 c_A c_B
d_x d_y phi  with the four-point formula at e = +-1 (and the first
derivatives with the two-point formula), so no derivative of a density has to
be written out by hand.

    E''_AB = sum_g [ w''_AB eps + w'_A eps'_B + w'_B eps'_A + w eps''_AB ]
    eps'_A = v . rho'_A,   eps''_AB = rho'_A . f . rho'_B + v . rho''_AB

LDA, GGA and meta-GGA.  tau = 1/2 sum_r d_r phi D d_r phi is a quadratic form
of the same gradient rows, so the displaced AOs carry its derivatives as well.
'''

import itertools
import numpy
from pyscf import lib
from pyscf.dft import numint, gen_grid
from socutils.grad import becke_deriv

# AOs below this (all derivative rows) on a block of points are dropped, and
# per-atom potentials below WEIGHT_CUTOFF on a block are skipped
AO_CUTOFF = 1e-13
WEIGHT_CUTOFF = 1e-14

# position of the derivative multi-index (sorted axes) in pyscf's AO array
_AO_IDX = {(): 0}
_n = 1
for _k in (1, 2, 3, 4):
    for _c in itertools.combinations_with_replacement(range(3), _k):
        _AO_IDX[_c] = _n
        _n += 1


def _d(ao, axes):
    return ao[_AO_IDX[tuple(sorted(axes))]]


def _rows(xctype):
    '''AO derivative rows the densities are built from.'''
    return [()] if xctype == 'LDA' else [(), (0,), (1,), (2,)]


def _nvar(xctype):
    '''density rows: rho [, grad rho] [, tau]'''
    return {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]


def _displaced_ao(ao, rows, cA, x, e1, cB=None, y=None, e2=0.):
    '''AO rows of phi(e1, e2) = phi + e1 cA d_x phi + e2 cB d_y phi
    + e1 e2 cA cB d_x d_y phi  (cA, cB: per-AO coefficients).'''
    out = []
    for r in rows:
        a = _d(ao, r).copy()
        a += e1 * cA * _d(ao, r + (x,))
        if cB is not None:
            a += e2 * cB * _d(ao, r + (y,))
            a += e1 * e2 * cA * cB * _d(ao, r + (x, y))
        out.append(a)
    return numpy.asarray(out)


def _rho(mol, ao_rows, dm, xctype):
    a = ao_rows[0] if xctype == 'LDA' else ao_rows
    r = numint.eval_rho(mol, a, dm, xctype=xctype, hermi=1, with_lapl=False)
    return r.reshape(_nvar(xctype), -1)


def _vmat(ao_rows, wv, xctype):
    '''sum_g wv_r(g) d rho_r / d D_mu,nu (r = value [, gradient rows] [, tau])'''
    a0 = ao_rows[0]
    m = a0.T.dot(a0 * wv[0][:, None])
    if xctype != 'LDA':
        for r in range(1, 4):
            t = ao_rows[r].T.dot(a0 * wv[r][:, None])
            m += t + t.T
    if xctype == 'MGGA':
        for r in range(1, 4):
            m += .5 * ao_rows[r].T.dot(ao_rows[r] * wv[4][:, None])
    return m


# Bilinear forms.  With Q(U, V) = U^T D V pointwise (D symmetric), the density
# rows are B(phi, phi) of
#     B_0 = Q(U_0, V_0),  B_r = Q(U_r, V_0) + Q(U_0, V_r),  B_tau = 1/2 sum_r Q(U_r, V_r),
# so with the displacement delta = c_A d_x phi (a column scaling of the AOs)
#     d rho = 2 B(delta, phi),   d2 rho = 2 B(delta_1, delta_2) + 2 B(delta_12, phi).
# phi D is computed once per block; the per-atom work is element-wise.

def _phid(ao_rows, dm):
    return [a.dot(dm) for a in ao_rows]


def _drho1(ao, rows, xctype, c, phid):
    '''d rho / dR_A,x (natm, 3, nvar, ng) of a symmetric dm (phid = phi D),
    all atoms at once.'''
    nvar = _nvar(xctype)
    ng = phid[0].shape[0]
    out = numpy.empty((c.shape[0], 3, nvar, ng))
    for x in range(3):
        d = [_d(ao, r + (x,)) for r in rows]
        t = [2 * d[0] * phid[0]]
        if xctype != 'LDA':
            t += [2 * (d[k] * phid[0] + d[0] * phid[k]) for k in range(1, 4)]
        if xctype == 'MGGA':
            t.append(sum(d[k] * phid[k] for k in range(1, 4)))
        for v, tv in enumerate(t):
            out[:, x, v] = c.dot(tv.T)
    return out


def _hess_bilinear(ao, rows, xctype, c, wv, dm, phid):
    '''sum_g wv . d2 rho / dR_A,x dR_B,y  (natm, natm, 3, 3) for a symmetric dm,
    weights wv (nvar, ng): 2 B(delta_1, delta_2) as c^T (M o D) c with
    M = sum_ab (d_x phi_a)^T W_ab (d_y phi_b), and 2 B(delta_12, phi).'''
    natm = c.shape[0]
    h = numpy.zeros((natm, natm, 3, 3))
    lda = xctype == 'LDA'
    for x in range(3):
        dx = [_d(ao, r + (x,)) for r in rows]
        for y in range(3):
            dy = [_d(ao, r + (y,)) for r in rows]
            Y = [wv[0][:, None] * dy[0]]
            if not lda:
                for k in range(1, 4):
                    Y[0] = Y[0] + wv[k][:, None] * dy[k]
                    Y.append(wv[k][:, None] * dy[0])
                    if xctype == 'MGGA':
                        Y[k] = Y[k] + .5 * wv[4][:, None] * dy[k]
            M = sum(dx[a].T.dot(Y[a]) for a in range(len(Y)))
            h[:, :, x, y] += 2 * c.dot(M * dm).dot(c.T)
            dxy = [_d(ao, r + (x, y)) for r in rows]
            s0 = wv[0][:, None] * phid[0]
            if not lda:
                s0 = s0 + sum(wv[k][:, None] * phid[k] for k in range(1, 4))
            q = numpy.einsum('gm,gm->m', dxy[0], s0)
            if not lda:
                for k in range(1, 4):
                    sk = wv[k][:, None] * phid[0]
                    if xctype == 'MGGA':
                        sk = sk + .5 * wv[4][:, None] * phid[k]
                    q += numpy.einsum('gm,gm->m', dxy[k], sk)
            h[:, :, x, y] += 2 * numpy.einsum('Am,Bm,m->AB', c, c, q)
    return h


def _vmat_bilinear(U, V, wv, xctype):
    '''m(U, V) with m(phi, phi) = _vmat(phi): the matrix of the bilinear
    density form B(U, V) weighted with wv.'''
    m = U[0].T.dot(V[0] * wv[0][:, None])
    if xctype != 'LDA':
        for k in range(1, 4):
            m += U[k].T.dot(V[0] * wv[k][:, None]) + U[0].T.dot(V[k] * wv[k][:, None])
    if xctype == 'MGGA':
        for k in range(1, 4):
            m += .5 * U[k].T.dot(V[k] * wv[4][:, None])
    return m


def _blocks(mf, blksize=2000, order=2):
    if order == 1:
        gen = ((c, w, w1, None) for c, w, w1 in becke_deriv.becke_weight_derivs1(mf.grids))
    else:
        gen = becke_deriv.becke_weight_derivs(mf.grids, order)
    for coords, w, w1, w2 in gen:
        for p0, p1 in lib.prange(0, w.size, blksize):
            yield (p0 == 0, coords[p0:p1], w[p0:p1], w1[..., p0:p1],
                   None if w2 is None else w2[..., p0:p1])


def _coef(mol, owner):
    '''c[A, mu] = delta_A,owner - delta_A,atom(mu)'''
    aoslices = mol.aoslice_by_atom()
    c = numpy.zeros((mol.natm, mol.nao_nr()))
    c[owner] += 1
    for b in range(mol.natm):
        p0, p1 = aoslices[b][2:]
        c[b, p0:p1] -= 1
    return c


def xc_derivs_grid_response(mf, dm0, with_hess=True, with_fock=True):
    '''Returns (E''_xc (natm, natm, 3, 3), {atom: dVxc/dR (3, 2n, 2n)}) at the
    fixed GHF density dm0, with the full grid response.'''
    mol = mf.mol
    ni = mf._numint
    ni1c = ni._to_numint1c()
    xc = mf.xc
    xctype = ni1c._xc_type(xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError('grid response for %s' % xctype)
    if mf.grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('grid response with becke_scheme %s' % mf.grids.becke_scheme)
    natm = mol.natm
    nao = mol.nao_nr()
    rows = _rows(xctype)
    nvar = _nvar(xctype)
    dms = (numpy.ascontiguousarray(dm0[:nao, :nao].real),
           numpy.ascontiguousarray(dm0[nao:, nao:].real))
    ao_deriv = 2 if xctype == 'LDA' else 3
    hess = numpy.zeros((natm, natm, 3, 3))
    v1 = {ia: numpy.zeros((3, 2, nao, nao)) for ia in range(natm)}

    # second derivatives of the weights only contracted with eps, after the
    # loop (becke_deriv.weight_hess_contract): eps per point, by owner atom
    eps_owner = [[] for _ in range(natm)]
    owner = -1
    for first, coords, w, w1, _ in _blocks(mf, order=1):
        if first:
            owner += 1
            c = _coef(mol, owner)
        ao = ni1c.eval_ao(mol, coords, deriv=ao_deriv)
        # AO screening: only the AOs (any derivative row) significant on the block
        idx = numpy.where(abs(ao).max(axis=(0, 1)) > AO_CUTOFF)[0]
        ao = numpy.ascontiguousarray(ao[:, :, idx])
        cb = c[:, idx]
        dmb = [numpy.ascontiguousarray(d[idx[:, None], idx]) for d in dms]
        ix = numpy.ix_(idx, idx)
        ng = w.size
        ao0 = numpy.asarray([_d(ao, r) for r in rows])
        rho = numpy.asarray([_rho(mol, ao0, d, xctype) for d in dmb])         # (2, nvar, ng)
        rho_in = rho[:, 0] if xctype == 'LDA' else rho
        exc, vxc, fxc = ni1c.eval_xc_eff(xc, rho_in, deriv=2, xctype=xctype)[:3]
        vxc = vxc.reshape(2, nvar, -1)
        fxc = fxc.reshape(2 * nvar, 2 * nvar, -1)
        eps = exc * (rho[0, 0] + rho[1, 0])

        phid = [_phid(ao0, d) for d in dmb]
        # first derivatives of the densities, (3 natm, 2 nvar, ng)
        drho = numpy.stack([_drho1(ao, rows, xctype, cb, phid[s]) for s in range(2)], axis=2)
        drho = drho.reshape(3 * natm, 2 * nvar, ng)
        deps = (drho * vxc.reshape(1, 2 * nvar, ng)).sum(axis=1)               # (3 natm, ng)
        dv = numpy.zeros_like(drho)                                            # first-order potential
        for k in range(2 * nvar):
            dv += fxc[None, :, k] * drho[:, k:k + 1]
        w1f = w1.reshape(3 * natm, ng)

        if with_hess:
            eps_owner[owner].append(eps)
            # w1_A deps_B + deps_A w1_B + sum_g w drho_A . dv_B, as dgemms over (A, x)
            t = w1f.dot(deps.T)
            t = t + t.T + (drho * w).reshape(3 * natm, -1).dot(dv.reshape(3 * natm, -1).T)
            hess += t.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
            for s in range(2):
                hess += _hess_bilinear(ao, rows, xctype, cb, w * vxc[s], dmb[s], phid[s])

        if with_fock:
            dv = dv.reshape(natm, 3, 2, nvar, ng)
            # skip (A, x) whose potential on the block is negligible
            wmax = abs(w1).max(axis=2) * abs(vxc).max()                        # (natm, 3)
            dvmax = abs(dv).max(axis=(3, 4)) * abs(w).max()                    # (natm, 3, 2)
            for s in range(2):
                # d/dR of the potential matrix at fixed weights: m(delta, phi) + its transpose
                G = [_vmat_bilinear([_d(ao, r + (x,)) for r in rows], ao0, w * vxc[s], xctype)
                     for x in range(3)]
                for A in range(natm):
                    has_c = abs(cb[A]).max() > 0
                    for x in range(3):
                        m = 0
                        if wmax[A, x] > WEIGHT_CUTOFF or dvmax[A, x, s] > WEIGHT_CUTOFF:
                            m = _vmat(ao0, w1[A, x] * vxc[s] + w * dv[A, x, s], xctype)
                        if has_c:
                            t = cb[A][:, None] * G[x]
                            m = m + t + t.T
                        if not isinstance(m, int):
                            v1[A][x, s][ix] += m

    if with_hess:
        eps_owner = [numpy.concatenate(e) for e in eps_owner]
        hess += becke_deriv.weight_hess_contract(mf.grids, eps_owner).transpose(0, 2, 1, 3)

    fock1 = {}
    for ia in range(natm):
        f = numpy.zeros((3, 2 * nao, 2 * nao), dtype=numpy.complex128)
        f[:, :nao, :nao] = v1[ia][:, 0]
        f[:, nao:, nao:] = v1[ia][:, 1]
        fock1[ia] = f
    return hess, fock1
