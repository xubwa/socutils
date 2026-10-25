#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
First and second derivatives of the Becke partition weights with respect to
the nuclear coordinates, for the grid response of exchange-correlation
Hessians.

The weights of the points owned by atom a are

    w_g = vol_g P_a(r_g) / sum_b P_b(r_g),    r_g = R_a + u_g,
    P_i = prod_{j != i} s(mu_ij),  mu_ij = (|r - R_i| - |r - R_j|) / R_ij,

with pyscf's radii adjustment mu -> mu + a_ij (1 - mu^2) and the Becke
smoothing s = (1 - p3)/2, p_{k+1} = (3 - p_k^2) p_k / 2 (original_becke).
The point moves with its atom, so every derivative includes the motion of
the grid.  The expression is a composition of distances, polynomials,
products and one reciprocal; it is differentiated exactly by second-order
forward-mode arithmetic (value, gradient, Hessian in the 3 natm nuclear
coordinates) rather than by hand.
'''

import numpy
from pyscf import gto
from pyscf.dft import gen_grid, radi


class _Jet:
    '''f, df/dX_k, d2f/dX_k dX_l for K nuclear coordinates on ng points.'''
    __slots__ = ('v', 'd', 'h')

    def __init__(self, v, d, h):
        self.v, self.d, self.h = v, d, h

    @staticmethod
    def const(v, K, order=2):
        v = numpy.asarray(v, dtype=float)
        ng = v.shape[-1] if v.ndim else 1
        v = numpy.broadcast_to(v, (ng,)).copy()
        return _Jet(v, numpy.zeros((K, ng)), numpy.zeros((K, K, ng)) if order > 1 else None)

    # h is None: first-order jet (no second derivatives carried)
    def __add__(self, o):
        if isinstance(o, _Jet):
            h = None if self.h is None else self.h + o.h
            return _Jet(self.v + o.v, self.d + o.d, h)
        return _Jet(self.v + o, self.d, self.h)
    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.v, -self.d, None if self.h is None else -self.h)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, _Jet):
            h = None
            if self.h is not None:
                h = (self.h * o.v + self.v * o.h
                     + self.d[:, None] * o.d[None] + o.d[:, None] * self.d[None])
            return _Jet(self.v * o.v, self.d * o.v + self.v * o.d, h)
        return _Jet(self.v * o, self.d * o, None if self.h is None else self.h * o)
    __rmul__ = __mul__

    def sqrt(self):
        s = numpy.sqrt(self.v)
        d = self.d / (2 * s)
        h = None
        if self.h is not None:
            h = self.h / (2 * s) - self.d[:, None] * self.d[None] / (4 * s**3)
        return _Jet(s, d, h)

    def recip(self):
        r = 1 / self.v
        d = -self.d * r**2
        h = None
        if self.h is not None:
            h = -self.h * r**2 + 2 * self.d[:, None] * self.d[None] * r**3
        return _Jet(r, d, h)


def _proton_numbers(mol):
    '''Atomic numbers, as pyscf.dft.radi uses for the radii adjustment of the
    grid itself.  mol.atom_charges() is the effective charge Z - n_core with an
    ECP (pyscf.grad.rks.grids_response_becke uses it, which is wrong then).'''
    from pyscf.data.elements import charge
    return numpy.asarray([charge(x) for x in mol.elements])


def proton_charge_mol(mol):
    '''Copy of mol whose atom_charges() are the atomic numbers, for pyscf's
    grid-response helpers (only the radii adjustment reads the charges).'''
    from pyscf.gto.mole import CHARGE_OF
    if not mol.has_ecp():
        return mol
    m = mol.copy()
    m._atm = m._atm.copy()
    m._atm[:, CHARGE_OF] = _proton_numbers(mol)
    return m


def grids_response_cc(grids):
    '''pyscf.grad.rks.grids_response_cc with the radii adjustment of the grid
    (atomic numbers, also with ECPs).'''
    import copy
    from pyscf.grad import rks as rks_grad
    if grids.mol.has_ecp():
        grids = copy.copy(grids)
        grids.mol = proton_charge_mol(grids.mol)
    return rks_grad.grids_response_cc(grids)


def _radii_adjust_table(grids):
    mol = grids.mol
    if grids.radii_adjust is None or grids.atomic_radii is None:
        return None
    charges = _proton_numbers(mol)
    if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
        rad = numpy.sqrt(grids.atomic_radii[charges]) + 1e-200
    elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
        rad = grids.atomic_radii[charges] + 1e-200
    else:
        raise NotImplementedError('radii_adjust %s' % grids.radii_adjust)
    rr = rad.reshape(-1, 1) * (1. / rad)
    a = .25 * (rr.T - rr)
    a[a < -.5] = -.5
    a[a > 0.5] = 0.5
    return a


def _pair_mu(u, Ra, Ri, Rj):
    '''mu_ij = (r_i - r_j) / R_ij on the points u + R_a and its derivatives
    with respect to the role coordinates (R_a, R_i, R_j) (9 local variables):
    mu (n,), dmu (9, n), and the pieces of d2mu (see _d2mu_contract).'''
    p = u + Ra
    di, dj = p - Ri, p - Rj
    ri = numpy.linalg.norm(di, axis=1)
    rj = numpy.linalg.norm(dj, axis=1)
    ni, nj = di.T / ri, dj.T / rj                            # (3, n)
    Rv = Ri - Rj
    R = numpy.linalg.norm(Rv)
    e = Rv / R
    t = 1 / R
    q = ri - rj
    n = u.shape[0]
    dq = numpy.vstack([ni - nj, -ni, nj])                     # (9, n)
    dR = numpy.concatenate([numpy.zeros(3), e, -e])           # (9,)
    dt = -t**2 * dR
    mu = q * t
    dmu = t * dq + q[None] * dt[:, None]
    return mu, dmu, (ri, rj, ni, nj, q, dq, t, dt, dR, e, R, n)


def _d2mu_contract(beta, parts):
    '''sum_g beta_g d2mu_g (9, 9).'''
    ri, rj, ni, nj, q, dq, t, dt, dR, e, R, n = parts
    I3 = numpy.eye(3)

    def msum(b, r, nn):
        # sum_g b (I - n n^T) / r
        br = b / r
        return I3 * br.sum() - numpy.einsum('g,xg,yg->xy', br, nn, nn)
    Mi, Mj = msum(beta, ri, ni), msum(beta, rj, nj)
    d2q = numpy.zeros((9, 9))
    d2q[0:3, 0:3] = Mi - Mj
    d2q[0:3, 3:6] = d2q[3:6, 0:3] = -Mi
    d2q[3:6, 3:6] = Mi
    d2q[0:3, 6:9] = d2q[6:9, 0:3] = Mj
    d2q[6:9, 6:9] = -Mj
    N = (I3 - numpy.outer(e, e)) / R
    d2R = numpy.zeros((9, 9))
    d2R[3:6, 3:6] = d2R[6:9, 6:9] = N
    d2R[3:6, 6:9] = d2R[6:9, 3:6] = -N
    d2t = -t**2 * d2R + 2 * t**3 * numpy.outer(dR, dR)
    bdq = dq.dot(beta)                                         # (9,)
    return t * d2q + numpy.outer(bdq, dt) + numpy.outer(dt, bdq) + beta.dot(q) * d2t


def _becke_log_s(nu_mu, a):
    '''For s_- = (1 - p3(nu))/2 and s_+ = (1 + p3(nu))/2, nu = mu + a (1 - mu^2):
    log s_+-, and their first and second derivatives with respect to mu.'''
    mu = nu_mu
    nu = mu + a * (1 - mu * mu)
    dnu = 1 - 2 * a * mu
    d2nu = -2 * a
    g, dg, d2g = nu, numpy.ones_like(nu), numpy.zeros_like(nu)
    for _ in range(3):
        f1 = 1.5 * (1 - g * g)                                 # f(x) = (3 - x^2) x / 2
        f2 = -3 * g
        g, dg, d2g = (3 - g * g) * g * .5, f1 * dg, f2 * dg * dg + f1 * d2g
    out = []
    for sign in (-1, 1):
        s = .5 * (1 + sign * g)
        # s ~ (1 - |nu|)^8 -> 0: P_i is then negligible (< 1e-30) and so are its
        # derivatives (~ s^(7/8)); treat the factor as constant there instead
        # of forming d log s ~ 1/(1 - |nu|), which overflows when squared
        small = s < 1e-30
        s = numpy.where(small, 1e-30, s)
        l1 = numpy.where(small, 0, sign * .5 * dg / s)          # d log s / d nu
        l2 = numpy.where(small, 0, sign * .5 * d2g / s - l1 * l1)
        out.append((numpy.log(s), l1 * dnu, l2 * dnu * dnu + l1 * d2nu))
    return out                                                 # [(s_-), (s_+)]


def _weight_hess_contract_loop(grids, eps_by_owner, blksize=1024):
    '''sum_g eps_g d2 w_g / dR_A,x dR_B,y  (natm, 3, natm, 3) for the Becke
    weights of becke_weight_derivs, without the per-point second derivatives:
    with L_i = log P_i = sum_j log s_ij and W = P_a / Z,

        d2 W / W = (G_a - z1)(G_a - z1)^T + H_a - sum_i (P_i / Z)(H_i + G_i G_i^T) + z1 z1^T,

    G_i = dL_i, H_i = d2L_i, z1 = sum_i (P_i / Z) G_i.  Each log s_ij depends on
    three atoms (owner a, i, j), so H_i is a sum of local 9 x 9 terms that are
    contracted with their weights before being scattered.  eps_by_owner[a]:
    the values on the points of atom a, in the order of becke_weight_derivs.'''
    mol = grids.mol
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('becke_scheme %s' % grids.becke_scheme)
    natm = mol.natm
    K = 3 * natm
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid, grids.radi_method,
                                            grids.level, grids.prune)
    R = mol.atom_coords()
    a_adj = _radii_adjust_table(grids)
    hess = numpy.zeros((K, K))
    pairs = [(i, j) for i in range(natm) for j in range(i)]

    def blocks(roles):
        return [slice(3 * x, 3 * x + 3) for x in roles]

    for ia in range(natm):
        u_all, vol_all = atom_grids_tab[mol.atom_symbol(ia)]
        eps_all = eps_by_owner[ia]
        for p0 in range(0, vol_all.size, blksize):
            u = u_all[p0:p0 + blksize]
            vol = vol_all[p0:p0 + blksize]
            eps = eps_all[p0:p0 + blksize]
            n = vol.size
            L = numpy.zeros((natm, n))
            G = numpy.zeros((natm, K, n))
            cache = []
            for i, j in pairs:
                mu, dmu, parts = _pair_mu(u, R[ia], R[i], R[j])
                aij = 0. if a_adj is None else float(a_adj[i, j])
                (lm, lm1, lm2), (lp, lp1, lp2) = _becke_log_s(mu, aij)
                L[i] += lm
                L[j] += lp
                bl = blocks((ia, i, j))
                for k, sl in enumerate(bl):
                    G[i, sl] += lm1 * dmu[3 * k:3 * k + 3]
                    G[j, sl] += lp1 * dmu[3 * k:3 * k + 3]
                cache.append((i, j, dmu, parts, lm1, lm2, lp1, lp2, bl))
            Lmax = L.max(axis=0)
            P = numpy.exp(L - Lmax)                            # common factor cancels in P / Z
            Z = P.sum(axis=0)
            pz = P / Z                                         # (natm, n)
            c = eps * vol * pz[ia]                             # eps_g w_g
            z1 = numpy.einsum('ig,ikg->kg', pz, G)
            D = G[ia] - z1
            h = (D * c).dot(D.T) + (z1 * c).dot(z1.T)
            for i in range(natm):
                h -= (G[i] * (c * pz[i])).dot(G[i].T)
            for i, j, dmu, parts, lm1, lm2, lp1, lp2, bl in cache:
                wi = c * ((i == ia) - pz[i])
                wj = c * ((j == ia) - pz[j])
                alpha = wi * lm2 + wj * lp2
                beta = wi * lm1 + wj * lp1
                hl = (dmu * alpha).dot(dmu.T) + _d2mu_contract(beta, parts)
                for k, sk in enumerate(bl):
                    for m, sm in enumerate(bl):
                        h[sk, sm] += hl[3 * k:3 * k + 3, 3 * m:3 * m + 3]
            hess += h
    return hess.reshape(natm, 3, natm, 3)


def _becke_pairs_vectorized(u, ia, R, a_adj, full=False):
    '''All pairs i > j at once on the points u + R_ia: log s_-, log s_+, their
    mu-derivatives and dmu with respect to the role coordinates (R_a, R_i,
    R_j): (pairs (i, j) arrays, lm, lm1, lp, lp1, dmu (npair, 9, n)).'''
    natm = R.shape[0]
    ii, jj = numpy.tril_indices(natm, -1)
    p = u + R[ia]
    d = p[None] - R[:, None]                                   # (natm, n, 3)
    r = numpy.linalg.norm(d, axis=2)                           # (natm, n)
    nvec = d / r[:, :, None]                                   # (natm, n, 3)
    Rv = R[ii] - R[jj]
    Rn = numpy.linalg.norm(Rv, axis=1)
    e = Rv / Rn[:, None]                                       # (npair, 3)
    t = 1 / Rn
    q = r[ii] - r[jj]                                          # (npair, n)
    mu = q * t[:, None]
    npair, n = mu.shape
    dmu = numpy.empty((npair, 9, n))
    ni, nj = nvec[ii].transpose(0, 2, 1), nvec[jj].transpose(0, 2, 1)     # (npair, 3, n)
    dt = (t * t)[:, None] * e                                  # d t / d R_j = +t^2 e, d / d R_i = -t^2 e
    dmu[:, 0:3] = t[:, None, None] * (ni - nj)
    dmu[:, 3:6] = -t[:, None, None] * ni - q[:, None] * dt[:, :, None]
    dmu[:, 6:9] = t[:, None, None] * nj + q[:, None] * dt[:, :, None]
    aij = numpy.zeros(npair) if a_adj is None else a_adj[ii, jj]
    (lm, lm1, lm2), (lp, lp1, lp2) = _becke_log_s(mu, aij[:, None])
    if full:
        return ii, jj, lm, lm1, lp, lp1, dmu, lm2, lp2, (r, nvec, e, t, q, Rn)
    return ii, jj, lm, lm1, lp, lp1, dmu


def _pair_hess_local(alpha, beta, dmu, ii, jj, parts):
    '''sum_g alpha_g dmu dmu^T + sum_g beta_g d2mu for all pairs, (npair, 9, 9),
    in the role coordinates (R_a, R_i, R_j).'''
    r, nvec, e, t, q, Rn = parts
    npair = ii.size
    I3 = numpy.eye(3)
    hl = numpy.einsum('pkg,pg,plg->pkl', dmu, alpha, dmu, optimize=True)

    def msum(atoms):
        # sum_g beta (I - n n^T) / r for the atom of each pair
        br = beta / r[atoms]                                   # (npair, n)
        nn = nvec[atoms]                                       # (npair, n, 3)
        return (I3[None] * br.sum(axis=1)[:, None, None]
                - numpy.einsum('pg,pgx,pgy->pxy', br, nn, nn, optimize=True))
    Mi, Mj = msum(ii), msum(jj)
    d2q = numpy.zeros((npair, 9, 9))
    d2q[:, 0:3, 0:3] = Mi - Mj
    d2q[:, 0:3, 3:6] = d2q[:, 3:6, 0:3] = -Mi
    d2q[:, 3:6, 3:6] = Mi
    d2q[:, 0:3, 6:9] = d2q[:, 6:9, 0:3] = Mj
    d2q[:, 6:9, 6:9] = -Mj
    N = (I3[None] - numpy.einsum('px,py->pxy', e, e)) / Rn[:, None, None]
    d2R = numpy.zeros((npair, 9, 9))
    d2R[:, 3:6, 3:6] = d2R[:, 6:9, 6:9] = N
    d2R[:, 3:6, 6:9] = d2R[:, 6:9, 3:6] = -N
    dR = numpy.zeros((npair, 9))
    dR[:, 3:6] = e
    dR[:, 6:9] = -e
    dt = -(t * t)[:, None] * dR
    d2t = (-(t * t)[:, None, None] * d2R
           + 2 * (t ** 3)[:, None, None] * numpy.einsum('pk,pl->pkl', dR, dR))
    ni, nj = nvec[ii].transpose(0, 2, 1), nvec[jj].transpose(0, 2, 1)     # (npair, 3, n)
    bdq = numpy.concatenate([numpy.einsum('pxg,pg->px', ni - nj, beta),
                             -numpy.einsum('pxg,pg->px', ni, beta),
                             numpy.einsum('pxg,pg->px', nj, beta)], axis=1)  # (npair, 9)
    hl += t[:, None, None] * d2q
    hl += numpy.einsum('pk,pl->pkl', bdq, dt) + numpy.einsum('pk,pl->pkl', dt, bdq)
    hl += (beta * q).sum(axis=1)[:, None, None] * d2t
    return hl


def weight_hess_contract(grids, eps_by_owner, blksize=1024):
    '''sum_g eps_g d2 w_g / dR_A,x dR_B,y  (natm, 3, natm, 3) for the Becke
    weights of becke_weight_derivs, without the per-point second derivatives:
    with L_i = log P_i = sum_j log s_ij and W = P_a / Z,

        d2 W / W = (G_a - z1)(G_a - z1)^T + H_a - sum_i (P_i / Z)(H_i + G_i G_i^T) + z1 z1^T,

    G_i = dL_i, H_i = d2L_i, z1 = sum_i (P_i / Z) G_i.  Each log s_ij depends on
    three atoms (owner a, i, j), so H_i is a sum of local 9 x 9 terms, which are
    contracted with their weights before being scattered; the outer products
    are weighted Gram matrices.  All atom pairs at once (arrays over pairs);
    eps_by_owner[a]: the values on the points of atom a, in the order of
    becke_weight_derivs.'''
    mol = grids.mol
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('becke_scheme %s' % grids.becke_scheme)
    natm = mol.natm
    K = 3 * natm
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid, grids.radi_method,
                                            grids.level, grids.prune)
    R = mol.atom_coords()
    a_adj = _radii_adjust_table(grids)
    ii, jj = numpy.tril_indices(natm, -1)
    npair = ii.size
    Ii = numpy.zeros((natm, npair))
    Ij = numpy.zeros((natm, npair))
    Ii[ii, numpy.arange(npair)] = 1
    Ij[jj, numpy.arange(npair)] = 1
    diag = numpy.arange(natm)
    x3 = numpy.arange(3)
    hess = numpy.zeros((K, K))
    for ia in range(natm):
        u_all, vol_all = atom_grids_tab[mol.atom_symbol(ia)]
        eps_all = eps_by_owner[ia]
        # coordinate indices of the three roles (owner, i, j) of every pair
        roles = [3 * numpy.full(npair, ia)[:, None] + x3, 3 * ii[:, None] + x3, 3 * jj[:, None] + x3]
        for p0 in range(0, vol_all.size, blksize):
            p1 = min(p0 + blksize, vol_all.size)
            n = p1 - p0
            u = u_all[p0:p1]
            ii, jj, lm, lm1, lp, lp1, dmu, lm2, lp2, parts = _becke_pairs_vectorized(
                u, ia, R, a_adj, full=True)
            L = Ii.dot(lm) + Ij.dot(lp)
            P = numpy.exp(L - L.max(axis=0))                   # common factor cancels in P / Z
            pz = P / P.sum(axis=0)                             # (natm, n)
            c = eps_all[p0:p1] * vol_all[p0:p1] * pz[ia]       # eps_g w_g
            # G[i] = d log P_i, (natm, natm, 3, n): role blocks of each pair
            X = lm1[:, None] * dmu                             # into G_i
            Y = lp1[:, None] * dmu                             # into G_j
            G = numpy.zeros((natm, natm, 3, n))
            G[:, ia] += (Ii.dot(X[:, 0:3].reshape(npair, -1))
                         + Ij.dot(Y[:, 0:3].reshape(npair, -1))).reshape(natm, 3, n)
            G[diag, diag] += (Ii.dot(X[:, 3:6].reshape(npair, -1))
                              + Ij.dot(Y[:, 6:9].reshape(npair, -1))).reshape(natm, 3, n)
            G[ii, jj] += X[:, 6:9]                             # one pair per (i, j)
            G[jj, ii] += Y[:, 3:6]
            G = G.reshape(natm, K, n)
            z1 = numpy.einsum('ig,ikg->kg', pz, G)
            D = G[ia] - z1
            h = (D * c).dot(D.T) + (z1 * c).dot(z1.T)
            for i in range(natm):
                h -= (G[i] * (c * pz[i])).dot(G[i].T)
            # local terms, weights ([i = a] - P_i / Z) c
            wi = c * ((ii == ia)[:, None] - pz[ii])
            wj = c * ((jj == ia)[:, None] - pz[jj])
            hl = _pair_hess_local(wi * lm2 + wj * lp2, wi * lm1 + wj * lp1, dmu, ii, jj, parts)
            for k in range(3):
                for m in range(3):
                    numpy.add.at(h, (roles[k][:, :, None], roles[m][:, None, :]),
                                 hl[:, 3 * k:3 * k + 3, 3 * m:3 * m + 3])
            hess += h
    return hess.reshape(natm, 3, natm, 3)


def becke_weight_derivs1(grids, blksize=1024):
    '''For each atom a: (coords, weights, w1) with w1 = dw/dR (natm, 3, ng) of
    becke_weight_derivs, analytically and vectorized over the atom pairs:

        dW = W (G_a - sum_i (P_i / Z) G_i),  G_i = d log P_i,

    i.e. dW = W sum_pairs [([i = a] - P_i/Z) L'_- + ([j = a] - P_j/Z) L'_+] dmu_ij,
    scattered to the atoms a, i, j.'''
    mol = grids.mol
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('becke_scheme %s' % grids.becke_scheme)
    natm = mol.natm
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid, grids.radi_method,
                                            grids.level, grids.prune)
    R = mol.atom_coords()
    a_adj = _radii_adjust_table(grids)
    ii, jj = numpy.tril_indices(natm, -1)
    # atom-pair incidence matrices: scatter over the pairs as a dgemm
    Ii = numpy.zeros((natm, ii.size))
    Ij = numpy.zeros((natm, ii.size))
    Ii[ii, numpy.arange(ii.size)] = 1
    Ij[jj, numpy.arange(ii.size)] = 1
    for ia in range(natm):
        u_all, vol_all = atom_grids_tab[mol.atom_symbol(ia)]
        ng = vol_all.size
        w = numpy.empty(ng)
        w1 = numpy.zeros((natm, 3, ng))
        for p0 in range(0, ng, blksize):
            p1 = min(p0 + blksize, ng)
            n = p1 - p0
            u = u_all[p0:p1]
            ii, jj, lm, lm1, lp, lp1, dmu = _becke_pairs_vectorized(u, ia, R, a_adj)
            L = Ii.dot(lm) + Ij.dot(lp)
            P = numpy.exp(L - L.max(axis=0))
            pz = P / P.sum(axis=0)
            W = vol_all[p0:p1] * pz[ia]
            alpha = ((ii == ia)[:, None] - pz[ii]) * lm1 + ((jj == ia)[:, None] - pz[jj]) * lp1
            ad = dmu * alpha[:, None, :]                       # (npair, 9, n)
            g = (Ii.dot(ad[:, 3:6].reshape(-1, 3 * n)) + Ij.dot(ad[:, 6:9].reshape(-1, 3 * n))).reshape(natm, 3, n)
            g[ia] += ad[:, 0:3].sum(axis=0)
            w[p0:p1] = W
            w1[:, :, p0:p1] = g * W
        yield u_all + R[ia], w, w1


def becke_weight_derivs(grids, order=2):
    '''For each atom a: (coords, weights, w1, w2) of its atomic grid, with
    w1 = dw/dR (natm, 3, ng) and w2 = d2w/dR dR (natm, 3, natm, 3, ng);
    order=1: first derivatives only (w2 is None).'''
    mol = grids.mol
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('becke_scheme %s' % grids.becke_scheme)
    natm = mol.natm
    K = 3 * natm
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid, grids.radi_method,
                                            grids.level, grids.prune)
    R = mol.atom_coords()
    a_adj = _radii_adjust_table(grids)

    # nuclear coordinates as jets (constant over the grid): dX/dX_k = e_k
    def coord_jet(i, x):
        d = numpy.zeros((K, 1))
        d[3 * i + x] = 1
        return _Jet(numpy.array([R[i, x]]), d, numpy.zeros((K, K, 1)) if order > 1 else None)

    Rj = [[coord_jet(i, x) for x in range(3)] for i in range(natm)]
    Rij = {}
    for i in range(natm):
        for j in range(i):
            d2 = sum((Rj[i][x] - Rj[j][x]) * (Rj[i][x] - Rj[j][x]) for x in range(3))
            Rij[i, j] = d2.sqrt().recip()                        # 1 / R_ij

    for ia in range(natm):
        u, vol = atom_grids_tab[mol.atom_symbol(ia)]
        ng = u.shape[0]
        # r_g - R_i = u_g + R_a - R_i
        dist = []
        for i in range(natm):
            if i == ia:
                dist.append(_Jet.const(numpy.linalg.norm(u, axis=1), K, order))
                continue
            comp = []
            for x in range(3):
                c = _Jet.const(u[:, x], K, order) + (Rj[ia][x] - Rj[i][x]) * _Jet.const(numpy.ones(ng), K, order)
                comp.append(c)
            dist.append(sum(c * c for c in comp).sqrt())
        P = [_Jet.const(numpy.ones(ng), K, order) for _ in range(natm)]
        for i in range(natm):
            for j in range(i):
                g = (dist[i] - dist[j]) * Rij[i, j]
                if a_adj is not None:
                    g = g + float(a_adj[i, j]) * (1 - g * g)
                for _ in range(3):
                    g = (3 - g * g) * g * .5
                P[i] = P[i] * (.5 * (1 - g))
                P[j] = P[j] * (.5 * (1 + g))
        W = P[ia] * sum(P[1:], P[0]).recip() * vol
        yield (u + R[ia], W.v, W.d.reshape(natm, 3, ng),
               None if W.h is None else W.h.reshape(natm, 3, natm, 3, ng))
