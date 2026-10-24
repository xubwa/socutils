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
    def const(v, K):
        v = numpy.asarray(v, dtype=float)
        ng = v.shape[-1] if v.ndim else 1
        v = numpy.broadcast_to(v, (ng,)).copy()
        return _Jet(v, numpy.zeros((K, ng)), numpy.zeros((K, K, ng)))

    def __add__(self, o):
        if isinstance(o, _Jet):
            return _Jet(self.v + o.v, self.d + o.d, self.h + o.h)
        return _Jet(self.v + o, self.d, self.h)
    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.v, -self.d, -self.h)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, _Jet):
            h = (self.h * o.v + self.v * o.h
                 + self.d[:, None] * o.d[None] + o.d[:, None] * self.d[None])
            return _Jet(self.v * o.v, self.d * o.v + self.v * o.d, h)
        return _Jet(self.v * o, self.d * o, self.h * o)
    __rmul__ = __mul__

    def sqrt(self):
        s = numpy.sqrt(self.v)
        d = self.d / (2 * s)
        h = self.h / (2 * s) - self.d[:, None] * self.d[None] / (4 * s**3)
        return _Jet(s, d, h)

    def recip(self):
        r = 1 / self.v
        d = -self.d * r**2
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


def becke_weight_derivs(grids):
    '''For each atom a: (coords, weights, w1, w2) of its atomic grid, with
    w1 = dw/dR (natm, 3, ng) and w2 = d2w/dR dR (natm, 3, natm, 3, ng).'''
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
        return _Jet(numpy.array([R[i, x]]), d, numpy.zeros((K, K, 1)))

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
                dist.append(_Jet.const(numpy.linalg.norm(u, axis=1), K))
                continue
            comp = []
            for x in range(3):
                c = _Jet.const(u[:, x], K) + (Rj[ia][x] - Rj[i][x]) * _Jet.const(numpy.ones(ng), K)
                comp.append(c)
            dist.append(sum(c * c for c in comp).sqrt())
        P = [_Jet.const(numpy.ones(ng), K) for _ in range(natm)]
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
               W.h.reshape(natm, 3, natm, 3, ng))
