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

LDA and GGA.
'''

import itertools
import numpy
from pyscf import lib
from pyscf.dft import numint, gen_grid
from socutils.grad import becke_deriv

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
    return [()] if xctype == 'LDA' else [(), (0,), (1,), (2,)]


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
    return r.reshape(len(ao_rows), -1)


def _vmat(ao_rows, wv, xctype):
    '''sum_g wv_r(g) d_r(phi_mu phi_nu) (r = value [, gradient rows])'''
    a0 = ao_rows[0]
    m = a0.T.dot(a0 * wv[0][:, None])
    if xctype != 'LDA':
        for r in range(1, 4):
            t = ao_rows[r].T.dot(a0 * wv[r][:, None])
            m += t + t.T
    return m


def _blocks(mf, blksize=2000):
    for coords, w, w1, w2 in becke_deriv.becke_weight_derivs(mf.grids):
        for p0, p1 in lib.prange(0, w.size, blksize):
            yield p0 == 0, coords[p0:p1], w[p0:p1], w1[..., p0:p1], w2[..., p0:p1]


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
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError('grid response for %s' % xctype)
    if mf.grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('grid response with becke_scheme %s' % mf.grids.becke_scheme)
    natm = mol.natm
    nao = mol.nao_nr()
    rows = _rows(xctype)
    nvar = len(rows)
    dms = (numpy.ascontiguousarray(dm0[:nao, :nao].real),
           numpy.ascontiguousarray(dm0[nao:, nao:].real))
    ao_deriv = 2 if xctype == 'LDA' else 3
    hess = numpy.zeros((natm, natm, 3, 3))
    v1 = {ia: numpy.zeros((3, 2, nao, nao)) for ia in range(natm)}

    owner = -1
    for first, coords, w, w1, w2 in _blocks(mf):
        if first:
            owner += 1
            c = _coef(mol, owner)
        ao = ni1c.eval_ao(mol, coords, deriv=ao_deriv)
        ao0 = numpy.asarray([_d(ao, r) for r in rows])
        rho = numpy.asarray([_rho(mol, ao0, d, xctype) for d in dms])         # (2, nvar, ng)
        rho_in = rho[:, 0] if xctype == 'LDA' else rho
        exc, vxc, fxc = ni1c.eval_xc_eff(xc, rho_in, deriv=2, xctype=xctype)[:3]
        vxc = vxc.reshape(2, nvar, -1)
        fxc = fxc.reshape(2, nvar, 2, nvar, -1)
        eps = exc * (rho[0, 0] + rho[1, 0])

        # first derivatives of the densities, (natm, 3, 2, nvar, ng)
        drho = numpy.zeros((natm, 3, 2, nvar, w.size))
        for A in range(natm):
            for x in range(3):
                ap = _displaced_ao(ao, rows, c[A], x, 1.)
                am = _displaced_ao(ao, rows, c[A], x, -1.)
                for s in range(2):
                    drho[A, x, s] = .5 * (_rho(mol, ap, dms[s], xctype) - _rho(mol, am, dms[s], xctype))
        deps = numpy.einsum('svg,Axsvg->Axg', vxc, drho)
        dv = numpy.einsum('svtug,Axtug->Axsvg', fxc, drho)                    # first-order potential

        if with_hess:
            hess += numpy.einsum('AxByg,g->ABxy', w2, eps)
            hess += numpy.einsum('Axg,Byg->ABxy', w1, deps)
            hess += numpy.einsum('Byg,Axg->ABxy', w1, deps)
            hess += numpy.einsum('g,Axsvg,Bysvg->ABxy', w, drho, dv)
            for A in range(natm):
                for x in range(3):
                    for B in range(A, natm):
                        for y in range(3):
                            if B == A and y < x:
                                continue
                            d2 = 0
                            for e1, e2, sg in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                                a = _displaced_ao(ao, rows, c[A], x, e1, c[B], y, e2)
                                d2 = d2 + sg * numpy.asarray([_rho(mol, a, d, xctype) for d in dms])
                            val = .25 * numpy.einsum('g,svg,svg', w, vxc, d2)
                            hess[A, B, x, y] += val
                            if (B, y) != (A, x):
                                hess[B, A, y, x] += val

        if with_fock:
            for A in range(natm):
                for x in range(3):
                    ap = _displaced_ao(ao, rows, c[A], x, 1.)
                    am = _displaced_ao(ao, rows, c[A], x, -1.)
                    for s in range(2):
                        wv = w1[A, x] * vxc[s] + w * dv[A, x, s]
                        m = _vmat(ao0, wv, xctype)
                        m += .5 * (_vmat(ap, w * vxc[s], xctype) - _vmat(am, w * vxc[s], xctype))
                        v1[A][x, s] += m

    fock1 = {}
    for ia in range(natm):
        f = numpy.zeros((3, 2 * nao, 2 * nao), dtype=numpy.complex128)
        f[:, :nao, :nao] = v1[ia][:, 0]
        f[:, nao:, nao:] = v1[ia][:, 1]
        fock1[ia] = f
    return hess, fock1
