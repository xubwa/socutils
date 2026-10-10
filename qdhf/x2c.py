#!/usr/bin/env python
# Copyright 2026 The socutils developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Spin-free exact two-component (sf-X2C-1e) Hartree-Fock in quadruple
precision.

The one-electron X2C decoupling follows pyscf.x2c.sfx2c1e.  With
xuncontract=True (default, as in pyscf) the decoupling is carried out in the
decontracted basis (one shell per unique exponent and angular momentum on
each atom) and the X2C Hamiltonian is contracted back,
h = C^T h_unc C, with the contraction coefficients C in quad precision.
With xuncontract=False the X2C transformation is done in the basis of mol.

    h4 = [[V, T], [T, W/(4c^2) - T]],  M4 = [[S, 0], [0, T/(2c^2)]]
    h4 C = M4 C E,   X = C_S C_L^{-1}  (positive energy solutions)
    S~ = S + X^T T X / (2c^2),  R = S^{-1/2} (S^{-1/2} S~ S^{-1/2})^{-1/2} S^{1/2}
    h_X2C = R^T (V + T X + X^T T - X^T T X + X^T W X / (4c^2)) R

with W = <p chi | V | . p chi> (pV.p).  approx = '1e' decouples the
molecular one-electron Dirac Hamiltonian; approx = 'atom1e' builds X from the
atomic one-electron problems (block-diagonal X), as in pyscf.

The two-electron interaction is the nonrelativistic Coulomb operator
(no picture change).
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from socutils.qdhf import qarray
from socutils.qdhf.qarray import QArray, qmp, hmp
from socutils.qdhf import integrals as qints
from socutils.qdhf import dhf


def _sym_pow(a, p):
    '''a^p for a real symmetric positive definite matrix'''
    w, v = qarray.eigh(a)
    v = v.real
    wp = [qmp.mpf(x)**p for x in w.to_mpmath()]
    return _QB.diag_scale(v, wp) @ v.T


_QB = dhf._QuadBackend


def _get_r(s, s1):
    '''R = S^{-1/2} (S^{-1/2} S~ S^{-1/2})^{-1/2} S^{1/2}'''
    s_isqrt = _sym_pow(s, qmp.mpf(-0.5))
    s_sqrt = _sym_pow(s, qmp.mpf(0.5))
    mid = _sym_pow(s_isqrt @ s1 @ s_isqrt, qmp.mpf(-0.5))
    return s_isqrt @ mid @ s_sqrt


def xmatrix(t, v, w, s, c):
    '''X matrix of the spin-free one-electron Dirac equation (RKB)'''
    n = s.shape[0]
    h4 = QArray.zeros((2*n, 2*n))
    m4 = QArray.zeros((2*n, 2*n))
    L = slice(0, n)
    S = slice(n, 2*n)
    h4[L, L] = v
    h4[L, S] = t
    h4[S, L] = t
    h4[S, S] = w * (1 / (4 * c**2)) - t
    m4[L, L] = s
    m4[S, S] = t * (1 / (2 * c**2))
    x = _sym_pow(m4, qmp.mpf(-0.5))
    e, a = qarray.eigh(x @ h4 @ x)
    a = x @ a.real
    cl = a[L, n:]
    cs = a[S, n:]
    # X C_L = C_S  <=>  C_L^T X^T = C_S^T
    return qarray.solve(cl.T, cs.T).T


def uncontract_mol(mol):
    '''Decontracted molecule and the quad precision coefficients C
    (xmol.nao x mol.nao) of the contracted AOs of mol in the AOs of xmol.

    Primitives with the same exponent and angular momentum on the same atom
    are merged (same as pyscf mol.decontract_basis(aggregate=True)).
    '''
    xbasis = {}
    for key, shells in mol._basis.items():
        prims = []
        for b in shells:
            if isinstance(b[1], (int, numpy.integer)):
                if b[1] != 0:
                    raise NotImplementedError('kappa-dependent spinor basis')
                b = [b[0]] + list(b[2:])
            l = b[0]
            for prim in b[1:]:
                if (l, prim[0]) not in prims:
                    prims.append((l, prim[0]))
        prims.sort(key=lambda x: (x[0], -x[1]))
        xbasis[key] = [[l, [e, 1.]] for l, e in prims]
    xmol = mol.copy(deep=False)
    xmol.build(False, False, basis=xbasis)

    xshell = {}
    for ib in range(xmol.nbas):
        assert xmol.bas_nprim(ib) == 1
        xshell[(xmol.bas_atom(ib), xmol.bas_angular(ib),
                float(xmol.bas_exp(ib)[0]))] = ib
    ao_loc = mol.ao_loc_nr(mol.cart)
    xao_loc = xmol.ao_loc_nr(xmol.cart)
    nao = ao_loc[-1]
    xnao = xao_loc[-1]
    c = numpy.empty((xnao, nao), dtype=object)
    c[:] = hmp.mpf(0)
    for ib, (ia, l, es, cs) in enumerate(qints.shell_data(mol)):
        cs = qints.normalized_coefficients(l, es, cs)
        nf = ao_loc[ib+1] - ao_loc[ib]
        nf //= cs.shape[1]
        for k, e in enumerate(es):
            # normalization of the uncontracted primitive in xmol
            nk = qints.normalized_coefficients(l, [e], numpy.ones((1, 1)))[0, 0]
            xb = xshell[(ia, l, float(e))]
            for ic in range(cs.shape[1]):
                for m in range(nf):
                    c[xao_loc[xb]+m, ao_loc[ib]+ic*nf+m] += cs[k, ic] / nk
    return xmol, QArray.from_mpf(c)


def get_xr(mol, light_speed=None, approx='1e', qi=None, omega=1.0):
    '''X and R matrices of sf-X2C-1e in the basis of mol (no decontraction).

    approx = '1e', 'atom1e' or 'rs1e'.  'rs1e' is the range-separated X2C-1e
    of Q. Sun (pyscf branch range-separated-x2c1e): every nucleus is split
    into V = V_sr + V_lr, V_lr = -Z erf(omega r)/r (the potential of a
    Gaussian countercharge exp(-omega^2 r^2)); X is built from V_sr only and
    V_lr is added untransformed, consistent with the untransformed
    electron-electron interaction.

    Returns a dict with x, r, the integrals t, v, w (pV.p), s and, for
    'rs1e', v_lr (v and w are then the short-range parts).
    '''
    if qi is None:
        qi = qints.QuadIntegrals(mol)
    c = dhf.get_light_speed(light_speed)
    n = qi.nao
    s = qi.ovlp_x()
    v = qi.nuc_x()
    sy = qi.ovlp_y()
    vy = qi.nuc_y()
    t = (sy[0::3, 0::3] + sy[1::3, 1::3] + sy[2::3, 2::3]) * hmp.mpf(0.5)
    w = vy[0::3, 0::3] + vy[1::3, 1::3] + vy[2::3, 2::3]

    v_lr = None
    if approx.upper() == 'RS1E':
        zeta = float(omega)**2
        v_lr = qi.nuc_x(zeta=zeta)
        vy_lr = qi.nuc_y(zeta=zeta)
        v = v - v_lr
        w = w - (vy_lr[0::3, 0::3] + vy_lr[1::3, 1::3] + vy_lr[2::3, 2::3])
        x = xmatrix(t, v, w, s, c)
    elif approx.upper() == '1E':
        x = xmatrix(t, v, w, s, c)
    elif approx.upper() in ('ATOM1E', 'ATOM'):
        x = QArray.zeros((n, n))
        aoslices = mol.aoslice_by_atom() if not mol.cart else mol.aoslice_nr_by_atom()
        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            if p0 == p1:
                continue
            sl = slice(p0, p1)
            v1 = qi.nuc_x(atoms=[ia])[sl, sl]
            vy1 = qi.nuc_y(atoms=[ia])[p0*3:p1*3, p0*3:p1*3]
            w1 = vy1[0::3, 0::3] + vy1[1::3, 1::3] + vy1[2::3, 2::3]
            x[sl, sl] = xmatrix(t[sl, sl], v1, w1, s[sl, sl], c)
    else:
        raise ValueError('Unknown approx %s' % approx)

    s1 = s + x.T @ t @ x * (1 / (2 * c**2))
    r = _get_r(s, s1)
    return {'x': x, 'r': r, 't': t, 'v': v, 'w': w, 's': s, 'c': c, 'qi': qi,
            'v_lr': v_lr}


def hcore_from_xr(d):
    '''h_X2C = R^T (V + T X + X^T T - X^T T X + X^T W X/(4c^2)) R'''
    x, r, t, v, w, c = d['x'], d['r'], d['t'], d['v'], d['w'], d['c']
    tx = t @ x
    h1 = v + tx + tx.T - x.T @ tx + x.T @ w @ x * (1 / (4 * c**2))
    h = r.T @ h1 @ r
    if d.get('v_lr') is not None:
        h = h + d['v_lr']
    return h


def picture_change(d, v_op, w_op):
    '''Picture-change transformed multiplicative operator
    R^T (O + X^T W_O X / (4c^2)) R, with W_O = <p chi| O |p chi>'''
    x, r, c = d['x'], d['r'], d['c']
    return r.T @ (v_op + x.T @ w_op @ x * (1 / (4 * c**2))) @ r


def get_hcore(mol, light_speed=None, approx='1e', qi=None, xuncontract=True, omega=1.0):
    '''sf-X2C-1e one-electron Hamiltonian in quad precision'''
    if xuncontract:
        xmol, contr = uncontract_mol(mol)
        h = get_hcore(xmol, light_speed, approx, xuncontract=False, omega=omega)
        return contr.T @ h @ contr
    return hcore_from_xr(get_xr(mol, light_speed, approx, qi, omega))


class SFX2C1E_RHF(dhf.RHF):
    '''Quad precision closed-shell RHF with the sf-X2C-1e Hamiltonian

    Attributes:
        approx : '1e' (default, molecular decoupling), 'atom1e' or 'rs1e'
        omega : range-separation parameter of 'rs1e' (default 1.0)
        xuncontract : decouple in the decontracted basis (default True)
        light_speed : speed of light (default lib.param.LIGHT_SPEED)
    '''
    approx = '1e'
    xuncontract = True
    omega = 1.0

    def __init__(self, mol, light_speed=None):
        dhf.RHF.__init__(self, mol)
        self.light_speed = light_speed

    def dump_flags(self, verbose=None):
        dhf.RHF.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('X2C approx = %s', self.approx)
        log.info('X2C xuncontract = %s', self.xuncontract)
        log.info('light speed = %s', qmp.nstr(qmp.mpf(dhf.get_light_speed(self.light_speed)), 34))
        return self

    def build_hamiltonian(self):
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        qi = qints.QuadIntegrals(mol)
        h = get_hcore(mol, self.light_speed, self.approx, qi, self.xuncontract, self.omega)
        t0 = log.timer('quad sf-X2C-1e hcore', *t0)
        eri = {('X', 'X'): qi.eri_xxxx()}
        log.timer('quad 2e integrals', *t0)
        e_nuc = qmp.mpf(qints.energy_nuc(mol))
        comps = [(QArray.eye(qi.nao), 'X')]
        self.ham = dhf.Hamiltonian4c(h, qi.ovlp_x(), comps, eri, e_nuc, None,
                                     k_factor=0.5, occ_number=2)
        return self.ham

    def _pyscf_guess(self):
        from pyscf import scf
        mf = scf.RHF(self.mol).sfx2c1e()
        mf.with_x2c.xuncontract = self.xuncontract
        mf.with_x2c.approx = self.approx
        mf.verbose = 0
        mf.conv_tol = 1e-11
        c0 = lib.param.LIGHT_SPEED
        try:
            lib.param.LIGHT_SPEED = float(dhf.get_light_speed(self.light_speed))
            mf.kernel()
        finally:
            lib.param.LIGHT_SPEED = c0
        self.e_tot_double = mf.e_tot
        return mf.make_rdm1()

SFX2C1E = SFX2C1E_RHF
