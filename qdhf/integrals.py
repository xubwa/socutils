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
Quadruple precision Gaussian integrals for the four-component
Dirac-Hartree-Fock method.

The basis set, the geometry and the nuclear model are taken from a PySCF
:class:`Mole` object.  The double precision numbers stored there
(exponents, contraction coefficients from the basis set data, coordinates
in Bohr, nuclear exponents) are taken as exact input; everything derived
from them (normalization factors, spherical harmonics, integrals) is
evaluated in quadruple precision.

Large component basis:  chi_mu (spherical or Cartesian GTOs, the same
functions and order as in libcint).  The small component basis is built
with the restricted kinetic balance (RKB) condition, chi^S = (sigma.p) chi/(2c),
which only requires the scalar integrals over the derivative functions
d chi_mu / d r_a.  The derivatives are expanded exactly in Cartesian
Gaussians with angular momenta l+1 and l-1.
'''

import ctypes
import math
from fractions import Fraction
import numpy
from pyscf import gto
from pyscf.gto import mole
from socutils.qdhf import qarray
from socutils.qdhf.qarray import QArray, QDTYPE, hmp, libqd

MAXL = 5  # max angular momentum supported by the C kernels (aux shells)
BAS_SLOTS = 5


def cart_comps(l):
    '''Cartesian components (lx, ly, lz) in the libcint order'''
    return [(i, j, l-i-j) for i in range(l, -1, -1) for j in range(l-i, -1, -1)]


def _poly_mul(p1, p2):
    out = {}
    for k1, v1 in p1.items():
        for k2, v2 in p2.items():
            k = (k1[0]+k2[0], k1[1]+k2[1], k1[2]+k2[2])
            out[k] = out.get(k, 0) + v1 * v2
    return out


def _solid_harmonic_poly(l, m):
    '''Unnormalized real regular solid harmonic r^l P_l^|m| {cos,sin}(|m|phi)
    as a polynomial {(i,j,k): Fraction}'''
    am = abs(m)
    # Q(x,y,z) = r^(l-am) d^am P_l(t)/dt^am, t = z/r
    q = {}
    for k in range(l // 2 + 1):
        n = l - 2 * k
        if n < am:
            continue
        b = Fraction((-1)**k * math.comb(l, k) * math.comb(2*l-2*k, l), 2**l)
        b *= Fraction(math.factorial(n), math.factorial(n - am))
        # z^(n-am) * (x^2+y^2+z^2)^k
        rk = {(0, 0, n-am): Fraction(1)}
        for _ in range(k):
            rk = _poly_mul(rk, {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 2): 1})
        for key, v in rk.items():
            q[key] = q.get(key, 0) + b * v
    # (x + i y)^am
    xy = {}
    for j in range(am + 1):
        c = math.comb(am, j)
        if m >= 0 and j % 2 == 0:
            xy[(am-j, j, 0)] = Fraction(c * (-1)**(j//2))
        elif m < 0 and j % 2 == 1:
            xy[(am-j, j, 0)] = Fraction(c * (-1)**((j-1)//2))
    return _poly_mul(xy, q)


_c2s_cache = {}
def cart2sph(l):
    '''Cartesian to real spherical transformation in hmp precision,
    c2s[cart, m], same convention (ordering, normalization, sign) as
    pyscf.gto.cart2sph(l)'''
    if l in _c2s_cache:
        return _c2s_cache[l]
    comps = cart_comps(l)
    pi4 = 4 * hmp.pi
    if l == 0:
        c2s = numpy.array([[1/hmp.sqrt(pi4)]], dtype=object)
    elif l == 1:
        c2s = numpy.zeros((3, 3), dtype=object)
        c2s[:] = hmp.mpf(0)
        for i in range(3):
            c2s[i, i] = hmp.sqrt(3 / pi4)
    else:
        ref = gto.cart2sph(l)
        idx = {c: i for i, c in enumerate(comps)}
        c2s = numpy.empty((len(comps), 2*l+1), dtype=object)
        c2s[:] = hmp.mpf(0)
        for im, m in enumerate(range(-l, l+1)):
            am = abs(m)
            norm = (2*l+1) / pi4 * hmp.factorial(l-am) / hmp.factorial(l+am)
            if m != 0:
                norm *= 2
            norm = hmp.sqrt(norm)
            for key, v in _solid_harmonic_poly(l, m).items():
                if v != 0:
                    c2s[idx[key], im] = norm * hmp.mpf(v.numerator) / v.denominator
            # align the phase with the libcint convention
            col = numpy.array([float(x) for x in c2s[:, im]])
            imax = numpy.argmax(abs(col))
            if col[imax] * ref[imax, im] < 0:
                c2s[:, im] = -c2s[:, im]
                col = -col
            if abs(col - ref[:, im]).max() > 1e-12:
                raise RuntimeError('cart2sph convention mismatch for l=%d' % l)
    _c2s_cache[l] = c2s
    return c2s


def _gaussian_int(n, a):
    '''int_0^inf x^n exp(-a x^2) dx'''
    n1 = hmp.mpf(n + 1) / 2
    return hmp.gamma(n1) / (2 * a**n1)


def _find_basis(mol, symb):
    basis = mol._basis
    puresymb = mole._rm_digit(symb)
    if symb in basis:
        return basis[symb]
    if puresymb in basis:
        return basis[puresymb]
    if symb[:2].upper() == 'X-':
        symb = symb[2:]
    elif symb[:6].upper() == 'GHOST-':
        symb = symb[6:]
    puresymb = mole._rm_digit(symb)
    if symb in basis:
        return basis[symb]
    if puresymb in basis:
        return basis[puresymb]
    return []


def shell_data(mol):
    '''Shells of mol in the order of mol._bas, with the raw data of the basis
    set (exponents and unnormalized contraction coefficients).
    Returns a list of (atom_id, l, exps[nprim], coefs[nprim,nctr]).
    '''
    shells = []
    for ia, atom in enumerate(mol._atom):
        for b in _find_basis(mol, atom[0]):
            angl = b[0]
            if isinstance(b[1], (int, numpy.integer)):
                kappa = b[1]
                b_coeff = numpy.array(sorted(b[2:], reverse=True))
            else:
                kappa = 0
                b_coeff = numpy.array(sorted(b[1:], reverse=True))
            if kappa != 0:
                raise NotImplementedError('kappa-dependent spinor basis')
            shells.append((ia, angl, b_coeff[:,0], b_coeff[:,1:]))
    if len(shells) != mol.nbas:
        raise RuntimeError('Failed to map the basis set of mol')
    for ib, (ia, l, es, cs) in enumerate(shells):
        if (ia != mol.bas_atom(ib) or l != mol.bas_angular(ib) or
            not numpy.array_equal(es, mol.bas_exp(ib)) or
            cs.shape[1] != mol.bas_nctr(ib)):
            raise RuntimeError('Failed to map the basis set of mol')
    return shells


def normalized_coefficients(l, es, cs):
    '''Contraction coefficients including the radial normalization, in hmp
    precision.  Same convention as pyscf.gto.mole.make_bas_env'''
    es = [hmp.mpf(float(e)) for e in es]
    nprim, nctr = cs.shape
    out = numpy.empty((nprim, nctr), dtype=object)
    for ic in range(nctr):
        c = [hmp.mpf(float(cs[ip,ic])) / hmp.sqrt(_gaussian_int(2*l+2, 2*es[ip]))
             for ip in range(nprim)]
        if mole.NORMALIZE_GTO:
            s = hmp.mpf(0)
            for i in range(nprim):
                for j in range(nprim):
                    s += c[i] * c[j] * _gaussian_int(2*l+2, es[i]+es[j])
            s = 1 / hmp.sqrt(s)
            c = [x * s for x in c]
        out[:,ic] = c
    return out


class QuadIntegrals:
    '''Quad precision scalar integrals over the large component functions
    chi_mu (X space) and their derivatives d chi_mu/d r_a (Y space).

    Attributes:
        nao : number of large component (scalar) functions
        X functions are ordered as mol AOs.  Y function index = 3*mu + a.
    '''
    def __init__(self, mol):
        if mol.cart:
            nao = mol.nao_cart()
        else:
            nao = mol.nao_nr()
        self.mol = mol
        self.nao = nao
        self._build()

    def _build(self):
        mol = self.mol
        env = []
        bas = []
        nao = self.nao

        def add_shell(l, es, cs, ia):
            bas.append([l, len(es), len(env), len(env)+len(es), len(env)+2*len(es)])
            env.extend(es)
            env.extend(cs)
            env.extend(hmp.mpf(float(x)) for x in mol.atom_coord(ia))
            return len(bas) - 1

        shls_L = []
        shls_S = []
        cx_entries = []   # (row, aux_cart_index, value) for the X space
        cy_entries = []   # (row, aux_cart_index, value) for the Y space
        offL = 0
        offS = 0
        ao_loc = mol.ao_loc_nr(mol.cart)
        for ib, (ia, l, es, cs) in enumerate(shell_data(mol)):
            if l + 1 > MAXL:
                raise NotImplementedError('Angular momentum l=%d not supported' % l)
            cs = normalized_coefficients(l, es, cs)
            es_h = [hmp.mpf(float(e)) for e in es]
            comps = cart_comps(l)
            if mol.cart:
                fac = cart2sph(l)[0, 0] if l <= 1 else hmp.mpf(1)
                ang = numpy.empty((len(comps), len(comps)), dtype=object)
                ang[:] = hmp.mpf(0)
                for i in range(len(comps)):
                    ang[i, i] = fac
            else:
                ang = cart2sph(l).T.copy()
            nf = ang.shape[0]
            idx_p = {c: i for i, c in enumerate(cart_comps(l+1))}
            idx_m = {c: i for i, c in enumerate(cart_comps(l-1))} if l > 0 else {}
            for ic in range(cs.shape[1]):
                c = list(cs[:, ic])
                shls_L.append(add_shell(l, es_h, c, ia))
                shls_S.append(add_shell(l+1, es_h, [-2*e*x for e, x in zip(es_h, c)], ia))
                p0 = offS
                offS += len(idx_p)
                if l > 0:
                    shls_S.append(add_shell(l-1, es_h, c, ia))
                    m0 = offS
                    offS += len(idx_m)
                ao0 = ao_loc[ib] + ic * nf
                for m in range(nf):
                    for k, (lx, ly, lz) in enumerate(comps):
                        a = ang[m, k]
                        if a == 0:
                            continue
                        cx_entries.append((ao0+m, offL+k, a))
                        lxyz = (lx, ly, lz)
                        for d in range(3):
                            row = (ao0 + m) * 3 + d
                            up = list(lxyz)
                            up[d] += 1
                            cy_entries.append((row, p0 + idx_p[tuple(up)], a))
                            if lxyz[d] > 0:
                                dn = list(lxyz)
                                dn[d] -= 1
                                cy_entries.append((row, m0 + idx_m[tuple(dn)], a * lxyz[d]))
                offL += len(comps)
        self._bas = numpy.asarray(bas, dtype=numpy.int32)
        self._env = qarray._raw_from_mpf(numpy.array(env, dtype=object))
        self.shls_L = numpy.asarray(shls_L, dtype=numpy.int32)
        self.shls_S = numpy.asarray(shls_S, dtype=numpy.int32)
        self.ncart_L = offL
        self.ncart_S = offS

        def build_c(entries, nrow, ncol):
            c = numpy.empty((nrow, ncol), dtype=object)
            c[:] = hmp.mpf(0)
            for r, k, v in entries:
                c[r, k] += v
            return QArray.from_mpf(c)
        self.cx = build_c(cx_entries, nao, offL)
        self.cy = build_c(cy_entries, nao*3, offS)

        charges = mol.atom_charges().astype(float)
        coords = mol.atom_coords()
        zetas = numpy.zeros(mol.natm)
        for ia in range(mol.natm):
            if mol._atm[ia, mole.NUC_MOD_OF] == mole.NUC_GAUSS:
                zetas[ia] = mol._env[mol._atm[ia, mole.PTR_ZETA]]
            elif mol._atm[ia, mole.NUC_MOD_OF] != mole.NUC_POINT:
                raise NotImplementedError('Nuclear model %d' % mol._atm[ia, mole.NUC_MOD_OF])
        self._charges = qarray._raw_from_double(charges)
        self._coords = qarray._raw_from_double(coords)
        self._zetas = qarray._raw_from_double(zetas)

    # ------------------------------------------------------------ raw ints
    def _int1e_cart(self, kind, shls_i, shls_j, atoms=None, zeta=None):
        ni = sum((self._bas[i,0]+1)*(self._bas[i,0]+2)//2 for i in shls_i)
        nj = sum((self._bas[i,0]+1)*(self._bas[i,0]+2)//2 for i in shls_j)
        out = numpy.empty((ni, nj), dtype=QDTYPE)
        ptr = qarray._ptr
        charges = self._charges
        zetas = self._zetas
        if zeta is not None:
            # Gaussian charge distributions exp(-zeta r^2) on all nuclei
            zetas = qarray._raw_from_double(numpy.full(self.mol.natm, float(zeta)))
        if atoms is not None:
            # nuclear attraction of the selected nuclei only
            z = self.mol.atom_charges().astype(float)
            mask = numpy.zeros_like(z)
            mask[list(atoms)] = 1
            charges = qarray._raw_from_double(z * mask)
        ierr = libqd.QDint1e(
            ctypes.c_int(kind), ptr(out),
            ptr(shls_i), ctypes.c_int(len(shls_i)),
            ptr(shls_j), ctypes.c_int(len(shls_j)),
            ptr(self._bas), ptr(self._env),
            ctypes.c_int(self.mol.natm), ptr(charges), ptr(self._coords),
            ptr(zetas))
        if ierr != 0:
            raise RuntimeError('QDint1e failed')
        return QArray(out)

    def _int2e_cart(self, shls, sym):
        ns = numpy.asarray([len(s) for s in shls], dtype=numpy.int32)
        nc = [sum((self._bas[i,0]+1)*(self._bas[i,0]+2)//2 for i in s) for s in shls]
        out = numpy.empty(nc, dtype=QDTYPE)
        ptr = qarray._ptr
        allshls = numpy.hstack(shls).astype(numpy.int32)
        ierr = libqd.QDint2e(ptr(out), ptr(allshls), ptr(ns), ctypes.c_int(sym),
                             ptr(self._bas), ptr(self._env))
        if ierr != 0:
            raise RuntimeError('QDint2e failed')
        return QArray(out)

    # ------------------------------------------------------------ public
    def ovlp_x(self):
        s = self._int1e_cart(0, self.shls_L, self.shls_L)
        return self.cx @ s @ self.cx.T

    def nuc_x(self, atoms=None, zeta=None):
        '''<chi_mu | V_nuc | chi_nu>.  atoms selects a subset of the nuclei.
        zeta: replace the nuclei by Gaussian charges exp(-zeta r^2), i.e.
        the long-range potential -Z erf(sqrt(zeta) r)/r'''
        v = self._int1e_cart(1, self.shls_L, self.shls_L, atoms, zeta)
        return self.cx @ v @ self.cx.T

    def ovlp_y(self):
        '''<d_a chi_mu | d_b chi_nu>,  index 3*mu+a'''
        s = self._int1e_cart(0, self.shls_S, self.shls_S)
        return self.cy @ s @ self.cy.T

    def nuc_y(self, atoms=None, zeta=None):
        '''<d_a chi_mu | V_nuc | d_b chi_nu>,  index 3*mu+a.  atoms selects a
        subset of the nuclei.  zeta: see nuc_x'''
        v = self._int1e_cart(1, self.shls_S, self.shls_S, atoms, zeta)
        return self.cy @ v @ self.cy.T

    def dip_x(self, comp):
        '''<chi_mu | r_comp | chi_nu> (origin at 0), comp = 0, 1, 2'''
        d = self._int1e_cart(2 + comp, self.shls_L, self.shls_L)
        return self.cx @ d @ self.cx.T

    def dip_y(self, comp):
        '''<d_a chi_mu | r_comp | d_b chi_nu>,  index 3*mu+a (origin at 0)'''
        d = self._int1e_cart(2 + comp, self.shls_S, self.shls_S)
        return self.cy @ d @ self.cy.T

    def get_jk_direct(self, dm):
        '''Integral-direct J, K in the X space for a real symmetric density
        matrix: J_ij = sum (ij|kl) dm_lk, K_il = sum (ij|kl) dm_jk'''
        dma = self.cx.T @ dm @ self.cx
        n = self.ncart_L
        vj = numpy.empty((n, n), dtype=QDTYPE)
        vk = numpy.empty((n, n), dtype=QDTYPE)
        ptr = qarray._ptr
        ierr = libqd.QDjk_direct(ptr(vj), ptr(vk), ptr(numpy.ascontiguousarray(dma.re)),
                                 ptr(self.shls_L), ctypes.c_int(len(self.shls_L)),
                                 ptr(self._bas), ptr(self._env))
        if ierr != 0:
            raise RuntimeError('QDjk_direct failed')
        return self.cx @ QArray(vj) @ self.cx.T, self.cx @ QArray(vk) @ self.cx.T

    def eri_xxxx(self):
        L = self.shls_L
        eri = self._int2e_cart([L, L, L, L], 7)
        return transform_eri(eri, self.cx, self.cx, self.cx, self.cx)

    def eri_yyxx(self):
        L = self.shls_L
        S = self.shls_S
        eri = self._int2e_cart([S, S, L, L], 3)
        return transform_eri(eri, self.cy, self.cy, self.cx, self.cx)

    def eri_yyyy(self):
        S = self.shls_S
        eri = self._int2e_cart([S, S, S, S], 7)
        return transform_eri(eri, self.cy, self.cy, self.cy, self.cy)


def transform_eri(eri, c1, c2, c3, c4):
    '''(ij|kl) -> sum c1[p,i] c2[q,j] c3[r,k] c4[s,l] (ij|kl)'''
    for c in (c4, c3, c2, c1):
        n1, n2, n3, n4 = eri.shape
        # contract the last index, then move the new index to the front
        x = qarray.matmul(eri.reshape(-1, n4), c, trans_b=True)
        eri = x.reshape(n1, n2, n3, -1).transpose(3, 0, 1, 2)
    return eri


def energy_nuc(mol):
    '''Nuclear repulsion energy (point charges) in hmp precision'''
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    e = hmp.mpf(0)
    for i in range(mol.natm):
        for j in range(i):
            r = hmp.sqrt(sum((hmp.mpf(float(coords[i,d])) - hmp.mpf(float(coords[j,d])))**2
                             for d in range(3)))
            e += hmp.mpf(int(charges[i])) * int(charges[j]) / r
    return e
