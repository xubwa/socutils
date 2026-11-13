#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Nuclear first and second derivatives of the spin-orbit ECP integrals
<mu| U_SO |nu> (pyscf intor 'ECPso', three components), which libcint/pyscf
do not provide.

The derivative of a Cartesian Gaussian is a combination of Gaussians,

    d/dx  (-2 alpha)^k x^a y^b z^c e^{-alpha r^2}
        = a (-2 alpha)^k x^(a-1) ... + (-2 alpha)^(k+1) x^(a+1) ... ,

so an n-th derivative of a shell of angular momentum l is a combination of
auxiliary shells l - n + 2k (k = 0..n) with contraction coefficients
(-2 alpha)^k c.  <d^m mu| U_SO |d^n nu> is therefore T_m^T <aux|U_SO|aux> T_n
with ECPso_cart integrals over the auxiliary shells and integer matrices T
(no new integral code), then transformed to spherical AOs.  libcint carries
the common factors of s and p functions in the Cartesian integrals, which the
auxiliary coefficients compensate.

With A(mu), B(nu) the AO centres, C the ECP centre and D_P the derivative
with respect to centre P (d phi(r - R) / dR = -d phi / dr):

    D_A = <d mu|.|nu>   D_B = <mu|.|d nu>   D_C = -(D_A + D_B)

(translational invariance), and the total derivative with respect to atom K
is the sum of D_P over the centres P sitting on K.  U_SO is real
anti-symmetric for any real functions, so <mu|U|d^n nu> = -<d^n nu|U|mu>.
The integrals of each ECP centre are taken separately by restricting the ECP
basis of the molecule to that centre.
'''

import itertools
import numpy
from pyscf.gto.mole import ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, ATOM_OF


def _cart_comps(l):
    '''(lx, ly, lz) of the Cartesian components of a shell, in pyscf order.'''
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _cart_index(l):
    return {c: i for i, c in enumerate(_cart_comps(l))}


def _common_fac(l):
    '''factor libcint applies to Cartesian s and p functions'''
    if l == 0:
        return 0.282094791773878143
    if l == 1:
        return 0.488602511902919921
    return 1.


def _deriv_terms(comp, axes):
    '''d/dr_axes[0] ... of (-2 alpha)^0 x^a y^b z^c e^{-alpha r^2}:
    {(k, (a', b', c')): integer coefficient} of (-2 alpha)^k x^a' y^b' z^c'.'''
    terms = {(0, tuple(comp)): 1}
    for x in axes:
        new = {}
        for (k, e), c in terms.items():
            if e[x] > 0:
                d = list(e)
                d[x] -= 1
                new[k, tuple(d)] = new.get((k, tuple(d)), 0) + c * e[x]
            u = list(e)
            u[x] += 1
            new[k + 1, tuple(u)] = new.get((k + 1, tuple(u)), 0) + c
        terms = new
    return terms


def _aux_shells(mol, n, env):
    '''Auxiliary shells for n-th derivatives: (bas rows, {(ish, k): index in
    the returned rows}).  Coefficients are appended to env.'''
    bas, where = [], {}
    for ish, b in enumerate(mol._bas):
        l, nprim, nctr = b[ANG_OF], b[NPRIM_OF], b[NCTR_OF]
        alpha = mol._env[b[PTR_EXP]:b[PTR_EXP] + nprim]
        c = mol._env[b[PTR_COEFF]:b[PTR_COEFF] + nprim * nctr].reshape(nctr, nprim)
        for k in range(n + 1):
            lk = l - n + 2 * k
            if lk < 0:
                continue
            ck = c * (-2 * alpha)**k * (_common_fac(l) / _common_fac(lk))
            nb = b.copy()
            nb[ANG_OF] = lk
            nb[PTR_COEFF] = len(env)
            env.extend(ck.ravel())
            where[ish, k] = len(bas)
            bas.append(nb)
    return bas, where


def _transforms(mol, n, bas, where, shl_offset, aux_loc):
    '''T[axes] (ncart_aux_all, ncart): d^n/dr_axes of every Cartesian AO of mol
    in terms of the auxiliary Cartesian functions.'''
    loc = mol.ao_loc_nr(cart=True)
    nall = aux_loc[-1]
    out = {}
    for axes in itertools.product(range(3), repeat=n):
        T = numpy.zeros((nall, loc[-1]))
        for ish, b in enumerate(mol._bas):
            l, nctr = b[ANG_OF], b[NCTR_OF]
            comps = _cart_comps(l)
            nc = len(comps)
            for ic, comp in enumerate(comps):
                for (k, e), coef in _deriv_terms(comp, axes).items():
                    lk = sum(e)
                    a = where[ish, k] + shl_offset
                    nck = (lk + 1) * (lk + 2) // 2
                    ie = _cart_index(lk)[e]
                    for ctr in range(nctr):
                        T[aux_loc[a] + ctr * nck + ie, loc[ish] + ctr * nc + ic] += coef
        out[axes] = T
    return out


def ecpso_derivs(mol, nbra, nket, ecp_atom=None):
    '''<d^nbra mu / dr^nbra| U_SO |d^nket nu / dr^nket>, shape
    (3,)*nbra + (3,)*nket + (3 [so], nao, nao), spherical (or Cartesian, if
    mol.cart) AOs; U_SO of ecp_atom only, or of all centres.'''
    molc = mol.copy()
    molc.cart = True
    env = list(mol._env)
    bra, wbra = _aux_shells(mol, nbra, env)
    ket, wket = _aux_shells(mol, nket, env)
    molc._bas = numpy.asarray(bra + ket, dtype=numpy.int32)
    molc._env = numpy.asarray(env)
    if ecp_atom is not None:
        molc._ecpbas = mol._ecpbas[mol._ecpbas[:, ATOM_OF] == ecp_atom]
    nb = len(bra)
    aux_loc = molc.ao_loc_nr(cart=True)
    ints = molc.intor('ECPso', comp=3, shls_slice=(0, nb, nb, molc.nbas))
    # bra rows: Cartesian offsets of the bra shells; ket rows: relative to the
    # first ket shell (the column offset of the shls_slice)
    Tb = _transforms(mol, nbra, bra, wbra, 0, aux_loc[:nb + 1])
    Tk = _transforms(mol, nket, ket, wket, 0, aux_loc[nb:] - aux_loc[nb])
    c2s = None if mol.cart else mol.cart2sph_coeff(normalized='sp')
    ncart = mol.ao_loc_nr(cart=True)[-1]
    nao = ncart if mol.cart else c2s.shape[1]
    out = numpy.zeros((3,) * (nbra + nket) + (3, nao, nao))
    for ab in Tb:
        for ak in Tk:
            m = numpy.einsum('pi,spq,qj->sij', Tb[ab], ints, Tk[ak])
            if c2s is not None:
                m = numpy.einsum('pi,spq,qj->sij', c2s, m, c2s)
            out[ab + ak] = m
    return out


def ecpso_ip(mol, ecp_atom=None):
    '''<d mu / dr_x| U_SO |nu>, (3 [x], 3 [so], nao, nao).'''
    return ecpso_derivs(mol, 1, 0, ecp_atom)


def _ecp_atoms(mol):
    return sorted(set(mol._ecpbas[:, ATOM_OF])) if mol.has_ecp_soc() else []


def ecpso_deriv_generator(mol):
    '''Returns h1(K) = d/dR_K <mu| U_SO |nu>, (3 [x], 3 [so], nao, nao).'''
    aoslices = mol.aoslice_by_atom()
    ecp_atoms = _ecp_atoms(mol)
    ip_all = ecpso_ip(mol)
    ip_c = {c: ecpso_ip(mol, c) for c in ecp_atoms}

    def ket(ip):
        # U_SO is real anti-symmetric for any real functions:
        # <mu|U|d nu> = - <d nu|U|mu>
        return -ip.transpose(0, 1, 3, 2)

    def hcore_deriv(K):
        p0, p1 = aoslices[K][2:]
        h = numpy.zeros_like(ip_all)
        h[:, :, p0:p1] -= ip_all[:, :, p0:p1]
        kt = ket(ip_all)
        h[:, :, :, p0:p1] -= kt[:, :, :, p0:p1]
        if K in ip_c:
            h += ip_c[K] + ket(ip_c[K])
        return h
    return hcore_deriv


def ecpso_deriv2_generator(mol):
    '''Returns h2(K, L) = d^2/dR_K,x dR_L,y <mu| U_SO |nu>,
    (3 [x], 3 [y], 3 [so], nao, nao).'''
    nao = mol.nao
    atom_of_ao = numpy.zeros(nao, dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        atom_of_ao[p0:p1] = ia
    blocks = {}
    for c in _ecp_atoms(mol):
        # D_A D_A = <dd mu|U|nu>, D_A,x D_B,y = <d_x mu|U|d_y nu>,
        # D_B D_B = <mu|U|dd nu> = -<dd nu|U|mu>
        aa = ecpso_derivs(mol, 2, 0, c)                        # (3, 3, 3, n, n)
        ab = ecpso_derivs(mol, 1, 1, c)
        bb = -aa.transpose(0, 1, 2, 4, 3)
        ba = ab.transpose(1, 0, 2, 3, 4)                       # D_B,x D_A,y
        M = {('A', 'A'): aa, ('A', 'B'): ab, ('B', 'A'): ba, ('B', 'B'): bb}
        # D_C = -(D_A + D_B)
        for P in 'AB':
            M[P, 'C'] = -(M[P, 'A'] + M[P, 'B'])
            M['C', P] = -(M['A', P] + M['B', P])
        M['C', 'C'] = M['A', 'A'] + M['A', 'B'] + M['B', 'A'] + M['B', 'B']
        blocks[c] = M

    def mask(P, K, c):
        '''(n, n) indicator that centre P of <mu|U_c|nu> sits on atom K'''
        if P == 'A':
            return numpy.repeat((atom_of_ao == K)[:, None], nao, axis=1)
        if P == 'B':
            return numpy.repeat((atom_of_ao == K)[None, :], nao, axis=0)
        return numpy.full((nao, nao), K == c)

    def hcore_deriv2(K, L):
        h = numpy.zeros((3, 3, 3, nao, nao))
        for c, M in blocks.items():
            for (P, Q), m in M.items():
                w = mask(P, K, c) & mask(Q, L, c)
                if w.any():
                    h += m * w
        return h
    return hcore_deriv2


def _pauli(d):
    from pyscf import lib
    s = .5 * lib.PauliMatrices
    lead = d.shape[:-3]
    nao = d.shape[-1]
    h = numpy.einsum('sxy,...spq->...xpyq', -1j * s, d)
    return h.reshape(lead + (2 * nao, 2 * nao))


def ghf_hcore_deriv(mol, K, gen=None):
    '''SO-ECP part of d hcore_GHF / dR_K, (3, 2n, 2n), with
    hcore_GHF += sum_s (-i/2) sigma_s <mu| U_SO^s |nu>  (pyscf.scf.ghf).'''
    if gen is None:
        gen = ecpso_deriv_generator(mol)
    return _pauli(gen(K))


def ghf_hcore_deriv2(mol, K, L, gen=None):
    '''SO-ECP part of d^2 hcore_GHF / dR_K dR_L, (3, 3, 2n, 2n).'''
    if gen is None:
        gen = ecpso_deriv2_generator(mol)
    return _pauli(gen(K, L))
