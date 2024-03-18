#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Analytical nuclear gradients for X2C1E method

Ref.
JCP 135, 084114 (2011); DOI:10.1063/1.3624397
Note 1: Though the title of this paper is sfx2c, but changing the
W matrix to the spin dependent one will give the two-component gradient.
Note 2: Due to the spin-orbit coupling from X2CAMF method is an atom only term,
it will not contribute to the nuclear gradients, and can be used directly.
Note 3: This implementation uses the j-adapted spinor basis, i.e. can be used for SpinorX2CHelper directly,
for SpinOrbitX2CHelper, the function for class will translate the j-adapted spinor basis to the spin-orbital one.
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e_grad
from pyscf.x2c.sfx2c1e_grad import _get_r1
from pyscf.x2c.x2c import _sigma_dot, _block_diag
from scipy.linalg import norm

def hcore_grad_generator_spin_orbital(x2cobj, mol=None):
    '''nuclear gradients of 2-component X2C1E core Hamiltonian
    '''
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff = x2cobj.get_xmol(mol)

    get_h1_xmol = gen_hfw(xmol, x2cobj.approx)
    def hcore_deriv(atm_id):
        h1 = get_h1_xmol(atm_id)
        if contr_coeff is not None:
            contr_coeff_2c = x2c._block_diag(contr_coeff)
            h1 = lib.einsum('pi,xpq,qj->xij', contr_coeff_2c, h1, contr_coeff_2c)
        return numpy.asarray(h1)
    return hcore_deriv


def hcore_deriv_generator_spin_orbital(self, mol=None, deriv=1):
    if deriv == 1:
        return hcore_grad_generator_spin_orbital(self, mol)
    elif deriv == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError

x2c.SpinOrbitalX2CHelper.hcore_deriv_generator = hcore_deriv_generator_spin_orbital # type: ignore

def _block_diag_xyz(mat):
    '''
    [a b c]
    ->
    [[a 0] [b 0] [c 0]]
    [[0 a] [0 b] [0 c]]
    '''
    return numpy.asarray([_block_diag(mat_i) for mat_i in mat])

def _sigma_dot_xyz(mat):
    '''sigma dot A x B + A dot B for A in [Ax, Ay, Az]'''
    return numpy.asarray([_sigma_dot(mat_i) for mat_i in mat])

def gen_hfw(mol, approx='1E'):
    approx = approx.upper()
    c = lib.param.LIGHT_SPEED

    h0, s0 = _get_h0_s0(mol)
    e0, c0 = scipy.linalg.eigh(h0, s0)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao_2c()
    if 'ATOM' in approx:
        x0 = numpy.zeros((nao,nao))
        for ia in range(mol.natm):
            ish0, ish1, p0, p1 = aoslices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            t1 = _block_diag(mol.intor('int1e_kin', shls_slice=shls_slice))
            s1 = _block_diag(mol.intor('int1e_ovlp', shls_slice=shls_slice))
            with mol.with_rinv_at_nucleus(ia):
                z = -mol.atom_charge(ia)
                v1 = _block_diag(z * mol.intor('int1e_rinv', shls_slice=shls_slice))
                w1 = _sigma_dot(z * mol.intor('int1e_sprinvsp', shls_slice=shls_slice))
            x0[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
    else:
        cl0 = c0[:nao, nao:]
        cs0 = c0[nao:, nao:]
        x0 = scipy.linalg.solve(cl0.T, cs0.T).T
    s_nesc0 = s0[:nao, :nao] + reduce(numpy.dot, (x0.conj().T, s0[nao:, nao:], x0))
    R0 = x2c._get_r(s0[:nao, :nao], s_nesc0)
    c_fw0 = numpy.vstack((R0, numpy.dot(x0, R0)))
    h0_fw_half = numpy.dot(h0, c_fw0)
    ovlp = _block_diag(mol.intor('int1e_ovlp'))
    kin = _block_diag(mol.intor('int1e_kin'))
    get_h1_etc = _gen_first_order_quantities(mol, e0, c0, x0, ovlp, kin, approx)

    def hcore_deriv(ia):
        h1_ao, s1_ao, e1, c1, x1, s_nesc1, R1, c_fw1 = get_h1_etc(ia)
        hfw1 = lib.einsum('xpi,pj->xij', c_fw1, h0_fw_half)
        hfw1 += hfw1 + hfw1.transpose(0,2,1)
        hfw1 += lib.einsum('pi,xpq,qj->xij', c_fw0, h1_ao, c_fw0)

        return hfw1
    return hcore_deriv

def _get_h0_s0(mol):
    c = lib.param.LIGHT_SPEED
    t = _block_diag(mol.intor_symmetric('int1e_kin'))
    v = _block_diag(mol.intor_symmetric('int1e_nuc'))
    s = _block_diag(mol.intor_symmetric('int1e_ovlp'))
    w = _block_diag(mol.intor('int1e_pnucp')) # temporary convert to spin free operato
    nao = s.shape[0]
    n2 = nao * 2
    dtype = numpy.result_type(t, v, w, s)
    h = numpy.zeros((n2,n2), dtype=dtype)
    m = numpy.zeros((n2,n2), dtype=dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    return h, m

def _gen_h1_s1(mol):
    c = lib.param.LIGHT_SPEED
    s1 = _block_diag_xyz(mol.intor('int1e_ipovlp', comp=3))
    t1 = _block_diag_xyz(mol.intor('int1e_ipkin', comp=3))
    v1 = _block_diag_xyz(mol.intor('int1e_ipnuc', comp=3))
    #w1 = _sigma_dot_xyz(mol.intor('int1e_ipspnucsp').reshape(3,4,nao,nao))
    w1 = _block_diag_xyz(mol.intor('int1e_ippnucp', comp=3))
    aoslices = mol.aoslice_by_atom()
    nao = s1.shape[1]
    n2 = nao * 2
    nao_nr = nao // 2

    def get_h1_s1(ia): # h1, s1 for dirac operator
        h1 = numpy.zeros((3,n2,n2), dtype=complex)
        m1 = numpy.zeros((3,n2,n2), dtype=complex)
        ish0, ish1, i0, i1 = aoslices[ia]
        with mol.with_rinv_origin(mol.atom_coord(ia)):
            z = mol.atom_charge(ia)
            rinv1 = _block_diag_xyz(-z*mol.intor('int1e_iprinv', comp=3))
            sprinvsp1 = _sigma_dot_xyz(-z*mol.intor('int1e_ipsprinvsp').reshape(3,4,nao//2,nao//2))
            #sprinvsp1 = _block_diag_xyz(-z*(1+0.j)*mol.intor('int1e_ipprinvp')) # use p to fake sp for debug purpose
            rinv1 [:, i0:i1, :] -= v1[:, i0:i1]
            rinv1 [:, i0+nao_nr:i1+nao_nr, :] -= v1[:, i0+nao_nr:i1+nao_nr, :]
            sprinvsp1[:, i0:i1, :] -= w1[:, i0:i1]
            sprinvsp1[:, i0+nao_nr:i1+nao_nr, :] -= w1[:, i0+nao_nr:i1+nao_nr, :]

        for i in range(3):
            s1cc = numpy.zeros((nao, nao), dtype=complex)
            t1cc = numpy.zeros((nao, nao), dtype=complex)
            s1cc[i0:i1,:] -= s1[i,i0:i1]
            s1cc[i0+nao_nr:i1+nao_nr,:] -= s1[i,i0+nao_nr:i1+nao_nr]
            s1cc[:,i0:i1]-= s1[i,i0:i1].T
            s1cc[:,i0+nao_nr:i1+nao_nr]-= s1[i,i0+nao_nr:i1+nao_nr].T
            t1cc[i0:i1,:] =-t1[i,i0:i1]
            t1cc[i0+nao_nr:i1+nao_nr,:] -= t1[i,i0+nao_nr:i1+nao_nr]
            t1cc[:,i0:i1]-= t1[i,i0:i1].T
            t1cc[:,i0+nao_nr:i1+nao_nr]-= t1[i,i0+nao_nr:i1+nao_nr].T
            v1cc = rinv1[i] + rinv1[i].T
            w1cc = sprinvsp1[i] + sprinvsp1[i].T

            h1[i,:nao,:nao] = v1cc
            h1[i,:nao,nao:] = t1cc
            h1[i,nao:,:nao] = t1cc
            h1[i,nao:,nao:] = w1cc * (.25/c**2) - t1cc
            m1[i,:nao,:nao] = s1cc
            m1[i,nao:,nao:] = t1cc * (.5/c**2)
        return h1, m1
    return get_h1_s1

def _gen_first_order_quantities(mol, e0, c0, x0, s0, t0, approx='1E'):
    c = lib.param.LIGHT_SPEED
    nao = e0.size // 2
    n2 = nao * 2

    epq = e0[:,None] - e0
    degen_mask = abs(epq) < 1e-7
    epq[degen_mask] = 1e200

    cl0 = c0[:nao,nao:]
    # cs0 = c0[nao:,nao:]
    # s0 = mol.intor('int1e_ovlp')
    # t0 = mol.intor('int1e_kin')
    t0x0 = numpy.dot(t0, x0) * (.5/c**2)
    s_nesc0 = s0[:nao,:nao] + numpy.dot(x0.T.conj(), t0x0)
    w_s, v_s = scipy.linalg.eigh(s0)
    w_sqrt = numpy.sqrt(w_s)
    s_nesc0_vbas = reduce(numpy.dot, (v_s.T.conj(), s_nesc0, v_s))
    R0_mid = numpy.einsum('i,ij,j->ij', 1./w_sqrt, s_nesc0_vbas, 1./w_sqrt)
    wr0, vr0 = scipy.linalg.eigh(R0_mid)
    wr0_sqrt = numpy.sqrt(wr0)
    # R0 in v_s basis
    R0 = numpy.dot(vr0/wr0_sqrt, vr0.T.conj())
    R0 *= w_sqrt
    R0 /= w_sqrt[:,None]
    # Transform R0 back
    R0 = reduce(numpy.dot, (v_s, R0, v_s.T.conj()))
    get_h1_s1 = _gen_h1_s1(mol)
    def get_first_order(ia):
        h1ao, s1ao = get_h1_s1(ia)
        h1mo = lib.einsum('pi,xpq,qj->xij', c0.conj(), h1ao, c0)
        s1mo = lib.einsum('pi,xpq,qj->xij', c0.conj(), s1ao, c0)
        if 'ATOM' in approx:
            e1 = c1_ao = x1 = None
            s_nesc1 = lib.einsum('pi,xpq,qj->xij', x0.conj(), s1ao[:,nao:,nao:], x0)
            s_nesc1+= s1ao[:,:nao,:nao]
        else:
            f1 = h1mo[:,:,nao:] - s1mo[:,:,nao:] * e0[nao:]
            c1 = f1 / -epq[:,nao:]
            e1 = f1[:,nao:]
            e1[:,~degen_mask[nao:,nao:]] = 0

            c1_ao = lib.einsum('pq,xqi->xpi', c0, c1)
            cl1 = c1_ao[:,:nao]
            cs1 = c1_ao[:,nao:]
            tmp = cs1 - lib.einsum('pq,xqi->xpi', x0.conj(), cl1)
            x1 = scipy.linalg.solve(cl0.T, tmp.reshape(-1,nao).T)
            x1 = x1.T.reshape(3,nao,nao)

            s_nesc1 = lib.einsum('xpi,pj->xij', x1.conj(), t0x0)
            s_nesc1 = s_nesc1 + s_nesc1.transpose(0,2,1)
            s_nesc1+= lib.einsum('pi,xpq,qj->xij', x0.conj(), s1ao[:,nao:,nao:], x0)
            s_nesc1+= s1ao[:,:nao,:nao]

        R1 = numpy.empty((3,nao,nao), dtype=complex)
        c_fw1 = numpy.empty((3,n2,nao), dtype=complex)
        for i in range(3):
            R1[i] = _get_r1((w_sqrt,v_s), s_nesc0_vbas,
                            s1ao[i,:nao,:nao], s_nesc1[i], (wr0_sqrt,vr0))
            c_fw1[i,:nao] = R1[i]
            c_fw1[i,nao:] = numpy.dot(x0, R1[i])
            if 'ATOM' not in approx:
                c_fw1[i,nao:] += numpy.dot(x1[i], R0)

        return h1ao, s1ao, e1, c1_ao, x1, s_nesc1, R1, c_fw1
    return get_first_order
