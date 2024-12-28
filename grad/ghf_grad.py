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
Non-relativistic generalized Hartree-Fock analytical nuclear gradients
'''


import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from socutils.somf import x2c_grad

from numpy.linalg import norm

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of GHF/GKS gradients

    Args:
        mf_grad : grad.ghf.Gradients or grad.gks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-GHF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3), dtype=complex)
    nao = mol.nao_nr()
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)

        if h1ao.shape[-1] == dm0.shape[-1]//2:
            h1ao = numpy.asarray([scipy.linalg.block_diag(h1ao[i],h1ao[i]) for i in range(3)])

        # Be careful with the contraction order when h1, vhf1, and dm0 are complex
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm0)
        de[k] += numpy.einsum('xij,ji->x', vhf[:,p0:p1], dm0[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ji->x', vhf[:,p0+nao:p1+nao], dm0[:,p0+nao:p1+nao]) * 2
        # s1 is real for ghf
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], dme0[:nao,p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ji->x', s1[:,p0:p1], dme0[nao:,p0+nao:p1+nao]) * 2

        # for density-fitting and dft
        de[k] += mf_grad.extra_force(ia, locals())
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)

    mf_grad.de_elec = de.real
    return de.real

def get_jk(mf_grad, mol = None, dm = None):
    '''
    First order derivative of HF Coulomb and exchange matrix (wrt electron coordinates)

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    nao = mol.nao_nr()

    # Density matrix preprocessing
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao*2,nao*2)
    nset = dms.shape[0]

    vj = numpy.zeros((nset, 3, nao*2, nao*2), dm.dtype)
    vk = numpy.zeros((nset, 3, nao*2, nao*2), dm.dtype)

    dms_spin = []

    if not (dm.dtype == numpy.complex128):
        for i in range(nset):
            dmi = dms[i]
            dmaa = dmi[:nao, :nao]
            dmbb = dmi[nao:, nao:]
            dmab = dmi[:nao, nao:]
            dmba = dmi[nao:, :nao]
            dms_spin.append((dmaa, dmbb, dmab, dmba))

        dms_spin = numpy.asarray(dms_spin).reshape(-1, nao, nao)
        j1, k1 = rhf_grad.get_jk(mol, dms_spin)
        j1 = j1.reshape(nset, 4, 3, nao, nao)
        k1 = k1.reshape(nset, 4, 3, nao, nao)

        vj[:, :, :nao,:nao] = j1[:,0] + j1[:,1]
        vj[:, :, nao:,nao:] = j1[:,0] + j1[:,1]
        vk[:, :, :nao,:nao] = k1[:,0]
        vk[:, :, nao:,nao:] = k1[:,1]
        vk[:, :, :nao,nao:] = k1[:,2]
        vk[:, :, nao:,:nao] = k1[:,3]
    else:
        for i in range(nset):
            dmi = dms[i]
            dmaa = dmi[:nao, :nao].real
            dmbb = dmi[nao:, nao:].real
            dmab = dmi[:nao, nao:].real
            dmba = dmi[nao:, :nao].real
            dms_spin.append((dmaa, dmbb, dmab, dmba))
        for i in range(nset):
            dmi = dms[i]
            dmaa = dmi[:nao, :nao].imag
            dmbb = dmi[nao:, nao:].imag
            dmab = dmi[:nao, nao:].imag
            dmba = dmi[nao:, :nao].imag
            dms_spin.append((dmaa, dmbb, dmab, dmba))

        dms_spin = numpy.asarray(dms_spin).reshape(-1, nao, nao)
        j1, k1 = rhf_grad.get_jk(mol, dms_spin)
        j1 = j1.reshape(2*nset, 4, 3, nao, nao)
        k1 = k1.reshape(2*nset, 4, 3, nao, nao)
        j1r = j1[:nset]
        j1i = j1[nset:]
        k1r = k1[:nset]
        k1i = k1[nset:]

        # imaginary part of vj should be zero for ghf
        vj[:, :, :nao,:nao] = j1r[:,0] + 1.j*j1i[:,0] + j1r[:,1] + 1.j*j1i[:,1]
        vj[:, :, nao:,nao:] = j1r[:,0] + 1.j*j1i[:,0] + j1r[:,1] + 1.j*j1i[:,1]
        vk[:, :, :nao,:nao] = k1r[:,0] + 1.j*k1i[:,0]
        vk[:, :, nao:,nao:] = k1r[:,1] + 1.j*k1i[:,1]
        vk[:, :, :nao,nao:] = k1r[:,2] + 1.j*k1i[:,2]
        vk[:, :, nao:,:nao] = k1r[:,3] + 1.j*k1i[:,3]

    vj = vj.reshape(out_shape)
    vk = vk.reshape(out_shape)

    return vj, vk

def get_veff(mf_grad, mol = None, dm = None):
    '''
    First order derivative of HF potential matrix (wrt electron coordinates)

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    vj, vk = get_jk(mf_grad, mol, dm)

    return vj - vk


def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix.
    '''
    return rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)


class Gradients(rhf_grad.GradientsMixin):
    '''Non-relativistic generalized Hartree-Fock gradients
    '''
    def __init__(self, method):
        rhf_grad.GradientsMixin.__init__(self, method)
        self.de_elec = None

    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm)
    
    def get_jk(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_jk(self, mol, dm)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

Grad = Gradients

from pyscf import scf
scf.ghf.GHF.Gradients = lib.class_as_method(Gradients)
