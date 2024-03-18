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
        if dm0.dtype is not numpy.complex128:
            new_dm0 = numpy.zeros(dm0.shape, dtype=numpy.complex128)
            new_dm0 += dm0
            dm0 = new_dm0
        if dm0.dtype is numpy.complex128:
            h1ao = h1ao * (1+0.j)

        if h1ao.shape[-1] != dm0.shape[-1]:
            h1ao = numpy.asarray([scipy.linalg.block_diag(h1ao[i],h1ao[i]) for i in range(3)])

        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
# s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0+nao:p1+nao], dm0[p0+nao:p1+nao]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1,:nao]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0+nao:p1+nao,nao:]) * 2

        #de[k] += mf_grad.extra_force(ia, locals())
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
    return de.real

def get_veff(mf_grad, mol, dm):
    '''
    First order derivative of HF potential matrix (wrt electron coordinates)

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    nao = mol.nao_nr()
    dmaa = dm[:nao, :nao]
    dmbb = dm[nao:, nao:]
    dmab = dm[:nao, nao:]
    dmba = dm[nao:, :nao]
    dms = numpy.stack((dmaa, dmbb, dmab, dmba))

    j1, k1 = mf_grad.get_jk(mol, dms)
    j1[0].shape
    vj = numpy.zeros((3, nao*2, nao*2), dm.dtype)
    vk = numpy.zeros((3, nao*2, nao*2), dm.dtype)

    vj[:, :nao,:nao] = j1[0] + j1[1]
    vj[:, nao:,nao:]=j1[0] + j1[1]
    vk[:, :nao,:nao]=k1[0]
    vk[:, nao:,nao:]=k1[1]
    vk[:, :nao,nao:]=k1[2]
    vk[:, nao:,:nao]=k1[3]

    return vj - vk


def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix.
    '''
    return rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)


class Gradients(rhf_grad.GradientsMixin):
    '''Non-relativistic generalized Hartree-Fock gradients
    '''
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

Grad = Gradients

from pyscf import scf
scf.ghf.GHF.Gradients = lib.class_as_method(Gradients)
