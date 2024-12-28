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
from pyscf import lib, df
from pyscf.lib import logger
from pyscf.df.grad import rhf as rhf_grad
from socutils.grad import ghf_grad
from socutils.somf import x2c_grad # for hcore_generator


def get_jk(mf_grad, mol = None, dm = None):
    '''
    First order derivative of HF Coulomb and exchange matrix (wrt electron coordinates)

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    nao = mol.nao_nr()
    natom = mol.natm

    # Density matrix preprocessing
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao*2,nao*2)
    nset = dms.shape[0]

    vj = numpy.zeros((nset, 3, nao*2, nao*2), dm.dtype)
    vk = numpy.zeros((nset, 3, nao*2, nao*2), dm.dtype)
    vj_aux = numpy.zeros((nset, nset, natom, 3))
    vk_aux = numpy.zeros((nset, nset, natom, 3))

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
        j1, k1 = rhf_grad.get_jk(mf_grad, mol, dms_spin, hermi=0)
        j1aux = j1.aux.reshape(nset, nset, 4, 4, natom, 3)
        k1aux = k1.aux.reshape(nset, nset, 4, 4, natom, 3)
        j1 = j1.reshape(nset, 4, 3, nao, nao)
        k1 = k1.reshape(nset, 4, 3, nao, nao)
        
        vj[:, :, :nao,:nao] = j1[:,0] + j1[:,1]
        vj[:, :, nao:,nao:] = j1[:,0] + j1[:,1]
        vk[:, :, :nao,:nao] = k1[:,0]
        vk[:, :, nao:,nao:] = k1[:,1]
        vk[:, :, :nao,nao:] = k1[:,2]
        vk[:, :, nao:,:nao] = k1[:,3]
        vj_aux[:,:] = j1aux[:,:,0,0] + j1aux[:,:,1,1] + j1aux[:,:,0,1] + j1aux[:,:,1,0]
        vk_aux[:,:] = k1aux[:,:,0,0] + k1aux[:,:,1,1] + k1aux[:,:,2,3] + k1aux[:,:,3,2]
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
        j1, k1 = rhf_grad.get_jk(mf_grad, mol, dms_spin, hermi=0)
        j1aux = j1.aux.reshape(nset*2, nset*2, 4, 4, natom, 3)
        k1aux = k1.aux.reshape(nset*2, nset*2, 4, 4, natom, 3)
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
        # only real part of aux response is needed
        vj_aux[:,:] = j1aux[:nset,:nset,0,0] + j1aux[:nset,:nset,1,1] + j1aux[:nset,:nset,0,1] + j1aux[:nset,:nset,1,0] \
                    - j1aux[nset:,nset:,0,0] - j1aux[nset:,nset:,1,1] - j1aux[nset:,nset:,0,1] - j1aux[nset:,nset:,1,0]
        vk_aux[:,:] = k1aux[:nset,:nset,0,0] + k1aux[:nset,:nset,1,1] + k1aux[:nset,:nset,2,3] + k1aux[:nset,:nset,3,2] \
                    - k1aux[nset:,nset:,0,0] - k1aux[nset:,nset:,1,1] - k1aux[nset:,nset:,2,3] - k1aux[nset:,nset:,3,2]
        
    vj = lib.tag_array(vj.reshape(out_shape), aux=numpy.array(vj_aux))
    vk = lib.tag_array(vk.reshape(out_shape), aux=numpy.array(vk_aux))

    return vj, vk


class Gradients(ghf_grad.Gradients):
    '''Non-relativistic generalized Hartree-Fock gradients
    '''
    def __init__(self, mf):
        ghf_grad.Gradients.__init__(self, mf)
    
    # Whether to include the response of DF auxiliary basis when computing
    # nuclear gradients of J/K matrices
    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df.df_jk._DFHF)
    
    def get_veff(self, mol=None, dm=None):
        vj, vk = self.get_jk(mol, dm)
        vhf = vj - vk
        if self.auxbasis_response:
            e1_aux = vj.aux.sum ((0,1))
            e1_aux -= numpy.trace (vk.aux, axis1=0, axis2=1)
            logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vhf = lib.tag_array(vhf, aux=e1_aux)
        return vhf

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['vhf'].aux[atom_id]
        else:
            return 0
        
    def get_jk(self, mol=None, dm=None):
        return get_jk(self, mol, dm)

Grad = Gradients

from pyscf import scf
from pyscf.df.df_jk import _DFHF
def nuc_grad_method(self):
    from pyscf.df.grad import rhf, rohf, uhf, rks, roks, uks
    if isinstance(self, scf.uhf.UHF):
        if isinstance(self, scf.hf.KohnShamDFT):
            return uks.Gradients(self)
        else:
            return uhf.Gradients(self)
    elif isinstance(self, scf.rohf.ROHF):
        if isinstance(self, scf.hf.KohnShamDFT):
            return roks.Gradients(self)
        else:
            return rohf.Gradients(self)
    elif isinstance(self, scf.rhf.RHF):
        if isinstance(self, scf.hf.KohnShamDFT):
            return rks.Gradients(self)
        else:
            return rhf.Gradients(self)
    elif isinstance(self, scf.ghf.GHF):
        if isinstance(self, scf.hf.KohnShamDFT):
            raise NotImplementedError
        else:
            return Gradients(self)
    else:
        raise NotImplementedError

_DFHF.Gradients = nuc_grad_method