#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Author: Xubo Wang <wangxubo0201@outlook.com>

'''Non-relativistic UKS analytical nuclear gradients'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad
from socutils import jhf_grad
from socutils.spinor_hf import spinor2sph, sph2spinor
from pyscf.dft import numint, gen_grid
from pyscf import __config__


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if mf.nlc != '':
        raise NotImplementedError('Non-local correlation functional is not supported.')
    if grids.coords is None:
        grids.build(with_non0tab=True)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        #if mf.nlc:
        #    assert 'VV10' in mf.nlc.upper()
        #    enlc, vnlc = rks_grad.get_vxc_full_response(
        #        ni, mol, nlcgrids, mf.xc+'__'+mf.nlc, dm[0]+dm[1],
        #        max_memory=max_memory, verbose=ks_grad.verbose)
        #    exc += enlc
        #    vxc += vnlc
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        #if mf.nlc:
        #    assert 'VV10' in mf.nlc.upper()
        #    enlc, vnlc = rks_grad.get_vxc(ni, mol, nlcgrids, mf.xc+'__'+mf.nlc,
        #                                  dm[0]+dm[1], max_memory=max_memory,
        #                                  verbose=ks_grad.verbose)
        #    vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)
    if dm.dtype == complex:
        vxc = vxc * (1+0.j)
    nao = mol.nao_nr()
    dmaa = dm[:nao, :nao]
    dmbb = dm[nao:, nao:]
    dmab = dm[:nao, nao:]
    dmba = dm[nao:, :nao]
    if abs(hyb) < 1e-10:
        dms = numpy.stack((dmaa, dmbb))
        j1 = ks_grad.get_j(mol, dms.real)
        vj = numpy.zeros((3, nao*2, nao*2), dm.dtype)
        vj[:, :nao, :nao] = j1[0] + j1[1]
        vj[:, nao:, nao:] = j1[0] + j1[1]
        vxc += vj

    else:
        dms = numpy.stack((dmaa, dmbb, dmab, dmba))
        j1, k1 = ks_grad.get_jk(mol, dms.real)
        vj = numpy.zeros((3, nao*2, nao*2), dm.dtype)
        vk = numpy.zeros((3, nao*2, nao*2), dm.dtype)
        vj[:, :nao, :nao] = j1[0] + j1[1]
        vj[:, nao:, nao:] = j1[0] + j1[1]
        vk[:, :nao, :nao] = k1[0]
        vk[:, nao:, nao:] = k1[1]
        vk[:, :nao, nao:] = k1[2]
        vk[:, nao:, :nao] = k1[3]
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            with mol.with_range_coulomb(omega):
                vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
        vxc += vj - vk

    return lib.tag_array(vxc, exc1_grid=exc)


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):

    if ni.collinear[0] != 'c':
        raise NotImplementedError('Noncollinear spin is not supported.')

    # currently only deal with colinear spin, so convert dm into a uks type of dm
    # and use the uks code completely, then fill them into the gks vxc

    xctype = ni._xc_type(xc_code)
    #nao_2c = nao * 2
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()

    ni1c = ni._to_numint1c()

    nao = mol.nao_nr()
    dm_a = dms[...,:nao,:nao].real.copy()
    dm_b = dms[...,nao:,nao:].real.copy()
    dm1 = (dm_a, dm_b)
    make_rho, nset, _ = ni1c._gen_rho_evaluator(mol, dm1, hermi, False, grids)

    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni1c.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[0], mask, xctype)
            rho_b = make_rho(1, ao[0], mask, xctype)
            vxc = ni1c.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc[:,0]
            aow = numint._scale_ao(ao[0], wv[0])
            rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
            aow = numint._scale_ao(ao[0], wv[1])
            rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni1c.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:4], mask, xctype)
            rho_b = make_rho(1, ao[:4], mask, xctype)
            vxc = ni1c.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni1c.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:10], mask, xctype)
            rho_b = make_rho(1, ao[:10], mask, xctype)
            vxc = ni1c.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5
            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[0], mol, ao, wv[0,4], mask, ao_loc, True)
            rks_grad._tau_grad_dot_(vmat[1], mol, ao, wv[1,4], mask, ao_loc, True)

    exc = numpy.zeros((mol.natm,3))

    vmat_2c = numpy.zeros((3,nao*2,nao*2))
    vmat_2c[:,:nao,:nao] = vmat[0]
    vmat_2c[:,nao:,nao:] = vmat[1]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat_2c

def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''

    if ni.collinear[0] != 'c':
        raise NotImplementedError('Noncollinear spin is not supported.')
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()

    ni1c = ni._to_numint1c()

    nao = mol.nao_nr()
    dm_a = dms[...,:nao,:nao].real.copy()
    dm_b = dms[...,nao:,nao:].real.copy()
    dm1 = (dm_a, dm_b)
    make_rho, nset, _ = ni1c._gen_rho_evaluator(mol, dm1, hermi, False, grids)

    excsum = 0
    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni1c.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[0], mask, xctype)
            rho_b = make_rho(1, ao[0], mask, xctype)
            exc, vxc = ni1c.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[:2]
            wv = weight * vxc[:,0]

            vtmp = numpy.zeros((3,nao,nao))
            aow = numint._scale_ao(ao[0], wv[0])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a+rho_b, weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dm1[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            aow = numint._scale_ao(ao[0], wv[1])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dm1[1]) * 2

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni1c.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[:4], mask, xctype)
            rho_b = make_rho(1, ao[:4], mask, xctype)
            exc, vxc = ni1c.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[:,0] *= .5

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[0], mask, ao_loc)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dm1[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[1], mask, ao_loc)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dm1[1]) * 2

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')

    elif xctype == 'MGGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni1c.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[:10], mask, xctype)
            rho_b = make_rho(1, ao[:10], mask, xctype)
            exc, vxc = ni1c.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[0], mask, ao_loc)
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[0,4], mask, ao_loc, True)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dm1[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[1], mask, ao_loc)
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[1,4], mask, ao_loc, True)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dm1[1]) * 2

    vmat_2c = numpy.zeros((3,nao*2,nao*2))
    vmat_2c[:,:nao,:nao] = vmat[0]
    vmat_2c[:,nao:,nao:] = vmat[1]
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat_2c


class Gradients(jhf_grad.Gradients):

    grid_response = getattr(__config__, 'grad_uks_Gradients_grid_response', False)

    def __init__(self, mf):
        jhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

    def dump_flags(self, verbose=None):
        jhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        dm_scalar = spinor2sph(mol, dm)
        print(numpy.linalg.norm(dm_scalar.real))
        print(numpy.linalg.norm(dm_scalar.imag))
        veff_scalar = get_veff(self, mol, dm_scalar)
        return sph2spinor(mol, veff_scalar)

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        if self.grid_response:
            vhf = envs['vhf']
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            return vhf.exc1_grid[atom_id]
        else:
            return 0

Grad = Gradients

from pyscf import socutils
socutils.dft.SpinorDFT.Gradients = lib.class_as_method(Gradients)
