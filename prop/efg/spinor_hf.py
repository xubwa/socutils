#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
# Modified from https://github.com/pyscf/properties 
#               pyscf/prop/efg/dhf.py
#

'''
Electric field gradients for relativistic 2-component JHF methods.
(In testing)
'''

import warnings
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.x2c.x2c import _decontract_spinor
from socutils.prop.efg.rhf import _get_quad_nuc
from socutils.tools.timer import Timer

warnings.warn('Module EFG is under testing')

def kernel(method, efg_nuc=None, dm=None, x_response=True):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** EFG for 2-component SCF methods (In testing) ********')
    timeit = Timer()
    xmol, contr_coeff_nr = method.with_x2c.get_xmol(method.mol)
    xmol, contr_coeff = _decontract_spinor(method.mol, method.with_x2c.xuncontract)
    mol = xmol
    npri, ncon = contr_coeff_nr.shape
    if efg_nuc is None:
        efg_nuc = range(mol.natm)

    c = lib.param.LIGHT_SPEED
    if dm is None:
        dm = method.make_rdm1()

    timeit.accumulate()
    from socutils.somf import x2c_grad
    t = mol.intor('int1e_kin_spinor')
    timeit.accumulate()
    s = mol.intor('int1e_ovlp_spinor')
    timeit.accumulate()
    v = mol.intor('int1e_nuc_spinor')
    timeit.accumulate()
    w = mol.intor('int1e_spnucsp_spinor')
    timeit.accumulate()
    a, e, xmat, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)
    timeit.accumulate()
    h4c = method.with_x2c.h4c
    m4c = method.with_x2c.m4c
    log.info('\nElectric Field Gradient Tensor Results')
    efg = []
    timeit.accumulate('zeroth order h4c and m4c')
    for i, atm_id in enumerate(efg_nuc):
        int_4c = _get_4cspinor_quadrupole_integrals(mol, atm_id)
        timeit.accumulate('quadrupole integral')
        efg_e = np.zeros((3,3), dtype=dm.dtype)
        for x in range(3):
            for y in range(x+1):
                #if Xresp:
                #    #int_2c = x2c_grad.get_hfw1(a, xmat, st, m4c, h4c, e, r, l, int_4c[x,y])
                #    int_2c = method.with_x2c.get_hfw1(int_4c[x,y], xresp=True)
                #else:
                #    #from socutils.somf.eamf import to_2c
                #    #int_2c = to_2c(xmat, r, int_4c[x,y])
                #    int_2c = method.with_x2c.get_hfw1(int_4c[x,
                int_2c = method.with_x2c.get_hfw1(int_4c[x,y], x_response=x_response)
                int_2c = reduce(np.dot, (contr_coeff.T.conj(), int_2c, contr_coeff))
                efg_e[x,y] = np.einsum('ij,ji->', int_2c, dm)
                efg_e[y,x] = efg_e[x,y]
        timeit.accumulate('transform efg integral')
        efg_nuc = _get_quad_nuc(mol, atm_id)
        v = efg_nuc - efg_e
        efg.append(v)
        log.info('\nAtom %d' % atm_id)
        log.info('Electronic EFG:')
        print(efg_e.real)
        log.info('Nuclear EFG:')
        print(efg_nuc.real)

        eigs, _ = np.linalg.eigh(v)
        log.info('Total EFG on main axis (Eigenvalues of total EFG tensors):')
        print(eigs)

    return np.asarray(efg).real

def _get_4cspinor_quadrupole_integrals(mol, atm_id):
    n2c = mol.nao_2c()
    # The electronic quadrupole operator (3 \vec{r} \vec{r} - r^2) / r^5
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        ipipv = mol.intor('int1e_ipiprinv_spinor', 9).reshape(3,3,n2c,n2c)
        ipvip = mol.intor('int1e_iprinvip_spinor', 9).reshape(3,3,n2c,n2c)
        h1LL = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1LL = h1LL + h1LL.conj().transpose(0,1,3,2)
        # trace = h1LL[0,0] + h1LL[1,1] + h1LL[2,2]
        # h1LL[0,0] -= trace
        # h1LL[1,1] -= trace
        # h1LL[2,2] -= trace

        ipipv = mol.intor('int1e_ipipsprinvsp_spinor', 9).reshape(3,3,n2c,n2c)
        ipvip = mol.intor('int1e_ipsprinvspip_spinor', 9).reshape(3,3,n2c,n2c)
        h1SS = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1SS = h1SS + h1SS.conj().transpose(0,1,3,2)
        # trace = h1SS[0,0] + h1SS[1,1] + h1SS[2,2]
        # h1SS[0,0] -= trace
        # h1SS[1,1] -= trace
        # h1SS[2,2] -= trace
    
    coord = mol.atom_coord(atm_id).reshape(1,3)
    aoLa, aoLb = mol.eval_gto('GTOval_spinor', coord)
    aoSa, aoSb = mol.eval_gto('GTOval_sp_spinor', coord)
    fcLL = np.einsum('ip,iq->pq', aoLa.conj(), aoLa)
    fcLL+= np.einsum('ip,iq->pq', aoLb.conj(), aoLb)
    fcSS = np.einsum('ip,iq->pq', aoSa.conj(), aoSa)
    fcSS+= np.einsum('ip,iq->pq', aoSb.conj(), aoSb)

    c = lib.param.LIGHT_SPEED
    int_4c = np.zeros((3, 3, n2c*2, n2c*2), dtype=fcLL.dtype)
    for x in range(3):
        for y in range(3):
            int_4c[x,y,:n2c,:n2c] = h1LL[x,y]
            int_4c[x,y,n2c:,n2c:] = h1SS[x,y] / 4.0 / c**2
            if x == y:
                int_4c[x,y,:n2c,:n2c] += 4*np.pi/3 * fcLL
                int_4c[x,y,n2c:,n2c:] += 4*np.pi/3 * fcSS / 4.0 / c**2
    return int_4c

EFG = kernel

from socutils.scf import spinor_hf
spinor_hf.JHF.EFG = lib.class_as_method(EFG)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = '''
O 0.          0. -0.12390941
H 0. -1.42993701  0.98326612
H 0.  1.42993701  0.98326612
'''
    mol.basis = 'uncccpvdz'
    mol.unit = 'Bohr'
    mol.build()
    from socutils.scf import x2camf_hf
    mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.EFG(efg_nuc=[0])
