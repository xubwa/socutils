#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
#

'''
Contact density for relativistic 2-component JHF methods.
(In testing)
'''

import warnings
from functools import reduce
import numpy as np
from pyscf import lib

warnings.warn('Module contact density is under testing')


def kernel(method, cd_nuc=None, dm=None, Xresp=False):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** Contact density for 2-component SCF methods (In testing) ********')
    if Xresp:
        log.info('Include the response of X2C transformation')
    else:
        log.info('Ignore the response of X2C transformation')

    mol = method.mol
    c = lib.param.LIGHT_SPEED
    n2c = mol.nao_2c()
    if cd_nuc is None:
        cd_nuc = range(mol.natm)
    if dm is None:
        dm = method.make_rdm1()

    coords = []
    log.info('\nContact Density Results')
    for atm_id in cd_nuc:
        coords.append(mol.atom_coord(atm_id))
    
    aoLa, aoLb = mol.eval_gto('GTOval_spinor', coords)
    aoSa, aoSb = mol.eval_gto('GTOval_sp_spinor', coords)

    t = mol.intor('int1e_kin_spinor')
    s = mol.intor('int1e_ovlp_spinor')
    v = mol.intor('int1e_nuc_spinor')
    w = mol.intor('int1e_spnucsp_spinor')
    from pyscf.socutils.somf import x2c_grad
    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)

    cont_den = []
    for atm_id in range(len(coords)):
        int_4c = np.zeros((n2c*2, n2c*2), dtype=dm.dtype)
        int_4c[:n2c,:n2c] = np.einsum('p,q->pq', aoLa[atm_id].conj(), aoLa[atm_id])
        int_4c[:n2c,:n2c]+= np.einsum('p,q->pq', aoLb[atm_id].conj(), aoLb[atm_id])
        int_4c[n2c:,n2c:] = np.einsum('p,q->pq', aoSa[atm_id].conj(), aoSa[atm_id]) / 4.0 / c**2
        int_4c[n2c:,n2c:]+= np.einsum('p,q->pq', aoSb[atm_id].conj(), aoSb[atm_id]) / 4.0 / c**2
        if Xresp:
            int_2c = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, int_4c)
        else:
            from pyscf.socutils.somf.eamf import to_2c
            int_2c = to_2c(x, r, int_4c)

        cont_den.append(np.einsum('ij,ji->', int_2c, dm))
        if cont_den[-1].imag > 1e-10:
            log.warn('Significant imaginary part found in contact density')
        log.info('\nAtom %d' % atm_id)
        log.info('Contact Density: %f' % cont_den[-1].real)

    return cont_den

ContDen = kernel

from pyscf.socutils.scf import spinor_hf
spinor_hf.JHF.ContDen = lib.class_as_method(ContDen)


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
    from pyscf.socutils.scf import x2camf_hf
    mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.ContDen()