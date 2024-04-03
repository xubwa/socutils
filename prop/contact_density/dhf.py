#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
#

'''
Contact density for relativistic 4-component DHF and DKS methods.
(In testing)
'''

import warnings
import numpy as np
from pyscf import lib

warnings.warn('Module contact density is under testing')


def kernel(method, cd_nuc=None, dm=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** Contact density for 4-component SCF methods (In testing) ********')

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
    for atm_id in range(len(coords)):
        int_4c = np.zeros((n2c*2, n2c*2), dtype=dm.dtype)
        int_4c[:n2c,:n2c] = np.einsum('p,q->pq', aoLa[atm_id].conj(), aoLa[atm_id])
        int_4c[:n2c,:n2c]+= np.einsum('p,q->pq', aoLb[atm_id].conj(), aoLb[atm_id])
        int_4c[n2c:,n2c:] = np.einsum('p,q->pq', aoSa[atm_id].conj(), aoSa[atm_id]) / 4.0 / c**2
        int_4c[n2c:,n2c:]+= np.einsum('p,q->pq', aoSb[atm_id].conj(), aoSb[atm_id]) / 4.0 / c**2
        cont_den = np.einsum('ij,ji->', int_4c, dm)
        if cont_den.imag > 1e-10:
            log.warn('Significant imaginary part found in contact density')
        log.info('\nAtom %d' % atm_id)
        log.info('Contact Density: %f' % cont_den.real)



ContDen = kernel

from pyscf import scf
scf.dhf.UHF.ContDen = lib.class_as_method(ContDen)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = [
        ["Ne" , (0.1 ,  0.     , 0.000)]]
    mol.basis = 'uncccpvdz'
    mol.build()

    mf = scf.DHF(mol).run()
    mf.ContDen()