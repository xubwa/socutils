#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Effective field operator for relativistic 4-component DHF and DKS methods.
(In testing)
'''

import warnings
import numpy as np
from pyscf import lib


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
    
    int1e = mol.intor('int1e_spspsp_spinor')
    
    int_4c = np.zeros((n2c*2, n2c*2), dtype=dm.dtype)
    int_4c[:n2c,n2c:] = -1.j * int1e
    int_4c[n2c:,:n2c] = 1.j * int1e
    
    e_eff = np.einsum('ij,ji->', int_4c, dm)
    if e_eff.imag > 1e-10:
        log.warn('Significant imaginary part found in contact density')
    log.info(f'Effective electric field: {e_eff}')



EffectiveField = kernel

from pyscf import scf
scf.dhf.UHF.EffectiveField = lib.class_as_method(EffectiveField)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = '''H 0.0 0.0 0.0\nF 0.0 0.0 1.1'''
    mol.basis = 'uncccpvdz'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = scf.DHF(mol).run()
    mf.chkfile='hf.chk'
    mf.init_guess='chkfile'
    mf.EffectiveField()
