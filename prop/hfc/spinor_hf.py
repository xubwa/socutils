#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
#

'''
Magnetic hyperfine constants for relativistic 2-component JHF methods
(In testing)
'''

import warnings, scipy
import numpy as np
from functools import reduce
from pyscf import lib
from socutils.prop.hfc.dhf import int_hfc_4c, au2MHz

warnings.warn('Module HFC is under testing')

def int_hfc_2c(method, atm_id, utm=True):
    mol = method.mol
    xmol, contr_coeff_nr = method.with_x2c.get_xmol(mol)
    npri, ncon = contr_coeff_nr.shape
    contr_coeff = np.zeros((npri*2,ncon*2))
    contr_coeff[0::2,0::2] = contr_coeff_nr
    contr_coeff[1::2,1::2] = contr_coeff_nr
    
    t = xmol.intor('int1e_kin_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_spnucsp_spinor')
    from socutils.somf import x2c_grad
    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)
    m4c_inv = np.dot(a, a.T.conj())

    h4c = method.with_x2c.h4c
    int_4c = int_hfc_4c(xmol, atm_id, utm, h4c, m4c_inv)
    int2c = []
    for xx in range(3):
        hfw1 = method.with_x2c.get_hfw1(int_4c[xx])
        int2c.append(reduce(np.dot, (contr_coeff.T.conj(), hfw1, contr_coeff)))

    return np.array(int2c)


def kernel(method, hfc_nuc=None, dm=None, utm=True, x_response=True):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** HFC for 4-component SCF methods (In testing) ********')
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)
    if dm is None:
        dm = method.make_rdm1()

    xmol, contr_coeff_nr = method.with_x2c.get_xmol(method.mol)
    npri, ncon = contr_coeff_nr.shape
    contr_coeff = np.zeros((npri*2,ncon*2))
    contr_coeff[0::2,0::2] = contr_coeff_nr
    contr_coeff[1::2,1::2] = contr_coeff_nr
    
    t = xmol.intor('int1e_kin_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_spnucsp_spinor')
    from socutils.somf import x2c_grad
    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)
    h1e_4c = h4c.copy()
    m4c_inv = np.dot(a, a.T.conj())

    h4c = method.with_x2c.h4c
    log.info('\nMagnetic Hyperfine Constants Results')
    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        int_4c = int_hfc_4c(xmol, atm_id, utm, h4c, m4c_inv)
        hfc_atm = np.zeros(3, dtype=dm.dtype)
        for xx in range(3):
            #int_2c = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, int_4c[xx])
            int_2c = method.with_x2c.get_hfw1(int_4c[xx], x_response=x_response)
            int_2c = reduce(np.dot, (contr_coeff.T.conj(), int_2c, contr_coeff))
            hfc_atm[xx] = np.einsum('ij,ji->', int_2c, dm).real
        hfc.append(hfc_atm)
        print('\nAtom %d' % atm_id)
        print('HFC (a.u.):')
        print(hfc_atm.real)
        print('HFC (MHz):')
        print(hfc_atm.real * au2MHz)
    return np.asarray(hfc).real

HFC = kernel
from socutils.scf.spinor_hf import JHF
JHF.HFC = lib.class_as_method(HFC)

if __name__ == '__main__':
    from pyscf import gto
    from socutils.scf import spinor_hf
    from socutils.somf import eamf

    mol = gto.M(
        atom = """
O        -0.00000000     0.00000000     0.68645937
H         0.00000000     0.00000000    -1.20326715
""",
        basis = 'ccpvdz', spin=1, unit="bohr", verbose = 4)
    mf = spinor_hf.SymmJHF(mol, symmetry="linear", occup={"-1/2":[4], "1/2":[4], "3/2":[0], "-3/2":[1]})
    mf.with_x2c = eamf.SpinorEAMFX2CHelper(mol, with_gaunt=False, with_breit=False, eamf="x2camf")
    mf.kernel()
    mf.HFC()
    
    
