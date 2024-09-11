#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
#

'''
G-factor for relativistic 4-component methods,
the corresponding operator:
    \frac{1}{2}[L + g_e S]
(In testing)
'''

import warnings, scipy
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.data.nist import G_ELECTRON, LIGHT_SPEED
from pyscf import scf

warnings.warn('Module G-factor is under testing')

from socutils.prop.gfac.spinor_hf import int_gfac_4c

def kernel(method, utm=False, dm=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** G-factor for 4-component SCF methods (In testing) ********')
    
    if dm is None:
        dm = method.make_rdm1()
    mol = method.mol

    log.info('\nG-factor [1/2(L + g_e S) operator] Results')
    n2c = mol.nao_2c()
    t = mol.intor('int1e_kin_spinor')
    s = mol.intor('int1e_ovlp_spinor')
    v = mol.intor('int1e_nuc_spinor')
    w = mol.intor('int1e_spnucsp_spinor')
    h4c = np.zeros((n2c*2, n2c*2), dtype=v.dtype)
    m4c = np.zeros((n2c*2, n2c*2), dtype=v.dtype)
    c = LIGHT_SPEED
    h4c[:n2c, :n2c] = v
    h4c[:n2c, n2c:] = t
    h4c[n2c:, :n2c] = t
    h4c[n2c:, n2c:] = w * (.25 / c**2) - t
    m4c[:n2c, :n2c] = s
    m4c[n2c:, n2c:] = t * (.5 / c**2)
    int_4c = int_gfac_4c(mol, utm=utm, h4c=h4c, m4c_inv=scipy.linalg.inv(m4c))

    xyz = ['x', 'y', 'z']
    gfac = np.zeros(3, dtype=complex)
    for xx in range(3):
        gfac[xx] = np.einsum('ij,ji->', int_4c[xx], dm)
        print('G-factor for %s: %.10f' % (xyz[xx], gfac[xx]))
    return gfac

Gfac = kernel

def get_hcore(dhfobj, mol, B_field = [0.0, 0.0, 0.0]):
    if mol.has_ecp():
        raise NotImplementedError
    from pyscf.scf import dhf
    n2c = mol.nao_2c()
    t = mol.intor('int1e_kin_spinor')
    s = mol.intor('int1e_ovlp_spinor')
    v = mol.intor('int1e_nuc_spinor')
    w = mol.intor('int1e_spnucsp_spinor')
    h4c = np.zeros((n2c*2, n2c*2), dtype=v.dtype)
    m4c = np.zeros((n2c*2, n2c*2), dtype=v.dtype)
    c = LIGHT_SPEED
    h4c[:n2c, :n2c] = v
    h4c[:n2c, n2c:] = t
    h4c[n2c:, :n2c] = t
    h4c[n2c:, n2c:] = w * (.25 / c**2) - t
    m4c[:n2c, :n2c] = s
    m4c[n2c:, n2c:] = t * (.5 / c**2)
    int_4c = int_gfac_4c(mol, utm=True, h4c=h4c, m4c_inv=scipy.linalg.inv(m4c))
    from pyscf.data.nist import LIGHT_SPEED as c
    for i in range(3):
        h4c += B_field[i] * int_4c[i] / c # the factor of half was taken in the integrals

    return h4c

from pyscf.scf import dhf
dhf.DHF.Gfac = lib.class_as_method(Gfac)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = '''
C       -0.00000000     0.00000000     1.19902577
O        0.00000000     0.00000000    -0.89955523
'''
    from socutils.tools.basis_parser import parse_genbas
    mol.basis = "unc-aug-ccpvtz"
    mol.charge = 1
    mol.spin = 1
    mol.unit = 'B'
    mol.nucmod = 'g'
    mol.build()
    chkfile = 'chk_4c.chk'
    def mfobj(B_field):
        from pyscf.scf import dhf
        mf = dhf.DHF(mol)
        mf.with_gaunt=False
        mf.chkfile = chkfile
        mf.init_guess = 'chkfile'
        mf.get_hcore = lambda mol = None: get_hcore(dhfobj=mf, mol=mol, B_field=B_field)
        mf.conv_tol = 1e-12
        return mf
    mf = mfobj([0.000, 0.000, 0.0001])
    from socutils.prop.effective_field.dhf import EffectiveField
    mf.kernel()
    mf1 = scf.DHF(mol)
    mf1.kernel(dm0=mf.make_rdm1())
    mf1.EffectiveField()
    gfac = mf1.Gfac()
    print(abs(gfac)*4.-G_ELECTRON)
    from socutils.scf import linear_dhf
    mf2 = linear_dhf.SymmDHF(mol, symmetry='linear')
    mf2.kernel()
    mf2.EffectiveField()
    mf2.Gfac()
    exit()
    mf.with_gaunt = True
    mf.with_breit = True
    mf.kernel(dm0 = mf.init_guess_by_chkfile(chkfile))
    gfac=mf1.Gfac()
    print(gfac)
    print((4.0*abs(gfac[2]) - G_ELECTRON)*10**6)
    
    
