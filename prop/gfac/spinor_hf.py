#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
#

'''
G-factor for relativistic 2-component JHF methods,
the corresponding operator:
    \frac{1}{2}[L + g_e S]
(In testing)
'''

import warnings, scipy
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.data.nist import G_ELECTRON, LIGHT_SPEED

warnings.warn('Module G-factor is under testing')

def int_gfac_4c(mol, utm = False, h4c = None, m4c_inv = None):
    factor = G_ELECTRON/8.0
    # r_i nabla_j
    xnx, xny, xnz, ynx, yny, ynz, znx, zny, znz = mol.intor("int1e_irp", comp=9)
    n2c = mol.nao_2c()
    nao = mol.nao_nr()
    int_4c = []
    for xx in range(3):
        # construct using scalar basis
        int_4c_LS = np.zeros((n2c, n2c), dtype=complex)
        if xx == 0:
            int_4c_LS[:nao, :nao] =  factor*znx + 0.25j*(-ynz+zny)
            int_4c_LS[nao:, nao:] = -factor*znx + 0.25j*(-ynz+zny)
            int_4c_LS[:nao, nao:] = -factor*(yny+znz) - 1.0j*factor*ynx
        elif xx == 1:
            int_4c_LS[:nao, :nao] =  factor*zny + 0.25j*(-znx+xnz)
            int_4c_LS[nao:, nao:] = -factor*zny + 0.25j*(-znx+xnz)
            int_4c_LS[:nao, nao:] =  factor*xny + 1.0j*factor*(xnx+znz)
        else:
            int_4c_LS[:nao, :nao] = -factor*(xnx+yny) + 0.25j*(-xny+ynx)
            int_4c_LS[nao:, nao:] = factor*(xnx+yny) + 0.25j*(-xny+ynx)
            int_4c_LS[:nao, nao:] =  factor*xnz - 1.0j*factor*ynz
        int_4c_LS[nao:, :nao] = int_4c_LS[:nao, nao:].conj()
        # transform into spinor basis
        from socutils.scf.spinor_hf import sph2spinor
        int_4c_LS = sph2spinor(mol, int_4c_LS)
        int_4c.append(np.zeros((n2c*2, n2c*2), dtype=complex))
        int_4c[xx][:n2c, n2c:] = int_4c_LS
        int_4c[xx][n2c:, :n2c] = int_4c_LS.conj().T

        # This is the unitary transformation form, which is equivalent to 
        # the use of restricted magnetic balance condition
        # Please see Lan Cheng, Molecular Physics, 121, e2113567, (2023)
        # DOI: 10.1080/00268976.2022.2113567
        if utm:
            if h4c is None or m4c_inv is None:
                raise ValueError('h4c and m4c_inv should be provided with utm=True')
            utm_tau = np.zeros_like(int_4c[xx])
            utm_tau[:n2c, n2c:] = -int_4c[xx][:n2c, n2c:]/2.0/LIGHT_SPEED**2
            utm_tau[n2c:, :n2c] = int_4c[xx][n2c:, :n2c]/2.0/LIGHT_SPEED**2
            int_4c[xx] += reduce(np.dot, (h4c, m4c_inv, utm_tau)) - reduce(np.dot, (utm_tau, m4c_inv, h4c))

    return int_4c

def kernel(method, dm=None, utm=True):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** G-factor for 2-component SCF methods (In testing) ********')
    xmol, contr_coeff_nr = method.with_x2c.get_xmol(method.mol)
    npri, ncon = contr_coeff_nr.shape
    contr_coeff = np.zeros((npri*2,ncon*2))
    contr_coeff[0::2,0::2] = contr_coeff_nr
    contr_coeff[1::2,1::2] = contr_coeff_nr
    
    if dm is None:
        dm = method.make_rdm1()
    t = xmol.intor('int1e_kin_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_spnucsp_spinor')
    from socutils.somf import x2c_grad
    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)
    m4c_inv = np.dot(a, a.T.conj())

    log.info('\nG-factor [1/2(L + g_e S) operator] Results')
    int_4c = int_gfac_4c(xmol, utm, h4c, m4c_inv)
    xyz = ['x', 'y', 'z']
    gfac = np.zeros(3)
    for xx in range(3):
        int_2c = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, int_4c[xx])
        int_2c = reduce(np.dot, (contr_coeff.T.conj(), int_2c, contr_coeff))
        gfac[xx] = np.einsum('ij,ji->', int_2c, dm).real
        print('G-factor for %s: %.10f' % (xyz[xx], gfac[xx]))
    return gfac

Gfac = kernel

def get_hcore(x2cobj, mol, B_field = [0.0, 0.0, 0.0]):
    if mol.has_ecp():
        raise NotImplementedError

    xmol, contr_coeff_nr = x2cobj.get_xmol()
    npri, ncon = contr_coeff_nr.shape
    contr_coeff = np.zeros((npri*2,ncon*2))
    contr_coeff[0::2,0::2] = contr_coeff_nr
    contr_coeff[1::2,1::2] = contr_coeff_nr
    n2c = xmol.nao_2c()
    t = xmol.intor('int1e_kin_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_spnucsp_spinor')
    h4c = np.zeros((n2c*2, n2c*2), dtype=v.dtype)
    m4c = np.zeros((n2c*2, n2c*2), dtype=v.dtype)
    c = LIGHT_SPEED
    h4c[:n2c, :n2c] = v
    h4c[:n2c, n2c:] = t
    h4c[n2c:, :n2c] = t
    h4c[n2c:, n2c:] = w * (.25 / c**2) - t
    m4c[:n2c, :n2c] = s
    m4c[n2c:, n2c:] = t * (.5 / c**2)
    
    gfac_4c = int_gfac_4c(xmol, utm = True, h4c = h4c, m4c_inv = scipy.linalg.inv(m4c))
    for i in range(3):
        h4c += B_field[i] * gfac_4c[i] / c # the factor of half was taken in the integrals
    e, a = scipy.linalg.eigh(h4c, m4c)
    cl = a[:n2c, n2c:]
    cs = a[n2c:, n2c:]
    x = np.linalg.solve(cl.T, cs.T).T

    st = m4c[:n2c, :n2c] + reduce(np.dot, (x.T.conj(), m4c[n2c:, n2c:], x))
    tx = reduce(np.dot, (h4c[:n2c, n2c:], x))
    l = h4c[:n2c, :n2c] + tx + tx.T.conj() + reduce(np.dot, (x.T.conj(), h4c[n2c:, n2c:], x))
    from pyscf.x2c import x2c
    sa = x2c._invsqrt(m4c[:n2c, :n2c])
    sb = x2c._invsqrt(reduce(np.dot, (sa, st, sa)))
    r = reduce(np.dot, (sa, sb, sa, m4c[:n2c, :n2c]))
    hcore = reduce(np.dot, (r.T.conj(), l, r))
    hcore = reduce(np.dot, (contr_coeff.T.conj(), hcore, contr_coeff))
    soc_matrix = x2cobj.get_soc_integrals()

    return hcore + soc_matrix

from socutils.scf import spinor_hf
spinor_hf.JHF.Gfac = lib.class_as_method(Gfac)


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
    mol.basis = {"C": parse_genbas("C:APVTZ-DE4"), "O": parse_genbas("O:APVTZ-DE4")}
    mol.charge = 1
    mol.spin = 1
    mol.unit = 'bohr'
    mol.nucmod = 'g'
    mol.build()
    irrep = {'1/2':[6], '-1/2':[5], '3/2':[1], '-3/2':[1]}
    def mfobj(B_field):
        from socutils.scf import spinor_hf, x2camf_hf
        mf = spinor_hf.SymmSpinorSCF(mol, symmetry="linear", occup = irrep)
        mf.chkfile='chk.chk'
        mf.init_guess = 'chkfile'
        mf.with_x2c = x2camf_hf.SpinorX2CAMFHelper(mol, with_gaunt=True, with_breit=True)
        mf.get_hcore = lambda mol = None: get_hcore(x2cobj=mf.with_x2c, mol=mol, B_field=B_field)
        mf.damp = 0.5
        mf.diis_start_cycle = 30
        mf.max_cycle = 100
        mf.conv_tol = 1e-14
        return mf
    mf = mfobj([0.0, 0.0, 0.0])
    mf.kernel()
    gfac = mf.Gfac()
    # cfour value 0.500563438766573
    # finite difference
    dx = 0.01
    mf1 = mfobj([0.0, 0.0, dx])
    mf2 = mfobj([0.0, 0.0,-dx])
    e1 = mf1.kernel()
    e2 = mf2.kernel()
    mf3 = mfobj([0.0, 0.0, 2*dx])
    mf4 = mfobj([0.0, 0.0,-2*dx])
    mf5 = mfobj([0.0, 0.0, 3*dx])
    mf6 = mfobj([0.0, 0.0,-3*dx])
    e3 = mf3.kernel()
    e4 = mf4.kernel()
    e5 = mf5.kernel()
    e6 = mf6.kernel()
    gfacz2 = (e1 - e2)/dx/2.0*LIGHT_SPEED
    gfacz4 = (8*e1 - 8*e2 - e3 + e4)/12.0/dx*LIGHT_SPEED
    gfacz6 = (45*e1 - 45*e2 - 9*e3 + 9*e4 + e5 - e6)/60.0/dx*LIGHT_SPEED
    print(gfac[2])
    print(gfacz2)
    print(gfacz4)
    print(gfacz6)
    print((4.0*abs(gfac[2]) - G_ELECTRON)*10**6)
    
    
