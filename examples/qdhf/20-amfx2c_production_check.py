#!/usr/bin/env python

'''
Check of the production amfX2C flavor 'dirac_amf' of socutils.somf.x2cmp.

With pcc=True, x2camf.amfi returns x2c2ePCC = to_2c_K(G4c_K) - G2c_K[D2c_K]
+ (h1e_x2c2e - h1e_x2c1e)_K.  The last term converts an X2C-1e decoupling to
the atomic Fock decoupling; 'dirac_amf' already decouples h4c + sum_K G4c_K,
so the term is counted twice.  h_amfx2c_fixed adds the pure two-electron
correction only (amfX2C of Knecht, Repisky, Jensen, Saue, JCP 157, 114106
(2022)); it agrees with the quad implementation qdhf.x2c4c 'amfx2c'
(He3, 3e-14 Hartree).
'''

import os, contextlib, numpy
os.environ['X2CAMF_TIMING'] = '0'
from functools import reduce
from pyscf import gto, lib, scf
lib.einsum = lambda s, *o, **k: numpy.einsum(s, *o, optimize=True)
import socutils
import x2camf
from pyscf.data import elements
from pyscf.lib import param
from pyscf.x2c.x2c import _decontract_spinor
from socutils.scf import spinor_hf
from socutils.somf import x2cmp as X
from x2camf.x2camf import construct_molecular_matrix


def h_amfx2c_fixed(x2cobj):
    mol = x2cobj.mol
    atm_ints = {}
    for atom in set(a[0] for a in mol._atom):
        z = elements.charge(gto.mole._std_symbol(atom))
        raw = gto.mole.uncontracted_basis(mol._basis[atom])
        shell = numpy.array([b[0] for b in raw]); exp_a = numpy.array([b[-1][0] for b in raw])
        atm_ints[atom] = x2camf.libx2camf.atm_integrals(0, z, shell[-1]+1, len(shell), 0, shell, exp_a,
                                                        speed_of_light=param.LIGHT_SPEED)
    xmol, _ = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    sl = xmol.aoslice_2c_by_atom()
    g = lambda i, four: construct_molecular_matrix(X.extract_ith_integral(atm_ints, i), sl, xmol, n2c, four)
    mf4 = scf.DHF(xmol)
    h4, s4 = mf4.get_hcore(), mf4.get_ovlp()
    atm_x, atm_r = g(0, False), g(1, False)
    f4_2e, f2_2e = g(5, True), g(6, False)
    x, st, r, h2c = X.x2c1e_hfw0_4cmat(h4 + f4_2e, s4, mol=xmol)          # amfX2C step 6
    dF = X.to_2c(atm_x, atm_r, f4_2e) - f2_2e                              # pure 2e PCE, atomic blocks
    h = X.to_2c(x, r, h4) + dF
    xmol, contr = _decontract_spinor(mol, x2cobj.xuncontract)
    return reduce(numpy.dot, (contr.T.conj(), h, contr))


def energy(mol, flavor):
    mf = spinor_hf.SCF(mol)
    mf.x2cmp(flavor='x2cmp' if flavor == 'amf_fixed' else flavor,
             with_gaunt=False, with_breit=False, with_pcc=(flavor == 'dirac_amf'))
    if flavor == 'amf_fixed':
        hc = h_amfx2c_fixed(mf.with_x2c)
        mf.with_x2c.get_hcore = lambda *a: hc.copy()
    mf.conv_tol = 1e-12
    mf.verbose = 0
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        return mf.kernel()


if __name__ == '__main__':
    # for a single atom amfX2C must equal x2cmp (both reproduce the atomic 4c Fock exactly)
    for atom, basis in (('He', 'unc-cc-pvdz'), ('Ne', 'unc-cc-pvdz'), ('Ar', 'unc-cc-pvdz'),
                        ('Kr', 'unc-cc-pvdz-dk')):
        mol = gto.M(atom=atom + ' 0 0 0', basis=basis, verbose=0)
        e = {fl: energy(mol, fl) for fl in ('x2cmp', 'dirac_amf', 'amf_fixed')}
        print('%-3s x2cmp % .12f   dirac_amf(pcc) - x2cmp % .3e   fixed amfX2C - x2cmp % .3e' % (
            atom, e['x2cmp'], e['dirac_amf'] - e['x2cmp'], e['amf_fixed'] - e['x2cmp']), flush=True)
