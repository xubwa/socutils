#!/usr/bin/env python
'''
Spin-orbit counterpart of 33-heavy_background_darwin.py: picture change of a
diffuse Gaussian background charge at the nucleus,

    dh = h_X2C[V + u_bg] - (h_X2C[V] + u_bg),

with the X2C-1e Hamiltonian including spin-orbit coupling (spin-orbital basis)
and the GHF density of the free atom obtained with it.  Printed: the ratio of
Tr D dh to the complete-basis nonrelativistic-limit value -(pi/2c^2) rho0 N_e,
next to the spin-free ratio (density from sf-X2C-1e RHF).

Usage:  python 43-heavy_background_darwin_so.py [symbols...]
'''
import sys
import numpy
from scipy.linalg import block_diag
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.x2c import x2c

CHARGE = {'Au': 1}           # Au+ (5d10) for a closed-shell density
syms = sys.argv[1:] or ['Ar', 'Kr', 'Xe', 'Au', 'Hg', 'Rn']
Q = 1.0
zeta = 1e-3
c = lib.param.LIGHT_SPEED


def sigma_dot(wso, wsf):
    q = numpy.vstack([1j * lib.PauliMatrices, numpy.eye(2)[None]])
    n = wsf.shape[-1] * 2
    return numpy.einsum('sxy,spq->xpyq', q, numpy.vstack([wso, wsf[None]])).reshape(n, n)


print('ratio = (Tr D dh) / (-(pi/2c^2) rho0 N_e);  dyall-v2z (uncontracted), point nucleus, c = %.3f, zeta = %g' % (c, zeta))
print('%-4s %4s %12s %12s' % ('atom', 'Z', 'SFX2C-1e', 'X2C-1e'))
for symb in syms:
    basis = {symb: gto.uncontract(gto.load('dyall-v2z', symb))}
    mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, charge=CHARGE.get(symb, 0), verbose=0)
    t = mol.intor('int1e_kin')
    s = mol.intor('int1e_ovlp')
    v = mol.intor('int1e_nuc')
    w = mol.intor('int1e_pnucp')
    wso = mol.intor('int1e_pnucxp', comp=3)
    mol.set_rinv_zeta(zeta)
    vb = Q * mol.intor('int1e_rinv')
    wb = Q * mol.intor('int1e_prinvp')
    wbso = Q * mol.intor('int1e_prinvxp', comp=3)
    mol.set_rinv_zeta(0)
    ref = -numpy.pi / (2 * c**2) * Q * (zeta / numpy.pi)**1.5 * mol.nelectron
    # spin-free
    h0 = x2c._get_hcore_fw(t, v, w, s, x2c._x2c1e_xmatrix(t, v, w, s, c), c)
    h1 = x2c._get_hcore_fw(t, v + vb, w + wb, s, x2c._x2c1e_xmatrix(t, v + vb, w + wb, s, c), c)
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h0
    mf.conv_tol = 1e-10
    mf.kernel()
    r_sf = numpy.einsum('ij,ji', mf.make_rdm1(), h1 - h0 - vb) / ref
    dm0 = mf.make_rdm1()
    # spin-orbit
    bd = lambda a: block_diag(a, a)
    g0 = x2c._x2c1e_get_hcore(bd(t), bd(v), sigma_dot(wso, w), bd(s), c)
    g1 = x2c._x2c1e_get_hcore(bd(t), bd(v + vb), sigma_dot(wso + wbso, w + wb), bd(s), c)
    mg = scf.GHF(mol)
    mg.get_hcore = lambda *args: g0
    mg.conv_tol = 1e-10
    mg.kernel(bd(dm0 * .5) + 0j)
    r_so = (numpy.einsum('ij,ji', mg.make_rdm1(), g1 - g0 - bd(vb)) / ref).real
    print('%-4s %4d %12.4f %12.4f   (GHF converged %s, E_SO - E_sf = %.6f)' % (
        symb, mol.atom_charge(0), r_sf, r_so, mg.converged, mg.e_tot - mf.e_tot), flush=True)
