#!/usr/bin/env python

'''
Background Darwin term of periodic X2C-1e for heavy atoms, computed without
any periodic code (molecular sf-X2C-1e, exact integrals, no density fitting).

In a periodic X2C-1e calculation the Ewald nuclear potential contains the
uniform background -rho of the nuclei.  Near an atom it is a smooth potential
with lap u = -4 pi rho, X2C-1e picture-changes it while the background in J
is not transformed (30-, 32-).  Here the background is modelled by a diffuse
Gaussian charge Q (zeta/pi)^1.5 exp(-zeta r^2) at the nucleus (uniform over the
atom for small zeta; rho0 = Q (zeta/pi)^1.5) and

    dh = h_X2C[V + u_bg] - (h_X2C[V] + u_bg)

is the picture change of the background alone.  To first order the energy
Tr D dh (X2C-1e density of the free atom) is compared with the complete-basis
nonrelativistic-limit value  -(pi/2c^2) rho0 N_e,  orbital by orbital, at
c = 137.036 and at c = 2000 (basis factor), so that the ratio of the two shows
the relativistic enhancement of the core contributions.

Last column: the estimate for the fcc crystal, -(pi/2c^2) rho N_e times the
computed ratio, with rho = Z / Omega.

Usage:  python 33-heavy_background_darwin.py [symbols...]
'''

import sys
import numpy
from pyscf import lib, gto, scf
lib.einsum = numpy.einsum   # pyscf lib.einsum vs numpy 2.x
from pyscf.x2c import x2c

# fcc lattice constants / Angstrom (low temperature).  Hg and Rn are not
# measured fcc solids: they are given the Au and Xe lattice constants only to
# show the Z dependence (marked 'hypothetical' in the output).
FCC = {'Ne': 4.464, 'Ar': 5.26, 'Kr': 5.72, 'Xe': 6.20, 'Au': 4.078, 'Rn': 6.2, 'Hg': 4.078}
HYPOTHETICAL = ('Hg', 'Rn')
CHARGE = {'Au': 1}           # Au+ (5d10) for a closed-shell density
syms = sys.argv[1:] or ['Ar', 'Kr', 'Xe', 'Au', 'Hg', 'Rn']
Q = 1.0
zetas = (1e-3, 3e-4)


def background_pc(symb, c, zeta):
    basis = {symb: gto.uncontract(gto.load('dyall-v2z', symb))}
    mol = gto.M(atom='%s 0 0 0' % symb, basis=basis, charge=CHARGE.get(symb, 0), verbose=0)
    t = mol.intor('int1e_kin')
    s = mol.intor('int1e_ovlp')
    v = mol.intor('int1e_nuc')
    w = mol.intor('int1e_pnucp')
    mol.set_rinv_zeta(zeta)
    vb = Q * mol.intor('int1e_rinv')
    wb = Q * mol.intor('int1e_prinvp')
    mol.set_rinv_zeta(0)
    x0 = x2c._x2c1e_xmatrix(t, v, w, s, c)
    h0 = x2c._get_hcore_fw(t, v, w, s, x0, c)
    x1 = x2c._x2c1e_xmatrix(t, v + vb, w + wb, s, c)
    h1 = x2c._get_hcore_fw(t, v + vb, w + wb, s, x1, c)
    dh = h1 - h0 - vb
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h0
    mf.conv_tol = 1e-10
    mf.kernel()
    occ = mf.mo_occ > 0
    cocc = mf.mo_coeff[:, occ]
    e_orb = numpy.einsum('pi,pq,qi->i', cocc, dh, cocc)
    rho0 = Q * (zeta / numpy.pi)**1.5
    ref = -numpy.pi / (2 * c**2) * rho0          # per electron
    return mol, mf.mo_energy[occ], e_orb / ref, 2 * e_orb.sum() / (ref * mol.nelectron)


print('ratio = (Tr D dh) / (-(pi/2c^2) rho0 N_e);  dyall-v2z (uncontracted), point nucleus')
for symb in syms:
    res = {}
    for c in (2000., 137.036):
        lib.param.LIGHT_SPEED = c
        for zeta in zetas:
            res[c, zeta] = background_pc(symb, c, zeta)
    mol = res[137.036, zetas[0]][0]
    z = mol.atom_charge(0)
    r_nr = res[2000., zetas[0]][3]
    r_rel = res[137.036, zetas[0]][3]
    r_rel2 = res[137.036, zetas[1]][3]
    a = FCC[symb] / 0.529177210903
    omega = a**3 / 4
    rho = z / omega
    est = r_rel * -numpy.pi / (2 * 137.036**2) * rho * z
    print('\n%s (Z = %d, N_e = %d, Z/c = %.3f)' % (symb, z, mol.nelectron, z / 137.036))
    print('  ratio  c=2000: %.4f   c=137: %.4f (zeta %g)  %.4f (zeta %g)   relativistic factor %.4f' % (
        r_nr, r_rel, zetas[0], r_rel2, zetas[1], r_rel / r_nr))
    print('  fcc a = %.3f A: rho = %.4f, dE = %.3e Eh/atom = %.1f meV/atom, pressure %.3f GPa%s' % (
        FCC[symb], rho, est, est * 27211.386, est / omega * 29421.02,
        '  (hypothetical lattice)' if symb in HYPOTHETICAL else ''))
    eps, f137 = res[137.036, zetas[0]][1:3]
    fnr = res[2000., zetas[0]][2]
    print('  orbital  eps(c=137)      ratio(c=137)  ratio(c=2000)')
    for i in range(len(eps)):
        print('  %4d  %14.5f  %10.4f  %10.4f' % (i, eps[i], f137[i], fnr[i]))
    sys.stdout.flush()
