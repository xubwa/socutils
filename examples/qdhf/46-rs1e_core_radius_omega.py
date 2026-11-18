#!/usr/bin/env python
'''
rs1e with an element-dependent omega_A = alpha / r_core,A: the short-range
potential -Z erfc(omega_A r)/r is confined to the atomic core (all shells
except the valence ns, np and (n-1)d).  r_core: rms radius of the outermost
core orbital of the free atom (nonrelativistic HF, see 45-/scratch estimate).
H has no core: its nucleus is not transformed (V_lr = V, only d_H).

alpha fixes what "covering the core" means:
    erfc(alpha) = 0.5   -> alpha = 0.477  (V_sr ~ V over the core, tail in the valence)
    erfc(alpha) = 1e-2  -> alpha = 1.821
    erfc(alpha) = 1e-4  -> alpha = 2.751  (V_sr confined within the core)

Bare nuclear frames, spin-free, double precision; deviations of the lowest
N/2 levels from X2C-1e (exact one-electron reference in the same basis).
'''
import sys
import numpy
import scipy.linalg
from pyscf import gto, lib
from pyscf.x2c import x2c

c = lib.param.LIGHT_SPEED
RCORE = {'Cl': 0.481, 'I': 0.985, 'Au': 1.204}
ALPHAS = [('erfc 0.5', 0.4769), ('erfc 1e-2', 1.8214), ('erfc 1e-4', 2.7511)]
MOLS = {
    'HCl': ('H 0 0 0; Cl 0 0 1.275', 'cc-pvdz'),
    'Cl2': ('Cl 0 0 0; Cl 0 0 1.988', 'cc-pvdz'),
    'HI': ('H 0 0 0; I 0 0 1.609', 'dyall-v2z'),
    'I2': ('I 0 0 0; I 0 0 2.666', 'dyall-v2z'),
    'AuH': ('Au 0 0 0; H 0 0 1.524', 'dyall-v2z'),
    'Au2': ('Au 0 0 0; Au 0 0 2.472', 'dyall-v2z'),
}


def build(atom, basis):
    bas = {}
    for e in set(a.split()[0] for a in atom.split(';')):
        b = basis if not (basis == 'dyall-v2z' and gto.charge(e) <= 2) else 'cc-pvdz'
        bas[e] = gto.uncontract(gto.load(b, e))
    return gto.M(atom=atom, basis=bas, verbose=0)


def nuc(mol, ia, omega=None):
    '''V and pV.p of nucleus ia; omega: long-range part -Z erf(omega r)/r'''
    with mol.with_rinv_origin(mol.atom_coord(ia)):
        if omega is not None:
            mol.set_rinv_zeta(omega**2)
        v = mol.intor('int1e_rinv')
        w = mol.intor('int1e_prinvp')
        mol.set_rinv_zeta(0)
    z = -mol.atom_charge(ia)
    return z * v, z * w


def hx(t, v, w, s):
    return x2c._x2c1e_get_hcore(t, v, w, s, c)


def h_rs1e(mol, t, s, v, w, omegas):
    vl = numpy.zeros_like(v)
    wl = numpy.zeros_like(w)
    d = numpy.zeros_like(v)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        va, wa = nuc(mol, ia)
        om = omegas[ia]
        vla, wla = (va, wa) if om is None else nuc(mol, ia, om)
        vl += vla
        wl += wla
        b = slice(p0, p1)
        d[b, b] = hx(t[b, b], va[b, b], wa[b, b], s[b, b]) - hx(
            t[b, b], va[b, b] - vla[b, b], wa[b, b] - wla[b, b], s[b, b]) - vla[b, b]
    return hx(t, v - vl, w - wl, s) + vl + d


if __name__ == '__main__':
    for name in sys.argv[1:] or list(MOLS):
        atom, basis = MOLS[name]
        mol = build(atom, basis)
        nocc = mol.nelectron // 2
        t, s = mol.intor('int1e_kin'), mol.intor('int1e_ovlp')
        v, w = mol.intor('int1e_nuc'), mol.intor('int1e_pnucp')
        e_ref = scipy.linalg.eigh(hx(t, v, w, s), s, eigvals_only=True)[:nocc]
        print('\n%s (%s unc.), bare frame, lowest %d sf levels; deviations rs1e - X2C-1e / microEh' % (name, basis, nocc))
        for lab, alpha in ALPHAS:
            oms = [None if mol.atom_symbol(i) == 'H' else alpha / RCORE[mol.atom_symbol(i)] for i in range(mol.natm)]
            e = scipy.linalg.eigh(h_rs1e(mol, t, s, v, w, oms), s, eigvals_only=True)[:nocc]
            dev = (e - e_ref) * 1e6
            print('  %-10s omega = %-16s level 1 % 9.3f   max |dev| % 9.3f (#%2d)   sum % 10.3f' % (
                lab, ','.join('H:-' if o is None else '%.2f' % o for o in oms), dev[0], abs(dev).max(),
                abs(dev).argmax() + 1, dev.sum()), flush=True)
