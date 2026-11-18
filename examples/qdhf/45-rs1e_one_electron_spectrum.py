#!/usr/bin/env python
'''
One-electron spectrum of rs1e against X2C-1e for the bare nuclear frame of a
molecule (spin-free, double precision, uncontracted basis).

For a one-electron Hamiltonian X2C-1e reproduces the positive-energy spectrum
of the Dirac matrix in the same basis exactly, and there are no long-range
spurious terms.  The deviation of the rs1e eigenvalues from the X2C-1e ones is
therefore the error of the range separation alone:

    h_rs1e = h_X2C[V_sr] + V_lr + sum_A d_A,
    V_lr,A = -Z_A erf(omega r_A)/r_A,  d_A = h_A[V_A] - h_A[V_sr,A] - V_lr,A

(an isolated atom is exactly X2C-1e).  Printed per omega: the deviation of
the lowest level, the largest deviation among the lowest N/2 levels (N: number
of electrons of the neutral molecule), and the deviation of their sum.

Usage:  python 45-rs1e_one_electron_spectrum.py [molecule ...]
'''
import sys
import numpy
import scipy.linalg
from pyscf import gto, lib
from pyscf.x2c import x2c

c = lib.param.LIGHT_SPEED
MOLS = {
    'HCl': ('H 0 0 0; Cl 0 0 1.275', 'cc-pvdz'),
    'Cl2': ('Cl 0 0 0; Cl 0 0 1.988', 'cc-pvdz'),
    'HI': ('H 0 0 0; I 0 0 1.609', 'dyall-v2z'),
    'I2': ('I 0 0 0; I 0 0 2.666', 'dyall-v2z'),
    'AuH': ('Au 0 0 0; H 0 0 1.524', 'dyall-v2z'),
    'Au2': ('Au 0 0 0; Au 0 0 2.472', 'dyall-v2z'),
}
omegas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def build(atom, basis):
    elems = set(a.split()[0] for a in atom.split(';'))
    bas = {}
    for e in elems:
        b = basis if not (basis == 'dyall-v2z' and gto.charge(e) <= 2) else 'cc-pvdz'
        bas[e] = gto.uncontract(gto.load(b, e))
    return gto.M(atom=atom, basis=bas, verbose=0)


def lr(mol, ia, omega):
    '''V_lr and pV_lr.p of nucleus ia: -Z erf(omega r)/r'''
    with mol.with_rinv_origin(mol.atom_coord(ia)):
        mol.set_rinv_zeta(omega**2)
        v = mol.intor('int1e_rinv')
        w = mol.intor('int1e_prinvp')
        mol.set_rinv_zeta(0)
    z = -mol.atom_charge(ia)
    return z * v, z * w


def full(mol, ia):
    with mol.with_rinv_origin(mol.atom_coord(ia)):
        v = mol.intor('int1e_rinv')
        w = mol.intor('int1e_prinvp')
    z = -mol.atom_charge(ia)
    return z * v, z * w


def h_x2c(t, v, w, s):
    return x2c._x2c1e_get_hcore(t, v, w, s, c)


def h_rs1e(mol, t, s, v, w, omega):
    vl = numpy.zeros_like(v)
    wl = numpy.zeros_like(w)
    d = numpy.zeros_like(v)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        va, wa = full(mol, ia)
        vla, wla = lr(mol, ia, omega)
        vl += vla
        wl += wla
        b = slice(p0, p1)
        tb, sb = t[b, b], s[b, b]
        d[b, b] = (h_x2c(tb, va[b, b], wa[b, b], sb) - h_x2c(tb, va[b, b] - vla[b, b], wa[b, b] - wla[b, b], sb)
                   - vla[b, b])
    return h_x2c(t, v - vl, w - wl, s) + vl + d


names = sys.argv[1:] or list(MOLS)
for name in names:
    atom, basis = MOLS[name]
    mol = build(atom, basis)
    nocc = mol.nelectron // 2
    t = mol.intor('int1e_kin')
    s = mol.intor('int1e_ovlp')
    v = mol.intor('int1e_nuc')
    w = mol.intor('int1e_pnucp')
    e_ref = scipy.linalg.eigh(h_x2c(t, v, w, s), s, eigvals_only=True)[:nocc]
    print('\n%s (%s uncontracted, nao %d), bare nuclear frame, lowest %d sf levels.  X2C-1e: e1 = %.6f, sum = %.6f Eh'
          % (name, basis, mol.nao, nocc, e_ref[0], e_ref.sum()))
    print('%7s %16s %22s %18s   (deviations rs1e - X2C-1e, microEh)' % ('omega', 'level 1', 'max |dev| lowest N/2', 'sum'))
    for om in omegas:
        e = scipy.linalg.eigh(h_rs1e(mol, t, s, v, w, om), s, eigvals_only=True)[:nocc]
        dev = (e - e_ref) * 1e6
        print('%7.2f %16.3f %16.3f (#%2d) %18.3f' % (om, dev[0], abs(dev).max(), abs(dev).argmax() + 1, dev.sum()),
              flush=True)
