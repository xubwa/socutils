#!/usr/bin/env python

'''
Which term of the sf-X2C-1e transformation produces the spurious long-range
tails?

System: fragment A (molecule) + fragment B (far He atom), V = V_A + V_B.

    h_X2C = R^T (V + T X + X^T T - X^T T X + X^T W X / (4c^2)) R

The nuclear potential of the other fragment O enters the block of fragment F
(F, O = A, B) through its picture-change transformed form

    PC(V_O) = R^T (V_O + X^T W_O X / (4c^2)) R,      W_O = <p chi | V_O | p chi>

while the electrons of O interact with the electrons of F through the
untransformed Coulomb operator.  The far neutral fragment is therefore
"transformed nuclei + untransformed electrons", and the mismatch is the
picture-change error operator

    Delta_O = PC(V_O) - V_O

        = (R^T V_O R - V_O)  +  R^T X^T W_O X R / (4c^2)
          [renormalization]     [small-component potential]

To leading order in 1/c^2 (X = 1 + O(1/c^2), R = 1 - S^-1 T / (4c^2) + ...)

    Delta_O^(2) = (1/4c^2) [ W_O - T S^-1 V_O - V_O S^-1 T ]

In a complete basis W_O - (p^2 V_O + V_O p^2)/2 = (1/2) nabla^2 V_O, i.e.
Delta_O^(2) is the Darwin term (1/8c^2) nabla^2 V_O, a contact term at the
nuclei of O whose matrix in the basis of F is exponentially small.  In a
finite basis the resolution of the identity in T S^-1 V_O is incomplete, and
Delta^(2) acquires long-range multipoles.  Each half (renormalization part,
small-component part) alone contains the monopole Z/R; only their sum is
short ranged up to the basis-set incompleteness.

Variants (applied to the F-F blocks, F = A, B, in the decontracted basis):

    full        h_X2C
    -Delta      h_X2C - Delta_O                (O nuclei untransformed for F)
    -Delta+D2   h_X2C - Delta_O + Delta_O^(2)  (only the leading term kept)
    -R part     h_X2C - (R^T V_O R - V_O)
    -W part     h_X2C - R^T X^T W_O X R / (4c^2)

For approx='atom1e', X and R do not depend on V_O, so V_O enters h_X2C only
through PC(V_O) and '-Delta' leaves no artifact at all.  For approx='1e', X
depends on V_O; '-Delta' leaves the O(1/c^4) field-squared polarization of
the decoupling.

Usage:  python 06-x2c_decomposition.py [basis]
'''

import sys
from pyscf import gto
from socutils import qdhf
from socutils.qdhf import x2c, dhf, qarray
from socutils.qdhf import integrals as qints
from socutils.qdhf.qarray import QArray, qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
bohr = 1 / 0.52917721092
distances = [80, 160, 320]

systems = [
    # label, fragment A, n of the expected leading tail
    ('HF...He', 'F 0 0 0; H 0 0 0.917', 2),
    ('He2...He', 'He 0 0 -1.5; He 0 0 1.5', 3),
    ('He...He', 'He 0 0 0', 4),
]
variants = ['full', '-Delta', '-Delta+D2', '-R part', '-W part']


def fragment_hcores(mol, nat_a, approx):
    '''hcore (contracted basis of mol) for all variants.  Atoms [0, nat_a)
    form fragment A, the rest fragment B.'''
    xmol, contr = x2c.uncontract_mol(mol)
    d = x2c.get_xr(xmol, approx=approx)
    qi, c = d['qi'], d['c']
    h_full = x2c.hcore_from_xr(d)
    aoslices = xmol.aoslice_by_atom()
    blocks = {'A': slice(aoslices[0][2], aoslices[nat_a-1][3]),
              'B': slice(aoslices[nat_a][2], aoslices[-1][3])}
    atoms = {'A': list(range(nat_a)), 'B': list(range(nat_a, mol.natm))}
    hs = {k: h_full.copy() for k in variants}
    t, s, x, r = d['t'], d['s'], d['x'], d['r']
    for f, o in (('A', 'B'), ('B', 'A')):
        v_o = qi.nuc_x(atoms=atoms[o])
        vy = qi.nuc_y(atoms=atoms[o])
        w_o = vy[0::3, 0::3] + vy[1::3, 1::3] + vy[2::3, 2::3]
        r_part = r.T @ v_o @ r - v_o
        w_part = r.T @ (x.T @ w_o @ x) @ r * (1 / (4 * c**2))
        delta = r_part + w_part
        tsv = t @ qarray.solve(s, v_o)
        delta2 = (w_o - tsv - tsv.T) * (1 / (4 * c**2))
        b = blocks[f]
        for name, corr in (('-Delta', -delta), ('-Delta+D2', delta2 - delta),
                           ('-R part', -r_part), ('-W part', -w_part)):
            hs[name][b, b] = hs[name][b, b] + corr[b, b]
    return {k: contr.T @ h @ contr for k, h in hs.items()}


def scf_energy(mol, hcore):
    qi = qints.QuadIntegrals(mol)
    ham = dhf.Hamiltonian4c(hcore, qi.ovlp_x(), [(QArray.eye(qi.nao), 'X')],
                            {('X', 'X'): qi.eri_xxxx()},
                            qmp.mpf(qints.energy_nuc(mol)), None,
                            k_factor=0.5, occ_number=2)
    mf = dhf.RHF(mol)
    mf.build_hamiltonian = lambda: ham
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: SCF not converged')
    return e


def x2c_energy(mol, approx):
    mf = qdhf.SFX2C1E_RHF(mol)
    mf.approx = approx
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 100
    mf.verbose = 0
    return mf.kernel()


for approx in ('1e', 'atom1e'):
    for label, geom_a, n in systems:
        mol_a = gto.M(atom=geom_a, basis=basis, verbose=0)
        mol_b = gto.M(atom='He', basis=basis, verbose=0)
        e_ref = x2c_energy(mol_a, approx) + x2c_energy(mol_b, approx)
        print('\nsf-X2C-1e(%s)  %s   R^%d dE (a.u.), local exponent in ()' % (approx, label, n))
        print('%-10s' % 'variant' + ''.join('        R=%-4d A          ' % r for r in distances))
        rows = {k: [] for k in variants}
        for r in distances:
            mol = gto.M(atom=geom_a + '; He 0 0 %g' % r, basis=basis, verbose=0)
            hs = fragment_hcores(mol, mol_a.natm, approx)
            for k in variants:
                rows[k].append(scf_energy(mol, hs[k]) - e_ref)
        for k in variants:
            line = '%-10s' % k
            for i, r in enumerate(distances):
                de = rows[k][i]
                if i > 0 and de * rows[k][i-1] > 0:
                    ex = -float(qmp.log(de / rows[k][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
                    ex = '(%6.3f)' % ex
                else:
                    ex = '        '
                line += '  % .8e %s' % (float(de * (qmp.mpf(r) * bohr)**n), ex)
            print(line, flush=True)
