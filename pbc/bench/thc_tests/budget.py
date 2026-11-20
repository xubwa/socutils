'''Memory budget of the THC route-1 pieces for a cell (counts only, no integrals): nao, smooth/compact, naux,
CX pairs within r_cut, LR G vectors at several omega, N_mu, vs the RSDF cderi.'''
import sys, runpy, numpy as np
from pyscf.df import addons as df_addons
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
name, sc, basis_py, ac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
sysm = cells.from_cif(name)
if sc != '1,1,1':
    sysm = cells.supercell(sysm, [int(x) for x in sc.split(',')])
B = dict(runpy.run_path(basis_py)['BASIS'])
lab = sys.argv[5] if len(sys.argv) > 5 else None
if lab:                                                    # first Eu relabelled (mixed basis)
    k, at = 0, []
    for e, x in sysm[1]:
        if e == 'Eu':
            e = lab if k == 0 else e
            k += 1
        at.append((e, x))
    sysm = (sysm[0], at)
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac))
labs = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
from pyscf import gto
aux = {l: auxgen_from_basis({gto.mole._std_symbol(l): B[l]}, contract=False, prune_lmax=False, threshold=1e-8)[gto.mole._std_symbol(l)] for l in labs}
naux = df_addons.make_auxmol(cell, aux).nao
nao = cell.nao
compact = np.repeat([cell.bas_exp(i).min() >= ac for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
aoat = np.repeat([cell.bas_atom(i) for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
ns = int((~compact).sum())
X = cell.atom_coords(); A = cell.lattice_vectors()
T = np.array([[i, j, k] for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)]) @ A
BOHR = 0.52917721092
print('%s %s: %d atoms, nao %d (split; original %d), smooth %d, compact %d, naux %d, vol %.0f A^3'
      % (name.split('/')[-1], sc, cell.natm, nao, full.nao, ns, nao - ns, naux, cell.vol * BOHR**3))
print('RSDF cderi (naux x npair, original basis): %.1f GB' % (naux * full.nao * (full.nao + 1) / 2 * 8e-9))
for rc in (5., 6.):
    # image pairs (c compact on A, x on B + L) within rc; each counted once
    npairs = 0
    for a in range(cell.natm):
        nc = int(compact[aoat == a].sum())
        if nc == 0:
            continue
        for b in range(cell.natm):
            d = np.linalg.norm(X[b] + T - X[a], axis=1) * BOHR
            npairs += nc * int((aoat == b).sum()) * int((d <= rc).sum())
    # local C storage: each image pair has aux of two atoms (~2 x naux/natm)
    print('r_cut %.0f A: CX image pairs %.2e; local C (pair x 2 atoms of aux) ~ %.1f GB' % (rc, npairs, npairs * 2 * naux / cell.natm * 8e-9))
nmu = 16 * ns
print('THC N_mu = 16 x %d = %d: V %.1f GB, A (naux x N_mu) %.1f GB' % (ns, nmu, nmu**2 * 8e-9, naux * nmu * 8e-9))
nCX = int((compact[:, None] | compact[None, :]).sum() / 2)
for w in (0.25, 0.5):
    kc = (2 * w)**2 * 2 * np.log(1e10)          # LR cutoff ~ G^2/(4 w^2) = ln(1/eps)
    nG = cell.vol / (6 * np.pi**2) * kc**1.5 / 2
    print('omega %.2f: ~%d LR G vectors (half); LR rows of the CX Gamma pairs (%.2e pairs): %.1f GB' % (w, nG, nCX, nCX * 2 * nG * 8e-9))
