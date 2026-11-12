'''far_cx_fit building blocks on a production cell: the grid quadrature (Q | c x_L)_SR = int c x_L u_Q of one explicit
image pair against the analytic 3c integral (A's aux only).  usage: far_check.py CIF BASIS_PY'''
import sys, runpy, numpy as np
from pyscf.df import incore as df_incore, addons as df_addons
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.thc_local import _atom_mol, _sr_potential_at, _lebedev
from pyscf import gto
from pyscf.dft import radi
B = dict(runpy.run_path(sys.argv[2])['BASIS'])
cell = cells.make_cell(cells.from_cif(sys.argv[1]), 1e-10, nucmod='G', basis=split_basis(B, 4.), max_memory=100000)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({gto.mole._std_symbol(e): B[e]}, contract=False, prune_lmax=False,
                            threshold=1e-8)[gto.mole._std_symbol(e)] for e in els}
auxcell = df_addons.make_auxmol(cell, aux)
compact = np.repeat([cell.bas_exp(i).min() >= 4. for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
X = cell.atom_coords(); ao_loc = cell.aoslice_by_atom()[:, 2:4]
A = [i for i in range(cell.natm) if cell.atom_symbol(i).startswith('Eu')][0]
a0, a1 = ao_loc[A]; cA = np.where(compact[a0:a1])[0]
alat = np.asarray(cell.lattice_vectors())
for Bt, t in ((A, alat[0]), ((A + 1) % cell.natm, np.zeros(3)), ((A + 2) % cell.natm, alat[1])):
    mol = _atom_mol(cell, [A, Bt], [X[A], X[Bt] + t], cell._basis)
    amol = _atom_mol(auxcell, [A], [X[A]], auxcell._basis)
    nbA = sum(1 for i in range(mol.nbas) if mol.bas_atom(i) == 0)
    with mol.with_range_coulomb(-0.5), amol.with_range_coulomb(-0.5):
        j3 = df_incore.aux_e2(mol, amol, intor='int3c2e', aosym='s1', shls_slice=(0, nbA, nbA, mol.nbas, 0, amol.nbas))
    j3 = j3[cA]                                                       # (n_c, n_x, nQ)
    for nrad, nang in ((300, 302), (500, 590)):
        r, dr = radi.mura_knowles(nrad, cell.atom_charge(A)); sel = r < np.sqrt(36 / 4.)
        r, wr = r[sel], (dr * r ** 2)[sel]
        xa, wa = _lebedev(nang)
        pts = (r[:, None, None] * xa[None]).reshape(-1, 3) + X[A]; w = (wr[:, None] * wa[None]).ravel()
        cv = _atom_mol(cell, [A], [X[A]], cell._basis).eval_gto('GTOval_sph', pts)[:, cA]
        xv = _atom_mol(cell, [Bt], [X[Bt] + t], cell._basis).eval_gto('GTOval_sph', pts)
        u = _sr_potential_at(amol, pts, 0.5)
        Jg = np.einsum('gc,gx,gq,g->cxq', cv, xv, u, w)
        print('B %d |t| %.1f bohr, grid %dx%d: max |J_grid - J| %.2e (max |J| %.2e)' % (Bt, np.linalg.norm(X[Bt] + t - X[A]),
              nrad, nang, abs(Jg - j3).max(), abs(j3).max()), flush=True)
