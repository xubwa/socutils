'''Exchange cost per SCF cycle: global RSDF cderi vs pyscf's direct RSJK (HANDOFF Open item 2, LinK option),
on one system and a series of its supercells, same real density for both.

For each supercell: the cell as in production (make_cell, precision 1e-10, basis rules), a minao density with its
natural orbitals as occupied factors, then
  RSDF (omega 0.5, auxgen uncontracted 1e-8, lr 'fit'):  build; K from the vectors (dm only) and occ-RI K;
  RSJK (omega 0.5, pyscf defaults: dd block to AFT):    build (supmol + qindex); direct SR K; LR (AFT) K;
and the relative difference of the two K matrices (same kernel, both without the G = 0 exchange correction).

usage: link_vs_rsdf.py SYSTEM BASIS_PY NX,NY,NZ [NX,NY,NZ ...]'''
import sys, time, runpy, numpy as np
from pyscf import lib, gto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from socutils.pbc.bench import cells
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.rsdf import RSDF

name, basis_py = sys.argv[1], sys.argv[2]
B = runpy.run_path(basis_py)['BASIS']
omega = 0.5
kpts = np.zeros((1, 3))
print('%s, %d threads' % (name, lib.num_threads()), flush=True)
for sc in sys.argv[3:]:
    sc = [int(x) for x in sc.split(',')]
    sysm = cells.supercell(cells.SYSTEMS[name], sc) if sc != [1, 1, 1] else cells.SYSTEMS[name]
    cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
    nao = cell.nao
    els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
    aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}

    dm = pscf.RHF(cell).get_init_guess(key='minao')
    S = cell.pbc_intor('int1e_ovlp')
    e, U = np.linalg.eigh(S)
    Sh, Shi = (U * e**.5) @ U.T, (U * e**-.5) @ U.T
    n, V = np.linalg.eigh(Sh @ dm @ Sh)
    keep = n > 1e-8
    C, occ = Shi @ V[:, keep], n[keep]
    dm_t = lib.tag_array(dm, mo_coeff=C, mo_occ=occ)
    print('\n== %s %s: %d atoms, nao %d, %d shells, nelec %.1f, %d occupied factors'
          % (name, sc, cell.natm, nao, cell.nbas, np.einsum('ij,ji', dm, S), len(occ)), flush=True)

    t = time.time()
    r = RSDF(cell, aux, omega=omega, precision=1e-10)
    r.cderi_in_memory = True
    r.verbose = 0
    r.build()
    t_rb = time.time() - t
    t = time.time()
    vk_df = r.get_jk(dm, hermi=1, with_j=False)[1]
    t_rk = time.time() - t
    t = time.time()
    r.get_jk(dm_t, hermi=1, with_j=False)
    t_ro = time.time() - t
    print('RSDF: naux %d, build %.0f s, K (dm) %.1f s, K (occ-RI) %.1f s' % (r.get_naoaux(), t_rb, t_rk, t_ro), flush=True)
    r = None

    t = time.time()
    rs = RangeSeparatedJKBuilder(cell, kpts=kpts)
    rs.verbose = 0
    rs.build(omega=omega)
    t_sb = time.time() - t
    dms = dm[None, None]
    t = time.time()
    vk_sr = rs._get_jk_sr(dms, hermi=1, kpts=kpts, with_j=False, with_k=True)[-1]   # stacked (J, K) when requested; K only here
    t_sk = time.time() - t
    t = time.time()
    vk_lr = rs._get_vk_lr(dms, hermi=1, kpts=kpts)
    t_lk = time.time() - t
    vk_rs = (np.asarray(vk_sr) + np.asarray(vk_lr)).reshape(nao, nao).real
    sup = rs.supmol_sr
    print('RSJK: supmol %d shells / %d AOs, smooth shells %d, mesh %s, direct_scf_tol %.1e; build %.0f s, SR K %.1f s, LR K %.1f s'
          % (sup.nbas, sup.nao, rs.cell_d.nbas if rs.cell_d is not None else 0, rs.mesh, rs.direct_scf_tol,
             t_sb, t_sk, t_lk), flush=True)
    print('K(RSJK) vs K(RSDF): max |diff| %.2e, max |K| %.2e' % (abs(vk_rs - vk_df).max(), abs(vk_df).max()), flush=True)
    print('SUMMARY %s %s nao %d: RSDF build %.0f K %.1f occK %.1f | RSJK build %.0f K %.1f (SR %.1f + LR %.1f)'
          % (name, 'x'.join(map(str, sc)), nao, t_rb, t_rk, t_ro, t_sb, t_sk + t_lk, t_sk, t_lk), flush=True)
    rs = None
