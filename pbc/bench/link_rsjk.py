'''Direct short-range exchange with pyscf's own machinery (HANDOFF Open item 2, LinK/ONX option).

pyscf.pbc.scf.rsjk.RangeSeparatedJKBuilder builds the images of the cell (ft_ao.ExtendedMole, strip_basis),
the SR Schwarz table (PBCVHFnr_int2e_q_cond -> qindex, log-scaled int16) and the direct SR J/K with density
screening (PBCVHF_direct_drv).  With cell.omega = -w only the SR (erfc) part is built (no AFT long range) and
with exclude_dd_block = False the SR part is complete.  This script

  1. times one direct SR K build (and J+K) on the converged Gamma density of a production run: the spinor
     density as its four complex spin blocks D_aa, D_bb, D_ab, D_ba in the real spherical AO basis;
  2. counts the shell quartets (i in the reference cell, j k l over images) that pass
     Q_ij |D_jk| Q_kl >= tau with Q from qindex and D the condensed density, for a sample of i
     (pyscf's Gamma convention: D on every image pair, as PBCVHF_direct_drv screens).

With FULL = True in CONFIG the builder runs as pyscf sets it up by default instead: exclude_dd_block = True (the
smooth-smooth block of the diffuse shells goes to the AFT long-range part, so the SR images reach only as far as
the compact shells need) and the long range on; the SR and LR parts of K are timed separately.

usage: link_rsjk.py CONFIG.py   (SYSTEM, SUPERCELL, BASIS_PY, ACTIVE_LABEL, SCF_NPZ, OMEGA, TAUS, NSAMPLE_MU, FULL)'''
import sys, time, runpy, numpy as np
from pyscf import lib
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.pbc.df.incore import LOG_ADJUST
from socutils.pbc.bench import cells

cfg = runpy.run_path(sys.argv[1])
omega = cfg['OMEGA']
B = runpy.run_path(cfg['BASIS_PY'])['BASIS']
sysm = cells.from_cif(cfg['SYSTEM']) if cfg['SYSTEM'].endswith('.cif') else cells.SYSTEMS[cfg['SYSTEM']]
if cfg.get('SUPERCELL'):
    sysm = cells.supercell(sysm, cfg['SUPERCELL'])
if cfg.get('ACTIVE_LABEL'):
    k, at = 0, []
    for e, x in sysm[1]:
        if e == 'Eu':
            e = cfg['ACTIVE_LABEL'] if k == 0 else e
            k += 1
        at.append((e, x))
    sysm = (sysm[0], at)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
full = cfg.get('FULL', False)
if not full:
    cell.omega = -omega
nao = cell.nao
print('%s: %d atoms, nao %d, %d shells' % (cfg['SYSTEM'], cell.natm, nao, cell.nbas), flush=True)

t = time.time()
rs = RangeSeparatedJKBuilder(cell, kpts=np.zeros((1, 3)))
if not full:
    rs.exclude_dd_block = False
    rs.allow_drv_nodddd = False
rs.verbose = 5
rs.build(omega=omega)
sup = rs.supmol_sr
print('RSJK build (%s, omega %.2f): %.0f s; rs_cell %d shells, supmol %d shells / %d AOs, direct_scf_tol %.1e, has_long_range %s'
      % ('pyscf default, dd block to AFT' if full else 'SR only', rs.omega, time.time() - t, rs.rs_cell.nbas, sup.nbas, sup.nao, rs.direct_scf_tol, rs.has_long_range()), flush=True)

# converged spinor density -> the four spin blocks in the real spherical basis
z = np.load(cfg['SCF_NPZ'])
D2 = (z['mo_coeff'] * z['mo_occ']) @ z['mo_coeff'].conj().T
c2 = np.vstack(cell.sph2spinor_coeff())
Dso = c2 @ D2 @ c2.conj().T
dms = np.array([Dso[:nao, :nao], Dso[nao:, nao:], Dso[:nao, nao:], Dso[nao:, :nao]])[:, None]   # (4, nk=1, nao, nao)
kpts = np.zeros((1, 3))
for wj, wk, lab in ((False, True, 'K'), (True, True, 'J+K')):
    t = time.time()
    vs = rs._get_jk_sr(dms, hermi=0, kpts=kpts, with_j=wj, with_k=wk)
    print('direct SR %s, 4 complex spin-block densities, %d threads: %.0f s' % (lab, lib.num_threads(), time.time() - t), flush=True)
if full:
    print('mesh for the LR / dd part %s, smooth shells %d' % (rs.mesh, rs.cell_d.nbas if rs.cell_d is not None else 0), flush=True)
    t = time.time()
    vk_lr = rs._get_vk_lr(dms, hermi=0, kpts=kpts)
    print('LR (AFT) K incl. the dd block, 4 densities: %.0f s' % (time.time() - t), flush=True)

# ---- quartet statistic from qindex and the condensed density (rs_cell shells)
rs_cell = rs.rs_cell
nimgs = sup.bas_mask.shape[2]
bmap = sup.bas_map
rs_sh = (bmap // nimgs) % rs_cell.nbas                                 # supmol shell -> rs_cell shell
img = bmap % nimgs                                                     # image id (0 = reference cell)
ao_loc = rs_cell.ao_loc
# rs_cell AO -> cell AO (rs_cell splits contracted shells into segments)
ao_map = lib.locs_to_indices(cell.ao_loc, rs_cell.bas_map)
Dabs = np.max(np.abs(dms[:, 0]), axis=0)[np.ix_(ao_map, ao_map)]
Dsh = lib.condense('NP_absmax', Dabs, ao_loc, ao_loc)                  # (rs_nbas, rs_nbas)
qidx = rs.qindex[0]                                                    # (sup nbas, sup nbas) int16, log(sqrt((ij|ij)))*LOG_ADJUST
Qrow = lambda r: np.exp(np.asarray(qidx[r], dtype=np.float32) / LOG_ADJUST)
Dcl = Dsh[np.ix_(rs_sh, rs_sh)].astype(np.float32)
home = np.where(img == 0)[0]
rng = np.random.default_rng(0)
mus = rng.choice(home, min(len(home), cfg.get('NSAMPLE_MU', 60)), replace=False)
scale = len(home) / len(mus)
qmax, dmax = float(np.exp(qidx.max() / LOG_ADJUST)), Dcl.max()
for tau in cfg['TAUS']:
    t = time.time()
    tot = 0.
    for mu in mus:
        a = Qrow(mu)
        lam = np.where(a * dmax * qmax >= tau)[0]
        if len(lam) == 0:
            continue
        W = a[lam, None] * Dcl[lam]
        sig = np.where(W.max(axis=0) * qmax >= tau)[0]
        Qsort = -np.sort(-Qrow(sig), axis=1)
        for k, s_ in enumerate(sig):
            thr = tau / np.maximum(W[:, s_], 1e-30)
            tot += np.searchsorted(-Qsort[k], -thr, side='right').sum()
    print('tau %.0e: surviving shell quartets (i in the reference cell, %d of %d sampled, scaled) %.3e  (%.0f s)'
          % (tau, len(mus), len(home), tot * scale, time.time() - t), flush=True)
