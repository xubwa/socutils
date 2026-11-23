"""Where does the orbital gradient of the (l, j, m_j)-constrained SCF point?  Neutral U, M_J = 6,
ADYVDZ-SOG contracted (as in ea_sph.py).  After the masked SCF has converged, the full (unmasked)
Fock matrix is built from its density and the occupied-virtual elements are grouped by the
(l, j, m_j) blocks of the occupied and virtual spinors; each spinor is also labelled by its
energy so that shells can be told apart (core / open shell / virtual)."""
import re, sys
import numpy as np
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools.basisopt import genbas
G = '/home/xwang405/scr4/zmr_runs/basis_opt/thuof/'
name, src = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('ADYVDZ-SOG', 'GENBAS.thuof_aug')
ent = genbas.read(G + src, 'U:' + name)
mol = gto.M(atom='U 0 0 0', basis={'U': genbas.to_pyscf(ent, kappa=True)}, verbose=0, nucmod='G', max_memory=50000)
CLOSED = {'s1/2': 7, 'p1/2': 5, 'p3/2': 5, 'd3/2': 3, 'd5/2': 3, 'f5/2': 1, 'f7/2': 1}
OPEN = {('f5/2', '5/2'): 1, ('f5/2', '3/2'): 1, ('f5/2', '1/2'): 1, ('d3/2', '3/2'): 1}
mf = spinor_hf.SymmSpinorSCF(mol, symmetry='sph').x2camf()
occ = {}
for key in mf.irrep_ao:
    m = re.match(r'([spdfghi])(\d+/2),(-?\d+/2)', key)
    n = CLOSED.get(m.group(1) + m.group(2), 0) + OPEN.get((m.group(1) + m.group(2), m.group(3)), 0)
    if n: occ[key] = [n]
mf.occupation = occ
lab = np.empty(mol.nao_2c(), dtype=object)
for key, idx in mf.irrep_ao.items():
    lab[idx] = key
mask = (lab[:, None] == lab[None, :])
veff0 = mf.get_veff
mf.get_veff = lambda *a, **k: veff0(*a, **k) * mask
mf.conv_tol = 1e-10
e = mf.kernel()
print('constrained SCF E = %.10f, converged %s' % (e, mf.converged), flush=True)
dm = mf.make_rdm1()
fock = mf.get_hcore() + veff0(mol, dm)                     # unmasked
C, eps, occd = mf.mo_coeff, mf.mo_energy.real, abs(mf.mo_occ) > 0.5
tags = np.asarray(mf.irrep_mo)
fo = C.conj().T @ fock @ C
g = fo[np.ix_(~occd, occd)]
gfull = np.sqrt(2 * np.sum(abs(g) ** 2))
gin = np.sqrt(2 * np.sum(abs(g[tags[~occd][:, None] == tags[occd][None, :]]) ** 2))
print('|g| unconstrained %.4f, inside the blocks %.2e' % (gfull, gin))
# name each spinor: block + rank within block (n counted from l+1 for occupied, 'v' for virtual)
def nm(i):
    t = tags[i]; l = 'spdfghi'.index(t[0])
    same = np.where(tags == t)[0]
    k = int(np.where(same[np.argsort(eps[same])] == i)[0][0])
    return '%d%s' % (k + l + 1, t) if occd[i] else 'v%d %s' % (k - sum(occd[same]) + 1, t)
oi, vi = np.where(occd)[0], np.where(~occd)[0]
w = abs(g) ** 2
order = np.dstack(np.unravel_index(np.argsort(-w, axis=None), w.shape))[0][:25]
print('largest occupied -> virtual couplings (|F_ai|, occupied spinor (eps), virtual spinor (eps)):')
for a, i in order:
    print('   %.4f   %-14s %+9.4f  ->  %-16s %+9.4f' % (np.sqrt(w[a, i]), nm(oi[i]), eps[oi[i]], nm(vi[a]), eps[vi[a]]))
# grouped by kind of pair
def lj(t): return t.split(',')[0]
groups = {}
for a in range(len(vi)):
    for i in range(len(oi)):
        to, tv = tags[oi[i]], tags[vi[a]]
        if to == tv: continue
        lo, lv = 'spdfghi'.index(to[0]), 'spdfghi'.index(tv[0])
        kind = 'same l, other j' if lo == lv else 'l differs by %d' % abs(lo - lv)
        sh = 'open' if eps[oi[i]] > -1.0 else ('outer core' if eps[oi[i]] > -10 else 'deep core')
        key = (kind, sh)
        groups[key] = groups.get(key, 0) + 2 * w[a, i]
tot = sum(groups.values())
print('share of |g|^2 by kind of rotation and by the occupied spinor (open: eps > -1 Eh, outer core: > -10 Eh):')
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print('   %-20s %-11s %5.1f %%' % (k[0], k[1], 100 * v / tot))
