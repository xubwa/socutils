"""One state, full SCF trace (verbose 4 log), to see how the open-shell SCF fails to converge.
    python scf_trace.py ADYVDZ-SOG contracted U
"""
import sys, re
import numpy as np
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.tools.basisopt import genbas
name, kind, state = sys.argv[1:4]
G = '/home/xwang405/scr4/zmr_runs/basis_opt/thuof/'
ent = genbas.read(G + ('GENBAS.thuof_aug' if name.startswith('A') else 'GENBAS.thuof'), 'U:' + name)
basis = genbas.to_pyscf(ent, kappa=True) if kind == 'contracted' else \
        [[l, [e, 1.0]] for l, (ex, c) in sorted(ent.shells.items()) for e in sorted(ex, reverse=True)]
CLOSED = {'s1/2': 7, 'p1/2': 5, 'p3/2': 5, 'd3/2': 3, 'd5/2': 3, 'f5/2': 1, 'f7/2': 1}
OPEN = {('f5/2', '5/2'): 1, ('f5/2', '3/2'): 1, ('f5/2', '1/2'): 1, ('d3/2', '3/2'): 1}
EXTRA = {'U': {}, 'U-1': {('p1/2', '1/2'): 1}, 'U-2': {('d3/2', '1/2'): 1}}[state]
charge = 0 if state == 'U' else -1
log = 'trace_%s_%s_%s.log' % (name, kind, state)
mol = gto.M(atom='U 0 0 0', basis={'U': basis}, charge=charge, spin=(92 - charge) % 2, verbose=4, output=log, nucmod='G')
mf = spinor_hf.SymmSpinorSCF(mol, symmetry='sph').x2camf()
occ = {}
for key in mf.irrep_ao:
    m = re.match(r'([spdfghi])(\d+/2),(-?\d+/2)', key)
    n = CLOSED.get(m.group(1) + m.group(2), 0) + {**OPEN, **EXTRA}.get((m.group(1) + m.group(2), m.group(3)), 0)
    if n: occ[key] = [n]
mf.occupation = occ
mf.conv_tol = 1e-9; mf.max_cycle = 150
mf.kernel()
