'''GAPW J grid settings on a production cell (default EuOCl prim), against the tightest setting
(Treutler-Ahlrichs 3000 radial points); route-1 THCExchange common part only (no RSDF cderi).

usage: gapw_eu.py CIF BASIS_PY'''
import sys, time, runpy, numpy as np
from pyscf import gto
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf
from socutils.pbc.thc import THCExchange
from socutils.pbc.thc_gapw import GAPWJ
B = dict(runpy.run_path(sys.argv[2])['BASIS'])
sysm = cells.from_cif(sys.argv[1])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, 4.), max_memory=150000)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({gto.mole._std_symbol(e): B[e]}, contract=False, prune_lmax=False,
                            threshold=1e-8)[gto.mole._std_symbol(e)] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
th = THCExchange(cell, aux, 4., nuc_zeta=spinor_scf.nuclear_zeta(full), local_rcut=6.).build_common()
ref = None
import os
cfg = [('treutler_ahlrichs', 3000), ('treutler_ahlrichs', 800), ('treutler_ahlrichs', 1000), ('treutler_ahlrichs', 1500),
       ('treutler_ahlrichs', 2000), ('mura_knowles', 800), ('mura_knowles', 1000), ('mura_knowles', 1500),
       ('mura_knowles', 2000), ('mura_knowles', 3000), ('becke', 1000), ('becke', 2000)]
for rad_scheme, nrad in cfg:
    nang, nb = 302, False
    t = time.time(); g = GAPWJ.__new__(GAPWJ); g.radial = rad_scheme; g.__init__(th, nrad=nrad, nang=nang)
    g.neighbours = nb; ts = time.time() - t
    t = time.time(); J = g.get_j(dm); t1 = time.time() - t
    t = time.time(); J = g.get_j(dm); t2 = time.time() - t
    if ref is None:
        ref = J
    print('%s nrad %d: setup %.0f s, get_j %.0f / %.0f s; vs tightest max %.2e, E_J %.2e'
          % (rad_scheme, nrad, ts, t1, t2, abs(J - ref).max(), .5 * np.einsum('ij,ji', dm, J - ref)), flush=True)
    g = None
