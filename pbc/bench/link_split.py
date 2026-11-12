'''Direct SR exchange (pyscf RSJK, SR kernel only, no dd exclusion) in the original basis and in basis sets split by
exponent intervals (sr_locality.split_basis): how much of the generally-contracted penalty is screening granularity.

usage: link_split.py SYSTEM PRECISION CUTS [CUTS ...]    CUTS: "none" or comma-separated exponents, e.g. 8 or 0.5,2,8,32'''
import sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.hybrid_k import exact_sr

name, prec = sys.argv[1], float(sys.argv[2])
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
dm0 = pscf.RHF(full).get_init_guess(key='minao')
Kref = None
for spec in sys.argv[3:]:
    if spec == 'none':
        cell, T = full, np.eye(full.nao)
    else:
        cuts = [float(x) for x in spec.split(',')]
        cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, cuts if len(cuts) > 1 else cuts[0]),
                               max_memory=150000)
        from pyscf.pbc import gto as pgto
        T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
    nprim = sum(cell.bas_nprim(i) * (2 * cell.bas_angular(i) + 1) for i in range(cell.nbas))
    print('\n== cuts %s: nao %d, %d shells, %d primitive functions' % (spec, cell.nao, cell.nbas, nprim), flush=True)
    t = time.time()
    K = T.T @ exact_sr(cell, T @ dm0 @ T.T, prec) @ T
    if Kref is None:
        Kref = K
    print('   SR K %.0f s; vs first: max %.1e, d(E_x) %.1e Eh' % (time.time() - t, abs(K - Kref).max(),
                                                               -.25 * np.einsum('ij,ji', dm0, K - Kref)), flush=True)
