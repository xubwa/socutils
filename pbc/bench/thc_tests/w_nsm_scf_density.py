'''W_NSM of the SCF density of EuOCl prim workdirs (examples/14 runs), e.g. to see the C4-forbidden W_x, W_y.

usage: python w_nsm_scf_density.py WORKDIR [WORKDIR ...]   (relative to P below)'''
import sys, runpy, numpy as np
from socutils.pbc.bench.cells import make_cell, from_cif
from socutils.pbc import atomic_prop
P = '/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/'
B = runpy.run_path(P + '../basis_rules.py')['BASIS']
cell = make_cell(from_cif('/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/struct/oxyhalides/COD_1542170.cif'), 1e-10,
                 nucmod='G', basis=B)
ieu = [i for i in range(cell.natm) if cell.atom_pure_symbol(i) == 'Eu']
molatom, op = atomic_prop.load_cfour_atom('/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/basis_test/cfour_eu/ETB0T10', 'Eu')
nsm = atomic_prop.atomic_block_operator(cell, ieu[0], molatom, op)
f = lambda v: np.array2string(np.asarray(v), precision=3)
for wd in sys.argv[1:]:
    z = np.load(P + wd + '/scf_hf.npz')
    dscf = (z['mo_coeff'] * z['mo_occ']) @ z['mo_coeff'].conj().T
    print('%-12s SCF W %s' % (wd, f(atomic_prop.expectation(nsm, dscf))), flush=True)
