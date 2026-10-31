'''THC route-1 total energy at a fixed density (minao guess) for several omega: exact Coulomb does not depend on omega,
so the spread measures the omega dependence of the local fit / THC error (for cells where SCF energies are not
comparable, e.g. open 4f shells).  usage: omega_fixed_dm.py CIF BASIS_PY OMEGA[:RCUT[:far]] ...   (RCUT Angstrom, default 6; far: far-field CX fit)'''
import sys, time, runpy, numpy as np
from pyscf import gto
from socutils.pbc.bench import cells
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.thc import THCJK
from socutils.scf import spinor_hf
B = dict(runpy.run_path(sys.argv[2])['BASIS'])
cell = cells.make_cell(cells.from_cif(sys.argv[1]), 1e-10, nucmod='G', basis=B, max_memory=150000)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({gto.mole._std_symbol(e): B[e]}, contract=False, prune_lmax=False,
                            threshold=1e-8)[gto.mole._std_symbol(e)] for e in els}
dm = None
for arg in sys.argv[3:]:
    parts = arg.split(':')
    om, rc = float(parts[0]), float(parts[1]) if len(parts) > 1 else 6.
    far = len(parts) > 2 and parts[2] == 'far'
    t = time.time()
    jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), omega=om, local_rcut=rc)
    jk.stream = True
    jk.far_fit = far                                           # explicit: the tag ':far' switches it on
    jk.build()
    mf = split_direct.periodic(spinor_hf.SpinorSCF(cell), jk)
    if dm is None:
        dm = mf.get_init_guess()
    vhf = mf.get_veff(cell, dm)
    e = mf.energy_tot(dm, h1e=mf.get_hcore(), vhf=vhf)
    print('omega %.2f r_cut %.1f far %s: E(dm) = %.10f, LR rows %d, build+E %.0f s' % (om, rc, far, e, jk.th.Llr_cx.shape[0], time.time() - t),
          flush=True)
