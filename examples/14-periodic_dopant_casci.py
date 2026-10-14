#!/usr/bin/env python
'''
Gamma-point periodic two-component calculation of a lanthanide dopant:
RSDF Coulomb integrals -> Kramers-restricted HF or KS (--xc) with one-centre
X2CAMF and the dopant 4f shell fractionally occupied (chosen by character) ->
CASCI(4f) on the 4f spinors.

    python 14-periodic_dopant_casci.py eu_yso_x2 --workdir run_eu --max-memory 400000

Every stage writes to --workdir and is reused on a second run:
  cderi.h5   the RSDF vectors, S, V_ne, E_nn (RSDF.load)      [the expensive step]
  scf_<xc>.npz    converged orbitals, occupations, energies    (SCF restart guess)
  cahf_<xc>.npz   CAHF orbitals (--no-cahf: CASCI on the fractional-occupation ones)
  casci_<xc>.npz  CASCI energies

Structures: socutils.pbc.bench.cells.SYSTEMS (eu_y2o3A, eu_yso, eu_yso_x2,
eu_yso_x3), or adapt make_cell for your own CIF/coordinates.  The basis (x2c-SVPall
with exponents < 0.1 removed) and the auxiliary basis (Lehtola's auxgen,
uncontracted) are those of the benchmarks; replace them for production.

Cost and memory (see pbc/DEVELOPMENT.md): the vectors take naux*nao^2/2 doubles
(eu_yso_x2: ~150 GB); with --max-memory above twice that they stay in memory,
otherwise they are streamed from cderi.h5 every SCF iteration.
'''
import os
import sys
import time
import argparse
import numpy
from pyscf import lib
from socutils.pbc.bench.cells import make_cell, SYSTEMS
from socutils.pbc.bench.auxgen_util import auxgen
from socutils.pbc.rsdf import RSDF
from socutils.pbc import spinor_scf, spinor_ks
from socutils.scf import spinor_hf, addons
import io
import contextlib
from socutils.mcscf import zcasci, zmcscf, zcahf, cahf_diis

p = argparse.ArgumentParser()
p.add_argument('system', choices=sorted(SYSTEMS))
p.add_argument('--workdir', default='run_dopant')
p.add_argument('--dopant', default='Eu')
p.add_argument('--nelec-open', type=int, default=6, help='4f electrons (Eu3+: 6)')
p.add_argument('--norb-open', type=int, default=14, help='4f spinors')
p.add_argument('--omega', type=float, default=0.5)
p.add_argument('--precision', type=float, default=1e-10)
p.add_argument('--max-memory', type=float, default=8000, help='MB')
p.add_argument('--nroots', type=int, default=13)
p.add_argument('--no-gaunt', action='store_true')
p.add_argument('--xc', default=None, help='KS functional (e.g. PBE, PBE0); default HF')
p.add_argument('--grids-level', type=int, default=4)
p.add_argument('--no-cahf', action='store_true', help='CASCI directly on the fractional-occupation orbitals')
p.add_argument('--cahf-max-cycle', type=int, default=100)
p.add_argument('--cahf-driver', choices=('superci', 'diis'), default='superci',
               help='superci: second-order (zmcscf.CASSCF.kernel); diis: cahf_diis.mc_diis')
args = p.parse_args()

os.makedirs(args.workdir, exist_ok=True)
f_cderi = os.path.join(args.workdir, 'cderi.h5')
f_scf = os.path.join(args.workdir, 'scf_%s.npz' % (args.xc or 'hf'))
f_cas = os.path.join(args.workdir, 'casci_%s.npz' % (args.xc or 'hf'))
f_cahf = os.path.join(args.workdir, 'cahf_%s.npz' % (args.xc or 'hf'))
log = lambda *a: print(*a, flush=True)

cell = make_cell(args.system, args.precision, nucmod='G', max_memory=args.max_memory)
idop = [i for i in range(cell.natm) if cell.atom_symbol(i) == args.dopant]
assert len(idop) == 1, 'one %s atom expected' % args.dopant
log('%s: %d atoms, nao %d, n2c %d, %d electrons' % (args.system, cell.natm, cell.nao, cell.nao_2c(), cell.nelectron))

# 1. RSDF (built once)
t = time.time()
if os.path.exists(f_cderi):
    r = RSDF.load(cell, f_cderi)
    r.max_memory = args.max_memory
    log('RSDF loaded from %s (naux kept %d)' % (f_cderi, r.naux_kept))
else:
    aux = auxgen(cell, contract=False, prune_lmax=False, threshold=1e-8)
    r = RSDF(cell, aux, omega=args.omega, precision=args.precision,
             nuc_zeta=spinor_scf.nuclear_zeta(cell))
    r.cderi_file = f_cderi
    r.verbose = 4
    r.stdout = sys.stdout
    r.build()
    log('RSDF built in %.0f s, naux kept %d, metric eigenvalues [%.1e, %.1e]'
        % (time.time() - t, r.naux_kept, *r.metric_eig))

# 2. KRHF + one-centre X2CAMF, dopant 4f shell fractionally occupied
if args.xc:
    mf = spinor_ks.periodic_ks(spinor_hf.KRHF(cell), r, xc=args.xc)
    mf.grids_level = args.grids_level
else:
    mf = spinor_scf.periodic(spinor_hf.KRHF(cell), r)
mf = mf.x2camf(with_gaunt=not args.no_gaunt, with_breit=not args.no_gaunt)
mf.max_cycle = 100
mf.conv_tol = 1e-9
mf.verbose = 4
mf.stdout = sys.stdout
idx = addons.spinor_ao_indices(cell, idop[0], l=3)
addons.projected_frac_occ(mf, idx, args.norb_open, args.nelec_open)
dm0 = None
if os.path.exists(f_scf):
    z = numpy.load(f_scf)
    dm0 = lib.tag_array((z['mo_coeff'] * z['mo_occ']) @ z['mo_coeff'].conj().T,
                        mo_coeff=z['mo_coeff'], mo_occ=z['mo_occ'])
    log('SCF restarted from %s' % f_scf)
t = time.time()
e_scf = mf.kernel(dm0)
numpy.savez(f_scf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy, e_tot=e_scf)
w = addons.subspace_weights(mf.get_ovlp(), mf.mo_coeff, idx)
log('SCF converged=%s E = %.10f (%.0f s, %d cycles); open-shell 4f weights %.3f..%.3f'
    % (mf.converged, e_scf, time.time() - t, mf.cycles, w[mf.open_shell_mo].min(), w[mf.open_shell_mo].max()))

# 3. CAHF: average-of-configuration (ensemble of all 4f^n determinants) orbital
# optimisation with the true Hamiltonian, from the fractional-occupation
# orbitals (or a previous CAHF run)
mo, ncore = addons.cas_order(mf)
if not args.no_cahf:
    if os.path.exists(f_cahf):
        z = numpy.load(f_cahf)
        mo = z['mo_coeff']
        log('CAHF orbitals loaded from %s (E = %.10f)' % (f_cahf, float(z['e_tot'])))
    else:
        mc = zmcscf.CASSCF(mf, args.norb_open, args.nelec_open, ncore=ncore)
        mc.fcisolver = zcahf.CAHF(cell)
        mc.max_cycle_macro = args.cahf_max_cycle
        mc.conv_tol = 1e-9
        mc.verbose = 4
        t = time.time()
        with open(os.path.join(args.workdir, 'cahf.log'), 'w') as fcahf, contextlib.redirect_stdout(fcahf):
            mc.stdout = fcahf
            if args.cahf_driver == 'superci':
                mc.kernel(mo)
            else:
                cahf_diis.mc_diis(mc, mo, conv_tol=1e-9)
        mo = mc.mo_coeff
        numpy.savez(f_cahf, mo_coeff=mo, e_tot=mc.e_tot, ncore=ncore)
        w = addons.subspace_weights(mf.get_ovlp(), mo[:, ncore:ncore + args.norb_open], idx)
        log('CAHF (%s) E = %.10f converged %s (%.0f s); active 4f weights %.3f..%.3f'
            % (args.cahf_driver, mc.e_tot, getattr(mc, 'converged', None), time.time() - t, w.min(), w.max()))

# 4. CASCI on the 4f spinors
mc = zcasci.CASCI(mf, args.norb_open, args.nelec_open, ncore=ncore)
mc.fcisolver.nroots = args.nroots
mc.verbose = 4
mc.stdout = sys.stdout
t = time.time()
mc.kernel(mo)
e = numpy.sort(numpy.atleast_1d(mc.e_tot))
numpy.savez(f_cas, e_tot=e, mo_coeff=mo, ncore=ncore)
log('CASCI(%de,%d) %.0f s; E0 = %.10f; levels (cm-1): %s'
    % (args.nelec_open, args.norb_open, time.time() - t, e[0],
       numpy.array2string((e - e[0]) * 219474.63, precision=1, max_line_width=200)))
