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
from socutils.pbc.bench.cells import make_cell, SYSTEMS, from_cif, from_cif_relaxed, supercell
from socutils.pbc.bench.auxgen_util import auxgen, auxgen_from_basis
import runpy
from pyscf import gto
from socutils.pbc.rsdf import RSDF
from socutils.pbc import spinor_scf, spinor_ks
from socutils.scf import spinor_hf, addons
import io
import contextlib
from socutils.mcscf import zcasci, zmcscf, zcahf, cahf_diis, cahf_fock, frozen_frac

p = argparse.ArgumentParser()
p.add_argument('system', help='a name of cells.SYSTEMS, or a CIF file (primitive cell)')
p.add_argument('--xyz', default=None, help='with a CIF: take the positions from the first atoms of this (relaxed) xyz')
p.add_argument('--supercell', default=None, help='supercell matrix of the (primitive) cell, e.g. "2,2,1" or 9 integers')
p.add_argument('--active', type=int, default=0,
               help='with several dopant atoms in the cell (stoichiometric compound): index (among them) of the '
                    'one whose open shell is active; the others are frozen as their configuration average')
p.add_argument('--workdir', default='run_dopant')
p.add_argument('--dopant', default='Eu')
p.add_argument('--nelec-open', type=int, default=6, help='4f electrons (Eu3+: 6)')
p.add_argument('--norb-open', type=int, default=14, help='4f spinors')
p.add_argument('--open-ref', default=None,
               help='npz with ref = reference open-shell spinors of one dopant atom (its AO block): the open shell '
                    'is chosen by projection on them placed on every dopant (addons.reference_frac_occ)')
p.add_argument('--active-label', default=None,
               help='relabel the active dopant atom (e.g. Eu1) so that it can carry its own basis (BASIS[label]); '
                    'the other dopant atoms keep BASIS[dopant] (mixed basis: the property site in the large basis)')
p.add_argument('--guess-from', default=None,
               help='workdir of a smaller cell (same CIF, its --supercell given by --guess-supercell): its converged '
                    'SCF density folded into this cell as the initial guess (blocks of atoms whose basis differs '
                    'are projected through the isolated-atom overlap)')
p.add_argument('--guess-supercell', default=None, help='supercell matrix of the --guess-from cell')
p.add_argument('--guess-active-label', default=None, help='--active-label of the --guess-from run')
p.add_argument('--jk', choices=('rsdf', 'thc'), default='rsdf',
               help="mean-field J/K: 'rsdf' (global cderi) or 'thc' (socutils.pbc.thc.THCJK: compact pairs locally "
                    "fitted, smooth pairs by THC, J by RI-J on the grid; no cderi; needs --lr gspace and "
                    "--active-eri three-index)")
p.add_argument('--thc-alpha-c', type=float, default=4., help='THCJK: compact/smooth exponent split')
p.add_argument('--thc-nmu', type=float, default=16., help='THCJK: interpolation points per smooth function')
p.add_argument('--thc-max-memory', type=float, default=None,
               help='THCJK: memory budget per MPI rank (MB, default --max-memory); the streamed build, the phi '
                    'storage and the K caches are sized to it')
p.add_argument('--thc-local-rcut', type=float, default=0., help='THCJK: pair-atom local fit of the compact pairs within this distance (A), non-robust K (0: Gamma Bloch-domain fit)')
p.add_argument('--active-eri', choices=('cderi', 'three-index'), default='cderi',
               help="CASCI (tu|vw): 'cderi' (_CDERIS on the stored vectors) or 'three-index' (RSDF.active_vectors: "
                    "SR B_P,tu fitted + exact LR rho_tu(G), no transformed global tensor; needs --lr gspace)")
p.add_argument('--open-l', type=int, default=3, help='angular momentum of the open shell (4f: 3)')
p.add_argument('--omega', type=float, default=0.5)
p.add_argument('--lr', choices=('fit', 'gspace'), default='fit',
               help="RSDF long range: 'fit' (whole kernel fitted) or 'gspace' (SR fitted, LR exact "
                    "plane-wave rows; ~V (2 omega)^3 extra rows, use a smaller --omega, e.g. 0.3)")
p.add_argument('--precision', type=float, default=1e-10)
p.add_argument('--max-memory', type=float, default=8000, help='MB')
p.add_argument('--lindep-scaled', action='store_true', help='RSDF lindep cut on the normalised metric')
p.add_argument('--cderi-in-memory', action='store_true',
               help='keep the RSDF vectors in memory only (needs --max-memory > 2x their size); not written to '
                    'cderi.h5, so a rerun rebuilds them (the SCF/CAHF/CASCI stage files are still reused)')
p.add_argument('--nroots', type=int, default=13)
p.add_argument('--no-gaunt', action='store_true')
p.add_argument('--xc', default=None, help='KS functional (e.g. PBE, PBE0); default HF')
p.add_argument('--basis-py', default=None,
               help='python file defining BASIS = {symbol: pyscf basis} (default: x2c-SVPall without exponents < 0.1); '
                    'the auxiliary basis is generated from it')
p.add_argument('--nsm-dir', default=None,
               help='dopant-atom CFOUR directory (config.py with BASIS_MAP/GENBAS, SOSFMO2C): <W_NSM> of CASCI root 0')
p.add_argument('--grids-level', type=int, default=4)
p.add_argument('--no-cahf', action='store_true', help='CASCI directly on the fractional-occupation orbitals')
p.add_argument('--cahf-max-cycle', type=int, default=100)
p.add_argument('--scf-conv-tol', type=float, default=1e-9, help='SCF energy threshold (large cells: DF noise ~1e-9)')
p.add_argument('--cahf-conv-tol', type=float, default=1e-9, help='CAHF energy threshold')
p.add_argument('--cahf-conv-grad', type=float, default=1e-5, help='CAHF orbital-gradient threshold')
p.add_argument('--cahf-driver', choices=('fock', 'superci', 'diis'), default='fock',
               help='fock: integral-free (cahf_fock, J/K of the core and 4f densities only); '
                    'superci: second-order (zmcscf.CASSCF.kernel); diis: cahf_diis.mc_diis')
args = p.parse_args()

# MPI (SPMD, --jk thc only): every rank runs this script; the THC build and every J/K (SCF, CAHF, CASCI core) are
# split over the ranks and summed, rank 0 broadcasts the orbitals/densities, so all ranks follow the same path.
# Only rank 0 prints and writes files.   mpirun -np N python -m mpi4py 14-periodic_dopant_casci.py ... --jk thc
comm = None
if any(k in os.environ for k in ('PMI_SIZE', 'PMI_RANK', 'OMPI_COMM_WORLD_SIZE', 'PMIX_RANK')):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        comm = MPI.COMM_WORLD
rank = comm.Get_rank() if comm is not None else 0
root = rank == 0
if comm is not None and args.jk != 'thc':
    raise SystemExit('MPI runs need --jk thc')
if root:
    os.makedirs(args.workdir, exist_ok=True)
if comm is not None:
    comm.Barrier()
f_cderi = os.path.join(args.workdir, 'cderi.h5')
f_scf = os.path.join(args.workdir, 'scf_%s.npz' % (args.xc or 'hf'))
f_cas = os.path.join(args.workdir, 'casci_%s.npz' % (args.xc or 'hf'))
f_cahf = os.path.join(args.workdir, 'cahf_%s.npz' % (args.xc or 'hf'))
log = (lambda *a: print(*a, flush=True)) if root else (lambda *a: None)
devnull = open(os.devnull, 'w')

basis = runpy.run_path(args.basis_py)['BASIS'] if args.basis_py else None
if args.system.lower().endswith('.cif'):
    system = from_cif_relaxed(args.system, args.xyz) if args.xyz else from_cif(args.system)
else:
    system = args.system
if args.supercell:
    m = [int(x) for x in args.supercell.split(',')]
    system = supercell(SYSTEMS[system] if isinstance(system, str) else system,
                       m if len(m) == 3 else numpy.reshape(m, (3, 3)))
if args.active_label:
    A_, atoms_ = SYSTEMS[system] if isinstance(system, str) else system
    k_, atoms2 = 0, []
    for e_, x_ in atoms_:
        if e_ == args.dopant:
            if k_ == args.active:
                e_ = args.active_label
            k_ += 1
        atoms2.append((e_, x_))
    system = (A_, atoms2)
if basis:
    cell = make_cell(system, args.precision, nucmod='G', max_memory=args.max_memory, basis=basis)
else:
    cell = make_cell(system, args.precision, nucmod='G', max_memory=args.max_memory)
idop_all = [i for i in range(cell.natm) if cell.atom_pure_symbol(i) == args.dopant]
ndop = len(idop_all)
idop = [idop_all[args.active]]
log('%s atoms %s; active %d' % (args.dopant, idop_all, idop[0]))
log('%s: %d atoms, nao %d, n2c %d, %d electrons' % (args.system, cell.natm, cell.nao, cell.nao_2c(), cell.nelectron))

def fold_guess():
    '''SCF density of a smaller cell of the same crystal (args.guess_from) in this cell's AO basis:
    D[a_mu, b_nu] = D_small[a'_mu, b'_nu] with a', b' the atoms of the small cell equivalent to a, b
    (fractional positions modulo the small lattice).  This is the same Gamma-point density of the
    small cell written in the larger cell; it has the electron count of this cell.  Where an atom
    carries a different basis here than its small-cell equivalent (e.g. the copy of a relabelled
    active atom), its rows/columns are projected with the isolated-atom overlaps.'''
    sysg = from_cif_relaxed(args.system, args.xyz) if args.xyz else from_cif(args.system)
    if args.guess_supercell:
        mg = [int(x) for x in args.guess_supercell.split(',')]
        sysg = supercell(sysg, mg if len(mg) == 3 else numpy.reshape(mg, (3, 3)))
    if args.guess_active_label:
        k_, at_ = 0, []
        for e_, x_ in sysg[1]:
            if e_ == args.dopant:
                if k_ == args.active:
                    e_ = args.guess_active_label
                k_ += 1
            at_.append((e_, x_))
        sysg = (sysg[0], at_)
    cg = make_cell(sysg, args.precision, nucmod='G', basis=basis) if basis else make_cell(sysg, args.precision, nucmod='G')
    zg = numpy.load(os.path.join(args.guess_from, 'scf_%s.npz' % (args.xc or 'hf')))
    dg = (zg['mo_coeff'] * zg['mo_occ']) @ zg['mo_coeff'].conj().T
    ag = numpy.asarray(cg.lattice_vectors())
    xg, x = cg.atom_coords(), cell.atom_coords()
    amap = []
    for ia in range(cell.natm):
        f = numpy.linalg.solve(ag.T, (x[ia][:, None] - xg.T)).T          # fractional differences to all small-cell atoms
        f -= numpy.round(f)
        dd = numpy.linalg.norm(f @ ag, axis=1)
        jb = int(numpy.argmin(dd))
        assert dd[jb] < 1e-4 and cg.atom_pure_symbol(jb) == cell.atom_pure_symbol(ia), (ia, dd[jb])
        amap.append(jb)
    sl, slg = cell.aoslice_2c_by_atom(), cg.aoslice_2c_by_atom()
    blocks = []                                     # per atom: (indices in this cell, projector from the small-cell block)
    for ia, jb in enumerate(amap):
        la, lb = cell.atom_symbol(ia), cg.atom_symbol(jb)
        if cell._basis[la] == cg._basis[lb]:
            pa = numpy.eye(slg[jb][3] - slg[jb][2])
        else:
            ma = gto.M(atom=[(la, (0, 0, 0))], basis={la: cell._basis[la]}, unit='B', spin=None, verbose=0)
            mb = gto.M(atom=[(lb, (0, 0, 0))], basis={lb: cg._basis[lb]}, unit='B', spin=None, verbose=0)
            pa = numpy.linalg.solve(ma.intor('int1e_ovlp_spinor'), gto.intor_cross('int1e_ovlp_spinor', ma, mb))
        blocks.append((numpy.arange(sl[ia][2], sl[ia][3]), numpy.arange(slg[jb][2], slg[jb][3]), pa))
    dm = numpy.zeros((cell.nao_2c(),) * 2, dtype=complex)
    for ia, (pi, gi, pa) in enumerate(blocks):
        for ib, (pj, gj, pb) in enumerate(blocks):
            dm[numpy.ix_(pi, pj)] = pa @ dg[numpy.ix_(gi, gj)] @ pb.conj().T
    return dm


# 1. RSDF (built once); with --jk thc only the object (CASCI three-index route, nothing stored) and THCJK
t = time.time()
if args.jk == 'thc':
    if args.lr != 'gspace' or args.active_eri != 'three-index' or args.xc:
        raise SystemExit('--jk thc needs --lr gspace and --active-eri three-index (HF only)')
    from socutils.pbc.thc import THCJK
    from socutils.pbc import split_direct
    if basis:
        aux = {}
        for lab in sorted(set(cell.atom_symbol(i) for i in range(cell.natm))):
            el = gto.mole._std_symbol(lab)
            aux[lab] = auxgen_from_basis({el: basis[lab]}, contract=False, prune_lmax=False, threshold=1e-8)[el]
    else:
        aux = auxgen(cell, contract=False, prune_lmax=False, threshold=1e-8)
    r = RSDF(cell, aux, omega=args.omega, precision=args.precision,
             nuc_zeta=spinor_scf.nuclear_zeta(cell), lr=args.lr)
    r.verbose = 4 if root else 0
    r.stdout = sys.stdout if root else devnull
    if args.thc_max_memory:
        cell.max_memory = args.thc_max_memory
    jkobj = THCJK(cell, aux, alpha_c=args.thc_alpha_c, nmu_factor=args.thc_nmu, omega=args.omega,
                  nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=args.thc_local_rcut or None)
    jkobj.stream = True                       # streamed V/A: memory bounded by the budget
    jkobj.verbose = root
    jkobj.comm = comm
    jkobj.build()
    cell.max_memory = args.max_memory
    log('THCJK built in %.0f s on %d rank(s): N_mu %d (smooth %d), %s'
        % (time.time() - t, comm.Get_size() if comm is not None else 1, len(jkobj.th.pts), len(jkobj.th.smooth),
           ', '.join('%s %.0f s' % kv for kv in jkobj.th.timings.items())))
elif os.path.exists(f_cderi):
    r = RSDF.load(cell, f_cderi)
    r.max_memory = args.max_memory
    log('RSDF loaded from %s (naux kept %d)' % (f_cderi, r.naux_kept))
else:
    if basis:
        # one auxgen per basis label (a relabelled atom has its own orbital basis)
        aux = {}
        for lab in sorted(set(cell.atom_symbol(i) for i in range(cell.natm))):
            el = gto.mole._std_symbol(lab)
            aux[lab] = auxgen_from_basis({el: basis[lab]}, contract=False, prune_lmax=False, threshold=1e-8)[el]
    else:
        aux = auxgen(cell, contract=False, prune_lmax=False, threshold=1e-8)
    r = RSDF(cell, aux, omega=args.omega, precision=args.precision,
             nuc_zeta=spinor_scf.nuclear_zeta(cell), lr=args.lr)
    r.lindep_scaled = args.lindep_scaled
    if args.cderi_in_memory:
        r.cderi_in_memory = True
    else:
        r.cderi_file = f_cderi
    r.verbose = 4
    r.stdout = sys.stdout
    r.build()
    log('RSDF built in %.0f s, naux kept %d, metric eigenvalues [%.1e, %.1e]'
        % (time.time() - t, r.naux_kept, *r.metric_eig))

# 2. KRHF + one-centre X2CAMF, dopant 4f shell fractionally occupied
if args.jk == 'thc':
    mf = split_direct.periodic(spinor_hf.KRHF(cell), jkobj)
elif args.xc:
    mf = spinor_ks.periodic_ks(spinor_hf.KRHF(cell), r, xc=args.xc)
    mf.grids_level = args.grids_level
else:
    mf = spinor_scf.periodic(spinor_hf.KRHF(cell), r)
mf = mf.x2camf(with_gaunt=not args.no_gaunt, with_breit=not args.no_gaunt)
mf.max_cycle = 100
mf.conv_tol = args.scf_conv_tol
mf.verbose = 4 if root else 0
mf.stdout = sys.stdout if root else devnull
idx = addons.spinor_ao_indices(cell, idop[0], l=args.open_l)
idx_all = addons.spinor_ao_indices(cell, idop_all, l=args.open_l)
if args.open_ref:
    ref1 = numpy.load(args.open_ref)['ref']
    sl = cell.aoslice_2c_by_atom()
    ref = numpy.zeros((cell.nao_2c(), ref1.shape[1] * ndop), dtype=complex)
    ref_lab = args.active_label or args.dopant      # the reference spinors are in the basis of this label
    for k, ia in enumerate(idop_all):
        r_k = ref1
        if sl[ia][3] - sl[ia][2] != ref1.shape[0]:
            # another basis on this site: least-squares projection, re-orthonormalised (isolated-atom overlaps)
            ma = gto.M(atom=[(cell.atom_symbol(ia), (0, 0, 0))], basis={cell.atom_symbol(ia): cell._basis[cell.atom_symbol(ia)]},
                       unit='B', spin=None, verbose=0)
            mr = gto.M(atom=[(ref_lab, (0, 0, 0))], basis={ref_lab: cell._basis[ref_lab]}, unit='B', spin=None, verbose=0)
            assert mr.nao_2c() == ref1.shape[0]
            sa = ma.intor('int1e_ovlp_spinor')
            r_k = numpy.linalg.solve(sa, gto.intor_cross('int1e_ovlp_spinor', ma, mr) @ ref1)
            w_, v_ = numpy.linalg.eigh(r_k.conj().T @ sa @ r_k)
            log('reference 4f projected onto the basis of atom %d (%s): retained norms %.4f..%.4f'
                % (ia, cell.atom_symbol(ia), w_.min(), w_.max()))
            r_k = r_k @ v_ @ numpy.diag(w_ ** -.5) @ v_.conj().T
        ref[sl[ia][2]:sl[ia][3], k * ref1.shape[1]:(k + 1) * ref1.shape[1]] = r_k
    addons.reference_frac_occ(mf, ref, args.norb_open * ndop, args.nelec_open * ndop)
else:
    addons.projected_frac_occ(mf, idx_all, args.norb_open * ndop, args.nelec_open * ndop)
dm0 = None
if os.path.exists(f_scf):
    z = numpy.load(f_scf)
    dm0 = lib.tag_array((z['mo_coeff'] * z['mo_occ']) @ z['mo_coeff'].conj().T,
                        mo_coeff=z['mo_coeff'], mo_occ=z['mo_occ'])
    log('SCF restarted from %s' % f_scf)
if dm0 is None and args.guess_from:
    dm0 = fold_guess()
    log('SCF guess folded from %s (tr DS = %.6f, %d electrons)'
        % (args.guess_from, numpy.einsum('ij,ji', dm0, mf.get_ovlp()).real, cell.nelectron))


def _checkpoint(envs):
    if not root:
        return
    numpy.savez(f_scf, mo_coeff=envs['mo_coeff'], mo_occ=envs['mo_occ'], mo_energy=envs['mo_energy'],
                e_tot=envs['e_tot'], converged=False)


mf.callback = _checkpoint
t = time.time()
e_scf = mf.kernel(dm0)
if comm is not None:                          # the same orbitals on every rank (rank 0's)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = comm.bcast((mf.mo_coeff, mf.mo_occ, mf.mo_energy) if root else None, root=0)
if root:
    numpy.savez(f_scf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy, e_tot=e_scf)
w = addons.subspace_weights(mf.get_ovlp(), mf.mo_coeff, idx_all)
log('SCF converged=%s E = %.10f (%.0f s, %d cycles); open-shell 4f weights %.3f..%.3f'
    % (mf.converged, e_scf, time.time() - t, mf.cycles, w[mf.open_shell_mo].min(), w[mf.open_shell_mo].max()))

# 3. CAHF: average-of-configuration (ensemble of all 4f^n determinants) orbital
# optimisation with the true Hamiltonian, from the fractional-occupation
# orbitals (or a previous CAHF run)
mo, ncore = addons.cas_order(mf)
mf_scf, dm_frozen = mf, 0
if ndop > 1:
    # localise the open space on the active and the other dopant atoms; the
    # others are frozen (FrozenFracSCF) and drop out of the orbital space
    nact = args.norb_open
    order = [idop[0]] + [i for i in idop_all if i != idop[0]]
    blocks, wts = frozen_frac.localize_sites(mf.get_ovlp(), mo[:, ncore:ncore + nact * ndop],
                                             [addons.spinor_ao_indices(cell, i, l=args.open_l) for i in order], nact)
    for i, w in zip(order, wts):
        if w is not None:
            log('open shell localised on atom %d: weights %s | rest %.2e..%.2e'
                % (i, numpy.array2string(w[:nact:2], precision=4), w[nact:].min(), w[nact:].max()))
    mo_a = blocks[0]
    mo = numpy.hstack([mo[:, :ncore], mo_a, mo[:, ncore + nact * ndop:]])
    if ndop == 2:
        mf = frozen_frac.FrozenFracSCF(mf_scf, blocks[1], args.nelec_open)
    else:
        mf = frozen_frac.FrozenFracSCF(mf_scf, blocks[1:], args.nelec_open * (ndop - 1))
    dm_frozen = mf.frac * mf.mo_frozen @ mf.mo_frozen.conj().T
cahf_ran = False
if not args.no_cahf:
    if os.path.exists(f_cahf):
        z = numpy.load(f_cahf)
        mo = z['mo_coeff']
        log('CAHF orbitals loaded from %s (E = %.10f)' % (f_cahf, float(z['e_tot'])))
    elif args.cahf_driver == 'fock':
        cahf_ran = True
        mc = cahf_fock.CAHF(mf, args.norb_open, args.nelec_open, ncore=ncore)
        mc.max_cycle = args.cahf_max_cycle
        mc.conv_tol = args.cahf_conv_tol
        mc.conv_tol_grad = args.cahf_conv_grad
        mc.verbose = 4 if root else 0
        mc.chkfile = f_cahf.replace('.npz', '_partial.npz') if root else None
        if mc.chkfile and os.path.exists(mc.chkfile):
            zp = numpy.load(mc.chkfile)
            if zp['mo_coeff'].shape == mo.shape and comm is None:
                mo = zp['mo_coeff']
                log('CAHF restarted from %s (cycle %d, |g| %.2e)' % (mc.chkfile, int(zp['cycle']), float(zp['gnorm'])))
        t = time.time()
        with open(os.path.join(args.workdir, 'cahf.log') if root else os.devnull, 'a') as fcahf:
            mc.stdout = fcahf
            mc.kernel(mo)
    else:
        cahf_ran = True
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
    if cahf_ran:
        mo = mc.mo_coeff
        if comm is not None:                  # one set of orbitals on every rank
            mo = comm.bcast(mo if root else None, root=0)
        if root:
            numpy.savez(f_cahf, mo_coeff=mo, e_tot=mc.e_tot, ncore=ncore)
        w = addons.subspace_weights(mf.get_ovlp(), mo[:, ncore:ncore + args.norb_open], idx)
        log('CAHF (%s) E = %.10f converged %s (%.0f s); active 4f weights %.3f..%.3f'
            % (args.cahf_driver, mc.e_tot, getattr(mc, 'converged', None), time.time() - t, w.min(), w.max()))

# 4. CASCI on the 4f spinors
mc = zcasci.CASCI(mf, args.norb_open, args.nelec_open, ncore=ncore)
if args.active_eri == 'three-index':
    from socutils.mcscf import eri_active

    def _h2eff(mo_coeff=None):
        mo_coeff = mc.mo_coeff if mo_coeff is None else mo_coeff
        t_ = time.time()
        e_ = r.active_eri(mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]) if root else None
        if comm is not None:                  # rank 0 only (the three-index route is not split)
            e_ = comm.bcast(e_, root=0)
        log('active (tu|vw) from the three-index route: %.0f s' % (time.time() - t_))
        return e_
    mc.get_h2eff = _h2eff
mc.fcisolver.nroots = args.nroots
mc.verbose = 4 if root else 0
mc.stdout = sys.stdout if root else devnull
t = time.time()
mc.kernel(mo)
e_all = numpy.atleast_1d(mc.e_tot)
e = numpy.sort(e_all)
ci0 = mc.ci[int(numpy.argmin(e_all))] if isinstance(mc.ci, (list, tuple)) else mc.ci
casdm1 = mc.fcisolver.make_rdm1(ci0, args.norb_open, args.nelec_open)   # <p^+ q>, ground state
if root:
    numpy.savez(f_cas, e_tot=e_all, mo_coeff=mo, ncore=ncore, casdm1_0=casdm1,
                ci=numpy.asarray(mc.ci if isinstance(mc.ci, (list, tuple)) else [mc.ci]))
log('CASCI(%de,%d) %.0f s; E0 = %.10f; levels (cm-1): %s'
    % (args.nelec_open, args.norb_open, time.time() - t, e[0],
       numpy.array2string((e - e[0]) * 219474.63, precision=1, max_line_width=200)))

# 5. W_NSM of the CASCI ground state: the dopant-atom X2C Schiff integrals (CFOUR
# SOSFMO2C) in the dopant's diagonal AO block, traced with the full (core +
# active) one-particle density
if args.nsm_dir and root:
    from socutils.pbc import atomic_prop
    molatom, nsm_atom = atomic_prop.load_cfour_atom(args.nsm_dir, args.dopant)
    nsm = atomic_prop.atomic_block_operator(cell, idop[0], molatom, nsm_atom)
    dm = mc.make_rdm1(mo_coeff=mo, ci=ci0) + dm_frozen
    w_core = atomic_prop.expectation(nsm, mo[:, :ncore] @ mo[:, :ncore].conj().T + dm_frozen)
    w = atomic_prop.expectation(nsm, dm)
    numpy.savez(os.path.join(args.workdir, 'wnsm_%s.npz' % (args.xc or 'hf')), w=w, w_core=w_core, dm=dm)
    log('W_NSM (root 0, %s): %s  |W| %.4e   (core %s, active %s)'
        % (args.nsm_dir, numpy.array2string(w, precision=5), numpy.linalg.norm(w),
           numpy.array2string(w_core, precision=5), numpy.array2string(w - w_core, precision=5)))

# 6. 7F0 -> 5D0 transition dipole d00 (root 49: the first state above the 49
# 7F_J states).  r is not defined on Gamma-point AOs; the 4f-4f transition
# density is local on the dopant, so the dipole is taken about its nucleus over
# the AO images near it (atomic_prop.local_dipole; rc = 0: dopant block only)
if len(e_all) > 50 and root:
    from socutils.pbc import atomic_prop
    k = 49
    rel = (e_all - e_all[0]) * 219474.63
    g = mc.fcisolver.trans_rdm1(mc.ci[0], mc.ci[k], args.norb_open, args.nelec_open)
    cas = mo[:, ncore:ncore + args.norb_open]
    tdm = cas @ g.T @ cas.conj().T
    log('d00: root %d at %.1f cm-1 (root 48 %.1f, root 50 %.1f)' % (k, rel[k], rel[k - 1], rel[k + 1]))
    res = atomic_prop.local_dipole(cell, idop[0], tdm, rcs=[x / 0.52917721092 for x in (0., 2.6, 2.9, 3.5)])
    out = []
    for rc, d, q in res:
        dr = atomic_prop.real_axis(d)
        out.append(dr)
        log('   rc %.1f A: d00 %s  |d| %.4e  (|q_trans| %.1e)' % (rc * 0.52917721092, numpy.array2string(dr, precision=6),
                                                             numpy.linalg.norm(dr), abs(q)))
    numpy.savez(os.path.join(args.workdir, 'd00_%s.npz' % (args.xc or 'hf')), root=k, gamma=g, d=numpy.array(out),
                rc_A=[0., 2.6, 2.9, 3.5], e_rel=rel)
