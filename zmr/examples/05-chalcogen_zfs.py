#!/usr/bin/env python
'''
Zero-field splittings of the X 3Sigma- ground states of the chalcogen
diatomics O2, SO, S2, SeO, SeS, Se2, TeO, TeS, TeSe and Te2: state-averaged
spinor CASSCF(8e, 12 spinors) with an X2C Hamiltonian (X2CMP by default) and
multi-state PC-NEVPT2 (union contraction) on top.

Molecules, bond lengths, basis sets (uncontracted aug-cc-pVTZ for O and S,
Dyall's aVTZ for Se and Te), active space and Cholesky threshold (1e-7) are
those of X. Wang, S. Wang, Y. Wu and L. Cheng, arXiv:2602.24236, Table 1.
The orbital optimization starts from the Kramers-restricted SCF of the
closed-shell dication (see 04-o2_zfs_active_spaces.py).

zmr uses the Cholesky vectors of the CASSCF, so that both steps work with the
same Hamiltonian.  Correlated space: the core spinors are the eigenvectors of
the core-core block of the generalized Fock matrix (state-averaged density);
the lowest ones are frozen, either all shells below the valence ns np
("valence") or all shells below the outermost (n-1)d of Se and Te ("d").
Virtual spinors above --emax hartree (default 100) are removed.

Usage:  python 05-chalcogen_zfs.py <molecule> [options]
    --model x2cmp|x2camf     X2C model potential or atomic mean field
    --ham dcb|dcg|dc         two-electron operator behind the model
    --nsa 3,6                state averages to run (3: X 0+, X +-1;
                             6: plus a 1Delta and b 1Sigma+)
    --core valence,d         frozen-core choices
    --emax 100               virtual cutoffs in hartree (0: keep all)
    --orbitals               print the quasi-canonical core and virtual shells
                             (energies, gaps) and stop: the table to choose from
    --nfrozen N, --nkeep N   frozen core / kept virtual spinors chosen by hand
    --uc                     also UC-NEVPT2 for the three-state model space
    --mrci ci,aqcc           internally contracted MRCI (+Q), ACPF, AQCC for the
                             first model space

Every stage is stored in the working directory (cderi.chk, hcore_*.npy,
scf.chk, mo_sa*.npy) and reused by a later run.  Lines of results start with
ZFS; one RESULT line per calculation is meant for collection.
'''
import argparse
from types import SimpleNamespace
import json
import os
import resource
import sys
import time

import numpy
from pyscf import gto, lib
from pyscf.x2c.x2c import _decontract_spinor
from socutils.mcscf import zmc_ao2mo, zmc_superci, zmcscf
from socutils.scf import spinor_hf
from socutils.somf import x2cmp
from socutils.tools import fcidump_rel

from socutils.zmr import fock
from socutils.zmr.active import symmetry
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import socutils_x2c
from socutils.zmr.mrci import MRCI
from socutils.zmr.mrci.caspt2 import CASPT2
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc.nevpt2 import UCNEVPT2

HARTREE2WAVENUMBER = 219474.6313632

# heavier atom first; experimental bond lengths in Angstrom
MOLECULES = {'O2': ('O', 'O', 1.20752), 'SO': ('S', 'O', 1.481), 'S2': ('S', 'S', 1.889),
             'SeO': ('Se', 'O', 1.648), 'SeS': ('Se', 'S', 2.037), 'Se2': ('Se', 'Se', 2.166),
             'TeO': ('Te', 'O', 1.825), 'TeS': ('Te', 'S', 2.230), 'TeSe': ('Te', 'Se', 2.372),
             'Te2': ('Te', 'Te', 2.557)}
# atoms of the p block: electrons in np, charge of the closed-shell ion of the
# SCF (ns^2 for np^1 - np^3, np^6 for np^4 and np^5) and the state averages
# (= model spaces).  All states of np^n: 6, 15, 20, 15, 6; np^4 also 3P alone
# (9).  For np^1 and np^5 the average over all six states is the
# configuration-averaged Hartree-Fock and its CASCI.
PBLOCK = {}
for _n, (_charge, _models, _elements) in enumerate((
        (1, (6,), 'B Al Ga In Tl'), (2, (15,), 'C Si Ge Sn Pb'), (3, (20,), 'N P As Sb Bi'),
        (-2, (9, 15), 'O S Se Te'), (-1, (6,), 'F Cl Br I')), start=1):
    for _el in _elements.split():
        PBLOCK[_el] = (_n, _charge, _models)
        MOLECULES[_el] = (_el, None, 0)
BASIS = {'O': 'uncaugccpvtz', 'S': 'uncaugccpvtz', 'F': 'uncaugccpvtz'}
# frozen spinors per atom, by period: all shells below ns np ('valence'), below
# (n-1)d ns np ('d'), or below the whole shell under the valence one ('shell':
# O 1s, S 2s2p, Se 3s3p3d, Te 4s4p4d correlated)
_PERIODS = ('B C N O F', 'Al Si P S Cl', 'Ga Ge As Se Br', 'In Sn Sb Te I', 'Tl Pb Bi')
FROZEN = {key: {el: nf for els, nf in zip(_PERIODS, counts) for el in els.split()}
          for key, counts in (('valence', (2, 10, 28, 46, 78)), ('d', (2, 10, 18, 36, 68)),
                              ('shell', (0, 2, 10, 28, 60)))}
HAMILTONIANS = {'dcb': (True, True), 'dcg': (True, False), 'dc': (False, False)}

parser = argparse.ArgumentParser()
parser.add_argument('molecule', choices=sorted(MOLECULES))
parser.add_argument('--model', default='x2cmp', choices=['x2cmp', 'x2camf'])
parser.add_argument('--ham', default='dcb', choices=sorted(HAMILTONIANS))
parser.add_argument('--nsa', default=None)
parser.add_argument('--core', default='valence,d')
parser.add_argument('--emax', default='100')
parser.add_argument('--mrci', default='',
                    help='internally contracted MRCI for the first model space: ci (with '
                         'Davidson corrections), acpf, aqcc, comma separated')
parser.add_argument('--mrci-plain', action='store_true',
                    help='with --corr and --mrci: also the MRCI without the correction')
parser.add_argument('--mrci-tol', type=float, default=5e-5,
                    help='residual norm at which the MRCI roots are converged')
parser.add_argument('--mrci-wmax', type=float, default=0.0,
                    help='largest dressed integral kept in the MRCI sigma, in elements '
                         '(Sigma.w_max; memory against time)')
parser.add_argument('--mrci-maxcycle', type=int, default=0,
                    help='MRCI: stop the Davidson iterations after this many (timing tests)')
parser.add_argument('--mrci-memory', action='store_true',
                    help='MRCI: report the memory of the setup and of one sigma, and stop')
parser.add_argument('--orbitals', action='store_true',
                    help='print the quasi-canonical core and virtual shells and stop')
parser.add_argument('--nfrozen', type=int, default=None,
                    help='number of frozen core spinors (overrides --core)')
parser.add_argument('--nkeep', type=int, default=None,
                    help='number of virtual spinors kept (overrides --emax)')
parser.add_argument('--uc', action='store_true')
parser.add_argument('--bond', type=float, default=None,
                    help='bond length in Angstrom instead of the experimental one (run in a '
                         'directory of its own)')
parser.add_argument('--dump-light', action='store_true',
                    help='with --dump: only the spinors, their labels and the frozen spinors')
parser.add_argument('--mrci-contraction', default='fic', choices=['fic', 'wk', 'none'],
                    help='contraction of the MRCI: fully internally contracted, only the doubly '
                         'external blocks, or none (uncontracted MRCISD)')
parser.add_argument('--natvirt', default='',
                    help='numbers of natural virtual spinors (of the MRCI of the full space) to '
                         'which the virtual space is truncated, e.g. 30,38,46: every method is '
                         'repeated in each truncated space (dump: %%(nv)d in the file name)')
parser.add_argument('--dump-sf', action='store_true',
                    help='with --dump: also <file>_sf.npz, the same spinors with a spin-free '
                         'Hamiltonian (sfx2c1e one-electron operator)')
parser.add_argument('--dump', default=None,
                    help='write the dense Hamiltonian of the correlated space to this .npz file '
                         '(%%(emax)g, %%(core)s, %%(nsa)d are filled in)')
parser.add_argument('--caspt2', default='',
                    help='CASPT2 (zmr.mrci.caspt2): fic (internally contracted), none '
                         '(uncontracted) or both, comma separated')
parser.add_argument('--models', default=None,
                    help='numbers of model states (lowest CASCI states) instead of the default, '
                         'e.g. 9 for the ground term 3P of np^2 or np^4')
parser.add_argument('--diagonal', action='store_true',
                    help='also the levels from the diagonal of the effective Hamiltonians: '
                         'the single-state second-order energies of the CASCI states')
parser.add_argument('--mcscf-grad', type=float, default=1e-6,
                    help='gradient threshold of the CASSCF; near-linear dependent sets (dyall '
                         'v4z of the 5p atoms) stall at about 2e-5 with a stationary energy')
parser.add_argument('--fock-h0', action='store_true',
                    help='also second-order perturbation theory with the Fock operator as '
                         'zeroth-order Hamiltonian of the active space (h0 = "fock")')
parser.add_argument('--corr', action='store_true',
                    help='X2Ccorr: four-component two-electron integrals in the active space')
parser.add_argument('--basis', default=None,
                    help='basis of all atoms instead of the default one, e.g. uncaugccpvqz '
                         '(run in a directory of its own: the work files do not carry the basis)')
parser.add_argument('--mcscf', default='superci', choices=['superci', 'sosci', 'cahf'],
                    help='orbitals: SA-CASSCF with the super-CI of socutils or with the SO-SCI '
                         '+ L-BFGS of its zmcscf-qn branch, or the configuration-averaged '
                         'Hartree-Fock of the active shell (cahf_diis of socutils)')
parser.add_argument('--sp-from', default=None,
                    help='take the s and p primitives from this basis (e.g. aug-cc-pv7z), the '
                         'higher angular momenta from --basis')
parser.add_argument('--basis-file', default=None,
                    help='Python file with one variable per element (PySCF internal format, '
                         'kappa allowed), e.g. a j-adapted contracted set; elements not in '
                         'the file keep --basis')
parser.add_argument('--tight', default='',
                    help='tight primitives added to the (uncontracted) basis, e.g. p:4,d:2: for '
                         'each l the exponents continue the series upwards with the ratio of '
                         'its two largest exponents; p:1223.0/182.9 adds these exponents')
parser.add_argument('--vvvv-memory', type=float, default=0.0,
                    help='bytes up to which the MRCI keeps (vv|vv) as one matrix '
                         '(Sigma.vvvv_dense_memory)')
parser.add_argument('--cas', default=None, help='active electrons,spinors (default 8,12; atoms 4,6)')
parser.add_argument('--scf-charge', type=int, default=None,
                    help='charge of the closed-shell species of the SCF')
parser.add_argument('--max-memory', type=int, default=12000)
args = parser.parse_args()
heavy, light, bond = MOLECULES[args.molecule]
if args.bond is not None:
    bond = args.bond
ATOM = light is None
# active electrons and spinors; closed-shell ion of the SCF; state averages
# (= model spaces): molecules X 3Sigma- (3) and all of pi*^2 (6), atoms 3P (9)
# and all of np^4 (15)
NELEC, NACT, SCF_CHARGE, MODELS = (8, 12, 2, (3, 6))
if ATOM:
    NELEC, SCF_CHARGE, MODELS = PBLOCK[heavy]
    NACT = 6
if args.models:
    MODELS = tuple(int(x) for x in args.models.split(','))
NROOTS = MODELS[-1]
if args.cas:
    NELEC, NACT = (int(x) for x in args.cas.split(','))
if args.scf_charge is not None:
    SCF_CHARGE = args.scf_charge
if args.nsa is None:
    args.nsa = ','.join(str(n) for n in MODELS)
with_gaunt, with_breit = HAMILTONIANS[args.ham]
tag = '%s(%s)' % (args.model.upper(), args.ham.upper())
T0 = time.time()


def stamp(text):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print('ZFS [%7.0f s, peak %5.1f GB] %s' % (time.time() - T0, rss, text), flush=True)


# ----------------------------------------------------------------------
# orbitals: Cholesky vectors, X2C one-electron operator, SCF, SA-CASSCF

geometry = '%s 0 0 0' % heavy if ATOM else '%s 0 0 0; %s 0 0 %s' % (heavy, light, bond)
def basis_of(element):
    if args.basis_file:
        import runpy
        sets = runpy.run_path(args.basis_file)
        if element in sets:
            print('ZFS %s: basis from %s' % (element, args.basis_file))
            return sets[element]
    name = args.basis or BASIS.get(element, 'dyallav3z')
    if not args.tight and not args.sp_from and not (args.basis and name.startswith('unc')):
        return name
    # (PySCF does not resolve the 'unc' prefix for every basis, e.g. aug-cc-pV6Z)
    shells = gto.uncontract(gto.basis.load(name.replace('unc', '', 1).lstrip('-'), element))
    if args.sp_from:
        sp = gto.uncontract(gto.basis.load(args.sp_from, element))
        shells = [sh for sh in sp if sh[0] < 2] + [sh for sh in shells if sh[0] >= 2]
        print('ZFS %s: s and p primitives of %s (%d s, %d p), higher l of %s' % (
            element, args.sp_from, sum(sh[0] == 0 for sh in shells),
            sum(sh[0] == 1 for sh in shells), name))
    if not args.tight:
        return shells
    for item in args.tight.split(','):
        l, n = 'spdfgh'.index(item.split(':')[0]), item.split(':')[1]
        if not n.isdigit():
            # explicit exponents, e.g. from an optimisation on the atom
            shells += [[l, [float(x), 1.0]] for x in n.split('/')]
            print('ZFS %s: tight %s primitives added with the exponents %s'
                  % (element, 'spdfgh'[l], n.replace('/', ', ')))
            continue
        n = int(n)
        exps = sorted((sh[1][0] for sh in shells if sh[0] == l), reverse=True)
        ratio = exps[0] / exps[1]
        shells += [[l, [exps[0] * ratio**(k + 1), 1.0]] for k in range(n)]
        print('ZFS %s: %d tight %s primitives added, ratio %.3f, largest exponent %.1f -> %.1f'
              % (element, n, 'spdfgh'[l], ratio, exps[0], exps[0] * ratio**n))
    # shells of one angular momentum together, exponents descending (the order of the
    # basis-set files; the tight ones were appended after the highest l)
    shells.sort(key=lambda sh: (sh[0], -sh[1][0]))
    return shells


mol = gto.M(atom=geometry,
            basis={el: basis_of(el) for el in (heavy, light) if el},
            charge=SCF_CHARGE, verbose=4,
            max_memory=args.max_memory, nucmod='G')
nao = mol.nao_nr()
if os.path.exists('cderi.chk'):
    cderi = lib.chkfile.load('cderi.chk', 'j3c')
else:
    cderi = zmc_ao2mo.chunked_cholesky_threeloop(mol, max_error=1e-7)
    cderi = lib.pack_tril(cderi.reshape(-1, nao, nao))
    lib.chkfile.save('cderi.chk', 'j3c', cderi)
stamp('%s: %d spherical AOs, %d spinors, %d Cholesky vectors (1e-7)'
      % (args.molecule, nao, mol.nao_2c(), cderi.shape[0]))

mf = spinor_hf.KRHF(mol).density_fit()
mf.with_x2c = x2cmp.X2CMP(mol, x2cmp=args.model, with_gaunt=with_gaunt, with_breit=with_breit)
hcore_file = 'hcore_%s_%s.npy' % (args.model, args.ham)
if os.path.exists(hcore_file) and not args.corr:
    mf.with_x2c.hcore = numpy.load(hcore_file)
else:
    numpy.save(hcore_file, mf.with_x2c.get_hcore(mol))
stamp('%s one-electron operator' % tag)
mf.with_df._cderi = 'cderi.chk'
mf.chkfile = 'scf_%s_%s_q%d.chk' % (args.model, args.ham, SCF_CHARGE)
if os.path.exists(mf.chkfile):
    mf.init_guess = 'chkfile'
mf.conv_tol = 1e-10
mf.kernel()
if not mf.converged:
    # oscillating SCF (SeO^2+, SeS^2+): level shift, then a few plain cycles
    mf.level_shift = 0.5
    mf.max_cycle = 200
    mf.kernel(dm0=mf.make_rdm1())
    mf.level_shift = 0
    mf.kernel(dm0=mf.make_rdm1())
if not mf.converged:
    # near-linear dependent basis (dyall v4z of the 5p atoms: overlap
    # eigenvectors are removed): energy and density are stationary but the
    # gradient norm in the full basis stays at about 1e-4.  Accepted if a run
    # with a loose gradient criterion ends at once at the same energy.
    e_stalled = mf.e_tot
    mf.conv_tol_grad = 1e-3
    mf.max_cycle = 20
    mf.kernel(dm0=mf.make_rdm1())
    if mf.converged and abs(mf.e_tot - e_stalled) < 1e-8:
        print('ZFS SCF accepted on the energy (change %.1e Eh); the gradient norm does '
              'not reach its threshold; %d of %d spinors kept'
              % (mf.e_tot - e_stalled, mf.mo_coeff.shape[1], mol.nao_2c()), flush=True)
    else:
        mf.converged = False
if not mf.converged:
    raise RuntimeError('SCF of the closed-shell ion did not converge')
stamp('KRHF of the closed-shell ion (charge %+d): E = %.10f' % (SCF_CHARGE, mf.e_tot))

# neutral molecule from here on
mf.mol.charge = 0
mf.mol.spin = int(mol.atom_charges().sum()) % 2
mf.mol.build()
ncore_total = int(mol.atom_charges().sum()) - NELEC


def casscf(nsa, mo0):
    '''SA-CASSCF orbitals and energy; stored and reused.'''
    if args.mcscf == 'cahf':
        # configuration-averaged Hartree-Fock of the active shell (socutils);
        # the same orbitals for every model space
        from socutils.mcscf import cahf_diis, zcahf
        stem = 'mo_cahf%d_%d_%s_%s' % (NELEC, NACT, args.model, args.ham)
        if os.path.exists(stem + '.npy') and os.path.exists(stem + '.json'):
            with open(stem + '.json') as f:
                return numpy.load(stem + '.npy'), json.load(f)['e_sa']
        mc = zmcscf.CASSCF(mf, NACT, NELEC)
        mc.max_cycle_macro = 100
        mc.fcisolver = zcahf.CAHF(mol)
        mf.mo_coeff = mo0
        mf.mo_occ = numpy.zeros(mo0.shape[1])
        mf.mo_occ[:ncore_total] = 1
        mc, _, _ = cahf_diis.mc_diis(mc, mo0, diis_start=20, scaling_factor=0.8)
        mo, e_sa = mc.mo_coeff, float(numpy.real(mc.e_tot))
        print('ZFS CAHF converged: %s' % getattr(mc, 'converged', 'unknown'), flush=True)
        numpy.save(stem + '.npy', mo)
        with open(stem + '.json', 'w') as f:
            json.dump({'e_sa': e_sa}, f)
        return mo, e_sa
    stem = 'mo_cas%d_%d_sa%d_%s_%s' % (NELEC, NACT, nsa, args.model, args.ham)
    old = 'mo_sa%d_%s_%s' % (nsa, args.model, args.ham)
    if (NELEC, NACT) in ((8, 12), (4, 6)) and os.path.exists(old + '.json'):
        stem = old      # name used before the active space became an option
    if os.path.exists(stem + '.npy') and os.path.exists(stem + '.json'):
        with open(stem + '.json') as f:
            return numpy.load(stem + '.npy'), json.load(f)['e_sa']
    mc = zmcscf.CASSCF(mf, NACT, NELEC).state_average(numpy.ones(nsa) / nsa)
    mc.max_cycle_macro = 100
    # the splitting is not variational: an orbital gradient of 1e-4 (the
    # default threshold) leaves it uncertain in the third digit
    if args.mcscf == 'sosci':
        # frozen orbital basis and Kramers pairs (2k, 2k+1) with phase +1 are required
        from socutils.mcscf import zmc_sosci
        from socutils.tools import kramers
        mc.canonicalize_ = False
        mc.natorb = False
        ovlp = mol.intor('int1e_ovlp_spinor')
        tc = kramers.time_reversal(mol, mo0[:, ::2])
        ph = numpy.einsum('mk,mn,nk->k', mo0[:, 1::2].conj(), ovlp, tc)
        if abs(ph).min() < 1 - 1e-5:
            raise RuntimeError('the starting orbitals are not Kramers pairs (2k, 2k+1)')
        mo0 = mo0.copy()
        mo0[:, 1::2] = tc
        converged, e_sa = zmc_sosci.mcscf_sosci(mc, mo0, bfgs=True, conv_tol=1e-11,
                                                conv_tol_grad=args.mcscf_grad, davidson_maxiter=30,
                                                trust_radius=0.2, bfgs_start_grad=1.0)[:2]
        mo = mc.mo_coeff
    else:
        converged, e_sa, _, _, mo, _ = zmc_superci.mcscf_superci(mc, mo0, conv_tol=1e-11,
                                                                 conv_tol_grad=args.mcscf_grad)
    e_sa = float(numpy.mean(e_sa))
    if not converged:
        raise RuntimeError('SA-%d CASSCF did not converge in %d macro iterations (E = %.10f)'
                           % (nsa, mc.max_cycle_macro, e_sa))
    numpy.save(stem + '.npy', mo)
    with open(stem + '.json', 'w') as f:
        json.dump({'e_sa': e_sa}, f)
    return mo, e_sa


# ----------------------------------------------------------------------
# zmr: correlated space, model states, NEVPT2

FROZEN_MO = [None]      # frozen spinors of the last correlated_space (for dump_hamiltonian)


def correlated_space(mo, nsa, core, emax):
    '''Hamiltonian in the CASSCF orbitals mo with the frozen core named core
    and the virtual cutoff emax.  Core and virtual orbitals are the
    eigenvectors of the generalized Fock matrix of the SA-nsa density.'''
    full = socutils_x2c.from_scf(mf, ncore=ncore_total, nact=NACT, mo_coeff=mo, mol=mol,
                                 cderi=cderi)
    act = ActiveSpace.from_eris(full)
    e, c = act.solve(NELEC, nsa)
    gamma = act.averaged_rdm1(c, NELEC)
    if args.orbitals:
        fock.orbital_report(full, gamma)
        sys.exit(0)
    eps, u = fock.canonical_core(full, gamma)
    if args.nfrozen is not None:
        nfrozen = args.nfrozen          # the user's choice, taken as it is
    else:
        nfrozen = FROZEN[core][heavy] + (FROZEN[core][light] if light else 0)
        nfrozen = min(nfrozen, ncore_total)
        if 0 < nfrozen < ncore_total and eps[nfrozen] - eps[nfrozen - 1] < 0.05:
            raise RuntimeError('no gap between frozen and correlated core spinors')
    eps = numpy.append(eps, numpy.nan)
    mo = mo.copy()
    mo[:, :ncore_total] = mo[:, :ncore_total] @ u
    eris = socutils_x2c.from_scf(mf, ncore=ncore_total - nfrozen, nact=NACT, nfrozen=nfrozen,
                                 mo_coeff=mo, mol=mol, cderi=cderi)
    FROZEN_MO[0] = mo[:, :nfrozen].copy()
    nvirt = eris.spaces.nvirt
    cut = 'none'
    if args.nkeep is not None or emax > 0:
        eris, kept, dropped = fock.truncate_virtuals(eris, gamma, emax, nkeep=args.nkeep)
        if len(dropped):
            cut = '%.1f | %.1f Eh' % (kept[-1], dropped[0])
    print('ZFS correlated space (%s core, emax %g): %d frozen + %d core + %d active + %d of %d '
          'virtual spinors; frozen up to %.3f Eh, correlated core from %.3f Eh; virtual cut: %s'
          % (core, emax, nfrozen, eris.spaces.ncore, NACT, eris.spaces.nvirt, nvirt,
             eps[nfrozen - 1] if nfrozen else numpy.nan, eps[nfrozen], cut))
    return eris, e[:nsa].mean()


def corr_active_eri(mo_act):
    '''X2Ccorr: (tu|vw) of the active spinors from the four-component
    Coulomb and Gaunt / Breit integrals, with the orbitals transformed back
    to the four-component picture.  socutils does it
    (tools.fcidump_rel.x2c_active_eri_4c, the integrals of from_x2c with
    fourC_int and with_full_coulomb), also for a contracted basis.'''
    eri = fcidump_rel.x2c_active_eri_4c(mf, mo_act, with_full_coulomb=True,
                                        with_gaunt=with_gaunt, with_breit=with_breit)
    n = mo_act.shape[1]
    return numpy.asarray(eri).reshape(n, n, n, n)


def corr_core_hamiltonian(mo_core, mo_act):
    '''One-electron part to go with corr_active_eri: four-component bare
    Hamiltonian and four-component mean field of the core spinors only
    (fourC_fcore of socutils' from_x2c), in the active spinors, and the core
    energy.  The X2C one-electron operator contains the mean-field
    two-electron spin-orbit term of the atomic densities, active electrons
    included, which the explicit four-component active integrals would count
    a second time.'''
    from pyscf import scf
    nc = mo_core.shape[1]
    xmol, mo_l, mo_s = fcidump_rel.x2c_4c_spinors(mf, numpy.hstack((mo_core, mo_act)))
    c4c = numpy.vstack((mo_l, mo_s))
    mf4c = scf.DHF(xmol)
    mf4c.with_gaunt = with_gaunt
    mf4c.with_breit = with_breit
    hcore = mf4c.get_hcore()
    dm = c4c[:, :nc] @ c4c[:, :nc].conj().T
    vj, vk = mf4c.get_jk(xmol, dm)
    vhf = vj - vk
    e_core = (mol.energy_nuc() + numpy.einsum('ij,ji', dm, hcore)
              + 0.5 * numpy.einsum('ij,ji', dm, vhf)).real
    return c4c[:, nc:].conj().T @ (hcore + vhf) @ c4c[:, nc:], e_core


def grouped(e, tol=1e-7):
    '''Level energies relative to the lowest one, in cm-1, and degeneracies.'''
    out, start = [], 0
    for k in range(1, len(e) + 1):
        if k == len(e) or e[k] - e[k - 1] > tol:
            out.append(((e[start:k].mean() - e[0]) * HARTREE2WAVENUMBER, k - start))
            start = k
    return out


def dump_hamiltonian(eris, act, civecs, e_cas, path):
    """Dense Hamiltonian of the correlated space for another program: h1, (pq|rs)
    (chemists' notation), the constant (nuclear repulsion and the frozen core,
    whose mean field is in h1), and Omega of every spinor.  Core and virtual
    spinors are rotated within their spaces to eigenvectors of the generalized
    Fock matrix and of J_z (the active ones are J_z-adapted already)."""
    sp = eris.spaces
    gamma = act.averaged_rdm1(civecs[:, :MODELS[0]], NELEC)
    f = fock.generalized_fock(eris, gamma)
    jz = socutils_x2c.jz_mo(mol, eris.mo_coeff)
    u = numpy.eye(sp.nmo, dtype=numpy.complex128)
    par = numpy.zeros_like(jz)
    if inversion_parity(eris.mo_coeff)[0] is not None:
        sl = mol.aoslice_2c_by_atom()
        sign = []
        for ib in range(mol.nbas):
            sign += [(-1) ** mol.bas_angular(ib)] * (mol.bas_len_spinor(ib) * mol.bas_nctr(ib))
        inv = numpy.zeros((mol.nao_2c(),) * 2)
        idx = numpy.arange(sl[0][3] - sl[0][2])
        inv[sl[0][2] + idx, sl[1][2] + idx] = numpy.array(sign)[sl[1][2]:sl[1][3]]
        inv[sl[1][2] + idx, sl[0][2] + idx] = numpy.array(sign)[sl[0][2]:sl[0][3]]
        par = eris.mo_coeff.conj().T @ mol.intor('int1e_ovlp_spinor') @ inv @ eris.mo_coeff
    for x in 'cv':
        s = sp.slice(x)
        if s.stop > s.start:
            # common eigenvectors of the Fock matrix, J_z and the inversion
            u[s, s] = numpy.linalg.eigh(f[s, s] + 1e-3 * jz[s, s] + 0.37e-3 * par[s, s])[1]
    new = eris.rotate(u)
    jz = u.conj().T @ jz @ u
    f = u.conj().T @ f @ u
    off = abs(jz - numpy.diag(jz.diagonal())).max()
    omega2 = numpy.rint(2 * jz.diagonal().real).astype(int)
    if args.dump_light:
        # spinors and labels only (for a Hamiltonian built elsewhere from them)
        parity, pdev = inversion_parity(new.mo_coeff)
        numpy.savez(path, ncore=sp.ncore, nact=sp.nact, nvirt=sp.nvirt, nelec=sp.ncore + NELEC,
                    omega2=omega2, parity=parity, eps=f.diagonal().real, e_casci=numpy.asarray(e_cas),
                    mo_coeff=new.mo_coeff, mo_frozen=FROZEN_MO[0], ecore=new.ecore)
        print('ZFS dump %s (spinors and labels only): %d spinors, J_z off-diagonal %.1e, parity '
              'deviation %.1e' % (path, sp.nmo, off, pdev), flush=True)
        return
    dense = new.to_dense()
    e_chk = ActiveSpace.from_eris(dense).solve(NELEC, len(e_cas))[0]
    parity, pdev = inversion_parity(new.mo_coeff)
    extra = {} if parity is None else dict(parity=parity, parity_mod=2)
    if parity is not None:
        print('ZFS dump parity of the spinors (0 g, 1 u), deviation from +-1 %.1e: %s'
              % (pdev, ' '.join('%d' % x for x in parity)), flush=True)
    numpy.savez(path, h1=dense.h1, eri=dense.eri_block('pppp'), ecore=dense.ecore,
                nelec=sp.ncore + NELEC, ncore=sp.ncore, nact=sp.nact, nvirt=sp.nvirt,
                omega2=omega2, eps=f.diagonal().real, e_casci=numpy.asarray(e_cas),
                mo_coeff=new.mo_coeff, **extra)
    print('ZFS dump %s: %d spinors (%d core, %d active, %d virtual), %d electrons, '
          'constant %.10f; J_z off-diagonal %.1e, |2 Omega - integer| %.1e; CASCI from the '
          'dense integrals differs by %.1e Eh'
          % (path, sp.nmo, sp.ncore, sp.nact, sp.nvirt, sp.ncore + NELEC, dense.ecore, off,
             abs(2 * jz.diagonal().real - omega2).max(), abs(e_chk - e_cas).max()), flush=True)
    print('ZFS dump 2 Omega of the spinors: ' + ' '.join('%d' % x for x in omega2), flush=True)
    if not args.dump_sf:
        return
    # the same spinors with a spin-free Hamiltonian: spin-free X2C one-electron
    # operator (sfx2c1e of pyscf) instead of the X2C one; the two-electron
    # integrals and the mean field of the frozen core are the bare Coulomb
    # operator already.  In a basis contracted per kappa the spinor space is
    # not closed under spin rotations, so the triplet is not exactly degenerate.
    from pyscf import scf
    ca, cb = mol.sph2spinor_coeff()
    xmol, contr = _decontract_spinor(mol, mf.with_x2c.xuncontract)
    if contr.shape[0] == contr.shape[1] and abs(contr - numpy.eye(len(contr))).max() < 1e-12:
        h0 = scf.RHF(mol).sfx2c1e().get_hcore()
        h_sf = ca.conj().T @ h0 @ ca + cb.conj().T @ h0 @ cb
    else:
        # contracted (also per kappa): the operator in the primitive spinor
        # basis, then the contraction that the X2C step uses
        h0 = scf.RHF(xmol).sfx2c1e().get_hcore()
        xa, xb = xmol.sph2spinor_coeff()
        h_sf = contr.conj().T @ (xa.conj().T @ h0 @ xa + xb.conj().T @ h0 @ xb) @ contr
    # how far the spinor spaces are from being closed under spin rotations
    # (sigma_x acting on a spinor, its component outside the space): for the
    # frozen spinors, the active ones and the whole basis
    snr = mol.intor('int1e_ovlp')

    def spin_defect(c):
        x = numpy.vstack((ca @ c, cb @ c))
        sx = numpy.vstack((cb @ c, ca @ c))
        big = numpy.kron(numpy.eye(2), snr)
        rest = sx - x @ numpy.linalg.solve(x.conj().T @ big @ x, x.conj().T @ big @ sx)
        return numpy.sqrt(numpy.einsum('ik,ij,jk->k', rest.conj(), big, rest).real.max())

    a_ = sp.slice('a')
    print('ZFS dump spin-rotation defect (norm of the part of sigma_x phi outside the space): '
          'frozen spinors %.1e, active spinors %.1e, frozen + all correlated spinors %.1e'
          % (spin_defect(FROZEN_MO[0]), spin_defect(new.mo_coeff[:, a_]),
             spin_defect(numpy.hstack((FROZEN_MO[0], new.mo_coeff)))), flush=True)
    h_so = mf.get_hcore()
    cf = FROZEN_MO[0]
    dm = cf @ cf.conj().T
    ecore = new.ecore + numpy.einsum('ij,ji->', dm, h_sf - h_so).real
    sf = type(new)(ecore, new.hcore_ao - h_so + h_sf, new.mo_coeff, sp, new.eri_fn, new.veff_fn)
    sf.active_charges = getattr(new, 'active_charges', None)
    dense = sf.to_dense()
    e_sf = ActiveSpace.from_eris(dense).solve(NELEC, len(e_cas))[0]
    path_sf = path.replace('.npz', '_sf.npz')
    numpy.savez(path_sf, h1=dense.h1, eri=dense.eri_block('pppp'), ecore=dense.ecore,
                nelec=sp.ncore + NELEC, ncore=sp.ncore, nact=sp.nact, nvirt=sp.nvirt,
                omega2=omega2, eps=f.diagonal().real, e_casci=e_sf, mo_coeff=new.mo_coeff)
    om = numpy.asarray(omega2)
    print('ZFS dump %s: spin-free Hamiltonian, constant %.10f; max |h1| between different Omega '
          '%.1e; CASCI energies %s; levels / cm-1 %s'
          % (path_sf, dense.ecore, abs(dense.h1[om[:, None] != om[None, :]]).max(),
             ' '.join('%.12f' % x for x in e_sf),
             ' '.join('%.4f' % ((x - e_sf[0]) * HARTREE2WAVENUMBER) for x in e_sf)), flush=True)
    # the triplet of the spin-free Hamiltonian at the correlated levels: it is
    # degenerate only in the full-CI limit of a spin-closed orbital space
    nev = PCNEVPT2(sf, NELEC, nstate=3, contraction='union')
    nev.verbose = 0
    e_nev = nev.kernel()
    ci = MRCI(sf, NELEC, nstate=3)
    ci.verbose = 0
    e_ci = numpy.sort(numpy.asarray(ci.kernel(nroots=3)).real)
    for label, e in (('PC-NEVPT2', e_nev), ('ic-MRCI', e_ci)):
        print('ZFS dump spin-free %s: E = %s; levels / cm-1 %s'
              % (label, ' '.join('%.10f' % x for x in e),
                 ' '.join('%.4f' % ((x - e[0]) * HARTREE2WAVENUMBER) for x in e)), flush=True)


class Model:
    '''The CASCI model space itself, in the form levels() expects.'''

    def __init__(self, e, c):
        self.civecs = c
        self.heff = numpy.diag(e).astype(complex)


def inversion_parity(c):
    '''Parity (0 g, 1 u) of the spinors c of a homonuclear diatomic molecule and
    the largest deviation of <p|i|p> from +-1.'''
    sl = mol.aoslice_2c_by_atom()
    if mol.natm != 2 or mol.atom_charge(0) != mol.atom_charge(1):
        return None, None
    sign = []
    for ib in range(mol.nbas):
        sign += [(-1) ** mol.bas_angular(ib)] * (mol.bas_len_spinor(ib) * mol.bas_nctr(ib))
    sign = numpy.array(sign, dtype=float)
    n = mol.nao_2c()
    half = sl[0][3] - sl[0][2]
    inv = numpy.zeros((n, n))
    idx = numpy.arange(half)
    inv[sl[0][2] + idx, sl[1][2] + idx] = sign[sl[1][2]:sl[1][3]]
    inv[sl[1][2] + idx, sl[0][2] + idx] = sign[sl[0][2]:sl[0][3]]
    s = mol.intor('int1e_ovlp_spinor')
    p = c.conj().T @ s @ inv @ c
    d = p.diagonal().real
    return (d < 0).astype(int), max(abs(abs(d) - 1).max(), abs(p - numpy.diag(p.diagonal())).max())


def natural_virtuals(eris, civecs):
    '''Virtual-virtual block of the one-particle density of the MRCI wave
    functions of the first model space (averaged over the roots), its
    eigenvalues (descending) and eigenvectors, and the Hamiltonian in the
    quasi-canonical orbitals of that MRCI, to which the vectors refer.'''
    n = MODELS[0]
    ci = MRCI(eris, NELEC, civecs=civecs[:, :n])
    ci.verbose = 0
    ci.kernel()
    cc = ci.basis.unpack(numpy.asarray(ci.ci))
    nv = ci.eris.spaces.nvirt
    dvv = numpy.zeros((nv, nv), dtype=numpy.complex128)
    for name, c in cc.items():
        npart = ci.basis.blocks[name].np
        if npart == 1:
            c1 = c.reshape(len(ci.ci), nv, -1)
            dvv += numpy.einsum('zak,zbk->ab', c1.conj(), c1)
        elif npart == 2:
            c2 = c.reshape(len(ci.ci), nv, nv, -1)
            dvv += numpy.einsum('zack,zbck->ab', c2.conj(), c2)
    dvv /= len(ci.ci)
    occ, vec = numpy.linalg.eigh(dvv.T)
    return occ[::-1], vec[:, ::-1], ci.eris


def run(mo, nsa, e_sa, core, emax, eris=None, nat=''):
    full_space = eris is None
    if full_space:
        eris, e_ref = correlated_space(mo, nsa, core, emax)
    # Jz-adapted active spinors: the model states become Omega eigenstates
    eris, sym = symmetry.adapt(eris)
    act = ActiveSpace.from_eris(eris)
    e_cas, civecs = act.solve(NELEC, NROOTS)
    a = eris.spaces.slice('a')
    jz_act = socutils_x2c.jz_mo(mol, eris.mo_coeff)[a, a]
    if args.dump and (full_space or '%(nv)' in args.dump):
        dump_hamiltonian(eris, act, civecs, e_cas, args.dump % dict(
            emax=emax, core=core, nsa=nsa, nv=eris.spaces.nvirt))

    def levels(pt, tol=1e-8):
        '''(energy, Omega values) of the levels of an effective Hamiltonian.'''
        c = pt.civecs
        n = c.shape[1]
        jz = numpy.array([[numpy.sum(jz_act * act.trans_rdm1(c[:, i], c[:, j], NELEC))
                           for j in range(n)] for i in range(n)])
        e, v = numpy.linalg.eigh(pt.heff)
        jz = v.conj().T @ jz @ v
        out, start = [], 0
        for k in range(1, n + 1):
            if k == n or e[k] - e[k - 1] > tol:
                omega = numpy.linalg.eigvalsh(jz[start:k, start:k])
                out.append((e[start:k].mean(), numpy.round(omega, 3) + 0.0))
                start = k
        return out

    def report(label, pt):
        lv = levels(pt)
        rel = [(e - lv[0][0]) * HARTREE2WAVENUMBER for e, _ in lv]
        # the splitting is the distance of the Omega = +-1 pair from the 0+ level
        if ATOM:
            print('ZFS %-40s E(ground) = %.10f  levels / cm-1 (degeneracy): %s'
                  % (label, lv[0][0], '  '.join('%.2f (%d)' % (r, len(om))
                                                for r, (_, om) in zip(rel, lv))), flush=True)
            return rel
        first = [abs(o).max() for _, o in lv[:2]]
        if len(lv) < 2 or first[0] > 0.5 or abs(first[1] - 1) > 0.1:
            print('ZFS     WARNING: the two lowest levels are not X 0+ and X +-1')
        print('ZFS %-40s E(X 0+) = %.10f  X(+-1) - X(0+) = %10.4f cm-1' % (label, lv[0][0], rel[1]))
        print('ZFS     levels / cm-1: '
              + '   '.join('%.3f (Omega %s)' % (r, ' '.join('%+g' % o for o in om))
                           for r, (_, om) in zip(rel, lv)), flush=True)
        return rel

    print('ZFS ' + '-' * 100)
    print('ZFS %s %s, SA-%d orbitals, %s core, emax %g%s; %d core, %d active, %d virtual spinors'
          % (args.molecule, tag, nsa, core, emax, nat, eris.spaces.ncore, NACT, eris.spaces.nvirt))
    occ = numpy.linalg.eigvalsh(act.averaged_rdm1(civecs[:, :nsa], NELEC))[::-1]
    print('ZFS natural occupations of the active spinors: ' + ' '.join('%.3f' % x for x in occ))
    if sym is not None:
        print('ZFS active-space symmetry: %d generators, largest discarded integral %.1e Eh'
              % (sym.ngen, sym.error))
    print('ZFS CASCI in zmr, average of the %d states: %.10f (CASSCF: difference %.1e Eh)'
          % (nsa, e_cas[:nsa].mean(), e_cas[:nsa].mean() - e_sa))
    result = {'molecule': args.molecule, 'hamiltonian': tag, 'nsa': nsa, 'core': core,
              'emax': emax, 'ncore': eris.spaces.ncore, 'nvirt': eris.spaces.nvirt,
              'e_sa': e_sa}
    rel = report('CASSCF', Model(e_cas, civecs))
    result['casscf'] = rel[1]
    result['casscf_levels'] = rel
    if args.corr:
        # X2Ccorr: CASCI with the four-component two-electron integrals in the
        # active space and, as one-electron part, the four-component bare
        # Hamiltonian with the mean field of the core spinors only; its level
        # shifts are a correction to add to the NEVPT2 levels.  (The MRCI
        # below still takes the X2C one-electron part with these integrals.)
        t0 = time.time()
        heff, aaaa, e_core = eris.active_hamiltonian()
        new = corr_active_eri(eris.mo_coeff[:, a])
        e_corr, c_corr = ActiveSpace(heff, new, e_core, charges=getattr(
            eris, 'active_charges', None)).solve(NELEC, NROOTS)
        corr_eri = new
        ref, cor = grouped(e_cas), grouped(e_corr)
        print('ZFS X2Ccorr CASCI: max |d(tu|vw)| = %.2e Eh; levels / cm-1 (degeneracy): %s'
              % (abs(new - aaaa).max(), '  '.join('%.3f (%d)' % x for x in cor)))
        if [d for _, d in ref] == [d for _, d in cor]:
            # not the correction: the X2C one-electron operator still holds the
            # mean-field two-electron spin-orbit term of the active electrons
            result['corr_integrals_only'] = [c[0] - r[0] for c, r in zip(cor, ref)]
            print('ZFS X2Ccorr, active integrals only (double counts the active-active '
                  'two-electron spin-orbit term), shifts / cm-1: '
                  + '  '.join('%+.3f' % x for x in result['corr_integrals_only']))
        stamp('    X2Ccorr active-space integrals: %.0f s' % (time.time() - t0))
        t0 = time.time()
        h4, e4 = corr_core_hamiltonian(mo[:, :ncore_total], eris.mo_coeff[:, a])
        e_c4, _ = ActiveSpace(h4, new, e4, charges=getattr(
            eris, 'active_charges', None)).solve(NELEC, NROOTS)
        cor4 = grouped(e_c4)
        print('ZFS X2Ccorr with the four-component core: E_core %.10f (X2C %.10f), '
              'max |d h_tu| = %.2e Eh; levels / cm-1 (degeneracy): %s'
              % (e4, e_core, abs(h4 - heff).max(),
                 '  '.join('%.3f (%d)' % x for x in cor4)))
        if [d for _, d in ref] == [d for _, d in cor4]:
            result['corr'] = [c[0] - r[0] for c, r in zip(cor4, ref)]
            print('ZFS X2Ccorr correction of the levels / cm-1: '
                  + '  '.join('%+.3f' % x for x in result['corr']))
        stamp('    four-component core mean field: %.0f s' % (time.time() - t0))
    # MRCI with the Hamiltonian as it is and, with --corr, with the
    # four-component active-space integrals (model states of that CASCI)
    variants_h = [('', civecs, None)]
    if args.corr and not args.mrci_plain:
        variants_h = []
    if args.corr:
        variants_h.append((' + X2Ccorr', c_corr, corr_eri))
    for functional, (suffix, cvec, act_eri) in [
            (f, v) for f in args.mrci.split(',') if f for v in variants_h]:
        t0 = time.time()
        n = MODELS[0]
        ci = MRCI(eris, NELEC, civecs=cvec[:, :n], functional=functional, active_eri=act_eri,
                  contraction=args.mrci_contraction)
        ci.verbose = 4
        if args.vvvv_memory:
            from socutils.zmr.mrci.sigma import Sigma
            Sigma.vvvv_dense_memory = args.vvvv_memory
        if args.mrci_wmax:
            from socutils.zmr.mrci.sigma import Sigma
            Sigma.w_max = args.mrci_wmax
        if args.mrci_maxcycle:
            ci.max_cycle = args.mrci_maxcycle
            ci.timing_lines = 30
        if args.mrci_memory:
            # where the memory goes: setup, one sigma, the arrays that stay
            stamp('    MRCI memory: before the setup')
            ci.build()
            stamp('    MRCI memory: block bases built, dimension %d' % ci.basis.dim)
            import cProfile
            import pstats
            prof = cProfile.Profile()
            prof.enable()
            for bra in ci.basis.names:
                for ket in ci.basis.names:
                    ci.sigma.compiled(bra, ket)
                stamp('    MRCI memory: contractions of bra block %s compiled' % bra)
            prof.disable()
            print('ZFS     profile of the setup')
            pstats.Stats(prof, stream=sys.stdout).sort_stats('tottime').print_stats(16)
            ci.sigma.compile_all()
            stamp('    MRCI memory: setup arrays released')
            # dense vectors, as in the iterations (the reference alone has only
            # its internal block)
            rng = numpy.random.default_rng(0)
            x = ci.reference_vectors()
            x = x + 0.01 * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
            prof = cProfile.Profile()
            ci.sigma.timings.clear()
            prof.enable()
            ci.sigma.matvec(x[:1])
            prof.disable()
            stamp('    MRCI memory: one sigma, one vector')
            ticks = sorted(ci.sigma.timings.items(), key=lambda kv: -kv[1])
            print('ZFS     one sigma: %.1f s in %d entries' % (sum(v for _, v in ticks), len(ticks)))
            for k, v in ticks[:45]:
                print('ZFS       %6.2f s  %s <- %s %s' % (v, k[0], k[1], k[2]))
            by_bra = {}
            for k, v in ticks:
                by_bra[k[0]] = by_bra.get(k[0], 0) + v
            print('ZFS     by bra block: ' + '  '.join('%s %.1f' % kv for kv in sorted(
                by_bra.items(), key=lambda kv: -kv[1])))
            pstats.Stats(prof, stream=sys.stdout).sort_stats('tottime').print_stats(22)
            ci.sigma.matvec(x)
            stamp('    MRCI memory: one sigma, %d vectors' % len(x))
            rows = ci.sigma.memory_report()
            print('ZFS     arrays kept between iterations: %.1f GB in all'
                  % (sum(b for _, b in rows) / 1e9))
            for label, b in rows[:40]:
                print('ZFS       %8.2f GB  %s' % (b / 1e9, label))
            sys.stdout.flush()
            sys.exit(0)
        ci.conv_tol = args.mrci_tol
        e = ci.kernel()
        # the internal parts of the roots identify the levels
        cas = ci.basis.blocks['cas']
        inner = cas.b @ ci.ci[:, :cas.size].T
        inner /= numpy.linalg.norm(inner, axis=0)
        label = {'ci': 'MRCI', 'acpf': 'MR-ACPF', 'aqcc': 'MR-AQCC'}[functional] + suffix
        print('ZFS     %s: dimension %d, converged %s, weights of the internal block %s'
              % (label, ci.basis.dim, ci.converged, ' '.join('%.4f' % w for w in ci.weights)))
        variants = [(label, e)]
        if functional == 'ci':
            variants += [('MRCI+Q (Davidson)' + suffix, ci.e_davidson),
                         ('MRCI+Q (renormalized)' + suffix, ci.e_davidson_renormalized)]
        for name, energies in variants:
            order = numpy.argsort(energies)
            rel = report('%s, %d states' % (name, n), Model(energies[order], inner[:, order]))
            key = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
            result['%s%d' % (key, n)] = rel[1]
            result['%s%d_levels' % (key, n)] = rel
        stamp('    %s, %d states: %.0f s' % (label, n, time.time() - t0))
        top = sorted(ci.sigma.timings.items(), key=lambda x: -x[1])[:8]
        print('ZFS     sigma, largest contributions (s): '
              + '  '.join('%s<-%s %s %.0f' % (k + (v,)) for k, v in top), flush=True)
    methods = [('pc', 'PC-NEVPT2 union', PCNEVPT2, dict(contraction='union'), MODELS)]
    if args.uc:
        methods.append(('uc', 'UC-NEVPT2', UCNEVPT2, {}, MODELS[:1]))
    if args.fock_h0:
        # the same drivers with the Fock operator as zeroth-order Hamiltonian
        # in the active space (a CASPT2 with decoupled classes)
        methods.append(('pf', 'PC-PT2(Fock H0) union', PCNEVPT2, dict(contraction='union'),
                        MODELS))
        if args.uc:
            methods.append(('uf', 'UC-PT2(Fock H0)', UCNEVPT2, {}, MODELS[:1]))
    for key, name, cls, kw, models in methods:
        for nstate in models:
            for e0 in ('state', 'average'):
                t0 = time.time()
                pt = cls(eris, NELEC, civecs=civecs[:, :nstate], e0=e0, **kw)
                if key in ('pf', 'uf'):
                    pt.h0 = 'fock'
                pt.kernel()
                if key in ('pf', 'uf'):
                    # heff refers to the model states rotated to diagonalize H0
                    pt.civecs = pt.civecs @ pt.model_rotation
                rel = report('%s, %d states, e0=%s' % (name, nstate, e0), pt)
                if args.diagonal and e0 == 'state':
                    # single-state energies: the diagonal of the effective
                    # Hamiltonian in the CASCI eigenstates (for the Fock H0
                    # without the rotation of the model states)
                    single = pt
                    if key in ('pf', 'uf'):
                        single = cls(eris, NELEC, civecs=civecs[:, :nstate], e0=e0, **kw)
                        single.h0, single.fock_rotate = 'fock', False
                        single.kernel()
                    rel_d = report('%s, %d states, diagonal of heff' % (name, nstate),
                                   Model(single.heff.diagonal().real, civecs[:, :nstate]))
                    result['%s%d_diag_levels' % (key, nstate)] = rel_d
                result['%s%d_%s' % (key, nstate, e0)] = rel[1]
                result['%s%d_%s_levels' % (key, nstate, e0)] = rel
                if key == 'pc' and nstate == MODELS[0] and e0 == 'state':
                    e2 = pt.h2_by_class
                    print('ZFS     E2 of model state 0 by class (Eh): '
                          + ' '.join('%s %.6f' % (k, v[0, 0].real) for k, v in e2.items()))
                    result['e2'] = float(pt.h2[0, 0].real)
                stamp('    %s, %d states, e0=%s: %.0f s' % (name, nstate, e0, time.time() - t0))
    for contraction in [c for c in args.caspt2.split(',') if c]:
        # CASPT2 proper: the full Fock operator in the first-order space,
        # internally contracted (fic) or in all external determinants (none)
        for nstate in MODELS:
            t0 = time.time()
            pt = CASPT2(eris, NELEC, civecs=civecs[:, :nstate], contraction=contraction)
            pt.verbose = 0
            pt.kernel()
            name = 'CASPT2 %s' % ('contracted' if contraction == 'fic' else 'uncontracted')
            rel = report('%s, %d states' % (name, nstate),
                         SimpleNamespace(heff=pt.heff,
                                         civecs=civecs[:, :nstate] @ pt.model_rotation))
            result['caspt2_%s%d_levels' % (contraction, nstate)] = rel
            stamp('    %s, %d states: %.0f s, dimension %d, %d .. %d products with F, '
                  'reference weights %.3f .. %.3f, converged %s'
                  % (name, nstate, time.time() - t0, pt.basis.dim, min(pt.niter),
                     max(pt.niter), pt.weights.min(), pt.weights.max(), pt.converged))
    print('RESULT ' + json.dumps(result), flush=True)
    if full_space and args.natvirt:
        # frozen natural orbitals: the n virtual spinors with the largest
        # occupation in the MRCI of the full space (never cutting inside a
        # group of equal occupations), and everything again in that space
        occ, vec, qc = natural_virtuals(eris, civecs)
        print('ZFS natural virtual spinors of the MRCI: %.5f electrons in %d spinors; '
              'occupations %s ...' % (occ.sum(), len(occ), ' '.join('%.2e' % x for x in occ[:12])))
        for n in (int(x) for x in args.natvirt.split(',')):
            while n < len(occ) and occ[n] > 0.98 * occ[n - 1]:
                n += 1
            print('ZFS %d natural virtual spinors kept: %.2f %% of the virtual occupation, '
                  'occupations %.2e | %.2e'
                  % (n, 100 * occ[:n].sum() / occ.sum(), occ[n - 1], occ[n] if n < len(occ) else 0.0),
                  flush=True)
            run(mo, nsa, e_sa, core, emax, eris=qc.with_virtuals(vec[:, :n]), nat=', %d natural virtuals' % n)


mo = mf.mo_coeff
for nsa in (int(x) for x in args.nsa.split(',')):
    mo, e_sa = casscf(nsa, mo)
    stamp('SA-%d CASSCF(%d,%d) %s: E(SA) = %.10f' % (nsa, NELEC, NACT, tag, e_sa))
    for core in args.core.split(','):
        for emax in (float(x) for x in args.emax.split(',')):
            run(mo, nsa, e_sa, core, emax)
stamp('done')
