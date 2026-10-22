#!/usr/bin/env python
'''
Zero-field splitting of O2 X 3Sigma_g- from state-averaged spinor CASSCF
orbitals, followed by multi-state PC-NEVPT2 with the union contraction (and
UC-NEVPT2), for two active spaces and two state averages.

Active space: the minimal one, 2 electrons in the 4 pi_g* spinors, or the
full valence 2p space, 8 electrons in 12 spinors (3 sigma_g, pi_u, pi_g*,
3 sigma_u).  State average: the three components of X 3Sigma_g-, or the six
lowest states (X 3Sigma_g-, a 1Delta_g, b 1Sigma_g+).

The orbital part follows the zmcscf O2 inputs: R(O-O) = 1.20752 A,
uncontracted aug-cc-pVTZ, Gaussian nuclei, X2CAMF Hamiltonian, spinor SCF of
O2^2+ (empty pi_g*) with Cholesky-decomposed integrals (1e-7), super-CI
CASSCF.  The SCF is Kramers restricted (KRHF) instead of the linear-symmetry
SCF of those inputs: the super-CI step of socutils is projected onto
Kramers-symmetric rotations and needs phase-clean Kramers pairs; with the
arbitrary pair phases of the symmetry-blocked SCF it stalls.  zmr then takes
exact four-index integrals in the CASSCF orbitals; the 1s spinors are frozen.

Usage:  python 04-o2_zfs_active_spaces.py [dcb|dc] [2,4|8,12] [3|6] [basis]
    dcb  Dirac-Coulomb-Breit atomic mean field (default)
    dc   Dirac-Coulomb atomic mean field
    2,4 (default) or 8,12: active electrons, active spinors
    3 (default) or 6: states in the average of the orbital optimization

The X2CAMF Hamiltonian has no two-electron spin-spin operator, so the
splitting is the spin-orbit part of the experimental value (3.96 cm^-1 in
total).  socutils writes cderi.chk to the working directory.
'''
import sys
import time

import numpy
from pyscf import gto, lib
from socutils.mcscf import zmc_ao2mo, zmc_superci, zmcscf
from socutils.scf import spinor_hf
from socutils.somf import x2cmp

from socutils.zmr.active import symmetry
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import socutils_x2c
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc.nevpt2 import UCNEVPT2

HARTREE2WAVENUMBER = 219474.6313632

variant = sys.argv[1] if len(sys.argv) > 1 else 'dcb'
NELEC, NACT = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '2,4').split(','))
NSA = int(sys.argv[3]) if len(sys.argv) > 3 else 3
basis = sys.argv[4] if len(sys.argv) > 4 else 'uncaugccpvtz'
with_gaunt, with_breit = {'dcb': (True, True), 'dc': (False, False)}[variant]
# spinors: 1s frozen, the other doubly occupied ones are the correlated core
NFROZEN = 4
NCORE = 16 - NELEC - NFROZEN

# closed-shell O2^2+ (empty pi_g*) as the starting point of the CASSCF
mol = gto.M(atom='O 0 0 0; O 0 0 1.20752', basis={'O': basis}, charge=2, verbose=4,
            max_memory=12000, nucmod='G')
cderi = zmc_ao2mo.chunked_cholesky_threeloop(mol, max_error=1e-7)
lib.chkfile.save('cderi.chk', 'j3c', lib.pack_tril(cderi.reshape(-1, mol.nao, mol.nao)))
mf = spinor_hf.KRHF(mol).density_fit()
mf.with_x2c = x2cmp.X2CMP(mol, x2cmp='x2camf', with_gaunt=with_gaunt, with_breit=with_breit)
mf.with_df._cderi = 'cderi.chk'
mf.kernel()

# neutral molecule: SA-CASSCF over the three components of the triplet, or
# over the six states that arise from pi_g*^2
mf.mol.charge = 0
mf.mol.build()
mc = zmcscf.CASSCF(mf, NACT, NELEC).state_average(numpy.ones(NSA) / NSA)
mc.max_cycle_macro = 100
# the splitting is not variational: an orbital gradient of 1e-4 (the default
# threshold) leaves it uncertain by 0.01 cm^-1
converged, e_sa, _, _, mo, _ = zmc_superci.mcscf_superci(mc, mf.mo_coeff, cderi=cderi,
                                                         conv_tol=1e-11, conv_tol_grad=1e-6)
e_sa = float(numpy.mean(e_sa))
if not converged:
    raise RuntimeError('SA-CASSCF did not converge in %d macro iterations (E = %.10f); '
                       'the splitting is not meaningful with these orbitals'
                       % (mc.max_cycle_macro, e_sa))

# exact integrals in the CASSCF orbitals, with the X2C one-electron operator of the SCF
mf_exact = spinor_hf.SCF(mol)
mf_exact.with_x2c = mf.with_x2c
mf_exact.verbose = 0
eris = socutils_x2c.from_scf(mf_exact, ncore=NCORE, nact=NACT, nfrozen=NFROZEN, mo_coeff=mo,
                             mol=mol)
# Jz-adapted active spinors: the model states become Omega eigenstates (same energies)
eris, sym = symmetry.adapt(eris)
act = ActiveSpace.from_eris(eris)
e_cas, civecs = act.solve(NELEC, 6)
jz_act = socutils_x2c.jz_mo(mol, eris.mo_coeff)[eris.spaces.slice('a'), eris.spaces.slice('a')]


def jz_model(c):
    '''Jz in the basis of the model states given as the columns of c.'''
    n = c.shape[1]
    return numpy.array([[numpy.sum(jz_act * act.trans_rdm1(c[:, i], c[:, j], NELEC))
                         for j in range(n)] for i in range(n)])


def levels(pt, tol=1e-8):
    '''Levels of the effective Hamiltonian: (energy, Omega values of its states).'''
    n = pt.civecs.shape[1]
    e, v = numpy.linalg.eigh(pt.heff)
    jz = v.conj().T @ jz_model(pt.civecs) @ v
    out, start = [], 0
    for k in range(1, n + 1):
        if k == n or e[k] - e[k - 1] > tol:
            omega = numpy.linalg.eigvalsh(jz[start:k, start:k])
            out.append((e[start:k].mean(), numpy.round(omega, 3) + 0.0))
            start = k
    return out


def report(label, pt, t0=None):
    lv = levels(pt)
    rel = [(e - lv[0][0]) * HARTREE2WAVENUMBER for e, _ in lv]
    e2 = '' if pt is cas else '  E2(X 0+) = %.8f' % (lv[0][0] - e_cas[0])
    print('ZFS %-44s E(X 0+) = %.10f  X(+-1) - X(0+) = %7.4f cm-1%s%s'
          % (label, lv[0][0], rel[1], e2, '' if t0 is None else '  [%.1f s]' % (time.time() - t0)))
    print('ZFS     levels / cm-1: '
          + '   '.join('%.3f (Omega %s)' % (r, ' '.join('%+g' % o for o in om))
                       for r, (_, om) in zip(rel, lv)), flush=True)
    return rel[1]


def splitting_by_class(pt):
    '''Contribution of every perturber class to X(+-1) - X(0+) for the
    three-state model space.  The model states are Omega eigenstates, so the
    class matrices are diagonal and the contributions add up to the change
    of the splitting.'''
    omega = numpy.round(jz_model(pt.civecs).diagonal().real)
    zero, pair = numpy.nonzero(omega == 0)[0], numpy.nonzero(omega != 0)[0]
    assert len(zero) == 1 and len(pair) == 2
    out, offdiag = [], 0.0
    for name, a in pt.h2_by_class.items():
        d = a.diagonal().real
        out.append('%s %+.4f' % (name, (d[pair].mean() - d[zero[0]]) * HARTREE2WAVENUMBER))
        offdiag = max(offdiag, abs(a - numpy.diag(a.diagonal())).max())
    print('ZFS     X(+-1) - X(0+) by class / cm-1: %s   (largest off-diagonal element %.0e Eh)'
          % ('  '.join(out), offdiag))


class Model:
    '''The CASCI model space itself, in the form report() expects.'''

    def __init__(self, nstate):
        self.civecs = civecs[:, :nstate]
        self.heff = numpy.diag(e_cas[:nstate]).astype(complex)


cas = Model(6)
print('ZFS ' + '=' * 100)
print('ZFS O2 X 3Sigma_g-, X2CAMF (%s), basis %s, %d spinors, R = 1.20752 A'
      % ({'dcb': 'Dirac-Coulomb-Breit', 'dc': 'Dirac-Coulomb'}[variant], basis, mo.shape[1]))
print('ZFS SA-CASSCF(%de, %d spinors; %d states): converged %s, E(SA) = %.10f'
      % (NELEC, NACT, NSA, converged, e_sa))
occ = numpy.linalg.eigvalsh(act.averaged_rdm1(civecs[:, :NSA], NELEC))[::-1]
print('ZFS natural occupations of the active spinors: ' + ' '.join('%.4f' % x for x in occ))
if sym is not None:
    print('ZFS active-space symmetry: %d generators, largest discarded integral %.1e Eh'
          % (sym.ngen, sym.error))
print('ZFS CASCI with exact integrals, average of the %d states: %.10f (CASSCF with CD '
      'integrals: difference %.1e)' % (NSA, e_cas[:NSA].mean(), e_cas[:NSA].mean() - e_sa))
report('CASSCF(%d,%d)' % (NELEC, NACT), cas)

# model space: the three components of X 3Sigma_g-, or the six lowest states
# (X, a 1Delta_g, b 1Sigma_g+; all states of the minimal active space)
for nstate in (3, 6):
    c = civecs[:, :nstate]
    for name, cls, kw in (('PC-NEVPT2 union', PCNEVPT2, dict(contraction='union')),
                          ('UC-NEVPT2', UCNEVPT2, {})):
        for e0 in ('state', 'average'):
            t0 = time.time()
            pt = cls(eris, NELEC, civecs=c, e0=e0, **kw)
            pt.kernel()
            report('%s, %d-state model space, e0=%s' % (name, nstate, e0), pt, t0)
            if name.startswith('PC') and nstate == 3:
                splitting_by_class(pt)
            if name.startswith('PC') and nstate == 3 and e0 == 'state':
                e2 = ' '.join('%s %.6f' % (k, v[0, 0].real) for k, v in pt.h2_by_class.items())
                print('ZFS     second-order energy of model state 0 by class (Eh): ' + e2)
                if NACT <= 6:
                    # the same matrix from explicit transition RDMs up to rank 4
                    # (nstate^2 nact^8 numbers: 62 GB for 12 spinors and 3 states)
                    rdm = PCNEVPT2(eris, NELEC, civecs=c, e0=e0, contraction='union',
                                   amplitudes='rdm')
                    rdm.kernel()
                    print('ZFS     RDM route (S and K from RDMs up to rank 4) vs CI-vector route: '
                          'max |dH_eff| = %.1e Eh' % abs(rdm.heff - pt.heff).max())
print('ZFS ' + '=' * 100)
