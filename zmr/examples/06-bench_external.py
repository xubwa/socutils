#!/usr/bin/env python
'''
Timing of NEVPT2 for a large external space, on a synthetic Hamiltonian of
chosen dimensions: random three-index integrals (so that the two-electron
integrals are positive semidefinite) and a diagonal one-electron operator
with a gap between core, active and virtual spin-orbitals.  No SCF, no basis
set: the cost of the correlation step alone, for any numbers of orbitals.

It uses only the public interface, so the same file runs with different
versions of zmr (set PYTHONPATH to the checkout to be timed):

    PYTHONPATH=~/apps/zmr      python 06-bench_external.py --nvirt 1000 --ncore 30
    PYTHONPATH=~/apps/zmr-dev  python 06-bench_external.py --nvirt 1000 --ncore 30

Counts are spin-orbitals.  The energies of the two runs must agree (same
seed); the NEVPT2 output has the wall time of every step.  --kramers makes the
orbitals Kramers pairs (time reversal used by the external sums).
'''
import argparse
import resource
import time

import numpy

from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.interface import pyscf_nr
from socutils.zmr.pc.nevpt2 import PCNEVPT2

parser = argparse.ArgumentParser()
parser.add_argument('--ncore', type=int, default=20)
parser.add_argument('--nact', type=int, default=8)
parser.add_argument('--nvirt', type=int, default=400)
parser.add_argument('--nelec', type=int, default=4, help='active electrons')
parser.add_argument('--nstate', type=int, default=3)
parser.add_argument('--naux', type=int, default=0, help='vectors (default 2.5 x orbitals)')
parser.add_argument('--e0', default='state')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--amplitudes', default='spectral', choices=('spectral', 'rdm'),
                    help='active part of PC-NEVPT2 from CI vectors or from RDMs')
parser.add_argument('--density', action='store_true',
                    help='also the unrelaxed ccvv density (zmr.uc.density), timed against '
                    'the ccvv energy')
parser.add_argument('--max-memory', type=float, default=0,
                    help='MB for the calculation (default: pyscf.lib.param.MAX_MEMORY)')
parser.add_argument('--kramers', action='store_true',
                    help='Kramers-paired orbitals (time-reversal symmetric one-electron '
                    'operator), so that the external sums use one of every pair of labels')
args = parser.parse_args()

nso = args.ncore + args.nact + args.nvirt
if nso % 2:
    raise SystemExit('ncore + nact + nvirt must be even')
nao = nso // 2
naux = args.naux or int(2.5 * nso)
rng = numpy.random.default_rng(args.seed)


class Dimensions:
    '''Stands in for a molecule: number of spatial functions, no nuclei.'''

    def nao_nr(self):
        return nao

    def energy_nuc(self):
        return 0.0


# vectors with a magnitude that keeps the perturbation small against the gaps
cderi = rng.normal(size=(naux, nao * (nao + 1) // 2)) * (0.3 / numpy.sqrt(naux * nao))
# orbital energies: core at -3 .. -1, active around 0, virtual at 1 .. 40
eps = numpy.concatenate([numpy.linspace(-3.0, -1.0, args.ncore),
                         numpy.linspace(-0.15, 0.15, args.nact),
                         numpy.geomspace(1.0, 40.0, args.nvirt)])
# a random unitary coefficient matrix in the (alpha, beta) x spatial basis
if args.kramers:
    # T|alpha mu> = |beta mu>, T|beta mu> = -|alpha mu>: the coefficients are the
    # Kramers-paired eigenvectors of a random time-reversal symmetric matrix, and the
    # orbital energies come in equal pairs
    from socutils.zmr import kramers
    t_ao = kramers.TimeReversal(numpy.concatenate([numpy.arange(nao, nso), numpy.arange(nao)]),
                                numpy.concatenate([numpy.ones(nao), -numpy.ones(nao)]))
    th = t_ao.matrix()
    x = rng.normal(size=(nso, nso)) + 1j * rng.normal(size=(nso, nso))
    x = x + x.conj().T
    c = kramers.paired_eigh(0.5 * (x + th @ x.conj() @ th.conj().T), t_ao)[1]
    for n in (args.ncore, args.nact, args.nvirt):
        if n % 2:
            raise SystemExit('--kramers needs even numbers of core, active and virtual orbitals')
    eps = eps.reshape(-1, 2).mean(axis=1).repeat(2)
else:
    c = numpy.linalg.qr(rng.normal(size=(nso, nso)) + 1j * rng.normal(size=(nso, nso)))[0]
hcore = c @ numpy.diag(eps) @ c.conj().T

t0 = time.time()
# ghf_eris wants the one-electron operator of one spin block and doubles it;
# hand it the full matrix through a tiny wrapper instead
from socutils.zmr.integrals import MOERIs
from socutils.zmr.interface.cholesky import CholeskyAO2MO
from socutils.zmr.spaces import OrbitalSpaces

source = CholeskyAO2MO(cderi, nao, lambda x: (x[:nao], x[nao:]))
eris = MOERIs(0.0, hcore, c, OrbitalSpaces(args.ncore, args.nact, args.nvirt), source, source.veff)
if args.kramers:
    eris.time_reversal = t_ao.rotated(c)
    assert eris.time_reversal is not None
act = ActiveSpace.from_eris(eris)
e_cas, civecs = act.solve(args.nelec, args.nstate)
print('BENCH %d core + %d active + %d virtual spin-orbitals, %d vectors, %d states; '
      'setup and CASCI %.1f s' % (args.ncore, args.nact, args.nvirt, naux, args.nstate,
                                  time.time() - t0), flush=True)
t0 = time.time()
pt = PCNEVPT2(eris, args.nelec, civecs=civecs, e0=args.e0, contraction='union',
              amplitudes=args.amplitudes)
if args.max_memory:
    pt.max_memory = args.max_memory
e = pt.kernel()
print('BENCH NEVPT2 wall %.1f s, peak memory %.1f GB' % (
    time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
print('BENCH E(CASCI) ' + ' '.join('%.10f' % x for x in e_cas))
print('BENCH E(NEVPT2) ' + ' '.join('%.10f' % x for x in e))
print('BENCH E2 by class (state 0): ' + ' '.join(
    '%s %.8f' % (k, v[0, 0].real) for k, v in pt.h2_by_class.items()))
if args.density:
    from socutils.zmr.uc import classes, density
    qc = pt.quasi_canonical
    ctx = classes.Context(qc.eris, qc.eps_core, qc.eps_virt, None)
    t0 = time.time()
    e_d, dcc, dvv = density.ccvv(ctx)
    t_d = time.time() - t0
    t_e = dict(pt.timings).get('ccvv', float('nan'))
    print('BENCH ccvv density %.1f s (energy %.1f s, ratio %.2f); E_ccvv %.10f (NEVPT2 %.10f)'
          % (t_d, t_e, t_d / t_e, e_d, pt.h2_by_class['ccvv'][0, 0].real), flush=True)
