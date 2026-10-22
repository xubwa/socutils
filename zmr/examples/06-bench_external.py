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
seed); the NEVPT2 output has the wall time of every step.
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
act = ActiveSpace.from_eris(eris)
e_cas, civecs = act.solve(args.nelec, args.nstate)
print('BENCH %d core + %d active + %d virtual spin-orbitals, %d vectors, %d states; '
      'setup and CASCI %.1f s' % (args.ncore, args.nact, args.nvirt, naux, args.nstate,
                                  time.time() - t0), flush=True)
t0 = time.time()
pt = PCNEVPT2(eris, args.nelec, civecs=civecs, e0=args.e0, contraction='union')
e = pt.kernel()
print('BENCH NEVPT2 wall %.1f s, peak memory %.1f GB' % (
    time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
print('BENCH E(CASCI) ' + ' '.join('%.10f' % x for x in e_cas))
print('BENCH E(NEVPT2) ' + ' '.join('%.10f' % x for x in e))
print('BENCH E2 by class (state 0): ' + ' '.join(
    '%s %.8f' % (k, v[0, 0].real) for k, v in pt.h2_by_class.items()))
