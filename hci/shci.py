#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Selected CI for spinor Hamiltonians: driver of the zshci program bundled in
socutils/lib/zshci.

SHCI is a CI solver in the sense of PySCF, with the interface and the
density-matrix conventions of :class:`socutils.fci.zfci.FCISolver`, so the two
can be exchanged::

    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = shci.SHCI(mol)
    mc.fcisolver.schedule = [[0, 1e-4]]
    mc.kernel()

This module replaces the earlier driver of Dice's ZFCI program.  Inputs
written for it keep working as far as the names go (sweep_iter, sweep_epsilon,
davidsonTol, dE, initialStates, scratchDirectory are still understood), with
one difference in meaning: the old program ignored the epsilon schedule and
always did a full CI, whereas the schedule now really selects determinants.
Set ``occslst = 'all'`` for a full CI in a small active space.

The program is run as a separate process: the integrals go to a FCIDUMP file
and the settings to a JSON file in ``scratch``; energies and density matrices
are read back from there.  The CI vectors stay on disk, so what ``kernel``
returns in their place, and what ``make_rdm1`` etc. expect, is the index of
the state.

Build the program with ``make`` at the top of socutils (or ``make`` inside
lib/zshci); set SOCUTILS_ZSHCI_EXE to use another executable.
'''

import itertools
import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy

from pyscf import lib
from pyscf.lib import logger

from socutils.tools import fcidump_rel

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXE = os.environ.get(
    'SOCUTILS_ZSHCI_EXE',
    os.path.join(_HERE, os.pardir, 'lib', 'zshci', 'bin', 'zshci'))


def _state_index(fcivec):
    '''The state a CI-vector handle refers to (None means the first state).'''
    if fcivec is None:
        return 0
    return int(fcivec)


class SHCI(lib.StreamObject):
    '''Selected CI solver.

    Attributes:
        executable : str
            The zshci program.
        mpi_prefix : str or None
            If set (e.g. 'mpirun -np 4'), prepended to the command; the
            executable must then be the MPI build (zshci_mpi).
        scratch : str
            Directory for the input, the integrals and the results.
        nroots : int
        schedule : list of [iteration, epsilon1]
            From that iteration on a determinant D_a is added when
            |H(a,i) c_i| > epsilon1 for some D_i in the space.
        max_iter : int or None
            Maximum number of selection iterations (program default if None).
        energy_tol : float
            Convergence of the selection iterations.
        davidson_tol : float
            Residual norm of the final eigenvectors.
        initial_dets : list of lists of occupied spinors, or None
            Starting determinants.  Default: the lowest nelec spinors.
        occslst : list of lists of occupied spinors, or 'all', or None
            If set, no selection is done: the Hamiltonian is diagonalized in
            exactly these determinants ('all': every determinant, i.e. full
            CI, feasible for small spaces only).
        orbsym : list of int, list of such lists, or None
            Additive symmetry label of every spinor; the calculation stays in
            the block of the starting determinants.  Several lists are
            several independent conserved quantities, e.g. 2*m_j of an axis
            and the parity.
        orbsym_mod : int, or list of int
            Labels add modulo this number; 0 for plain addition.  One number
            per list of orbsym if there are several.
        restrictions : list of dict(orbitals=[...], min=..., max=...)
            Bounds on the number of electrons in sets of spinors.
        pt : dict or None
            Perturbative correction, e.g. dict(method='semistochastic',
            epsilon2=1e-7, epsilon2_large=1e-5, target_error=1e-5).  The
            energies returned by kernel then include it; the density matrices
            stay variational.
        extra : dict
            Merged into the input as is, for options not listed here.
    '''

    def __init__(self, mol=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.executable = DEFAULT_EXE
        self.mpi_prefix = None
        self.scratch = tempfile.mkdtemp(prefix='zshci_', dir=lib.param.TMPDIR)
        self.nroots = 1
        self.schedule = [[0, 1e-4]]
        self.max_iter = None
        self.energy_tol = 1e-8
        self.davidson_tol = 1e-8
        self.initial_dets = None
        self.occslst = None
        self.orbsym = None
        self.orbsym_mod = 0
        self.restrictions = []
        self.pt = None
        self.extra = {}
        self.integral_tol = 1e-14
        # Number type of the calculation: 'auto' runs in real arithmetic when
        # h1e and eri are passed as real arrays, in complex arithmetic
        # otherwise; 'real' or 'complex' force one.  'real' with integrals
        # that have an imaginary part is an error of the program.
        self.scalar = 'auto'

        # Names of the earlier Dice driver, honoured if set (see the module
        # docstring).
        self.sweep_iter = None
        self.sweep_epsilon = None
        self.davidsonTol = None
        self.dE = None
        self.initialStates = None
        self.scratchDirectory = None

        self.converged = False
        self.eci = None
        self.ci = None
        # Spin is not a good quantum number for a spinor wave function; see
        # socutils.fci.zfci.FCISolver.
        self.spin_square = None
        self.states_spin_square = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('executable = %s', self.executable)
        log.info('scratch = %s', self.scratch)
        log.info('nroots = %d', self.nroots)
        if self.occslst is None:
            log.info('schedule = %s', self.schedule)
        else:
            log.info('diagonalization in a given determinant list')
        if self.pt:
            log.info('pt = %s', self.pt)
        return self

    def _apply_legacy_names(self, log):
        '''Copies settings given under the names of the earlier driver.'''
        used = []
        if self.sweep_epsilon:
            iters = self.sweep_iter if self.sweep_iter else [0]
            self.schedule = [[int(i), float(e)]
                             for i, e in zip(iters, self.sweep_epsilon)]
            used.append('sweep_iter/sweep_epsilon')
        for old, new in (('davidsonTol', 'davidson_tol'), ('dE', 'energy_tol'),
                         ('initialStates', 'initial_dets'),
                         ('scratchDirectory', 'scratch')):
            value = getattr(self, old)
            if value is not None:
                setattr(self, new, value)
                used.append(old)
        if used:
            log.note('SHCI: settings taken from legacy names: %s. Note that '
                     'the epsilon schedule now selects determinants; the '
                     'earlier driver always ran a full CI.', ', '.join(used))

    def _input(self, norb, nelec, nroots):
        if not isinstance(nelec, (int, numpy.integer)):
            nelec = sum(nelec)
        inp = {
            'integrals': os.path.join(self.scratch, 'FCIDUMP'),
            'prefix': self.scratch,
            'nroots': int(nroots),
            'davidson': {'tol': self.davidson_tol},
            'rdm': {'one': True, 'two': True, 'transition': nroots > 1},
        }
        if self.occslst is None:
            inp['mode'] = 'select'
            dets = self.initial_dets
            if dets is None:
                dets = [list(range(nelec))]
            select = {'schedule': [[int(i), float(e)] for i, e in self.schedule],
                      'energy_tol': self.energy_tol}
            if self.max_iter is not None:
                select['max_iter'] = int(self.max_iter)
            inp['select'] = select
        else:
            inp['mode'] = 'diag'
            if isinstance(self.occslst, str):
                if self.occslst != 'all':
                    raise ValueError("occslst must be a list or 'all'")
                dets = itertools.combinations(range(norb), nelec)
            else:
                dets = self.occslst
        inp['dets'] = [[int(p) for p in det] for det in dets]
        if self.orbsym is not None:
            sets = self.orbsym
            if numpy.ndim(sets) == 1:
                sets = [sets]
            mods = numpy.atleast_1d(self.orbsym_mod)
            if len(mods) == 1:
                mods = [mods[0]] * len(sets)
            if len(mods) != len(sets):
                raise ValueError('orbsym_mod needs one entry per label set')
            inp['symmetry'] = [{'labels': [int(x) for x in labels],
                                'mod': int(mod)}
                               for labels, mod in zip(sets, mods)]
        if self.restrictions:
            inp['restrictions'] = self.restrictions
        if self.pt:
            inp['pt'] = self.pt
        inp.update(self.extra)
        return inp

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nroots=None, ecore=0,
               verbose=None, **kwargs):
        '''Runs zshci.  ci0 is ignored.

        Returns:
            (energy, 0) for one root, otherwise (energies, [0, 1, ...]); the
            second item stands for the CI vectors (see the module docstring).
        '''
        log = logger.new_logger(self, verbose)
        if nroots is None:
            nroots = self.nroots
        self._apply_legacy_names(log)
        if not os.path.isfile(self.executable):
            raise RuntimeError(
                'zshci executable %s not found; build it with "make" in '
                'socutils or socutils/lib/zshci, or set SOCUTILS_ZSHCI_EXE'
                % self.executable)
        os.makedirs(self.scratch, exist_ok=True)
        self.norb = norb
        self.nelec = nelec

        inp = self._input(norb, nelec, nroots)
        if self.scalar not in ('auto', 'real', 'complex'):
            raise ValueError("scalar must be 'auto', 'real' or 'complex'")
        if self.scalar == 'auto':
            real = numpy.isrealobj(h1e) and numpy.isrealobj(eri)
            inp.setdefault('scalar', 'real' if real else 'complex')
        else:
            inp.setdefault('scalar', self.scalar)
        ecore = complex(ecore)
        fcidump_rel.from_integrals(
            inp['integrals'], numpy.asarray(h1e, dtype=complex),
            numpy.asarray(eri, dtype=complex).reshape(norb * norb, norb * norb),
            norb, nelec, nuc=ecore.real, tol=self.integral_tol)
        input_file = os.path.join(self.scratch, 'input.json')
        with open(input_file, 'w') as f:
            json.dump(inp, f)

        # Results of an earlier call must not survive a failed run.
        for name in os.listdir(self.scratch):
            if name == 'shci.e' or name == 'wfn.h5' or name.startswith('rdm_'):
                os.remove(os.path.join(self.scratch, name))

        command = [self.executable, input_file]
        if self.mpi_prefix:
            command = self.mpi_prefix.split() + command
        env = dict(os.environ)
        env.setdefault('OMP_NUM_THREADS', str(lib.num_threads()))
        output_file = os.path.join(self.scratch, 'output.txt')
        with open(output_file, 'w') as out:
            status = subprocess.call(command, stdout=out,
                                     stderr=subprocess.STDOUT, env=env)
        if log.verbose >= logger.DEBUG:
            with open(output_file) as out:
                log.debug('%s', out.read())
        if status != 0:
            with open(output_file) as out:
                tail = ''.join(out.readlines()[-20:])
            raise RuntimeError('zshci failed with status %d; the end of %s:\n%s'
                               % (status, output_file, tail))

        e = numpy.fromfile(os.path.join(self.scratch, 'shci.e'))
        if e.size != nroots:
            raise RuntimeError('zshci wrote %d energies for %d roots'
                               % (e.size, nroots))
        self.converged = True
        if nroots == 1:
            self.eci, self.ci = e[0], 0
        else:
            self.eci, self.ci = e, list(range(nroots))
        return self.eci, self.ci

    def _read(self, bra, ket, name):
        path = os.path.join(self.scratch, 'rdm_%d_%d.h5' % (bra, ket))
        if not os.path.isfile(path):
            raise RuntimeError('%s not found; run kernel first' % path)
        with h5py.File(path, 'r') as f:
            if name not in f:
                raise RuntimeError('%s holds no %s' % (path, name))
            return f[name][()]

    def make_rdm1(self, fcivec, norb, nelec, **kwargs):
        '''dm1[p,q] = <p^+ q>'''
        state = _state_index(fcivec)
        return self._read(state, state, 'dm1')

    def make_rdm2(self, fcivec, norb, nelec, **kwargs):
        '''dm2[p,q,r,s] = <p^+ r^+ s q>'''
        state = _state_index(fcivec)
        return self._read(state, state, 'dm2')

    def make_rdm12(self, fcivec, norb, nelec, **kwargs):
        state = _state_index(fcivec)
        return (self._read(state, state, 'dm1'),
                self._read(state, state, 'dm2'))

    def trans_rdm1(self, cibra, ciket, norb, nelec, **kwargs):
        '''t[p,q] = <bra| p^+ q |ket>.

        zshci writes transition densities with the bra fixed at state 0, so
        one of the two states must be state 0.
        '''
        bra, ket = _state_index(cibra), _state_index(ciket)
        if bra == ket:
            return self._read(bra, bra, 'dm1')
        if bra == 0:
            return self._read(0, ket, 'dm1')
        if ket == 0:
            # <bra| p^+ q |0> = conj(<0| q^+ p |bra>)
            return self._read(0, bra, 'dm1').conj().T
        raise NotImplementedError(
            'transition densities between two excited states (%d, %d) are '
            'not written by zshci' % (bra, ket))

    def wavefunction(self):
        '''The determinants and CI vectors of the last calculation.

        Returns:
            occslst : list of tuples, the occupied spinors of each determinant
            ci : array (nroots, ndet), real or complex as the calculation
        '''
        with h5py.File(os.path.join(self.scratch, 'wfn.h5'), 'r') as f:
            words = f['dets'][()]
            ci = f['ci_real'][()]
            if 'ci_imag' in f:
                ci = ci + 1j * f['ci_imag'][()]
        occslst = []
        for row in words:
            occ = []
            for k, word in enumerate(row):
                word = int(word)
                while word:
                    low = word & -word
                    occ.append(64 * k + low.bit_length() - 1)
                    word ^= low
            occslst.append(tuple(occ))
        return occslst, ci


ZSHCI = SHCI
