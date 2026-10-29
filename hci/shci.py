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
        restart : bool
            If True, every call of kernel after the first starts from the
            determinants and CI vectors of the previous call (same norb,
            nelec and nroots) instead of initial_dets, and uses only the
            last epsilon1 of the schedule.  Meant for orbital optimization,
            where the Hamiltonian changes little between calls.  The space
            then only grows from call to call: determinants selected at
            earlier orbitals stay.
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
        self.restart = False
        self._restart_key = None
        self.orbsym = None
        self.orbsym_mod = 0
        self.restrictions = []
        self.pt = None
        self.extra = {}
        self.integral_tol = 1e-14
        # 'hdf5': the two-electron integrals go to the program
        # antisymmetrized in a binary file (fcidump_rel.from_integrals_h5);
        # 'fcidump': as a text FCIDUMP file, far slower to write and read.
        self.integral_format = 'hdf5'
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
                dets = self._default_dets(norb, nelec)
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
                dets = self._all_dets(norb, nelec)
            else:
                dets = self.occslst
        inp['dets'] = [[int(p) for p in det] for det in dets]
        symmetry = self._symmetry_input(norb)
        if symmetry is not None:
            inp['symmetry'] = symmetry
        if self.restrictions:
            inp['restrictions'] = self.restrictions
        if self.pt:
            inp['pt'] = self.pt
        inp.update(self.extra)
        return inp

    def _default_dets(self, norb, nelec):
        '''The starting determinants when none are given.'''
        return [list(range(nelec))]

    def _all_dets(self, norb, nelec):
        '''Every determinant of the space, for occslst = 'all'.'''
        return itertools.combinations(range(norb), nelec)

    def _symmetry_input(self, norb):
        '''The "symmetry" entry of the input, or None.'''
        if self.orbsym is not None:
            sets = self.orbsym
            if numpy.ndim(sets) == 1:
                sets = [sets]
            mods = numpy.atleast_1d(self.orbsym_mod)
            if len(mods) == 1:
                mods = [mods[0]] * len(sets)
            if len(mods) != len(sets):
                raise ValueError('orbsym_mod needs one entry per label set')
            return [{'labels': [int(x) for x in labels], 'mod': int(mod)}
                    for labels, mod in zip(sets, mods)]
        return None

    def _write_integrals(self, inp, h1e, eri, norb, nelec, ecore):
        '''Writes the integral file named in the input `inp`, which may be
        amended.'''
        if self.scalar not in ('auto', 'real', 'complex'):
            raise ValueError("scalar must be 'auto', 'real' or 'complex'")
        if self.scalar == 'auto':
            real = numpy.isrealobj(h1e) and numpy.isrealobj(eri)
            inp.setdefault('scalar', 'real' if real else 'complex')
        else:
            inp.setdefault('scalar', self.scalar)
        if self.integral_format not in ('hdf5', 'fcidump'):
            raise ValueError("integral_format must be 'hdf5' or 'fcidump'")
        if self.integral_format == 'hdf5':
            inp['integrals'] = os.path.join(self.scratch, 'integrals.h5')
        fcidump_rel.from_integrals(
            inp['integrals'], numpy.asarray(h1e, dtype=complex),
            numpy.asarray(eri, dtype=complex).reshape(norb * norb, norb * norb),
            norb, nelec, nuc=complex(ecore).real, tol=self.integral_tol)

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
        self._write_integrals(inp, h1e, eri, norb, nelec, ecore)

        # Restart from the wave function of the previous call.
        key = (norb, repr(nelec), nroots)
        previous = os.path.join(self.scratch, 'wfn.h5')
        restart_file = os.path.join(self.scratch, 'restart.h5')
        if (self.restart and inp['mode'] == 'select'
                and self._restart_key == key and os.path.isfile(previous)):
            os.replace(previous, restart_file)
            del inp['dets']
            inp['detfile'] = restart_file
            inp['select']['schedule'] = [[0, float(self.schedule[-1][1])]]
            log.debug('zshci restarts from the previous wave function')
        self._restart_key = None

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
        self._restart_key = key
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


class SpatialSHCI(SHCI):
    '''Selected CI for a non-relativistic Hamiltonian over spatial orbitals
    (a restricted reference), with the interface and the spin-traced density
    matrices of the PySCF FCI solvers (pyscf.fci.direct_spin1)::

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.fcisolver = shci.SpatialSHCI(mol)
        mc.fcisolver.schedule = [[0, 1e-4]]
        mc.kernel()

    h1e and eri are real and over norb spatial orbitals; nelec is
    (nalpha, nbeta) or their sum together with the attribute spin
    (nalpha - nbeta).  The program works with the spin orbitals 2P (alpha) and
    2P + 1 (beta) of spatial orbital P, and determinants (initial_dets,
    occslst, wavefunction()) are lists of those.  The spin projection is
    conserved; the total spin is not imposed, so with several roots states of
    different spin come out together, as in direct_spin1.

    Attributes in addition to those of SHCI:
        spin : int or None
            nalpha - nbeta when nelec is given as one number (default 0 or 1).
        orbsym : list of int, or None
            Irreducible representation of every orbital as PySCF numbers
            those of D2h and its subgroups (0 to 7, products being exclusive
            ors).  Other groups (linear molecules) are not understood; leave
            orbsym unset for them.
    '''

    def __init__(self, mol=None):
        super().__init__(mol)
        self.spin = None
        self.wfnsym = None

    def _unpack_nelec(self, nelec):
        if isinstance(nelec, (int, numpy.integer)):
            spin = self.spin if self.spin is not None else nelec % 2
            return (nelec + spin) // 2, (nelec - spin) // 2
        return int(nelec[0]), int(nelec[1])

    def _input(self, norb, nelec, nroots):
        nalpha, nbeta = self._unpack_nelec(nelec)
        self._nelec_ab = (nalpha, nbeta)
        inp = super()._input(norb, nalpha + nbeta, nroots)
        inp['orbitals'] = 'spatial'
        return inp

    def _default_dets(self, norb, nelec):
        nalpha, nbeta = self._nelec_ab
        return [[2 * p for p in range(nalpha)] +
                [2 * p + 1 for p in range(nbeta)]]

    def _all_dets(self, norb, nelec):
        nalpha, nbeta = self._nelec_ab
        return [[2 * p for p in alpha] + [2 * p + 1 for p in beta]
                for alpha in itertools.combinations(range(norb), nalpha)
                for beta in itertools.combinations(range(norb), nbeta)]

    def _symmetry_input(self, norb):
        if self.orbsym is None or len(self.orbsym) == 0:
            return []
        orbsym = numpy.asarray(self.orbsym, dtype=int)
        if orbsym.size != norb or orbsym.min() < 0 or orbsym.max() > 7:
            raise ValueError('orbsym must give one irreducible representation '
                             'of D2h or a subgroup (0 to 7) per orbital')
        sets = []
        for bit in range(3):
            labels = (orbsym >> bit) & 1
            if labels.any():
                sets.append({'labels': [int(x) for x in labels], 'mod': 2})
        return sets

    def _write_integrals(self, inp, h1e, eri, norb, nelec, ecore):
        from pyscf import ao2mo
        from pyscf.tools import fcidump
        if numpy.iscomplexobj(h1e) or numpy.iscomplexobj(eri):
            raise ValueError('integrals over spatial orbitals must be real')
        nalpha, nbeta = self._nelec_ab
        fcidump.from_integrals(
            inp['integrals'], numpy.asarray(h1e), ao2mo.restore(8, eri, norb),
            norb, nalpha + nbeta, nuc=float(numpy.real(ecore)),
            ms=nalpha - nbeta, tol=self.integral_tol)

    def trans_rdm1(self, cibra, ciket, norb, nelec, **kwargs):
        '''t[p,q] = <bra| q^+ p |ket>, the convention of
        pyscf.fci.direct_spin1.trans_rdm1 (the transpose of SHCI.trans_rdm1).
        '''
        return super().trans_rdm1(cibra, ciket, norb, nelec).T

    def spin_square(self, fcivec, norb, nelec):
        raise NotImplementedError('zshci does not evaluate <S^2>')


class UnrestrictedSHCI(SpatialSHCI):
    '''Selected CI on an unrestricted reference: different spatial orbitals
    for the two spins.  Interface and conventions of pyscf.fci.direct_uhf:

        h1e = (h1_a, h1_b)
        eri = (eri_aa, eri_ab, eri_bb),  eri_ab[p,q,r,s] = (p_a q_a|r_b s_b)
        nelec = (nalpha, nbeta)

    The program holds the three blocks as given (n^4/2 numbers, four times
    what a restricted calculation of the same size takes), antisymmetrizes
    them as needed, and returns density matrices over the spin orbitals.
    orbsym (D2h-subgroup numbers) must then be the same for the alpha and the
    beta orbital of each index.
    '''

    def _input(self, norb, nelec, nroots):
        inp = super()._input(norb, nelec, nroots)
        inp['orbitals'] = 'unrestricted'
        inp['integrals'] = os.path.join(self.scratch, 'integrals.h5')
        return inp

    def _write_integrals(self, inp, h1e, eri, norb, nelec, ecore):
        from pyscf import ao2mo

        def full(x, mixed=False):
            x = numpy.asarray(x)
            if x.ndim == 4:
                return x
            if mixed:
                # (npair, npair) with fourfold symmetry.
                x = lib.unpack_tril(x.reshape(-1, norb * (norb + 1) // 2))
                x = lib.unpack_tril(x.reshape(-1, norb * norb).T.copy())
                return x.reshape(norb, norb, norb, norb).transpose(2, 3, 0, 1)
            return ao2mo.restore(1, x, norb)

        if any(numpy.iscomplexobj(x) for x in tuple(h1e) + tuple(eri)):
            raise ValueError('integrals of an unrestricted reference must be '
                             'real')
        nalpha, nbeta = self._nelec_ab
        with h5py.File(inp['integrals'], 'w') as f:
            f['h1_a'] = numpy.asarray(h1e[0], dtype=float)
            f['h1_b'] = numpy.asarray(h1e[1], dtype=float)
            f['eri_aa'] = full(eri[0])
            f['eri_ab'] = full(eri[1], mixed=True)
            f['eri_bb'] = full(eri[2])
            f.attrs['ecore'] = float(numpy.real(ecore))
            f.attrs['nelec'] = numpy.int32(nalpha + nbeta)

    def make_rdm1s(self, fcivec, norb, nelec, **kwargs):
        '''(dm1a, dm1b), dm1a[p,q] = <p_a^+ q_a>.'''
        state = _state_index(fcivec)
        dm1 = self._read(state, state, 'dm1')
        return dm1[0::2, 0::2], dm1[1::2, 1::2]

    def make_rdm12s(self, fcivec, norb, nelec, **kwargs):
        '''((dm1a, dm1b), (dm2aa, dm2ab, dm2bb)) as pyscf.fci.direct_uhf:
        dm2ab[p,q,r,s] = <p_a^+ r_b^+ s_b q_a>.'''
        state = _state_index(fcivec)
        dm2 = self._read(state, state, 'dm2')
        return (self.make_rdm1s(fcivec, norb, nelec),
                (dm2[0::2, 0::2, 0::2, 0::2], dm2[0::2, 0::2, 1::2, 1::2],
                 dm2[1::2, 1::2, 1::2, 1::2]))

    def make_rdm1(self, fcivec, norb, nelec, **kwargs):
        dm1a, dm1b = self.make_rdm1s(fcivec, norb, nelec)
        return dm1a + dm1b

    def make_rdm2(self, fcivec, norb, nelec, **kwargs):
        return self.make_rdm12(fcivec, norb, nelec)[1]

    def make_rdm12(self, fcivec, norb, nelec, **kwargs):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = self.make_rdm12s(fcivec, norb,
                                                               nelec)
        return (dm1a + dm1b,
                dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb)

    def trans_rdm1(self, cibra, ciket, norb, nelec, **kwargs):
        t = super().trans_rdm1(cibra, ciket, norb, nelec)
        return t[0::2, 0::2] + t[1::2, 1::2]

