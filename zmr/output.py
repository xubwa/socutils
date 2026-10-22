'''
Printed output of a NEVPT2 calculation, in the manner of ORCA's NEVPT2
section: settings and dimensions first, one line per step as it starts and
finishes, the second-order energies by class, the eigenvalues of the
effective Hamiltonian and the timings.

verbose = 0: nothing; 1 (default): everything, per-state tables only for up
to MAX_STATES model states; 2: per-state tables for every state.
'''

import os
import sys
import time

HARTREE2WAVENUMBER = 219474.6313632
HARTREE2EV = 27.211386245988
WIDTH = 78
MAX_STATES = 6


class Printer:
    # seconds between two heartbeat lines inside a step
    interval = 30.0

    def __init__(self, verbose=1, stdout=None):
        self.verbose = int(verbose)
        self.stdout = sys.stdout if stdout is None else stdout
        self._t0 = None
        self._open = False
        self._last = 0.0

    def __call__(self, text=''):
        if self.verbose:
            self.stdout.write(text + '\n')
            self.stdout.flush()

    def rule(self, char='-'):
        self(char * WIDTH)

    def title(self, text):
        self.rule()
        self(text.center(WIDTH))
        self.rule()

    def item(self, label, value):
        self(' %-36s ... %s' % (label, value))

    def start(self, label):
        '''First half of a progress line; done() completes it.'''
        self._t0 = self._last = time.perf_counter()
        if self.verbose:
            self.stdout.write(' %-36s ... ' % label)
            self.stdout.flush()
            self._open = True

    def done(self, note=''):
        dt = time.perf_counter() - self._t0
        lead = '' if self._open else ' %-36s ... ' % ''
        self._open = False
        self('%s%sdone in %8.1f sec' % (lead, note + ' ... ' if note else '', dt))
        return dt

    def progress(self, what, i, n):
        '''Heartbeat inside a step: item i of n of a loop (what: 'block',
        'node', 'state', ...).  Printed at most every `interval` seconds, so
        fast steps stay on one line.'''
        now = time.perf_counter()
        if not self.verbose or self._t0 is None or now - self._last < self.interval:
            return
        self._last = now
        if self._open:
            self.stdout.write('\n')
            self._open = False
        self('      ... %s %d of %d (%.0f%%), %.0f sec' % (what, i, n, 100.0 * i / max(n, 1),
                                                         now - self._t0))

    def stage(self, what):
        '''Heartbeat for a part of a step that is not a loop.'''
        now = time.perf_counter()
        if not self.verbose or self._t0 is None or now - self._last < self.interval:
            return
        self._last = now
        if self._open:
            self.stdout.write('\n')
            self._open = False
        self('      ... %s, %.0f sec' % (what, now - self._t0))


def threads():
    return os.environ.get('OMP_NUM_THREADS', 'not set')


def header(log, pt, act, eris, e_ref):
    sp = eris.spaces
    log()
    log.title('zmr  NEVPT2')
    log.item('Method', pt.method_label())
    log.item('Zeroth-order Hamiltonian', 'Dyall; effective Hamiltonian of the model space')
    log.item('Model states', len(e_ref))
    log.item('E0 in the denominators',
             'state specific' if pt.e0 == 'state' else 'common (weighted average)')
    log.item('Active electrons', pt.nelec)
    log.item('Correlated orbitals', '%d core, %d active, %d virtual'
             % (sp.ncore, sp.nact, sp.nvirt))
    dims = []
    for dn in (-2, -1, 0, 1, 2):
        try:
            dims.append('%d' % act.dim(act.target(pt.nelec, dn)))
        except Exception:
            dims.append('-')
    log.item('Determinants, sectors N-2 .. N+2', ' '.join(dims))
    for label, value in pt.settings():
        log.item(label, value)
    log.item('OMP_NUM_THREADS', threads())
    log()
    log(' Reference energies of the model states (diagonal of the model Hamiltonian):')
    shown = len(e_ref) if log.verbose > 1 else min(len(e_ref), 20)
    for i in range(shown):
        log('   STATE %3d:  E0 = %18.10f Eh  %12.2f cm-1'
            % (i, e_ref[i], (e_ref[i] - e_ref.min()) * HARTREE2WAVENUMBER))
    if shown < len(e_ref):
        log('   ... %d more states' % (len(e_ref) - shown))
    log()


def results(log, pt):
    names = list(pt.h2_by_class)
    nstate = pt.hmodel.shape[0]
    e_ref = pt.hmodel.diagonal().real
    log()
    log.rule('=')
    log('NEVPT2 Results'.center(WIDTH))
    log.rule('=')
    full = log.verbose > 1 or nstate <= MAX_STATES
    if full:
        for i in range(nstate):
            log('   STATE %d' % i)
            for name in names:
                log('  Class %-5s:  dE = %18.12f' % (name, pt.h2_by_class[name][i, i].real))
            de = pt.h2[i, i].real
            log('  ' + '-' * 44)
            log('  Total Energy Correction : dE = %18.12f' % de)
            log('  Reference Energy        : E0 = %18.12f' % e_ref[i])
            log('  Total Energy (E0+dE)    : E  = %18.12f' % (e_ref[i] + de))
            log()
    else:
        log(' Diagonal second-order energies (verbose = 2 prints them by class):')
        log('   STATE          E0 / Eh             dE / Eh          E0+dE / Eh')
        for i in range(min(nstate, 40)):
            de = pt.h2[i, i].real
            log('   %5d  %18.10f  %16.10f  %18.10f' % (i, e_ref[i], de, e_ref[i] + de))
        if nstate > 40:
            log('   ... %d more states' % (nstate - 40))
        log()
        log(' Second-order energy by class, averaged over the model states:')
        for name in names:
            log('  Class %-5s:  dE = %18.12f' % (name, pt.h2_by_class[name].diagonal().real.mean()))
        log()
    if nstate > 1:
        off = pt.h2 - __import__('numpy').diag(pt.h2.diagonal())
        log(' Largest off-diagonal second-order coupling : %.3e Eh' % abs(off).max())
        log()
    log(' Eigenvalues of the effective Hamiltonian:')
    log('   ROOT         E / Eh            cm-1            eV')
    e = pt.e_tot
    shown = len(e) if log.verbose > 1 else min(len(e), 400)
    for i in range(shown):
        d = e[i] - e[0]
        log('   %4d  %18.10f  %12.2f  %12.5f' % (i, e[i], d * HARTREE2WAVENUMBER, d * HARTREE2EV))
    if shown < len(e):
        log('   ... %d more roots' % (len(e) - shown))


def decay_results(log, pt):
    '''The CVS energies and the contributions of the decay channels left
    out by the CVS, evaluated with the CVS eigenvectors, and the energies
    with them included.'''
    import numpy
    e, u = numpy.linalg.eigh(pt.heff)
    by = pt.h2_decay_by_class
    disc = sum(by[n] for n in ('ccaa', 'caaa') if n in by)
    cont = by.get('ccav', 0)

    def expect(m):
        if numpy.isscalar(m):
            return numpy.zeros(len(e))
        return numpy.einsum('ki,kl,li->i', u.conj(), m, u).real
    qd, qc = expect(disc), expect(cont)
    log()
    log(' Core-valence separation: decay channels (not included above)')
    log('   Q discrete  : ccaa, caaa with electrons from outer shells (lower states, no electron emitted)')
    log('   Q continuum : ccav with both electrons from outer shells (one electron emitted)')
    if pt.decay_window is not None:
        tau, n, w = pt.decay_window
        log('   window      : |denominator| < %.4f Eh left out: %d terms, weight sum |u|^2 = %.3e'
            % (tau, n, w))
    log('   ROOT     E(CVS) / Eh     Q discrete    Q continuum   E(CVS+Q) / Eh    Q / eV')
    shown = len(e) if log.verbose > 1 else min(len(e), 40)
    for i in range(shown):
        log('   %4d  %16.10f  %12.8f  %12.8f  %16.10f  %8.4f'
            % (i, e[i], qd[i], qc[i], pt.e_tot_decay[i],
               (pt.e_tot_decay[i] - e[i]) * HARTREE2EV))
    if shown < len(e):
        log('   ... %d more roots' % (len(e) - shown))
    log('   (Q discrete, Q continuum: expectation values in the CVS eigenvectors;')
    log('    E(CVS+Q): eigenvalues of the effective Hamiltonian with all of Q)')


def warnings_summary(log, messages):
    if messages:
        log()
        log.rule('!')
        for m in messages:
            log(' ' + m)
        log.rule('!')


def timings(log, steps, total, name='NEVPT2'):
    log()
    log.rule()
    log(('TIMINGS %s' % name).center(WIDTH))
    log.rule()
    for label, t in steps:
        log('   %-40s ... %10.1f sec (%5.1f%%)' % (label, t, 100 * t / max(total, 1e-12)))
    log('   %-40s ... %10.1f sec' % ('Total', total))
    log()


# smallest energy denominator met in the step that is running (reported by
# the contraction engines, read and reset by the NEVPT2 driver)
_min_denominator = None
# below this (hartree) a class is reported as affected by an intruder state
INTRUDER_THRESHOLD = 0.01


def denominator(value):
    global _min_denominator
    if _min_denominator is None or value < _min_denominator:
        _min_denominator = float(value)


def take_denominator():
    global _min_denominator
    value, _min_denominator = _min_denominator, None
    return value


def intruder_message(name, value):
    if value <= 0:
        kind = 'non-positive'
    else:
        kind = 'small'
    return ('WARNING: class %s has a %s energy denominator (smallest %.4f Eh): intruder '
            'state.  The second-order energy of this class is not reliable.'
            % (name, kind, value))


# heartbeat of the step that is running; set by the NEVPT2 driver for the
# duration of kernel() and called from the loops of the contraction engines
_progress = None
_stage = None


def progress(what, i, n):
    if _progress is not None:
        _progress(what, i, n)


def stage(what):
    if _stage is not None:
        _stage(what)


# ----------------------------------------------------------------------
# MRCI

def mrci_header(log, ci, e_ref):
    sp = ci.eris0.spaces
    log()
    log.title('zmr  MRCI')
    scheme = ci.contraction if isinstance(ci.contraction, str) else 'user defined'
    names = {'fic': 'fully internally contracted', 'wk': 'doubly external blocks contracted '
             '(Werner-Knowles type)', 'none': 'uncontracted MRCISD'}
    log.item('Method', {'ci': 'MRCI (singles and doubles)', 'acpf': 'MR-ACPF',
                        'aqcc': 'MR-AQCC'}[ci.functional])
    log.item('Contraction', names.get(scheme, scheme))
    log.item('Internal block', 'all CAS determinants (reference relaxes)'
             if ci.internal == 'cas' else 'the model states')
    log.item('Model states', len(e_ref))
    log.item('Active electrons', ci.nelec)
    log.item('Correlated orbitals', '%d core, %d active, %d virtual'
             % (sp.ncore, sp.nact, sp.nvirt))
    log.item('Integrals', 'three-index (Cholesky / RI)'
             if getattr(ci.eris0, 'factorized', False) else 'four-index')
    if ci.active_eri is not None:
        log.item('Active two-electron integrals', 'replaced (active_eri)')
    log.item('Metric threshold of the contraction', '%.1e' % ci.lindep)
    log.item('Convergence (residual norm)', '%.1e' % ci.conv_tol)
    log.item('OMP_NUM_THREADS', threads())
    log()
    log(' Reference energies of the model states:')
    for i, e in enumerate(e_ref):
        log('   STATE %3d:  E0 = %18.10f Eh  %12.2f cm-1'
            % (i, e, (e - min(e_ref)) * HARTREE2WAVENUMBER))
    log()


def mrci_blocks(log, ci):
    log()
    log(' Blocks of the MRCI space (external labels x active functions):')
    log('   block   ext. labels   functions   of sector   dimension')
    for name in ci.basis.names:
        b = ci.basis.blocks[name]
        log('   %-5s  %12d  %10d  %10d  %10d'
            % (name, b.size // b.m, b.m, b.b.shape[0], b.size))
    log('   %-5s  %12s  %10s  %10s  %10d' % ('total', '', '', '', ci.basis.dim))
    log()


def mrci_iteration_header(log, label=''):
    log()
    log(' Davidson iterations%s' % (': ' + label if label else ''))
    log('   ITER  space       E(lowest) / Eh    max |dE| / Eh   max |res|     sec')


def mrci_results(log, ci):
    log()
    log.rule('=')
    log('MRCI Results'.center(WIDTH))
    log.rule('=')
    label = {'ci': 'MRCI', 'acpf': 'MR-ACPF', 'aqcc': 'MR-AQCC'}[ci.functional]
    for i, e in enumerate(ci.e_tot):
        log('   ROOT %d' % i)
        log('  %-28s: E = %18.10f' % ('Total energy (%s)' % label, e))
        log('  %-28s: E = %18.10f' % ('Rotated reference', ci.e_rotated_ref[i]))
        log('  %-28s: dE = %17.10f' % ('Correlation energy', e - ci.e_rotated_ref[i]))
        log('  %-28s: c0^2 = %10.6f' % ('Weight of the internal block', ci.weights[i]))
        w = ci.block_weights[i]
        log('  Weights of the other blocks : '
            + '  '.join('%s %.5f' % (n, w[n]) for n in ci.basis.names if n != 'cas'))
        if ci.functional == 'ci':
            log('  %-28s: E = %18.10f' % ('Davidson correction (+Q)', ci.e_davidson[i]))
            log('  %-28s: E = %18.10f' % ('renormalized (+Q)', ci.e_davidson_renormalized[i]))
        log()
    if not ci.converged:
        log(' WARNING: the Davidson iterations did not converge')
        log()
    variants = [(label, ci.e_tot)]
    if ci.functional == 'ci':
        variants += [('MRCI+Q (Davidson)', ci.e_davidson),
                     ('MRCI+Q (renormalized)', ci.e_davidson_renormalized)]
    for name, e in variants:
        e = sorted(e)
        log(' Energies, %s:' % name)
        log('   ROOT         E / Eh            cm-1            eV')
        for i, x in enumerate(e):
            d = x - e[0]
            log('   %4d  %18.10f  %12.2f  %12.5f'
                % (i, x, d * HARTREE2WAVENUMBER, d * HARTREE2EV))
        log()
