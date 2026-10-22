'''
Internally contracted MRCI (singles and doubles) for spinor Hamiltonians.

    ci = MRCI(eris, nelec, civecs=c)        # or nstate=
    e = ci.kernel()                         # one root per model state

contraction: 'fic' (fully internally contracted, default), 'wk' (only the
doubly external blocks contracted), 'none' (uncontracted MRCISD) or a dict
block -> 'pc' | 'full'; internal: 'cas' (reference coefficients relaxed,
default) or 'states'.

functional: 'ci' (default), or the approximately size-extensive 'acpf' /
'aqcc': the stationary points of

    (E - E_ref) = <Psi| H - E_ref |Psi> / (<Psi_int|Psi_int> + g <Psi_ext|Psi_ext>)

with g = 2 / n (ACPF) or 1 - (n - 3)(n - 2) / (n (n - 1)) (AQCC) for n
correlated electrons, Psi_int the part in the internal block and E_ref the
CASCI energy of the model state; one eigenproblem per state.

After kernel(): e_tot, e_ref, weights (of the internal block), e_davidson
(E + (1 - c0^2)(E - E_ref) and the renormalized form), ci (coefficients).
'''

import resource
import time

import numpy

from socutils.zmr import fock, output
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.mrci import davidson
from socutils.zmr.mrci.basis import ProductBasis
from socutils.zmr.mrci.sigma import Sigma


def _peak_gb():
    '''Largest resident size of the process so far, GB.'''
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


class MRCI:
    # spaces up to this dimension are diagonalized densely
    dense_limit = 600
    conv_tol = 5e-5           # residual norm of every root (the default of Dice)
    max_cycle = 100
    # trial vectors per root before the Davidson subspace is restarted
    max_space = 12
    # printed output: 0 nothing, 1 settings, blocks, steps, iterations, results
    verbose = 1
    stdout = None
    timing_lines = 6          # sigma contributions listed in the timings
    heartbeat = 30.0
    # roots by maximum overlap with the model states instead of the lowest ones
    follow = False
    # bytes of Davidson trial vectors kept in memory; more go to mapped scratch
    # files (ZMR_TMPDIR or PySCF's TMPDIR)
    davidson_incore = 4e10
    follow_floor = 0.02

    def __init__(self, eris, nelec, civecs=None, nstate=None, contraction='fic',
                 internal='cas', functional='ci', lindep=1e-10, active_eri=None):
        # active_eri: (tu|vw) that replace those of eris in the all-active
        # part of H (e.g. four-component integrals for the active spinors)
        self.active_eri = active_eri
        self.eris0 = eris
        self.nelec = nelec
        self.contraction = contraction
        self.internal = internal
        self.functional = functional
        self.lindep = lindep
        act = self._active_space(eris)
        if civecs is None:
            civecs = act.solve(nelec, nstate or 1)[1]
        civecs = numpy.asarray(civecs)
        self.civecs = civecs[:, None] if civecs.ndim == 1 else civecs
        self._act0 = act

    def _active_space(self, eris):
        if self.active_eri is None:
            return ActiveSpace.from_eris(eris)
        heff, _, e_core = eris.active_hamiltonian()
        return ActiveSpace(heff, self.active_eri, e_core,
                           charges=getattr(eris, 'active_charges', None))

    def build(self, log=None):
        '''Quasi-canonical core and virtual orbitals (they make the
        zeroth-order energies a good preconditioner; the MRCI energies do
        not depend on them), the block bases and the sigma evaluator.'''
        log = log or output.Printer(0)
        steps = self._steps = []
        log.start('Averaged density, Fock matrix')
        gamma = self._act0.averaged_rdm1(self.civecs, self.nelec)
        qc = fock.quasi_canonicalize(self.eris0, gamma)
        self.eris = qc.eris
        self.eps_core, self.eps_virt = qc.eps_core, qc.eps_virt
        steps.append(('Fock matrix, quasi-canonical orbitals', log.done()))
        if hasattr(self.eris, 'prefetch') and getattr(self.eris, 'factorized', False):
            log.start('Three-index integrals')
            self.eris.prefetch(('av', 'ca', 'av', 'ca'))
            steps.append(('Three-index integrals', log.done()))
        log.start('Block bases')
        self.act = self._active_space(self.eris)
        self.basis = ProductBasis(self.eris, self.act, self.nelec, self.civecs,
                                  self.contraction, self.internal, self.lindep)
        steps.append(('Block bases', log.done('dimension %d' % self.basis.dim)))
        output.mrci_blocks(log, self)
        self.sigma = Sigma(self.eris, self.act, self.basis, self.active_eri)
        self.hdiag = self.basis.diagonal(self.eps_core, self.eps_virt, self.act.e_core)
        hc = self.civecs.conj().T @ self.act.apply_h(self.nelec, self.civecs)
        self.e_ref = hc.diagonal().real + self.act.e_core
        return self

    def _prepare(self, log):
        '''Contractions of all block pairs, bra block by bra block.'''
        t0 = time.perf_counter()
        for bra in self.basis.names:
            log.start('Contractions for block %s' % bra)
            n = 0
            for k, ket in enumerate(self.basis.names):
                full, staged = self.sigma.compiled(bra, ket)
                n += len(full) + sum(len(c) for _, c in staged)
                log.progress('ket block', k + 1, len(self.basis.names))
            log.done('%d contractions, peak memory %.1f GB' % (n, _peak_gb()))
        self.sigma.compile_all()
        self._steps.append(('Dressed integrals, contractions', time.perf_counter() - t0))
        kept = sum(b for _, b in self.sigma.memory_report())
        log.item('Arrays kept between iterations', '%.2f GB' % (kept / 1e9))

    def hamiltonian(self, chunk=256):
        '''Dense Hamiltonian in the product basis (small spaces only).'''
        n = self.basis.dim
        h = numpy.empty((n, n), dtype=numpy.complex128)
        eye = numpy.eye(n)
        for k0 in range(0, n, chunk):
            h[:, k0:k0 + chunk] = self.sigma.matvec(eye[k0:k0 + chunk]).T
        return h

    def reference_vectors(self):
        '''The model states in the product basis: (nstate, dim).'''
        cas = self.basis.blocks['cas']
        x = numpy.zeros((self.civecs.shape[1], self.basis.dim), dtype=numpy.complex128)
        x[:, :cas.size] = (cas.b.conj().T @ self.civecs).T
        return x

    def _metric(self):
        n = self.eris.spaces.ncore + self.nelec        # correlated electrons
        g = {'ci': 1.0, 'acpf': 2.0 / n,
             'aqcc': 1.0 - (n - 3.0) * (n - 2.0) / (n * (n - 1.0))}[self.functional]
        m = numpy.full(self.basis.dim, g)
        m[:self.basis.blocks['cas'].size] = 1.0
        return m

    def _solve(self, shift, scale, nroots, x0, log=None, label=''):
        '''Lowest eigenpairs of scale (H - shift) scale.'''
        log = log or output.Printer(0)
        dim = self.basis.dim
        t0 = time.perf_counter()
        if dim <= self.dense_limit:
            log.start('Dense Hamiltonian, diagonalization')
            h = self.hamiltonian()
            self.hermiticity = abs(h - h.conj().T).max()
            h = 0.5 * (h + h.conj().T) - shift * numpy.eye(dim)
            e, c = numpy.linalg.eigh(scale[:, None] * h * scale[None, :])
            self._steps.append(('Dense diagonalization' + label, log.done()))
            return e[:nroots], c[:, :nroots].T

        def aop(xs):
            xs = numpy.asarray(xs) * scale
            return (self.sigma.matvec(xs) - shift * xs) * scale

        diag = (self.hdiag - shift) * scale**2
        # followed roots are interior ones: functions of other symmetries lie
        # below them, and a small denominator would amplify the components
        # that the numerical symmetry breaking puts on them
        floor = self.follow_floor if self.follow else 1e-6

        def precond(r, e0):
            d = diag - e0
            d[abs(d) < floor] = floor
            return r / d

        output.mrci_iteration_header(log, label.strip(', '))
        state = {'t': time.perf_counter(), 'n': 0}

        def callback(icyc, space, e, de, norms, seconds):
            now = time.perf_counter()
            log('   %4d  %5d  %18.10f  %14.2e  %10.2e  %7.1f'
                % (icyc, space, e.min() + shift, de.max() if numpy.isfinite(de).all() else
                   numpy.nan, norms.max(), now - state['t']))
            state.update(t=now, n=icyc + 1)

        conv, e, c = davidson.davidson(aop, list(x0), precond, nroots, tol=self.conv_tol,
                                       max_cycle=self.max_cycle, max_space=self.max_space,
                                       callback=callback, follow=self.follow,
                                       incore=self.davidson_incore)
        self.converged = self.converged and bool(conv)
        log('   %s after %d iterations, peak memory %.1f GB%s'
            % ('converged' if conv else 'NOT CONVERGED', state['n'], _peak_gb(),
               ' (trial vectors in scratch files)' if davidson.last.get('on_disk') else ''))
        self._steps.append(('Davidson iterations' + label, time.perf_counter() - t0))
        d = davidson.last
        self._steps.append(('  H on the trial vectors', d['aop']))
        self._steps.append(('  Ritz vectors, residuals', d['ritz']))
        self._steps.append(('  preconditioner', d['precond']))
        self._steps.append(('  orthogonalization', d['orth']))
        return numpy.asarray(e), numpy.asarray(c)

    def kernel(self, nroots=None):
        log = output.Printer(self.verbose, self.stdout)
        log.interval = self.heartbeat
        t_start = time.perf_counter()
        nstate = self.civecs.shape[1]
        nroots = nroots or nstate
        hc = self.civecs.conj().T @ self._act0.apply_h(self.nelec, self.civecs)
        output.mrci_header(log, self, hc.diagonal().real + self._act0.e_core)
        self.build(log)
        # every sigma vector that takes longer than the heartbeat reports its time, the
        # peak memory and its slowest contractions with their steps
        self.sigma.progress, self.sigma.report_seconds = log, self.heartbeat
        if self.basis.dim > self.dense_limit:
            self._prepare(log)
        ref = self.reference_vectors()
        self.converged = True
        if self.functional == 'ci':
            x0 = list(ref[:nroots])
            if nroots > nstate:
                # further guesses: the lowest zeroth-order functions
                order = numpy.argsort(self.hdiag)
                for k in order:
                    if len(x0) == nroots:
                        break
                    v = numpy.zeros(self.basis.dim, dtype=numpy.complex128)
                    v[k] = 1.0
                    if max(abs(numpy.vdot(x, v)) for x in x0) < 0.9:
                        x0.append(v)
            e, c = self._solve(0.0, numpy.ones(self.basis.dim), nroots, x0, log)
        else:
            if nroots != nstate:
                raise ValueError('acpf / aqcc give one root per model state')
            scale = 1.0 / numpy.sqrt(self._metric())
            e, c = [], []
            for j in range(nstate):
                de, y = self._solve(self.e_ref[j], scale, j + 1, list(ref[:j + 1] / scale),
                                    log, ', state %d' % j)
                x = y * scale
                x /= numpy.linalg.norm(x, axis=1)[:, None]
                k = int(numpy.argmax(abs(x.conj() @ ref[j])))
                e.append(self.e_ref[j] + de[k])
                c.append(x[k])
            e, c = numpy.asarray(e), numpy.asarray(c)
        self.e_tot = numpy.asarray(e).real
        self.ci = numpy.asarray(c)
        self._analyze()
        output.mrci_results(log, self)
        steps = list(self._steps)
        steps.append(('  sigma itself, %d vectors' % self.sigma.calls, self.sigma.wall))
        steps.append(('  sigma, sum of the listed contractions', sum(self.sigma.timings.values())))
        top = sorted(self.sigma.timings.items(), key=lambda x: -x[1])[:self.timing_lines]
        for k, v in top:
            steps.append(('  sigma %s <- %s %s' % (k[0], k[1], k[2].lstrip('_')), v))
            if self.timing_lines > 6:
                # steps of the contraction
                rows = sorted(((x, name) for (key, name), x in self.sigma.steps.items()
                               if key == k), reverse=True)
                steps += [('      %s' % name, x) for x, name in rows if x >= 0.005 * max(v, 1e-9)]
        output.timings(log, steps, time.perf_counter() - t_start, 'MRCI')
        if self.timing_lines > 6 and self.sigma.peak_growth:
            log('\n Growth of the peak memory during sigma, GB:')
            for k, v in sorted(self.sigma.peak_growth.items(), key=lambda x: -x[1])[:12]:
                log('   %s <- %s %-16s %6.1f' % (k[0], k[1], k[2].lstrip('_'), v))
        return self.e_tot

    def _analyze(self):
        '''Weight and energy of the internal part of every root; Davidson
        corrections with the rotated reference.'''
        cas = self.basis.blocks['cas']
        v = (self.ci[:, :cas.size] / numpy.linalg.norm(self.ci, axis=1)[:, None]) @ cas.b.T
        w = (abs(v)**2).sum(axis=1)
        hv = self.act.apply_h(self.nelec, numpy.ascontiguousarray(v.T))
        e_rot = (v.conj() * hv.T).sum(axis=1).real / w + self.act.e_core
        self.weights = w
        norm = numpy.linalg.norm(self.ci, axis=1)**2
        self.block_weights = [
            {n: float((abs(root[self.basis.offsets[k]:self.basis.offsets[k + 1]])**2).sum() / nrm)
             for k, n in enumerate(self.basis.names)} for root, nrm in zip(self.ci, norm)]
        self.e_rotated_ref = e_rot
        corr = self.e_tot - e_rot
        self.e_davidson = self.e_tot + (1 - w) * corr
        self.e_davidson_renormalized = self.e_tot + (1 - w) / w * corr
