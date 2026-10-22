'''
Multi-state NEVPT2 drivers (conventions.md sections 7-9).

NEVPT2Base holds everything shared by the uncontracted and the contracted
variants: model states, the averaged Fock matrix and quasi-canonical
orbitals, the model-space Hamiltonian, the E0 shifts, the loop over the
perturber classes and the Hermitization.  A subclass only maps the
transition amplitudes and sector spectrum of one class to the matrix
a[I, J] = <I|V R(E0_J) V|J>.
'''

import time
import warnings

import numpy

from socutils.zmr import fock, output
from socutils.zmr import lib as zlib
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.lib import linalg as la
from socutils.zmr.uc import classes, engine


class NEVPT2Base:
    '''Common multi-state NEVPT2 driver with the Dyall Hamiltonian.

    Args:
        eris: integrals (any orbitals; core/virtual blocks are
            quasi-canonicalized here with the averaged Fock matrix)
        nelec: number of active electrons
        nstate: number of model states (lowest CASCI roots), or
        civecs: explicit model CI vectors as columns (overrides nstate)
        weights: state weights of the averaged 1-RDM and of the common E0
        e0: 'state' (default) or 'average'
        method: 'exact' or 'laplace', or a dict {class name: method}
        nquad: number of Laplace quadrature points

    After kernel():
        heff: (nstate, nstate) effective Hamiltonian
        h2_by_class: dict class -> (nstate, nstate) second-order contribution
        e_tot: eigenvalues of heff (also available as .energies)
        timings: wall time per class (including the first diagonalization of
            the sectors it needs)
    '''

    # Hermitize a even with a common E0 (needed when the first-order space
    # depends on the ket state)
    hermitize_always = False
    # CI vectors are complex unless a subclass works in real arithmetic and
    # the model states are real
    dtype = numpy.complex128
    # the model states come from the full spectrum of the N-electron sector
    # (needed anyway by the uncontracted classes) or from a partial solve
    full_reference_spectrum = True

    # printed output (zmr.output): 0 nothing, 1 settings, progress of every
    # step, energies by class, eigenvalues, timings; 2 also per-state tables
    # for many states.  stdout: file object to write to (default sys.stdout)
    verbose = 1
    stdout = None
    # seconds between two heartbeat lines inside a long step
    heartbeat = 30.0
    name = 'NEVPT2'

    def method_label(self):
        return self.name

    def settings(self):
        '''(label, value) pairs of method-specific settings for the output.'''
        out = [('Core-valence separation', 'on' if self.cvs else 'off')]
        if self.h0 != 'dyall':
            out.append(('Zeroth-order Hamiltonian', 'Fock operator (active block), '
                        'model states rotated to diagonalize it'))
        if self.skip_classes:
            out.append(('Classes left out', ' '.join(self.skip_classes)))
        return out
    # Core-hole model states (active orbitals below part of the core, e.g. a
    # 2p shell with holes): excitations that move core electrons into the
    # active orbitals are degenerate with the Auger continuum.  cvs = True
    # applies the core-valence separation: the classes that add electrons to
    # the active space (ccaa, ccav, caaa) are restricted to the core orbitals
    # below the active ones.  skip_classes names classes to leave out.
    #
    # The separation is by the number of holes: with the active shell holding
    # the holes and "inner" core orbitals deeper than it, a perturber keeps
    # the holes as deep as in the reference (P) if ccaa takes both electrons,
    # ccav at least one and caaa its electron from the inner core.  The
    # remaining labels (Q, the decay channels) are evaluated separately and
    # reported, not added: ccaa, caaa give discrete lower states, ccav the
    # Auger continuum (one electron in a virtual orbital), summed with the
    # symmetric window |denominator| >= cvs_window (principal value).
    # cvs_core: indices of the inner core orbitals among the quasi-canonical
    # correlated core orbitals (ascending energy); None: those below the
    # lowest active orbital energy.
    # Zeroth-order Hamiltonian in the active space: 'dyall' (H_act with its
    # two-electron part: NEVPT2) or 'fock' (the one-electron operator
    # sum_tu f_tu t^+ u with the generalized Fock matrix of the averaged
    # density).  With 'fock' the method is a CASPT2 whose H0 is the Fock
    # operator without its core-active, active-virtual and core-virtual
    # blocks, so that the classes stay decoupled; the model states are
    # rotated among themselves to diagonalize it (as in XMS-CASPT2), which
    # makes the result invariant under rotations of the model states.
    h0 = 'dyall'
    # h0 = 'fock' only.  False: the model states are kept as given and each
    # takes E0_J = <J|F|J>; the diagonal of heff is then the single-state
    # second-order energy of every model state (its off-diagonal elements
    # are those of a multi-state treatment without the rotation, which is
    # not invariant under rotations of the model states).
    fock_rotate = True
    cvs = False
    cvs_core = None
    cvs_decay = True
    cvs_window = 0.05
    skip_classes = ()

    @zlib.single_threaded_numpy
    def __init__(self, eris, nelec, nstate=1, civecs=None, weights=None,
                 e0='state', method='exact', nquad=None):
        self.eris0 = eris
        self.nelec = nelec
        self._act = None
        if civecs is None:
            _, civecs = self.active_space0().solve(nelec, nstate,
                                                   full=self.full_reference_spectrum)
        civecs = numpy.asarray(civecs)
        civecs = civecs.astype(numpy.result_type(civecs, self.dtype))
        self.civecs = civecs[:, None] if civecs.ndim == 1 else civecs
        nstate = self.civecs.shape[1]
        self.weights = (numpy.full(nstate, 1.0 / nstate) if weights is None
                        else numpy.asarray(weights, dtype=float))
        if e0 not in ('state', 'average'):
            raise ValueError('e0 must be "state" or "average"')
        self.e0 = e0
        self.method = method
        self.nquad = nquad

    def make_active_space(self, eris):
        return ActiveSpace.from_eris(eris)

    def perturber_classes(self):
        '''The perturber classes; several entries may share a name (the
        spin components of a class), their matrices are added.'''
        return classes.ALL_CLASSES

    def active_space0(self):
        '''Active-space Hamiltonian of the input orbitals.  It is unchanged by
        the core/virtual quasi-canonicalization (h_eff contains only the core
        trace), so it is built and diagonalized once and reused.'''
        if self._act is None:
            self._act = self.make_active_space(self.eris0)
        return self._act

    def _fock_zeroth_order(self, act, eris, qc, c):
        '''Active space with the one-electron zeroth-order Hamiltonian
        sum_tu f_tu t^+ u, the model states rotated to diagonalize it in the
        model space (block by block, so that symmetry labels are kept), the
        rotation and the zeroth-order energies.'''
        if type(act) is not ActiveSpace or getattr(self, 'amplitudes', None) == 'rdm':
            raise NotImplementedError('h0 = "fock" needs the spinor active space and the '
                                      'CI-vector route')
        sa = eris.spaces.slice('a')
        na = act.norb
        act_pt = ActiveSpace(qc.fock[sa, sa], numpy.zeros((na,) * 4), 0.0, charges=act.charges)
        fm = c.conj().T @ act_pt.apply_h(self.nelec, c)
        fm = 0.5 * (fm + fm.conj().T)
        fm[abs(fm) < 1e-10] = 0.0
        n = fm.shape[0]
        if not self.fock_rotate:
            return act_pt, c, numpy.eye(n, dtype=fm.dtype), fm.diagonal().real.copy()
        u = numpy.zeros((n, n), dtype=fm.dtype)
        w = numpy.zeros(n)
        todo = list(range(n))
        while todo:
            block, grow = {todo[0]}, [todo[0]]
            while grow:
                k = grow.pop()
                for j in numpy.nonzero(fm[k])[0]:
                    if j not in block:
                        block.add(int(j))
                        grow.append(int(j))
            idx = numpy.array(sorted(block))
            wb, ub = numpy.linalg.eigh(fm[numpy.ix_(idx, idx)])
            u[numpy.ix_(idx, idx)] = ub
            w[idx] = wb
            todo = [k for k in todo if k not in block]
        return act_pt, c @ u, u, w

    def _method_for(self, name):
        if isinstance(self.method, dict):
            return self.method.get(name, 'exact')
        return self.method

    def contract(self, name, blocks, basis, phi, lam, shifts, col=None):
        '''a[I, J] of one class from the functions phi[J, X, :] = O_X |Psi_J>
        and the first-order space spanned by the columns of basis with
        energies lam, by the method selected for this class.  col: only that
        column J is computed.'''
        method = self._method_for(name)
        if method == 'exact':
            return engine.exact_contract(blocks, basis, phi, lam, shifts, col=col)
        if method == 'laplace':
            return self.contract_t(name, blocks, engine.project_functions(basis, phi), lam,
                                   shifts, col)
        raise ValueError('unknown method %r' % method)

    def contract_t(self, name, blocks, t, lam, shifts, col=None):
        '''As contract for amplitudes t[J, k, X] that are already projected
        onto the first-order space.'''
        method = self._method_for(name)
        if method == 'exact':
            return engine.exact_contract_t(blocks, t, lam, shifts, col)
        if method == 'laplace':
            from socutils.zmr.uc import laplace_engine
            return laplace_engine.laplace_contract(blocks, t, lam, shifts, self.nquad,
                                                   name=name, col=col)
        raise ValueError('unknown method %r' % method)

    def class_matrix(self, pclass, blocks, act, civecs, shifts):
        '''a[I, J] of one class from its external blocks, the active space
        (quasi-canonical orbitals) and the model CI vectors.'''
        raise NotImplementedError

    @zlib.single_threaded_numpy
    def kernel(self):
        log = output.Printer(self.verbose, self.stdout)
        log.interval = self.heartbeat
        output._progress, output._stage = log.progress, log.stage
        try:
            return self._kernel(log)
        finally:
            output._progress = output._stage = None

    def _kernel(self, log):
        t_start = time.perf_counter()
        steps = []
        act = self.active_space0()
        c = self.civecs
        nstate = c.shape[1]
        hmodel_act = c.conj().T @ act.apply_h(self.nelec, c)
        output.header(log, self, act, self.eris0, hmodel_act.diagonal().real + act.e_core)
        log.start('Averaged density, Fock matrix')
        gamma = act.averaged_rdm1(self.civecs, self.nelec, self.weights)
        qc = fock.quasi_canonicalize(self.eris0, gamma)
        eris = qc.eris
        steps.append(('Fock matrix, quasi-canonical orbitals', log.done()))
        for label, eps in (('Core orbital energies', qc.eps_core),
                           ('Virtual orbital energies', qc.eps_virt)):
            if len(eps):
                log.item(label, '%.4f .. %.4f Eh' % (eps.min(), eps.max()))
        if hasattr(eris, 'prefetch'):
            # every block of the eight classes is (PQ|PQ), P in {a, v}, Q in {c, a}
            if getattr(eris, 'factorized', False):
                log.start('Three-index integrals cd_pa, cd_vc')
            else:
                log.start('Integral transformation (av ca|av ca)')
            eris.prefetch(('av', 'ca', 'av', 'ca'))
            steps.append(('Integral transformation', log.done()))
        e_core = act.e_core
        act_pt, e0_act = act, None
        if self.h0 == 'fock':
            act_pt, c, u, e0_act = self._fock_zeroth_order(act, eris, qc, c)
            hmodel_act = u.conj().T @ hmodel_act @ u
            self.model_rotation = u
            log.item('Zeroth-order energies of the model states',
                     '%.4f .. %.4f Eh' % (e0_act.min(), e0_act.max()))
        elif self.h0 != 'dyall':
            raise ValueError('h0 must be "dyall" or "fock"')
        fcore = fock.generalized_fock(eris, numpy.zeros_like(gamma))
        ctx = classes.Context(eris, qc.eps_core, qc.eps_virt, fcore)
        ctx_cvs = ctx_decay = ctx
        if self.cvs:
            sa = eris.spaces.slice('a')
            if self.cvs_core is None:
                inner = numpy.nonzero(qc.eps_core < qc.fock[sa, sa].diagonal().real.min())[0]
            else:
                inner = numpy.asarray(self.cvs_core, dtype=int)
            self.cvs_core = inner
            is_inner = numpy.zeros(eris.spaces.ncore, dtype=bool)
            is_inner[inner] = True

            def keep_p(name, ii, jj):
                if jj is None:
                    return is_inner[ii]
                n = is_inner[ii].astype(int) + is_inner[jj]
                return n == 2 if name == 'ccaa' else n >= 1
            ctx_cvs = classes.core_filter(ctx, keep_p)
            ctx_decay = classes.core_filter(ctx, lambda name, ii, jj: ~keep_p(name, ii, jj))
            log.item('Inner core orbitals (CVS)', '%d of %d' % (len(inner), eris.spaces.ncore))

        hmodel = hmodel_act + e_core * numpy.eye(nstate)
        self.hmodel_act = hmodel_act
        self.active_space = act
        e_diag = hmodel.diagonal().real
        # zeroth-order energies of the model states in the active space
        e0_states = e_diag - e_core if e0_act is None else e0_act
        if self.e0 == 'state':
            # degenerate states share their E0 exactly (one set of
            # denominators per level instead of one per state)
            shifts = _snap(e0_states, 1e-10)
        else:
            shifts = numpy.full(nstate, numpy.dot(self.weights, e0_states))

        h2_by_class = {}
        self.timings = {}
        self.min_denominators = {}
        intruders = []
        for pclass in self.perturber_classes():
            t0 = time.perf_counter()
            a = numpy.zeros((nstate, nstate), dtype=c.dtype)
            if pclass.name in self.skip_classes:
                continue
            log.start('Class %s' % pclass.name)
            self._class_info = None
            output.take_denominator()
            self._log = log
            blocks = engine.class_blocks(pclass, ctx_cvs if pclass.name in ('ccaa', 'ccav', 'caaa') else ctx)
            if blocks:
                a = self.class_matrix(pclass, blocks, act_pt, c, shifts)
            if self.e0 == 'state' or self.hermitize_always:
                a = 0.5 * (a + a.conj().T)
            h2_by_class[pclass.name] = h2_by_class.get(pclass.name, 0) + a
            self.timings[pclass.name] = (self.timings.get(pclass.name, 0.0)
                                         + time.perf_counter() - t0)
            nlabel = engine.nlabel(blocks)
            del blocks
            if hasattr(eris, 'release_blocks'):
                # the four-index blocks of this class are not needed again
                eris.release_blocks()
            note = '%d ext. labels' % nlabel
            if self._class_info:
                note += ', %d functions, %d kept' % self._class_info
            dmin = output.take_denominator()
            if dmin is not None:
                note += ', min. denominator %.4f' % dmin
                self.min_denominators[pclass.name] = min(
                    dmin, self.min_denominators.get(pclass.name, dmin))
            steps.append(('Class %s' % pclass.name, log.done(note)))
            if dmin is not None and dmin < output.INTRUDER_THRESHOLD:
                message = output.intruder_message(pclass.name, dmin)
                intruders.append(message)
                log(' ' + message)
                warnings.warn(message)

        self.h2_decay_by_class = {}
        self.decay_window = None
        if self.cvs and self.cvs_decay:
            self._decay_channels(log, ctx_decay, act_pt, c, shifts, steps)

        self.quasi_canonical = qc
        self.gamma = gamma
        self.shifts = shifts
        self.hmodel = hmodel
        self.h2_by_class = h2_by_class
        self.h2 = sum(h2_by_class.values())
        self.heff = hmodel + self.h2
        self.e_tot = self.energies = numpy.linalg.eigvalsh(self.heff)
        if self.h2_decay_by_class:
            self.h2_decay = sum(self.h2_decay_by_class.values())
            self.heff_decay = self.heff + self.h2_decay
            self.e_tot_decay = numpy.linalg.eigvalsh(self.heff_decay)
        output.results(log, self)
        if self.h2_decay_by_class:
            output.decay_results(log, self)
        output.warnings_summary(log, intruders)
        output.timings(log, steps, time.perf_counter() - t_start)
        return self.e_tot


def _decay_channels(self, log, ctx, act, c, shifts, steps):
    '''Second-order contributions of the labels left out by the CVS: ccaa
    and caaa (holes filled from outer shells, discrete states below the
    reference) and ccav (one electron emitted: the Auger continuum, summed
    outside the window |denominator| < cvs_window).'''
    for pclass in self.perturber_classes():
        name = pclass.name
        if name not in ('ccaa', 'caaa', 'ccav') or name in self.skip_classes:
            continue
        blocks = engine.class_blocks(pclass, ctx)
        if not blocks:
            continue
        log.start('Decay channels %s' % name)
        output.take_denominator()
        self._class_info = None
        engine.window_stats[:] = [0, 0.0]
        engine.WINDOW = self.cvs_window if name == 'ccav' else None
        try:
            a = self.class_matrix(pclass, blocks, act, c, shifts)
        finally:
            engine.WINDOW = None
        if self.e0 == 'state' or self.hermitize_always:
            a = 0.5 * (a + a.conj().T)
        self.h2_decay_by_class[name] = self.h2_decay_by_class.get(name, 0) + a
        dmin = output.take_denominator()
        note = '%d ext. labels' % engine.nlabel(blocks)
        if dmin is not None:
            note += ', min. denominator %.4f' % dmin
        if name == 'ccav':
            self.decay_window = (self.cvs_window, engine.window_stats[0],
                                 engine.window_stats[1])
            note += ', %d terms in the window' % engine.window_stats[0]
        steps.append(('Decay channels %s' % name, log.done(note)))


NEVPT2Base._decay_channels = _decay_channels


def _snap(values, tol):
    '''Replace values chained by gaps <= tol by their mean.'''
    values = numpy.array(values, dtype=float)
    order = numpy.argsort(values, kind='stable')
    cuts = numpy.nonzero(numpy.diff(values[order]) > tol)[0] + 1
    for idx in numpy.split(order, cuts):
        values[idx] = values[idx].mean()
    return values


def invariant_span(act, ntarget, phi, tol=1e-11):
    '''Exact reduction of the first-order space of an uncontracted class.

    If the span of the functions O_X |Psi_J> is invariant under H_act (e.g.
    O = 1 with eigenstate model vectors, the ccvv class), every right-hand
    side lies in it and the resolvent commutes with the projector, so the
    class can be evaluated in that span instead of the whole sector.

    Returns (basis, energies) of the span, or None if it is not invariant
    (or not smaller than the sector).'''
    from socutils.zmr.pc import spectral
    nstate, nx, dim = phi.shape
    if nstate * nx >= dim:
        return None
    q, _ = spectral.orthonormal_span(phi.reshape(nstate * nx, dim), lindep=1e-12)
    if q.shape[1] == 0:
        return q, numpy.zeros(0)
    hq = act.apply_h(ntarget, q)
    k = la.gemm(q, hq, 'C', 'N')
    if abs(hq - la.gemm(q, k)).max() > tol * max(1.0, abs(k).max()):
        return None
    mu, w = la.eigh(k)
    return la.gemm(q, w), mu


class UCNEVPT2(NEVPT2Base):
    name = 'UC-NEVPT2 (uncontracted in the active space)'
    '''Fully uncontracted NEVPT2: the complete sector spectrum is the
    first-order space of every external label.'''

    def class_matrix(self, pclass, blocks, act, civecs, shifts):
        phi = engine.class_functions(pclass, act, civecs, self.nelec)
        if phi is None:
            nstate = civecs.shape[1]
            return numpy.zeros((nstate, nstate), dtype=civecs.dtype)
        ntarget = act.target(self.nelec, pclass.dn)
        space = invariant_span(act, ntarget, phi)
        if space is None:
            sector = act.sector(ntarget)
            space = (sector.vecs, sector.lam)
        return self.contract(pclass.name, blocks, space[0], phi, space[1], shifts)
