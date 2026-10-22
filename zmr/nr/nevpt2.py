'''
Non-relativistic multi-state UC- and PC-NEVPT2 (spin-restricted orbitals).

Arguments as the spinor drivers (zmr.uc.nevpt2, zmr.pc.nevpt2), with

    eris: zmr.nr.integrals (real spatial orbitals, closed-shell core)
    nelec: (n_alpha, n_beta) active electrons; all model states are in this
        sector (CI vectors in the determinant order of pyscf.fci)

The Fock matrix is the spin-free generalized Fock matrix of the averaged
spin-summed 1-RDM and the perturber functions are spin-free
(zmr.nr.classes), so the results do not depend on the spin projection of the
model states.  Model states of different spin can be mixed in one effective
Hamiltonian as long as they share (n_alpha, n_beta).
'''

import numpy

from socutils.zmr.lib import linalg as la
from socutils.zmr.nr import classes, rdm
from socutils.zmr.nr.active import ActiveSpace
from socutils.zmr.pc import nevpt2 as pc_nevpt2
from socutils.zmr.pc import spectral
from socutils.zmr.uc import nevpt2 as uc_nevpt2


class _SpinRestricted:

    # real model states stay real
    dtype = numpy.float64

    def make_active_space(self, eris):
        return ActiveSpace.from_eris(eris)

    def perturber_classes(self):
        return classes.ALL_CLASSES

    def pattern_functions(self, pclass, act, civecs):
        '''[(target sector, phi[J, P, :])] for the patterns of a class whose
        target sector exists.'''
        return [(target, numpy.ascontiguousarray(phi))
                for target, phi in pclass.functions(act, civecs.T, self.nelec)
                if act.dim(target) > 0]


class UCNEVPT2(_SpinRestricted, uc_nevpt2.UCNEVPT2):
    '''Fully uncontracted NEVPT2 for spin-restricted orbitals.'''

    def __init__(self, eris, nelec, *args, **kwargs):
        uc_nevpt2.UCNEVPT2.__init__(self, eris, tuple(nelec), *args, **kwargs)

    def class_matrix(self, pclass, blocks, act, civecs, shifts):
        nstate = civecs.shape[1]
        a = numpy.zeros((nstate, nstate), dtype=civecs.dtype)
        # the external strings of different spin patterns are orthogonal: every
        # pattern is an independent problem in its own sector
        for target, phi in self.pattern_functions(pclass, act, civecs):
            space = uc_nevpt2.invariant_span(act, target, phi)
            if space is None:
                sector = act.sector(target)
                space = (sector.vecs, sector.lam)
            a += self.contract(pclass.name, blocks, space[0], phi, space[1], shifts)
        return a


class _Stack:
    '''Direct sum of sectors (one per spin pattern).'''

    def __init__(self, act, targets):
        self.act = act
        self.targets = targets
        dims = [act.dim(t) for t in targets]
        self.offsets = numpy.concatenate([[0], numpy.cumsum(dims)])
        self.dim = int(self.offsets[-1])

    def slices(self):
        return [slice(self.offsets[k], self.offsets[k + 1]) for k in range(len(self.targets))]

    def apply_h(self, q):
        return numpy.concatenate([self.act.apply_h(t, q[sl])
                                  for t, sl in zip(self.targets, self.slices())])

    def sector(self):
        '''Eigen-decomposition of the direct sum, or None if a sector is
        matrix free.'''
        if not all(self.act.is_dense(t) for t in self.targets):
            return None
        lam = numpy.empty(self.dim)
        vecs = numpy.zeros((self.dim, self.dim))
        for t, sl in zip(self.targets, self.slices()):
            sec = self.act.sector(t)
            lam[sl] = sec.lam
            vecs[sl, sl] = sec.vecs
        self.lam, self.vecs = lam, vecs
        return self


class PCNEVPT2(_SpinRestricted, pc_nevpt2.PCNEVPT2):
    '''Spin-free partially contracted NEVPT2 for spin-restricted orbitals.

    amplitudes: how the contracted space of a class is built:
        'spectral'  from CI vectors: the functions are orthonormalized in
                    determinant space and H_act is applied to each of them
        'rdm'       from the metric and Koopmans matrices of the functions,
                    which are reduced quantities of the model states of
                    3-RDM cost; the 4-RDM terms go through intermediate
                    states (zmr.nr.rdm).  Real model states only.
        'auto'      (default) class by class: 'rdm' if the functions are
                    fewer than the determinants they live in, 'spectral'
                    otherwise (many states) and for complex model states
    Both give the same energies.'''

    def __init__(self, eris, nelec, *args, **kwargs):
        names = ('nstate', 'civecs', 'weights', 'e0', 'method', 'nquad', 'contraction',
                 'lindep', 'amplitudes')
        kwargs.update(zip(names, args))
        route = kwargs.pop('amplitudes', 'auto')
        if route not in ('spectral', 'rdm', 'auto'):
            raise ValueError('amplitudes must be "spectral", "rdm" or "auto"')
        pc_nevpt2.PCNEVPT2.__init__(self, eris, tuple(nelec), **kwargs)
        self.amplitudes = route
        if route == 'rdm' and numpy.iscomplexobj(self.civecs):
            raise NotImplementedError('amplitudes="rdm" needs real model states')
        self._reduced = None
        self.routes = {}

    def settings(self):
        out = pc_nevpt2.PCNEVPT2.settings(self)
        how = {'spectral': out[0][1],
               'rdm': 'reduced matrices of 3-RDM cost + intermediate states',
               'auto': 'reduced matrices or CI vectors, chosen for every class'}
        out[0] = (out[0][0], how[self.amplitudes])
        return out

    # amplitudes='rdm': form the Koopmans matrices completely and record
    # their asymmetry in rdm_info (a check of the formulas; costs half a
    # matrix product more per sector)
    rdm_check = False
    # largest spin-flip asymmetry of the CI vectors up to which the matrices
    # of the alpha and beta sectors are taken from each other (zmr.nr.rdm)
    rdm_flip_tol = 1e-9

    def use_rdm(self, pclass, act, nstate):
        '''Route of one class.  'auto': the reduced matrices have the
        dimension M of the functions of all states, the CI-vector route
        never more than the number D of determinants of the stacked target
        sectors; M < D decides.'''
        if self.amplitudes != 'auto':
            return self.amplitudes == 'rdm'
        if numpy.iscomplexobj(self.civecs):
            return False
        dim = sum(act.dim(act.target(self.nelec, p.dn)) for p in pclass.patterns(act.norb))
        return nstate * sum(pclass.nparams(act.norb)) < dim

    def reduced(self, act, civecs):
        '''Reduced matrices of the model states (zmr.nr.rdm.Reduced), cached
        for the classes of one kernel() call.'''
        if self._reduced is None or self._reduced[0] is not civecs:
            self._reduced = (civecs, rdm.Reduced(act, civecs, self.nelec, self.rdm_check,
                                                 self.rdm_flip_tol,
                                                 diagonal=self.contraction == 'state'))
        return self._reduced[1]

    def _kernel(self, log):
        self.rdm_info = None
        self.routes = {}
        try:
            return pc_nevpt2.PCNEVPT2._kernel(self, log)
        finally:
            if self._reduced is not None:
                red = self._reduced[1]
                # the vectors and intermediate states are not kept
                self.rdm_info = {'asymmetry': red.asymmetry if red.check else None,
                                 'max_basis': red.max_basis,
                                 'spin_flip': red.flip_parities() is not None,
                                 'flip_error': red.flip_error}
                self._reduced = None

    def _class_matrix_rdm(self, pclass, blocks, act, civecs, shifts):
        nstate = civecs.shape[1]
        a = numpy.zeros((nstate, nstate))
        matrices = self.reduced(act, civecs).class_matrices(pclass)
        if matrices is None:
            return a
        s, k = matrices
        nx = s.shape[1]
        if self.contraction == 'union':
            smat = s.reshape(nstate * nx, nstate * nx)
            c, mu = rdm.contracted_space(smat, k.reshape(smat.shape), self.lindep)
            self._class_info = (nstate * nx, len(mu))
            if len(mu) == 0:
                return a
            # t[I, j, X] = sum_{L,Y} c[(L,Y), j] S[L, Y, I, X]
            t = la.gemm(c, smat, 'T', 'N').reshape(len(mu), nstate, nx).transpose(1, 0, 2)
            return self.contract_t(pclass.name, blocks, t, mu, shifts)
        for j in range(nstate):
            c, mu = rdm.contracted_space(s[j, :, j, :], k[j, :, j, :], self.lindep)
            if len(mu) == 0:
                continue
            t = la.gemm(c, numpy.ascontiguousarray(s[j]).reshape(nx, nstate * nx), 'T', 'N')
            t = t.reshape(len(mu), nstate, nx).transpose(1, 0, 2)
            a[:, j] = self.contract_t(pclass.name, blocks, t, mu, shifts, col=j)[:, j]
        return a

    def class_matrix(self, pclass, blocks, act, civecs, shifts):
        nstate = civecs.shape[1]
        rdm_route = self.use_rdm(pclass, act, nstate)
        self.routes.setdefault(pclass.name, []).append('rdm' if rdm_route else 'spectral')
        if rdm_route:
            return self._class_matrix_rdm(pclass, blocks, act, civecs, shifts)
        a = numpy.zeros((nstate, nstate), dtype=civecs.dtype)
        parts = self.pattern_functions(pclass, act, civecs)
        if not parts:
            return a
        # spin-free functions: the pattern components stacked, H = direct sum
        stack = _Stack(act, [target for target, _ in parts])
        phi = numpy.ascontiguousarray(numpy.concatenate([f for _, f in parts], axis=2))
        nx = phi.shape[1]

        def space(rows):
            return spectral.contracted_space(rows, stack.apply_h, stack.sector, self.lindep)

        if self.contraction == 'union':
            b, mu = space(phi.reshape(nstate * nx, stack.dim))
            if len(mu) == 0:
                return a
            return self.contract(pclass.name, blocks, b, phi, mu, shifts)
        for j in range(nstate):
            b, mu = space(phi[j])
            if len(mu) == 0:
                continue
            a[:, j] = self.contract(pclass.name, blocks, b, phi, mu, shifts, col=j)[:, j]
        return a
