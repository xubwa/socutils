'''
Multi-state NEVPT4(SD): Dyall partitioning to fourth order in the energy
with the first- and second-order wave functions in the singles and doubles
of the CAS (the first-order interacting space), docs/derivations/nevpt4-sd.md.

Multi-partitioning Bloch expansion (Zaitsevskii-Malrieu): column J of the
effective Hamiltonian uses the resolvent R_J = Q (E0_J - H_D)^-1 Q, Q the
complement of the model space P,

    x_J = R_J Q V |J>
    y_J = R_J Q [V x_J - sum_K x_K H1_KJ]
    z_J = R_J Q [V y_J - sum_K y_K H1_KJ (- sum_K x_K H2_KJ)]
    H_eff = H_PP + H2 + H3 + H4,   Hn_IJ = <I| V |x_J, y_J, z_J>,

H1_KJ = <K|V|J> = <K|H|J> - E0_J delta_KJ (H0 = sum_J E0_J |J><J| on the
model space, Q H_D Q outside; diagonal for CAS eigenstates), then Hermitized,
1/2 (H_eff + H_eff^+).  The renormalization term in parentheses is left out
by default: for one state it is -E2 <psi1|psi1>, whose omission makes the
fourth order size consistent when the triples and quadruples of the
second-order wave function are left out (Grimme 2001; Kempfer, Sivalingam,
Neese, JCTC 2025).  For one state the orders are the Rayleigh-Schroedinger
energies E2 = <V x>, E3 = <x|V - E1|x>, E4 = <w|R|w>, w = Q (V - E1) x.

    e0='state'    E0_J = <J|H_D|J> (QD-NEVPT2 at second order)
    e0='average'  E0 = sum_J w_J <J|H_D|J> for every column

    contraction='union'  (default) the PC functions of all reference
                         functions, one space for every column
                'none'   the uncontracted singles and doubles
                'state'  the first-order space of column J: the PC functions
                         of |J> and of the reference functions degenerate
                         with it (QD-NEVPT2); CAS eigenstates as reference
                         functions (H1 diagonal), no renormalization term.
                         At fourth order the second-order wave function of
                         each column is truncated to its own first-order
                         space, and the loss differs between reference
                         functions of different structure (O2 X,
                         PVDZ-X2C-TP: the fourth-order terms of 0+ and +-1
                         differ by 6.6e-4 Eh, against 2e-7 with 'union');
                         for comparisons only.  The contracted spaces of
                         'union' depend on the model space (spinor
                         excitations change the symmetry, so every reference
                         function adds functions to every sector);
                         docs/derivations/nevpt4-sd.md

The space is zmr's MRCI product basis (zmr.mrci.basis) in quasi-canonical
core and virtual orbitals of the averaged density, where H_D is diagonal
(MRCI.hdiag); H x is the MRCI sigma.  Four sigma evaluations per state.
'''

import time

import numpy

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.mrci.basis import ProductBasis
from socutils.zmr.mrci.sigma import Sigma


class _Space:
    '''One product basis with its sigma, the Dyall energies and the model
    states in it.'''

    def __init__(self, eris, act, nelec, civecs_basis, civecs_model, contraction, eps_core,
                 eps_virt, lindep):
        mode = 'none' if contraction == 'none' else 'fic'
        self.basis = ProductBasis(eris, act, nelec, civecs_basis, mode, 'cas', lindep)
        self.sigma = Sigma(eris, act, self.basis, None)
        self.sigma.compile_all()
        self.hdiag = self.basis.diagonal(eps_core, eps_virt, act.e_core)
        cas = self.basis.blocks['cas']
        p = numpy.zeros((civecs_model.shape[1], self.basis.dim), dtype=numpy.complex128)
        p[:, :cas.size] = (cas.b.conj().T @ civecs_model).T
        self.p = p                                   # (nstate, dim), orthonormal rows

    def h(self, x):
        return self.sigma.matvec(numpy.atleast_2d(x))

    def q(self, x):
        '''Projection onto the complement of the model space (rows of x).'''
        x = numpy.atleast_2d(x)
        return x - (x @ self.p.conj().T) @ self.p

    def v(self, x, hx):
        '''V x = H x - H_D x.'''
        return hx - self.hdiag * x

    def resolvent(self, e0, x, tol):
        d = e0 - self.hdiag
        out = numpy.zeros_like(x)
        m = numpy.abs(d) > tol
        out[..., m] = x[..., m] / d[m]
        return out


class NEVPT4SD:
    '''Multi-state NEVPT4(SD).

    Args:
        eris: integrals (zmr.integrals)
        nelec: active electrons
        nstate: number of model states (lowest CASCI roots), or
        civecs: model CI vectors as columns (overrides nstate)
        weights: weights of the averaged density (Fock matrix) and of the
            common E0 (default equal)
        e0: 'state' or 'average'
        contraction: 'union' (default), 'none' or 'state' (see the module
            docstring: unbalanced at fourth order)
        renormalization: include -sum_K x_K H2_KJ in the fourth order
            (contraction 'union' or 'none')
        lindep: threshold of the metric of the PC functions

    After kernel():
        hmodel, h2, h3, h4: (nstate, nstate) Bloch contributions (columns J)
        heff: Hermitized H_PP + H2 + H3 + H4
        e2, e3, e4: eigenvalues of the Hermitized matrices to that order
        e_tot = e4; norm1: <psi1_J|psi1_J>
    '''

    resolvent_tol = 1e-10
    # reference functions closer than this in energy (Eh) share their PC
    # functions with contraction='state'
    degeneracy_tol = 1e-8
    verbose = 1

    def __init__(self, eris, nelec, nstate=None, civecs=None, weights=None, e0='state',
                 contraction='union', renormalization=False, lindep=1e-10):
        if e0 not in ('state', 'average'):
            raise ValueError('e0 must be "state" or "average"')
        if contraction not in ('state', 'union', 'none'):
            raise ValueError('contraction must be "state", "union" or "none"')
        if renormalization and contraction == 'state':
            raise ValueError('the renormalization term needs one first-order space for all '
                             'states (contraction "union" or "none")')
        self.eris0 = eris
        self.nelec = nelec
        self.e0 = e0
        self.contraction = contraction
        self.renormalization = renormalization
        self.lindep = lindep
        act = ActiveSpace.from_eris(eris)
        if civecs is None:
            civecs = act.solve(nelec, nstate or 1)[1]
        civecs = numpy.asarray(civecs, dtype=numpy.complex128)
        self.civecs = civecs[:, None] if civecs.ndim == 1 else civecs
        n = self.civecs.shape[1]
        self.weights = numpy.full(n, 1.0 / n) if weights is None else numpy.asarray(weights, float)
        self._act0 = act

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def build(self):
        t0 = time.perf_counter()
        gamma = self._act0.averaged_rdm1(self.civecs, self.nelec, self.weights)
        qc = fock.quasi_canonicalize(self.eris0, gamma)
        self.eris = qc.eris
        self.eps_core, self.eps_virt = qc.eps_core, qc.eps_virt
        self.act = ActiveSpace.from_eris(self.eris)
        args = (self.eris, self.act, self.nelec)
        tail = (self.contraction, self.eps_core, self.eps_virt, self.lindep)
        n = self.civecs.shape[1]
        if self.contraction == 'state':
            # the PC functions of column J come from J and the reference functions
            # degenerate with it: any basis of a degenerate set (a Kramers pair, the
            # components of a J level) gives the same spaces
            e = numpy.einsum('pi,pi->i', self.civecs.conj(),
                             self.act.apply_h(self.nelec, self.civecs)).real
            groups = []
            for j in range(n):
                for g in groups:
                    if abs(e[g[0]] - e[j]) < self.degeneracy_tol:
                        g.append(j)
                        break
                else:
                    groups.append([j])
            self.spaces = [None] * n
            for g in groups:
                s = _Space(*args, self.civecs[:, g], self.civecs, *tail)
                for j in g:
                    self.spaces[j] = s
            self.degenerate_sets = groups
        else:
            self.spaces = [_Space(*args, self.civecs, self.civecs, *tail)] * n
        self._log('NEVPT4(SD), %s contraction, E0 %s: dimension %s, built in %.1f s'
                  % (self.contraction, self.e0,
                     ' '.join(str(d) for d in sorted({sp.basis.dim for sp in self.spaces})),
                     time.perf_counter() - t0))
        return self

    def kernel(self):
        if not hasattr(self, 'spaces'):
            self.build()
        t0 = time.perf_counter()
        n = self.civecs.shape[1]
        sp0 = self.spaces[0]
        hp = sp0.h(sp0.p)
        hmodel = sp0.p.conj() @ hp.T                 # <I|H|J>
        hdmodel = sp0.p.conj() @ (sp0.hdiag * sp0.p).T
        if self.contraction == 'state' and abs(hmodel - numpy.diag(hmodel.diagonal())).max() > 1e-8:
            raise ValueError('contraction "state" needs CAS eigenstates as model states')
        e_d = hdmodel.diagonal().real
        e0 = e_d.copy() if self.e0 == 'state' else numpy.full(n, self.weights @ e_d)
        # <K|V|J> with H0 = sum_J E0_J |J><J| in the model space (E0 1 for e0='average')
        h1 = hmodel - numpy.diag(e0)
        self.hmodel, self.h1 = hmodel, h1
        h2 = numpy.zeros((n, n), dtype=numpy.complex128)
        h3, h4 = h2.copy(), h2.copy()
        xs, ys = [None] * n, [None] * n
        norm1 = numpy.zeros(n)
        tol = self.resolvent_tol
        shared = self.contraction != 'state'
        # first and second order for all columns (the fourth order needs all
        # of them when the renormalization term is included)
        for j in range(n):
            s = self.spaces[j]
            hpj = hp[j] if s is sp0 else s.h(s.p[j])[0]
            x = s.resolvent(e0[j], s.q(hpj)[0], tol)
            hx = s.h(x)[0]
            h2[:, j] = s.p.conj() @ s.v(x, hx)
            xs[j] = (x, hx)
            norm1[j] = numpy.vdot(x, x).real
        for j in range(n):
            s = self.spaces[j]
            x, hx = xs[j]
            rhs = s.v(x, hx) - (sum(xs[k][0] * h1[k, j] for k in range(n)) if shared
                                else x * h1[j, j])
            y = s.resolvent(e0[j], s.q(rhs)[0], tol)
            hy = s.h(y)[0]
            h3[:, j] = s.p.conj() @ s.v(y, hy)
            ys[j] = (y, hy)
        for j in range(n):
            s = self.spaces[j]
            y, hy = ys[j]
            rhs = s.v(y, hy) - (sum(ys[k][0] * h1[k, j] for k in range(n)) if shared
                                else y * h1[j, j])
            if self.renormalization:
                rhs = rhs - sum(xs[k][0] * h2[k, j] for k in range(n))
            z = s.resolvent(e0[j], s.q(rhs)[0], tol)
            hz = s.h(z)[0]
            h4[:, j] = s.p.conj() @ s.v(z, hz)
        self.h2, self.h3, self.h4, self.norm1 = h2, h3, h4, norm1

        def herm(m):
            return 0.5 * (m + m.conj().T)
        self.e2 = numpy.linalg.eigvalsh(herm(hmodel + h2))
        self.e3 = numpy.linalg.eigvalsh(herm(hmodel + h2 + h3))
        self.heff = herm(hmodel + h2 + h3 + h4)
        self.e4 = self.e_tot = numpy.linalg.eigvalsh(self.heff)
        self._log('NEVPT4(SD): %.1f s, %d sigma evaluations per state' % (time.perf_counter() - t0, 4))
        return self.e_tot
