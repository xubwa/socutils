'''
Multi-state NEVPT3: the effective Hamiltonian of zmr.pt4.nevpt4 to third
order (docs/derivations/nevpt4-sd.md), with the sigma evaluations that the
third order needs and nothing more.

Column J of the Bloch expansion uses x_J = R_J Q V |J> and
y_J = R_J Q [V x_J - sum_K x_K H1_KJ] (shared spaces; x_J H1_JJ for
contraction='state').  Since x_J and y_J lie in the complement Q of the model
space and V is Hermitian in the orthonormal product basis,

    H2_IJ = <I| V x_J> = <V I | x_J>,    H3_IJ = <I| V y_J> = <V I | y_J>,

so V y_J (the sigma of y_J, needed by the fourth order only) is never formed:
one sigma of the model states and one of the x_J, each over all states of a
space at once.  The third order does not depend on the contraction of the
first-order space ('union', 'none' and the PC functions of any set of
reference functions give the same energies), which is what makes it the
level at which the multi-state energies are both exact order by order for
each state (e0='state') and balanced between states.
'''

import time

import numpy

from socutils.zmr.pt4.nevpt4 import NEVPT4SD


class NEVPT3(NEVPT4SD):
    '''Multi-state NEVPT3 (arguments of NEVPT4SD without renormalization).

    After kernel():
        hmodel, h1, h2, h3: (nstate, nstate) Bloch contributions (columns J)
        heff: Hermitized H_PP + H2 + H3
        e2, e3: eigenvalues of the Hermitized matrices to that order;
        e_tot = e3; norm1: <psi1_J|psi1_J>
    '''

    def __init__(self, eris, nelec, nstate=None, civecs=None, weights=None, e0='state',
                 contraction='union', lindep=1e-10):
        super().__init__(eris, nelec, nstate=nstate, civecs=civecs, weights=weights, e0=e0,
                         contraction=contraction, lindep=lindep)

    def kernel(self):
        if not hasattr(self, 'spaces'):
            self.build()
        t0 = time.perf_counter()
        n = self.civecs.shape[1]
        tol = self.resolvent_tol
        shared = self.contraction != 'state'
        # distinct spaces and the columns that use each
        groups = []
        for j, s in enumerate(self.spaces):
            for g in groups:
                if g[0] is s:
                    g[1].append(j)
                    break
            else:
                groups.append((s, [j]))
        sp0 = self.spaces[0]
        # V |I> of the model states in every space (one sigma of all of them)
        vp = {}
        for s, _ in groups:
            vp[id(s)] = s.v(s.p, s.h(s.p))
        hmodel = sp0.p.conj() @ (vp[id(sp0)] + sp0.hdiag * sp0.p).T        # <I|H|J>
        hdmodel = sp0.p.conj() @ (sp0.hdiag * sp0.p).T
        if self.contraction == 'state' and abs(hmodel - numpy.diag(hmodel.diagonal())).max() > 1e-8:
            raise ValueError('contraction "state" needs CAS eigenstates as model states')
        e_d = hdmodel.diagonal().real
        e0 = e_d.copy() if self.e0 == 'state' else numpy.full(n, self.weights @ e_d)
        h1 = hmodel - numpy.diag(e0)
        self.hmodel, self.h1 = hmodel, h1
        h2 = numpy.zeros((n, n), dtype=numpy.complex128)
        h3 = h2.copy()
        norm1 = numpy.zeros(n)
        nsigma = 0
        for s, cols in groups:
            v = vp[id(s)]
            qv = s.q(v[cols])
            x = numpy.stack([s.resolvent(e0[j], qv[k], tol) for k, j in enumerate(cols)])
            vx = s.v(x, s.h(x))
            nsigma += 2
            for k, j in enumerate(cols):
                h2[:, j] = v.conj() @ x[k]
                norm1[j] = numpy.vdot(x[k], x[k]).real
            for k, j in enumerate(cols):
                rhs = vx[k] - (sum(x[cols.index(m)] * h1[m, j] for m in cols) if shared
                               else x[k] * h1[j, j])
                y = s.resolvent(e0[j], s.q(rhs)[0], tol)
                h3[:, j] = v.conj() @ y
        self.h2, self.h3, self.norm1 = h2, h3, norm1

        def herm(m):
            return 0.5 * (m + m.conj().T)
        self.e2 = numpy.linalg.eigvalsh(herm(hmodel + h2))
        self.heff = herm(hmodel + h2 + h3)
        self.e3 = self.e_tot = numpy.linalg.eigvalsh(self.heff)
        self._log('NEVPT3: %.1f s, %d batched sigma evaluations' % (time.perf_counter() - t0, nsigma))
        return self.e_tot
