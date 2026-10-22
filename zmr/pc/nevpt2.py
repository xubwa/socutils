'''
Partially contracted multi-state NEVPT2.

Two multi-state contractions are available:

    contraction='union' (default)  the first-order space of every external
        label is spanned by the PC functions of all model states.  It is
        invariant under unitary rotations of the model space (with equal
        weights and e0='average').
    contraction='state'  each ket state J uses only its own PC functions,
        as in the QD-NEVPT2 of Angeli et al.; the second-order matrix is
        Hermitized, 1/2 [<I|V psi1_J> + <psi1_I|V|J>], for both E0 choices.
        Not rotation invariant.

Both coincide for a single state.
'''

import numpy

from socutils.zmr import output

from socutils.zmr.active import rdm as active_rdm
from socutils.zmr.pc import rdm_terms, spectral
from socutils.zmr.uc import engine
from socutils.zmr.uc.nevpt2 import NEVPT2Base


class PCNEVPT2(NEVPT2Base):
    '''PC-NEVPT2; arguments as NEVPT2Base plus

        contraction: 'union' or 'state'
        lindep: threshold on the eigenvalues of the metric of the contracted
            functions
        amplitudes: 'spectral' (contracted functions expanded in the sector
            eigenstates) or 'rdm' (metric and H_act matrices from transition
            RDMs up to rank 3 through the Wick-generated formulas, the 4-RDM
            terms through intermediate states; no sector is diagonalized)
    '''

    name = 'PC-NEVPT2'

    def method_label(self):
        return 'PC-NEVPT2 (%s contraction)' % self.contraction

    def settings(self):
        how = ('CI vectors (sector Hamiltonians)' if self.amplitudes == 'spectral'
               else 'RDMs up to rank 3 + intermediate states')
        return ([('First-order functions from', how),
                 ('Threshold of linear dependence', '%.1e' % self.lindep)]
                + NEVPT2Base.settings(self))

    # only the model states of the N-electron sector are needed
    full_reference_spectrum = False

    def __init__(self, eris, nelec, nstate=1, civecs=None, weights=None, e0='state',
                 method='exact', nquad=None, contraction='union', lindep=1e-10,
                 amplitudes='spectral'):
        NEVPT2Base.__init__(self, eris, nelec, nstate, civecs, weights, e0, method, nquad)
        if contraction not in ('union', 'state'):
            raise ValueError('contraction must be "union" or "state"')
        if amplitudes not in ('spectral', 'rdm'):
            raise ValueError('amplitudes must be "spectral" or "rdm"')
        self.contraction = contraction
        self.lindep = lindep
        self.amplitudes = amplitudes
        self.hermitize_always = contraction == 'state'
        self._dms = None

    def class_matrix(self, pclass, blocks, act, civecs, shifts):
        if self.amplitudes == 'rdm':
            return self._class_matrix_rdm(pclass, blocks, act, civecs, shifts)
        nstate = civecs.shape[1]
        ntarget = act.target(self.nelec, pclass.dn)
        a = numpy.zeros((nstate, nstate), dtype=civecs.dtype)
        if act.dim(ntarget) == 0:
            return a
        # contracted functions O_X |Psi_I> in the determinant basis: (I, X, D)
        phi = engine.class_functions(pclass, act, civecs, self.nelec)
        nx, dim = phi.shape[1], phi.shape[2]

        def space(rows):
            return spectral.contracted_space_det(rows, lambda: act.hmat(ntarget),
                                                 lambda: act.sector(ntarget), self.lindep,
                                                 act.blocks(ntarget))

        if self.contraction == 'union':
            output.stage('%d functions built, first-order space' % (nstate * nx))
            b, mu = space(phi.reshape(nstate * nx, dim))
            self._class_info = (nstate * nx, len(mu))
            if len(mu) == 0:
                return a
            output.stage('first-order space of dimension %d, sum over external labels' % len(mu))
            return self.contract(pclass.name, blocks, b, phi, mu, shifts)
        for j in range(nstate):
            output.progress('state', j + 1, nstate)
            b, mu = space(phi[j])
            if len(mu) == 0:
                continue
            a[:, j] = self.contract(pclass.name, blocks, b, phi, mu, shifts, col=j)[:, j]
        return a

    def _rdms(self, act, civecs):
        if self._dms is None:
            if rdm_terms.check_invariance(act, civecs, self.nelec) > 1e-8:
                raise ValueError('amplitudes="rdm" needs a model space invariant under '
                                 'H_act (CASCI eigenstates or their mixtures)')
            self._dms = {k: active_rdm.make_rdm(act, civecs, self.nelec, k)
                         for k in range(rdm_terms.max_rank() + 1)}
        return self._dms

    def _class_matrix_rdm(self, pclass, blocks, act, civecs, shifts):
        dms = self._rdms(act, civecs)
        def intermediate(name, items):
            return rdm_terms.intermediate_terms(name, items, act, civecs, self.nelec)

        s, kc = rdm_terms.class_matrices(pclass.name, dms, act.heff, act.aaaa,
                                         pclass.subset(act.norb), intermediate)
        k = rdm_terms.full_k(s, kc, self.hmodel_act)
        nstate, _, nx, _ = s.shape
        smat = s.transpose(0, 2, 1, 3).reshape(nstate * nx, nstate * nx)
        kmat = k.transpose(0, 2, 1, 3).reshape(nstate * nx, nstate * nx)
        if self.contraction == 'union':
            c, mu = rdm_terms.contracted_space(smat, kmat, self.lindep)
            self._class_info = (nstate * nx, len(mu))
            # t'[I, j, X] = sum_{L,Y} conj(c[(L,Y), j]) S[L, I, Y, X]
            t = numpy.einsum('lyj,liyx->ijx', c.conj().reshape(nstate, nx, -1), s, optimize=True)
            return self.contract_t(pclass.name, blocks, t, mu, shifts)
        a = numpy.zeros((nstate, nstate), dtype=numpy.complex128)
        for j in range(nstate):
            output.progress('state', j + 1, nstate)
            c, mu = rdm_terms.contracted_space(s[j, j], k[j, j], self.lindep)
            t = numpy.einsum('yj,iyx->ijx', c.conj(), s[j], optimize=True)
            aj = self.contract_t(pclass.name, blocks, t, mu, shifts, col=j)
            a[:, j] = aj[:, j]
        return a
