'''
CASPT2 in the product basis of the MRCI code.

    pt = CASPT2(eris, nelec, nstate=3)          # or civecs=
    e = pt.kernel()

Zeroth-order Hamiltonian

    H0 = P F P + S F S,      F = sum_pq f_pq p^+ q,

with the generalized Fock matrix f of the state-averaged density, all of its
blocks (core-active, active-virtual and core-virtual included), P the model
states and S the external blocks of the product basis: the internally
contracted first-order interacting space of the model states
(contraction='fic', the union over the states) or all determinants with one
or two core holes / virtual particles (contraction='none').  The functions of
the active space outside the model states do not interact with them and are
left out.  No IPEA shift; level_shift adds a real shift to the denominators
without the usual correction of the energy.

Multi-state treatment as in XMS-CASPT2: the model states are rotated among
themselves to diagonalize P F P, each rotated state J has E0_J = <J|F|J> and
the first-order function (F_SS - E0_J) x_J = - S H |J>; the effective
Hamiltonian <I|H|J> + 1/2 (<I|H|x_J> + <x_I|H|J>) is diagonalized.  One model
state gives the single-state theory.

The classes are coupled by the off-diagonal blocks of F, so the equations are
solved iteratively (GMRES with the diagonal of F_SS as preconditioner); sigma
of the one-electron operator F is the sigma of the MRCI code without its
two-electron terms.  PC-NEVPT2 with h0 = 'fock' (zmr.uc.nevpt2) is this theory
without the off-diagonal blocks of F.

After kernel(): e_tot, heff, e2 (second-order energies of the rotated
states), e_ref, model_rotation, e0, weights (1 / (1 + |x_J|^2)), niter.
'''

import time

import numpy
import scipy.sparse.linalg

from socutils.zmr import fock, output
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.mrci.mrci import MRCI
from socutils.zmr.mrci.sigma import Sigma


class _OneElectron:
    '''Integral source of the operator sum_pq f_pq p^+ q for Sigma.'''
    ecore = 0.0

    def __init__(self, f, spaces):
        self.h1 = f
        self.spaces = spaces

    def eri_block(self, labels):
        raise RuntimeError('a one-electron operator has no two-electron integrals')


class FockSigma(Sigma):
    '''F x in the product basis: the terms of the MRCI sigma with a
    one-electron integral, and F restricted to the active orbitals as the
    all-active part.'''

    def __init__(self, f, spaces, act, basis):
        na = act.norb
        Sigma.__init__(self, _OneElectron(f, spaces), act, basis,
                       active_eri=numpy.zeros((na,) * 4, dtype=numpy.complex128))

    def terms(self, bra, ket):
        if (bra, ket) not in self._terms:
            ham, over = Sigma.terms(self, bra, ket)
            ham = [t for t in ham if all(name != 'eri' for name, _ in t.tensors)]
            self._terms[bra, ket] = (ham, over)
        return self._terms[bra, ket]


class CASPT2(MRCI):
    conv_tol = 1e-8            # relative residual of the first-order equations
    max_cycle = 500
    level_shift = 0.0

    def __init__(self, eris, nelec, civecs=None, nstate=None, contraction='fic',
                 lindep=1e-10):
        MRCI.__init__(self, eris, nelec, civecs=civecs, nstate=nstate,
                      contraction=contraction, internal='states', lindep=lindep)

    def kernel(self):
        log = output.Printer(self.verbose, self.stdout)
        log.interval = self.heartbeat
        t_start = time.perf_counter()
        self.build(log)
        basis, sp = self.basis, self.eris.spaces
        nstate = self.civecs.shape[1]
        ncas = basis.blocks['cas'].size
        if basis.names[0] != 'cas' or ncas != nstate:
            raise RuntimeError('the internal block must hold the model states')

        # model states and S H |J>
        ref = self.reference_vectors()                       # (nstate, dim)
        log.start('H on the model states')
        v = self.sigma.matvec(ref)
        self._steps.append(('H on the model states', log.done()))
        hmodel = ref.conj() @ v.T
        hmodel = 0.5 * (hmodel + hmodel.conj().T)

        # Fock operator of the averaged density, in the quasi-canonical orbitals
        gamma = self._act0.averaged_rdm1(self.civecs, self.nelec)
        f = fock.generalized_fock(self.eris, gamma)
        log.start('Contractions of the Fock operator')
        fsig = self.fock_sigma = FockSigma(f, sp, self.act, basis)
        fsig.compile_all()
        self._steps.append(('Contractions of the Fock operator', log.done()))
        fref = fsig.matvec(ref)
        fmodel = ref.conj() @ fref.T
        fmodel = 0.5 * (fmodel + fmodel.conj().T)
        e0, u = numpy.linalg.eigh(fmodel)
        self.model_rotation, self.e0 = u, e0
        ref = u.T @ ref
        v = u.T @ v
        hmodel = u.conj().T @ hmodel @ u
        v[:, :ncas] = 0.0                                    # S H |J>

        # diagonal of F_SS: orbital energies and <b_k|F_act|b_k>
        sa = sp.slice('a')
        fact = ActiveSpace(f[sa, sa], numpy.zeros((sp.nact,) * 4), 0.0)
        saved = {}
        for name in basis.names:
            blk = basis.blocks[name]
            saved[name] = blk.mu
            blk.mu = numpy.einsum('dk,dk->k', blk.b.conj(), fact.apply_h(blk.nelec, blk.b)).real
        sc = sp.slice('c')
        diag = basis.diagonal(self.eps_core, self.eps_virt, float(numpy.trace(f[sc, sc]).real))
        for name in basis.names:
            basis.blocks[name].mu = saved[name]

        dim = basis.dim

        def fss(x):
            x = numpy.array(x, dtype=numpy.complex128)
            x[:ncas] = 0.0
            y = fsig.matvec(x[None])[0]
            y[:ncas] = 0.0
            return y

        x = numpy.zeros((nstate, dim), dtype=numpy.complex128)
        self.niter = []
        self.converged = True
        log.start('First-order equations')
        for j in range(nstate):
            shift = e0[j] - self.level_shift
            d = diag - shift
            d[:ncas] = 1.0
            d = numpy.where(abs(d) < 1e-3, 1e-3, d)
            count = [0]

            def aop(z, shift=shift):
                count[0] += 1
                return fss(z) - shift * _external(z, ncas)

            a = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=aop, dtype=numpy.complex128)
            m = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=lambda z, d=d: z / d,
                                                   dtype=numpy.complex128)
            xj, info = _solve(a, -v[j], m, self.conv_tol, self.max_cycle)
            xj[:ncas] = 0.0
            res = numpy.linalg.norm(aop(xj) + v[j]) / max(numpy.linalg.norm(v[j]), 1e-300)
            if info != 0 or res > 100 * self.conv_tol:
                self.converged = False
            x[j] = xj
            self.niter.append(count[0])
            log.progress('state', j + 1, nstate)
        self._steps.append(('First-order equations', log.done(
            '%d states, %d .. %d products with F' % (nstate, min(self.niter), max(self.niter)))))

        a2 = v.conj() @ x.T                                   # <I|H|x_J>
        self.e_ref = hmodel.diagonal().real
        self.e2 = a2.diagonal().real
        self.weights = 1.0 / (1.0 + numpy.einsum('jd,jd->j', x.conj(), x).real)
        self.first_order = x
        self.heff = hmodel + 0.5 * (a2 + a2.conj().T)
        self.e_tot = numpy.linalg.eigvalsh(self.heff)
        self.wall = time.perf_counter() - t_start
        return self.e_tot


def _external(z, ncas):
    out = numpy.array(z, dtype=numpy.complex128)
    out[:ncas] = 0.0
    return out


def _solve(a, b, m, tol, maxiter, restart=40):
    """GMRES (scipy's MINRES does not take complex Hermitian matrices)."""
    try:
        return scipy.sparse.linalg.gmres(a, b, M=m, rtol=tol, atol=0.0, restart=restart,
                                         maxiter=maxiter)
    except TypeError:                      # older scipy: the keyword is tol
        return scipy.sparse.linalg.gmres(a, b, M=m, tol=tol, atol=0.0, restart=restart,
                                         maxiter=maxiter)
