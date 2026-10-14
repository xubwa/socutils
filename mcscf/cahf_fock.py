#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Integral-free (Fock-build driven) CAHF for one open shell.
#
# The configuration-averaged energy of n electrons in m open-shell spinors
# (the ensemble of all C(m, n) determinants, zcahf.CAHF) depends on the
# orbitals only through two densities,
#     D_c = C_c C_c^+,        D_o = f C_o C_o^+,     f = n/m,
#     E = tr h (D_c + D_o) + 1/2 tr D_c G(D_c) + tr D_o G(D_c)
#         + a/2 tr D_o G(D_o) + E_nuc,
#     a = (n - 1) m / (n (m - 1))   (open-open pair-averaging factor),
# G(D) = J(D) - K(D).  Energy and orbital gradient therefore need only the
# J/K of D_c and D_o from mf.get_jk -- no integral transformation, so the
# same code runs on any J/K backend (molecular DF, periodic RSDF, ...).
#
# With F_c = h + G(D_c) + G(D_o) and F_o = h + G(D_c) + a G(D_o) (MO basis),
# the gradient of E(C exp(kappa)) is the anti-Hermitian
#     g = [F_c, P_c] + f [F_o, P_o],
# nonzero only in the core-open, core-virtual and open-virtual blocks.  The
# optimiser is L-BFGS on kappa with the diagonal model Hessian
#     H_pq = 2 [n_p (F^(p)_qq - F^(p)_pp) - n_q (F^(q)_qq - F^(q)_pp)]
# as preconditioner and a backtracking line search on the exact energy
# (one J/K pair per energy).
#
# Periodic (socutils.pbc.spinor_scf): mf.get_jk carries the G = 0-free
# kernel and mf.energy_nuc() the wavefunction constants, exactly what the
# ensemble of N-electron determinants needs; the G = 0 shift of K
# (mf.k_shift() S D S) only enters the preconditioner.
#
# Usage:
#     e, mo, mo_energy = cahf_fock.kernel(mf, mo, ncore, 14, 6)
# or mc = cahf_fock.CAHF(mf, 14, 6, ncore=ncore).run(mo)
#

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger


def _tag(c, occ):
    return lib.tag_array(numpy.dot(c * occ, c.conj().T), mo_coeff=c, mo_occ=numpy.full(c.shape[1], occ))


class CAHF(lib.StreamObject):
    '''Single-open-shell CAHF driven by J/K builds of the core and open-shell
    densities (see the module notes).'''

    def __init__(self, mf, ncas, nelecas, ncore=None):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.ncas = ncas
        self.nelecas = int(sum(nelecas)) if not numpy.isscalar(nelecas) else int(nelecas)
        if ncore is None:
            ncore = int(round(self.mol.nelectron - self.nelecas))
        self.ncore = ncore
        self.max_cycle = 100
        self.conv_tol = 1e-9
        self.conv_tol_grad = 1e-5
        self.max_stepsize = .3        # largest |kappa| element of one step
        self.lbfgs_space = 12
        self.hdiag_floor = .05
        self.mo_coeff = None
        self.mo_energy = None
        self.e_tot = None
        self.converged = False
        self.cycles = 0

    @property
    def f(self):
        return self.nelecas / self.ncas

    @property
    def a(self):
        n, m = self.nelecas, self.ncas
        return 0. if n <= 1 else (n - 1) * m / (n * (m - 1))

    # ------------------------------------------------------------------
    def _veffs(self, mo):
        '''G(D_c), G(D_o) in the AO basis.'''
        mf, ncore, nocc = self._scf, self.ncore, self.ncore + self.ncas
        if ncore:
            vj, vk = mf.get_jk(self.mol, _tag(mo[:, :ncore], 1.))
            gc = vj - vk
        else:
            gc = 0
        vj, vk = mf.get_jk(self.mol, _tag(mo[:, ncore:nocc], self.f))
        return gc, vj - vk

    def energy_and_fock(self, mo, h1e=None):
        '''Energy, MO-basis F_c and F_o.'''
        if h1e is None:
            h1e = self._h1e
        ncore, nocc, f, a = self.ncore, self.ncore + self.ncas, self.f, self.a
        gc, go = self._veffs(mo)
        dc = numpy.dot(mo[:, :ncore], mo[:, :ncore].conj().T)
        do = f * numpy.dot(mo[:, ncore:nocc], mo[:, ncore:nocc].conj().T)
        e = (numpy.einsum('ij,ji', h1e, dc + do) + .5 * numpy.einsum('ij,ji', dc, gc)
             + numpy.einsum('ij,ji', do, gc) + .5 * a * numpy.einsum('ij,ji', do, go)).real
        e += self._scf.energy_nuc()
        fc = mo.conj().T @ (h1e + gc + go) @ mo
        fo = mo.conj().T @ (h1e + gc + a * go) @ mo
        return e, fc, fo

    def gradient(self, fc, fo):
        ncore, nocc = self.ncore, self.ncore + self.ncas
        nmo = fc.shape[0]
        pc = numpy.zeros(nmo)
        pc[:ncore] = 1
        po = numpy.zeros(nmo)
        po[ncore:nocc] = 1
        g = fc * pc[None, :] - pc[:, None] * fc + self.f * (fo * po[None, :] - po[:, None] * fo)
        return g

    def _mask(self, nmo):
        ncore, nocc = self.ncore, self.ncore + self.ncas
        cls = numpy.zeros(nmo, dtype=int)
        cls[ncore:nocc] = 1
        cls[nocc:] = 2
        m = cls[:, None] > cls[None, :]       # lower triangle between classes: p in later class
        return m

    def hdiag(self, fc, fo):
        ncore, nocc, f = self.ncore, self.ncore + self.ncas, self.f
        nmo = fc.shape[0]
        nu = 0.
        k_shift = getattr(self._scf, 'k_shift', None)
        if callable(k_shift):
            nu = k_shift()
        n = numpy.zeros(nmo)
        n[:ncore] = 1
        n[ncore:nocc] = f
        # Fock of each orbital's own class; exchange G = 0 shift of the
        # occupied levels (periodic) only in this model Hessian
        occ = n.copy()
        dc = fc.diagonal().real - nu * occ
        do = fo.diagonal().real - nu * occ
        A = numpy.array([dc, do])[(numpy.arange(nmo) >= ncore).astype(int)]   # A[p, q] = e^(p)_q
        dA = A.diagonal()
        H = 2 * (n[:, None] * (A - dA[:, None]) - n[None, :] * (dA[None, :] - A.T))
        return H

    # ------------------------------------------------------------------
    def kernel(self, mo_coeff=None):
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        mo = numpy.asarray(mo_coeff)
        nmo = mo.shape[1]
        self._h1e = self._scf.get_hcore()
        mask = self._mask(nmo)
        log.info('Fock-driven CAHF: ncore %d, %d electrons in %d spinors (f = %.4f, a = %.4f), %d parameters',
                 self.ncore, self.nelecas, self.ncas, self.f, self.a, int(mask.sum()))

        e, fc, fo = self.energy_and_fock(mo)
        g = self.gradient(fc, fo)[mask]
        H = numpy.maximum(self.hdiag(fc, fo)[mask], self.hdiag_floor)
        s_hist, y_hist = [], []
        self.converged = False
        de = 0.
        for it in range(self.max_cycle):
            gnorm = numpy.linalg.norm(g)
            log.info('CAHF cycle %d  E = %.12f  dE = %.3e  |g| = %.3e', it, e, de, gnorm)
            if gnorm < self.conv_tol_grad and abs(de) < self.conv_tol and it > 0:
                self.converged = True
                break
            # two-loop L-BFGS on the real parameter vector (Re, Im)
            gv = numpy.concatenate([g.real, g.imag])
            Hv = numpy.concatenate([H, H])
            q = gv.copy()
            alphas = []
            for s, y in reversed(list(zip(s_hist, y_hist))):
                rho = 1. / numpy.dot(y, s)
                al = rho * numpy.dot(s, q)
                q -= al * y
                alphas.append((rho, al))
            r = q / Hv
            for (s, y), (rho, al) in zip(zip(s_hist, y_hist), reversed(alphas)):
                b = rho * numpy.dot(y, r)
                r += s * (al - b)
            dv = -r
            if numpy.dot(dv, gv) >= 0:      # not a descent direction: reset
                s_hist, y_hist = [], []
                dv = -gv / Hv
            mx = numpy.abs(dv).max()
            if mx > self.max_stepsize:
                dv *= self.max_stepsize / mx
            npar = g.size
            step = 1.
            for ls in range(8):
                k = numpy.zeros((nmo, nmo), dtype=complex)
                k[mask] = step * (dv[:npar] + 1j * dv[npar:])
                k = k - k.conj().T
                mo_new = mo @ scipy.linalg.expm(k)
                e_new, fc_new, fo_new = self.energy_and_fock(mo_new)
                if e_new < e + 1e-4 * step * numpy.dot(dv, gv) or e_new - e < 1e-12:
                    break
                step *= .5
                log.debug('  line search: step %.3g, dE %.3e', step, e_new - e)
            g_new = self.gradient(fc_new, fo_new)[mask]
            s = step * dv
            y = numpy.concatenate([g_new.real, g_new.imag]) - gv
            if numpy.dot(s, y) > 1e-12 * numpy.dot(y, y):
                s_hist.append(s)
                y_hist.append(y)
                if len(s_hist) > self.lbfgs_space:
                    s_hist.pop(0)
                    y_hist.pop(0)
            de = e_new - e
            mo, e, fc, fo, g = mo_new, e_new, fc_new, fo_new, g_new
        self.cycles = it
        if not self.converged:
            log.warn('Fock-driven CAHF not converged in %d cycles (|g| = %.3e)', self.max_cycle,
                     numpy.linalg.norm(g))
        mo, mo_energy = self.canonicalize(mo, fc, fo)
        self.mo_coeff, self.mo_energy, self.e_tot = mo, mo_energy, e
        log.timer('Fock-driven CAHF', *t0)
        log.note('CAHF E = %.12f  converged %s (%d cycles)', e, self.converged, self.cycles)
        return e, mo, mo_energy

    run = kernel

    def canonicalize(self, mo, fc, fo):
        '''Diagonalise F_c in the core and virtual blocks and F_o in the open shell.'''
        ncore, nocc = self.ncore, self.ncore + self.ncas
        mo = mo.copy()
        e = numpy.zeros(mo.shape[1])
        for (p0, p1), fk in (((0, ncore), fc), ((ncore, nocc), fo), ((nocc, mo.shape[1]), fc)):
            if p1 > p0:
                w, v = numpy.linalg.eigh(fk[p0:p1, p0:p1])
                mo[:, p0:p1] = mo[:, p0:p1] @ v
                e[p0:p1] = w
        return mo, e


def kernel(mf, mo_coeff, ncore, ncas, nelecas, **kwargs):
    mc = CAHF(mf, ncas, nelecas, ncore=ncore)
    for k, v in kwargs.items():
        setattr(mc, k, v)
    return mc.kernel(mo_coeff)
