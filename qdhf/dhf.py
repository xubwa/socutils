#!/usr/bin/env python
# Copyright 2026 The socutils developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Four-component Dirac-Hartree-Fock (Dirac-Coulomb, LLLL+SSLL+SSSS) in
quadruple precision (IEEE binary128, ~34 significant digits).

Two Hamiltonians are available:

* ``integrals='quad'`` (default): all integrals are computed in quadruple
  precision by the McMurchie-Davidson kernels of libqdhf.  The 4c basis is
  the spin-orbital basis {chi_mu alpha, chi_mu beta} for the large
  component and the RKB basis (sigma.p) chi_mu {alpha,beta} / (2c) for the
  small component.  It spans the same space as the j-adapted spinor basis
  of pyscf.scf.dhf, so the two programs give the same energy.  The
  integrals agree with libcint to double precision and the SCF energy is
  accurate to ~1e-28 Hartree.

* ``integrals='libcint'``: the spinor integrals of libcint (exactly the
  Hamiltonian of pyscf.scf.dhf.DHF) are promoted to quadruple precision and
  the SCF equations are solved in quadruple precision.  The integrals carry
  the double precision rounding errors of libcint (relative error ~1e-15),
  so the result is a tightly converged solution of the double precision
  Hamiltonian, not a quadruple precision result of the exact Hamiltonian.

The SCF is first converged in double precision (pyscf.scf.dhf or the same
algorithm in numpy), then refined by quad precision DIIS iterations.

Example:

>>> from pyscf import gto
>>> from pyscf import qdhf
>>> mol = gto.M(atom='Ne', basis='cc-pvdz')
>>> mf = qdhf.DHF(mol).run()
>>> print(mf.e_tot)    # mpmath number with 34 significant digits
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from socutils.qdhf import qarray
from socutils.qdhf.qarray import QArray, qmp, hmp
from socutils.qdhf import integrals as qints


# --------------------------------------------------------------------------
# backends: the same SCF code runs on numpy arrays (double precision) and on
# QArray objects (quad precision)
# --------------------------------------------------------------------------
class _NumpyBackend:
    name = 'double'

    @staticmethod
    def eigh(a):
        return numpy.linalg.eigh(a)

    @staticmethod
    def trace_dot(a, b):
        return numpy.einsum('ij,ji->', a, b)

    @staticmethod
    def vdot_real(a, b):
        return numpy.vdot(a, b).real

    @staticmethod
    def to_list(w):
        return list(w)

    @staticmethod
    def solve(b, rhs):
        return numpy.linalg.solve(numpy.asarray(b, dtype=float), rhs)

    @staticmethod
    def diag_scale(u, w):
        '''u * w[None,:] for a list of real numbers w'''
        return u * numpy.asarray(w, dtype=float)

    @staticmethod
    def make_rdm1(c_occ):
        return c_occ @ c_occ.conj().T

    @staticmethod
    def real(x):
        return float(numpy.real(x))

    @staticmethod
    def to_mpf(x):
        return qmp.mpf(float(x))

    @staticmethod
    def concat_columns(cols):
        return numpy.hstack([c.reshape(-1, 1) for c in cols])

    @staticmethod
    def column(x, i):
        return x[:, i]


class _QuadBackend:
    name = 'quad'

    @staticmethod
    def eigh(a):
        return qarray.eigh(a)

    @staticmethod
    def trace_dot(a, b):
        return qarray.trace_dot(a, b)

    @staticmethod
    def vdot_real(a, b):
        v = qarray.vdot(a, b)
        return qmp.re(v)

    @staticmethod
    def to_list(w):
        return list(w.to_mpmath())

    @staticmethod
    def solve(b, rhs):
        x = qmp.lu_solve(qmp.matrix(b), qmp.matrix(rhs))
        return [x[i] for i in range(len(rhs))]

    @staticmethod
    def diag_scale(u, w):
        n = len(w)
        d = QArray.zeros((n, n))
        for i in range(n):
            d[i, i] = QArray.scalar(w[i])
        return u @ d

    @staticmethod
    def make_rdm1(c_occ):
        return qarray.matmul(c_occ, c_occ.conj(), trans_b=True)

    @staticmethod
    def real(x):
        return qmp.re(x)

    @staticmethod
    def to_mpf(x):
        return qmp.mpf(x)

    @staticmethod
    def concat_columns(cols):
        return qarray.concatenate([c.reshape(-1, 1) for c in cols], axis=1)

    @staticmethod
    def column(x, i):
        return x[:, i]


# --------------------------------------------------------------------------
# 4c Hamiltonian
# --------------------------------------------------------------------------
class Hamiltonian4c:
    '''Four-component Hamiltonian in an "expansion" representation.

    Each 4c basis function I is expanded in scalar functions of several
    function spaces f.  For component group p, T_p[I,g] is the coefficient of
    the scalar function g (in space f(p)) in the 4-spinor I.  Then

        S_IJ   = sum_p (T_p^* S_f T_p^T)_IJ
        (IJ|KL) = sum_pq sum_{g} T_p^*[I,g1] T_p[J,g2] T_q^*[K,g3] T_q[L,g4] (g1g2|g3g4)

    Attributes:
        hcore, ovlp : 4c matrices
        comps : list of (T_p, f)
        eri : dict {(f,g): (g1 g2|g3 g4) tensor with g1,g2 in f, g3,g4 in g}
        energy_nuc : nuclear repulsion
    '''
    def __init__(self, hcore, ovlp, comps, eri, energy_nuc, light_speed,
                 k_factor=1, occ_number=1):
        self.hcore = hcore
        self.ovlp = ovlp
        self.comps = comps
        self.energy_nuc = energy_nuc
        # light_speed = None for nonrelativistic Hamiltonians
        self.light_speed = light_speed
        # Fock = h + J - k_factor * K,  D = occ_number * C_occ C_occ^H
        # (k_factor = 1/2, occ_number = 2 for closed-shell RHF)
        self.k_factor = k_factor
        self.occ_number = occ_number
        self.eri = eri
        self._eri_j = {}
        self._eri_k = {}
        for (f, g), e in eri.items():
            n1, n2, n3, n4 = e.shape
            self._eri_j[f, g] = e.reshape(n1*n2, n3*n4)
            # [g1,g4,g2,g3] for exchange
            self._eri_k[f, g] = e.transpose(0, 3, 1, 2).reshape(n1*n4, n2*n3)
        self.nao_4c = hcore.shape[0]

    def to_double(self):
        comps = [(t.to_numpy(), f) for t, f in self.comps]
        eri = {k: v.to_numpy() for k, v in self.eri.items()}
        c = None if self.light_speed is None else float(self.light_speed)
        return Hamiltonian4c(self.hcore.to_numpy(), self.ovlp.to_numpy(), comps,
                             eri, float(self.energy_nuc), c,
                             self.k_factor, self.occ_number)

    def get_jk(self, dm):
        dmT = dm.T
        rho = {}
        for t, f in self.comps:
            r = t.conj().T @ dmT @ t
            rho[f] = r if f not in rho else rho[f] + r
        w = {}
        for (f, g), e in self._eri_j.items():
            x = (e @ rho[g].reshape(-1, 1)).reshape(rho[f].shape)
            w[f] = x if f not in w else w[f] + x
        vj = None
        for t, f in self.comps:
            x = t.conj() @ w[f] @ t.T
            vj = x if vj is None else vj + x

        # exchange, batched over the component pairs of the same (f,g) block
        groups = {}
        for p, (tp, f) in enumerate(self.comps):
            for q, (tq, g) in enumerate(self.comps):
                groups.setdefault((f, g), []).append((p, q))
        vk = None
        for (f, g), pqs in groups.items():
            if (f, g) not in self._eri_k:
                continue
            cols = []
            shapes = []
            for p, q in pqs:
                tp = self.comps[p][0]
                tq = self.comps[q][0]
                r = tp.T @ dm @ tq.conj()
                shapes.append(r.shape)
                cols.append(r.reshape(-1))
            if isinstance(dm, QArray):
                rmat = qarray.concatenate([c.reshape(-1, 1) for c in cols], axis=1)
            else:
                rmat = numpy.hstack([c.reshape(-1, 1) for c in cols])
            kmat = self._eri_k[f, g] @ rmat
            for i, (p, q) in enumerate(pqs):
                tp = self.comps[p][0]
                tq = self.comps[q][0]
                n1 = tp.shape[1]
                n4 = tq.shape[1]
                ks = kmat[:, i].reshape(n1, n4)
                x = tp.conj() @ ks @ tq.T
                vk = x if vk is None else vk + x
        return vj, vk


def _sigma():
    sx = numpy.array([[0, 1], [1, 0]], dtype=complex)
    sy = numpy.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = numpy.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def get_light_speed(c=None):
    if c is None:
        c = lib.param.LIGHT_SPEED
    return hmp.mpf(c) if not isinstance(c, str) else hmp.mpf(c)


def build_rkb_hamiltonian(mol, light_speed=None, verbose=None):
    '''4c Dirac-Coulomb Hamiltonian with quad precision integrals'''
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    c = get_light_speed(light_speed)
    qi = qints.QuadIntegrals(mol)
    n = qi.nao
    n4 = n * 4
    sx = qi.ovlp_x()
    vx = qi.nuc_x()
    sy = qi.ovlp_y()
    vy = qi.nuc_y()
    t0 = log.timer('quad 1e integrals', *t0)

    # expansion coefficients of the 4-spinors
    t_up = []
    for s in range(2):
        t = numpy.zeros((n4, n))
        t[s*n:(s+1)*n] = numpy.eye(n)
        t_up.append(QArray.from_numpy(t))
    # small component, chi^S_{mu s} = (sigma.p) chi_mu |s> / (2c)
    #   lower component p' = sum_a (-i/(2c)) (sigma_a)_{p's} d_a chi_mu
    sigma = _sigma()
    t_low = []
    for pp in range(2):
        t = numpy.zeros((n4, 3*n), dtype=complex)
        for s in range(2):
            for a in range(3):
                # -i * sigma_a[pp,s], the 1/(2c) factor is applied below
                v = -1j * sigma[a][pp, s]
                if v != 0:
                    for mu in range(n):
                        t[2*n+s*n+mu, 3*mu+a] = v
        t_low.append(QArray.from_numpy(t) * (1 / (2 * c)))
    comps = [(t_up[0], 'X'), (t_up[1], 'X'), (t_low[0], 'Y'), (t_low[1], 'Y')]

    ovlp = None
    vnuc = None
    for t, f in comps:
        sf, vf = (sx, vx) if f == 'X' else (sy, vy)
        x = t.conj() @ sf @ t.T
        y = t.conj() @ vf @ t.T
        ovlp = x if ovlp is None else ovlp + x
        vnuc = y if vnuc is None else vnuc + y
    # kinetic energy T = 1/2 <d_a chi|d_a chi>
    tkin = (sy[0::3, 0::3] + sy[1::3, 1::3] + sy[2::3, 2::3]) * hmp.mpf(0.5)
    hcore = vnuc.as_complex()
    n2 = 2 * n
    for s in range(2):
        L = slice(s*n, (s+1)*n)
        S = slice(n2+s*n, n2+(s+1)*n)
        hcore[L, S] = hcore[L, S] + tkin
        hcore[S, L] = hcore[S, L] + tkin
    hcore[n2:, n2:] = hcore[n2:, n2:] - ovlp[n2:, n2:] * (2 * c**2)

    eri_xxxx = qi.eri_xxxx()
    t0 = log.timer('quad (LL|LL) integrals', *t0)
    eri_yyxx = qi.eri_yyxx()
    t0 = log.timer('quad (SS|LL) integrals', *t0)
    eri_yyyy = qi.eri_yyyy()
    t0 = log.timer('quad (SS|SS) integrals', *t0)
    eri = {('X', 'X'): eri_xxxx,
           ('Y', 'X'): eri_yyxx,
           ('X', 'Y'): eri_yyxx.transpose(2, 3, 0, 1),
           ('Y', 'Y'): eri_yyyy}
    e_nuc = qmp.mpf(qints.energy_nuc(mol))
    return Hamiltonian4c(hcore, ovlp, comps, eri, e_nuc, c)


def _symmetrize_spinor_eri(eri, pair_symmetric):
    '''Impose (ij|kl) = (ji|lk)^* (and (ij|kl) = (kl|ij) for the LLLL and
    SSSS blocks) exactly.  The libcint integrals satisfy them only up to
    rounding errors, which would limit the convergence of the quad precision
    SCF.'''
    if pair_symmetric:
        eri = (eri + eri.transpose(2, 3, 0, 1)) * .5
    eri = (eri + eri.transpose(1, 0, 3, 2).conj()) * .5
    return eri


def build_libcint_hamiltonian(mol, light_speed=None, verbose=None):
    '''The Hamiltonian of pyscf.scf.dhf (libcint spinor integrals, double
    precision) promoted to quad precision'''
    c = get_light_speed(light_speed)
    c1 = 1 / (2 * c)
    n2c = mol.nao_2c()
    n4 = n2c * 2
    s = QArray.from_numpy(mol.intor_symmetric('int1e_ovlp_spinor'))
    t = QArray.from_numpy(mol.intor_symmetric('int1e_spsp_spinor'))
    vn = QArray.from_numpy(mol.intor_symmetric('int1e_nuc_spinor'))
    wn = QArray.from_numpy(mol.intor_symmetric('int1e_spnucsp_spinor'))
    L = slice(0, n2c)
    S = slice(n2c, n4)
    half = hmp.mpf(0.5)
    hcore = QArray.zeros((n4, n4), iscomplex=True)
    hcore[L, L] = vn
    hcore[L, S] = t * half
    hcore[S, L] = t * half
    hcore[S, S] = wn * c1**2 - t * half
    ovlp = QArray.zeros((n4, n4), iscomplex=True)
    ovlp[L, L] = s
    ovlp[S, S] = t * c1**2

    tl = numpy.zeros((n4, n2c))
    tl[:n2c] = numpy.eye(n2c)
    ts = numpy.zeros((n4, n2c))
    ts[n2c:] = numpy.eye(n2c)
    comps = [(QArray.from_numpy(tl), 'L'), (QArray.from_numpy(ts), 'S')]
    llll = QArray.from_numpy(_symmetrize_spinor_eri(mol.intor('int2e_spinor'), True))
    ssll = QArray.from_numpy(_symmetrize_spinor_eri(mol.intor('int2e_spsp1_spinor'), False)) * c1**2
    ssss = QArray.from_numpy(_symmetrize_spinor_eri(mol.intor('int2e_spsp1spsp2_spinor'), True)) * c1**4
    eri = {('L', 'L'): llll,
           ('S', 'L'): ssll,
           ('L', 'S'): ssll.transpose(2, 3, 0, 1),
           ('S', 'S'): ssss}
    e_nuc = qmp.mpf(qints.energy_nuc(mol))
    return Hamiltonian4c(hcore, ovlp, comps, eri, e_nuc, c)


# --------------------------------------------------------------------------
# SCF driver
# --------------------------------------------------------------------------
def _canonical_orth(xp, s, lindep):
    w, u = xp.eigh(s)
    w = xp.to_list(w)
    keep = [i for i, x in enumerate(w) if x > lindep]
    u = u[:, keep[0]:]
    assert keep == list(range(keep[0], len(w)))
    wk = [1 / qmp.sqrt(qmp.mpf(w[i])) if xp is _QuadBackend else 1 / numpy.sqrt(w[i])
          for i in keep]
    return xp.diag_scale(u, wk), len(w) - len(keep)


def _get_occ(xp, mo_energy, nelec, c, log):
    '''Occupy the lowest nelec positive energy states (the lowest nelec
    states for nonrelativistic Hamiltonians, c = None)'''
    e = [float(x) for x in xp.to_list(mo_energy)]
    if c is None:
        pes = list(range(len(e)))
    else:
        pes = [i for i, x in enumerate(e) if x > -1.999 * float(c)**2]
    occ = pes[:nelec]
    if len(occ) < nelec:
        raise RuntimeError('Not enough positive energy states')
    if len(pes) > nelec and e[pes[nelec]] - e[pes[nelec-1]] < 1e-3:
        log.warn('HOMO %.15g == LUMO %.15g', e[pes[nelec-1]], e[pes[nelec]])
    return occ


def scf_iterations(ham, xp, nelec, dm0, conv_tol, conv_tol_grad, max_cycle,
                   diis_space, lindep, log):
    '''Roothaan-Hall iterations with DIIS.

    Returns converged, e_tot, mo_energy, mo_coeff, occ_idx, dm
    '''
    s = ham.ovlp
    h = ham.hcore
    c = ham.light_speed
    nocc = nelec // ham.occ_number
    if nocc * ham.occ_number != nelec:
        raise ValueError('%d electrons cannot be distributed in orbitals '
                         'with occupation %d' % (nelec, ham.occ_number))
    x, nlindep = _canonical_orth(xp, s, lindep)
    if nlindep > 0:
        log.info('%s: %d linearly dependent functions removed', xp.name, nlindep)
    xh = x.conj().T
    dm = dm0
    e_last = None
    f_list = []
    e_list = []
    converged = False
    mo_energy = mo_coeff = occ = None

    for cycle in range(max_cycle):
        vj, vk = ham.get_jk(dm)
        if ham.k_factor == 1:
            veff = vj - vk
        else:
            veff = vj - vk * ham.k_factor
        fock = h + veff
        e1 = xp.real(xp.trace_dot(h, dm))
        e2 = xp.real(xp.trace_dot(veff, dm)) / 2
        e_tot = e1 + e2 + ham.energy_nuc

        fds = fock @ dm @ s
        err = xh @ (fds - fds.conj().T) @ x
        err_norm = qmp.sqrt(xp.to_mpf(xp.vdot_real(err, err)))
        de = None if e_last is None else e_tot - e_last
        if de is None:
            log.info('%s cycle %d  E= %s', xp.name, cycle, qmp.nstr(qmp.mpf(e_tot), 34))
        else:
            log.info('%s cycle %d  E= %s  delta_E= %.3g  |[F,D]|= %.3g', xp.name, cycle,
                     qmp.nstr(qmp.mpf(e_tot), 34), float(de), float(err_norm))
        if de is not None and abs(de) < conv_tol and err_norm < conv_tol_grad:
            converged = True
            break
        e_last = e_tot

        # DIIS
        f_list.append(fock)
        e_list.append(err)
        if len(f_list) > diis_space:
            f_list.pop(0)
            e_list.pop(0)
        nd = len(f_list)
        if nd > 1:
            b = [[0] * (nd+1) for _ in range(nd+1)]
            for i in range(nd):
                for j in range(i+1):
                    b[i][j] = b[j][i] = xp.vdot_real(e_list[i], e_list[j])
                b[i][nd] = b[nd][i] = -1
            b[nd][nd] = 0
            rhs = [0] * nd + [-1]
            try:
                coef = xp.solve(b, rhs)
                fock = None
                for i in range(nd):
                    ci = coef[i] if xp is _NumpyBackend else qmp.mpf(coef[i])
                    fock = f_list[i] * ci if fock is None else fock + f_list[i] * ci
            except (ZeroDivisionError, numpy.linalg.LinAlgError):
                log.warn('DIIS singular, use the last Fock matrix')
                fock = f_list[-1]

        mo_energy, c1 = xp.eigh(xh @ fock @ x)
        mo_coeff = x @ c1
        occ = _get_occ(xp, mo_energy, nocc, c, log)
        c_occ = mo_coeff[:, occ[0]:occ[-1]+1]
        assert occ == list(range(occ[0], occ[-1]+1))
        dm = xp.make_rdm1(c_occ)
        if ham.occ_number != 1:
            dm = dm * ham.occ_number

    if converged:
        # orbitals of the converged Fock matrix
        mo_energy, c1 = xp.eigh(xh @ fock @ x)
        mo_coeff = x @ c1
        occ = _get_occ(xp, mo_energy, nocc, c, log)
    return converged, e_tot, mo_energy, mo_coeff, occ, dm


def spinor_to_spin_orbital_dm(mol, dm):
    '''Transform a 4c density matrix of pyscf.scf.dhf (j-adapted spinor
    basis) to the spin-orbital basis used by the quad Hamiltonian'''
    ca, cb = mol.sph2spinor_coeff()
    if mol.cart:
        c2s = mol.cart2sph_coeff()
        ca = c2s @ ca
        cb = c2s @ cb
    u = numpy.vstack([ca, cb])
    n2, n2c = u.shape
    u4 = numpy.zeros((n2*2, n2c*2), dtype=complex)
    u4[:n2, :n2c] = u
    u4[n2:, n2c:] = u
    return u4 @ dm @ u4.conj().T


class DHF(lib.StreamObject):
    '''Quadruple precision four-component Dirac-Hartree-Fock

    Attributes:
        integrals : str
            'quad' (quad precision integrals, default) or 'libcint' (libcint
            double precision spinor integrals promoted to quad precision)
        light_speed : float, str or mpmath number
            Speed of light.  Default is lib.param.LIGHT_SPEED (taken as an
            exact number).  A string is parsed in quad precision.
        conv_tol : float
            Energy convergence threshold of the quad precision iterations
        conv_tol_grad : float
            Threshold of the norm of the orbital gradient [F,D]
        init_guess : str
            'pyscf' (default) converges pyscf.scf.dhf.DHF in double precision
            first; 'double' converges the same Hamiltonian in double
            precision with the numpy implementation of this module.

    Saved results:
        e_tot : mpmath.mpf (113-bit mantissa)
        mo_energy, mo_coeff : QArray
        mo_occ : numpy array
    '''
    conv_tol = 1e-28
    conv_tol_grad = 1e-14
    max_cycle = 60
    diis_space = 8
    lindep = 1e-15
    init_guess = 'pyscf'

    def __init__(self, mol, integrals='quad', light_speed=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.integrals = integrals
        self.light_speed = light_speed
        self.converged = False
        self.e_tot = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.ham = None
        self.e_tot_double = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n******** %s ********', self.__class__)
        log.info('integrals = %s', self.integrals)
        log.info('light speed = %s', qmp.nstr(qmp.mpf(get_light_speed(self.light_speed)), 34))
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %g', self.conv_tol_grad)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('init_guess = %s', self.init_guess)
        return self

    def build_hamiltonian(self):
        if self.integrals == 'quad':
            self.ham = build_rkb_hamiltonian(self.mol, self.light_speed, self.verbose)
        elif self.integrals == 'libcint':
            self.ham = build_libcint_hamiltonian(self.mol, self.light_speed, self.verbose)
        else:
            raise ValueError('Unknown integrals %s' % self.integrals)
        return self.ham

    def _pyscf_guess(self):
        from pyscf.scf import dhf
        mol = self.mol
        mf = dhf.DHF(mol)
        mf.verbose = 0
        c = get_light_speed(self.light_speed)
        c0 = lib.param.LIGHT_SPEED
        try:
            lib.param.LIGHT_SPEED = float(c)
            mf.conv_tol = 1e-11
            mf.kernel()
        finally:
            lib.param.LIGHT_SPEED = c0
        self.e_tot_double = mf.e_tot
        dm = mf.make_rdm1()
        if self.integrals == 'quad':
            dm = spinor_to_spin_orbital_dm(mol, dm)
        return dm

    def kernel(self, dm0=None):
        log = logger.new_logger(self)
        self.dump_flags()
        t0 = (logger.process_clock(), logger.perf_counter())
        ham = self.build_hamiltonian()
        t0 = log.timer('Hamiltonian', *t0)
        nelec = self.mol.nelectron
        lindep = self.lindep

        ham_d = ham.to_double()
        if dm0 is None:
            if self.init_guess == 'pyscf':
                dm0 = self._pyscf_guess()
            else:
                # core guess, converged in double precision
                x, _ = _canonical_orth(_NumpyBackend, ham_d.ovlp, lindep)
                e, c1 = numpy.linalg.eigh(x.conj().T @ ham_d.hcore @ x)
                c = x @ c1
                nocc = nelec // ham_d.occ_number
                occ = _get_occ(_NumpyBackend, e, nocc, ham_d.light_speed, log)
                dm0 = _NumpyBackend.make_rdm1(c[:, occ[0]:occ[-1]+1]) * ham_d.occ_number
            conv, e, _, _, _, dm0 = scf_iterations(
                ham_d, _NumpyBackend, nelec, dm0, 1e-10, 1e-6, 100,
                self.diis_space, lindep, log)
            if not conv:
                log.warn('Double precision SCF not converged')
            self.e_tot_double = e
            t0 = log.timer('double precision SCF', *t0)

        if not isinstance(dm0, QArray):
            dm0 = numpy.asarray(dm0)
            if ham.hcore.iscomplex:
                dm0 = dm0.astype(complex)
            else:
                dm0 = dm0.real
            dm0 = QArray.from_numpy(dm0)
        conv, e, mo_e, mo_c, occ, dm = scf_iterations(
            ham, _QuadBackend, nelec, dm0, self.conv_tol, self.conv_tol_grad,
            self.max_cycle, self.diis_space, lindep, log)
        log.timer('quad precision SCF', *t0)
        self.converged = conv
        self.e_tot = qmp.mpf(e)
        self.mo_energy = mo_e
        self.mo_coeff = mo_c
        self.mo_occ = numpy.zeros(mo_e.shape[0])
        self.mo_occ[occ] = ham.occ_number
        name = self.__class__.__name__
        if conv:
            log.note('converged quad precision %s energy = %s', name,
                     qmp.nstr(self.e_tot, 34))
        else:
            log.note('quad precision %s not converged, energy = %s', name,
                     qmp.nstr(self.e_tot, 34))
        return self.e_tot

    def make_rdm1(self):
        occidx = numpy.where(self.mo_occ > 0)[0]
        c_occ = self.mo_coeff[:, occidx[0]:occidx[-1]+1]
        dm = _QuadBackend.make_rdm1(c_occ)
        if self.mo_occ[occidx[0]] != 1:
            dm = dm * int(self.mo_occ[occidx[0]])
        return dm

    @property
    def e_tot_str(self):
        return qmp.nstr(self.e_tot, 34)


# --------------------------------------------------------------------------
# nonrelativistic Hartree-Fock
# --------------------------------------------------------------------------
def build_nr_hamiltonian(mol, method='RHF', verbose=None):
    '''Nonrelativistic Hamiltonian with quad precision integrals.

    method = 'RHF': closed-shell restricted HF in the AO basis
    method = 'GHF': spin-orbital basis {chi_mu alpha, chi_mu beta}
    '''
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    qi = qints.QuadIntegrals(mol)
    n = qi.nao
    s = qi.ovlp_x()
    v = qi.nuc_x()
    sy = qi.ovlp_y()
    # kinetic energy T = 1/2 <d_a chi|d_a chi>
    t = (sy[0::3, 0::3] + sy[1::3, 1::3] + sy[2::3, 2::3]) * hmp.mpf(0.5)
    h = t + v
    t0 = log.timer('quad 1e integrals', *t0)
    eri = {('X', 'X'): qi.eri_xxxx()}
    t0 = log.timer('quad 2e integrals', *t0)
    e_nuc = qmp.mpf(qints.energy_nuc(mol))
    if method.upper() == 'RHF':
        comps = [(QArray.eye(n), 'X')]
        return Hamiltonian4c(h, s, comps, eri, e_nuc, None,
                             k_factor=0.5, occ_number=2)
    elif method.upper() == 'GHF':
        comps = []
        for spin in range(2):
            tm = numpy.zeros((2*n, n))
            tm[spin*n:(spin+1)*n] = numpy.eye(n)
            comps.append((QArray.from_numpy(tm), 'X'))
        hcore = QArray.zeros((2*n, 2*n))
        ovlp = QArray.zeros((2*n, 2*n))
        for spin in range(2):
            sl = slice(spin*n, (spin+1)*n)
            hcore[sl, sl] = h
            ovlp[sl, sl] = s
        return Hamiltonian4c(hcore, ovlp, comps, eri, e_nuc, None)
    raise ValueError('Unknown method %s' % method)


class RHF(DHF):
    '''Quadruple precision nonrelativistic closed-shell restricted
    Hartree-Fock.  The integrals, SCF and the results have the same layout
    as in :class:`DHF` (e_tot is an mpmath number, mo_energy/mo_coeff are
    QArray objects).

    init_guess = 'pyscf' takes the converged pyscf.scf.RHF density matrix
    as the initial guess; 'double' converges the quad Hamiltonian rounded to
    double precision first.
    '''
    _method = 'RHF'

    def __init__(self, mol):
        DHF.__init__(self, mol, integrals='quad', light_speed=None)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n******** %s ********', self.__class__)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %g', self.conv_tol_grad)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('init_guess = %s', self.init_guess)
        return self

    def build_hamiltonian(self):
        self.ham = build_nr_hamiltonian(self.mol, self._method, self.verbose)
        return self.ham

    def _pyscf_guess(self):
        from pyscf import scf
        mf = getattr(scf, self._method)(self.mol)
        mf.verbose = 0
        mf.conv_tol = 1e-11
        mf.kernel()
        self.e_tot_double = mf.e_tot
        return mf.make_rdm1()


class GHF(RHF):
    '''Quadruple precision nonrelativistic generalized (spin-orbital)
    Hartree-Fock.  The lowest mol.nelectron spin orbitals are occupied, so
    it can be used for open-shell systems.  For closed-shell molecules it
    gives the RHF energy.  The initial guess is converged by pyscf.scf.GHF.
    '''
    _method = 'GHF'
