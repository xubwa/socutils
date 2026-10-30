'''
State-averaged MR-LDSRG(2) for a spin-orbital / spinor Hamiltonian.

    Hbar = H + sum_k 1/k! [...[H, A], ..., A]_{0,1,2},   A = T - T^+,  T = T1 + T2

Every commutator is truncated to its scalar, one- and two-body parts in the generalized
normal order of the ensemble of the model states (recursive single-commutator
approximation); T is determined by the DSRG flow equation with flow parameter s,

    T(new) = [Hbar_od + T Delta] (1 - exp(-s Delta^2)) / Delta,

iterated to self-consistency.  The levels are the eigenvalues of the active-space part of
Hbar (reference relaxation): one transformation for all model states.

NIVO: the two-body blocks with three or four virtual indices of every commutator are
dropped; those of the bare Hamiltonian enter the first commutator, from the three-index
integrals (eris.cd_block).

    pt = LDSRG2(eris, nelec, civecs=civecs, s=0.5)
    pt.kernel()
    pt.e_tot        levels (eigenvalues of the relaxed active Hamiltonian)
    pt.civecs       their CI vectors;  pt.heff = diag(e_tot)

References: Li, Evangelista, J. Chem. Phys. 144, 164114 (2016); 148, 124106 (2018);
NIVO: Li, Evangelista, J. Chem. Phys. 150, 144107 (2019).  The commutator expressions
are those of forte (zmr.dsrg.commutator).
'''

import time
from itertools import product

import numpy

from socutils.zmr.active import rdm
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.dsrg import commutator as cm
from socutils.zmr.lib import linalg as la


def cumulants(act, civecs, nelec, weights=None, max_order=3):
    '''Averaged 1-RDM and cumulants of the model states in the layout of
    zmr.dsrg.commutator: gamma1[p, q] = <p^+ q>, lambda2 from <p^+ q^+ s r>,
    lambda3 from <p^+ q^+ r^+ u t s>.'''
    civecs = numpy.asarray(civecs)
    nstate = civecs.shape[1]
    if weights is None:
        weights = numpy.full(nstate, 1.0 / nstate)
    g = [0.0, 0.0, 0.0]
    for w, c in zip(weights, civecs.T):
        for k in range(1, max_order + 1):
            g[k - 1] = g[k - 1] + w * rdm.make_rdm(act, c[:, None], nelec, k, 0, 0)
    g1 = g[0]
    g2 = g[1].transpose(0, 2, 1, 3)
    l2 = (g2 - numpy.einsum('pr,qs->pqrs', g1, g1) + numpy.einsum('ps,qr->pqrs', g1, g1))
    if max_order < 3:
        return g1, l2, None
    g3 = g[2].transpose(0, 2, 4, 1, 3, 5)
    es = lambda sub, *ops: numpy.einsum(sub + '->pqrstu', *ops, optimize=True)
    l3 = (g3
          - es('ps,qrtu', g1, g2) + es('pt,qrsu', g1, g2) + es('pu,qrts', g1, g2)
          - es('qt,prsu', g1, g2) + es('qs,prtu', g1, g2) + es('qu,prst', g1, g2)
          - es('ru,pqst', g1, g2) + es('rs,pqut', g1, g2) + es('rt,pqsu', g1, g2)
          + 2 * (es('ps,qt,ru', g1, g1, g1) + es('pt,qu,rs', g1, g1, g1)
                 + es('pu,qs,rt', g1, g1, g1))
          - 2 * (es('ps,qu,rt', g1, g1, g1) + es('pu,qt,rs', g1, g1, g1)
                 + es('pt,qs,ru', g1, g1, g1)))
    return g1, l2, l3


class _ConjB(dict):
    """Conjugated three-index blocks of eris, read when first used."""

    def __init__(self, eris):
        super().__init__()
        self.eris = eris

    def __missing__(self, lab):
        self[lab] = b = numpy.conj(self.eris.cd_block(lab))
        return b


class LDSRG2:
    '''MR-LDSRG(2) with reference relaxation for the model states civecs.

    Args:
        eris: Hamiltonian with three-index integrals (eris.cd_block)
        nelec: number of active electrons
        civecs: model states as columns (default: the nstate lowest CASCI states)
        s: flow parameter in Eh^-2
        active_eri: (tu|vw) replacing the active block of the two-electron integrals,
            e.g. the picture-change corrected integrals; the model states, the Fock
            matrix and the transformation all use it

    Attributes:
        nivo: drop three- and four-virtual blocks of the commutators (default True;
            False needs the dense four-index integrals of all blocks)
        max_comm, comm_tol: length of the BCH series, norm of the last commutator
        conv_tol, conv_tol_amp: convergence of the energy and of the amplitudes
        max_cycle, diis_space
        e_corr: correlation energy of the ensemble (before relaxation)
        e_tot, civecs, heff: relaxed levels, their CI vectors, diag(e_tot)
    '''

    nivo = True
    max_comm = 20
    comm_tol = 1e-9
    conv_tol = 1e-8
    conv_tol_amp = 1e-6
    max_cycle = 60
    diis_space = 6
    single_precision_above = 2.0e10    # bytes of the pair-packed (vv|vv)
    packed_ladder = True
    verbose = True

    def __init__(self, eris, nelec, civecs=None, nstate=None, weights=None, s=0.5,
                 active_eri=None):
        self.eris = eris
        self.nelec = nelec
        self.s = s
        self.weights = weights
        self.active_eri = active_eri
        heff, aaaa, e_core = eris.active_hamiltonian()
        self._heff0, self._ecore0 = heff, e_core
        self._dv = None
        if active_eri is not None:
            self._dv = numpy.asarray(active_eri).reshape(aaaa.shape) - aaaa
            aaaa = aaaa + self._dv
        self.act = ActiveSpace(heff, aaaa, e_core, charges=getattr(eris, 'active_charges', None))
        if civecs is None:
            if nstate is None:
                raise ValueError('give civecs or nstate')
            civecs = self.act.solve(nelec, nstate)[1]
        self.ref_civecs = numpy.asarray(civecs)
        self.nstate = self.ref_civecs.shape[1]
        self.timings = {}
        self.converged = False
        self.t1 = self.t2 = None

    def log(self, text):
        if self.verbose:
            print('LDSRG2 ' + text, flush=True)

    # ------------------------------------------------------------------ setup
    def build(self):
        '''Semicanonical orbitals, cumulants and integrals of the reference.'''
        t0 = time.time()
        eris, act = self.eris, self.act
        sp = eris.spaces
        c = self.ref_civecs
        w = self.weights
        if w is None:
            w = numpy.full(self.nstate, 1.0 / self.nstate)
        hc = act.apply_h(self.nelec, c)
        self.e_ref = float(sum(wi * numpy.vdot(c[:, i], hc[:, i]).real
                               for i, wi in enumerate(w)) + act.e_core)
        g1, l2, l3 = cumulants(act, c, self.nelec, w)
        fock = eris.generalized_fock(g1)
        a = sp.slice('a')
        if self._dv is not None:
            fock = fock.copy()
            fock[a, a] += (numpy.einsum('pqtu,tu->pq', self._dv, g1)
                           - numpy.einsum('putq,tu->pq', self._dv, g1))
        u = numpy.eye(sp.nmo, dtype=numpy.complex128)
        eps = {}
        for x in 'cav':
            sl = sp.slice(x)
            if sl.stop > sl.start:
                eps[x], u[sl, sl] = la.eigh(numpy.ascontiguousarray(fock[sl, sl]))
            else:
                eps[x] = numpy.zeros(0)
        self.eps = eps
        self.u_act = ua = u[a, a].copy()
        sc = eris.rotate(u)
        fock = u.conj().T @ fock @ u
        self.dims = dims = {'c': sp.ncore, 'a': sp.nact, 'v': sp.nvirt}
        fs = numpy.conj(fock)
        self.F = cm.Blocks({x + y: fs[sp.slice(x), sp.slice(y)].copy()
                            for x, y in product('cav', repeat=2)})
        # operators a'^+_p = sum_t a^+_t u[t, p]
        uc = ua.conj()
        g1 = numpy.einsum('ip,ij,jq->pq', ua, g1, uc, optimize=True)
        l2 = numpy.einsum('ip,jq,ijkl,kr,ls->pqrs', ua, ua, l2, uc, uc, optimize=True)
        if l3 is not None:
            l3 = numpy.einsum('ip,jq,kr,ijklmn,ls,mt,nu->pqrstu', ua, ua, ua, l3, uc, uc, uc,
                              optimize=True)
        self.gamma1, self.lambda2 = g1, l2
        na = sp.nact
        self.cum = {'Gamma1': {'cc': cm.EYE, 'aa': g1},
                    'Eta1': {'aa': numpy.eye(na) - g1, 'vv': cm.EYE},
                    'Lambda2': {'aaaa': l2},
                    'Lambda3': None if l3 is None else {'aaaaaa': l3}}
        extra = {}
        if self._dv is not None:
            # (tu|vw) -> semicanonical active spinors -> conj(<tv||uw>)
            dv = numpy.einsum('ip,jq,ijkl,kr,ls->pqrs', uc, ua, self._dv, uc, ua, optimize=True)
            dv = dv.transpose(0, 2, 1, 3)
            extra['aaaa'] = numpy.conj(dv - dv.transpose(0, 1, 3, 2))
        self._w = None
        if getattr(sc, 'factorized', False):
            # three-index integrals, conjugated: V[pq, rs] = conj(<pq||rs>)
            self.V = cm.BareV(_ConjB(sc), dense_vvvv=not self.nivo, extra=extra)
            if self.nivo and self.packed_ladder and sp.nvirt > 1:
                npair = sp.nvirt * (sp.nvirt - 1) // 2
                self._w = self.V.packed_vvvv(
                    single=16.0 * npair**2 > self.single_precision_above)
        else:
            self.V = cm.DenseV(sc, extra=extra)
        self.log('reference: E(average) = %.10f; %d core + %d active + %d virtual; setup %.0f s'
                 % (self.e_ref, sp.ncore, sp.nact, sp.nvirt, time.time() - t0))
        return self

    # ------------------------------------------------------------- amplitudes
    @property
    def _t1_labels(self):
        return [h + p for h in 'ca' for p in 'av' if h + p != 'aa']

    @property
    def _t2_labels(self):
        return [x for x in (''.join(l) for l in product('ca', 'ca', 'av', 'av')) if x != 'aaaa']

    def _rden(self, lab):
        '''(1 - exp(-s D^2)) / D and exp(-s D^2) for the block lab of T.'''
        n = len(lab) // 2
        d = 0.0
        for k, x in enumerate(lab):
            shape = [1] * len(lab)
            shape[k] = -1
            d = d + (self.eps[x] if k < n else -self.eps[x]).reshape(shape)
        ex = numpy.exp(-self.s * d * d)
        small = abs(d) < 1e-7
        r = numpy.where(small, self.s * d, (1.0 - ex) / numpy.where(small, 1.0, d))
        return r, ex

    def init_amps(self):
        t1, t2 = cm.Blocks(), cm.Blocks()
        for lab in self._t1_labels:
            t1[lab] = self.F[lab] * self._rden(lab)[0]
        for lab in self._t2_labels:
            t2[lab] = self.V.get(lab) * self._rden(lab)[0]
        return t1, t2

    def _update(self, t, h, labels):
        new = cm.Blocks()
        for lab in labels:
            r, ex = self._rden(lab)
            new[lab] = t[lab] * (1.0 - ex)
            if h.get(lab) is not None:
                new[lab] += h[lab] * r
        return new

    def _ladder(self, t2, c2, alpha):
        '''c2[ij, rs] += alpha/2 sum_ab V[ab, rs] t2[ij, ab], four virtual indices.'''
        nv = self.dims['v']
        iu = numpy.triu_indices(nv, 1)
        for hh in ('cc', 'ca', 'aa'):
            t = t2.get(hh + 'vv')
            if t is None or t.size == 0:
                continue
            n0, n1 = t.shape[:2]
            tp = numpy.ascontiguousarray(t[:, :, iu[0], iu[1]]).reshape(n0 * n1, -1)
            x = (tp.astype(self._w.dtype, copy=False) @ self._w).reshape(n0, n1, -1)
            full = numpy.zeros((n0, n1, nv, nv), dtype=numpy.complex128)
            full[:, :, iu[0], iu[1]] = x
            full[:, :, iu[1], iu[0]] = -x
            if hh + 'vv' in c2:
                c2[hh + 'vv'] += alpha * full
            else:
                c2[hh + 'vv'] = alpha * full

    # -------------------------------------------------------------------- Hbar
    def _keep2(self, lab):
        return (not self.nivo) or lab.count('v') <= 2

    def make_hbar(self, t1, t2, comm_tol=None):
        '''(scalar, one-body, two-body hole-hole-particle-particle blocks) of Hbar;
        the scalar is the correlation energy of the ensemble.'''
        comm_tol = self.comm_tol if comm_tol is None else comm_tol
        hh_pp = set(self._t2_labels) | {'aaaa'}
        h0 = 0.0
        h1 = cm.Blocks({k: v.copy() for k, v in self.F.items()})
        h2 = cm.Blocks({lab: self.V.get(lab).copy() for lab in hh_pp})
        o1, o2 = self.F, self.V
        self.comm_norms = []
        for n in range(1, self.max_comm + 1):
            ladder = self._ladder if (n == 1 and self._w is not None) else None
            c0, c1, c2 = cm.commutator(o1, o2, t1, t2, self.cum, self.dims, 1.0 / n,
                                       self._keep2, ladder=ladder, timings=self.timings)
            cm.add_hc_1(c1)
            cm.add_hc_2(c2)
            h0 += 2.0 * numpy.real(c0)
            for lab, c in c1.items():
                h1[lab] += c
            for lab in hh_pp:
                c = c2.get(lab)
                if c is not None:
                    if c[1] == 1.0:
                        h2[lab] += c[0]
                    else:
                        h2[lab] -= c[0]
            norm = numpy.sqrt(c1.norm2() + sum(
                (1 + (k[0] != k[1])) * (1 + (k[2] != k[3])) * numpy.vdot(v, v).real
                for k, v in c2.items()))
            self.comm_norms.append(norm)
            o1, o2 = c1, c2
            if norm < comm_tol:
                break
        return h0, h1, h2

    # ------------------------------------------------------------------ kernel
    def solve_amplitudes(self):
        if self.t1 is None:
            t1, t2 = self.init_amps()
        else:
            t1, t2 = self.t1, self.t2
        diis_t, diis_e = [], []
        e_old = self.e_ref
        self.converged = False
        self.log('%4s %20s %10s %9s %9s %8s %8s %5s %8s' % (
            'iter', 'E(average)', 'dE', '|dT1|', '|dT2|', '|T1|', '|T2|', 'ncomm', 't / s'))
        for it in range(1, self.max_cycle + 1):
            t0 = time.time()
            h0, h1, h2 = self.make_hbar(t1, t2)
            e = self.e_ref + h0
            t1n = self._update(t1, h1, self._t1_labels)
            t2n = self._update(t2, h2, self._t2_labels)
            d1 = numpy.sqrt(sum(numpy.linalg.norm(t1n[k] - t1[k])**2 for k in t1n))
            d2 = numpy.sqrt(sum(numpy.linalg.norm(t2n[k] - t2[k])**2 for k in t2n))
            self.log('%4d %20.12f %10.2e %9.2e %9.2e %8.5f %8.5f %5d %8.1f' % (
                it, e, e - e_old, d1, d2, numpy.sqrt(t1.norm2()), numpy.sqrt(t2.norm2()),
                len(self.comm_norms), time.time() - t0))
            if abs(e - e_old) < self.conv_tol and max(d1, d2) < self.conv_tol_amp:
                # Hbar belongs to (t1, t2)
                self.converged = True
                break
            e_old = e
            if self.diis_space > 0:
                labs = [(self._t1_labels, t1, t1n), (self._t2_labels, t2, t2n)]
                vec = numpy.concatenate([n[k].ravel() for ll, o, n in labs for k in ll])
                old = numpy.concatenate([o[k].ravel() for ll, o, n in labs for k in ll])
                # history in single precision when large; the extrapolation only takes
                # differences from it
                dt = numpy.complex64 if vec.nbytes > 2**30 else numpy.complex128
                diis_t.append(vec.astype(dt))
                diis_e.append((vec - old).astype(dt))
                del old
                if len(diis_t) > self.diis_space:
                    diis_t.pop(0)
                    diis_e.pop(0)
                m = len(diis_t)
                if m > 1:
                    b = numpy.zeros((m + 1, m + 1))
                    for x in range(m):
                        for y in range(x, m):
                            b[x, y] = b[y, x] = numpy.vdot(diis_e[x], diis_e[y]).real
                    b[m, :m] = b[:m, m] = -1.0
                    rhs = numpy.zeros(m + 1)
                    rhs[m] = -1.0
                    try:
                        c = numpy.linalg.solve(b, rhs)[:m]
                    except numpy.linalg.LinAlgError:
                        c = numpy.zeros(m)
                        c[-1] = 1.0
                    for ck, tk in zip(c[:-1], diis_t[:-1]):
                        vec += ck * (tk - diis_t[-1])
                    pos = 0
                    for ll, o, n in labs:
                        for k in ll:
                            size = n[k].size
                            n[k] = vec[pos:pos + size].reshape(n[k].shape).copy()
                            pos += size
            t1, t2 = t1n, t2n
        if not self.converged:
            self.log('WARNING: amplitudes not converged in %d iterations' % self.max_cycle)
        self.log('norms of the nested commutators: ' + ' '.join('%.1e' % x for x in self.comm_norms))
        self.t1, self.t2 = t1, t2
        self.e_corr = h0
        self.e_dsrg = self.e_ref + h0
        self.hbar1 = h1['aa']
        self.hbar2 = h2['aaaa']
        return self.e_dsrg

    def relaxed_active_space(self):
        '''ActiveSpace of the active part of Hbar, in the active spinors of eris.'''
        g1, l2 = self.gamma1, self.lambda2
        h1, h2 = self.hbar1, self.hbar2
        e0 = (-numpy.einsum('uv,uv->', h1.conj(), g1)
              - 0.25 * numpy.einsum('uvxy,uvxy->', h2.conj(), l2)
              + 0.5 * numpy.einsum('uvxy,ux,vy->', h2.conj(), g1, g1)).real + self.e_dsrg
        h2 = numpy.conj(h2)                                          # <pq||rs>
        h1 = numpy.conj(h1) - numpy.einsum('uxvy,xy->uv', h2, g1)
        ua = self.u_act
        uc = ua.conj()
        h1 = numpy.einsum('ip,pq,jq->ij', ua, h1, uc, optimize=True)
        h2 = numpy.einsum('ip,jq,pqrs,kr,ls->ijkl', ua, ua, h2, uc, uc, optimize=True)
        # 1/4 <tv||uw> t^+ v^+ w u = 1/2 (tu|vw) t^+ v^+ w u with (tu|vw) = <tv||uw> / 2
        aaaa = 0.5 * h2.transpose(0, 2, 1, 3)
        return ActiveSpace(h1, aaaa, e0, charges=getattr(self.eris, 'active_charges', None))

    def kernel(self):
        t0 = time.time()
        self.build()
        self.solve_amplitudes()
        self.act_relaxed = self.relaxed_active_space()
        self.e_tot, self.civecs = self.act_relaxed.solve(self.nelec, self.nstate)
        self.heff = numpy.diag(self.e_tot).astype(numpy.complex128)
        self.log('E(average, unrelaxed) = %.10f, E(average, relaxed) = %.10f, %.0f s' % (
            self.e_dsrg, numpy.mean(self.e_tot), time.time() - t0))
        return self.e_tot
