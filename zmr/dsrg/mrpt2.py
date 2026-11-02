'''
State-averaged DSRG-MRPT2 with reference relaxation, spin-orbital / spinor Hamiltonian.

With the semicanonical orbitals of the ensemble and Delta = e_i + e_j - e_a - e_b of the
excitation (i j -> a b), the first-order amplitudes and renormalized integrals are

    T2[ij, ab] = V[ij, ab] (1 - exp(-s Delta^2)) / Delta
    T1[i, a]   = (F[i, a] + X[i, a]) (1 - exp(-s Delta^2)) / Delta,   X = [F0, T2]_1
    V~ = V (1 + exp(-s Delta^2)),   F~ = F + (F + X) exp(-s Delta^2)

on the excitation and de-excitation blocks (all hole-particle blocks but the active one),
F0 the active-active block of the Fock matrix.  The energy of the ensemble is

    E = E_ref + 1/2 <[H~, T - T^+]>_0

and the levels are the eigenvalues of the active part of H + 1/2 [H~, T - T^+]_{1,2}
(relax_reference 'once').  The core-core-virtual-virtual block of T2 is never stored:
its energy is summed over the core spinors, from the three-index integrals.

    pt = DSRGMRPT2(eris, nelec, civecs=civecs, s=0.5)
    pt.kernel()
    pt.e_dsrg       energy of the ensemble before the relaxation
    pt.e_tot        levels;  pt.civecs their CI vectors

References: Li, Evangelista, J. Chem. Theory Comput. 11, 2097 (2015);
J. Chem. Phys. 144, 164114 (2016).
'''

import time

import numpy

from socutils.zmr.dsrg import commutator as cm
from socutils.zmr.dsrg.ldsrg2 import LDSRG2
from socutils.zmr.lib import linalg as la


def _hole_particle(lab):
    '''Excitation (holes then particles) or de-excitation label, not all active.'''
    n = len(lab) // 2
    if lab == 'a' * len(lab):
        return False
    return ((all(x in 'ca' for x in lab[:n]) and all(x in 'av' for x in lab[n:]))
            or (all(x in 'av' for x in lab[:n]) and all(x in 'ca' for x in lab[n:])))


class _Renormalized:
    '''Two-body operator V with the hole-particle blocks multiplied by 1 + exp(-s D^2).'''

    def __init__(self, V, rden):
        self.V, self.rden = V, rden
        self.cache = {}
        self.dense_vvvv = V.dense_vvvv

    def get(self, lab):
        if not _hole_particle(lab):
            return self.V.get(lab)
        # one placement of every block, as BareV
        clab, perm, sign, cj = cm.hermitian_canonical(lab)
        if clab not in self.cache:
            self.cache[clab] = (cm.materialize(self.V.get(clab))
                                * (1.0 + self.rden(clab)[1]))
        c = self.cache[clab]
        return c if perm == (0, 1, 2, 3) else (c.transpose(perm), sign, cj)


class DSRGMRPT2(LDSRG2):
    '''DSRG-MRPT2 for the model states civecs (arguments as LDSRG2).

    Attributes:
        e_ref, e_corr, e_dsrg: reference, correlation and total energy of the ensemble
        e_ccvv: the core-core-virtual-virtual part of e_corr
        e_tot, civecs, heff: relaxed levels, their CI vectors, diag(e_tot)
        relax: False skips the reference relaxation (e_tot = e_dsrg)
    '''

    vvvv = None
    mem_t2 = 1.5
    mem_v3 = 0.0
    relax = True

    def log(self, text):
        if self.verbose:
            print('DSRG-MRPT2 ' + text, flush=True)

    @property
    def _t2_labels(self):
        return [lab for lab in LDSRG2._t2_labels.fget(self) if lab != 'ccvv']

    def _rden(self, lab):
        # the label of a de-excitation has the Delta of its excitation
        n = len(lab) // 2
        if all(x in 'av' for x in lab[:n]) and all(x in 'ca' for x in lab[n:]):
            lab = lab[n:] + lab[:n]
            r, ex = LDSRG2._rden(self, lab)
            perm = tuple(range(n, 2 * n)) + tuple(range(n))
            return r.transpose(perm), ex.transpose(perm)
        return LDSRG2._rden(self, lab)

    def amplitudes(self):
        '''First-order T1, T2 and the renormalized F~, V~.'''
        t2 = cm.Blocks()
        for lab in self._t2_labels:
            t2[lab] = cm.materialize(self.V.get(lab)) * self._rden(lab)[0]
        x = cm.Blocks()
        f0 = cm.Blocks({'aa': self.F['aa']})
        con = lambda spec, *ops, **kw: cm.contract(spec, *ops, dims=self.dims, **kw)
        g1 = self.cum['Gamma1']
        keep = lambda lab: lab in self._t1_labels
        con('ijab,bk,kj->ia', t2, f0, g1, out=x, keep=keep)
        con('ijau,vj,uv->ia', t2, f0, g1, out=x, scale=-1.0, keep=keep)
        t1 = cm.Blocks()
        ft = cm.Blocks({k: v.copy() for k, v in self.F.items()})
        for lab in self._t1_labels:
            r, ex = self._rden(lab)
            fx = self.F[lab] + x[lab] if lab in x else self.F[lab]
            t1[lab] = fx * r
            ft[lab] = self.F[lab] + fx * ex
            ft[lab[::-1]] = ft[lab].conj().T.copy()
        return t1, t2, ft, _Renormalized(self.V, self._rden)

    def energy_ccvv(self):
        '''1/4 sum_mnef |<mn||ef>|^2 (1 + exp(-s D^2)) (1 - exp(-s D^2)) / D, one core
        spinor m at a time (n < m).'''
        nc, nv = self.dims['c'], self.dims['v']
        if nc < 2 or nv < 2:
            return 0.0
        if not isinstance(self.V, cm.BareV):
            v = cm.materialize(self.V.get('ccvv'))
            r, ex = LDSRG2._rden(self, 'ccvv')
            return 0.25 * float(numpy.sum(abs(v)**2 * (1.0 + ex) * r))
        b = self.V.B['cv'][()]                        # (naux, nc, nv), read whole
        naux = b.shape[0]
        right = numpy.ascontiguousarray(b.reshape(naux, nc * nv))
        ec, ev = self.eps['c'], self.eps['v']
        s = self.s
        e = 0.0
        for m in range(1, nc):
            left = numpy.ascontiguousarray(b[:, m, :].T)               # e, P
            x = la.gemm(left, right[:, :m * nv]).reshape(nv, m, nv)     # (me|nf)[e, n, f]
            v = x - x.transpose(2, 1, 0)
            d = (ec[m] + ec[:m])[None, :, None] - ev[:, None, None] - ev[None, None, :]
            ex = numpy.exp(-s * d * d)
            small = abs(d) < 1e-7
            r = numpy.where(small, s * d, (1.0 - ex) / numpy.where(small, 1.0, d))
            e += float(numpy.sum((v.real**2 + v.imag**2) * (1.0 + ex) * r))
        return 0.5 * e

    def kernel(self):
        t0 = time.time()
        self.build()
        t1, t2, ft, vt = self.amplitudes()
        self.t1, self.t2 = t1, t2
        t = time.time()
        c0 = cm._scalar(ft, vt, t1, t2, self.cum['Gamma1'], self.cum['Eta1'],
                        self.cum['Lambda2'], self.cum['Lambda3'],
                        lambda spec, *ops, **kw: cm.contract(spec, *ops, dims=self.dims, **kw))
        self.timings['energy'] = time.time() - t
        t = time.time()
        self.e_ccvv = self.energy_ccvv()
        self.timings['energy ccvv'] = time.time() - t
        self.e_corr = float(numpy.real(c0)) + self.e_ccvv
        self.e_dsrg = self.e_ref + self.e_corr
        self.log('E(average) = %.10f, E_corr = %.10f (ccvv %.10f)' % (
            self.e_dsrg, self.e_corr, self.e_ccvv))
        if not self.relax:
            self.e_tot = numpy.full(self.nstate, self.e_dsrg)
            return self.e_tot
        t = time.time()
        _, c1, c2 = cm.commutator(ft, vt, t1, t2, self.cum, self.dims, 0.5,
                                  lambda lab: lab == 'aaaa', scalar=False, active=True)
        cm.add_hc_1(c1)
        cm.add_hc_2(c2)
        self.hbar1 = self.F['aa'] + c1['aa'] if 'aa' in c1 else self.F['aa'].copy()
        c, sign = c2.get('aaaa') if c2.get('aaaa') is not None else (0.0, 1.0)
        self.hbar2 = cm.materialize(self.V.get('aaaa')) + sign * c
        self.act_relaxed = self.relaxed_active_space()
        self.e_tot, self.civecs = self.act_relaxed.solve(self.nelec, self.nstate)
        self.heff = numpy.diag(self.e_tot).astype(numpy.complex128)
        self.timings['relaxation'] = time.time() - t
        self.log('E(average, relaxed) = %.10f, %.1f s' % (numpy.mean(self.e_tot),
                                                          time.time() - t0))
        return self.e_tot
