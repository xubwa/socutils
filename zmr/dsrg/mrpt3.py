'''
State-averaged DSRG-MRPT3 with reference relaxation, spin-orbital / spinor Hamiltonian.

H = F0 + H1 with F0 the diagonal blocks of the semicanonical Fock matrix; A = T - T^+;
[X, A]' is the commutator truncated to one- and two-body terms and projected on the
hole-particle (excitation and de-excitation) blocks.  With the first-order T(1) of
DSRG-MRPT2,

    O1  = [F0, A(1)]'
    O2  = [O1, A(1)]' + 2 [H1, A(1)]'
    T(2) from O2 / 2 as T(1) from H1 (DSRG source operator, same denominators)
    E   = E_ref + E2 + E3_1 + E3_2 + E3_3
    E3_1 = -1/6 <[[O1, A(1)]', A(1)]>,  E3_2 = 1/2 <[O1 + 2 H1, A(2)]>,
    E3_3 = 1/2 <[O2 / 2 + [F0, A(2)]', A(1)]>

(<[X, A]> = 2 Re of the scalar of [X, T]; E2 is the DSRG-MRPT2 energy.)  The levels are
the eigenvalues of the active part of H + the same commutators with one- and two-body
active output.  The four-virtual block of H1 enters [H1, A(1)] as one matrix product
with the pair-packed (vv|vv) (zmr.dsrg.ldsrg2).

    pt = DSRGMRPT3(eris, nelec, civecs=civecs, s=0.5)
    pt.kernel()

Reference: Li, Evangelista, J. Chem. Phys. 146, 124132 (2017).
'''

import time

import numpy

from socutils.zmr.dsrg import commutator as cm
from socutils.zmr.dsrg.ldsrg2 import LDSRG2
from socutils.zmr.dsrg.mrpt2 import DSRGMRPT2, _hole_particle


def _array(x):
    '''A block as returned by get (array, (array, sign), (array, sign, conj)).'''
    if x is None or isinstance(x, numpy.ndarray):
        return x
    blk, sign = x[0], x[1]
    if len(x) == 3 and x[2]:
        blk = blk.conj()
    return sign * blk


class _Sum:
    '''sum_k c_k X_k of two-body operators, hole-particle blocks only.'''

    dense_vvvv = False

    def __init__(self, *terms):
        self.terms = terms
        self.cache = {}

    def get(self, lab):
        if not _hole_particle(lab):
            return None
        if lab not in self.cache:
            out = None
            for c, op in self.terms:
                x = _array(op.get(lab))
                if x is not None:
                    out = c * x if out is None else out + c * x
            self.cache[lab] = out
        return self.cache[lab]


def _sum1(*terms):
    out = cm.Blocks()
    for c, op in terms:
        for lab, x in op.items():
            if not _hole_particle(lab):
                continue
            out[lab] = out[lab] + c * x if lab in out else c * x
    return out


def _od(lab):
    return _hole_particle(lab)


class DSRGMRPT3(DSRGMRPT2):
    '''DSRG-MRPT3 for the model states civecs (arguments as LDSRG2).

    Attributes:
        e_pt2, e_pt3 = (E3_1, E3_2, E3_3): energy contributions of the ensemble
        e_dsrg, e_tot, civecs, heff: as DSRGMRPT2
    '''

    packed_ladder = True

    def log(self, text):
        if self.verbose:
            print('DSRG-MRPT3 ' + text, flush=True)

    @property
    def _t2_labels(self):
        return LDSRG2._t2_labels.fget(self)

    def _source_amplitudes(self, o1, o2, factor):
        '''T from the hole-particle blocks of factor * O as T(1) from H1.'''
        t2 = cm.Blocks()
        for lab in self._t2_labels:
            x = _array(o2.get(lab))
            t2[lab] = (factor * x if x is not None else
                       numpy.zeros(tuple(self.dims[c] for c in lab), dtype=complex)
                       ) * self._rden(lab)[0]
        x = cm.Blocks()
        f0 = cm.Blocks({'aa': self.F['aa']})
        keep = lambda lab: lab in self._t1_labels
        cm.contract('ijab,bk,kj->ia', t2, f0, self.cum['Gamma1'], dims=self.dims, out=x,
                    keep=keep)
        cm.contract('ijau,vj,uv->ia', t2, f0, self.cum['Gamma1'], dims=self.dims, out=x,
                    scale=-1.0, keep=keep)
        t1 = cm.Blocks()
        for lab in self._t1_labels:
            y = factor * o1[lab] if lab in o1 else 0.0
            if lab in x:
                y = y + x[lab]
            t1[lab] = y * self._rden(lab)[0] if not numpy.isscalar(y) else \
                numpy.zeros(self.F[lab].shape, dtype=complex)
        return t1, t2

    def _comm(self, o1, o2, t1, t2, ladder=None):
        '''[O, A]' (one- and two-body hole-particle blocks).'''
        _, c1, c2 = cm.commutator(o1, o2, t1, t2, self.cum, self.dims, 1.0, _od,
                                  ladder=ladder, timings=self.timings, scalar=False)
        for lab in list(c1):
            if not _od(lab):
                del c1[lab]
        cm.add_hc_1(c1)
        cm.add_hc_2(c2)
        return c1, c2

    def _scalar(self, o1, o2, t1, t2):
        return float(numpy.real(cm._scalar(
            o1, o2, t1, t2, self.cum['Gamma1'], self.cum['Eta1'], self.cum['Lambda2'],
            self.cum['Lambda3'],
            lambda spec, *ops, **kw: cm.contract(spec, *ops, dims=self.dims, **kw))))

    def _active(self, o1, o2, t1, t2, alpha):
        _, c1, c2 = cm.commutator(o1, o2, t1, t2, self.cum, self.dims, alpha,
                                  lambda lab: lab == 'aaaa', scalar=False, active=True)
        return c1, c2

    def kernel(self):
        t0 = time.time()
        self.build()
        tim = self.timings
        t1, t2, ft, vt = self.amplitudes()
        self.t1, self.t2 = t1, t2
        empty = cm.AntiBlocks()
        f0 = cm.Blocks({x + x: self.F[x + x] for x in 'cav'})
        f1 = cm.Blocks({lab: self.F[lab] for lab in self.F if _od(lab)})

        t = time.time()
        self.e_pt2 = self._scalar(ft, vt, t1, t2)
        tim['E2'] = time.time() - t
        t = time.time()
        a1_1, a1_2 = self._comm(f0, empty, t1, t2)                      # [F0, A1]'
        aa1_1, aa1_2 = self._comm(a1_1, a1_2, t1, t2)                   # [[F0, A1]', A1]'
        e31 = -self._scalar(aa1_1, aa1_2, t1, t2) / 6.0
        tim['E3_1'] = time.time() - t
        t = time.time()
        h1_1, h1_2 = self._comm(f1, self.V, t1, t2,
                                ladder=self._ladder if self._w is not None else None)
        tim['[H1, A1]'] = time.time() - t
        o2_1 = _sum1((1.0, aa1_1), (2.0, h1_1))
        o2_2 = _Sum((1.0, aa1_2), (2.0, h1_2))
        del h1_1, h1_2
        t = time.time()
        u1, u2 = self._source_amplitudes(o2_1, o2_2, 0.5)                # T(2)
        x1 = _sum1((1.0, a1_1), (2.0, f1))
        x2 = _Sum((1.0, a1_2), (2.0, self.V))
        e32 = self._scalar(x1, x2, u1, u2)
        tim['E3_2'] = time.time() - t
        t = time.time()
        b1, b2 = self._comm(f0, empty, u1, u2)                          # [F0, A2]'
        y1 = _sum1((0.5, o2_1), (1.0, b1))
        y2 = _Sum((0.5, o2_2), (1.0, b2))
        e33 = self._scalar(y1, y2, t1, t2)
        tim['E3_3'] = time.time() - t
        self.e_pt3 = (e31, e32, e33)
        self.e_corr = self.e_pt2 + e31 + e32 + e33
        self.e_dsrg = self.e_ref + self.e_corr
        self.log('E(average) = %.10f; E2 %.10f, E3 %.10f %.10f %.10f' % (
            self.e_dsrg, self.e_pt2, e31, e32, e33))
        if not self.relax:
            self.e_tot = numpy.full(self.nstate, self.e_dsrg)
            return self.e_tot
        t = time.time()
        ft_od = _sum1((1.0, ft))
        vt_od = _Sum((1.0, vt))
        z1 = _sum1((1.0, ft_od), (0.5, o2_1), (1.0, b1), (-1.0 / 6.0, aa1_1))
        z2 = _Sum((1.0, vt_od), (0.5, o2_2), (1.0, b2), (-1.0 / 6.0, aa1_2))
        c1, c2 = self._active(z1, z2, t1, t2, 0.5)
        d1, d2 = self._active(x1, x2, u1, u2, 0.5)
        for lab, v in d1.items():
            c1[lab] = c1[lab] + v if lab in c1 else v
        for lab in list(d2):
            if lab in c2:
                c2[lab] = c2[lab] + d2[lab]
            else:
                c2[lab] = d2[lab]
        cm.add_hc_1(c1)
        cm.add_hc_2(c2)
        self.hbar1 = self.F['aa'] + c1['aa'] if 'aa' in c1 else self.F['aa'].copy()
        g = c2.get('aaaa')
        self.hbar2 = self.V.get('aaaa') + (_array(g) if g is not None else 0.0)
        self.act_relaxed = self.relaxed_active_space()
        self.e_tot, self.civecs = self.act_relaxed.solve(self.nelec, self.nstate)
        self.heff = numpy.diag(self.e_tot).astype(numpy.complex128)
        tim['relaxation'] = time.time() - t
        self.log('E(average, relaxed) = %.10f, %.1f s' % (numpy.mean(self.e_tot),
                                                          time.time() - t0))
        return self.e_tot
