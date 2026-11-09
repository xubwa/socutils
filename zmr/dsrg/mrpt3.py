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
with rows of the pair-packed (vv|vv) built from the three-index integrals as they are
used (nothing of size v^4 is stored).

    pt = DSRGMRPT3(eris, nelec, civecs=civecs, s=0.5)
    pt.kernel()

Reference: Li, Evangelista, J. Chem. Phys. 146, 124132 (2017).
'''

import time

import numpy

from socutils.symm import double_group as dg
from socutils.zmr import dpd
from socutils.zmr.dsrg import commutator as cm
from socutils.zmr.dsrg.ldsrg2 import LDSRG2
from socutils.zmr.dsrg.mrpt2 import DSRGMRPT2, _hole_particle
from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la


class _Sum:
    '''sum_k c_k X_k of two-body operators, hole-particle blocks only.'''

    dense_vvvv = False

    def __init__(self, *terms):
        self.terms = terms

    def get(self, lab):
        # formed when asked for, not kept
        if not _hole_particle(lab):
            return None
        out = None
        for c, op in self.terms:
            x = cm.materialize(op.get(lab))
            if x is not None:
                out = c * x if out is None else out + c * x
        return out


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

    vvvv = 'direct'
    mem_t2 = 4.5
    mem_v3 = 5.0

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
            x = cm.materialize(o2.get(lab))
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

    def _comm(self, o1, o2, t1, t2, ladder=None, large=None):
        '''[O, A]' (one- and two-body hole-particle blocks).'''
        t2c = None
        if self.dpd_storage:
            # T2 in the DPD layout, converted once per amplitude set
            cache = self.__dict__.setdefault('_t2c', {})
            if id(t2) not in cache or cache[id(t2)][0] is not t2:
                cache.clear()
                cache[id(t2)] = (t2, cm.dpd_blocks(t2, self.dims))
            t2c = cache[id(t2)][1]
        _, c1, c2 = cm.commutator(o1, o2, t1, t2, self.cum, self.dims, 1.0, _od,
                                  ladder=ladder, timings=self.timings, scalar=False,
                                  keep1=_od, large=large, dpd_out=self.dpd_storage,
                                  T2c=t2c)
        cm.add_hc_1(c1)
        # the excitation blocks only; the de-excitation blocks are their conjugates
        return c1, cm.HermitianHP.from_blocks(c2)

    def _three_virtual(self, C1, C2, Y, T1, T2, alpha):
        '''The terms of [V, T] with V[ab, ek] (three virtual indices, k core or active)
        and hole-particle output, one pass over V[ab, ek] in chunks of k:
            C1[i, e]    += alpha/2 V[ab, ek] T2[ij, ab] gamma[k, j]
            Y[i, k, p, q] += 1/2 T1[i, a] V[ak, pq]              V[ak, pq] = V[pq, ak]^*
            C2[ij, u, e] += alpha/2 V[ab, ue] T2[ij, ab]         V[ab, ue] = -V[ab, eu]
            C2[ij, r, s] += alpha V[ue, rs] T2[ij, ue]          V[ue, rs] = -V[rs, eu]^*
            C2[ij, r, s] -= alpha T2[ij, bx] gamma[x, y] V[by, rs]  V[by, rs] = V[rs, by]^*
        (forte: MRDSRG_SO, the terms of "abrk,ijab", "ia,arpq", "abrs,ijab", "byrs").'''
        d = self.dims
        nv = d['v']
        g = self.gamma1
        hs = [h for h in 'ca' if d[h]]
        # the two-body parts dense here (row slices of chunks), added to C2 and Y at the
        # end (in the DPD layout where those keep it)
        C2_out, Y_out = C2, Y
        C2, Y = {}, {}
        hh = [x + y for x in hs for y in hs if cm.is_canonical(x + y + 'vv')]
        sym = getattr(d, 'sym', None)
        if sym is not None:
            group, irs = sym
            pr = lambda p, q: numpy.asarray(dg.direct_prod(p, q, group)).reshape(-1)
            p_vv = pr(irs['v'], irs['v'])                               # (a, b)
            p_hh = {lab: pr(irs[lab[0]], irs[lab[1]]) for lab in hh}    # (i, j)
        for x in hs:
            # W[k, i, a, b] = sum_j T2[i, j, a, b] gamma[k, j]
            w = {}
            for i in hs:
                t = T2.get(i + x + 'vv')
                if t is None:
                    continue
                w[i] = (t.transpose(1, 0, 2, 3) if x == 'c' else
                        numpy.einsum('ijab,kj->kiab', t, g, optimize=True))
            ty = {}
            if x == 'a':
                for lab in hh:
                    t = T2.get(lab + 'va')
                    if t is not None:
                        ty[lab] = numpy.einsum('ijbx,xy->ijby', t, g, optimize=True)
            vsym = None if sym is None else (group, irs['v'], irs[x])
            for k0, k1, v in self.V.vvvx_chunks(x, self._available(), sym=vsym):  # v[a, b, e, k]
                nk = k1 - k0
                if sym is not None:
                    p_vk = pr(irs['v'], irs[x][k0:k1])                  # (e, k)
                    p_kv = pr(irs[x][k0:k1], irs['v'])                  # (k, e)
                vm = v.reshape(nv * nv, nv * nk)
                for i, wi in w.items():
                    ni = wi.shape[1]
                    wk = kernels.transposed(wi[k0:k1], (1, 0, 2, 3)).reshape(ni, -1)   # i, (k a b)
                    vk = kernels.transposed(v, (3, 0, 1, 2)).reshape(-1, nv)            # (k a b), e
                    c = 0.5 * alpha * la.gemm(wk, vk)
                    C1[i + 'v'] = C1[i + 'v'] + c if i + 'v' in C1 else c
                    # Y[i, k, p, q] = 1/2 conj(sum_a conj(T1[i, a]) V[p, q, a, k])
                    t1 = T1.get(i + 'v')
                    if t1 is not None:
                        vt = kernels.transposed(v, (2, 3, 0, 1)).reshape(nv, -1)       # a, (k p q)
                        y = 0.5 * la.gemm(numpy.ascontiguousarray(t1.conj()), vt).conj()
                        lab = i + x + 'vv'
                        if lab not in Y:
                            Y[lab] = numpy.zeros((d[i], d[x], nv, nv), dtype=complex)
                        Y[lab][:, k0:k1] += y.reshape(d[i], nk, nv, nv)
                if x != 'a':
                    continue
                for lab in hh:
                    t = T2.get(lab + 'vv')
                    if t is not None:
                        n0, n1 = t.shape[:2]
                        c = -0.5 * alpha * cm.blocked_gemm(
                            numpy.ascontiguousarray(t.reshape(n0 * n1, -1)), vm,
                            p_hh[lab] if sym else None, p_vv if sym else None,
                            p_vk if sym else None)
                        c = c.reshape(n0, n1, nv, nk).transpose(0, 1, 3, 2)
                        if lab + 'av' not in C2:
                            C2[lab + 'av'] = numpy.zeros((n0, n1, d['a'], nv), dtype=complex)
                        C2[lab + 'av'][:, :, k0:k1] += c
                    t = T2.get(lab + 'av')
                    if t is not None:
                        # a, b = active, virtual (both orders):
                        # alpha sum_ue V[ue, rs] T2[ij, ue], V[ue, rs] = -V[rs, eu]^*
                        n0, n1 = t.shape[:2]
                        tu = numpy.ascontiguousarray(t[:, :, k0:k1].conj()).reshape(n0 * n1, -1)
                        vu = kernels.transposed(v, (3, 2, 0, 1)).reshape(-1, nv * nv)
                        c = -alpha * cm.blocked_gemm(
                            tu, vu, p_hh[lab] if sym else None,
                            p_kv if sym else None, p_vv if sym else None
                        ).conj().reshape(n0, n1, nv, nv)
                        if lab + 'vv' not in C2:
                            C2[lab + 'vv'] = numpy.zeros((n0, n1, nv, nv), dtype=complex)
                        C2[lab + 'vv'] += c
                    if lab in ty:
                        n0, n1 = ty[lab].shape[:2]
                        # sum_by conj(ty)[ij, b, y] V[r, s, b, y], conjugated
                        tb = numpy.ascontiguousarray(ty[lab][:, :, :, k0:k1].conj()).reshape(n0 * n1, -1)
                        vb = kernels.transposed(v, (2, 3, 0, 1)).reshape(-1, nv * nv)
                        c = -alpha * cm.blocked_gemm(
                            tb, vb, p_hh[lab] if sym else None,
                            p_vk if sym else None, p_vv if sym else None
                        ).conj().reshape(n0, n1, nv, nv)
                        if lab + 'vv' not in C2:
                            C2[lab + 'vv'] = numpy.zeros((n0, n1, nv, nv), dtype=complex)
                        C2[lab + 'vv'] += c
                del v, vm
        for out, part in ((C2_out, C2), (Y_out, Y)):
            for lab in list(part):
                x = part.pop(lab)
                cur = dict.get(out, lab)
                if cur is None and getattr(out, 'dpd', False) and all(
                        c in getattr(d, 'legs', {}) for c in lab):
                    cur = out[lab] = dpd.Tensor([d.legs[c] for c in lab], cm.TWO_BODY, 2,
                                                d.sym[0])
                if isinstance(cur, dpd.Tensor):
                    cm.add_block(cur, x)
                elif cur is None:
                    out[lab] = x
                else:
                    cur += x

    def _scalar(self, o1, o2, t1, t2):
        return float(numpy.real(cm._scalar(
            o1, o2, t1, t2, self.cum['Gamma1'], self.cum['Eta1'], self.cum['Lambda2'],
            self.cum['Lambda3'],
            lambda spec, *ops, **kw: cm.contract(spec, *ops, dims=self.dims, **kw))))

    def _active(self, o1, o2, t1, t2, alpha):
        _, c1, c2 = cm.commutator(o1, o2, t1, t2, self.cum, self.dims, alpha,
                                  lambda lab: lab == 'aaaa', scalar=False, active=True)
        return c1, c2

    def _relax_part(self, o1, o2, t1, t2):
        '''Add the active part of 1/2 [O, T] to the effective Hamiltonian.'''
        if not self.relax:
            return
        c1, c2 = self._active(o1, o2, t1, t2, 0.5)
        for lab, v in c1.items():
            self._hb1[lab] = self._hb1[lab] + v if lab in self._hb1 else v
        for lab, v in c2.items():
            self._hb2[lab] = self._hb2[lab] + v if lab in self._hb2 else v

    def kernel(self):
        # every intermediate adds its part of the effective Hamiltonian when it is made
        # and is released after its last use
        t0 = time.time()
        t = time.time()
        self.build()
        self._stage('setup', t)
        self._hb1, self._hb2 = cm.Blocks(), cm.AntiBlocks()
        t = time.time()
        t1, t2, ft, vt = self.amplitudes()
        self._stage('T(1)', t)
        self.t1, self.t2 = t1, t2
        empty = cm.AntiBlocks()
        f0 = cm.Blocks({x + x: self.F[x + x] for x in 'cav'})
        f1 = cm.Blocks({lab: self.F[lab] for lab in self.F if _od(lab)})

        t = time.time()
        self.e_pt2 = self._scalar(ft, vt, t1, t2)
        self._relax_part(_sum1((1.0, ft)), _Sum((1.0, vt)), t1, t2)
        del ft, vt
        self._stage('E2', t)

        t = time.time()
        a1_1, a1_2 = self._comm(f0, empty, t1, t2)                      # [F0, A1]'
        aa1_1, aa1_2 = self._comm(a1_1, a1_2, t1, t2)                   # [[F0, A1]', A1]'
        e31 = -self._scalar(aa1_1, aa1_2, t1, t2) / 6.0
        self._stage('E3_1', t)

        t = time.time()
        large = None
        if isinstance(self.V, cm.BareV):
            self.V.skip_three_virtual = True
            large = self._three_virtual
        h1_1, h1_2 = self._comm(f1, self.V, t1, t2,
                                ladder=self._ladder if self._w is not None else None,
                                large=large)
        if isinstance(self.V, cm.BareV):
            self.V.skip_three_virtual = False
        self._stage('[H1, A1]', t)
        if getattr(self.V, 'timings', None):
            self.log('three-index parts: ' + ', '.join(
                '%s %.1f s' % kv for kv in sorted(self.V.timings.items())))
        rebuilt = {k: n for k, n in getattr(self.V, 'nbuild', {}).items() if n > 1}
        if rebuilt:
            self.log('integral blocks built more than once (cache_memory %.0f MB): %s'
                     % (self.V.cache_memory, rebuilt))
        t = time.time()
        # O2 = [[F0, A1]', A1]' + 2 [H1, A1]'; E3_3 = <[O2 / 2 + [F0, A2]', A1]> / 2 in two
        # parts, the first one here
        o2_1 = _sum1((1.0, aa1_1), (2.0, h1_1))
        o2_2 = _Sum((1.0, aa1_2), (2.0, h1_2))
        e33 = 0.5 * self._scalar(o2_1, o2_2, t1, t2)
        # -1/6 [[[F0, A1]', A1]', A1] + 1/2 [O2, A1] = [aa1 / 3 + h1, A1]
        self._relax_part(_sum1((1.0 / 3.0, aa1_1), (1.0, h1_1)),
                         _Sum((1.0 / 3.0, aa1_2), (1.0, h1_2)), t1, t2)
        u1, u2 = self._source_amplitudes(o2_1, o2_2, 0.5)                # T(2)
        del aa1_1, aa1_2, h1_1, h1_2, o2_1, o2_2
        self._stage('T(2)', t)

        t = time.time()
        x1 = _sum1((1.0, a1_1), (2.0, f1))
        x2 = _Sum((1.0, a1_2), (2.0, self.V))
        e32 = self._scalar(x1, x2, u1, u2)
        self._relax_part(x1, x2, u1, u2)
        del a1_1, a1_2, x1, x2
        self._stage('E3_2', t)

        t = time.time()
        b1, b2 = self._comm(f0, empty, u1, u2)                          # [F0, A2]'
        e33 += self._scalar(b1, b2, t1, t2)
        self._relax_part(b1, b2, t1, t2)
        del b1, b2, u1, u2
        self._stage('E3_3', t)

        self.e_pt3 = (e31, e32, e33)
        self.e_corr = self.e_pt2 + e31 + e32 + e33
        self.e_dsrg = self.e_ref + self.e_corr
        self.log('E(average) = %.10f; E2 %.10f, E3 %.10f %.10f %.10f' % (
            self.e_dsrg, self.e_pt2, e31, e32, e33))
        if not self.relax:
            self.e_tot = numpy.full(self.nstate, self.e_dsrg)
            return self.e_tot
        t = time.time()
        c1, c2 = self._hb1, self._hb2
        cm.add_hc_1(c1)
        cm.add_hc_2(c2)
        self.hbar1 = self.F['aa'] + c1['aa'] if 'aa' in c1 else self.F['aa'].copy()
        g = c2.get('aaaa')
        self.hbar2 = (cm.materialize(self.V.get('aaaa'))
                      + (cm.materialize(g) if g is not None else 0.0))
        self.act_relaxed = self.relaxed_active_space()
        self.e_tot, self.civecs = self.act_relaxed.solve(self.nelec, self.nstate)
        self.heff = numpy.diag(self.e_tot).astype(numpy.complex128)
        self._stage('relaxation', t)
        self.log('E(average, relaxed) = %.10f, %.1f s' % (numpy.mean(self.e_tot),
                                                          time.time() - t0))
        return self.e_tot
