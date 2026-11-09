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
integrals (eris.cd_block).  The pair-packed (vv|vv) of that ladder is built once from the
three-index integrals and kept in memory, or written to a temporary HDF5 file and read in
row blocks, when it does not fit in max_memory.

    pt = LDSRG2(eris, nelec, civecs=civecs, s=0.5)
    pt.kernel()
    pt.e_tot        levels (eigenvalues of the relaxed active Hamiltonian)
    pt.civecs       their CI vectors;  pt.heff = diag(e_tot)

References: Li, Evangelista, J. Chem. Phys. 144, 164114 (2016); 148, 124106 (2018);
NIVO: Li, Evangelista, J. Chem. Phys. 150, 144107 (2019).  The commutator expressions
are those of forte (zmr.dsrg.commutator).
'''

import resource
import time
from itertools import product

import numpy
from pyscf import lib

from socutils.zmr.active import rdm
from socutils.zmr.active.sectors import ActiveSpace
from socutils.symm import double_group as dg
from socutils.zmr import dpd, symmetry_sectors
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
        use_symmetry: with eris.orbsym (double-group sectors, zmr.symmetry_sectors), the
            orbitals of every space are grouped by sector and the contractions skip the
            blocks that vanish by symmetry (default True; NdO+, Coov*: MRPT3 1.6x,
            MR-LDSRG(2) iterations 3x faster, same energies)
        dpd_storage: with use_symmetry, the two-body parts of the nested commutators and
            their intermediates in the DPD layout (zmr.dpd: only the blocks allowed by
            symmetry are stored; default True)
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
    max_memory = lib.param.MAX_MEMORY  # MB
    # skip the blocks that vanish by double-group symmetry (eris.orbsym)
    use_symmetry = True
    # with the sectors: the two-body parts of the nested commutators in the DPD layout
    dpd_storage = True
    # (vv|vv) ladder: 'stored' (in memory or on disk, applied in every iteration),
    # 'direct' (rebuilt from the three-index integrals for each use), None (not needed)
    vvvv = 'stored'
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

    # arrays of the size of T2 held at the same time, and v^3 blocks of the smallest batch
    # (memory_estimate)
    mem_t2 = 8.0
    mem_v3 = 1.5

    read_memory = 1000.0               # MB of three-index integrals read at a time, at least

    def memory_estimate(self, naux):
        '''MB the calculation needs beyond what is in use after the semicanonicalization:
        mem_t2 arrays of the size of T2, mem_v3 blocks of v^3 (the smallest batch), the
        largest three-index block while it is written to disk, the read buffer, the
        integral blocks with at most two virtual indices (cached), 10 % for the rest.'''
        sp = self.eris.spaces
        nh, npart = sp.ncore + sp.nact, sp.nact + sp.nvirt
        t2 = 16e-6 * nh**2 * npart**2
        return 1.1 * (self.mem_t2 * t2 + self.mem_v3 * 16e-6 * sp.nvirt**3
                      + 16e-6 * naux * sp.nvirt**2 + self.read_memory
                      + self._cached_blocks(naux))

    def _cached_blocks(self, naux):
        '''MB of the integral blocks with at most two virtual indices (one placement each),
        the working set of the cache.'''
        if not naux:
            return 0.0
        n = {'c': self.eris.spaces.ncore, 'a': self.eris.spaces.nact,
             'v': self.eris.spaces.nvirt}
        labs = {cm.hermitian_canonical(''.join(x))[0] for x in product('cav', repeat=4)}
        return sum(16e-6 * n[l[0]] * n[l[1]] * n[l[2]] * n[l[3]]
                   for l in labs if l.count('v') <= 2)

    def _available(self, fraction=0.5):
        '''MB for a batch: fraction of max_memory less what is in use (at least 100).'''
        return max(100.0, fraction * (self.max_memory - lib.current_memory()[0]))

    def _stage(self, name, t0):
        '''Record the time of a stage since t0 and log it with the memory in use.'''
        self.timings[name] = self.timings.get(name, 0.0) + time.time() - t0
        self.log('%-12s %8.1f s, memory %7.0f MB (peak %7.0f MB)' % (
            name, time.time() - t0, lib.current_memory()[0],
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6))

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
        # semicanonical orbitals within the double-group sectors, if any
        sym = getattr(eris, 'orbsym', None)
        self.orbsym = {}
        for x in 'cav':
            sl = sp.slice(x)
            if sl.stop > sl.start:
                # with sectors: grouped, every sector of a space a range of orbitals
                eps[x], u[sl, sl], self.orbsym[x] = symmetry_sectors.eigh(
                    fock[sl, sl], None if sym is None else numpy.asarray(sym)[sl],
                    grouped=True)
            else:
                eps[x], self.orbsym[x] = numpy.zeros(0), None
        if sym is None:
            self.orbsym = None
        self.eps = eps
        self.u_act = ua = u[a, a].copy()
        sc = eris.rotate(u)
        fock = u.conj().T @ fock @ u
        # with double-group sectors the contractions skip the blocks that vanish by symmetry
        sym = None
        if self.orbsym is not None and self.use_symmetry:
            sym = (eris.doublegroupname, self.orbsym)
        self.dims = dims = cm.Dims({'c': sp.ncore, 'a': sp.nact, 'v': sp.nvirt}, sym=sym)
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
        self.cum = {'Gamma1': cm.with_signs({'cc': cm.EYE, 'aa': g1}, cm.DENSITY),
                    'Eta1': cm.with_signs({'aa': numpy.eye(na) - g1, 'vv': cm.EYE}, cm.DENSITY),
                    'Lambda2': cm.with_signs({'aaaa': l2}, cm.CUMULANT2),
                    'Lambda3': None if l3 is None else cm.with_signs({'aaaaaa': l3},
                                                                     cm.CUMULANT3)}
        extra = {}
        if self._dv is not None:
            # (tu|vw) -> semicanonical active spinors -> conj(<tv||uw>)
            dv = numpy.einsum('ip,jq,ijkl,kr,ls->pqrs', uc, ua, self._dv, uc, ua, optimize=True)
            dv = dv.transpose(0, 2, 1, 3)
            extra['aaaa'] = numpy.conj(dv - dv.transpose(0, 1, 3, 2))
        self._w = None
        factorized = getattr(sc, 'factorized', False)
        naux = sc.cd_block('cc').shape[0] if factorized else 0
        self.min_memory = lib.current_memory()[0] + self.memory_estimate(naux)
        self.log('memory: at least %.0f MB needed (%.0f MB in use), max_memory %.0f MB'
                 % (self.min_memory, lib.current_memory()[0], self.max_memory))
        if self.max_memory < self.min_memory:
            raise MemoryError('%s needs max_memory >= %.0f MB (max_memory = %.0f MB)'
                              % (type(self).__name__, self.min_memory, self.max_memory))
        if factorized:
            # three-index integrals on disk (conjugated when read: V[pq, rs] = conj(<pq||rs>)); the
            # cache of integral blocks gets half of the memory beyond the minimum
            spare = self.max_memory - self.min_memory
            self.V = cm.BareV(cm.ThreeIndexFile(sc), dense_vvvv=not self.nivo, extra=extra,
                              cache_memory=self._cached_blocks(naux) + 0.5 * spare,
                              read_memory=max(self.read_memory, min(4000.0, 0.1 * spare)))
            if self.dpd_storage and dims.legs and dims.sym is not None:
                # integral blocks in the DPD layout: only the elements allowed by symmetry
                self.V.legs, self.V.group = dims.legs, dims.sym[0]
            if self.nivo and self.vvvv and sp.nvirt > 1:
                self._w = self._store_vvvv() if self.vvvv == 'stored' else 'direct'
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

    def _dpd_amps(self, lab):
        '''Whether the amplitudes (and Hbar) of the two-body block lab are kept in the DPD
        layout: with dpd_storage and the double-group sectors of all its spaces.'''
        return (self.dpd_storage and len(lab) == 4 and getattr(self.dims, 'sym', None)
                is not None and all(x in self.dims.legs for x in lab))

    def _t2_tensor(self, lab, x=None):
        '''A dpd.Tensor of the two-body block lab (zero; with x, the block x as returned
        by get).'''
        legs = self.dims.legs
        t = dpd.Tensor([legs[c] for c in lab], cm.TWO_BODY, 2, self.dims.sym[0],
                       empty=x is not None)
        if x is not None:
            cm.add_block(t, x, beta=0.0)
        return t

    def _rden_dpd(self, lab):
        '''_rden(lab) as two dpd.Tensors of the layout of _t2_tensor(lab), computed block
        by block from the orbital energies; kept (the energies are fixed after build).'''
        cache = self.__dict__.setdefault('_rden_cache', {})
        key = (lab, id(self.eps))
        if key in cache:
            return cache[key]
        legs = [self.dims.legs[x] for x in lab]
        r, ex = self._t2_tensor(lab), self._t2_tensor(lab)

        def den(sec):
            d = 0.0
            for k, x in enumerate(lab):
                e = self.eps[x][legs[k].offsets[sec[k]]:legs[k].offsets[sec[k] + 1]]
                shape = [1] * 4
                shape[k] = -1
                d = d + (e if k < 2 else -e).reshape(shape)
            return d

        def fill_r(sec, blk):
            d = den(sec)
            small = abs(d) < 1e-7
            blk[...] = numpy.where(small, self.s * d, (1.0 - numpy.exp(-self.s * d * d))
                                   / numpy.where(small, 1.0, d))

        def fill_ex(sec, blk):
            d = den(sec)
            blk[...] = numpy.exp(-self.s * d * d)
        dpd.map_blocks(r, fill_r)
        dpd.map_blocks(ex, fill_ex)
        cache[key] = r, ex
        return r, ex

    def init_amps(self):
        t1, t2 = cm.Blocks(), cm.Blocks()
        for lab in self._t1_labels:
            t1[lab] = self.F[lab] * self._rden(lab)[0]
        for lab in self._t2_labels:
            if self._dpd_amps(lab):
                t = self._t2_tensor(lab, self.V.get(lab))
                t.data *= self._rden_dpd(lab)[0].data
                t2[lab] = t
            else:
                t2[lab] = cm.materialize(self.V.get(lab)) * self._rden(lab)[0]
        return t1, t2

    def _update(self, t, h, labels):
        new = cm.Blocks()
        for lab in labels:
            if isinstance(t[lab], dpd.Tensor):
                r, ex = self._rden_dpd(lab)
                n = new[lab] = self._t2_tensor(lab)
                n.data[...] = t[lab].data * (1.0 - ex.data)
                hl = h.get(lab)
                if hl is not None:
                    if not isinstance(hl, dpd.Tensor):
                        hl = self._t2_tensor(lab, hl)
                    n.data += hl.data * r.data
                continue
            r, ex = self._rden(lab)
            new[lab] = t[lab] * (1.0 - ex)
            if h.get(lab) is not None:
                new[lab] += h[lab] * r
        return new

    def _vvvv_pair_groups(self):
        '''{pair irrep: indices of the pairs a < b of that irrep (in the order of
        numpy.triu_indices)} of the virtual orbitals, None without the sectors.'''
        vsym = self._virtual_sym()
        if vsym is None:
            return None
        nv = self.dims['v']
        pirr = numpy.asarray(dg.direct_prod(vsym[1], vsym[1], vsym[0]))[numpy.triu_indices(nv, 1)]
        return {int(g): numpy.nonzero(pirr == g)[0] for g in numpy.unique(pirr)}

    def _store_vvvv(self):
        '''The pair-packed V[ab, rs] (a < b, r < s), in memory if it takes less than
        half of the memory beyond the minimum, else in a temporary HDF5 file.  With the
        double-group sectors only its diagonal blocks by the irrep of the pairs:
        {pair irrep: V[ab, rs] of the pairs of that irrep}.'''
        t0 = time.time()
        nv = self.dims['v']
        npair = nv * (nv - 1) // 2
        groups = self._vvvv_pair_groups()
        if groups is not None:
            return self._store_vvvv_sym(groups, t0)
        size = 16.0 * npair**2 / 1e6
        batch = cm.BareV.vvvv_batch(nv, self._available())
        if size < 0.5 * (self.max_memory - self.min_memory):
            w = numpy.empty((npair, npair), dtype=complex)
            where = 'memory'
        else:
            self._vvvv_file = lib.H5TmpFile()
            w = self._vvvv_file.create_dataset('vvvv', (npair, npair), 'c16')
            where = 'file %s' % self._vvvv_file.filename
        for r0, rows in self.V.vvvv_rows(batch, sym=self._virtual_sym()):
            w[r0:r0 + len(rows)] = rows
        self.log('(vv|vv) of %d pairs (%.0f MB) in %s, %.0f s' % (
            npair, size, where, time.time() - t0))
        return w

    def _store_vvvv_sym(self, groups, t0):
        nv = self.dims['v']
        npair = nv * (nv - 1) // 2
        size = sum(16.0 * len(ix)**2 for ix in groups.values()) / 1e6
        pirr = numpy.empty(npair, dtype=int)
        pos = numpy.empty(npair, dtype=numpy.int64)
        for g, ix in groups.items():
            pirr[ix] = g
            pos[ix] = numpy.arange(len(ix))
        if size < 0.5 * (self.max_memory - self.min_memory):
            w = {g: numpy.empty((len(ix), len(ix)), dtype=complex) for g, ix in groups.items()}
            where = 'memory'
        else:
            self._vvvv_file = lib.H5TmpFile()
            w = {g: self._vvvv_file.create_dataset('vvvv%d' % g, (len(ix), len(ix)), 'c16')
                 for g, ix in groups.items()}
            where = 'file %s' % self._vvvv_file.filename
        # the row batches of the construction within the memory left by the blocks
        mem = self._available() - (size if where == 'memory' else 0.0)
        batch = cm.BareV.vvvv_batch(nv, max(100.0, mem))
        for r0, rows in self.V.vvvv_rows(batch, sym=self._virtual_sym()):
            rows = numpy.ascontiguousarray(rows, dtype=complex)
            rid = numpy.arange(r0, r0 + len(rows))
            for g in numpy.unique(pirr[rid]):
                sel = numpy.nonzero(pirr[rid] == g)[0]
                blk = lib.take_2d(rows, sel, groups[int(g)])
                p = pos[rid[sel]]
                if where == 'memory':
                    w[int(g)][p] = blk
                else:
                    # increasing positions: one write of the rows between the first and last
                    lo, hi = int(p[0]), int(p[-1]) + 1
                    if hi - lo == len(p):
                        w[int(g)][lo:hi] = blk
                    else:
                        w[int(g)][list(p)] = blk
        self.log('(vv|vv) of %d pairs in %d blocks by the irrep of the pairs (%.0f MB, %.0f MB '
                 'without symmetry) in %s, %.0f s' % (npair, len(groups), size,
                                                    16.0 * npair**2 / 1e6, where,
                                                    time.time() - t0))
        return w

    def _vvvv_blocks(self):
        '''(first row, rows) of the pair-packed V[ab, rs], within the free memory.'''
        nv = self.dims['v']
        if isinstance(self._w, str):
            yield from self.V.vvvv_rows(cm.BareV.vvvv_batch(nv, self._available()),
                                        sym=self._virtual_sym())
            return
        if isinstance(self._w, numpy.ndarray):
            yield 0, self._w
            return
        npair = self._w.shape[0]
        step = max(1, int(self._available() * 1e6 / (16.0 * npair)))
        for r0 in range(0, npair, step):
            yield r0, self._w[r0:r0 + step]

    def _virtual_sym(self):
        '''(doublegroupname, irreps of the virtual orbitals) when the contractions use
        the double-group sectors, else None.'''
        sym = getattr(self.dims, 'sym', None)
        return None if sym is None else (sym[0], sym[1]['v'])

    def _ladder(self, t2, c2, alpha):
        '''c2[ij, rs] += alpha/2 sum_ab V[ab, rs] t2[ij, ab], four virtual indices: one
        pass over the rows of the pair-packed (vv|vv) for all hole pairs.  With the
        double-group sectors, the product is blocked by the irrep of the pairs: t2[ij, ab]
        and V[ab, rs] vanish unless ij, ab, rs have the same pair irrep.'''
        nv = self.dims['v']
        iu = numpy.triu_indices(nv, 1)
        # the hole-hole-vv blocks dense (those kept in the DPD layout expanded here)
        t2 = {hh + 'vv': cm.materialize(t2.get(hh + 'vv')) for hh in ('cc', 'ca', 'aa')}
        hhs = [hh for hh in ('cc', 'ca', 'aa')
               if t2.get(hh + 'vv') is not None and t2[hh + 'vv'].size]
        if not hhs:
            return
        tp = numpy.ascontiguousarray(numpy.concatenate(
            [t2[hh + 'vv'][:, :, iu[0], iu[1]].reshape(-1, iu[0].size) for hh in hhs]),
            dtype=numpy.complex128)
        x = numpy.zeros(tp.shape, dtype=numpy.complex128)
        sym = self.dims.sym if getattr(self.dims, 'sym', None) else None
        if sym is not None:
            group, irs = sym
            pv = numpy.asarray(dg.direct_prod(irs['v'], irs['v'], group))[iu]
            ph = numpy.concatenate([numpy.asarray(dg.direct_prod(irs[hh[0]], irs[hh[1]],
                                                                 group)).reshape(-1)
                                    for hh in hhs])
            blocks = [(numpy.nonzero(ph == p_)[0], numpy.nonzero(pv == p_)[0])
                      for p_ in numpy.unique(pv)]
            blocks = [(ij, ab) for ij, ab in blocks if ij.size]
        tim = getattr(self.V, 'timings', {})
        if isinstance(self._w, dict):
            # the diagonal blocks by pair irrep: x[ij, rs] = sum_ab t[ij, ab] W_g[ab, rs]
            groups = self._vvvv_pair_groups()
            t = time.time()
            for g, wg in self._w.items():
                ij = numpy.nonzero(ph == g)[0]
                ab = groups[g]
                if ij.size == 0 or ab.size == 0:
                    continue
                tg = lib.take_2d(tp, ij, ab)
                if isinstance(wg, numpy.ndarray):
                    xg = la.gemm(tg, wg)
                else:
                    xg = numpy.zeros((ij.size, ab.size), dtype=numpy.complex128)
                    step = max(1, int(self._available() * 1e6 / (16.0 * ab.size)))
                    for r0 in range(0, ab.size, step):
                        xg += la.gemm(numpy.ascontiguousarray(tg[:, r0:r0 + step]),
                                      wg[r0:r0 + step])
                lib.takebak_2d(x, xg, ij, ab)
            if isinstance(tim, dict):
                tim['vvvv apply'] = tim.get('vvvv apply', 0.0) + time.time() - t
        for r0, rows in ([] if isinstance(self._w, dict) else self._vvvv_blocks()):
            t = time.time()
            if sym is None:
                x += la.gemm(numpy.ascontiguousarray(tp[:, r0:r0 + len(rows)]), rows)
                if isinstance(tim, dict):
                    tim['vvvv apply'] = tim.get('vvvv apply', 0.0) + time.time() - t
                continue
            rows = numpy.asarray(rows)
            for ij, ab in blocks:
                loc = ab[(ab >= r0) & (ab < r0 + len(rows))]
                if loc.size == 0:
                    continue
                lib.takebak_2d(x, la.gemm(lib.take_2d(tp, ij, loc),
                                          lib.take_2d(rows, loc - r0, ab)), ij, ab)
            if isinstance(tim, dict):
                tim['vvvv apply'] = tim.get('vvvv apply', 0.0) + time.time() - t
        del tp
        pos = 0
        for hh in hhs:
            n0, n1 = t2[hh + 'vv'].shape[:2]
            xb = x[pos:pos + n0 * n1].reshape(n0, n1, -1)
            pos += n0 * n1
            full = numpy.zeros((n0, n1, nv, nv), dtype=numpy.complex128)
            full[:, :, iu[0], iu[1]] = xb
            full[:, :, iu[1], iu[0]] = -xb
            lab = hh + 'vv'
            cur = dict.get(c2, lab)
            if cur is None and getattr(c2, 'dpd', False):
                legs = self.dims.legs
                cur = c2[lab] = dpd.Tensor([legs[x] for x in lab], cm.TWO_BODY, 2,
                                           self.dims.sym[0])
            if isinstance(cur, dpd.Tensor):
                cm._add_dense(cur, full, alpha)
            elif cur is not None:
                cur += alpha * full
            else:
                c2[lab] = alpha * full

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
        # contiguous copies: never the cached integral blocks themselves (h2 is updated in
        # place), and C order (numpy.array keeps the layout of a transposed view)
        h2 = cm.Blocks({lab: self._t2_tensor(lab, self.V.get(lab)) if (
            self._dpd_amps(lab) and isinstance(t2.get(lab), dpd.Tensor))
            else numpy.array(cm.materialize(self.V.get(lab)), dtype=numpy.complex128,
                             order='C', copy=True) for lab in hh_pp})
        o1, o2 = self.F, self.V
        # T2 in the DPD layout once for all products of all commutators
        t2c = cm.dpd_blocks(t2, self.dims) if self.dpd_storage else None
        self.comm_norms = []
        for n in range(1, self.max_comm + 1):
            ladder = self._ladder if (n == 1 and self._w is not None) else None
            c0, c1, c2 = cm.commutator(o1, o2, t1, t2, self.cum, self.dims, 1.0 / n,
                                       self._keep2, ladder=ladder, timings=self.timings,
                                       dpd_out=self.dpd_storage, T2c=t2c)
            cm.add_hc_1(c1)
            cm.add_hc_2(c2)
            h0 += 2.0 * numpy.real(c0)
            for lab, c in c1.items():
                h1[lab] += c
            for lab in hh_pp:
                c = c2.get(lab)
                if c is not None:
                    if isinstance(h2[lab], dpd.Tensor):
                        cm.add_block(h2[lab], c)
                    else:
                        cm.add_to_dense(h2[lab], c[0], c[1])
            data = lambda v: v.data if isinstance(v, dpd.Tensor) else v
            norm = numpy.sqrt(c1.norm2() + sum(
                (1 + (k[0] != k[1])) * (1 + (k[2] != k[3])) * numpy.vdot(data(v), data(v)).real
                for k, v in c2.items()))
            self.comm_norms.append(norm)
            if self.verbose > 1:
                gb = lambda b: sum((x.data if isinstance(x, dpd.Tensor) else x).nbytes
                                   for x in b.values()) / 1e9
                cache = getattr(self.V, 'cache', {})
                self.log('  commutator %2d: memory %7.0f MB; C2 %.1f GB, H2 %.1f GB, T2 %.1f GB, '
                         'integral cache %.1f GB; dense expansions of DPD tensors %d (%.1f GB)'
                         % (n, lib.current_memory()[0], gb(c2), gb(h2), gb(t2), gb(cache),
                            dpd.DENSIFIED[0], dpd.DENSIFIED[1] / 1e9))
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
        adiis = None
        if self.diis_space > 0:
            adiis = lib.diis.DIIS()
            # vectors and errors in a temporary HDF5 file (pyscf.lib.diis keeps those
            # below 80 MB in memory), read back one vector at a time
            adiis.space, adiis.min_space = self.diis_space, 2
            adiis.incore = False
            adiis.verbose = 0
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
            data = lambda v: v.data if isinstance(v, dpd.Tensor) else v
            d1 = numpy.sqrt(sum(numpy.linalg.norm(t1n[k] - t1[k])**2 for k in t1n))
            d2 = numpy.sqrt(sum(numpy.linalg.norm(data(t2n[k]) - data(t2[k]))**2
                                for k in t2n))
            self.log('%4d %20.12f %10.2e %9.2e %9.2e %8.5f %8.5f %5d %8.1f' % (
                it, e, e - e_old, d1, d2, numpy.sqrt(t1.norm2()), numpy.sqrt(t2.norm2()),
                len(self.comm_norms), time.time() - t0))
            if abs(e - e_old) < self.conv_tol and max(d1, d2) < self.conv_tol_amp:
                # Hbar belongs to (t1, t2)
                self.converged = True
                break
            e_old = e
            if adiis is not None:
                labs = [(self._t1_labels, t1, t1n), (self._t2_labels, t2, t2n)]
                # DPD blocks as their stored elements only
                vec = numpy.concatenate([data(n[k]).ravel() for ll, o, n in labs for k in ll])
                old = numpy.concatenate([data(o[k]).ravel() for ll, o, n in labs for k in ll])
                vec = adiis.update(vec, xerr=vec - old)
                del old
                pos = 0
                for ll, o, n in labs:
                    for k in ll:
                        size = data(n[k]).size
                        if isinstance(n[k], dpd.Tensor):
                            n[k].data[...] = vec[pos:pos + size]
                        else:
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
