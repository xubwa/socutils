'''The amplitude equations of the AO-driven spinor CCSD on packed amplitudes.

The equations are those of zccsd_direct.update_amps (PySCF's GCCSD with the
intermediates of Stanton, Gauss et al., bare integrals); only the way each
term is evaluated differs:

- T2, tau and the residual are packed, [(i>j), (a>b)] in
  numpy.tril_indices(n, -1) order; nothing of o^2 v^2 elements is made by
  fancy indexing: the layouts of the terms come from the OpenMP kernels of
  zccsdt.zccsd.kernels, and every matrix product goes through zccsdt.blas
  (the BLAS of the kernel library: with MKL on GNU OpenMP, the kernels and
  the products share one pool of threads and numpy's BLAS stays idle);
- the two o^3 v^3 terms, W(mbej) += <mn||ef> t2 and the ring t2 W, are
  ZGEMMs in the "ring layout" [(i a), (j b)], in which t2 and <mn||ef> are
  symmetric matrices.  The first pass runs over slabs of the rows (m e) and
  completes W(mbej) there; the integrals <mn||ef>, <mb||je> and <ma||ef>
  are each read once, by rows, in this pass.  The second pass runs over
  blocks of the columns (j b): the product with t2, antisymmetrized into the
  packed residual block by block.  The rows and columns of t2 are generated
  from the packed amplitudes; the two-index intermediates that contract
  <mn||ef> with t2 are partial traces of the first product.  Each complex
  product is three real GEMMs instead of four (A_r B_r, A_i B_i and
  (A_r + A_i) (B_r + B_i): the "3M" scheme) when the memory for the real
  parts is there;
- the five terms with <ma||ef> use the array packed over e > f (half of
  ovvv): one GEMM with tau for a slab, and the t1 terms of one occupied
  index at a time, unpacked in the cache;
- W(mnij) and the occupied-occupied Fock intermediate act on the packed
  pairs as one matrix; the terms that multiply t2 from the virtual side run
  over blocks of pairs;
- the pp ladder runs in the scalar AO basis over blocks of pairs i > j.
  The alpha-beta block of the back-transformed tau is split into its
  symmetric and antisymmetric parts in the two AO indices: the antisymmetric
  part joins alpha-alpha and beta-beta on the packed antisymmetrized
  integrals (mu si|nu rho) - (mu rho|nu si), the symmetric part takes the
  packed symmetrized integrals.  Two symmetric real operators replace the
  nao^2 x nao^2 matrix, and the real GEMMs do o^2 nao^4 multiply-adds.  The
  operators (zccsdt.zccsd.aoladder) are kept, as square matrices of 2 nao^4
  bytes each or in parts of their rows down to the triangles, or made from
  the AO integrals in every iteration when they do not fit (integral-direct:
  about nao^4 / 8 integrals for every block of pairs).

Kramers pairs.  For spinors in time-reversal pairs, ordered [unbarred,
barred] among the occupied and among the virtual ones (zccsdt.zccsd.kramers;
``eris.kramers``), every tensor obeys X[Kp, Kq, ...] = +-conj(X[p, q, ...]),
and so does every term of the equations.  The expensive terms are then
evaluated for half of their output and completed by this relation: the rows
(m e) of W(mbej) and of the ring product with unbarred m, the <ma||ef> terms
for unbarred m (the integrals with barred m are never read), the ladder for
one pair (i>j) of each two partners.  The cheap terms run as before.

Memory.  Nothing but three packed T2 (the amplitudes, tau, the residual) and
arrays smaller than those is assumed to fit: plan_step sizes the slabs and
blocks from what ``max_memory`` leaves, keeps W(mbej) (o^2 v^2 elements) in
memory only if the smallest blocks still fit next to it (else it is a mapped
array of a scratch file, written by rows and read by columns), and takes the
integrals that zccsdt.zccsd.ao2mo has put into scratch files by rows.
Smaller blocks mean that rows of t2 are generated more often; MemoryError
below the working set of one occupied index.  With an integral-direct ladder
a block of pairs is as large as the memory allows next to a unit of AO
integrals: the integrals are gone through once per block.

kernel() is the CCSD iteration on these amplitudes: the scheme of PySCF's
driver (Jacobi step, damping, DIIS on the vector [t1, packed t2], the same
convergence test and log lines) with T2 packed from the first guess to the
converged amplitudes.

Timers (zccsdt.timing): 'amps.imds', 'amps.ovvv', 'amps.ring', 'amps.t2',
'amps.ladder', 'amps.t1' with sub-timers, 'amps.kramers' (the two- and
three-index intermediates completed by time reversal); 'amps.diis',
'amps.energy'.
'''

from dataclasses import dataclass
from types import SimpleNamespace

import numpy
import numpy as np
from pyscf.lib import logger

from socutils.cc.fast import blas, memory, scratch
from pyscf.lib import diis as _pyscf_diis
from socutils.cc.fast.timing import timer
from socutils.cc.fast import kernels as K

PAIR_WORK = 2 << 30         # bytes of the arrays of a block of pairs, at most
# Three real GEMMs for a complex product from this dimension o v on: they save
# 2 (o v)^3 multiply-adds per product and cost passes over (o v)^2 elements;
# on 8 cores the two break even near o v = 2300 (H2O/cc-pVQZ: 2200)
THREE_M_MIN = 3000


def _blocks(n, size):
    size = max(1, min(n, int(size)))
    return [(p, min(p + size, n)) for p in range(0, n, size)]


def _size(nbytes):
    return '%.2f GB' % (nbytes / 1e9) if nbytes >= 1e9 else '%.1f MB' % (nbytes / 1e6)


def _gemm3(A3, B3, alpha=1.0, C=None, acc=False):
    '''C (+)= alpha A B for complex matrices given as their parts
    (K.split3): three real GEMMs.'''
    P = [blas.gemm(a, b) for a, b in zip(A3, B3)]
    return K.combine3(*P, alpha, C, acc)


@dataclass
class Plan:
    '''How one step is sliced (plan_step); blocks are lists of (start, stop).'''
    mblocks: list       # slabs of the first occupied index: rows of the ring layout, <ma||ef>
    jblocks: list       # blocks of the occupied index of the columns of the ring layout
    pblocks: list       # blocks of pairs of the terms on the virtual indices
    lblocks: list       # blocks of pairs of the ladder
    ladder_work: int    # bytes of a unit of AO integrals of the ladder (0: operators in memory)
    w_mapped: bool      # W(mbej) is a mapped array of a scratch file
    three_m: bool       # the o^3 v^3 products by three real GEMMs each
    diis_incore: bool   # the DIIS vectors are in memory
    work: int           # bytes of the largest working set of this plan
    need: int           # bytes of the smallest one
    budget: object      # bytes there were (None: unknown, everything in memory)


def step_arrays(nocc, nvir):
    '''Bytes of the arrays of a step that plan_step counts as always there
    next to its three packed T2: W(mnij) and its Fock part, Nq, the t1
    <mb||je> array, o^4.  (What the Lambda equations and the orbital gradient
    keep in their place is theirs to count.)'''
    o, v = int(nocc), int(nvir)
    npo = K.npair(o)
    return 16 * (3 * npo * npo + 2 * o * v * npo + 2 * o * o * o * v + 4 * o ** 4)


# Parts of a DIIS vector with fewer elements stay in memory when the plan puts
# the vectors into a scratch file (t1; of the packed T2 a megabyte at most).
# The default of zccsdt.cc.diis, 1e7 elements, kept the 7.5 million of
# benzene/cc-pVDZ with a frozen core in memory twelve times: 1.4 GB that the
# plan had counted as none.
DIIS_SMALL = 1 << 16


def diis_space(mycc, plan, size, log=None, what='T2'):
    '''The number of DIIS vectors: mycc.diis_space.'''
    return mycc.diis_space


class DIIS:
    '''pyscf.lib.diis.DIIS on the parts of a vector given as a list (e.g.
    [t1, packed t2]): they are joined into one vector and split again.
    ``incore``: keep the vectors in memory, else in pyscf's HDF5 file.'''

    def __init__(self, space=6, filename=None, incore=False, incore_size=None):
        self._diis = _pyscf_diis.DIIS()
        self._diis.space = space
        self._diis.incore = bool(incore)
        if filename:
            self._diis.filename = filename

    def update(self, parts):
        shapes = [numpy.shape(x) for x in parts]
        vec = numpy.hstack([numpy.ravel(x) for x in parts])
        vec = self._diis.update(vec)
        out, p0 = [], 0
        for shape in shapes:
            n = int(numpy.prod(shape))
            out.append(vec[p0:p0+n].reshape(shape))
            p0 += n
        return out

    def close(self):
        self._diis = None


def plan_step(nocc, nvir, nao, budget=None, mapped=(), diis_space=6, diis_incore=None,
              three_m=None, kramers=False, ladder_units=None):
    '''The Plan of one step within ``budget`` bytes (None: unlimited).
    ``mapped``: the integrals that are mapped arrays, of 'V' (<mn||ef>), 'E'
    (<mb||je>), 'ovvv_p'; their rows are read through the mapping and count
    as memory while they are used.  ``ladder_units``: None if the AO
    operators of the ladder are in memory, else (least, most) bytes of a unit
    of their AO integrals (aoladder: computed in every iteration, or read
    from a scratch file); a block of pairs of the ladder is then as large as
    the memory allows, since the units are gone through once per block.
    ``diis_incore``: keep the DIIS vectors in memory (None: if they take no
    more than a third of what is left).  ``three_m``: three real GEMMs for a
    complex product (None: if o v >= THREE_M_MIN and the blocks of columns
    that fit with their parts cover a quarter of the occupied space).
    ``kramers``: the step uses Kramers pairs (a block of columns then holds
    an occupied spinor and its partner: two at least).  MemoryError if the
    budget is below Plan.need.'''
    o, v, nao = int(nocc), int(nvir), int(nao)
    n1, npo, npv, n2 = o * v, K.npair(o), K.npair(v), 2 * nao
    budget = None if budget is None else int(budget)
    P = 16 * npo * npv                              # one packed T2
    col = 16 * n1 * v                               # the rows (m e) of one m, the columns (j b) of one j
    # t2, tau, the residual, and step_arrays
    fixed = 3 * P + step_arrays(o, v)
    row = col * (('V' in mapped) + ('E' in mapped)) + 16 * v * npv * ('ovvv_p' in mapped)
    pair = 32 * v * v + 16 * o * v                  # virtual-side terms of one pair
    lad = max(16 * n2 * (n2 + v), 24 * n2 * n2, 16 * v * (n2 + v))      # the ladder of one pair
    if kramers:         # its rows of tau and of the result are arrays of their own
        lad += 32 * npv
    ufloor, umost = (0, 0) if ladder_units is None else (int(x) for x in ladder_units)

    def rows_work(nm, nj, w_mapped, m3):
        '''Pass 1: the columns of t2, the mapped rows (with 3M: the parts of
        <mn||ef>) and the product block.'''
        block = nj * 16 * v * v * (1.5 * m3 + (nj < o))
        return int((1 + 0.5 * m3) * nj * col
                   + nm * (row + 1.5 * m3 * col + col * w_mapped + block))

    def cols_work(nj, w_mapped, m3):
        '''Pass 2: a block of columns of the product, the same number of
        rows of t2 and, through the mapping, the columns of W; with 3M the
        parts of W and the three real products.'''
        if not m3:
            return (2 + w_mapped) * nj * col
        return int((4 + w_mapped) * nj * col + 1.5 * nj * nj * 16 * v * v)

    jmin = 2 if kramers else 1                      # the smallest block of columns

    def blocks_floor(w_mapped, m3):
        return max(cols_work(jmin, w_mapped, m3), rows_work(1, jmin, w_mapped, m3), pair,
                   lad + ufloor)

    need = fixed + blocks_floor(True, False)
    if three_m is None and n1 < THREE_M_MIN:
        three_m = False
    if budget is None:
        m3 = three_m is not False
        return Plan([(0, o)], [(0, o)], _blocks(npo, PAIR_WORK // pair),
                    _blocks(npo, npo if umost else PAIR_WORK // lad), umost, False, m3,
                    True if diis_incore is None else diis_incore,
                    fixed + o * col + cols_work(o, False, m3)
                    + (2 * diis_space * P if diis_incore is not False else 0), need, None)
    if budget < need:
        raise MemoryError(
            'the CCSD equations need %s (three packed T2 and the blocks of one occupied '
            'orbital: %d occupied and %d virtual spinors), %s are left of max_memory'
            % (_size(need), o, v, _size(budget)))
    def blocks(m3):
        '''(w_mapped, diis_incore, bytes left for the blocks, nj, nm).'''
        rem = budget - fixed
        # W(mbej) in memory if the smallest blocks still fit next to it
        w_mapped = bool(o * col + blocks_floor(False, m3) > rem)
        if not w_mapped:
            rem -= o * col
        incore = bool(2 * diis_space * P <= rem // 3) if diis_incore is None else diis_incore
        if incore:
            rem = max(rem - 2 * diis_space * P, blocks_floor(True, m3))
        nj = o
        while nj > jmin and max(cols_work(nj, w_mapped, m3),
                                rows_work(1, nj, w_mapped, m3)) > rem:
            nj -= 1
        nm = o
        while nm > 1 and rows_work(nm, nj, w_mapped, m3) > rem:
            nm -= 1
        return w_mapped, incore, rem, nj, nm

    # three real products for a complex one if their blocks of columns still
    # cover a quarter of the occupied space: with smaller blocks the passes
    # over the parts cost more than the GEMMs save
    m3 = bool(three_m)
    if three_m is None and blocks_floor(True, True) <= budget - fixed:
        m3 = 4 * blocks(True)[3] >= o
    w_mapped, diis_incore, rem, nj, nm = blocks(m3)
    lrem = min(rem, PAIR_WORK)
    npb = max(1, min(npo, lrem // pair))
    if umost:           # units of AO integrals next to the largest block of pairs that fits
        uwork = min(umost, max(ufloor, min(rem // 4, rem - lad)))
        nlb = max(1, min(npo, (rem - uwork) // lad))
    else:
        uwork, nlb = 0, max(1, min(npo, lrem // lad))
    work = (budget - rem) + max(rows_work(nm, nj, w_mapped, m3), cols_work(nj, w_mapped, m3),
                                npb * pair, nlb * lad + uwork)
    return Plan(_blocks(o, nm), _blocks(o, nj), _blocks(npo, npb), _blocks(npo, nlb), uwork,
                w_mapped, m3, diis_incore, work, need, budget)


class Residual:
    '''The CCSD equations for the integrals ``eris`` (the container of
    DirectZCCSD with the stored <ma||ef>).

    update_packed(t1, t2p) is one Jacobi step on the packed amplitudes;
    update(t1, t2) does the same on t2[i,j,a,b], as zccsd_direct.update_amps.
    ``max_memory`` (MB for the process, as in PySCF; None: 80% of what the
    node and the job leave) and the DIIS settings go into the Plan (``plan``:
    one given explicitly).'''

    def __init__(self, eris, max_memory=None, diis_space=6, diis_incore=None, plan=None,
                 three_m=None):
        o = self.nocc = int(eris.nocc)
        fock = np.asarray(eris.fock)
        v = self.nvir = fock.shape[0] - o
        n1 = o * v
        self.foo = np.ascontiguousarray(fock[:o, :o])
        self.fov = np.ascontiguousarray(fock[:o, o:])
        self.fvv = np.ascontiguousarray(fock[o:, o:])
        self.mo_e = np.asarray(eris.mo_energy, dtype=np.float64)
        io, jo = K.tril(o)
        npo = K.npair(o)

        with timer('amps.setup'):
            # ring layouts: V[(m e), (n f)] = <mn||ef>, E[(m e), (j b)] = -<mb||je>,
            # and <mn||ef> on packed pairs; made by zccsdt.zccsd.ao2mo, else from
            # the blocks oovv and ovov
            V, E = getattr(eris, 'oovv_r', None), getattr(eris, 'ovov_r', None)
            oovv_p = getattr(eris, 'oovv_p', None)
            if V is None:
                V = np.ascontiguousarray(np.asarray(eris.oovv).transpose(0, 2, 1, 3))
            if E is None:
                E = -np.ascontiguousarray(np.asarray(eris.ovov).transpose(0, 3, 2, 1))
            if oovv_p is None:
                oovv_p = K.pack_t2(np.ascontiguousarray(np.asarray(eris.oovv)))
            self.V, self.E = V.reshape(n1, n1), E.reshape(n1, n1)
            self.oovv_p = np.ascontiguousarray(oovv_p)                  # in memory
            self.oooo_p = np.ascontiguousarray(np.asarray(eris.oooo)[io, jo][:, io, jo])
            ooov = self.ooov = np.ascontiguousarray(np.asarray(eris.ooov))  # <mn||ie>: [m,n,i,e]
            self.ooov_r = np.ascontiguousarray(ooov.transpose(0, 3, 2, 1))  # [m, e, j, n]
            # [i, e, (m>n)]
            self.ooov_en = np.ascontiguousarray(ooov[io, jo].transpose(1, 2, 0))
            # <ma||ef> packed over e > f: [m, a, (e>f)]
            ovvv_p = getattr(eris, 'ovvv_p', None)
            if ovvv_p is None:
                av, bv = K.tril(v)
                ovvv_p = np.ascontiguousarray(np.asarray(eris.ovvv)[:, :, av, bv])
            self.ovvv_p = ovvv_p

        # AO side of the ladder
        self.nao = eris.nao_sph
        self.C2 = np.ascontiguousarray(eris.mo_ao_spin[:, o:])          # (2 nao, nvir)
        self.C2c = self.C2.conj()
        self.aolad = eris.aoladder          # the AO operators of the ladder (zccsd.aoladder)

        # the pairs (m, k), k = 0 .. o-1, that contain the occupied m: packed
        # row and the sign of t2[m, k] against it (k = m: no row)
        k = np.arange(o)
        self.pair_rows = []
        for m in range(o):
            hi, lo = np.maximum(m, k), np.minimum(m, k)
            rows = hi * (hi - 1) // 2 + lo
            rows[m] = -1
            self.pair_rows.append((rows, np.where(k < m, 1.0, -1.0) * (k != m)))

        # -P(ij) sum_m F[m,j] t2[i,m] as a matrix G[(k>l), (i>j)] on the pairs,
        #   G = d(l,i) F[k,j] - d(l,j) F[k,i] - d(k,i) F[l,j] + d(k,j) F[l,i]:
        # (row, column, the two indices of F, sign) of its nonzeros
        x, kk, ll = k[None, :], io[:, None], jo[:, None]
        rows = np.broadcast_to(np.arange(npo)[:, None], (npo, o))

        def pair(a, b):
            hi, lo = np.maximum(a, b), np.minimum(a, b)
            return hi * (hi - 1) // 2 + lo
        self._g = []
        for mask, cols, f0, sign in ((x < ll, pair(ll, x), kk, 1.0), (x > ll, pair(x, ll), kk, -1.0),
                                     (x < kk, pair(kk, x), ll, -1.0), (x > kk, pair(x, kk), ll, 1.0)):
            self._g.append((rows[mask], cols[mask], np.broadcast_to(f0, (npo, o))[mask],
                            np.broadcast_to(x, (npo, o))[mask], sign))

        # Kramers pairs: partners and signs of the pairs, one pair of each two partners
        self.paired = bool(getattr(eris, 'kramers', False)) and o % 2 == 0 and v % 2 == 0
        self.kramers = self.paired          # use the symmetry of the pairs
        if self.paired:
            from socutils.cc.fast import kramers
            self.kpo, self.sgo = kramers.pair_maps(o)
            self.kpv, self.sgv = (np.ascontiguousarray(x) for x in kramers.pair_maps(v))
            self.ko, self.so = kramers.partner_signs(o)
            pairs = np.arange(npo)
            self.kpairs = pairs[self.kpo >= pairs]
        self._scratch = None        # the file of W(mbej) when it is mapped
        self._w = None
        self.plan = plan
        self.gave_way = False       # the kept AO operators were dropped for the step
        if plan is None:
            self.replan(max_memory, diis_space, diis_incore, three_m)
        elif plan is not False:     # False: the caller plans (replan) before the first step
            self._set_ladder()

    def replan(self, max_memory=None, diis_space=6, diis_incore=None, three_m=None, reserve=0):
        '''A new Plan for the memory there is now (see plan_step), less
        ``reserve`` bytes for arrays that the plan does not know (those of
        the Lambda equations, zccsd.lam).'''
        names = ('V', 'E', 'ovvv_p')
        mapped = [n for n in names if scratch.is_mapped(getattr(self, n))]
        self.close()
        lad = self.aolad
        self._keep_ladder(max_memory)       # the AO operators, if they were dropped (ccsd_t)
        self.kramers = self.paired

        def plan():
            budget = memory.budget(max_memory)
            if budget is not None:
                budget -= int(reserve)
            return plan_step(self.nocc, self.nvir, self.nao, budget, mapped,
                             diis_space, diis_incore, three_m, self.paired, lad.unit_range())

        try:
            self.plan = plan()
        except MemoryError as err:
            # the AO operators of the ladder that are kept in memory may be what
            # is missing: without them, their integrals in every iteration
            if lad.mode != 'memory':
                raise
            if not lad.auto:
                raise MemoryError(
                    '%s; the AO operators of the ladder are kept in memory (%s), ao_ladder = '
                    '\'direct\' computes them in every iteration' % (err, _size(lad.nbytes))
                ) from None
            lad.to_direct()
            self.plan = plan()
            self.gave_way = True
        self._set_ladder()
        return self.plan

    def _keep_ladder(self, max_memory):
        '''The AO operators of the ladder made again if they were dropped, in
        units of integrals sized for the memory there is now: when they were
        first made nothing else was there (half of the budget for a unit), and
        after (T) the amplitudes, the sources of lambda and whatever else has
        come since are.  Kept in memory they give way to the mode 'direct' if
        they do not fit any more (or MemoryError, if the mode was asked for).'''
        from socutils.cc.fast import aoladder
        lad = self.aolad
        if lad.mode == 'direct' or lad.same is not None:
            return
        budget = memory.budget(max_memory)
        if budget is not None:
            floor = aoladder.unit_floor(lad.mol)
            kept = aoladder.stored_bytes(lad.mol, lad.nblocks) if lad.mode == 'memory' else 0
            room = budget - kept
            if room < floor:
                if lad.mode != 'memory' or not lad.auto:
                    raise MemoryError(
                        'the AO operators of the ladder (%s kept%s) cannot be made again: '
                        '%s are left of max_memory, a unit of their integrals takes %s; '
                        'ao_ladder = \'direct\' computes them in every iteration'
                        % (_size(kept), ' in memory' if lad.mode == 'memory' else '',
                           _size(budget), _size(floor)))
                lad.to_direct()
                self.gave_way = True
                return
            lad.set_work(max(floor, room // 2))
        lad.keep()

    def _set_ladder(self):
        '''The units of AO integrals of the ladder as the plan sizes them.'''
        if self.plan.ladder_work and self.aolad.mode == 'direct':
            self.aolad.set_work(self.plan.ladder_work)

    def describe(self):
        '''One line on the memory plan of a step.'''
        p, o = self.plan, self.nocc
        nb = lambda b: max(b1 - b0 for b0, b1 in b)                     # noqa: E731
        text = ('packed T2 %s; working set %s%s; W(mbej) %s; passes over %d and %d of %d '
                'occupied spinors, %s; %d and %d pairs per block'
                % (_size(16 * K.npair(o) * K.npair(self.nvir)), _size(p.work),
                   '' if p.budget is None else ' of %s' % _size(p.budget),
                   'in a scratch file' if p.w_mapped else 'in memory',
                   nb(p.mblocks), nb(p.jblocks), o,
                   'three real GEMMs per product' if p.three_m else 'ZGEMM',
                   nb(p.pblocks), nb(p.lblocks)))
        if self.kramers:
            text += '; Kramers pairs: half of the rows, of <ma||ef> and of the pairs of the ladder'
        if p.ladder_work:
            text += '; the AO integrals of the ladder in units of %s' % _size(p.ladder_work)
            if self.gave_way:
                text += ' (the step did not fit next to the operators kept in memory)'
        return text + '; DIIS vectors in %s' % ('memory' if p.diis_incore else 'a scratch file')

    def close(self):
        '''Release W(mbej) and its scratch file.'''
        self._w = None
        if self._scratch is not None:
            self._scratch.close()
            self._scratch = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _w_array(self):
        '''W(mbej) in the ring layout: a new array, or the mapped one.'''
        n1 = self.nocc * self.nvir
        if not self.plan.w_mapped:
            return np.empty((n1, n1), dtype=np.complex128)
        if self._w is None:
            self._scratch = scratch.Scratch(
                prefix='zccsd_w_', what='W(mbej)',
                hint=' or give the calculation more memory (max_memory)')
            self._w = self._scratch.create('W', (n1, n1), np.complex128)
        return self._w

    # -- amplitudes and energy on packed pairs -------------------------------
    def init_amps(self):
        '''(E(MP2), t1, t2p): the guess of PySCF's GCCSD.'''
        o = self.nocc
        eo, ev = self.mo_e[:o], self.mo_e[o:]
        t1 = self.fov / (eo[:, None] - ev)
        t2p = K.div_packed(K.copy(self.oovv_p, conj=True), eo, ev)
        return float(K.dotu(self.oovv_p, t2p).real), t1, t2p

    def energy(self, t1, t2p):
        '''E_corr = f_ia t_ia + sum_{i>j, a>b} <ij||ab> tau_ij^ab.'''
        t1 = np.ascontiguousarray(t1, dtype=np.complex128)
        taup = K.outer_pp(t1, t1, 0.5, K.copy(np.ascontiguousarray(t2p, dtype=np.complex128)))
        return np.sum(self.fov * t1) + K.dotu(self.oovv_p, taup)

    # -- the pp ladder in the AO basis --------------------------------------
    def ladder(self, taup, R, left=False):
        '''R[(i>j), (a>b)] += sum_ef (ae|bf) tau_ij^ef = 1/2 sum_ef <ab||ef> tau_ij^ef.
        With Kramers pairs: for one pair (i>j) of each two partners, the
        other by time reversal.  ``left``: 1/2 sum_ef tau_ij^ef <ef||ab>, the
        ladder of the Lambda equations (the integral is the complex
        conjugate: the two sets of coefficients change places).'''
        v, nao = self.nvir, self.nao
        n2 = 2 * nao
        C2, C2c = (self.C2c, self.C2) if left else (self.C2, self.C2c)
        for p0, p1 in self.plan.lblocks:
            if self.kramers:                        # the pairs of the block: rows of tau
                idx = self.kpairs[p0:p1]
                if not idx.size:
                    break
                n, rows = idx.size, np.ascontiguousarray(taup[idx])
                out = np.zeros((n, taup.shape[1]), dtype=np.complex128)
            else:
                n, rows, out = p1 - p0, taup[p0:p1], R[p0:p1]
            with timer('ladder.ao'):
                # X[x, p, y] = sum_ef C2[x, e] tau[p, e, f] C2[y, f], x and y over (spin, AO)
                tf = K.unpack_tril(rows, v, 'apb')                      # [e, p, f]
                H = blas.gemm(C2, tf.reshape(v, n * v))
                del tf
                X = blas.gemm(H.reshape(n2 * n, v), C2.T)               # [x, p, y]
                del H
            with timer('ladder.pack'):
                AR, SR = K.ladder_pack(X, n, nao)
                del X
            # the real AO operators on real and imaginary parts at once
            GA, GS = self.aolad.apply(AR, SR)
            del AR, SR
            with timer('ladder.pack'):
                G = K.ladder_unpack(GA, GS, n, nao)                     # [x, p, y]
                del GA, GS
            with timer('ladder.ao'):
                # L[p] = C2^+ G[p] C2^*
                H = blas.gemm(C2c.T, G.reshape(n2, n * n2))
                del G
                L = blas.gemm(H.reshape(v * n, n2), C2c)                # [a, p, b]
                del H
            with timer('ladder.pack'):
                K.pack_tril_add(L.reshape(v, n, v), 1.0, out, 'apb')
                del L
                if self.kramers:
                    K.add_rows(R, idx, np.ones(n), out)
                    K.kfill_packed(out, np.where(self.kpo[idx] != idx, self.kpo[idx], -1),
                                   self.sgo[idx], self.kpv, self.sgv, R)
            del out, rows
        self.aolad.release()
        return R

    # -- the two passes over the ring layout -----------------------------------
    def _pass_rows(self, s, W, ring=0.5):
        '''Pass 1, over slabs of the first occupied index m: the rows (m e)
        of W(mbej) - the bare integral, its t1 terms, the <ma||ef> t1 term
        and the product <mn||ef> t2 - and everything else that reads
        <mn||ef>, <mb||je> and <ma||ef>.

        ``ring``: the factor of the product <mn||ef> t2 in W, 1/2 in the
        amplitude equations; with 1 W is the element <mb|Hbar|ej> of the
        similarity-transformed Hamiltonian, which the Lambda equations need
        (zccsdt.zccsd.lam).  They also ask (s.fold_kw) for KW[m, i, (j b)] =
        sum_e t1[i,e] (<mb||ej> + ring sum_nf <mn||ef> t2[n,j,f,b]), added
        to s.Nq[m, b, (i>j)] antisymmetrized in i and j, slab by slab, and
        give no residual (s.R is None).'''
        o, v = self.nocc, self.nvir
        n1, npo, npv = o * v, K.npair(o), K.npair(v)
        plan, V, E = self.plan, self.V, self.E
        t1, t1T, t1c, t2p, taup, R = s.t1, s.t1T, s.t1.conj(), s.t2p, s.taup, s.R
        fold_kw = getattr(s, 'fold_kw', False)
        io, jo = K.tril(o)
        trace = 0.5 / ring          # the partial traces are those of 1/2 V T
        single = len(plan.jblocks) == 1
        s.g = np.empty((o, v), dtype=np.complex128)         # sum_nf <mn||ef> t1[n,f]
        # Kt[m, i, (j b)] = sum_e t1[i,e] E[(m e), (j b)]
        s.Kt = np.empty((o, o, n1), dtype=np.complex128)
        t1row = t1.reshape(1, n1)
        # Nq[m, a, (i>j)] = <ij||ma>* + sum_{e>f} <ma||ef> tau[(ij), (ef)]
        s.Nq = np.conj(self.ooov_en)
        s.X1 = np.zeros((o, v), dtype=np.complex128)
        s.r1 = np.zeros((o, v), dtype=np.complex128)
        s.foo_t = np.zeros((o, o), dtype=np.complex128)     # partial traces of 1/2 V T
        s.fvv_t = np.zeros((v, v), dtype=np.complex128)
        s.fvv_a = np.zeros((v, v), dtype=np.complex128)     # [a, e] = sum_mf <ma||ef> t1[m,f]
        fv = np.empty((v, v), dtype=np.complex128)
        x1 = np.empty((o, v), dtype=np.complex128)
        T = None                # the columns of t2, or their three real parts
        m3 = plan.three_m
        mblocks = plan.mblocks
        if self.kramers:        # the unbarred m only; the rest by time reversal
            ho = o // 2
            mblocks = _blocks(ho, max(m1 - m0 for m0, m1 in mblocks))
        for m0, m1 in mblocks:
            nm = m1 - m0
            nr, rows = nm * v, slice(m0 * v, m1 * v)
            Vb, Eb, Wb = V[rows], E[rows], W[rows]
            with timer('amps.ring'), timer('ring.small'):
                # -<mb||je> - sum_n t1[n,b] (<mn||je> - sum_f <mn||ef> t1[j,f])
                s.g[m0:m1] = blas.gemm(Vb, t1row.T).reshape(nm, v)
                Z = blas.gemm(Vb.reshape(nr * o, v), t1T).reshape(nr, o, o)     # [(m e), n, j]
                N = self.ooov_r[m0:m1].reshape(nr, o, o) - Z.transpose(0, 2, 1)
                del Z
                K.copy(Eb, Wb)
                blas.gemm(N.reshape(nr * o, o), t1, C=Wb.reshape(nr * o, v), beta=1.0)
                del N
                # for - t1[m,a] t1[i,e] <mb||ej> and the T1 equation
                for m in range(m0, m1):
                    blas.gemm(t1, Eb[(m - m0) * v:(m - m0 + 1) * v], C=s.Kt[m])
                blas.gemm(t1row[:, rows], Eb, C=s.r1.reshape(1, n1), beta=1.0)

            with timer('amps.ovvv'):
                # the five terms with <ma||ef>
                O = self.ovvv_p[m0:m1]                                  # [m, a, (e>f)]
                with timer('ovvv.gemm'):
                    blas.gemm(O.reshape(nm * v, npv), taup.T,
                              C=s.Nq[m0:m1].reshape(nm * v, npo), beta=1.0)
                with timer('ovvv.t1'):
                    for m in range(m0, m1):
                        Om = O[m - m0]
                        prow, sgn = self.pair_rows[m]
                        if R is not None:
                            # sum_e t1[i,e] <ab||em> as conj(sum_e t1*[i,e] <me||ab>)
                            Z = blas.gemm(t1c, Om)
                            K.add_rows(R, prow, sgn, Z, conj=True)
                            if self.kramers:        # the same term of the partner of m
                                brow, bsgn = self.pair_rows[m + ho]
                                K.add_rows_k(R, brow, -bsgn * self.so, Z, self.ko, self.kpv,
                                             self.sgv)
                            del Z
                        # sum_f <mb||ef> t1[j,f] into W(mbej), Fvv and the T1 equation
                        K.ovvv_t1(Om, t1, m, Wb[(m - m0) * v:(m - m0 + 1) * v], fv, x1)
                        s.fvv_a += fv
                        # sum_{e>f} <ma||ef> t2[m,i,(ef)]: the tau term less its t1 t1 part
                        c = np.maximum(prow, 0)
                        s.X1 += ((s.Nq[m][:, c] - np.conj(self.ooov_en[m][:, c])) * sgn).T - x1
                scratch.unmap(O)
                del O

            with timer('amps.ring'):
                # + 1/2 sum_nf <mn||ef> t2[n,j,f,b]: the o^3 v^3 term of W; its
                # partial traces are the <mn||ef> t2 terms of Foo and Fvv
                if fold_kw:             # t1 W before the product, for its part of KW
                    KW = np.empty((nm, o, n1), dtype=np.complex128)
                    for m in range(m0, m1):
                        blas.gemm(t1, Wb[(m - m0) * v:(m - m0 + 1) * v], C=KW[m - m0])
                if m3:
                    with timer('ring.layout'):
                        Vb = K.split3(Vb)
                for j0, j1 in plan.jblocks:
                    nj = j1 - j0
                    with timer('ring.layout'):
                        if T is None or not single:                     # [(n f), (j b)]
                            T = (K.ring_cols3 if m3 else K.ring_cols)(t2p, o, v, j0, j1)
                    W4 = Wb.reshape(nm, v, o, v)[:, :, j0:j1]           # [m, e, j, b]
                    both = range(max(m0, j0), min(m1, j1))              # m as a row and a column
                    if single:
                        with timer('ring.layout'):
                            tro = np.einsum('meje->mj', W4)
                            trv = sum(W4[m - m0, :, m - j0] for m in both)
                        with timer('ring.gemm'):
                            if m3:
                                _gemm3(Vb, T, ring, Wb, acc=True)
                            else:
                                blas.gemm(Vb, T, C=Wb, alpha=ring, beta=1.0)
                        with timer('ring.layout'):
                            s.foo_t[m0:m1, j0:j1] += trace * (np.einsum('meje->mj', W4) - tro)
                            s.fvv_t += trace * (sum(W4[m - m0, :, m - j0] for m in both) - trv)
                    else:
                        with timer('ring.gemm'):
                            X4 = _gemm3(Vb, T, ring) if m3 else blas.gemm(Vb, T, alpha=ring)
                            X4 = X4.reshape(nm, v, nj, v)
                        with timer('ring.layout'):
                            s.foo_t[m0:m1, j0:j1] += trace * np.einsum('meje->mj', X4)
                            for m in both:
                                s.fvv_t += trace * X4[m - m0, :, m - j0]
                            W4 += X4
                            del X4
                if fold_kw:             # sum_e t1[i,e] (<mb||ej> + the product), into Nq
                    for m in range(m0, m1):
                        x = (s.Kt[m] + blas.gemm(t1, Wb[(m - m0) * v:(m - m0 + 1) * v])
                             - KW[m - m0]).reshape(o, o, v)
                        s.Nq[m] += (x[io, jo] - x[jo, io]).T
                    del KW, x
            if self.kramers:                # the rows of the partners of these m
                with timer('amps.ring'), timer('ring.layout'):
                    K.kfill_ring(W, o, v, m0, m1, ho)
                brows = W[(m0 + ho) * v:(m1 + ho) * v]
                scratch.written(brows)
                scratch.unmap(brows)
            scratch.written(Wb)
            for a in (V[rows], Eb, Wb):
                scratch.unmap(a)
            del Vb
        s.T = T if single else None                         # all of t2: kept for pass 2
        if self.kramers:
            # what was summed over the unbarred m, or made for them, completed
            from socutils.cc.fast.kramers import fill_barred, image
            with timer('amps.kramers'):
                fill_barred(s.g)
                fill_barred(s.Kt.reshape(o, o, o, v))
                fill_barred(s.Nq, pairs=(2,))
                fill_barred(s.foo_t)
                for x in (s.X1, s.r1, s.fvv_t, s.fvv_a):
                    x += image(x)

    def _pass_columns(self, s):
        '''Pass 2, over blocks of the columns (j b): the ring term
        sum_me t2[i,m,a,e] W[(m e), (j b)] and the t1 t1 <mb||ej> term,
        antisymmetrized into the packed residual.  With Kramers pairs a
        block is a range of unbarred j and their partners, and the rows
        (i a) with barred i come from those with unbarred i.'''
        o, v = self.nocc, self.nvir
        n1, plan = o * v, self.plan
        m3, jblocks, iblocks = plan.three_m, plan.jblocks, plan.jblocks
        ho = o // 2
        if self.kramers:
            size = max(j1 - j0 for j0, j1 in jblocks)
            jblocks = _blocks(ho, max(1, size // 2))
            iblocks = _blocks(ho, size)
        nrow = ho * v if self.kramers else n1       # the rows that are multiplied
        for j0, j1 in jblocks:
            nj = j1 - j0
            if self.kramers:                        # the columns of j0:j1 and of their partners
                ncol, whole = 2 * nj * v, nj == ho
                parts = (slice(j0 * v, j1 * v), slice((j0 + ho) * v, (j1 + ho) * v))
            else:
                ncol, whole = nj * v, nj == o
                parts = (slice(j0 * v, j1 * v),)
            # the columns of W: all of it, one block, or two blocks (j and their
            # partners), which are multiplied one after the other into the two
            # halves of the columns of r4 - no copy of them is made, except for
            # the real parts of the three real GEMMs
            cols = [slice(None)] if whole else list(parts)
            WJ = W3 = None
            if m3:
                with timer('ring.layout'):
                    if len(cols) == 1:
                        WJ = s.W[:, cols[0]]
                    else:
                        WJ = np.empty((n1, ncol), dtype=np.complex128)
                        WJ[:, :nj * v], WJ[:, nj * v:] = s.W[:, cols[0]], s.W[:, cols[1]]
                    W3 = K.split3(WJ)
                scratch.unmap(s.W)
                if whole and not plan.w_mapped:             # the complex W is not needed again
                    s.W = None
                del WJ
            r4 = np.empty((n1, ncol), dtype=np.complex128)              # [(i a), (j b)]
            width = ncol // len(cols)

            def times(T, rows):
                '''r4[rows] = T W[:, the columns of the block].'''
                if m3:
                    _gemm3(T, W3, C=r4[rows])
                else:
                    for k, c in enumerate(cols):
                        blas.gemm(T, s.W[:, c], C=r4[rows, k * width:(k + 1) * width])

            with timer('ring.gemm'):
                if s.T is not None:
                    times([x[:nrow] for x in s.T] if m3 else s.T[:nrow], slice(0, nrow))
                    s.T = None
                else:
                    for i0, i1 in iblocks:                              # rows of t2: it is symmetric
                        with timer('ring.layout'):
                            Ti = (K.ring_cols3 if m3 else K.ring_cols)(s.t2p, o, v, i0, i1)
                        times([x.T for x in Ti] if m3 else Ti.T, slice(i0 * v, i1 * v))
                        del Ti
            W3 = None
            with timer('ring.small'):
                # - t1[m,a] t1[i,e] <mb||ej> (E is <mb||ej> in the ring layout)
                r43 = r4.reshape(o, v, ncol)
                for i in range(nrow // v):
                    for k, c in enumerate(cols):
                        blas.gemm(s.t1T, s.Kt[:, i, c], C=r43[i][:, k * width:(k + 1) * width],
                                  alpha=-1.0, beta=1.0)
            with timer('ring.pack'):
                if self.kramers:
                    K.kfill_ring(r4, o, v, 0, ho, nj)
                    K.asym_pack_cols(r4, o, v, j0, j1, 1.0, s.R)
                    K.asym_pack_cols(r4, o, v, j0 + ho, j1 + ho, 1.0, s.R, offset=nj * v)
                else:
                    K.asym_pack_cols(r4, o, v, j0, j1, 1.0, s.R)
            del r4
            if s.W is not None:
                scratch.unmap(s.W)
        s.W = None

    def symmetric(self, t1, t2p, tol=1e-6):
        '''Do the amplitudes obey the time-reversal symmetry of the Kramers
        pairs (kramers.symmetric: all of t1 and a few rows of t2p)?'''
        from socutils.cc.fast.kramers import symmetric
        return symmetric(t1, t2p, tol)

    # -- one step -------------------------------------------------------------
    def _imds_pairs(self, s):
        '''s.taup, and (foo_x, Wp): sum_ne <mn||ie> t1[n,e] and W(mnij) on
        packed pairs [(m>n), (i>j)] with the tau <mn||ef> term of the ladder,
        i.e. <mn|Hbar|ij>.'''
        o, v = self.nocc, self.nvir
        io, jo = K.tril(o)
        taup = s.taup = K.outer_pp(s.t1, s.t1, 0.5, K.copy(s.t2p))
        # tmp[m, n, i, j] = sum_e <mn||ie> t1[j, e]
        tmp = blas.gemm(self.ooov.reshape(o ** 3, v), s.t1T).reshape(o, o, o, o)
        foo_x = np.einsum('mnin->mi', tmp)
        tp = tmp[io, jo]
        Wp = self.oooo_p + tp[:, io, jo] - tp[:, jo, io]
        blas.gemm(self.oovv_p, taup.T, C=Wp, beta=1.0)
        return foo_x, Wp

    def _fock(self, s, foo_x):
        '''(Fov, Ft, tF, Foo, Fvv) after pass 1: the one-particle
        intermediates with their diagonals, Ft = Fov t1^T and tF = t1^T Fov.
        Foo + Ft / 2 and Fvv - tF / 2 are the occupied and the virtual block
        of the similarity-transformed Hamiltonian.'''
        Fov = self.fov + s.g
        Ft, tF = blas.gemm(Fov, s.t1.T), blas.gemm(s.t1T, Fov)          # o x o, v x v
        Foo = self.foo + 0.5 * Ft + foo_x + s.foo_t
        Fvv = self.fvv - 0.5 * tF - s.fvv_t.T - s.fvv_a
        return Fov, Ft, tF, Foo, Fvv

    def update_packed(self, t1, t2p, level_shift=0.0):
        '''(t1new, t2pnew) from t1[i,a] and t2p[(i>j),(a>b)].  With Kramers
        pairs the amplitudes must obey their symmetry (symmetric()): those
        that do not are passed through the equations without it.'''
        o, v = self.nocc, self.nvir
        npo = K.npair(o)
        io, jo = K.tril(o)
        fov = self.fov
        eo, ev = self.mo_e[:o], self.mo_e[o:] + level_shift
        t1 = np.ascontiguousarray(t1, dtype=np.complex128)
        t2p = np.ascontiguousarray(t2p, dtype=np.complex128)
        t1T = np.ascontiguousarray(t1.T)
        s = SimpleNamespace(t1=t1, t1T=t1T, t2p=t2p)
        if self.kramers and not self.symmetric(t1, t2p):
            self.kramers = False
            try:
                return self.update_packed(t1, t2p, level_shift)
            finally:
                self.kramers = True

        with timer('amps.imds'):
            foo_x, Wp = self._imds_pairs(s)
            taup = s.taup
            R = s.R = K.copy(self.oovv_p, conj=True)

        s.W = self._w_array()
        self._pass_rows(s, s.W)
        with timer('amps.ring'):
            self._pass_columns(s)
        r1, Nq = s.r1, s.Nq
        del s.Kt

        with timer('amps.t2'):
            Fov, Ft, tF, Foo, Fvv = self._fock(s, foo_x)
            Foo[np.diag_indices(o)] -= eo
            Fvv[np.diag_indices(v)] -= ev
            F2o = Foo + 0.5 * Ft
            F2v = Fvv - 0.5 * tF
            # occupied side: W(mnij) and -P(ij) F2o[m,j] t2[i,m] as one matrix on
            # the pairs, applied to tau; the t1 t1 part of tau is taken back
            for rows, cols, f0, f1, sign in self._g:
                Wp[rows, cols] += sign * F2o[f0, f1]
            blas.gemm(Wp.T, taup, C=R, beta=1.0)
            del Wp
            K.outer_pp(t1, blas.gemm(F2o.T, t1), 1.0, R)
            # virtual side, in blocks of pairs
            F2vT = np.ascontiguousarray(F2v.T)
            for p0, p1 in self.plan.pblocks:
                n = p1 - p0
                t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)   # [(a p), e]
                Y = blas.gemm(t2u, F2vT)                                # sum_e t2[p,a,e] F2v[b,e]
                Nb = Nq[:, :, p0:p1].reshape(o, v * n)
                blas.gemm(Nb.T, t1, C=Y, beta=1.0)                      # + sum_m Nq[m,a,p] t1[m,b]
                K.asym_pack2(Y.reshape(v, n, v), 1.0, R[p0:p1], 'apb')
                # - 1/2 sum_mne <mn||ie> t2[m,n,a,e] of the T1 equation
                r1 += blas.gemm(self.ooov_en[:, :, p0:p1].reshape(o, v * n), t2u)
                del t2u, Y, Nb
            del Nq, s.Nq

        with timer('amps.ladder'):
            self.ladder(taup, R)

        with timer('amps.t1'):
            r1 += blas.gemm(t1, Fvv.T) - blas.gemm(Foo.T, t1) + s.X1 + fov.conj()
            K.t2p_matvec(t2p, Fov, r1)
            K.div_packed(R, eo, ev)
            r1 /= eo[:, None] - ev
        return r1, R

    def update(self, t1, t2, level_shift=0.0):
        '''(t1new, t2new) on t2[i,j,a,b]: update_packed between a pack and an
        unpack.'''
        o, v = self.nocc, self.nvir
        with timer('amps.unpack'):
            t2p = K.pack_t2(np.ascontiguousarray(t2, dtype=np.complex128))
        t1new, t2pnew = self.update_packed(t1, t2p, level_shift)
        with timer('amps.unpack'):
            t2new = K.unpack_t2(t2pnew, o, v)
        return t1new, t2new


def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6, verbose=None,
           callback=None):
    '''The CCSD iterations of pyscf.cc.ccsd.kernel on packed amplitudes:
    (converged, E_corr, t1, t2) with t2[i,j,a,b] unpacked at the end.

    The DIIS vector is [t1, t2 packed over i > j and a > b] as in PySCF's
    GCCSD, so the iterates are the same; it is kept in memory if the memory
    plan of the step says so or mycc.incore_complete is set, else in a
    mapped scratch file (mycc.diis_file; zccsdt.cc.diis).
    ``callback(envs)`` is called after every step, as in PySCF, with a dict
    of the names mycc, eris, res, istep, t1, t2p, t1new, t2pnew (the new
    amplitudes, packed), eccsd, eold, adiis, log.'''
    log = logger.new_logger(mycc, verbose)
    name = mycc.__class__.__name__
    settings = (mycc.max_memory, mycc.diis_space, True if mycc.incore_complete else None,
                getattr(mycc, 'three_m', None))
    if getattr(eris, 'residual', None) is None:
        eris.residual = Residual(eris, *settings[:3], three_m=settings[3])
    else:                               # the memory may have changed since the last run
        eris.residual.replan(*settings)
    res = eris.residual
    o, v = res.nocc, res.nvir
    if t1 is None and t2 is None:
        mycc.emp2, t1, t2p = res.init_amps()
        log.info('Init t2, MP2 energy = %.15g', mycc.emp2)
    else:
        guess = res.init_amps() if t1 is None or t2 is None else None
        t1 = guess[1] if t1 is None else np.ascontiguousarray(t1, dtype=np.complex128)
        t2p = (guess[2] if t2 is None
               else K.pack_t2(np.ascontiguousarray(t2, dtype=np.complex128)))

    if res.kramers and not res.symmetric(t1, t2p):
        log.warn('The amplitudes given do not have the time-reversal symmetry of the '
                 'Kramers pairs: the equations are solved without it')
        res.kramers = False
    log.info('CCSD equations: %s', res.describe())
    kind, text = blas.backend()
    log.info('Matrix products: %s', text)
    if kind != 'mkl':
        log.warn('The matrix products of the CCSD run on %s, not on MKL in the OpenMP pool of '
                 'the kernels: the results are the same, large calculations are slower '
                 '(benzene/cc-pVDZ with OpenBLAS: 1.4 times per iteration).  For MKL build csrc '
                 'with -DZC_MKL_DIR=<MKL lib directory>.', text)
    if len(blas.openmp_runtimes()) > 1:
        log.warn('More than one OpenMP runtime is loaded (%s): the kernels and the BLAS may '
                 'run on separate pools of threads that compete for the cores.  With MKL, '
                 'MKL_THREADING_LAYER=GNU in the environment puts them on one.',
                 ', '.join(blas.openmp_runtimes()))

    def energy(t1, t2p):
        with timer('amps.energy'):
            e = res.energy(t1, t2p)
        if abs(e.imag) > 1e-4:
            log.warn('Non-zero imaginary part found in %s energy %s', name, e)
        return e.real

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = energy(t1, t2p)
    log.info('Init E_corr(%s) = %.15g', name, eccsd)
    adiis = None
    if mycc.diis:
        adiis = DIIS(space=diis_space(mycc, res.plan, t2p.size, log), filename=mycc.diis_file,
                     incore=res.plan.diis_incore, incore_size=DIIS_SMALL)
    conv = False
    try:
        for istep in range(max_cycle):
            t1new, t2pnew = res.update_packed(t1, t2p, mycc.level_shift)
            if callback is not None:
                # not locals(): the frame would keep its values - the old and
                # the new packed amplitudes - alive through the next step
                callback(dict(mycc=mycc, eris=eris, res=res, istep=istep, t1=t1, t2p=t2p,
                              t1new=t1new, t2pnew=t2pnew, eccsd=eccsd, eold=eold,
                              adiis=adiis, log=log))
            with timer('amps.diis'):
                normt = np.sqrt(K.diff_norm2(t1new, t1) + K.diff_norm2(t2pnew, t2p))
                if mycc.iterative_damping < 1.0:
                    alpha = mycc.iterative_damping
                    t1new = (1 - alpha) * t1 + alpha * t1new
                    t2pnew *= alpha
                    t2pnew += (1 - alpha) * t2p
                t1, t2p = t1new, t2pnew
                t1new = t2pnew = None
                if (adiis is not None and istep >= mycc.diis_start_cycle
                        and abs(eccsd - eold) < mycc.diis_start_energy_diff):
                    x1, x2 = adiis.update([t1, t2p])
                    t1, t2p = x1.reshape(o, v), x2.reshape(t2p.shape)
                    log.debug1('DIIS for step %d', istep)
            eold, eccsd = eccsd, energy(t1, t2p)
            log.info('cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                     istep + 1, name, eccsd, eccsd - eold, normt)
            cput1 = log.timer('%s iter' % name, *cput1)
            if abs(eccsd - eold) < tol and normt < tolnormt:
                conv = True
                break
    finally:
        if adiis is not None:
            adiis.close()
    log.timer(name, *cput0)
    return conv, eccsd, t1, K.unpack_t2(np.ascontiguousarray(t2p), o, v)

