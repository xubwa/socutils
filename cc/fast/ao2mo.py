'''Spinor MO integrals of a CCSD from the scalar AO integrals, slab by slab,
within a memory budget.

For two-component spinors and the bare Coulomb operator (X2C-1e, X2CAMF,
GHF) the integrals are, with the alpha and beta components C^s of the MOs
in the scalar AO basis,

    (pq|rs) = sum_{s,s'} sum C^s*[mu,p] C^s[nu,q] (mu nu|la si) C^s'*[la,r] C^s'[si,s]:

the AO integrals are real, have 8-fold symmetry and carry no spin.

The ket pass is that of nrr_fast of socutils (zccsdt.zccsd.nrr_fast), in C;
the bra pass is here (_Bra): PySCF's s4 AO fill, and the two quarter
transformations as real matrix products over whole blocks of AO pairs.
(nrr_fast's own bra pass calls a kernel for every AO pair and spin
component, each with three n x n arrays of its own to allocate and fill and
with a product on a matrix of zeros - the real integrals taken as complex
ones -, and its result has to be transposed for the ket pass: 2 to 3 times
the time of this one, bench/profile_ao2mo.py.)  This module organizes both
for a CCSD:

- only the integrals that are needed are made: (ip|qr) with i occupied for
  the blocks of a CCSD with an AO-driven ladder (occupied_blocks), (ae|bf)
  for the packed <ab||ef> of a ladder in the MO basis (packed_vvvv);
- every slab is antisymmetrized into its blocks at once, so that no file of
  half-sorted integrals is written and read back.

Nothing is assumed to fit into memory but the working set of one orbital
(``max_memory`` in MB for the whole process as in PySCF; None: 80% of what
the node and the memory limit of the job leave):

- the passes run over batches of the first MO index and, within a bra pass,
  over ranges of the AO shell pairs (la si), sized by plan_slabs: the AO
  integrals of a range, the half-transformed integrals of a batch and one
  slab of MO integrals are what is in memory.  Smaller batches mean that
  the AO integrals are computed more often, not that anything is written;
- the integrals that are produced stay in memory only while they take no
  more than a third of the budget together, the smallest blocks first; the
  others are mapped arrays of a scratch file (zccsdt.scratch), filled slab
  by slab;
- a budget below the working set of one orbital and one shell pair raises
  MemoryError with what is needed.

ladder_operators builds the two AO operators of the AO-driven ladder in shell
batches, under the same rules.  Timers (zccsdt.timing): 'ao2mo.bra',
'ao2mo.ket', 'ao2mo.sort', 'ao2mo.ladder'.
'''

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo, outcore

from socutils.cc.fast import blas, memory, scratch
from socutils.cc.fast.timing import timer
from socutils.cc.fast import kernels as K
from socutils.cc.fast import nrr_fast

SLAB_BYTES = 4 << 30        # the largest slab of MO integrals
BRA_BYTES = 1 << 30         # the arrays of a block of AO pairs in the bra pass, at most;
                            # and the AO integrals of a range, packed
WORK_BYTES = 16 << 30       # what the passes use at most, however much there is
STORE_SHARE = 1 / 3         # share of the budget for the integrals kept in memory
BLOCKS = ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv')


def spin_components(mol, mo_coeff, motype='j-spinor'):
    '''Alpha and beta components (nao, nmo) of spinor MOs in the scalar AO
    basis: ``mo_coeff`` in the j-adapted spinor basis of PySCF ('j-spinor')
    or already as the alpha block stacked on the beta block ('ghf').'''
    nao = mol.nao_nr()
    if motype == 'j-spinor':
        ca, cb = mol.sph2spinor_coeff()
        return np.dot(ca, mo_coeff), np.dot(cb, mo_coeff)
    assert motype == 'ghf' and mo_coeff.shape[0] == 2 * nao
    return np.asarray(mo_coeff[:nao]), np.asarray(mo_coeff[nao:])


def _threads():
    return max(1, lib.num_threads())


def _size(nbytes):
    return '%.2f GB' % (nbytes / 1e9) if nbytes >= 1e9 else '%.1f MB' % (nbytes / 1e6)


def _store_bytes(budget, need):
    '''Bytes of the budget for the integrals that stay in memory: a share of
    it, and never what the smallest passes need (None: no limit known).'''
    if budget is None:
        return None
    return int(max(0, min(STORE_SHARE * budget, budget - need)))


@dataclass
class Plan:
    '''Sizes of the passes of one transformation (plan_slabs).'''
    nbra: int           # first-index orbitals per bra pass
    nket: int           # first-index orbitals per slab of the ket pass
    nkl: int            # AO pairs (la si) per range of a bra pass, at most
    work: int           # bytes of the passes with these sizes
    need: int           # bytes of the smallest passes: one orbital, one shell pair
    budget: object      # bytes that were given (None: no limit known)


def plan_slabs(mol, nfirst, nq, budget=None, sort_arrays=1):
    '''Plan for the integrals (iq|rs) of ``nfirst`` first orbitals and
    ``nq`` orbitals in each of the other three places, within ``budget``
    bytes (None: WORK_BYTES).  ``sort_arrays``: arrays of one orbital (nq^3)
    that each thread needs to sort a slab into its place.'''
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    ao_loc = mol.ao_loc_nr()
    pmax = int((ao_loc[1:] - ao_loc[:-1]).max()) ** 2        # AO pairs of a shell pair, at most
    nthr = _threads()

    def work(nbra, nket, nkl):
        return (16 * npair * nq * nbra                       # half transform of a batch
                + 8 * npair * nkl                            # AO integrals of a range, packed
                + _bra_pair(nao, nq, nbra) * _bra_block(nao, nq, nbra, nkl)    # a block of them at work
                + nao * (80 * nq + 48 * nbra)                # MO coefficients and their parts
                + 16 * nq ** 3 * (nket + sort_arrays * min(nthr, nket)))    # slab and its sorting

    need = work(1, 1, pmax)
    if budget is not None and need > budget:
        raise MemoryError('the integral transformation needs %s for one orbital at a time '
                          '(%d AO functions, %d MOs per index), %s are left of max_memory'
                          % (_size(need), nao, nq, _size(budget)))
    w = WORK_BYTES if budget is None else min(int(budget), WORK_BYTES)
    nket = int(max(1, min(nfirst, w // 3 // (16 * nq ** 3 * (1 + sort_arrays)),
                          max(1, SLAB_BYTES // (16 * nq ** 3)))))
    # (half for the half transform: every batch computes the AO integrals)
    nbra = int(max(1, min(nfirst, w // 2 // (16 * npair * nq))))
    nket = min(nket, nbra)
    while work(nbra, nket, pmax) > w and nbra > 1:
        nbra = max(1, nbra // 2)
        nket = min(nket, nbra)
    # the AO integrals of a range take BRA_BYTES at most: a longer range
    # saves nothing (4.9 s for the pairs of AuH in ranges of 826, 5.1 s in
    # ranges of 2975), and what the passes do not need stays free
    top = int(max(pmax, min(npair, BRA_BYTES // (8 * npair))))
    nkl = top
    if work(nbra, nket, nkl) > w:
        rest = w - work(nbra, nket, 0)      # (that has one AO pair at work)
        pair = _bra_pair(nao, nq, nbra)
        nkl = rest // (8 * npair + pair)    # a range as one block
        if nkl * pair > BRA_BYTES:          # ... or as several
            nkl = (rest - BRA_BYTES) // (8 * npair)
        nkl = int(max(pmax, min(top, nkl)))
    return Plan(nbra, nket, nkl, work(nbra, nket, nkl), need, budget)


def _bra_pair(nao, nq, nbra):
    '''Bytes that one AO pair of a block of the bra pass takes: the integrals
    unpacked to a square, their quarter transform for the batch of first
    orbitals (three real parts for each of the two spin components), and the
    three real products of one first orbital.'''
    return 8 * nao * nao + 48 * nao * nbra + 24 * nq


def _bra_block(nao, nq, nbra, nkl):
    '''AO pairs per block of the bra pass, of the ``nkl`` of a range.'''
    return int(max(1, min(nkl, BRA_BYTES // _bra_pair(nao, nq, nbra))))


class _Bra:
    '''The bra pass for batches of first orbitals i and the orbitals q:

        vin[(i q), k] = sum_s sum_{mu nu} conj(C^s[mu,i]) C^s[nu,q] (mu nu|k)

    for all AO pairs k = (la >= si), in the shell-block order of PySCF's s4
    fill, which nrr_fast._e2 takes.  The AO integrals are computed for one
    range of shell pairs at a time (``ranges``: of
    pyscf.ao2mo.outcore.guess_shell_ranges) and are real and symmetric in
    (mu nu); a block of them is unpacked to squares, and then

    - quarter 1: one real GEMM for the whole block, G[(i s part), (k mu)] =
      sum_nu B[(i s part), nu] (mu nu|k) with the real part, the imaginary
      part and their sum of conj(C^s[:, i]) for both spin components;
    - quarter 2: for each i, the complex product of C^s[:, q]^T with
      G[i s]^T as three real GEMMs for each spin component, combined into
      the rows (i q) of vin: nothing is transposed afterwards.

    6 n (n + nq) real multiply-adds for an AO pair and a first orbital (n
    AO functions).  The arrays of a block of AO pairs take BRA_BYTES at
    most (or those of one AO pair), and they are allocated once for all
    batches.'''

    def __init__(self, mol, Cqa, Cqb, ranges, nbra):
        self.mol, self.ranges = mol, ranges
        nao, nq = Cqa.shape
        self.nao, self.nq = nao, nq
        self.npair = nao * (nao + 1) // 2
        # (the data of libcint for the shells, made once for all ranges; no screening)
        self.opt = _ao2mo.AO2MOpt(mol, 'int2e_sph')
        nkl = max(r[2] for r in ranges)
        self.block = _bra_block(nao, nq, nbra, nkl)
        self.eri = np.empty((nkl, self.npair))
        self.sq = np.empty((self.block, nao, nao))
        self.G = np.empty((6 * nbra, self.block * nao))
        self.P = np.empty((3, nq, self.block))
        # C^s[:, q]^T: real part, imaginary part and their sum
        self.A = [[np.ascontiguousarray(x) for x in (C.real.T, C.imag.T, C.real.T + C.imag.T)]
                  for C in (Cqa, Cqb)]

    def __call__(self, Cia, Cib):
        mol, nao, nq, P = self.mol, self.nao, self.nq, self.P
        bi = Cia.shape[1]
        # conj(C^s[:, i]): the same three, [(i s part), nu]
        B = np.empty((bi, 2, 3, nao))
        for s, C in enumerate((Cia, Cib)):
            B[:, s, 0], B[:, s, 1] = C.real.T, -C.imag.T
            B[:, s, 2] = B[:, s, 0] + B[:, s, 1]
        B = B.reshape(6 * bi, nao)
        vin = np.empty((bi * nq, self.npair), dtype=np.complex128)
        k0 = 0
        for r in self.ranges:
            with timer('ao2mo.fill'):
                eri = _ao2mo.nr_e1fill('int2e_sph', r[:3], mol._atm, mol._bas, mol._env, 's4', 1,
                                       self.opt, self.eri)[0]
            for c0 in range(0, r[2], self.block):
                c1 = min(r[2], c0 + self.block)
                nb = c1 - c0
                sq = lib.unpack_tril(eri[c0:c1], lib.SYMMETRIC, out=self.sq)
                blas.gemm(B, sq.reshape(nb * nao, nao).T, C=self.G[:6 * bi, :nb * nao])
                for i in range(bi):
                    for part in range(3):
                        for s in (0, 1):
                            Gi = self.G[(2 * i + s) * 3 + part, :nb * nao].reshape(nb, nao)
                            blas.gemm(self.A[s][part], Gi.T, C=P[part][:, :nb], beta=float(s))
                    K.combine3(P[0][:, :nb], P[1][:, :nb], P[2][:, :nb], 1.0,
                               vin[i * nq:(i + 1) * nq, k0 + c0:k0 + c1])
            k0 += r[2]
        return vin


def slabs(mol, Ca, Cb, first, rest=None, plan=None):
    '''Generator of (i0, i1, g): g[i - i0, q, r, s] = (iq|rs) in chemists'
    notation for the first index i in first = (start, stop) and the other
    three in rest = (start, stop) (default: all MOs).  ``g`` is a buffer that
    the next slab overwrites.'''
    nao, nmo = Ca.shape
    q0, q1 = (0, nmo) if rest is None else rest
    nq = q1 - q0
    npair = nao * (nao + 1) // 2
    if plan is None:
        plan = plan_slabs(mol, first[1] - first[0], nq)
    Cqa = np.asarray(Ca[:, q0:q1], dtype=np.complex128, order='F')
    Cqb = np.asarray(Cb[:, q0:q1], dtype=np.complex128, order='F')
    ranges = outcore.guess_shell_ranges(mol, True, plan.nkl)
    bra = _Bra(mol, Cqa, Cqb, ranges, min(plan.nbra, first[1] - first[0]))
    g = np.empty((plan.nket * nq, nq * nq), dtype=np.complex128)
    for b0 in range(first[0], first[1], plan.nbra):
        b1 = min(b0 + plan.nbra, first[1])
        with timer('ao2mo.bra'):
            vin = bra(Ca[:, b0:b1], Cb[:, b0:b1])
        for i0 in range(b0, b1, plan.nket):
            i1 = min(i0 + plan.nket, b1)
            n = (i1 - i0) * nq
            with timer('ao2mo.ket'):
                nrr_fast._e2(mol, vin[(i0 - b0) * nq:(i1 - b0) * nq], Cqa, Cqb,
                             (0, nq, 0, nq), out=g[:n])
            yield i0, i1, g[:n].reshape(i1 - i0, nq, nq, nq)
        del vin


class Arrays(dict):
    '''The arrays of a transformation by name.  ``scratch`` keeps the file of
    the mapped ones open (None: all are in memory); ``plan`` is the Plan of
    the passes.'''
    scratch = None
    plan = None

    def on_disk(self):
        return [k for k, a in self.items() if scratch.is_mapped(a)]

    def describe(self):
        '''One line on where the arrays are and how the passes were sliced.'''
        gb = lambda keys: ', '.join('%s %s' % (k, _size(self[k].nbytes)) for k in keys)  # noqa: E731
        disk = self.on_disk()
        text = 'in memory: ' + (gb([k for k in self if k not in disk]) or 'nothing')
        if disk:
            text += '; in a scratch file: ' + gb(disk)
        p = self.plan
        if p is not None:
            text += ('; passes of %d and %d orbitals, %d AO pairs per range (%s)'
                     % (p.nbra, p.nket, p.nkl, _size(p.work)))
        return text


def _allocate(shapes, dtype, mem_bytes, what):
    '''Arrays for ``shapes`` (a dict): the smallest in memory as long as they
    take no more than ``mem_bytes`` together (None: all of them), the others
    as mapped arrays of a scratch file.'''
    size = {k: np.dtype(dtype).itemsize * int(np.prod(s, dtype=np.int64))
            for k, s in shapes.items()}
    out, used = Arrays(), 0
    for k in sorted(shapes, key=size.get):
        if mem_bytes is None or used + size[k] <= mem_bytes:
            out[k] = np.empty(shapes[k], dtype=dtype)
            used += size[k]
        else:
            if out.scratch is None:
                out.scratch = scratch.Scratch(prefix='zccsd_', what=what,
                                              hint=' or give the calculation more memory '
                                                   '(max_memory)')
            out[k] = out.scratch.create(k, shapes[k], dtype)
    return out, used


def occupied_blocks(mol, Ca, Cb, nocc, with_ovvv=True, max_memory=None, plan=None,
                    names=None):
    '''The antisymmetrized blocks <pq||rs> with a first occupied index: an
    Arrays dict with oooo, ooov, oovv, ovov, ovvo and (``with_ovvv``) ovvv,
    or with the blocks in ``names``, which may also ask for the layouts of
    zccsdt.zccsd.amps (pairs in numpy.tril_indices(n, -1) order):

    'ovvv_p'  <ia||bc> packed over b > c, [i, a, (b>c)]: half of ovvv;
    'oovv_p'  <ij||ab> packed over i > j and a > b, [(i>j), (a>b)];
    'oovv_r'  <ij||ab> as [i, a, j, b];
    'ovov_r'  -<ib||ja> as [i, a, j, b].

    ``max_memory`` (MB) decides which of them are mapped scratch arrays and
    how the passes are sliced (``plan``: an explicit Plan for the latter).'''
    nmo = Ca.shape[1]
    nvir = nmo - nocc
    dim = {'o': nocc, 'v': nvir}
    if names is None:
        names = BLOCKS[:5 + bool(with_ovvv)]
    bv, cv = np.tril_indices(nvir, -1)
    special = {'ovvv_p': (nocc, nvir, bv.size),
               'oovv_p': (nocc * (nocc - 1) // 2, bv.size),
               'oovv_r': (nocc, nvir, nocc, nvir), 'ovov_r': (nocc, nvir, nocc, nvir)}
    shapes = {k: special[k] if k in special else [dim[c] for c in k] for k in names}
    budget = memory.budget(max_memory)
    need = plan_slabs(mol, nocc, nmo).need if plan is None else plan.need
    out, used = _allocate(shapes, np.complex128, _store_bytes(budget, need), 'the MO integrals')
    if plan is None:
        plan = plan_slabs(mol, nocc, nmo, None if budget is None else budget - used)
    out.plan = plan
    o, v = slice(0, nocc), slice(nocc, nmo)
    # <iq||rs> = (ir|qs) - (is|qr) from gi[q, r, s] = (iq|rs): the two parts of each block
    parts = {'oooo': ((o, o, o), (o, o, o)), 'ooov': ((o, o, v), (v, o, o)),
             'oovv': ((v, o, v), (v, o, v)), 'ovov': ((o, v, v), (v, v, o)),
             'ovvo': ((v, v, o), (o, v, v))}

    def sort(i, gi):                                    # one orbital per thread
        for k in names:
            if k in parts:
                x, y = parts[k]
                np.subtract(gi[x].transpose(1, 0, 2), gi[y].transpose(1, 2, 0), out=out[k][i])
        if 'ovvv' in names or 'ovvv_p' in names:       # <ia||bc> = (ib|ac) - (ic|ab)
            t = np.ascontiguousarray(gi[v, v, v].transpose(1, 0, 2))    # [a, b, c] = (ib|ac)
            if 'ovvv' in names:
                np.subtract(t, t.transpose(0, 2, 1), out=out['ovvv'][i])
            if 'ovvv_p' in names:
                np.subtract(t[:, bv, cv], t[:, cv, bv], out=out['ovvv_p'][i])
        if 'oovv_r' in names:                           # <ij||ab> = (ia|jb) - (ib|ja) as [a, j, b]
            t = gi[v, o, v]
            np.subtract(t, t.transpose(2, 1, 0), out=out['oovv_r'][i])
        if 'ovov_r' in names:                           # -<ib||ja> = (ia|bj) - (ij|ba) as [a, j, b]
            np.subtract(gi[v, v, o].transpose(0, 2, 1), gi[o, v, v].transpose(2, 0, 1),
                        out=out['ovov_r'][i])
        if 'oovv_p' in names and i:                     # the pairs (i, j < i)
            t = gi[v, :i, v].transpose(1, 0, 2)         # [j, a, b] = (ia|jb)
            np.subtract(t[:, bv, cv], t[:, cv, bv],
                        out=out['oovv_p'][i * (i - 1) // 2:i * (i + 1) // 2])

    with ThreadPoolExecutor(min(_threads(), plan.nket)) as pool:
        for i0, i1, g in slabs(mol, Ca, Cb, (0, nocc), None, plan):
            with timer('ao2mo.sort'):
                list(pool.map(lambda i: sort(i, g[i - i0]), range(i0, i1)))
    return out


def packed_vvvv(mol, Ca, Cb, nocc, max_memory=None, plan=None):
    '''<ab||ef> packed over a > b and e > f, rows and columns in the order of
    numpy.tril_indices(nvir, -1): an Arrays dict with 'vvvv_tril' (in memory
    or a mapped scratch array, by ``max_memory`` in MB) and the pair indices
    as its attribute ``idx`` = (ti, tj).'''
    nmo = Ca.shape[1]
    nvir = nmo - nocc
    ti, tj = np.tril_indices(nvir, -1)
    npair = ti.size
    budget = memory.budget(max_memory)
    need = plan_slabs(mol, nvir, nvir, sort_arrays=0).need if plan is None else plan.need
    out, used = _allocate({'vvvv_tril': (npair, npair)}, np.complex128,
                          _store_bytes(budget, need), 'the packed <ab||cd> integrals')
    if plan is None:
        plan = plan_slabs(mol, nvir, nvir, None if budget is None else budget - used,
                          sort_arrays=0)
    out.plan, out.idx = plan, (ti, tj)
    vv = out['vvvv_tril']

    def sort_a(a, ga):
        # ga[e, b, f] = (ae|bf); <ab||ef> = (ae|bf) - (af|be) for b < a: one
        # (e, f) matrix at a time, in the cache
        for b in range(a):
            P = np.ascontiguousarray(ga[:, b, :])
            P -= P.T.copy()
            vv[a * (a - 1) // 2 + b] = P[ti, tj]

    with ThreadPoolExecutor(min(_threads(), plan.nket)) as pool:
        for a0, a1, g in slabs(mol, Ca, Cb, (nocc, nmo), (nocc, nmo), plan):
            with timer('ao2mo.sort'):
                list(pool.map(lambda a: sort_a(a - nocc, g[a - a0]), range(a0, a1)))
    return out


def ladder_operators(mol, max_memory=None, names=('ao_eri', 'W_same')):
    '''AO operators of the AO-driven pp ladder (DirectZCCSD), an Arrays dict
    with those in ``names``:

    'W_same'  the antisymmetrized integral (mu si|nu rho) - (mu rho|nu si)
              packed over mu > nu and si > rho (pair indices: the attribute
              ``idx`` = (pi, pj)), for antisymmetric AO amplitudes;
    'W_plus'  (mu nu|la si) + (mu si|la nu), rows packed over nu >= si (half
              of it on nu = si) and columns over mu >= la, for symmetric ones;
    'ao_eri'  (mu nu|la si) as the full matrix [(nu si), (mu la)].

    Built from the AO integrals of batches of shells in the first place; in
    memory or mapped by ``max_memory`` (MB).'''
    nao, nbas, ao_loc = mol.nao_nr(), mol.nbas, mol.ao_loc_nr()
    pi, pj = np.tril_indices(nao, -1)
    si, sj = np.tril_indices(nao)
    budget = memory.budget(max_memory)
    # a batch of AO functions: its integrals and the two gathers of a W
    per_ao = 8 * nao ** 3 + 16 * nao * si.size
    need = 2 * int((ao_loc[1:] - ao_loc[:-1]).max()) * per_ao           # one shell
    if budget is not None and need > budget:
        raise MemoryError('the AO operators of the ladder need %s for one shell at a time '
                          '(%d AO functions), %s are left of max_memory'
                          % (_size(need), nao, _size(budget)))
    shapes = {'ao_eri': (nao, nao, nao, nao), 'W_same': (pi.size, pi.size),
              'W_plus': (si.size, si.size)}
    out, used = _allocate({k: shapes[k] for k in names}, np.float64,
                          _store_bytes(budget, need), 'the AO operators of the ladder')
    out.idx = (pi, pj)
    ao4, W, Wp = out.get('ao_eri'), out.get('W_same'), out.get('W_plus')
    w = WORK_BYTES if budget is None else min(budget - used, WORK_BYTES)
    nthr = _threads()
    with timer('ao2mo.ladder'), ThreadPoolExecutor(nthr) as pool:
        sh0 = 0
        while sh0 < nbas:
            sh1 = sh0 + 1
            while sh1 < nbas and int(ao_loc[sh1 + 1] - ao_loc[sh0]) * per_ao <= w // 2:
                sh1 += 1
            m0, m1 = int(ao_loc[sh0]), int(ao_loc[sh1])
            E = mol.intor('int2e', shls_slice=(sh0, sh1, 0, nbas, 0, nbas, 0, nbas))
            E = E.reshape(m1 - m0, nao, nao, nao)             # [nu, mu, si, la] = (nu mu|si la)
            if ao4 is not None:
                ao4[m0:m1] = E.transpose(0, 2, 1, 3)          # [nu, si, mu, la]
            # rows of W_same with mu in the batch: (mu si|nu rho) - (mu rho|nu si)
            r0, r1 = m0 * (m0 - 1) // 2, m1 * (m1 - 1) // 2
            edges = [r0 + (r1 - r0) * j // nthr for j in range(nthr + 1)]
            # rows of W_plus with nu in the batch: (nu mu|si la) + (nu la|si mu)
            q0, q1 = m0 * (m0 + 1) // 2, m1 * (m1 + 1) // 2
            qedges = [q0 + (q1 - q0) * j // nthr for j in range(nthr + 1)]

            def rows(j):
                a, b = edges[j], edges[j + 1]
                if W is not None and b > a:
                    mu, nu = pi[a:b, None] - m0, pj[a:b, None]
                    W[a:b] = E[mu, pi[None, :], nu, pj[None, :]]
                    W[a:b] -= E[mu, pj[None, :], nu, pi[None, :]]
                a, b = qedges[j], qedges[j + 1]
                if Wp is not None and b > a:
                    nu, sg = si[a:b, None] - m0, sj[a:b, None]
                    Wp[a:b] = E[nu, si[None, :], sg, sj[None, :]]
                    Wp[a:b] += E[nu, sj[None, :], sg, si[None, :]]
                    Wp[a:b][si[a:b] == sj[a:b]] *= 0.5
            list(pool.map(rows, range(nthr)))
            sh0 = sh1
    if ao4 is not None:
        out['ao_eri'] = ao4.reshape(nao * nao, nao * nao)
    return out
