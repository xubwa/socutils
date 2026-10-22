'''
Determinant strings of the active space and single creation / annihilation
tables between electron-number sectors.

Strings are uint64 bit strings sorted in ascending order, which is the
pyscf.fci.cistring order; phases follow conventions.md section 4.
'''

import functools
import itertools
import numpy

MAX_ORB = 63
_ONE = numpy.uint64(1)


def _popcount(x):
    return numpy.bitwise_count(x).astype(numpy.int64)


@functools.lru_cache(maxsize=None)
def gen_strings(norb, nelec):
    '''All strings with nelec of norb orbitals occupied, ascending (cached,
    read-only).'''
    if norb > MAX_ORB:
        raise NotImplementedError('more than %d active orbitals' % MAX_ORB)
    if nelec < 0 or nelec > norb:
        strs = numpy.zeros(0, dtype=numpy.uint64)
    else:
        strs = [sum(1 << p for p in c) for c in itertools.combinations(range(norb), nelec)]
        strs = numpy.asarray(sorted(strs), dtype=numpy.uint64)
    strs.flags.writeable = False
    return strs


def _address(sorted_strs, strs):
    if len(sorted_strs) == 0:
        return numpy.full(numpy.shape(strs), -1, dtype=numpy.int64)
    idx = numpy.minimum(numpy.searchsorted(sorted_strs, strs), len(sorted_strs) - 1)
    return numpy.where(sorted_strs[idx] == strs, idx, -1)


def single_op_table(norb, nelec, dagger):
    '''Map of a_p^+ (dagger=True) or a_p from the nelec sector.

    Returns:
        target: (norb, dim) address in the nelec +- 1 sector, -1 where the
            operator annihilates the string
        sign: (norb, dim) int64, +-1 or 0
    '''
    src = gen_strings(norb, nelec)
    dst = gen_strings(norb, nelec + (1 if dagger else -1))
    target = numpy.full((norb, len(src)), -1, dtype=numpy.int64)
    sign = numpy.zeros((norb, len(src)), dtype=numpy.int64)
    for p in range(norb):
        bit = _ONE << numpy.uint64(p)
        occupied = (src & bit) != 0
        ok = ~occupied if dagger else occupied
        new = src ^ bit
        par = _popcount(src >> numpy.uint64(p + 1)) & 1
        addr = _address(dst, new[ok]) if ok.any() else numpy.zeros(0, dtype=numpy.int64)
        target[p, ok] = addr
        sign[p, ok] = 1 - 2 * par[ok]
    return target, sign
