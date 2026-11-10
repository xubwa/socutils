#!/usr/bin/env python
'''
Where the time of one exact-K build of socutils.scf.jk_multi goes, by angular
class (l_i l_j | l_k l_l) and by integral engine (lanes / jk_ssss / generated
kernels / libcint), measured by the build itself (the semi-direct store's
measuring state, lib/vhf/jk_multi.c; the lanes' times include their digestion).

    python prof_k_classes.py mol.xyz def2-tzvp [--tol 1e-12] [--top 30]

The density is the minao guess (the classes and their weights are those of a
full SCF build).
'''
import argparse
import ctypes
import time
import numpy
from pyscf import gto, scf, lib
from socutils.scf import jk_multi
from socutils.hessian import ghf as ghf_hess

ENGINES = {0: 'libcint', 1: 'ssss', 2: 'gen', 3: 'lanes'}


def profile(mol, dm, tol):
    libv = ghf_hess._load_socvhf()
    st = jk_multi.Store(mol)
    t = time.perf_counter()
    jk_multi.get_jk(mol, dm, False, True, tol=tol, hermi=1, store=st)
    wall = time.perf_counter() - t
    n = libv.SOCVHFjk_store_ntype(st._ptr)
    ct = numpy.zeros(n ** 4)
    cc = numpy.zeros(n ** 4)
    stype = numpy.zeros(mol.nbas, dtype=numpy.int32)
    libv.SOCVHFjk_store_classes(st._ptr, ct.ctypes.data_as(ctypes.c_void_p), cc.ctypes.data_as(ctypes.c_void_p),
                                stype.ctypes.data_as(ctypes.c_void_p))
    rep = [int(numpy.nonzero(stype == k)[0][0]) for k in range(n)]
    ls = mol._bas[:, 1]
    agg = {}
    for idx in numpy.nonzero(cc)[0]:
        ti, tj, tk, tl = idx // n ** 3, idx // n ** 2 % n, idx // n % n, idx % n
        sh = numpy.asarray([rep[ti], rep[tj], rep[tk], rep[tl]], dtype=numpy.int32)
        e = libv.SOCVHFjk_engine(sh.ctypes.data_as(ctypes.c_void_p), mol._bas.ctypes.data_as(ctypes.c_void_p))
        l4 = [int(ls[s]) for s in sh]
        key = (tuple(sorted(l4[:2], reverse=True)) + tuple(sorted(l4[2:], reverse=True)), ENGINES[e])
        a = agg.setdefault(key, [0., 0.])
        a[0] += ct[idx] / 3e9       # TSC ticks, ~3 GHz
        a[1] += cc[idx]
    return wall, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz')
    ap.add_argument('basis')
    ap.add_argument('--tol', type=float, default=1e-12)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--charge', type=int, default=0)
    a = ap.parse_args()
    lines = open(a.xyz).read().splitlines()
    mol = gto.M(atom='\n'.join(lines[2:]), basis=a.basis, charge=a.charge, verbose=0)
    dm = scf.RHF(mol).get_init_guess(key='minao')
    wall, agg = profile(mol, dm, a.tol)
    tot = sum(v[0] for v in agg.values())
    print('%s / %s: nao %d, %d threads, tol %.0e: K build %.2f s wall, integral engines %.1f core-s' % (
        a.xyz, a.basis, mol.nao, lib.num_threads(), a.tol, wall, tot))
    byeng = {}
    for (l, e), v in agg.items():
        b = byeng.setdefault(e, [0., 0.])
        b[0] += v[0]
        b[1] += v[1]
    for e, v in sorted(byeng.items(), key=lambda kv: -kv[1][0]):
        print('  %-8s %8.1f core-s %5.1f%%  %12d quartets' % (e, v[0], 100 * v[0] / tot, v[1]))
    print('class        engine     core-s  share    quartets  ns/quartet')
    for (l, e), v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print('(%d%d|%d%d)     %-8s %8.2f %5.1f%% %11d %8.0f' % (l + (e, v[0], 100 * v[0] / tot, v[1],
                                                                    1e9 * v[0] / v[1])))


if __name__ == '__main__':
    main()
