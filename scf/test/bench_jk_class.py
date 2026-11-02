#!/usr/bin/env python
'''
Cost of one angular class of two-electron integrals as the J / K builds of
socutils.scf.jk_multi meet it (lib/vhf/jk_bench.c): the unique shell quartets
(i >= j, k >= l, ij >= kl) passing Schwarz screening whose class is
(l_i l_j | l_k l_l), evaluated by libcint and by our kernels, nothing else.

    python bench_jk_class.py mol.xyz cc-pvdz --cls 0000          # one class
    python bench_jk_class.py mol.xyz cc-pvdz --cls all           # every class present
    python bench_jk_class.py mol.xyz cc-pvdz --keep 0 --cls 0000 # s shells of the basis only

Prints per class: quartets, primitive quartets, wall time per engine, ns per
quartet and per primitive quartet (x threads: per-core cost), and whether
the checksums (sum of all integrals) of the engines agree.
'''
import argparse
import ctypes
import itertools
import numpy
from pyscf import gto, lib
from pyscf.gto import moleintor
from socutils.scf import jk_multi
from socutils.hessian import ghf as ghf_hess

ENGINES = {0: 'libcint', 1: 'jk_ssss'}


def engines_for(cls):
    '''jk_ssss.c: s / p quartets with at most two p shells'''
    return [0, 1] if max(cls) <= 1 and sum(cls) <= 2 else [0]


def run(mol, cls, engine, tol):
    libv = ghf_hess._load_socvhf()
    q = jk_multi._q_cond(mol)
    atm, bas, env = mol._atm, mol._bas, mol._env
    opt = moleintor.make_cintopt(atm, bas, env, 'int2e_sph')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    c = numpy.asarray(cls, dtype=numpy.int32)
    stats = numpy.zeros(5)
    libv.SOCVHFjk_bench_class.restype = ctypes.c_double
    s = libv.SOCVHFjk_bench_class(
        ctypes.c_int(engine), c.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(tol), opt, ao_loc.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
        stats.ctypes.data_as(ctypes.c_void_p))
    return s, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xyz')
    ap.add_argument('basis')
    ap.add_argument('--cls', default='all', help="e.g. 0000, 1000 (l_i l_j l_k l_l), comma-separated, or 'all'")
    ap.add_argument('--keep', default=None, help='angular momenta of the basis to keep, e.g. 0 or 01')
    ap.add_argument('--tol', type=float, default=1e-12)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    atom = open(a.xyz).read()
    if a.xyz.endswith('.xyz'):
        atom = '\n'.join(atom.strip().splitlines()[2:])
    basis = a.basis
    if a.keep is not None:
        keep = {int(x) for x in a.keep}
        mol0 = gto.M(atom=atom, basis=a.basis, verbose=0)
        basis = {e: [b for b in gto.load(a.basis, e) if b[0] in keep] for e in mol0.elements}
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    ls = sorted(set(mol._bas[:, gto.ANG_OF]))
    print('%s / %s%s: nao %d, nbas %d, l %s, threads %d, tol %.0e' % (
        a.xyz, a.basis, '' if a.keep is None else ' (l in %s)' % a.keep, mol.nao, mol.nbas, ls,
        lib.num_threads(), a.tol))
    if a.cls == 'all':
        classes = [c for c in itertools.product(ls, repeat=4)]
    else:
        classes = [tuple(int(x) for x in c) for c in a.cls.split(',')]
    print('%-9s %11s %13s %8s %9s %10s %11s  %s' % (
        'class', 'quartets', 'prim-quart', 'engine', 'wall s', 'ns/quart', 'ns/prim', 'checksum'))
    nt = lib.num_threads()
    for cls in classes:
        ref = None
        for eng in engines_for(cls):
            run(mol, cls, eng, a.tol)                                  # warm-up
            best = None
            for r in range(a.reps):
                s, st = run(mol, cls, eng, a.tol)
                best = st if best is None or st[3] < best[3] else best
            if best[0] == 0:
                break
            ok = '' if ref is None else ('agrees' if abs(s - ref) <= 1e-10 * max(1, abs(ref)) else
                                         'DIFFERS %.3e' % (s - ref))
            ref = s if ref is None else ref
            print('(%d%d|%d%d)   %11d %13d %8s %9.4f %10.1f %11.2f  %s' % (
                *cls, best[0], best[1], ENGINES[eng], best[3], best[3] * nt * 1e9 / best[0],
                best[3] * nt * 1e9 / max(1, best[1]), ok), flush=True)


if __name__ == '__main__':
    main()
