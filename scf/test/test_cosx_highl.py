#!/usr/bin/env python
'''COSX at high angular momentum (h .. k): the potential integrals A_ab(g) and their centre derivatives against
libcint (int1e_grids, int1e_grids_ip) per shell-pair class, the K kernel's A (one- and two-centre, near and far
field, by parity class) through a one-point grid, and the K gradient / Fock1 against finite differences.'''
import ctypes
import unittest
import numpy
from pyscf import gto
from socutils.scf import cosx


def _shells(rng, lmax, nctr=1, nprim=2):
    out = []
    for l in range(lmax + 1):
        es = sorted(rng.uniform(.4, 6., nprim), reverse=True)
        out.append([l] + [[e] + list(rng.uniform(.3, 1., nctr)) for e in es])
    return out


class _OnePoint:
    def __init__(self, p, nbas):
        self.coords = numpy.asarray(p, dtype=float).reshape(1, 3)
        self.weights = numpy.ones(1)
        self.non0tab = numpy.ones((1, nbas), dtype=numpy.uint8)


@unittest.skipIf(cosx._lib() is None or cosx._lib().SOCDFTcosx_lmax() < 7, 'COSX kernel of l >= 7 not built')
class KnownValues(unittest.TestCase):
    def test_pair_ints_vs_libcint(self):
        rng = numpy.random.default_rng(5)
        mol = gto.M(atom='Xe 0 0 0; Kr 0.4 0.9 -1.1', basis={'Xe': _shells(rng, 7), 'Kr': _shells(rng, 7)},
                    verbose=0, spin=None)
        pts = numpy.array([c + r * v / numpy.linalg.norm(v) for c in (numpy.zeros(3), mol.atom_coord(1))
                           for r, v in zip((.05, .5, 1.8, 5.), rng.normal(size=(4, 3)))])
        ref = mol.intor('int1e_grids_sph', grids=pts)
        ip = mol.intor('int1e_grids_ip_sph', grids=pts)
        with __import__('pyscf').lib.temporary_env(cosx, PRIM_TOL=0., PAIR_TOL=0.):
            k = cosx.COSX(mol, grids=_OnePoint(pts[0], mol.nbas), fit=False).build()
        lib = k._libdft
        atm, bas, env, ao_loc = k._env
        self.assertTrue(lib.SOCDFTcosx_build_grad(ctypes.c_void_p(k._opt), atm.ctypes.data_as(ctypes.c_void_p),
                                                  bas.ctypes.data_as(ctypes.c_void_p),
                                                  env.ctypes.data_as(ctypes.c_void_p)))
        xyz = numpy.ascontiguousarray(pts.T)
        ng = len(pts)
        for I in range(mol.nbas):
            for J in range(I + 1):
                di, dj = ao_loc[I + 1] - ao_loc[I], ao_loc[J + 1] - ao_loc[J]
                A = numpy.zeros((di * dj, ng))
                dA = numpy.zeros((3, di * dj, ng))
                self.assertEqual(lib.SOCDFTcosx_pair_ints(
                    ctypes.c_void_p(k._opt), ctypes.c_int(I), ctypes.c_int(J), xyz.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ng), A.ctypes.data_as(ctypes.c_void_p), dA.ctypes.data_as(ctypes.c_void_p)), 0)
                A = A.reshape(di, dj, ng).transpose(2, 0, 1)
                dA = dA.reshape(3, di, dj, ng).transpose(0, 3, 1, 2)
                si, sj = slice(ao_loc[I], ao_loc[I + 1]), slice(ao_loc[J], ao_loc[J + 1])
                a = ref[:, si, sj]
                if mol.bas_atom(I) == mol.bas_atom(J):
                    d = -(ip[:, :, si, sj] + ip[:, :, sj, si].transpose(0, 1, 3, 2))
                else:
                    d = -ip[:, :, si, sj]
                # libcint's own error at (7, 7) two-centre is ~1e-10 (a 40-digit reference: kernel 4e-14)
                self.assertLess(abs(A - a).max() / abs(a).max(), 1e-9, (I, J))
                self.assertLess(abs(dA - d).max() / abs(d).max(), 1e-9, (I, J))

    def test_k_kernel_one_point(self):
        # K_d[k, nu] = phi_k(g)^2 A_{nu d}(g) for D_d = e_k e_d^T (k a diffuse probe): the K kernel's A, by parity
        # class for the one-centre pairs, near and far field, on one point at a time
        rng = numpy.random.default_rng(7)
        mol = gto.M(atom='Xe 0 0 0; Kr 0.4 0.9 -1.1; Ne 0 0 0',
                    basis={'Xe': _shells(rng, 5, 2, 3), 'Kr': _shells(rng, 4, 1, 3), 'Ne': [[0, [1e-5, 1.]]]},
                    verbose=0, spin=None)
        nao = mol.nao
        kp = nao - 1
        dms = numpy.zeros((nao, nao, nao))
        for d in range(nao):
            dms[d, kp, d] = 1
        pts = [r * v / numpy.linalg.norm(v) for r, v in zip((0., .1, .6, 1.5, 4., 9.), rng.normal(size=(6, 3)))]
        pts += [mol.atom_coord(1) + .3 * rng.normal(size=3)]
        ref = mol.intor('int1e_grids_sph', grids=numpy.array(pts))
        for i, p in enumerate(pts):
            k = cosx.COSX(mol, grids=_OnePoint(p, mol.nbas), fit=False, tol=0.)
            k.ao_cutoff = 0.
            vk = k.get_k(dms, hermi=0)
            phik = mol.eval_gto('GTOval_sph', numpy.asarray(p)[None])[0, kp]
            a = vk[:, kp, :].T / phik ** 2
            self.assertLess(abs(a[:kp, :kp] - ref[i, :kp, :kp]).max() / abs(ref[i, :kp, :kp]).max(), 1e-11, i)

    def test_derivatives_fd(self):
        # fixed grid, no fit: d tr(D K[D]) / dR and d K / dR C against central differences
        rng = numpy.random.default_rng(11)
        basA = [[0, [1.5, 1.]]] + _shells(rng, 5)[4:]
        basB = [[1, [.9, 1.]]] + _shells(rng, 4)[3:]
        atom = [['Xe', [0., 0., 0.]], ['Kr', [0.3, 1.0, -1.2]]]

        def make(at):
            return gto.M(atom=at, basis={'Xe': basA, 'Kr': basB}, unit='Bohr', verbose=0, spin=None)
        mol = make(atom)
        grids = cosx.make_grids(mol, intacc=3.0, angular_grid=(14, 14, 26, 26, 14))
        C = numpy.linalg.qr(rng.normal(size=(mol.nao, 4)))[0] * .5
        dm = 2 * C @ C.T

        def kobj(m):
            k = cosx.COSX(m, grids=grids, fit=False, tol=1e-14)
            k.ao_cutoff = 1e-14
            return k
        g = kobj(mol).grad(dm)
        f1 = kobj(mol).fock1(C)
        h = 1e-4
        for ia, x in ((0, 2), (1, 0)):
            at_p = [[s, list(c)] for s, c in atom]
            at_m = [[s, list(c)] for s, c in atom]
            at_p[ia][1][x] += h
            at_m[ia][1][x] -= h
            mp, mm = make(at_p), make(at_m)
            ge = (kobj(mp).energy(dm) - kobj(mm).energy(dm)) / (2 * h)
            fe = (kobj(mp).get_k(dm) @ C - kobj(mm).get_k(dm) @ C) / (2 * h)
            self.assertAlmostEqual(g[ia, x], ge, delta=1e-6 * max(1, abs(ge)))
            self.assertLess(abs(f1[ia, x] - fe).max(), 2e-6 * max(1, abs(fe).max()))


if __name__ == '__main__':
    unittest.main()
