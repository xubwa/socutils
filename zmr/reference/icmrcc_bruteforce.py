'''
Brute-force internally contracted MRCC (ic-MRCC(SD), one reference state) in
the determinant basis, for validation (docs/derivations/ic-mrcc.md).

Everything is a dense matrix over all determinants of the electron number:

    excitation operators  every elementary string p^+ r (p in active +
                          virtual, r in core + active) and p^+ q^+ s r
                          (p < q, r < s, same ranges), not all indices active
    FIC basis             B = V Y, V = [E_mu |Psi_0>], Y = U s^-1/2 from the
                          eigenvectors of V^+ V above lindep
    T(C)                  sum_mu (Y C)_mu E_mu, so T |Psi_0> = B C
    H_bar                 sum_{k <= m} ad_T^k (H) / k!  (m = None: every term
                          until it vanishes; T raises the external level, so
                          the series ends)
    equations             B^+ H_bar |Psi_0> = 0,  E = <Psi_0| H_bar |Psi_0>

The perturbation orders on H = H_D + V (orders()) follow from the same
operators without fitting.  Nothing of zmr.mrci or zmr.pc is used.
'''

import itertools

import numpy

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.reference import dets
from socutils.zmr.reference.nevpt2_bruteforce import dyall_operator


def excitation_operators(sp):
    '''[(daggers, orbs)] of the elementary excitation strings.'''
    act = set(range(sp.ncore, sp.nocc))
    cre = list(range(sp.ncore, sp.nmo))
    ann = list(range(sp.nocc))
    ops = []
    for p in cre:
        for r in ann:
            if not {p, r} <= act:
                ops.append(((True, False), (p, r)))
    for p, q in itertools.combinations(cre, 2):
        for r, s in itertools.combinations(ann, 2):
            if not {p, q, r, s} <= act:
                ops.append(((True, True, False, False), (p, q, s, r)))
    return ops


def _matrix(op, strs):
    return dets.operator_matrix(op, strs).toarray()


def _comm(a, b):
    return a @ b - b @ a


class ICMRCC:
    '''Args: eris (spaces with ncore, nact, nvirt), nelec active electrons,
    civec of the reference in the CAS determinants (dets.cas_strings order).'''

    def __init__(self, eris, nelec, civec, lindep=1e-10):
        sp = eris.spaces
        self.eris, self.sp, self.nelec = eris, sp, nelec
        self.strs = dets.gen_strings(range(sp.nmo), sp.ncore + nelec)
        cas = dets.cas_strings(sp.ncore, sp.nact, nelec)
        self.psi0 = dets.SparseVector(cas, civec).on(self.strs)
        self.psi0 = self.psi0 / numpy.linalg.norm(self.psi0)
        self.ops = excitation_operators(sp)
        self.mats = [_matrix([(d, numpy.array([o]), numpy.array([1.0]))], self.strs)
                     for d, o in self.ops]
        v = numpy.stack([m @ self.psi0 for m in self.mats], axis=1)
        s, u = numpy.linalg.eigh(v.conj().T @ v)
        keep = s > lindep
        self.y = u[:, keep] / numpy.sqrt(s[keep])
        self.b = v @ self.y
        self.h = _matrix(dets.hamiltonian_operator(eris.h1, eris.eri_block('pppp'), eris.ecore),
                         self.strs)

    @property
    def nbasis(self):
        return self.b.shape[1]

    def t_matrix(self, c):
        amps = self.y @ c
        out = numpy.zeros_like(self.h)
        for a, m in zip(amps, self.mats):
            if a != 0:
                out += a * m
        return out

    @staticmethod
    def hbar(h, t, m=None):
        out, term, k = h.copy(), h, 1
        while m is None or k <= m:
            term = _comm(term, t) / k
            if not abs(term).max() > 0:
                break
            out = out + term
            k += 1
        return out

    def residual(self, c, h=None, m=None):
        '''(B^+ H_bar |Psi_0>, <Psi_0| H_bar |Psi_0>)'''
        h = self.h if h is None else h
        hb = self.hbar(h, self.t_matrix(c), m)
        hp = hb @ self.psi0
        return self.b.conj().T @ hp, numpy.vdot(self.psi0, hp)

    def kernel(self, h=None, m=None, tol=1e-11, max_cycle=200, ndiis=8):
        '''Solve the projected equations; returns (energy, C).  Newton-like
        steps with the linear matrix B^+ (H - E_ref) B, extrapolated by DIIS.'''
        h = self.h if h is None else h
        eref = numpy.vdot(self.psi0, h @ self.psi0).real
        a = self.b.conj().T @ (h @ self.b) - eref * numpy.eye(self.nbasis)
        c = numpy.zeros(self.nbasis, dtype=numpy.complex128)
        hist = []
        for _ in range(max_cycle):
            r, e = self.residual(c, h, m)
            if abs(r).max() < tol:
                return e.real, c
            step = -numpy.linalg.solve(a, r)
            hist.append((c + step, step))
            hist = hist[-ndiis:]
            if len(hist) > 1:
                d = numpy.stack([x[1] for x in hist])
                n = len(hist)
                bm = numpy.ones((n + 1, n + 1), dtype=numpy.complex128)
                bm[:n, :n] = d.conj() @ d.T
                bm[n, n] = 0
                rhs = numpy.zeros(n + 1, dtype=numpy.complex128)
                rhs[n] = 1
                w = numpy.linalg.solve(bm, rhs)[:n]
                c = sum(wi * x[0] for wi, x in zip(w, hist))
            else:
                c = c + step
        raise RuntimeError('ic-MRCC did not converge')

    def orders(self, hd, v):
        '''Perturbation orders on H = H_D + V (H_D |Psi_0> = E0 |Psi_0>):
        dict with e0, e1, e2, e3 of ic-MRCC, the PC-NEVPT3 energy
        e3_pc = <psi1| V - E1 |psi1> (psi1 = B C1), c1, c2.'''
        p = self.psi0
        e0 = numpy.vdot(p, hd @ p).real
        e1 = numpy.vdot(p, v @ p).real
        a0 = self.b.conj().T @ (hd @ self.b) - e0 * numpy.eye(self.nbasis)
        c1 = -numpy.linalg.solve(a0, self.b.conj().T @ (v @ p))
        t1 = self.t_matrix(c1)
        e2 = numpy.vdot(p, _comm(v, t1) @ p)
        rhs = _comm(v, t1) + 0.5 * _comm(_comm(hd, t1), t1)
        c2 = -numpy.linalg.solve(a0, self.b.conj().T @ (rhs @ p))
        t2 = self.t_matrix(c2)
        x3 = (_comm(v, t2) + 0.5 * _comm(_comm(v, t1), t1)
              + 0.5 * (_comm(_comm(hd, t1), t2) + _comm(_comm(hd, t2), t1))
              + _comm(_comm(_comm(hd, t1), t1), t1) / 6)
        e3 = numpy.vdot(p, x3 @ p)
        psi1 = self.b @ c1
        e3_pc = numpy.vdot(psi1, (v - e1 * numpy.eye(len(p))) @ psi1)
        return dict(e0=e0, e1=e1, e2=e2.real, e3=e3.real, e3_pc=e3_pc.real,
                    e2_imag=abs(e2.imag), e3_imag=abs(e3.imag), c1=c1, c2=c2)


def dyall_split(eris, nelec, civec):
    '''(eris in quasi-canonical orbitals of the reference density, H_D as an
    operator) for orders(); the CAS vector is unchanged (active orbitals are
    not rotated).'''
    act = ActiveSpace.from_eris(eris)
    civec = numpy.asarray(civec, dtype=numpy.complex128)
    qc = fock.quasi_canonicalize(eris, act.averaged_rdm1(civec[:, None], nelec, numpy.ones(1)))
    return qc.eris, dyall_operator(qc.eris, qc.eps_core, qc.eps_virt)
