#!/usr/bin/env python
'''
Multiwavelet (VAMPyR/MRCPP) closed-shell nonrelativistic Hartree-Fock of He2
(He at z = -+1.5 Angstrom, Gaussian nuclear charges exp(-zeta r^2)) and the
complete-basis-set c^-4 coefficient of the sf-X2C-1e (approx='1e')
picture-change error of the quadrupole operator

    O = r^2 P2(cos theta) = z^2 - (x^2 + y^2)/2     (origin at the midpoint)

    dM2(c) = Tr[D (PC(O) - O)] = a2/c^2 + a4/c^4 + ...
    a2 = (1/8) <lap O> = 0                                 (O harmonic)
    a4 = sum_i 2 [ (5/32) <phi_i|[T,[T,O]]|phi_i> + (1/8) <phi_i|W_V T^-1 [O,T]|phi_i> ]
       = a4_H1 + a4_H2kin + a4_W
    a4_H1    = (1/32) <[T,[T,O]]>     (= H1 = 1/2 [[R,V],R^-1] in the CBS limit)
    a4_H2kin = (4/32) <[T,[T,O]]>     (from the -T/2 part of X2 in H2)
    a4_W     = (1/8)  <W_V T^-1 [O,T]> (from the T^-1 W_V/4 part of X2 in H2)
    [O,T] = grad O . grad,  [T,[T,O]] = sum_jk O_jk d_j d_k,
    <phi|[T,[T,O]]|phi> = -(2 <d_z phi|d_z phi> - <d_x phi|d_x phi> - <d_y phi|d_y phi>)
    <phi|W_V T^-1 [O,T]|phi> = sum_a <V d_a phi | d_a u>,  u = T^-1 (grad O . grad phi)
       (expression A);  cross-check B with W_V = TV + VT + lap V/2:
       1/2 sum_a <d_a phi|d_a (V u)> + <V phi|grad O . grad phi> + 2 pi <rho_nuc phi|u>
Far charge Z_B on +z at distance R:  E_spurious -> -Z_B dM2 / R^3, i.e.
R^3 E = -Z_B dM2 (= -2 dM2 for script 14's Z_B = 2).

HF: two doubly occupied orbitals, fixed-shift Helmholtz iterations
    phi_i <- G_mu[ mu^2 phi_i + 2 sum_j phi_j F_ji - 2 (V + J - K) phi_i ]
with DIIS on the orbital residuals, Loewdin orthonormalization, run through a
sequence of precisions (orbitals carried over).

Usage: python mw_he2_quadrupole.py [prec1,prec2,...] [order] [L]
Env: QDHF_ZETA (default 1e4)
'''
import os
import sys
import time
import numpy as np
from vampyr import vampyr3d as vp

precs = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '1e-5,1e-6').split(',')]
order = int(sys.argv[2]) if len(sys.argv) > 2 else 9
L = int(sys.argv[3]) if len(sys.argv) > 3 else 32
zeta = float(os.environ.get('QDHF_ZETA', '1e4'))
zh = 1.5 / 0.52917721092           # pyscf BOHR
nuclei = [(2.0, [0.0, 0.0, -zh]), (2.0, [0.0, 0.0, zh])]
t0 = time.time()
mra = vp.MultiResolutionAnalysis(box=[-L, L], order=order)
print('He2 MW NR-HF: order %d box [-%d,%d] zeta %g  z_He = +-%.10f bohr' % (order, L, L, zeta, zh))


def log(*a):
    print(*a, '(%.0f s)' % (time.time() - t0), flush=True)


class Ctx:
    def __init__(self, prec):
        self.prec = prec
        self.P = vp.ScalingProjector(mra, prec)
        self.D = vp.ABGVDerivative(mra, 0.5, 0.5)
        self.Pois = vp.PoissonOperator(mra, prec)
        rho = None
        for z, pos in nuclei:
            g = self.P(vp.GaussFunc(alpha=z * (zeta / np.pi)**1.5, beta=zeta, position=pos))
            rho = g if rho is None else rho + g
        self.rho = rho
        self.V = -4 * np.pi * self.Pois(rho)

    def grad(self, f):
        return [self.D(f, a) for a in range(3)]


def tsum(trees):
    '''sum of a sequence of FunctionTrees (python sum starts from int 0)'''
    out = None
    for t in trees:
        out = t if out is None else out + t
    return out


def lowdin(phis):
    n = len(phis)
    s = np.array([[vp.dot(phis[i], phis[j]) for j in range(n)] for i in range(n)])
    w, v = np.linalg.eigh(s)
    x = v @ np.diag(w**-0.5) @ v.T
    return [tsum(float(x[j, i]) * phis[j] for j in range(n)) for i in range(n)]


def fock(ctx, phis):
    n = len(phis)
    rho = 2.0 * tsum(p * p for p in phis)
    J = 4 * np.pi * ctx.Pois(rho)
    vphi = []
    for i in range(n):
        kphi = tsum(phis[j] * (4 * np.pi * ctx.Pois(phis[j] * phis[i])) for j in range(n))
        vphi.append((ctx.V + J) * phis[i] - kphi)
    grads = [ctx.grad(p) for p in phis]
    T = np.array([[0.5 * sum(vp.dot(grads[i][a], grads[j][a]) for a in range(3))
                   for j in range(n)] for i in range(n)])
    Vm = np.array([[vp.dot(phis[i], vphi[j]) for j in range(n)] for i in range(n)])
    Vm = 0.5 * (Vm + Vm.T)
    Vne = np.array([vp.dot(phis[i], ctx.V * phis[i]) for i in range(n)])
    Jd = np.array([vp.dot(phis[i], J * phis[i]) for i in range(n)])
    F = T + Vm
    # E = sum_i 2 h_ii + <J>_i - <K>_i ;  F_ii = h_ii + <J>_i - <K>_i
    h = np.diag(T) + Vne
    E = float(np.sum(h + np.diag(F)))
    return F, vphi, E


def scf(ctx, phis, mu, maxiter=80, ndiis=6):
    G = vp.HelmholtzOperator(mra, exp=mu, prec=ctx.prec)
    hist = []
    E_old = 0.0
    for it in range(maxiter):
        F, vphi, E = fock(ctx, phis)
        n = len(phis)
        new = [G(float(mu**2) * phis[i] + 2.0 * tsum(float(F[j, i]) * phis[j] for j in range(n)) - 2.0 * vphi[i])
               for i in range(n)]
        res = [new[i] - phis[i] for i in range(n)]
        rnorm = np.sqrt(sum(r.squaredNorm() for r in res))
        hist.append((phis, res))
        hist = hist[-ndiis:]
        m = len(hist)
        B = -np.ones((m + 1, m + 1))
        B[m, m] = 0
        for a in range(m):
            for b in range(m):
                B[a, b] = sum(vp.dot(hist[a][1][i], hist[b][1][i]) for i in range(n))
        rhs = np.zeros(m + 1)
        rhs[m] = -1
        try:
            cvec = np.linalg.solve(B, rhs)[:m]
        except np.linalg.LinAlgError:
            cvec = np.zeros(m)
            cvec[-1] = 1
        phis = [tsum(float(cvec[k]) * (hist[k][0][i] + hist[k][1][i]) for k in range(m)).crop(ctx.prec)
                for i in range(n)]
        phis = lowdin(phis)
        log('  prec %.0e it %2d  E_elec % .12f  dE % .2e  |res| %.2e  eps %s' % (
            ctx.prec, it, E, E - E_old, rnorm, np.array2string(np.diag(F), precision=8)))
        if rnorm < ctx.prec and abs(E - E_old) < 0.1 * ctx.prec:
            break
        E_old = E
    F, vphi, E = fock(ctx, phis)
    return phis, E, F


def analyse(ctx, phis):
    P = ctx.P
    xf = [P(lambda r, a=a: r[a]) for a in range(3)]
    gO = [-1.0 * xf[0], -1.0 * xf[1], 2.0 * xf[2]]          # grad O
    Of = xf[2] * xf[2] - 0.5 * (xf[0] * xf[0] + xf[1] * xf[1])
    tot = dict(Q=0.0, K=0.0, WA=0.0, WB=0.0, a2=0.0)
    for i, p in enumerate(phis):
        g = ctx.grad(p)
        nn = [vp.dot(g[a], g[a]) for a in range(3)]
        K = -(2 * nn[2] - nn[0] - nn[1])
        gop = gO[0] * g[0] + gO[1] * g[1] + gO[2] * g[2]     # [O,T] phi
        u = 2 * ctx.Pois(gop)                                  # T^-1 [O,T] phi
        du = ctx.grad(u)
        WA = sum(vp.dot(ctx.V * g[a], du[a]) for a in range(3))
        dvu = ctx.grad(ctx.V * u)
        WB = (0.5 * sum(vp.dot(g[a], dvu[a]) for a in range(3)) + vp.dot(ctx.V * p, gop)
              + 2 * np.pi * vp.dot(ctx.rho * p, u))
        a2 = -0.25 * sum(vp.dot(g[a], gO[a] * p) for a in range(3))
        Q = vp.dot(p, Of * p)
        log('  orbital %d: <O> % .10e  <[T,[T,O]]> % .10e  <W T^-1[O,T]> A % .10e B % .10e  a2-check % .1e'
            % (i, Q, K, WA, WB, a2))
        for k, val in zip(('Q', 'K', 'WA', 'WB', 'a2'), (Q, K, WA, WB, a2)):
            tot[k] += 2 * val
    h1 = tot['K'] / 32
    h2k = tot['K'] * 4 / 32
    wa, wb = tot['WA'] / 8, tot['WB'] / 8
    print('  RESULT prec %.0e: Q_NR = <O> = % .10e   a2 = (1/4)<W_O-TO-OT> = % .2e (exact 0)' % (
        ctx.prec, tot['Q'], tot['a2']))
    print('  RESULT prec %.0e: a4_H1 = % .8e  a4_H2kin = % .8e  a4_W = % .8e (A) % .8e (B)' % (
        ctx.prec, h1, h2k, wa, wb))
    print('  RESULT prec %.0e: a4 = % .8e (A)  % .8e (B)' % (ctx.prec, h1 + h2k + wa, h1 + h2k + wb),
          flush=True)


# guess: STO-6G 1s (zeta 1.6875) on each He, sigma_g / sigma_u
sto6g = [(23.10303149, 0.00916359628), (4.235915534, 0.04936149294), (1.185056519, 0.16853830490),
         (0.4070988982, 0.37056279970), (0.1580884151, 0.41649152980), (0.06510953954, 0.13033408410)]
ctx = Ctx(precs[0])
s1 = []
for z, pos in nuclei:
    f = None
    for a, d in sto6g:
        a = a * 1.6875**2
        g = ctx.P(vp.GaussFunc(alpha=d * (2 * a / np.pi)**0.75, beta=a, position=pos))
        f = g if f is None else f + g
    s1.append(f)
phis = lowdin([s1[0] + s1[1], s1[0] - s1[1]])
mu = np.sqrt(-2 * float(os.environ.get('MW_EPS_SHIFT', '-0.918')))
for prec in precs:
    if prec != ctx.prec:
        ctx = Ctx(prec)
    log('== precision %.0e, int rho_nuc = %.12f' % (prec, ctx.rho.integrate()))
    phis, E, F = scf(ctx, phis, mu)
    log('  E_elec(HF) = % .12f   eps = %s   nodes %s' % (
        E, np.array2string(np.diag(F), precision=10), [p.nNodes() for p in phis]))
    analyse(ctx, phis)
log('done')
