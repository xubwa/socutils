#!/usr/bin/env python
'''
Complete-basis-set (multiwavelet, VAMPyR/MRCPP) value of the c^-4 coefficient
of the sf-X2C-1e picture-change error of the dipole operator, one-electron
model HeH2+ (He at 0, H at +z R_AB, Gaussian nuclear charges exp(-zeta r^2)),
matched to gauss_heh2p_dipole.py.

Theory (complete basis, modified sf-Dirac, operators T = p^2/2, W_V = p.V p):
    X~ = 1 + X2/c^2 + ...,   X2 = T^-1 W_V / 4 - (V + T)/2
    dM_l = <phi_FW|PC(O) - O|phi_FW>
         = c^-2 (1/8) <lap O>                                  (= 0, O harmonic)
         + c^-4 [ (5/32) <[T,[T,O]]> + (1/8) Re<psi|W_V T^-1 [O,T]|psi> ] + O(c^-6)
with psi the nonrelativistic state.  For O = z: [O,T] = d/dz, [T,[T,z]] = 0,
    a4 = (1/8) <psi| W_V d_z T^-1 |psi> = (1/8) sum_a <V d_a psi | d_a d_z u>,
    u = T^-1 psi = 2 (-lap)^-1 psi  (one Poisson solve).
Cross-check (eigenstate form, W_V = TV + VT + lap V/2, T psi = (E - V) psi):
    <W_V d_z u> = <(E-V) psi|V d_z u> + <V psi|d_z psi> + 2 pi <rho_nuc psi|d_z u>
(lap V = 4 pi rho_nuc for the Gaussian nuclear charge density rho_nuc).

Usage: python mw_heh2p_dipole.py [prec] [order] [L]
Env: QDHF_ZETA (default 1e4), QDHF_RAB (default 2.0)
'''
import os
import sys
import time
import numpy as np
from vampyr import vampyr3d as vp

prec = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
order = int(sys.argv[2]) if len(sys.argv) > 2 else 7
L = int(sys.argv[3]) if len(sys.argv) > 3 else 32
zeta = float(os.environ.get('QDHF_ZETA', '1e4'))
rab = float(os.environ.get('QDHF_RAB', '2.0'))
nuclei = [(2.0, [0.0, 0.0, 0.0]), (1.0, [0.0, 0.0, rab])]

t0 = time.time()
mra = vp.MultiResolutionAnalysis(box=[-L, L], order=order)
P = vp.ScalingProjector(mra, prec)
D = vp.ABGVDerivative(mra, 0.5, 0.5)
Pois = vp.PoissonOperator(mra, prec)


def gauss(alpha, beta, pos):
    return P(vp.GaussFunc(alpha=alpha, beta=beta, position=pos))


# nuclear charge density (positive) and V = -(-lap)^-1 4 pi rho  (attractive)
rho = None
for z, pos in nuclei:
    g = gauss(z * (zeta / np.pi)**1.5, zeta, pos)
    rho = g if rho is None else rho + g
V = -4 * np.pi * Pois(rho)
print('MW: prec %.1e order %d box [-%g,%g]  zeta %g  R_AB %g' % (prec, order, L, L, zeta, rab))
print('  int rho_nuc = %.12f (3)' % rho.integrate())


def kinetic(psi):
    return 0.5 * sum(vp.dot(D(psi, a), D(psi, a)) for a in range(3))


def energy(psi):
    return kinetic(psi) + vp.dot(psi, V * psi)


# guess: STO-6G 1s (zeta = 2) on He plus a little on H
sto6g = [(23.10303149, 0.00916359628), (4.235915534, 0.04936149294), (1.185056519, 0.16853830490),
         (0.4070988982, 0.37056279970), (0.1580884151, 0.41649152980), (0.06510953954, 0.13033408410)]
psi = None
for a, d in sto6g:
    a = a * 4.0
    g = gauss(d * (2 * a / np.pi)**0.75, a, [0.0, 0.0, 0.0])
    psi = g if psi is None else psi + g
psi = psi + 0.1 * gauss(1.0, 1.0, [0.0, 0.0, rab])
psi.normalize()
E = energy(psi)
# fixed-shift Helmholtz iteration: (-lap + mu^2) psi = (mu^2 + 2E - 2V) psi
E_shift = float(os.environ.get('MW_E_SHIFT', '-2.5'))
mu = np.sqrt(-2 * E_shift)
G = vp.HelmholtzOperator(mra, exp=mu, prec=prec)
print('  Helmholtz operator built (mu = %.4f), %.0f s' % (mu, time.time() - t0), flush=True)
for it in range(200):
    psi_new = G((mu**2 + 2 * E) * psi - 2 * (V * psi))
    psi_new.normalize()
    upd = (psi_new - psi).norm()
    psi = psi_new.crop(prec)
    psi.normalize()
    E = energy(psi)
    if it % 5 == 0 or upd < 10 * prec:
        print('  it %3d  E % .12f  |dpsi| %.2e  (%.0f s)' % (it, E, upd, time.time() - t0), flush=True)
    if upd < prec:
        break
E = energy(psi)
print('  E_NR = % .12f  (%.0f s, %d nodes)' % (E, time.time() - t0, psi.nNodes()))

zf = P(lambda r: r[2])
mu_nr = vp.dot(psi, zf * psi)
print('  <z>_NR = % .12f' % mu_nr)

u = 2 * Pois(psi)                       # T^-1 psi
dpsi = [D(psi, a) for a in range(3)]
duz = D(u, 2)
# expression A: sum_a <V d_a psi | d_a d_z u>
wa = sum(vp.dot(V * dpsi[a], D(duz, a)) for a in range(3))
# expression B (eigenstate form)
wb = (vp.dot((E * psi - V * psi), V * duz) + vp.dot(V * psi, dpsi[2])
      + 2 * np.pi * vp.dot(rho * psi, duz))
print('  <W_V T^-1 d_z>:  A % .10e   B % .10e' % (wa, wb))
print('  a4 = <W_V T^-1 d_z>/8:  A % .10e   B % .10e' % (wa / 8, wb / 8))
print('  check: <[T,d_z]>-like  <psi|d_z psi> = % .2e (should be 0)' % vp.dot(psi, dpsi[2]))
print('  total time %.0f s' % (time.time() - t0))
