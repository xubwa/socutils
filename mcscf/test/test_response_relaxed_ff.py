'''Relaxed one-electron property of a state of a state-averaged spinor CASSCF through the
Z-vector of zmc_response, against finite fields with the SA-CASSCF redone in the field:

    d E_k / d lambda = tr(O D_k) + tr(O D_z),   J z = -G_k,

O a spinor AO operator added to h.  Water 6-31G, CAS(4e in 8 spinors), SA over the
singlet ground state and the triplet (four states), the target the ground state.

    python mcscf/test/test_response_relaxed_ff.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_response
from socutils.fci import zfci

NROOTS = 4
W = [1.0 / NROOTS] * NROOTS
mol = gto.M(atom='O 0 0 0.1173; H 0 0.7172 -0.4892; H 0 -0.8072 -0.4492', basis='6-31g',
            verbose=0)
mf = spinor_hf.KRHF(mol).density_fit()
mf.kernel()
h0 = mf.get_hcore()
op = mol.intor('int1e_r_spinor')[2]
assert op.shape == h0.shape


def casscf(lam, mo0=None, ci0=None):
    mc = zmcscf.CASSCF(mf, 8, 4)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.fcisolver.nroots = NROOTS
    mc = mc.state_average_(W)
    mc.get_hcore = lambda *a: h0 + lam * op
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-6
    mc.verbose = 0
    mc.kernel(mf.mo_coeff if mo0 is None else mo0)
    mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), W, verbose=False)
    print('  SA-CASSCF lambda %+.0e: |G| after polish %.1e' % (lam, gn), flush=True)
    return mc, mo, ci


def state_energy(mc, mo, ci, k):
    return zmc_response.sa_energy(mc, mo, [ci[k]], [1.0])


mc, mo, ci = casscf(0.0)
resp = zmc_response.SAResponse(mc, mo, ci, W)
print('state energies', np.round(resp.e, 8), flush=True)
k = 0
z = zmc_response.zvector(resp, zmc_response.state_gradient(resp, k))
dk = zmc_response.state_density(resp, k)
dz = zmc_response.lagrangian_density(resp, z)
unrel = np.einsum('ij,ji', op, dk).real
relaxed = unrel + np.einsum('ij,ji', op, dz).real
fd = {}
for h in (2e-3, 1e-3):
    vals = []
    for s in (1, -1):
        m, mo2, ci2 = casscf(s * h, mo)
        vals.append(state_energy(m, mo2, ci2, k))
    fd[h] = (vals[0] - vals[1]) / (2 * h)
ff = (4 * fd[1e-3] - fd[2e-3]) / 3
print('state %d: unrelaxed %.8f  relaxed %.8f  finite field %.8f (h 1e-3: %.8f)'
      % (k, unrel, relaxed, ff, fd[1e-3]))
assert abs(relaxed - ff) < 1e-6, (relaxed, ff)
print('relaxed spinor SA-CASSCF property PASSES')
