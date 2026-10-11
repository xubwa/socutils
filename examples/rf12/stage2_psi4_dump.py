#!/usr/bin/env python
'''
Run with Psi4 (>= 1.11): dump Psi4's own MP2-F12/3C(FIX) ingredients
(orbital spaces, Fock/exchange in the OBS+CABS basis, MO integrals with
Psi4's 6-Gaussian fit of the Slater geminal) so that the energy assembly
of socutils.rf12.nr.mp2f12 can be checked independently of the geminal
and of the integral library.

    psi4-python stage2_psi4_dump.py H2O aug-cc-pvdz 0 out.npz [laplace DT]

geminal: Psi4's default 6-Gaussian fit of -exp(-r12) (Tew-Klopper), or with
'laplace DT' a Gaussian expansion of the exact Slater function from the
Laplace transform  exp(-r) = int_0^inf (2 sqrt(pi))^-1 s^-3/2 exp(-1/(4s))
exp(-s r^2) ds  on a uniform grid in t = ln s with spacing DT.
'''
import sys
import numpy as np
import psi4
from psi4.driver import qcdb

GEOMS = {
    'He': "0 1\nHe 0 0 0\nsymmetry c1\nunits bohr",
    'Ne': "0 1\nNe 0 0 0\nsymmetry c1\nunits bohr",
    'H2O': "0 1\nO 0.0 0.0 -0.124038860300\nH 0.0 -1.431430901356 0.984293362719\n"
           "H 0.0 1.431430901356 0.984293362719\nsymmetry c1\nunits bohr\nno_reorient\nno_com",
}

name, basis, nfz, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
psi4.set_memory('6 GB')
psi4.core.set_output_file(out + '.log', False)
mol = psi4.geometry(GEOMS[name])
psi4.set_options({'basis': basis, 'scf_type': 'pk', 'e_convergence': 1e-12, 'd_convergence': 1e-10,
                  'puream': True})
e_hf, wfn = psi4.energy('scf', return_wfn=True)
obs = wfn.basisset()
combined = qcdb.libmintsbasisset.BasisSet.pyconstruct_combined(
    mol.to_string(dtype='psi4'), ['BASIS', 'CABS_BASIS'], [basis, basis + '-optri'],
    ['ORBITAL', 'F12'], [basis, basis])
cabs_bs = psi4.core.BasisSet.construct_from_pydict(mol, combined, combined['puream'])

OBS = wfn.alpha_orbital_space('p', 'SO', 'ALL')
RI = psi4.core.OrbitalSpace.build_ri_space(cabs_bs, 1.0e-8)
CABS = psi4.core.OrbitalSpace.build_cabs_space(OBS, RI, 1.0e-6)

mints = psi4.core.MintsHelper(cabs_bs)
S = mints.ao_overlap().np
S_co = mints.ao_overlap(cabs_bs, obs).np
X = RI.C().np                                   # orthonormalizer (zero columns for lindep)
C_obs = X @ X.T @ S_co @ wfn.Ca().np            # OBS MOs expressed in the combined AO basis
C_cabs = CABS.C().np
C1 = np.hstack([C_obs, C_cabs])
nocc = wfn.doccpi()[0]
nmo = C_obs.shape[1]
print('orthonormality', abs(C1.T @ S @ C1 - np.eye(C1.shape[1])).max())

# Fock and exchange in the "1" basis
H = mints.ao_kinetic().np + mints.ao_potential().np
jk = psi4.core.JK.build(cabs_bs, jk_type='PK')
jk.set_memory(int(4e9))
jk.initialize()
jk.C_left_add(psi4.core.Matrix.from_array(C_obs[:, :nocc]))
jk.compute()
J = jk.J()[0].np
Kao = jk.K()[0].np
fock = C1.T @ (H + 2 * J - Kao) @ C1
K = C1.T @ Kao @ C1

sp = {'a': C1[:, nfz:nocc], 'o': C1[:, :nocc], 'v': C1[:, nocc:nmo], 'P': C1[:, :nmo],
      'c': C1[:, nmo:], '1': C1}
if len(sys.argv) > 5 and sys.argv[5] == 'laplace':
    dt = float(sys.argv[6])
    t = np.arange(-14.0, 34.0, dt)
    sk = np.exp(t)
    wk = dt * sk * sk ** -1.5 * np.exp(-0.25 / sk) / (2 * np.sqrt(np.pi))
    cgtg = [(float(a), float(-w)) for a, w in zip(sk, wk) if w > 1e-14]
    print('Laplace expansion: %d Gaussians' % len(cgtg))
else:
    cgtg = mints.f12_cgtg(1.0)
M = lambda a: psi4.core.Matrix.from_array(a)


def phys(kind, lab):
    c = [sp[x] for x in lab]
    # <12|34> = (13|24)
    args = (M(c[0]), M(c[2]), M(c[1]), M(c[3]))
    if kind == 'G':
        t = mints.mo_eri(*args)
    elif kind == 'F':
        t = mints.mo_f12(cgtg, *args)
    elif kind == 'FG':
        t = mints.mo_f12g12(cgtg, *args)
    elif kind == 'F2':
        t = mints.mo_f12_squared(cgtg, *args)
    elif kind == 'U':
        t = mints.mo_f12_double_commutator(cgtg, *args)
    t = t.np.reshape(c[0].shape[1], c[2].shape[1], c[1].shape[1], c[3].shape[1])
    return t.transpose(0, 2, 1, 3).copy()


need = [('G', 'aaPP'), ('G', 'aaoc'), ('F', 'aaPP'), ('F', 'aaoc'), ('F', 'aavc'), ('F', 'aa11'),
        ('F2', 'aaaa'), ('F2', 'aaa1'), ('FG', 'aaaa'), ('U', 'aaaa')]
data = {'%s_%s' % k: phys(*k) for k in need}
np.savez(out, fock=fock, K=K, e_hf=e_hf, nocc=nocc, nfz=nfz, nmo=nmo, nri=C1.shape[1], **data)
print('dumped', out, C1.shape)
