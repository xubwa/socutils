#!/usr/bin/env python
'''
The x2cmp-1x flavor: x2cmp with one-centre exchange (cf. Neese's SOMF(1X)).

x2cmp decouples F4 = h4 + (J - K)[sum_K D_K] with the molecular J and K of the
superposed atomic densities.  x2cmp-1x keeps J molecular (exact, from the
spherical atomic densities with one-electron-type libcint integrals) and
takes K one-centre: atom blocks K_K[D_K] (Coulomb + Gaunt/Breit exchange).

  flavor='x2cmp_1x'       atom-block-diagonal one-centre K
  flavor='x2cmp_1x_proj'  K in projector form  S_.K S_KK^-1 K_K S_KK^-1 S_K.
                          (same as x2cmp_1x with with_x2c.proj_1x = True)
  with_x2c.j_1x = 'direct' builds J with a molecular get_j instead.

The projector form stays on x2cmp when diffuse functions are added: for HCl
(uncontracted cc-pVDZ + 8 diffuse s,p shells, min eig S 7e-7, Dirac-Coulomb)
E - E(x2cmp) is -2.7e-6 Eh for x2cmp_1x_proj at every k, while the block form
drifts from -1.1e-5 to -1.1e-4 Eh.  ('x2cmp-1x' spellings are accepted.)
'''
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom='H 0 0 0; Cl 0 0 1.275', basis='cc-pvdz', verbose=4)

e_mp = spinor_hf.SCF(mol).x2cmp().kernel()
e_1x = spinor_hf.SCF(mol).x2cmp(flavor='x2cmp-1x').kernel()
e_1xp = spinor_hf.SCF(mol).x2cmp(flavor='x2cmp_1x_proj').kernel()

mf = spinor_hf.SCF(mol).x2cmp(flavor='x2cmp_1x')
mf.with_x2c.proj_1x = True       # switch, same Hamiltonian as 'x2cmp_1x_proj'
e_sw = mf.kernel()

print('E(x2cmp)              = %.10f' % e_mp)
print('E(x2cmp_1x)      - mp = %.3e' % (e_1x - e_mp))
print('E(x2cmp_1x_proj) - mp = %.3e' % (e_1xp - e_mp))
print('switch vs flavor      = %.1e' % (e_sw - e_1xp))
