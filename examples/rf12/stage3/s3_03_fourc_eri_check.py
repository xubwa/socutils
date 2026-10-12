import numpy
from pyscf import gto, scf, lib
from socutils.rf12 import fourc
mol = gto.M(atom='Ne 0 0 0; H 0 0 1.8', unit='bohr', basis='cc-pvdz', charge=1, verbose=0)
c = 137.035999084
with fourc.light_speed(c):
    mf = scf.DHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    print('DHF', mf.e_tot)
    n2c = mol.nao_2c()
    mo = mf.mo_coeff
    nocc = mol.nelectron
    o = slice(n2c, n2c + nocc)
    sel = numpy.r_[n2c:n2c + 3, n2c + nocc:n2c + nocc + 4]      # a few positive-energy orbitals
    orb = fourc.orbitals(mol, mo[:, sel], c)
    mine = fourc.mo_eri(mol, 'G', orb, orb, orb, orb)
    # pyscf spinor integrals, chemist (ij|kl), small basis sigma.p chi / 2c
    LL = mol.intor('int2e_spinor')
    SL = mol.intor('int2e_spsp1_spinor')
    SS = mol.intor('int2e_spsp1spsp2_spinor')
    cL, cS = mo[:n2c, sel], mo[n2c:, sel] / (2 * c)
    t = lambda I, a, b, cc, d: numpy.einsum('ijkl,ip,jq,kr,ls->pqrs', I, a.conj(), b, cc.conj(), d, optimize=True)
    chem = t(LL, cL, cL, cL, cL) + t(SL, cS, cS, cL, cL) + t(SL, cS, cS, cL, cL).transpose(2, 3, 0, 1) + t(SS, cS, cS, cS, cS)
    ref = chem.transpose(0, 2, 1, 3)      # <pr|qs> = (pq|rs)
    print('max |mine - pyscf| =', abs(mine - ref).max(), ' max |ref| =', abs(ref).max())
