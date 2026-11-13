'''Zero-dimensional check of a periodic Coulomb builder with tight all-electron
functions: ZnH2 (def2-SVP, Zn exponents up to 8e4, point nuclei) in cubic boxes.

The total Coulomb energy of a neutral, dipole-free molecule in a box must
approach the molecular value as E(L) - E_mol = A/L^5, so (E(L) - E_mol)*L^5
has to be constant over L.  An error that does not vanish with L shows up as
a drift of that column.  Reference run (pyscf 2.14):

    builder                L=20    L=26    L=32    (E - E_mol)*L^5
    RSJK J + GDF nuc       392.1   390.2   389.5   -> constant: passes
    GDF J + GDF nuc        343.0   208.0  -125.2   -> GDF J has a constant
                                                      -1.5e-5 Eh fitting error

Run:  python check_zero_dim_pyscf.py   (about 5 minutes)
'''
import time
import numpy as np
from pyscf import gto, scf
from pyscf.pbc import df as pdf
from pyscf.pbc.scf import rsjk
from socutils.pbc import coulomb_reference as cr


def main(boxes=(20., 26., 32.), precision=1e-8):
    mol = gto.M(atom='Zn 0 0 0; H 0 0 -3.0; H 0 0 3.0', unit='B', basis='def2-svp', verbose=0)
    dm = scf.RHF(mol).run().make_rdm1()
    nao = mol.nao
    m = cr.molecular_coulomb(mol, dm)
    dip, tr2 = cr.dipole_and_second_moment(mol, dm)
    S = mol.intor('int1e_ovlp')
    print('ZnH2: nao %d, |dipole| %.1e, molecular E_coul %.10f' % (nao, np.linalg.norm(dip), m['e_coul']))

    def report(tag, L, vj, vnuc, enn, t):
        e = .5 * np.einsum('ij,ji', dm, vj) + np.einsum('ij,ji', dm, vnuc) + enn
        de = e - m['e_coul']
        dV = (vj + vnuc) - (m['vj'] + m['vnuc'])
        c = -2 * np.pi / (3 * L**3) * tr2
        print('%-18s L=%3.0f  E-E_mol=%10.3e  (E-E_mol)*L^5=%9.2f  max|dV - c S|=%8.1e  %4.0fs'
              % (tag, L, de, de * L**5, abs(dV - c * S).max(), t))

    for L in boxes:
        cell = cr.molecule_in_box(mol, L, precision=precision)
        t = time.time()
        gdf = pdf.GDF(cell)
        vnuc = np.asarray(gdf.get_nuc()).reshape(nao, nao).real
        enn = cell.energy_nuc()
        vj = gdf.get_jk(dm, with_k=False)[0]
        report('GDF J', L, vj, vnuc, enn, time.time() - t)
        t = time.time()
        jk = rsjk.RangeSeparatedJKBuilder(cell)
        jk.build()
        vj = jk.get_jk(dm[None], hermi=1, kpts=np.zeros((1, 3)), with_k=False)[0][0]
        report('RSJK J', L, vj, vnuc, enn, time.time() - t)


if __name__ == '__main__':
    main()
