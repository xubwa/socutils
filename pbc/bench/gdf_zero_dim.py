'''Separate DF fitting error from periodic-integral error: YH3 in a vacuum box.

The same auxiliary basis is used in molecular DF (pure fitting error, vs the
exact molecular J) and in periodic GDF (vs RSJK J in the same box).  In vacuum
the two agree, i.e. the GDF breakdown seen in dense Y2O3 with compact
auxiliary functions comes from the lattice sums, not from the auxiliary basis.

    python gdf_zero_dim.py
'''
import time
import numpy as np
from pyscf import gto, scf, df
from pyscf.df import addons as dfaddons
from pyscf.pbc import df as pdf
from pyscf.pbc.scf import rsjk
from pyscf.pbc.df import rsdf_builder
from socutils.pbc import coulomb_reference as cr
from socutils.pbc.bench.auxgen_util import auxgen


def main(L=22.):
    r = 1.95 / 0.52917721092
    mol = gto.M(atom=[('Y', (0, 0, 0))] + [('H', (r * np.cos(t), r * np.sin(t), 0))
                                           for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)],
                unit='B', basis='x2c-svpall', verbose=0)
    dm = scf.RHF(mol).run().make_rdm1()
    ej = lambda v: .5 * np.einsum('ij,ji', dm, v)
    j_exact = scf.hf.get_jk(mol, dm, with_k=False)[0]
    cell = cr.molecule_in_box(mol, L, precision=1e-10)
    jk = rsjk.RangeSeparatedJKBuilder(cell)
    jk.build()
    j_box = jk.get_jk(dm[None], hermi=1, kpts=np.zeros((1, 3)), with_k=False)[0][0]
    print('YH3 nao %d, box L = %.0f bohr' % (mol.nao, L), flush=True)
    for tag, mkaux in [('ETB beta=2.0', lambda: dfaddons.aug_etb(mol, beta=2.0)),
                       ('ETB beta=1.3', lambda: dfaddons.aug_etb(mol, beta=1.3)),
                       ('auxgen unc. tau=1e-8', lambda: auxgen(mol, contract=False, prune_lmax=False, threshold=1e-8))]:
        aux = mkaux()
        mdf = df.DF(mol, auxbasis=aux)
        mdf.build()
        e_mol = ej(mdf.get_jk(dm, with_k=False)[0] - j_exact)
        for eig in (False, True):
            rsdf_builder._RSGDFBuilder.j2c_eig_always = eig
            g = pdf.GDF(cell)
            g.auxbasis = aux
            t = time.time()
            g.build()
            e_box = ej(g.get_jk(dm, with_k=False)[0] - j_box)
            print('%-22s naux %4d | molecular DF dE_J %9.1e | periodic GDF (eigh=%-5s) dE_J %9.1e  %.0fs'
                  % (tag, mdf.get_naoaux(), e_mol, eig, e_box, time.time() - t), flush=True)
        rsdf_builder._RSGDFBuilder.j2c_eig_always = False


if __name__ == '__main__':
    main()
