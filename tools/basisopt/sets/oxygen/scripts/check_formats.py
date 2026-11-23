"""The GENBAS entries and the PySCF files give the same energies (sf-X2C-1e RHF + CCSD; He atom, H2 at 1.4 a0)."""
import os
from pyscf import gto, scf, cc
from socutils.tools.basisopt import genbas
here = os.path.dirname(os.path.abspath(__file__))
for z in 'DTQ5':
    py = genbas.read_pyscf(os.path.join(here, 'pyscf', 'pv%sz_x2c.py' % z.lower()))
    for el, atom in (('H', 'H 0 0 0; H 0 0 1.4'), ('He', 'He 0 0 0')):
        gb = genbas.to_pyscf(genbas.read(os.path.join(here, 'GENBAS'), '%s:PV%sZ-X2C' % (el.upper(), z)), kappa=True)
        out = []
        for b in (gb, py[el]):
            m = gto.M(atom=atom, unit='Bohr', basis={el: b}, verbose=0)
            mf = scf.RHF(m).sfx2c1e(); mf.conv_tol = 1e-11; mf.kernel()
            c = cc.CCSD(mf); c.conv_tol = 1e-10; c.kernel()
            out.append((mf.e_tot, c.e_corr, m.nao_nr()))
        print('%-2s PV%sZ-X2C  nao %2d  E(HF) %.7f  E(corr) %.6f   GENBAS - PySCF file: %.1e %.1e' % (
            el, z, out[0][2], out[0][0], out[0][1], out[0][0] - out[1][0], out[0][1] - out[1][1]))
