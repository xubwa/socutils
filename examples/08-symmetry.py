import argparse
from pyscf import scf, gto
from socutils.scf import spinor_hf
from socutils.somf import amf, eamf
from socutils.tools import basis_parser 

mol = gto.M(
        atom=f'''
U             -0.00000000     0.00000000     0.21892071
O              0.00000000     0.00000000    -3.25817608
''',
        #basis='unc-ano',
        basis={'O':'uncccpvtz',
               'U':'uncano')},
        nucmod='G',
        unit='Bohr',
        verbose=5)
mf = spinor_hf.SpinorSymmSCF(mol, symmetry='linear',
        occup={
               '1/2':[28,0,0,0,1],
              '-1/2':[28,1,0,0,0],
               '3/2':[14,0,0,0,1],
              '-3/2':[14,0,0,0,0],
               '5/2':[5, 0,0,0,1],
              '-5/2':[5, 0,0,0,0],
               '7/2':[1, 0,0,0,0],
              '-7/2':[1, 0,0,0,0]
})
with_gaunt=True
with_breit=True
with_aoc=False
amf_type='x2camf'

mf.with_x2c = eamf.SpinorEAMFX2CHelper(mol, eamf=amf_type, with_gaunt=with_gaunt, with_breit=with_breit, with_aoc=with_aoc)
mf.diis_start_cycle=20
mf.damp=0.4
mf.kernel()
