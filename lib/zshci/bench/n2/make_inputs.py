'''
Non-relativistic benchmark: N2 at 1.5 Angstrom, cc-pVDZ, frozen 1s, 10
electrons in 26 orbitals, D2h. Writes, into the directory given as the first
argument, FCIDUMP (spatial orbitals, ORBSYM in Molpro's numbering), input.json
for zshci and input.dat for Dice, with the same epsilon1 schedule (the further
arguments; default 1e-3 2e-4 5e-5), Davidson threshold and deterministic PT.

    python make_inputs.py DIR [epsilon1 ...]
'''

import json
import os
import sys

from pyscf import gto, mcscf, scf
from pyscf.symm.param import IRREP_ID_MOLPRO
from pyscf.tools import fcidump

directory = os.path.abspath(sys.argv[1])
epsilons = [float(x) for x in sys.argv[2:]] or [1e-3, 2e-4, 5e-5]
os.makedirs(directory, exist_ok=True)

mol = gto.M(atom='N 0 0 0; N 0 0 1.5', basis='cc-pvdz', symmetry='d2h',
            verbose=0)
mf = scf.RHF(mol).run()
norb, nelec = mol.nao - 2, 10
mc = mcscf.CASCI(mf, norb, nelec)
h1, ecore = mc.get_h1eff()
orbsym = [IRREP_ID_MOLPRO['D2h'][i]
          for i in mf.mo_coeff.orbsym[mc.ncore:mc.ncore + norb]]
integrals = os.path.join(directory, 'FCIDUMP')
fcidump.from_integrals(integrals, h1, mc.get_h2eff(), norb, nelec, nuc=ecore,
                       ms=0, orbsym=orbsym, tol=1e-14)

schedule = [[2 * k, e] for k, e in enumerate(epsilons)]
max_iter = 2 * len(epsilons) + 4
with open(os.path.join(directory, 'input.json'), 'w') as f:
    json.dump({
        'mode': 'select', 'orbitals': 'spatial', 'integrals': integrals,
        'prefix': '.', 'nroots': 1, 'dets': [list(range(nelec))],
        'select': {'schedule': schedule, 'max_iter': max_iter,
                   'energy_tol': 1e-6, 'davidson_tol_loose': 5e-5},
        'davidson': {'tol': 5e-5},
        'pt': {'method': 'deterministic', 'epsilon2': 1e-6},
        'hij_threshold': 0.0, 'output_level': 0}, f, indent=1)
with open(os.path.join(directory, 'input.dat'), 'w') as f:
    f.write('nocc %d\n%s\nend\n' % (nelec, ' '.join(map(str, range(nelec)))))
    f.write('orbitals %s\nnroots 1\npointGroup d2h\n' % integrals)
    f.write('schedule\n%send\n' % ''.join('%d %g\n' % (i, e)
                                          for i, e in schedule))
    f.write('davidsonTol 5e-5\ndE 1e-6\nmaxiter %d\ndeterministic\n'
            'epsilon2 1e-6\nnoio\n' % max_iter)
print('%d orbitals, E(RHF) = %.10f; inputs in %s' % (norb, mf.e_tot, directory))
