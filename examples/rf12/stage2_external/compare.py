#!/usr/bin/env python
'''
Compare external MP2-F12/3C(FIX) numbers with Psi4 1.11 (6-Gaussian geminal,
rf12/test/data/psi4_aug-cc-pvdz.json) and with socutils (exact Slater geminal).

    python compare.py results.json [--ours]

results.json (fill in by hand from the program outputs; Eh):
  {"orca":   {"He_ae": {"hf": ..., "mp2": ..., "f12": ..., "singles": ...}, ...},
   "molpro": {...}}
  f12 = F12 correction to the MP2 correlation energy WITHOUT CABS singles.
--ours: also run socutils.rf12.nr.mp2f12 (exact STG) for each case (minutes).
'''
import os
import sys
import json
import socutils.rf12

ref = json.load(open(os.path.join(os.path.dirname(socutils.rf12.__file__), 'test', 'data', 'psi4_aug-cc-pvdz.json')))
KEYS = {'hf': 'SCF TOTAL ENERGY', 'mp2': 'MP2 CORRELATION ENERGY', 'f12': 'MP2-F12 CORRECTION ENERGY',
        'singles': 'F12 CABS CORRECTION ENERGY'}
GEOM = {'He': 'He 0 0 0', 'Ne': 'Ne 0 0 0',
        'H2O': 'O 0 0 -0.124038860300; H 0 -1.431430901356 0.984293362719; H 0 1.431430901356 0.984293362719'}


def ours(case):
    from pyscf import gto, scf
    from socutils.rf12.nr import mp2f12, gbs
    name, fc = case.split('_')
    els = ['O', 'H'] if name == 'H2O' else [name]
    mol = gto.M(atom=GEOM[name], unit='bohr', basis=gbs.basis('aug-cc-pvdz', els), verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
    m = mp2f12.MP2F12(mf, gbs.basis('aug-cc-pvdz-optri', els), frozen=1 if fc == 'fc' else 0)
    m.kernel()
    return dict(hf=mf.e_tot, mp2=m.e_mp2, f12=m.e_f12, singles=m.e_singles)


res = json.load(open(sys.argv[1]))
do_ours = '--ours' in sys.argv
for prog, cases in res.items():
    for case, vals in cases.items():
        o = ours(case) if do_ours else {}
        for k, v in vals.items():
            line = '%-7s %-7s %-8s %18.10f  -Psi4(6G) %+10.3f uEh' % (prog, case, k, v, (v - ref[case][KEYS[k]]) * 1e6)
            if do_ours:
                line += '  -socutils(STG) %+10.3f uEh' % ((v - o[k]) * 1e6)
            print(line)
