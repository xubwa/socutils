#!/usr/bin/env python
'''
Stage-2 external anchor: write ORCA and Molpro input DRAFTS for the
MP2-F12/3C(FIX) cases of rf12/test/data/psi4_aug-cc-pvdz.json
(He ae; Ne ae/fc; H2O ae/fc; aug-cc-pVDZ / aug-cc-pVDZ-OptRI; gamma = 1).

    python make_inputs.py [outdir]

Every keyword marked VERIFY was written from memory, NOT checked against
the program manuals or the installed versions.  Check each one before
trusting a number, in particular:
  * conventional (non-DF/non-RI) F12 integrals vs the program default (DF/RI)
  * ansatz 3C(FIX) and the SP fixed amplitudes 1/2, 1/4
  * the geminal: exponent 1.0, and whether it is the exact Slater function
    or a Gaussian fit (Psi4: 6-Gaussian fit; socutils: exact Slater)
  * CABS basis = aug-cc-pVDZ-OptRI, and how the CABS is orthogonalised
  * CABS singles reported separately; frozen core on/off
'''
import os
import sys

GEOM_BOHR = {
    'He': [('He', 0., 0., 0.)],
    'Ne': [('Ne', 0., 0., 0.)],
    'H2O': [('O', 0., 0., -0.124038860300),
            ('H', 0., -1.431430901356, 0.984293362719),
            ('H', 0., 1.431430901356, 0.984293362719)],
}
CASES = [('He', 'ae'), ('Ne', 'ae'), ('Ne', 'fc'), ('H2O', 'ae'), ('H2O', 'fc')]


def molpro(name, fc):
    atoms = GEOM_BOHR[name]
    core = '' if fc == 'fc' else 'core,0'           # VERIFY: Molpro default is frozen core
    return '''***,{name} {fc} MP2-F12/3C(FIX) stage-2 anchor
memory,1000,m
gthresh,energy=1.d-12,orbital=1.d-10
! VERIFY: Molpro reads XYZ geometries in Angstrom; the coordinates below are in bohr (add the bohr option or convert)
geomtyp=xyz
geometry={{
{nat}
{name}
{geom_xyz}
}}
basis={{default=aug-cc-pVDZ
       ri=aug-cc-pVDZ/OPTRI}}                  ! VERIFY: name of the OptRI set in the Molpro library
explicit,ri_basis=ri,gem_beta=1.0             ! VERIFY: gem_beta = Slater exponent; check the Gaussian-fit options
hf
{{mp2-f12,ansatz=3C(FIX),cabs_singles=1{core_opt}}}  ! VERIFY: conventional (non-DF) MP2-F12 available?
! fallback if only DF is available (differences of order uEh-0.1 mEh expected):
! {{df-mp2-f12,ansatz=3C(FIX),cabs_singles=1}}
'''.format(name=name, fc=fc, nat=len(atoms),
           geom_xyz='\n'.join('%s %.12f %.12f %.12f' % a for a in atoms),
           core_opt=(';' + core) if core else '')


def orca(name, fc):
    atoms = GEOM_BOHR[name]
    geom = '\n'.join('  %-2s %18.12f %18.12f %18.12f' % a for a in atoms)
    frozen = 'FrozenCore' if fc == 'fc' else 'NoFrozenCore'
    return '''# {name} {fc} MP2-F12/3C(FIX) stage-2 anchor (DRAFT)
# VERIFY: '! MP2-F12' should be the conventional (non-RI) F12 code; '! RI-MP2-F12' is the RI one
# VERIFY: which ansatz/approximation ORCA's MP2-F12 uses (needs 3C(FIX), SP amplitudes)
# VERIFY: name of the CABS basis (aug-cc-pVDZ-OptRI) in ORCA's library; else read it via %basis GTOName
! MP2-F12 aug-cc-pVDZ VeryTightSCF {frozen}
%basis
  CABS "aug-cc-pVDZ-OptRI"      # VERIFY keyword and basis name
end
%mp2
  F12 true                      # VERIFY
  # gamma (geminal exponent) = 1.0  VERIFY: keyword name, default value, Slater vs Gaussian fit
end
* xyz 0 1
{geom}
*
# VERIFY: ORCA reads Angstrom by default -> add '! Bohrs' or convert
'''.format(name=name, fc=fc, frozen=frozen, geom=geom)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'inputs'
    os.makedirs(out, exist_ok=True)
    for name, fc in CASES:
        open(os.path.join(out, '%s_%s.molpro.inp' % (name, fc)), 'w').write(molpro(name, fc))
        open(os.path.join(out, '%s_%s.orca.inp' % (name, fc)), 'w').write(orca(name, fc))
    print('wrote %d inputs to %s' % (2 * len(CASES), out))
