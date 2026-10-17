#!/usr/bin/env python
'''
AVAS automated active space selection on a two-component (spinor) reference.

socutils.mcscf.avas is the spinor analogue of pyscf.mcscf.avas: reference AOs
are selected by matching spinor labels (a scalar-style pattern like 'F 2p'
matches both 2p1/2 and 2p3/2 spinors), the projector is complex Hermitian,
and all sizes are counted in spinor orbitals (one electron per orbital).
In the nonrelativistic limit it reproduces the scalar AVAS: every projector
eigenvalue appears twice (Kramers pair), ncas doubles, nelecas is unchanged.
'''
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import avas, zcasci

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

mf = spinor_hf.SCF(mol).x2camf()
mf.kernel()

# valence active space from F 2p + H 1s reference AOs
ncas, nelecas, mo = avas.avas(mf, ['F 2p', 'H 1s'])
print('AVAS active space: %d electrons in %d spinor orbitals' % (nelecas, ncas))

mc = zcasci.CASCI(mf, ncas, nelecas)
mc.kernel(mo)
print('E(CASCI) = %.10f' % mc.e_tot)

# use IAOs instead of raw minao AOs as the reference (with_iao=True), and
# tighten/loosen the selection with threshold
ncas, nelecas, mo = avas.avas(mf, ['F 2p', 'H 1s'], with_iao=True, threshold=0.2)
print('AVAS(with_iao) active space: %d electrons in %d spinor orbitals'
      % (nelecas, ncas))

# Atomic x2camf reference instead of the tabulated 'minao' basis: occupied
# spinors of spherically averaged atomic SCF solutions (aufbau/AOC
# occupations from the x2camf tabulation).  Recommended for elements beyond
# Kr, where 'minao' is truncated from cc-pVTZ-PP (no core functions, no
# relativistic contraction); the atomic reference has j-split relativistic
# radial functions and full core coverage for every element.
from socutils.lo import atomic_ref
c_ref, labels = atomic_ref.atomic_minao(mf)
ncas, nelecas, mo = avas.avas(mf, ['F 2p', 'H 1s'], minao=(c_ref, labels))
print('AVAS(atomic ref) active space: %d electrons in %d spinor orbitals'
      % (nelecas, ncas))
