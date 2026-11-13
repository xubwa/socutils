#
# Author: Xubo Wang <wangxubo0201@gmail.com>
#

'''
The fast spinor CCSD and CCSD(T), moved from zccsdt (zccsdt.zccsd).

- ZCCSD (zccsd.py): pp ladder in the MO basis on <ab||ef> packed over
  (a>b, e>f).
- DirectZCCSD (zccsd_direct.py): pp ladder in the scalar AO basis (no vvvv),
  the amplitude equations on packed amplitudes within max_memory (amps.py),
  OpenMP kernels (lib/zccsd, kernels.py), Kramers pairs of a Kramers-
  restricted reference (kramers.py), integrals slab by slab (ao2mo.py,
  nrr_fast.py), (T) in batches of virtual pairs (triples.py).

DIIS is pyscf.lib.diis.DIIS.  Not moved yet (use zccsdt.zccsd): Lambda,
densities and dipole moment, orbital relaxation, CC2.

    >>> from socutils.cc.fast import DirectZCCSD
    >>> mycc = DirectZCCSD(mf).run()     # mf: a spinor SCF (socutils, pyscf x2c)
    >>> et = mycc.ccsd_t()
'''

from socutils.cc.fast.zccsd import ZCCSD
from socutils.cc.fast.zccsd_direct import DirectZCCSD
