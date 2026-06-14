#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
"""Backwards-compatibility shim.

The density-fitted spinor SCF used to live here as a parallel copy of the
whole ``spinor_hf`` module.  Its more efficient density-fitting ``get_jk``
(the block-wise MO half-transform of the DF 3-index integrals) has been merged
into :mod:`socutils.scf.spinor_hf`, which is now the single implementation and
whose ``.density_fit()`` uses it.

This module re-exports everything from :mod:`socutils.scf.spinor_hf` so that
existing ``from socutils.scf import spinor_hf_df`` imports keep working.
"""

from socutils.scf.spinor_hf import *  # noqa: F401,F403
from socutils.scf.spinor_hf import (  # noqa: F401  (also expose non-public names)
    density_fit,
    _DFJHF,
    spinor2sph,
    sph2spinor,
    get_hcore,
    SpinorSCF,
    SymmSpinorSCF,
    KRHF,
    SCF,
)
