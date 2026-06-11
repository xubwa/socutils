#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# TDDFT / TDA for a two-component (j-adapted spinor) reference.
#
# A spinor mean field is, structurally, a GHF/GKS reference with a single
# complex n2c x n2c MO coefficient matrix.  The linear-response sigma vector
# (pyscf.tdscf.ghf) is basis agnostic -- it only needs mf.gen_response, which
# returns (J - c_x K + f_xc) acting on the (complex) transition densities.  The
# spinor reference supplies that through the GHF response template
# (socutils.scf.spinor_hf binds it as gen_response), with J/K from the spinor
# get_jk and the xc kernel from socutils.dft.numint_2c.SpinorNumInt2C (which
# also makes the collinear f_xc work with complex transition densities).
#
# Validated vs pyscf's 2-component GKS-TDA/TDDFT (non-relativistic integrals,
# no SOC) to machine precision.  With SOC integrals the orbitals are genuinely
# complex; set mf._numint.collinear='mcol' for the spin-flip (non-collinear)
# response.
#

from pyscf.tdscf import ghf as _ghf


class TDA(_ghf.TDA):
    pass


# The Casida/RPA solver in tdscf.ghf is named TDHF; driven through
# mf.gen_response it gives full TDDFT for a KS reference (the response carries
# f_xc) or TDHF for a bare spinor HF reference.
class TDDFT(_ghf.TDHF):
    pass


TDHF = RPA = TDDFT


def TDDFT_factory(mf):
    return TDDFT(mf)
