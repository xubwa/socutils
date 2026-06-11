#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# TDA (Tamm-Dancoff) for a two-component (j-adapted spinor) reference.
#
# A spinor mean field is, structurally, a GHF/GKS reference with a single
# complex n2c x n2c MO coefficient matrix.  The linear-response sigma vector
# (pyscf.tdscf.ghf) is basis agnostic -- it only needs mf.gen_response, which
# returns (J - c_x K + f_xc) acting on the (complex) transition densities.  The
# spinor reference supplies that through the GHF response template (bound as
# SpinorSCF.gen_response): J/K from the spinor get_jk, f_xc from the
# 2-component SpinorNumInt2C (which also makes the collinear f_xc work with
# complex transition densities).
#
# Only TDA is exposed here.  TDA sets B=0, so its response matrix A is genuinely
# Hermitian and the Davidson solve (lr_eig.eigh) is robust.  The full RPA/TDDFT
# (Casida) problem is k-Hermitian (non-Hermitian) on a Krein space with an
# indefinite metric [Furche & Chen, J. Chem. Phys. 163, 174104 (2025)]; its
# iterative solution is delicate near eta-neutral roots and is left aside for
# now.
#
# Validated vs pyscf's 2-component GKS-TDA (non-relativistic integrals, no SOC)
# to machine precision (LDA / GGA / hybrid).  With SOC integrals the orbitals
# are genuinely complex; set mf._numint.collinear='mcol' for the spin-flip
# (non-collinear) response.
#

from pyscf import lib
from pyscf.tdscf import ghf as _ghf


class _KernelXCMixin:
    '''Allow the response (A) xc kernel to differ from the ground-state xc.

    Set ``xc_kernel`` to e.g. ``'LDA,VWN'`` to build the TDA response with an
    (A)LDA kernel on top of a GGA/hybrid ground state.  The whole response xc --
    the local f_xc *and* the exact-exchange fraction -- then comes from
    ``xc_kernel`` (so an 'LDA,VWN' kernel carries no HF exchange; keep the exact
    exchange with e.g. ``'0.2*HF + 0.8*LDA, VWN'``), evaluated on the
    ground-state density.  Leave it None to use the ground-state functional.
    '''
    xc_kernel = None

    def kernel(self, *args, **kwargs):
        # _gen_ghf_response returns a vind closure that reads mf.xc lazily (at
        # Davidson time), so the kernel xc must stay set for the whole solve,
        # not just while the response is generated.
        if self.xc_kernel is None:
            return super().kernel(*args, **kwargs)
        with lib.temporary_env(self._scf, xc=self.xc_kernel):
            return super().kernel(*args, **kwargs)


class TDA(_KernelXCMixin, _ghf.TDA):
    pass


CIS = TDA
