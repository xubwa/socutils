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


class _PairedRPA(_ghf.TDHF):
    '''Full RPA/TDDFT solved with the paired-trial-vector Davidson
    (socutils.tdscf.lr_davidson), which represents the indefinite Krein
    metric correctly for complex orbitals.  Experimental -- TDA remains the
    recommended default.'''

    conv_tol = 1e-6

    def kernel(self, x0=None, nstates=None):
        import numpy as np
        from pyscf.lib import logger
        from socutils.tdscf import lr_davidson
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        conv, e, zs, nmv = lr_davidson.paired_eig(
            vind, hdiag, nroots=nstates, x0=x0, conv_tol=self.conv_tol,
            max_cycle=self.max_cycle,
            pos_tol=getattr(self, 'positive_eig_threshold', 1e-3),
            verbose=self.verbose > 4)
        log.debug('TDRPA paired Davidson: %d matvecs', nmv)
        self.converged = conv
        self.e = e

        mask = self.get_frozen_mask()
        mo_occ = self._scf.mo_occ[mask]
        nocc = int(np.count_nonzero(mo_occ > 0))
        nvir = mo_occ.size - nocc
        # zs are normalized to |X|^2 - |Y|^2 = 1 already
        self.xy = [(z[:nocc*nvir].reshape(nocc, nvir),
                    z[nocc*nvir:].reshape(nocc, nvir)) for z in zs]
        log.timer('TDRPA (paired Davidson)', *cpu0)
        self._finalize()
        return self.e, self.xy


class TDRPA(_KernelXCMixin, _PairedRPA):
    pass
