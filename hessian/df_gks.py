#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Density-fitted (RI-J) two-electron parts of the GKS Hessian of a Kramers
closed shell, for functionals without exact exchange.

The Coulomb energy of a GHF density depends only on the total density
rho = Re(D_aa + D_bb) (real symmetric for a Hermitian D), so its derivatives
are those of a restricted closed shell with that density.  They are taken
from pyscf.df.hessian.rhf (full response of the auxiliary basis,
auxbasis_response = 2) through a restricted proxy whose occupied orbitals are
the natural orbitals of rho scaled by sqrt(n / 2); the exchange-correlation
terms, the hcore (including X2C) and the coupled-perturbed equations (with
the density-fitted response of the GKS object) are those of hessian.gks.
'''

import numpy
from pyscf import scf
from pyscf.df import df_jk


def is_density_fitted(mf):
    return isinstance(mf, df_jk._DFHF) and getattr(mf, 'with_df', None) is not None


def _rhf_proxy(hessobj, dm0):
    '''pyscf.df.hessian.rhf.Hessian of a restricted proxy carrying the total
    density of dm0; returns (hessian object, mo_coeff, mo_occ, mo_energy).'''
    from pyscf.df.hessian import rhf as df_rhf_hess
    mf = hessobj.base
    mol = hessobj.mol
    nao = mol.nao_nr()
    rho = (dm0[:nao, :nao] + dm0[nao:, nao:]).real
    n, U = numpy.linalg.eigh(.5 * (rho + rho.T))
    keep = n > 1e-12
    mo = U[:, keep] * numpy.sqrt(n[keep] / 2)
    occ = numpy.full(mo.shape[1], 2.)
    proxy = scf.RHF(mol).density_fit(auxbasis=mf.with_df.auxbasis)
    proxy.with_df = mf.with_df
    proxy.mo_coeff = mo
    proxy.mo_occ = occ
    proxy.mo_energy = numpy.zeros(mo.shape[1])
    h = df_rhf_hess.Hessian(proxy)
    h.max_memory = hessobj.max_memory
    h.verbose = hessobj.verbose
    return h, mo, occ, proxy.mo_energy


def partial_hess_j(hessobj, dm0, atmlst):
    '''Second derivatives of E_J = 1/2 (rho|rho) (density fitted) at fixed
    dm0, (natm, natm, 3, 3).'''
    from pyscf.df.hessian import rhf as df_rhf_hess
    h, mo, occ, e = _rhf_proxy(hessobj, dm0)
    ej = df_rhf_hess._partial_hess_ejk(h, e, mo, occ, atmlst, hessobj.max_memory,
                                       hessobj.verbose, with_k=False)[1]
    return ej


def veff_deriv1_j(hessobj, dm0, atmlst):
    '''d J[dm0] / dR at fixed dm0 (density fitted), {atom: (3, 2n, 2n)}.'''
    from pyscf.df.hessian import rhf as df_rhf_hess
    h, mo, occ, e = _rhf_proxy(hessobj, dm0)
    nao = hessobj.mol.nao_nr()
    out = {}
    for ia, h1, vj1, vk1 in df_rhf_hess._gen_jk(h, mo, occ, None, atmlst, hessobj.verbose,
                                                with_k=False):
        v = numpy.zeros((3, 2 * nao, 2 * nao), dtype=numpy.complex128)
        v[:, :nao, :nao] = vj1
        v[:, nao:, nao:] = vj1
        out[ia] = v
    return out
