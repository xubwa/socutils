#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Spinor (j-adapted) <-> GHF (scalar AO x spin) representations of the same
two-component calculation, for the nuclear derivatives.

The j-adapted spinor AOs are a fixed unitary combination U = sph2spinor_coeff
of the scalar AOs times spin, so a spinor SCF solution is a GHF solution with
MO coefficients U C (and the same orbital energies and occupations), and TDA
amplitudes in the MO basis are the same in both representations.  The nuclear
gradient, the Hessian and the TDA excited-state gradient of a spinor
calculation are therefore those of the equivalent GHF / GKS object
(socutils.grad.ghf_grad / gks_grad, socutils.hessian, socutils.grad.tdghf_grad):

    mf = spinor_hf.SCF(mol).x2camf().run()
    g = mf.Gradients().kernel()           # GHF gradient code
    h = mf.Hessian().kernel()
    td = mf.TDA().run()
    ge = td.Gradients().kernel(state=1)
'''

import numpy
from pyscf.lib import logger
from pyscf.scf import hf


def _spin_orbital_x2c(mol, x2c_spinor):
    '''The spin-orbital X2C helper with the settings of a spinor one.'''
    from socutils.somf.x2cmp import SpinOrbitalX2CMPHelper
    return SpinOrbitalX2CMPHelper(mol, x2cmp=x2c_spinor.amf_type,
                                  with_gaunt=x2c_spinor.gaunt, with_breit=x2c_spinor.breit,
                                  with_pcc=x2c_spinor.pcc, with_aoc=x2c_spinor.aoc)


def spinor2ghf_coeff(mol, c):
    '''GHF MO coefficients of spinor MO coefficients c.'''
    so_contr = getattr(mol, 'so_contr', None)
    if so_contr is not None:          # symmetry-adapted spinor basis
        c = so_contr.dot(c)
    return numpy.vstack(mol.sph2spinor_coeff()).dot(c)


def to_ghf(mf):
    '''Equivalent socutils GHF / GKS object of a converged spinor SCF / KS.'''
    from socutils.scf import ghf
    from socutils.dft import gks
    mol = mf.mol
    if getattr(mf, 'with_df', None) is not None:
        raise NotImplementedError('nuclear derivatives of density-fitted spinor SCF')
    if isinstance(mf, hf.KohnShamDFT):
        g = gks.GKS(mol, xc=mf.xc)
        g.grids = mf.grids
        g.nlc = getattr(mf, 'nlc', '')
        ni, ni_s = g._numint, mf._numint
        for key in ('collinear', 'spin_samples', 'collinear_thrd', 'collinear_samples',
                    'mcol_backend', 'mcol_kernel', 'ncol_s_thrd', 'ncol_fxc_thrd'):
            if hasattr(ni_s, key):
                setattr(ni, key, getattr(ni_s, key))
    else:
        g = ghf.GHF(mol)
    if getattr(mf, 'with_x2c', None) is not None:
        g.with_x2c = _spin_orbital_x2c(mol, mf.with_x2c)
    for key in ('conv_tol', 'conv_tol_grad', 'conv_tol_cpscf', 'max_memory', 'verbose', 'stdout'):
        if hasattr(mf, key):
            setattr(g, key, getattr(mf, key))
    g.mo_coeff = spinor2ghf_coeff(mol, mf.mo_coeff)
    g.mo_energy = mf.mo_energy
    g.mo_occ = mf.mo_occ
    g.e_tot = mf.e_tot
    g.converged = mf.converged
    return g


def to_ghf_tda(td):
    '''TDA object on the equivalent GHF / GKS reference, with the amplitudes,
    excitation energies and response settings of a spinor TDA.'''
    from socutils.tdscf import gks as td_gks
    g = to_ghf(td._scf)
    tdg = td_gks.TDA(g)
    for key in ('nstates', 'conv_tol', 'xc_kernel', 'collinear', 'frozen', 'verbose'):
        if hasattr(td, key):
            setattr(tdg, key, getattr(td, key))
    tdg.e = td.e
    tdg.xy = td.xy
    tdg.converged = td.converged
    return tdg


def check_energy(mf, g=None):
    '''|E(GHF representation) - E(spinor)| for a converged spinor mf.'''
    if g is None:
        g = to_ghf(mf)
    e = g.energy_tot(g.make_rdm1())
    logger.info(mf, 'spinor -> GHF: E = %.12f (spinor %.12f)', e, mf.e_tot)
    return abs(e - mf.e_tot)


# methods of the spinor classes
def _gradients(self):
    return to_ghf(self).Gradients()


def _hessian(self):
    import socutils.hessian.ghf  # noqa: F401
    import socutils.hessian.gks  # noqa: F401
    return to_ghf(self).Hessian()


def _td_gradients(self):
    from socutils.grad import tdghf_grad
    return tdghf_grad.Gradients(to_ghf_tda(self))
