#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''Screened-nuclear spin-orbit (SNSO) X2C-1e Hamiltonian.

The one-electron X2C Hamiltonian overestimates spin-orbit splittings because
it carries the bare nuclear spin-orbit operator and none of the two-electron
(SSO/SOO) screening.  Boettger's recipe [1] damps the spin-orbit part of
``W = (sigma.p) V (sigma.p)`` between two basis functions by a factor that
depends only on their angular momenta and the nuclear charge::

    W_so[i,j] *= 1 - sqrt(Q(l_i) Q(l_j)) / Z

with ``Q(l) = 0, 2, 10, 28, 60, 110`` for l = s..h.  Row 1 of
:data:`BOETTGER_Q` is the refit of Filatov, Zou and Cremer [2]; the remaining
rows are further refits carried over from the original implementation, and
``scaled=3`` picks one of rows 3-9 per atom by its Z range.

This module is pyscf's ``SpinorX2CHelper`` / ``SpinOrbitalX2CHelper`` with
the scaling inserted between the ``W`` integral and the Foldy-Wouthuysen
transform;
everything else (X matrix, decontraction, picture change) is pyscf's.  Hang
it on a spinor (or, with ``SpinOrbitalX2CHelper``, a GHF) mean field exactly
like any other X2C helper::

    mf = spinor_hf.SpinorSCF(mol)
    mf.with_x2c = snso.X2C(mol)
    mf.with_x2c.boettger = True
    mf.with_x2c.scaled = 0

[1] J. C. Boettger, Phys. Rev. B 62, 7809 (2000).
[2] M. Filatov, W. Zou, D. Cremer, J. Chem. Theory Comput. 9, 3092 (2013).
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.x2c import x2c as _x2c
from pyscf.x2c.x2c import _get_hcore_fw, _x2c1e_xmatrix, _x2c1e_get_hcore

__all__ = ['SpinorX2CHelper', 'SpinOrbitalX2CHelper', 'X2C', 'boettger_damp',
           'boettger_scale', 'get_hcore']

# Q(l) for l = s, p, d, f, g, h, one row per ``scaled`` value.
BOETTGER_Q = numpy.array([
    [0.0, 2.  , 10.  , 28.  , 60., 110.],   # 0: Boettger (2000)
    [0.0, 2.32, 10.64, 28.38, 60., 110.],   # 1: mSNSO, Filatov/Zou/Cremer (2013)
    [0.0, 2.97, 11.93, 29.84, 64., 115.],   # 2
    [0.0, 2.97, 11.93, 29.84, 64., 115.],   # 3: Z <  2   (scaled=3 dispatches by Z)
    [0.0, 2.80, 11.93, 29.84, 64., 115.],   # 4: Z < 10
    [0.0, 2.95, 11.93, 29.84, 64., 115.],   # 5: Z < 18
    [0.0, 3.09, 11.49, 29.84, 64., 115.],   # 6: Z < 36
    [0.0, 3.02, 11.91, 29.84, 64., 115.],   # 7: Z < 54
    [0.0, 2.85, 12.31, 30.61, 64., 115.],   # 8: Z < 86
    [0.0, 2.85, 12.31, 30.61, 64., 115.],   # 9: Z >= 86
])
_Z_EDGES = (2, 10, 18, 36, 54, 86)
_L_OF = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5}


def _q_row(scaled, z):
    if scaled == 3:
        return 3 + numpy.searchsorted(_Z_EDGES, z, side='right')
    return scaled


def boettger_damp(xmol, scaled=0, spinor=True):
    '''The ``1 - sqrt(Q(l_i) Q(l_j)) / Z`` factor for every pair of basis
    functions of ``xmol`` -- in its j-adapted spinor basis, or in the scalar
    AO basis (``spinor=False``) for the spin-orbital (GHF) picture.'''
    if spinor:
        labels, slices, n = xmol.spinor_labels(0), xmol.offset_2c_by_atom(), xmol.nao_2c()
    else:
        labels, slices, n = xmol.ao_labels(None), xmol.offset_nr_by_atom(), xmol.nao_nr()
    q_over_z = numpy.zeros(n)
    for ia, (ish0, ish1, p0, p1) in enumerate(slices):
        z = xmol.atom_charge(ia)
        q = BOETTGER_Q[_q_row(scaled, z)]
        for i in range(p0, p1):
            l = _L_OF[labels[i][2].upper().lstrip('0123456789')[0]]
            q_over_z[i] = min(z, q[l]) / z
    return 1. - numpy.minimum(numpy.sqrt(numpy.outer(q_over_z, q_over_z)), 1.)


def boettger_scale(xmol, w, scaled=0):
    '''Damp the spin-orbit part of the spinor-basis ``W`` by Boettger's
    screening factors; the spin-free part is left alone.'''
    wsf = xmol.intor_symmetric('int1e_pnucp_spinor')
    return wsf + (w - wsf) * boettger_damp(xmol, scaled, spinor=True)


class _SNSO:
    '''The two knobs shared by both pictures.

    Attributes:
        boettger: apply the screening (default False -- plain X2C-1e).
        scaled: which row of :data:`BOETTGER_Q` to use; 3 picks a row per
            atom by its nuclear charge.
    '''
    boettger = False
    scaled = 0
    _keys = {'boettger', 'scaled'}

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('SNSO (Boettger) screening = %s, scaled = %d', self.boettger, self.scaled)
        return self


class SpinorX2CHelper(_SNSO, _x2c.SpinorX2CHelper):
    '''X2C-1e in the j-adapted spinor basis, with optional SNSO screening of
    the nuclear spin-orbit integrals.'''

    def get_hcore(self, mol=None):
        '''2-component X2C FW Hamiltonian; ``pyscf.x2c.x2c.X2CHelperBase.get_hcore``
        with ``W`` screened before the transform when ``boettger`` is set.'''
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff_nr = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        if self.boettger:
            w = boettger_scale(xmol, w, self.scaled)

        if 'get_xmat' in self.__dict__:
            x = self.get_xmat(xmol)
            h1 = _get_hcore_fw(t, v, w, s, x, c)
        elif 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_2c_by_atom()
            n2c = xmol.nao_2c()
            x = numpy.zeros((n2c,n2c), dtype=numpy.complex128)
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
                t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                    w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
            h1 = _get_hcore_fw(t, v, w, s, x, c)
        else:
            h1 = _x2c1e_get_hcore(t, v, w, s, c)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp_spinor')
            s21 = mole.intor_cross('int1e_ovlp_spinor', xmol, mol)
            c = lib.cho_solve(s22, s21)
            h1 = reduce(numpy.dot, (c.T.conj(), h1, c))
        elif self.xuncontract:
            np, nc = contr_coeff_nr.shape
            if np == h1.shape[0]:
                contr_coeff = contr_coeff_nr
            else:
                contr_coeff = numpy.zeros((np*2,nc*2))
                contr_coeff[0::2,0::2] = contr_coeff_nr
                contr_coeff[1::2,1::2] = contr_coeff_nr
            h1 = reduce(numpy.dot, (contr_coeff.T.conj(), h1, contr_coeff))
        return h1


class SpinOrbitalX2CHelper(_SNSO, _x2c.SpinOrbitalX2CHelper):
    '''The same Hamiltonian in the spin-orbital (GHF) basis: the spinor
    hcore rotated with ``spinor2sph``.  The spin-orbital picture is a view of
    the spinor one, not a second implementation.'''

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        x = SpinorX2CHelper(mol)
        x.__dict__.update(self.__dict__)
        from socutils.tools.spinor2sph import spinor2sph
        return spinor2sph(mol, x.get_hcore(mol))


X2C = SpinorX2CHelper


def get_hcore(mol, scaled=0):
    '''SNSO-screened X2C-1e hcore in the j-adapted spinor basis.'''
    x = SpinorX2CHelper(mol)
    x.boettger = True
    x.scaled = scaled
    return x.get_hcore(mol)
