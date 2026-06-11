#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Transition properties (transition density matrix, transition dipole moment,
oscillator strength) for spinor CASCI/CASSCF calculations.

The functions work with any fcisolver providing a
trans_rdm1(ci_bra, ci_ket, ncas, nelecas) method, e.g. zfci.FCISolver and
zfci.SelectedCI.  The Dice interface (socutils.hci.shci) is also supported,
since there mc.ci holds the root indices that its trans_rdm1 expects.
'''

from functools import reduce
import numpy
from pyscf import lib


def charge_center(mol):
    charges = mol.atom_charges()
    return numpy.einsum('z,zx->x', charges, mol.atom_coords()) / charges.sum()

def _state_ci(ci, state):
    if isinstance(ci, (list, tuple)):
        return ci[state]
    if state != 0:
        raise ValueError(f'state {state} requested but only one CI vector '
                         'is available; set fcisolver.nroots')
    return ci

def trans_rdm1_ao(mc, state_i=0, state_j=1, mo_coeff=None, ci=None):
    '''Transition density matrix in the AO spinor basis, defined such that
    the matrix element of a one-electron operator O is

        <i|O|j> = einsum('uv,vu->', O_ao, dm)

    with O_ao[u,v] = <chi_u|O|chi_v>.  This requires dm = C t^T C^dagger
    where t[p,q] = <i|p^+ q|j>; with complex spinor orbitals the transpose
    matters (C t C^dagger contracts the operator with t transposed and is
    wrong for transition properties).

    For state_i == state_j the core contribution is included; for different
    states it vanishes because the CI vectors are orthogonal.
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    ncore = mc.ncore
    ncas = mc.ncas
    ci_i = _state_ci(ci, state_i)
    ci_j = _state_ci(ci, state_j)
    t_dm1 = mc.fcisolver.trans_rdm1(ci_i, ci_j, ncas, mc.nelecas)
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    dm = reduce(numpy.dot, (mo_cas, t_dm1.T, mo_cas.conj().T))
    if state_i == state_j:
        mo_core = mo_coeff[:, :ncore]
        dm = dm + numpy.dot(mo_core, mo_core.conj().T)
    return dm

def _dipole_integrals(mol, nao, origin):
    '''AO dipole integrals (3, nao, nao) in the 2c spinor basis, or in the
    4c basis (following pyscf.scf.dhf.dip_moment) when nao = 2*nao_2c.'''
    n2c = mol.nao_2c()
    with mol.with_common_orig(origin):
        ll_dip = mol.intor_symmetric('int1e_r_spinor', comp=3)
        if nao == n2c:
            return ll_dip
        elif nao == n2c * 2:
            ss_dip = mol.intor_symmetric('int1e_sprsp_spinor', comp=3)
            c = lib.param.LIGHT_SPEED
            ao_dip = numpy.zeros((3, nao, nao), dtype=numpy.complex128)
            ao_dip[:, :n2c, :n2c] = ll_dip
            ao_dip[:, n2c:, n2c:] = ss_dip * (.5/c)**2
            return ao_dip
        else:
            raise ValueError(f'AO dimension {nao} matches neither the 2c '
                             f'({n2c}) nor the 4c ({2*n2c}) spinor basis')

def transition_dipole(mc, state_i=0, state_j=1, origin=None):
    '''Electronic (transition) dipole moment <i|-r|j> in a.u. (length gauge).

    For state_i != state_j the result is independent of the gauge origin;
    for state_i == state_j it is the electronic dipole of that state with
    respect to origin (the nuclear charge center by default), so that
    adding the nuclear contribution gives the total dipole moment.
    '''
    mol = mc.mol
    if origin is None:
        origin = charge_center(mol)
    t_dm1 = trans_rdm1_ao(mc, state_i, state_j)
    ao_dip = _dipole_integrals(mol, t_dm1.shape[0], origin)
    return -numpy.einsum('xij,ji->x', ao_dip, t_dm1)

def oscillator_strength(mc, state_i=0, state_j=1, e_i=None, e_j=None,
                        origin=None):
    '''Oscillator strength f = 2/3 * dE * |<i|r|j>|^2 (length gauge).

    Energies are taken from mc.fcisolver.eci unless given explicitly.
    For transitions between degenerate states (Kramers partners), sum the
    oscillator strengths over all degenerate components.

    Returns:
        (f, f_components) with f = f_components.sum()
    '''
    if e_i is None or e_j is None:
        e = numpy.asarray(mc.fcisolver.eci).real
        if e_i is None: e_i = e[state_i]
        if e_j is None: e_j = e[state_j]
    de = abs(e_j - e_i)
    t_dip = transition_dipole(mc, state_i, state_j, origin)
    f_comp = 2./3. * de * numpy.abs(t_dip)**2
    return f_comp.sum(), f_comp
