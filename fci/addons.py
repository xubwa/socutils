#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Transition properties (transition density matrix, electric transition
dipole moments, oscillator strengths, Einstein coefficients and radiative
lifetimes) and angular momentum analysis for spinor CASCI/CASSCF
calculations.

Only the electric dipole (length gauge) channel is provided: magnetic
multipole operators for 2c (X2C) wavefunctions require picture-change
transformed operators derived from the relativistic current, and the bare
nonrelativistic L+2S form is not valid in this framework.

The functions work with any fcisolver providing a
trans_rdm1(ci_bra, ci_ket, ncas, nelecas) method, e.g. zfci.FCISolver and
zfci.SelectedCI.  The Dice interface (socutils.hci.shci) is also supported,
since there mc.ci holds the root indices that its trans_rdm1 expects.

Transitions between states from two different determinant spaces (e.g. a
valence ground state and core-hole states from a SelectedCI calculation)
are supported by passing the second CASCI object as mc_ket, as long as the
two calculations share the same orbitals and active space.
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.fci import cistring
from pyscf.data import nist

# atomic time unit in seconds, hbar/E_h
T_AU = nist.PLANCK / (2 * numpy.pi * nist.HARTREE2J)


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

def _solver_occslst(mc):
    '''Determinant list of the CASCI object's fcisolver; full determinant
    space in cistring order for full CI solvers.'''
    occslst = getattr(mc.fcisolver, 'occslst', None)
    if occslst is not None:
        return numpy.asarray(occslst)
    nelec = mc.nelecas
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    return numpy.asarray(cistring.gen_occslst(range(mc.ncas), nelec))

def _trans_rdm1_act(mc, state_i, state_j, mc_ket=None):
    '''Active space transition density t[p,q] = <i|p^+ q|j>.  With mc_ket,
    bra and ket states come from different CI calculations (e.g. different
    determinant lists) sharing the same orbitals and active space.'''
    if mc_ket is None or mc_ket is mc:
        ci_i = _state_ci(mc.ci, state_i)
        ci_j = _state_ci(mc.ci, state_j)
        return mc.fcisolver.trans_rdm1(ci_i, ci_j, mc.ncas, mc.nelecas)

    from socutils.fci import zfci
    if mc.ncas != mc_ket.ncas or mc.ncore != mc_ket.ncore:
        raise ValueError('bra and ket CASCI objects must share the same '
                         'active space')
    if mc.mo_coeff is not mc_ket.mo_coeff and \
       not numpy.allclose(mc.mo_coeff, mc_ket.mo_coeff):
        raise ValueError('bra and ket CASCI objects must share mo_coeff')
    ci_i = _state_ci(mc.ci, state_i)
    ci_j = _state_ci(mc_ket.ci, state_j)
    return zfci.trans_rdm1_dets(ci_i, ci_j, mc.ncas,
                                _solver_occslst(mc), _solver_occslst(mc_ket))

def trans_rdm1_ao(mc, state_i=0, state_j=1, mo_coeff=None, mc_ket=None):
    '''Transition density matrix in the AO spinor basis, defined such that
    the matrix element of a one-electron operator O is

        <i|O|j> = einsum('uv,vu->', O_ao, dm)

    with O_ao[u,v] = <chi_u|O|chi_v>.  This requires dm = C t^T C^dagger
    where t[p,q] = <i|p^+ q|j>; with complex spinor orbitals the transpose
    matters (C t C^dagger contracts the operator with t transposed and is
    wrong for transition properties).

    The core contribution <i|j> * dm_core is included; it survives only
    when the bra and ket states are not orthogonal (in particular for
    state_i == state_j).
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    ncore = mc.ncore
    ncas = mc.ncas
    nelec = mc.nelecas
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    t_dm1 = _trans_rdm1_act(mc, state_i, state_j, mc_ket)
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    dm = reduce(numpy.dot, (mo_cas, t_dm1.T, mo_cas.conj().T))
    if ncore > 0 and nelec > 0:
        # Tr t = <i|N|j> = nelec <i|j>
        ovlp = numpy.trace(t_dm1) / nelec
        if abs(ovlp) > 1e-14:
            mo_core = mo_coeff[:, :ncore]
            dm = dm + ovlp * numpy.dot(mo_core, mo_core.conj().T)
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

def transition_dipole(mc, state_i=0, state_j=1, origin=None, mc_ket=None):
    '''Electronic (transition) dipole moment <i|-r|j> in a.u. (length gauge).

    For orthogonal states the result is independent of the gauge origin;
    for state_i == state_j it is the electronic dipole of that state with
    respect to origin (the nuclear charge center by default), so that
    adding the nuclear contribution gives the total dipole moment.

    Kwargs:
        mc_ket: a second CASCI object providing the ket states, for
            transitions between two different determinant spaces.
    '''
    mol = mc.mol
    if origin is None:
        origin = charge_center(mol)
    t_dm1 = trans_rdm1_ao(mc, state_i, state_j, mc_ket=mc_ket)
    ao_dip = _dipole_integrals(mol, t_dm1.shape[0], origin)
    return -numpy.einsum('xij,ji->x', ao_dip, t_dm1)

def _spin_ao(mol, nao):
    '''Electron spin matrices s = sigma/2 in the 2c AO spinor basis'''
    if nao != mol.nao_2c():
        raise NotImplementedError('spin matrices implemented for the 2c '
                                  'spinor basis only')
    return 0.5 * mol.intor('int1e_sigma_spinor', comp=3)

def _orb_angmom_ao(mol, nao, origin):
    '''Orbital angular momentum matrices L = -i r x nabla (about origin)
    in the 2c AO spinor basis'''
    if nao != mol.nao_2c():
        raise NotImplementedError('angular momentum matrices implemented '
                                  'for the 2c spinor basis only')
    with mol.with_common_orig(origin):
        # int1e_cg_irxp = <r x nabla> (anti-hermitian)
        return -1j * mol.intor('int1e_cg_irxp_spinor', comp=3)

def oscillator_strength(mc, state_i=0, state_j=1, e_i=None, e_j=None,
                        origin=None, mc_ket=None):
    '''Electric dipole oscillator strength f = 2/3 dE |<i|r|j>|^2
    (length gauge).

    Energies are taken from the fcisolver.eci attributes unless given
    explicitly.  For transitions between degenerate states (Kramers
    multiplets), only the sum of f over the degenerate components is
    physically meaningful.

    Returns:
        (f, f_components) with f = f_components.sum()
    '''
    if e_i is None:
        e_i = numpy.asarray(mc.fcisolver.eci).real.reshape(-1)[state_i]
    if e_j is None:
        e_j = numpy.asarray((mc_ket or mc).fcisolver.eci).real.reshape(-1)[state_j]
    de = abs(e_j - e_i)
    t_dip = transition_dipole(mc, state_i, state_j, origin, mc_ket=mc_ket)
    f_comp = 2./3. * de * numpy.abs(t_dip)**2
    return f_comp.sum(), f_comp

def einstein_coefficient_a(mc, state_upper, state_lower, origin=None,
                           mc_ket=None):
    '''Einstein spontaneous emission coefficient A (in s^-1) for the
    electric dipole channel of the transition state_upper -> state_lower:

        A_E1 = 4/3 alpha^3 w^3 |<u|r|l>|^2

    (atomic units, converted to s^-1).  state_lower belongs to mc_ket when
    given.  Higher multipole channels (M1, E2, ...) are not included; for
    2c wavefunctions they require picture-change transformed operators.
    '''
    e_u = numpy.asarray(mc.fcisolver.eci).real.reshape(-1)[state_upper]
    e_l = numpy.asarray((mc_ket or mc).fcisolver.eci).real.reshape(-1)[state_lower]
    w = abs(e_u - e_l)
    alpha = 1. / lib.param.LIGHT_SPEED
    t_dip = transition_dipole(mc, state_upper, state_lower, origin, mc_ket=mc_ket)
    return 4./3. * alpha**3 * w**3 * (numpy.abs(t_dip)**2).sum() / T_AU

def radiative_lifetime(mc, state, lower_states=None, origin=None):
    '''Radiative lifetime (in seconds) of a state, from the sum of the
    Einstein A coefficients (E1 only) over all lower-lying states.

    For an initial state belonging to a degenerate multiplet, average the
    decay rates over the degenerate components (high-temperature limit:
    tau = n / sum_i 1/tau_i).
    '''
    e = numpy.asarray(mc.fcisolver.eci).real.reshape(-1)
    if lower_states is None:
        lower_states = [i for i in range(len(e)) if e[i] < e[state]]
    a_tot = 0.
    for i in lower_states:
        a_tot += einstein_coefficient_a(mc, state, i, origin)
    return 1. / a_tot

def angular_momentum_square(mc, state=0, kind='spin', origin=None):
    '''Expectation value of S^2, L^2 or J^2 = (L+S)^2 for a CASCI state,
    intended for state classification (spin composition, atomic j labels).

    Caveat: the bare (untransformed) operators are contracted with the 2c
    wavefunction, i.e. no X2C picture-change transformation is applied to
    S and L.  The resulting values are therefore diagnostics rather than
    rigorous observables; the associated error is O(alpha^2) and does not
    affect the classification of valence states.

    The two-electron part is evaluated with the active space 1- and 2-RDMs;
    the closed core shell is assumed to carry no net angular momentum
    (exact for S^2 of a closed-shell core in the absence of SOC, and for
    L^2/J^2 when the core orbitals are s-type or complete shells).

    Args:
        kind: 'spin' for S^2, 'orb' for L^2, 'total' for J^2
    '''
    mol = mc.mol
    if origin is None:
        origin = charge_center(mol)
    ncore, ncas = mc.ncore, mc.ncas
    nao = mc.mo_coeff.shape[0]
    if kind == 'spin':
        mats = _spin_ao(mol, nao)
    elif kind == 'orb':
        mats = _orb_angmom_ao(mol, nao, origin)
    elif kind == 'total':
        mats = _orb_angmom_ao(mol, nao, origin) + _spin_ao(mol, nao)
    else:
        raise ValueError(kind)
    mo_cas = mc.mo_coeff[:, ncore:ncore+ncas]
    mats = lib.einsum('up,xuv,vq->xpq', mo_cas.conj(), mats, mo_cas)
    dm1, dm2 = mc.fcisolver.make_rdm12(_state_ci(mc.ci, state), ncas,
                                       mc.nelecas)
    # <(sum_i o_i)^2> = sum_pq (o.o)_pq <p^+ q>
    #                 + sum_pqrs o_pq o_rs <p^+ r^+ s q>
    val = 0.
    for m in mats:
        val += numpy.einsum('pq,pq->', m @ m, dm1)
        val += numpy.einsum('pq,rs,pqrs->', m, m, dm2)
    return val.real

def spin_square(mc, state=0):
    '''<S^2> of a CASCI state and the effective multiplicity 2S+1 obtained
    from S(S+1) = <S^2>.  With spin-orbit coupling, <S^2> interpolates
    between the pure-spin values (0 for singlet, 2 for triplet, ...) and
    quantifies the singlet/triplet composition of a SOC-mixed state.
    '''
    ss = angular_momentum_square(mc, state, kind='spin')
    s = 0.5 * (numpy.sqrt(1 + 4*ss) - 1)
    return ss, 2*s + 1
