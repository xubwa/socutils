#!/usr/bin/env python
'''
Atomic x2camf reference orbitals ("atomic minao") for IAO / IBO / AVAS.

Instead of a tabulated scalar minimal basis (pyscf's 'minao', which for
elements beyond Kr is truncated from cc-pVTZ-PP and lacks core functions,
and whose radial contractions are nonrelativistic), the reference minimal
set is the occupied spinors of the spherically averaged atomic Dirac-HF
solved inside the x2camf library (the same atomic calculation that builds
the X2CAMF / X2CMP Hamiltonian), in its two-component (X2C) picture.  This
gives reference AOs with relativistic radial functions (j-split shells),
full all-electron core coverage for every element, and Kramers symmetry by
construction.

The atomic calculation runs in the uncontracted basis of the element.  Its
two-component Fock and density matrices are block diagonal in the (l, j,
m_j) channels of the j-adapted spinor AOs; within each channel the
occupied space is spanned by the natural orbitals of the density (fractional
occupations of open shells, threshold tol), and the reference spinors are
the canonical orbitals of the two-component Fock matrix in that space.  They are projected onto the molecular (possibly
contracted) basis of the atom, C = S_cc^-1 S_cu C_u.

Usage:

>>> mf = spinor_hf.SCF(mol).x2camf().run()
>>> c_ref, labels = atomic_ref.atomic_minao(mf)
>>> ncas, nelecas, mo = avas.avas(mf, ['I 5p', 'H 1s'], minao=(c_ref, labels))
'''

import numpy
import scipy.linalg
from pyscf import gto, lib
from pyscf.data.elements import is_ghost_atom


def _single_atom(symb, basis):
    z = gto.charge(symb)
    return gto.M(atom=[[symb, (0., 0., 0.)]], basis={symb: basis}, spin=z % 2, verbose=0)


def _uncontracted_atom(symb, basis):
    '''Single-atom Mole in the uncontracted basis, shells sorted by
    (l, -exponent): the AO order of the x2camf atomic calculation.'''
    prims = sorted({(b[0], b[-1][0]) for b in gto.mole.uncontracted_basis(basis)},
                   key=lambda p: (p[0], -p[1]))
    atmol = _single_atom(symb, [[l, [e, 1.]] for l, e in prims])
    shell = numpy.array([l for l, e in prims])
    exp_a = numpy.array([e for l, e in prims])
    return atmol, shell, exp_a


def atomic_spinors(symb, basis, with_gaunt=True, with_breit=True, gaussian_nuc=False,
                   with_aoc=False, tol=1e-4):
    '''Occupied spinors of the spherically averaged atomic Dirac-HF of the
    x2camf library, two-component picture, uncontracted basis.

    Returns:
        (atmol, coeff, labels, mo_energy, mo_occ): atmol is the uncontracted
        single-atom Mole of the coefficients; labels are "nlj,mj" (e.g.
        "4p3/2,-1/2"), n counted from l + 1 by orbital energy within each
        channel; mo_occ the (fractional) occupations from the atomic density.
    '''
    from x2camf import libx2camf
    atmol, shell, exp_a = _uncontracted_atom(symb, basis)
    z = gto.charge(symb)
    flavor = int(with_gaunt) + (int(with_breit) << 1) + (int(gaussian_nuc) << 2) + (int(with_aoc) << 3)
    res = libx2camf.atm_integrals(flavor, z, shell[-1] + 1, len(shell), 0,
                                  shell, exp_a, speed_of_light=lib.param.LIGHT_SPEED)
    # results: 4 fock_2c, 12 den_2c
    fock = numpy.asarray(res[4])
    dm = numpy.asarray(res[12])
    s = atmol.intor('int1e_ovlp_spinor')
    if fock.shape != s.shape:
        raise RuntimeError('atomic x2camf matrices of %s have dimension %d, expected %d'
                           % (symb, fock.shape[0], s.shape[0]))

    # (l, j, m_j) channels of the j-adapted spinor AOs
    channels = {}
    for i, (_, _, nlj, mj) in enumerate(atmol.spinor_labels(fmt=False)):
        channels.setdefault((nlj.lstrip('0123456789'), mj), []).append(i)

    nao = s.shape[0]
    coeff, labels, mo_energy, mo_occ = [], [], [], []
    for (lj, mj), idx in channels.items():
        idx = numpy.asarray(idx)
        sb = s[numpy.ix_(idx, idx)]
        sds = sb.dot(dm[numpy.ix_(idx, idx)]).dot(sb)
        # occupied space of the channel: natural orbitals of the density
        n, cno = scipy.linalg.eigh(sds, sb)
        cno = cno[:, n > tol]
        nocc = cno.shape[1]
        if nocc == 0:
            continue
        # canonical orbitals within it (separates the degenerate fully
        # occupied shells, 1s/2s/...)
        e, u = numpy.linalg.eigh(cno.conj().T.dot(fock[numpy.ix_(idx, idx)]).dot(cno))
        c = cno.dot(u)
        cfull = numpy.zeros((nao, nocc), dtype=c.dtype)
        cfull[idx] = c
        l = 'spdfghik'.index(lj[0])
        coeff.append(cfull)
        labels += ['%d%s,%s' % (k + l + 1, lj, mj) for k in range(nocc)]
        mo_energy.append(e)
        mo_occ.append(numpy.einsum('pi,pq,qi->i', c.conj(), sds, c).real)
    coeff = numpy.hstack(coeff)
    mo_energy = numpy.hstack(mo_energy)
    mo_occ = numpy.hstack(mo_occ)
    # the canonical occupied spinors must carry the whole atomic density
    if abs(mo_occ.sum() - z) > 1e-6:
        raise RuntimeError('occupied atomic spinors of %s hold %.8f electrons, expected %d'
                           % (symb, mo_occ.sum(), z))
    order = numpy.argsort(mo_energy, kind='stable')
    return atmol, coeff[:, order], [labels[i] for i in order], mo_energy[order], mo_occ[order]


def _x2c_settings(helper):
    '''Atomic-calculation settings of an X2CAMF / X2CMP helper (defaults:
    Gaunt + Breit, point nucleus).'''
    if helper is None:
        return dict(with_gaunt=True, with_breit=True, gaussian_nuc=False, with_aoc=False)
    return dict(with_gaunt=bool(getattr(helper, 'gaunt', True)),
                with_breit=bool(getattr(helper, 'breit', True)),
                gaussian_nuc=bool(getattr(helper, 'gau_nuc', False)),
                with_aoc=bool(getattr(helper, 'aoc', False)))


def atomic_minao(mf_or_mol, with_gaunt=None, with_breit=None):
    '''Reference minimal set of occupied atomic x2camf spinors for IAO/AVAS.

    Args:
        mf_or_mol : a spinor mean-field object (the x2camf settings are taken
            from its with_x2c helper) or a Mole (defaults: Gaunt + Breit on).

    Returns:
        (c_ref, labels): c_ref (nao_2c x nref) complex array of the occupied
        atomic spinors of every (non-ghost) atom, block-diagonal per atom in
        the molecular spinor AO basis (not orthonormal after the projection
        onto a contracted basis); labels are spinor_labels-style strings
        "ia symb nlj,mj" (e.g. "0 Br 4p3/2,-1/2") for selection by regex.
    '''
    if isinstance(mf_or_mol, gto.MoleBase):
        mol = mf_or_mol
        helper = None
    else:
        mol = mf_or_mol.mol
        helper = getattr(mf_or_mol, 'with_x2c', None)
    settings = _x2c_settings(helper)
    if with_gaunt is not None:
        settings['with_gaunt'] = with_gaunt
    if with_breit is not None:
        settings['with_breit'] = with_breit

    cache = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if is_ghost_atom(symb) or symb in cache:
            continue
        basis = mol._basis[symb]
        atmol, coeff, orb_labels = atomic_spinors(symb, basis, **settings)[:3]
        # projection onto the molecular basis of the element
        cmol = _single_atom(symb, basis)
        s_cc = cmol.intor('int1e_ovlp_spinor')
        s_cu = gto.intor_cross('int1e_ovlp_spinor', cmol, atmol)
        cache[symb] = (scipy.linalg.solve(s_cc, s_cu.dot(coeff), assume_a='pos'), orb_labels)

    slices = mol.aoslice_2c_by_atom()
    nref = sum(cache[mol.atom_symbol(ia)][0].shape[1]
               for ia in range(mol.natm)
               if not is_ghost_atom(mol.atom_symbol(ia)))
    c_ref = numpy.zeros((mol.nao_2c(), nref), dtype=complex)
    labels = []
    col = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if is_ghost_atom(symb):
            continue
        coeff, orb_labels = cache[symb]
        p0, p1 = slices[ia][2:4]
        assert p1 - p0 == coeff.shape[0], \
            'atomic/molecular basis mismatch for atom %d %s' % (ia, symb)
        n = coeff.shape[1]
        c_ref[p0:p1, col:col+n] = coeff
        labels += ['%d %s %s' % (ia, mol.atom_pure_symbol(ia), ol)
                   for ol in orb_labels]
        col += n
    return c_ref, labels
