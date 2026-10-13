#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Numerically exact reference values for periodic Coulomb quantities at the
# Gamma point.  Used to validate the production periodic J (and later K)
# builders; nothing here is meant to be fast.
#
# Everything is built from analytic formulas and *molecular* integrals only
# (libcint through pyscf.gto), so the references are independent of the
# lattice-sum estimators in pyscf.pbc.  Only pure geometry helpers
# (lattice/reciprocal vectors) are taken from the Cell object.
#
# Conventions (shared with pyscf.pbc and with the production code):
#   * the G = 0 component of the electrostatic potential is dropped
#     ("tin-foil" boundary, zero average potential);
#   * nuclei are normalised Gaussian charges Z (zeta/pi)^1.5 exp(-zeta r^2);
#     the point-nucleus limit is zeta -> infinity;
#   * the Coulomb energy of a Gaussian charge distribution includes the
#     self-interaction of each Gaussian, 0.5 q^2 sqrt(2 alpha/pi).
#

import numpy as np
from scipy.special import erf, erfc
from pyscf import gto
from pyscf.gto.ft_ao import ft_aopair


def lattice_translations(a, rcut):
    '''All lattice vectors T = n.a with |T| <= rcut.  a: rows are the lattice vectors.'''
    a = np.asarray(a, dtype=float)
    b = np.linalg.inv(a).T  # b_i . a_j = delta_ij; n_i = T . b_i
    nmax = np.ceil(rcut * np.linalg.norm(b, axis=1)).astype(int)
    rng = [np.arange(-n, n + 1) for n in nmax]
    n = np.array(np.meshgrid(*rng, indexing='ij')).reshape(3, -1).T
    T = n @ a
    return T[np.linalg.norm(T, axis=1) <= rcut]


def reciprocal_vectors(a, gcut):
    '''All nonzero reciprocal lattice vectors G with |G| <= gcut.'''
    a = np.asarray(a, dtype=float)
    b = 2 * np.pi * np.linalg.inv(a).T
    G = lattice_translations(b, gcut)
    return G[np.linalg.norm(G, axis=1) > 1e-12]


def cell_volume(a):
    return abs(np.linalg.det(np.asarray(a, dtype=float)))


#
# Point charges
#
def ewald_point_charges(a, coords, charges, eta=1.0, prec=1e-16):
    '''Ewald energy per cell of a neutral lattice of point charges (no self energy).'''
    coords = np.asarray(coords, dtype=float)
    q = np.asarray(charges, dtype=float)
    assert abs(q.sum()) < 1e-12, 'cell must be neutral'
    vol = cell_volume(a)
    lnp = -np.log(prec)

    rcut = (np.sqrt(lnp) + 2.0) / eta
    T = lattice_translations(a, rcut + np.ptp(coords, axis=0).max())
    e_real = 0.0
    for i in range(len(q)):
        r = np.linalg.norm(coords[:, None, :] + T[None, :, :] - coords[i], axis=2)
        mask = r > 1e-12
        e_real += 0.5 * q[i] * np.sum(q[:, None] * np.where(mask, erfc(eta * r) / np.where(mask, r, 1.0), 0.0))

    gcut = 2 * eta * (np.sqrt(lnp) + 2.0)
    G = reciprocal_vectors(a, gcut)
    G2 = np.einsum('gx,gx->g', G, G)
    S = np.exp(-1j * G @ coords.T) @ q
    e_recip = 2 * np.pi / vol * np.sum(abs(S)**2 * np.exp(-G2 / (4 * eta**2)) / G2)

    e_self = -eta / np.sqrt(np.pi) * np.sum(q**2)
    return e_real + e_recip + e_self


#
# Gaussian charges
#
def gaussian_charges_recip(a, coords, charges, alphas, prec=1e-16):
    '''Coulomb energy per cell of a neutral lattice of normalised Gaussian charges,
    as a pure reciprocal-space sum (no splitting).  Includes Gaussian self energies.'''
    coords = np.asarray(coords, dtype=float)
    q = np.asarray(charges, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    assert abs(q.sum()) < 1e-12, 'cell must be neutral'
    vol = cell_volume(a)
    # slowest decaying factor: exp(-G^2/(2 alpha_max)) in |S|^2
    gcut = np.sqrt(2 * alphas.max() * (-np.log(prec) + 10.0))
    G = reciprocal_vectors(a, gcut)
    G2 = np.einsum('gx,gx->g', G, G)
    S = (np.exp(-1j * G @ coords.T) * np.exp(-G2[:, None] / (4 * alphas[None, :]))) @ q
    return 2 * np.pi / vol * np.sum(abs(S)**2 / G2)


def gaussian_charges_ewald(a, coords, charges, alphas, eta=1.0, prec=1e-16):
    '''Same quantity as gaussian_charges_recip, via an Ewald split with parameter eta.

    The pair kernel of Gaussians i, j is erf(sqrt(g_ij) r)/r, g_ij = a_i a_j/(a_i + a_j).
    It is split into [erf(sqrt(g_ij) r) - erf(b_ij r)]/r (real space) plus
    erf(b_ij r)/r (reciprocal space), 1/b_ij^2 = 1/g_ij + 1/eta^2.  The constant G = 0
    piece of the real-space kernel cancels for a neutral cell.'''
    coords = np.asarray(coords, dtype=float)
    q = np.asarray(charges, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    assert abs(q.sum()) < 1e-12, 'cell must be neutral'
    vol = cell_volume(a)
    lnp = -np.log(prec)

    g = alphas[:, None] * alphas[None, :] / (alphas[:, None] + alphas[None, :])
    beta = 1.0 / np.sqrt(1.0 / g + 1.0 / eta**2)
    rcut = (np.sqrt(lnp) + 2.0) / beta.min()
    T = lattice_translations(a, rcut + np.ptp(coords, axis=0).max())
    e_real = 0.0
    for i in range(len(q)):
        r = np.linalg.norm(coords[:, None, :] + T[None, :, :] - coords[i], axis=2)
        sg = np.sqrt(g[i])[:, None]
        bt = beta[i][:, None]
        rs = np.where(r > 1e-12, r, 1.0)
        kern = np.where(r > 1e-12, (erf(sg * r) - erf(bt * r)) / rs,
                        2 * (sg - bt) / np.sqrt(np.pi))
        e_real += 0.5 * q[i] * np.sum(q[:, None] * kern)

    gcut = np.sqrt(4 * (lnp + 10.0) / (1.0 / (2 * alphas.max()) + 1.0 / eta**2))
    G = reciprocal_vectors(a, gcut)
    G2 = np.einsum('gx,gx->g', G, G)
    S = (np.exp(-1j * G @ coords.T) * np.exp(-G2[:, None] / (4 * alphas[None, :]))) @ q
    e_recip = 2 * np.pi / vol * np.sum(abs(S)**2 * np.exp(-G2 / (4 * eta**2)) / G2)
    return e_real + e_recip


def gaussian_self_energy(charges, alphas):
    '''sum_i 0.5 q_i^2 sqrt(2 alpha_i/pi): self Coulomb energy of normalised Gaussians.'''
    q = np.asarray(charges, dtype=float)
    return 0.5 * np.sum(q**2 * np.sqrt(2 * np.asarray(alphas, dtype=float) / np.pi))


#
# AO density + Gaussian nuclei
#
def _image_mole(cell, rpair):
    '''Molecule made of cell-0 atoms followed by every image atom within rpair of
    some cell-0 atom.  Returns (mol, ao_fold) where ao_fold maps each AO of mol to
    the corresponding AO index of the cell.'''
    a = cell.lattice_vectors()
    coords0 = cell.atom_coords()
    natm0 = cell.natm
    span = np.ptp(coords0, axis=0).max() if natm0 > 1 else 0.0
    T = lattice_translations(a, rpair + span)
    T = T[np.argsort(np.linalg.norm(T, axis=1), kind='stable')]
    assert np.linalg.norm(T[0]) < 1e-12

    atoms = []
    origin = []
    for t in T:
        for ia in range(natm0):
            x = coords0[ia] + t
            if np.linalg.norm(t) < 1e-12 or \
               np.min(np.linalg.norm(coords0 - x, axis=1)) <= rpair:
                atoms.append((cell.atom_symbol(ia), x))
                origin.append(ia)

    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = 'Bohr'
    mol.basis = cell._basis
    mol.cart = cell.cart
    mol.verbose = 0
    mol.spin = sum(gto.charge(s) for s, _ in atoms) % 2
    mol.build(False, False)

    ao_loc0 = cell.aoslice_by_atom()
    ao_fold = []
    for ia in origin:
        p0, p1 = ao_loc0[ia, 2:4]
        ao_fold.extend(range(p0, p1))
    ao_fold = np.asarray(ao_fold)
    assert np.all(ao_fold[:cell.nao] == np.arange(cell.nao))
    return mol, ao_fold


def coulomb_reference(cell, dm, zeta, ke_cutoff, rpair, blksize=2048):
    '''Gamma-point periodic Coulomb quantities, evaluated as a pure reciprocal
    space sum with analytic Fourier transforms of explicit AO pairs.

    Args:
        cell : pyscf.pbc.gto.Cell (only geometry and basis are used)
        dm : (nao, nao) real symmetric Gamma-point AO density matrix
        zeta : float or (natm,) Gaussian nuclear exponents
        ke_cutoff : spherical cutoff |G|^2/2 <= ke_cutoff (Hartree)
        rpair : AO pairs (mu in cell 0, nu in image) are kept when the atoms of
                nu lie within rpair of a cell-0 atom

    Returns dict with
        vj    : Coulomb matrix of the electron density, sum_T <mu_0|phi_e|nu_T>
        vnuc  : electron-nucleus attraction matrix
        e_ee, e_en, e_nn : the three parts of the Coulomb energy per cell
                           (e_nn includes the Gaussian nuclear self energies)
        e_coul : e_ee + e_en + e_nn
        ngrids, nimg_ao, max_imag
    '''
    dm = np.asarray(dm)
    assert dm.ndim == 2 and not np.iscomplexobj(dm)
    a = cell.lattice_vectors()
    vol = cell_volume(a)
    nao = cell.nao
    coords0 = cell.atom_coords()
    Z = cell.atom_charges().astype(float)
    zeta = np.broadcast_to(np.asarray(zeta, dtype=float), (cell.natm,))

    mol, ao_fold = _image_mole(cell, rpair)
    nbas0 = cell.nbas
    shls_slice = (0, nbas0, 0, mol.nbas)
    dm_img = dm[:, ao_fold]  # D_{mu nu} for every image of nu

    G = reciprocal_vectors(a, np.sqrt(2 * ke_cutoff))
    G2 = np.einsum('gx,gx->g', G, G)
    ng = len(G)

    rho_n = (np.exp(-1j * G @ coords0.T) * np.exp(-G2[:, None] / (4 * zeta[None, :]))) @ Z / vol
    rho_e = np.empty(ng, dtype=complex)
    for g0 in range(0, ng, blksize):
        g1 = min(g0 + blksize, ng)
        F = ft_aopair(mol, G[g0:g1], shls_slice=shls_slice)
        rho_e[g0:g1] = np.einsum('gij,ij->g', F, dm_img) / vol

    coulG = 4 * np.pi / G2
    phi_e = coulG * rho_e
    phi_n = coulG * rho_n
    vj = np.zeros((nao, len(ao_fold)), dtype=complex)
    vn = np.zeros((nao, len(ao_fold)), dtype=complex)
    for g0 in range(0, ng, blksize):
        g1 = min(g0 + blksize, ng)
        F = ft_aopair(mol, G[g0:g1], shls_slice=shls_slice)
        vj += np.einsum('gij,g->ij', F.conj(), phi_e[g0:g1])
        vn -= np.einsum('gij,g->ij', F.conj(), phi_n[g0:g1])

    vj_cell = np.zeros((nao, nao), dtype=complex)
    vn_cell = np.zeros((nao, nao), dtype=complex)
    np.add.at(vj_cell.T, ao_fold, vj.T)
    np.add.at(vn_cell.T, ao_fold, vn.T)
    max_imag = max(abs(vj_cell.imag).max(), abs(vn_cell.imag).max())

    e_ee = 2 * np.pi * vol * np.sum(abs(rho_e)**2 / G2)
    e_en = -4 * np.pi * vol * np.sum((rho_e.conj() * rho_n).real / G2)
    e_nn = 2 * np.pi * vol * np.sum(abs(rho_n)**2 / G2)
    return {
        'vj': vj_cell.real, 'vnuc': vn_cell.real,
        'e_ee': e_ee, 'e_en': e_en, 'e_nn': e_nn, 'e_coul': e_ee + e_en + e_nn,
        'ngrids': ng, 'nimg_ao': len(ao_fold), 'max_imag': max_imag,
    }
