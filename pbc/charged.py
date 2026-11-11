'''
Charged periodic cells.

Periodic Coulomb sums drop the G = 0 term, i.e. every cell is put in a uniform neutralizing background.  For a
cell of net charge q = sum(Z) - nelectron the energy then misses the electrostatic energy of the lattice of
charges q in that background, -q^2 nu_M / 2 per cell (nu_M: the Madelung constant of the cell's lattice,
tools.pbc.madelung; Makov and Payne, PRB 51, 4014 (1995), first order):

    E(isolated) ~= E(periodic) + q^2 nu_M / 2.

The remaining finite-size error is the one of the neutral cell (He+ / He in cubic boxes, X2CAMF GHF: after the
correction the charged and the neutral errors agree to 1e-6 Eh at L = 10..40 bohr and fall off as 1/L^3), so the
Makov-Payne quadrupole term is not added.  nu_M is that of the cell itself, whatever the k-point mesh: the charge
density is periodic with the cell.

Orbital energies of a charged cell are shifted by a constant (the G = 0 potential is dropped); take ionization or
attachment energies from total energies.

Usage (any pyscf.pbc SCF object, e.g. a GHF / GKS with the X2CAMF helper of socutils.pbc.amf):

    mf = charged.correct(mf)       # mf.energy_nuc() and mf.e_tot then include q^2 nu_M / 2
'''

import numpy
from pyscf import lib
from pyscf.pbc import tools


def net_charge(cell):
    '''Net charge of the cell: nuclear charges minus electrons (cell.charge).'''
    return float(cell.atom_charges().sum() - cell.nelectron)


def charge_correction(cell):
    '''q^2 nu_M / 2 (Hartree), the energy the periodic G = 0-free sums miss for a cell of net charge q.'''
    q = net_charge(cell)
    if q == 0:
        return 0.
    if cell.dimension != 3:
        raise NotImplementedError('charged-cell correction for low-dimensional cells')
    return .5 * q * q * tools.pbc.madelung(cell, numpy.zeros(3))


def correct(mf):
    '''Add the charged-cell correction to mf.energy_nuc (so to e_tot); returns mf.  A no-op for a neutral cell.
    The correction is stored as mf.charge_correction.'''
    cell = mf.cell
    corr = charge_correction(cell)
    mf.charge_correction = corr
    if corr == 0:
        return mf
    energy_nuc0 = mf.energy_nuc

    def energy_nuc(*args, **kwargs):
        return energy_nuc0(*args, **kwargs) + corr
    mf.energy_nuc = energy_nuc
    lib.logger.info(mf, 'charged cell: net charge %g, q^2 nu_M / 2 = %.10f added to energy_nuc',
                    net_charge(cell), corr)
    return mf
