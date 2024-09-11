import numpy as np
import scipy

from pyscf.data import nist
from pyscf.data.elements import MASSES

au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * np.pi)

def fit(x, y, order):
    coeff = np.polyfit(x, y, order)
    p = np.poly1d(coeff)
    solution = scipy.optimize.minimize_scalar(p, bounds=(x[0], x[-1]), method='bounded')
    x = solution.x
    min_val = p(x)
    force_const = p.deriv().deriv()(x)
    bond_length = x
    print(f'Equilibrium bond length = {x:.6f} Angstrom, force constant = {force_const:.6f} Hartree/Angstrom^2, minimum energy = {min_val:.8f} Hartree')
    return bond_length, force_const


def au2cm(mol, fc):
    mol.atom_charges()
    mass0 = MASSES[mol.atom_charges()[0]]
    mass1 = MASSES[mol.atom_charges()[1]]
    reduced_mass = mass0 * mass1 / (mass0 + mass1)
    bohr = nist.BOHR
    force_const_au = np.sqrt(fc * bohr ** 2 / reduced_mass)
    force_const_cm = force_const_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2
    return force_const_cm

