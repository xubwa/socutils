#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
"""Add-on helpers for the spinor SCF methods (:mod:`socutils.scf.spinor_hf`).

Mirrors :mod:`pyscf.scf.addons`: small utilities that *decorate* a mean-field
object (typically by overriding ``get_occ``) rather than living on the SCF
class itself.  Use as, e.g.::

    from socutils.scf import spinor_hf, addons
    mf = spinor_hf.SCF(mol).x2camf()
    addons.even_occ(mf, norb_open=2, nelec_open=1)   # in place
    mf.kernel()
"""

import numpy
from pyscf.lib import logger


def even_occ(mf, norb_open=0, nelec_open=0):
    '''Even (uniform) fractional occupation of an *explicitly specified* open shell.

    Distributes ``nelec_open`` electrons evenly over the ``norb_open`` open-shell
    spinors that sit just above the ``nclose = mol.nelectron - nelec_open``
    fully-occupied spinors, by overriding ``mf.get_occ`` in place.  The open
    shell is given explicitly (unlike :func:`pyscf.scf.addons.frac_occ`, which
    auto-detects degenerate frontier orbitals).

    This only sets the *occupation numbers* and then runs the ordinary
    mean-field.  It is well defined for DFT (the energy is a functional of the
    fractionally-occupied density), but it is **not** a true average-of-
    configuration / CAHF treatment: it does not include the inter-shell
    coupling (averaged Slater--Condon J/K) coefficients that the AOC/CAHF energy
    functional carries, so for HF it is an approximation, not the AOC energy.

    Returns the same ``mf`` object (so it can be chained).
    '''
    mol = mf.mol
    nclose = mol.nelectron - nelec_open
    frac = nelec_open / norb_open if norb_open else 0.0

    def get_occ(mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nclose] = 1.0
        mo_occ[nclose:nclose + norb_open] = frac
        logger.info(mf, '  fractional open-shell occupation = %.6f over %d spinors',
                    frac, norb_open)
        logger.info(mf, '  open-shell orbital energies = %s',
                    mo_energy[nclose:nclose + norb_open])
        logger.debug(mf, '  mo_energy = %s', mo_energy)
        return mo_occ

    mf.get_occ = get_occ
    return mf
