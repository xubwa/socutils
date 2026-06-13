#!/usr/bin/env python
'''
Generalized Hartree-Fock (GHF) driver with an X2C spin-orbit Hamiltonian.

This is the spin-orbital (GHF) analogue of :mod:`socutils.scf.spinor_hf`.  The
spin-orbit physics is attached through ``with_x2c`` with the ``.x2camf()`` /
``.x2cmp()`` shortcuts, exactly as ``spinor_hf.SCF(mol).x2camf()`` does for the
spinor representation -- mirroring PySCF's ``mf.x2c()``::

    from socutils.scf import ghf
    mf = ghf.GHF(mol).x2camf()
    mf.kernel()

The spinor and GHF styles describe the same Hamiltonian and agree to numerical
precision; the GHF style stays close to PySCF's native GHF toolchain.
'''

from pyscf.scf import ghf
from pyscf.lib import logger


class GHF(ghf.GHF):
    '''GHF with an X2C spin-orbit Hamiltonian attached through ``with_x2c``.'''

    _keys = {'with_x2c'}

    def __init__(self, mol):
        ghf.GHF.__init__(self, mol)
        self.with_x2c = None

    def _attach_x2cmp(self, flavor, with_gaunt, with_breit, with_pcc, with_aoc):
        from socutils.somf.x2cmp import SpinOrbitalX2CMPHelper
        self.with_x2c = SpinOrbitalX2CMPHelper(
            self.mol, x2cmp=flavor, with_gaunt=with_gaunt, with_breit=with_breit,
            with_pcc=with_pcc, with_aoc=with_aoc)
        self._keys = self._keys.union(['with_x2c'])
        terms = ['Dirac-Coulomb']
        if with_gaunt: terms.append('Gaunt')
        if with_breit: terms.append('Breit')
        if with_pcc: terms.append('PCC')
        logger.info(self, 'X2C SOC Hamiltonian: flavor=%s, terms = %s',
                    flavor, ' + '.join(terms))
        return self

    def x2camf(self, flavor='x2camf', with_gaunt=True, with_breit=True,
               with_pcc=False, with_aoc=False):
        '''Attach an X2CAMF (atomic-mean-field) spin-orbit Hamiltonian to this
        GHF object, in place, and return it -- the GHF analogue of PySCF's
        ``mf.x2c()``.  The SOC enters through ``self.with_x2c`` and is applied
        in :meth:`get_hcore`.

        >>> mf = GHF(mol).x2camf()               # Gaunt + Breit
        >>> mf.kernel()
        '''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def x2cmp(self, flavor='x2cmp', with_gaunt=True, with_breit=True,
              with_pcc=False, with_aoc=False):
        '''Attach an X2C molecular-mean-field (molecular picture-change) spin-
        orbit Hamiltonian to this GHF object, in place, and return it.  Same
        usage as :meth:`x2camf`, with the molecular-mean-field flavor.

        >>> mf = GHF(mol).x2cmp()
        >>> mf.kernel()
        '''
        return self._attach_x2cmp(flavor, with_gaunt, with_breit, with_pcc, with_aoc)

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        if self.with_x2c is not None:
            return self.with_x2c.get_hcore(mol)
        return ghf.GHF.get_hcore(self, mol)


SCF = GHF
