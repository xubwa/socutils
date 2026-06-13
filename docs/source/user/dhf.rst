Four-component Dirac-Hartree-Fock
=================================

Alongside the two-component spinor and GHF drivers (:doc:`scf`), socutils
provides a few specialized **four-component** Dirac-Hartree-Fock drivers, built
on PySCF's ``dhf`` module.  These are mainly reference and benchmarking tools:
the full four-component treatment is the parent theory from which the X2C
two-component Hamiltonians are derived.

Symmetry-adapted DHF
--------------------

``scf.linear_dhf.SymmDHF(mol, symmetry='linear', occup=None)`` is a
symmetry-adapted four-component DHF.  It shares the occupation-control logic of
the two-component ``spinor_hf.SymmSpinorSCF`` (an ``occup`` dictionary keyed by
irrep label selects the configuration) but operates in the four-component
picture, and it supports **linear** symmetry only.

.. warning::

   As with ``SymmSpinorSCF(symmetry='linear')``, the molecule **must** be
   placed along the ``z`` axis; the irrep assignment assumes this and does not
   check the geometry.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import linear_dhf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = linear_dhf.SymmDHF(mol, symmetry='linear')
   e = mf.kernel()

Spin-free DHF
-------------

``scf.sfdhf.SpinFreeDHF(mol)`` is a spin-free (scalar-relativistic)
four-component DHF -- it keeps the four-component structure but removes the
spin-orbit coupling, which is useful for separating scalar-relativistic from
spin-orbit effects.  ``scf.sfdhf.SpinFreeDKS`` is the corresponding
Kohn-Sham (DFT) variant.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import sfdhf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = sfdhf.SpinFreeDHF(mol)
   e = mf.kernel()

Fractional-occupation DHF
-------------------------

``scf.frac_dhf.FRAC_RDHF(mol, nopen=0, nact=0)`` is a Kramers-restricted
four-component DHF that allows fractional occupation of an open shell:
``nact`` electrons are spread evenly over the ``nopen`` frontier orbitals.
This is the four-component tool for averaging over a degenerate open shell
(for instance to obtain a spherically symmetric atomic reference).  It requires
the optional ``zquatev`` library (see :doc:`../install`).

.. code-block:: python

   from pyscf import gto
   from socutils.scf import frac_dhf

   mol = gto.M(atom='C 0 0 0', basis='ccpvdz', spin=2, verbose=4)

   # spread the 2 open-shell electrons evenly over the 3 frontier orbitals
   mf = frac_dhf.FRAC_RDHF(mol, nopen=3, nact=2)
   e = mf.kernel()
