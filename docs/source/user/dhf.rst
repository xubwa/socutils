Four-component Dirac-Hartree-Fock
=================================

Alongside the two-component spinor and GHF drivers (:doc:`scf`), socutils
provides a few specialized **four-component** Dirac-Hartree-Fock drivers, built
on PySCF's ``dhf`` module.  These are mainly reference and benchmarking tools:
the full four-component treatment is the parent theory from which the X2C
two-component Hamiltonians are derived.

Double-group symmetric DHF
--------------------------

``scf.double_group_symm.SymAdaptedDHF`` and ``SymAdaptedRDHF`` (Kramers
restricted) are the four-component counterparts of the double-group spinor SCF
(:doc:`scf`): same groups, irrep names, ``irrep_nelec`` and symmetry-frame
requirement.  The small component carries the same irrep as the large one.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import double_group_symm

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', symmetry=True, verbose=4)

   mf = double_group_symm.SymAdaptedRDHF(mol)
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
