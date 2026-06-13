Spin-orbit mean-field Hamiltonians (somf)
=========================================

socutils introduces spin-orbit coupling through an **exact two-component
(X2C)** one-electron Hamiltonian.  Starting from the four-component
Dirac picture, the X2C transformation decouples the electronic block into a
two-component operator that already contains the one-electron spin-orbit
interaction.  The two-electron spin-orbit contribution is added as a
mean-field operator, which is where the different *flavors* below differ.

Flavors
-------

Two flavors are recommended and supported as user entry points:

``x2camf``
    Atomic mean-field.  The two-electron spin-orbit mean field is built from
    converged atomic four-component calculations and added to the molecular
    X2C core, performing the spin separation atom by atom.

``x2cmp``
    Molecular picture-change.  The mean-field correction is evaluated in the
    full molecular four-component picture before the two-component reduction.

Both are correct; ``x2cmp`` is the more complete of the two.  They are exposed
through the driver shortcuts (next section).

Attaching the Hamiltonian
-------------------------

The recommended way to use either flavor is the driver shortcuts on a spinor
SCF object (see :doc:`scf`):

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SCF(mol).x2camf()   # atomic mean-field flavor
   # or
   mf = spinor_hf.SCF(mol).x2cmp()    # molecular picture-change flavor
   e = mf.kernel()

``.x2camf()`` defaults to the ``x2camf`` flavor and ``.x2cmp()`` to the
``x2cmp`` flavor.  Under the hood both build a
``somf.x2cmp.SpinorX2CMPHelper`` and store it on ``mf.with_x2c``; the active
two-electron terms are logged when the shortcut is called.

The atomic-mean-field integrals are provided by the optional ``x2camf``
package (see :doc:`../install`); ``spinor_hf`` imports without it, and the
dependency is only needed when the shortcut is invoked.

Two-electron SOC terms
----------------------

Both shortcuts accept the same keyword arguments controlling which two-electron
relativistic corrections enter the mean field:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Keyword
     - Default
     - Meaning
   * - ``with_gaunt``
     - ``True``
     - include the Gaunt (spin-other-orbit) term
   * - ``with_breit``
     - ``True``
     - include the full Breit interaction (Gaunt + gauge)
   * - ``with_pcc``
     - ``False``
     - include the two-electron picture-change correction
   * - ``with_aoc``
     - ``False``
     - average-of-configuration treatment of the atomic reference

For example, a Dirac-Coulomb-only spin-orbit Hamiltonian:

.. code-block:: python

   mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)

Constructing the helper directly
--------------------------------

The shortcuts cover the common case.  If you need to set options that the
shortcuts do not expose, build the helper yourself and assign it to
``with_x2c``:

.. code-block:: python

   from socutils.scf import spinor_hf
   from socutils.somf.x2cmp import SpinorX2CMPHelper

   mf = spinor_hf.SCF(mol)
   mf.with_x2c = SpinorX2CMPHelper(mol, x2cmp='x2camf',
                                   with_gaunt=True, with_breit=True,
                                   with_pcc=False, with_aoc=False)
   e = mf.kernel()

GHF (spin-orbital) path
-----------------------

The same Hamiltonian can be used with PySCF's generalized Hartree-Fock by
attaching the spin-orbital helper to an X2C-wrapped GHF object:

.. code-block:: python

   from pyscf import scf
   from socutils.somf.x2cmp import SpinOrbitalX2CMPHelper

   gmf = scf.GHF(mol).x2c()
   gmf.with_x2c = SpinOrbitalX2CMPHelper(mol, x2cmp='x2camf',
                                         with_gaunt=True, with_breit=True)
   e = gmf.kernel()

This reproduces the spinor result to numerical precision -- the spinor and
spin-orbital pictures describe the same Hamiltonian.

Caching the integrals
----------------------

Building the X2C spin-orbit core is the expensive one-time step of a
calculation.  The helper can dump the assembled integrals to a checkpoint and
reload them on a later run with ``mf.with_x2c.save_hcore(filename)`` and
``load_hcore(filename)`` (default ``x2cmp.chk``); the ``x2camf`` flavor
additionally caches its per-atom mean fields (``amf.chk``) so repeated
calculations on the same atoms reuse them.

Deprecated helper
-----------------

``somf.amf.SpinorX2CAMFHelper`` is the original atomic-mean-field helper.  It
is numerically identical to ``SpinorX2CMPHelper(x2cmp='x2camf')`` and is now
deprecated: constructing it emits a ``DeprecationWarning``.  Use the
``.x2camf()`` shortcut, or ``SpinorX2CMPHelper`` directly, instead.
