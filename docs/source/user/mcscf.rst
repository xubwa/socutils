Multiconfigurational SCF (mcscf)
================================

socutils provides CASCI and CASSCF on a two-component reference -- a spinor
mean field (``spinor_hf``) or a GHF object (``ghf.GHF``).  Because the
reference is two-component, the active space is counted in **spinor
(spin-orbital) orbitals**: ``ncas`` active spinors holding ``nelecas``
electrons, with ``nelecas`` no larger than ``ncas``.

CASCI
-----

``zcasci.CASCI(mf, ncas, nelecas, ncore=None)`` runs a complete-active-space
configuration interaction on top of a converged mean field.  ``kernel()``
returns ``(e_tot, e_cas, ci, ...)`` and stores ``mc.e_tot`` / ``mc.e_cas`` /
``mc.ci``.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zcasci

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SCF(mol).x2camf()
   mf.kernel()

   mc = zcasci.CASCI(mf, 8, 6)   # 6 electrons in 8 active spinors
   mc.kernel()
   print(mc.e_tot, mc.e_cas)

The default CI solver is socutils' own ``fci.FCISolver`` (see
`Full CI and selected CI`_ below).  Useful attributes:

* ``ncore`` -- number of core (doubly counted) orbitals; inferred from the
  electron count if not given;
* ``frozen`` -- orbitals to keep frozen;
* ``natorb`` -- transform the active space to natural orbitals;
* ``canonicalization`` -- canonicalize the core/external blocks (default
  ``True``);
* ``fcisolver`` -- the CI solver, which can be replaced (see below).

CASSCF
------

``zmcscf.CASSCF(mf, ncas, nelecas, ncore=None, frozen=None)`` additionally
optimizes the orbitals.  ``kernel()`` drives a **super-CI** orbital optimizer:
each macro-iteration solves the active-space CI problem, builds the orbital
gradient and an approximate Hessian, and takes a Kramers-paired orbital
rotation step, repeating until the energy and orbital gradient are converged.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zmcscf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   # the orbital optimizer builds its integrals from a density-fitted
   # reference, so the mean field must be density-fitted (or .cholesky())
   mf = spinor_hf.SCF(mol).x2camf().density_fit()
   mf.kernel()

   mc = zmcscf.CASSCF(mf, 8, 6)   # 6 electrons in 8 active spinor orbitals
   mc.kernel()
   print(mc.e_tot)

Requirements
~~~~~~~~~~~~

* **zquatev** -- the orbital step is solved with the Kramers-paired
  (quaternion) eigensolver, so the bundled ``zquatev`` solver must be built
  (see :doc:`../install`); ``kernel()`` raises a clear error if it is missing.
* **a density-fitted reference** -- the optimizer builds its two-electron
  integrals by Cholesky/DF transformation from ``mf``, so the mean field must
  carry a ``with_df``: attach it with ``.density_fit()`` or ``.cholesky()``
  (otherwise ``kernel()`` raises ``Either with_df or cderi must be provided``).
  See :ref:`the Cholesky decomposition section <cholesky-decomposition>` for the
  CD route and its on-disk caching.

Options
~~~~~~~

The optimization is controlled by attributes set on the ``CASSCF`` object
(defaults in parentheses):

* ``max_cycle_macro`` (``20``) -- maximum number of macro-iterations;
* ``max_stepsize`` (``0.2``) -- trust radius capping each orbital-rotation step;
* ``conv_tol`` (``1e-8``) -- energy convergence threshold;
* ``conv_tol_grad`` (``None`` → ``sqrt(conv_tol)`` ``= 1e-4``) -- orbital-gradient
  convergence threshold;
* ``natorb`` (``True``) -- at each macro-iteration rotate the active orbitals to
  natural orbitals (eigenvectors of the active 1-RDM, ordered by descending
  occupation);
* ``canonicalize_`` (``True``) -- diagonalize the core and virtual blocks of the
  effective Fock matrix so the inactive/virtual orbitals come out canonical;
* ``frozen`` (``None``) -- orbitals excluded from rotation; an ``int`` freezes the
  lowest ``frozen`` orbitals, a list/array freezes the listed indices;
* ``freeze_pair`` (``None``) -- a pair of index sets ``(set_i, set_j)`` whose
  mutual rotations are frozen (the rest are still optimized);
* ``irrep`` (``None``) -- per-orbital symmetry labels; rotations are then allowed
  only between orbitals carrying the same label.

Convergence and results
~~~~~~~~~~~~~~~~~~~~~~~~~~

A macro-iteration is accepted as converged when **both** the energy change and
the orbital-gradient norm fall below their thresholds
(``abs(dE) < conv_tol`` and ``norm(grad) < conv_tol_grad``); otherwise the loop
stops at ``max_cycle_macro``.  ``kernel()`` returns
``(e_tot, e_cas, ci, mo_coeff, mo_energy)`` and sets the attributes

* ``mc.e_tot`` -- total CASSCF energy;
* ``mc.e_cas`` -- active-space (CI) energy;
* ``mc.ci`` -- the active-space CI vector;
* ``mc.mo_coeff`` / ``mc.mo_energy`` -- optimized (canonical) orbitals and their
  energies;
* ``mc.converged`` -- whether both convergence criteria were met.

.. note::

   For tightly-bound cases the default ``max_cycle_macro = 20`` can stop a step
   or two before ``abs(dE) < 1e-8`` even though the gradient is already below
   ``conv_tol_grad`` (so ``mc.converged`` is ``False`` while the energy is
   essentially converged).  Raise ``mc.max_cycle_macro`` (e.g. to ``40``) to
   reach full convergence.

Full CI and selected CI
-----------------------

socutils ships its own spinor CI module, ``socutils.fci``, which is the
recommended (and default) CI solver.  It is a drop-in replacement for both
PySCF's ``fci_dhf_slow`` and the Dice-based SHCI interface (``socutils.hci``).

``fci.FCISolver`` (alias ``fci.FCI``)
    Exact full CI by direct construction and diagonalization of the
    Hamiltonian in the determinant basis.  All roots come from a single
    diagonalization, which avoids the Davidson convergence problems that the
    Kramers-degenerate roots cause for iterative solvers.  This is the default
    ``mc.fcisolver``; set ``mc.fcisolver.nroots`` for several states.

    .. code-block:: python

       from socutils.mcscf import zcasci
       from socutils.fci import zfci

       mc = zcasci.CASCI(mf, 8, 6)
       mc.fcisolver = zfci.FCISolver(mol)   # (this is also the default)
       mc.fcisolver.nroots = 4
       mc.kernel()
       print(mc.fcisolver.eci)              # the individual root energies

``fci.SelectedCI(mol, occslst=...)``
    Diagonalization in a chosen list of determinants (the replacement for the
    SHCI interface).  Combined with ``zfci.gen_ras_occslst`` it also expresses
    RASCI-type determinant spaces.

The companion ``fci.addons`` module provides post-processing for any solver
that exposes ``trans_rdm1`` (``FCISolver`` and ``SelectedCI``): transition
dipoles, oscillator strengths, Einstein coefficients / radiative lifetimes,
and ``spin_square`` / ``angular_momentum_square`` for analysing states.  See
the ``examples/fci`` directory.

Configuration-averaged solvers
-------------------------------

``zcahf`` provides configuration-averaged solvers that return averaged density
matrices instead of solving a CI -- useful for averaging over a degenerate
open shell:

``zcahf.CAHF(mol)``
    Configuration-averaged Hartree-Fock: spreads the active electrons evenly
    over the active orbitals (a single averaged configuration).

    .. code-block:: python

       from socutils.mcscf import zcasci, zcahf

       mc = zcasci.CASCI(mf, 8, 6)
       mc.fcisolver = zcahf.CAHF(mol)
       mc.kernel()

``zcahf.MultiSlater(mol, det_list, weight_list)``
    Average over an explicit list of Slater determinants with given weights.

``zcahf.MultiZCAHF(mol, orb_open, elec_open)``
    Configuration averaging over multiple open shells, specified by the
    open-shell orbital and electron counts.
