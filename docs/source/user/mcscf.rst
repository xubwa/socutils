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
optimizes the orbitals.  ``kernel()`` drives the super-CI orbital optimizer
directly, alternating CI and orbital steps until convergence:

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zmcscf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   # the orbital optimizer builds its integrals from a density-fitted
   # reference, so the mean field must be density-fitted
   mf = spinor_hf.SCF(mol).x2camf().density_fit()
   mf.kernel()

   mc = zmcscf.CASSCF(mf, 8, 6)
   mc.kernel()
   print(mc.e_tot)

The orbital optimizer uses a Kramers-paired eigensolver and therefore requires
the optional ``zquatev`` library (see :doc:`../install`); ``kernel()`` raises a
clear error if it is missing.  It builds its two-electron integrals from a
density-fitted (Cholesky) reference, so the mean field must be density-fitted
with ``.density_fit()``.  Options controlling the optimization include:

* ``frozen`` -- frozen orbitals (int or list);
* ``max_stepsize`` -- trust radius for the orbital rotation (default ``0.2``);
* ``conv_tol`` / ``conv_tol_grad`` -- energy / gradient convergence;
* ``freeze_pair`` -- freeze rotations between two specified orbital sets;
* ``irrep`` -- restrict rotations to within matching irrep labels;
* ``natorb`` / ``canonicalize_`` -- natural-orbital / canonicalization
  post-processing (both default ``True``).

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
