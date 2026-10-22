'''Non-relativistic (spin-restricted) UC- and PC-NEVPT2.

Real spatial orbitals with a closed-shell core, determinants |I_alpha I_beta>
in the order of pyscf.fci, sectors named by (n_alpha, n_beta).  The perturber
classes are those of the spinor code resolved by spin (zmr.nr.classes); the
contraction engines and the multi-state drivers are shared with it.

    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2

    eris, nelec = pyscf_nr.nr_from_casci(mc)        # nelec = (n_alpha, n_beta)
    pt = PCNEVPT2(eris, nelec, civecs=pyscf_nr.nr_civecs(mc))
    pt.kernel()
'''

from socutils.zmr.nr.nevpt2 import PCNEVPT2, UCNEVPT2  # noqa: F401
