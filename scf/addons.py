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


def spinor_ao_indices(mol, atoms, l=None):
    '''Indices of the (j-adapted spinor) basis functions on the given atom
    indices, optionally only of angular momentum l.'''
    atoms = set(numpy.atleast_1d(atoms).tolist())
    loc = mol.ao_loc_2c()
    idx = [numpy.arange(loc[sh], loc[sh + 1]) for sh in range(mol.nbas)
           if mol.bas_atom(sh) in atoms and (l is None or mol.bas_angular(sh) == l)]
    return numpy.concatenate(idx) if idx else numpy.zeros(0, dtype=int)


def subspace_weights(s, mo_coeff, idx):
    '''Weight of each MO in the span of the basis functions idx:
    w_i = c_i^+ S[:, F] S_FF^-1 S[F, :] c_i (1 for an MO inside the span).'''
    sf = s[:, idx]
    x = numpy.linalg.solve(s[numpy.ix_(idx, idx)], sf.conj().T @ mo_coeff)
    return numpy.einsum('pi,pi->i', (sf @ x).conj(), mo_coeff).real


def projected_frac_occ(mf, idx, norb_open, nelec_open, w_min=.5):
    '''Even fractional occupation of an open shell chosen by character: among
    the MOs whose weight in the span of the basis functions idx (e.g. the f
    functions of a dopant, :func:`spinor_ao_indices`) is at least w_min, the
    norb_open lowest in energy; the remaining mol.nelectron - nelec_open
    electrons by aufbau among the other MOs.  The span of all f functions also
    contains the virtual f-polarisation orbitals (weight ~1, high energy), so
    the energy order among the candidates picks the valence shell, as AVAS
    does with a minimal atomic basis.  With fewer than norb_open candidates
    the largest weights are taken (with a warning).

    Unlike :func:`even_occ` the open shell is chosen by character, not by
    orbital energy, which matters for a 4f shell in a solid: HF puts the
    occupied 4f levels well below the ligand bands.  Same caveat as
    :func:`even_occ` (fractional occupation, not AOC/CAHF).  Kramers partners
    have equal weights and are selected together, so a Kramers-restricted
    solver keeps equal pair occupations.  The selected MO indices of the last
    call are stored in mf.open_shell_mo.
    '''
    mol = mf.mol
    nclose = mol.nelectron - nelec_open
    frac = nelec_open / norb_open
    idx = numpy.asarray(idx)

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        w = subspace_weights(mf.get_ovlp(), mo_coeff, idx)
        cand = numpy.where(w >= w_min)[0]
        if len(cand) >= norb_open:
            open_mo = numpy.sort(cand[numpy.argsort(mo_energy[cand], kind='stable')[:norb_open]])
        else:
            logger.warn(mf, 'projected_frac_occ: only %d MOs with weight >= %.2f, taking the %d largest weights',
                        len(cand), w_min, norb_open)
            open_mo = numpy.sort(numpy.argsort(-w, kind='stable')[:norb_open])
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[open_mo] = frac
        rest = numpy.array([i for i in numpy.argsort(mo_energy, kind='stable') if i not in set(open_mo)])
        mo_occ[rest[:nclose]] = 1.0
        mf.open_shell_mo = open_mo
        logger.info(mf, '  open shell: %d MOs x %.6f electrons, weights %.3f..%.3f, '
                    'energies %.4f..%.4f; %d candidates with weight >= %.2f',
                    norb_open, frac, w[open_mo].min(), w[open_mo].max(),
                    mo_energy[open_mo].min(), mo_energy[open_mo].max(), len(cand), w_min)
        e_occ = mo_energy[rest[:nclose]]
        if len(rest) > nclose:
            logger.info(mf, '  closed shell HOMO %.6f LUMO %.6f', e_occ.max(), mo_energy[rest[nclose]])
        return mo_occ

    mf.get_occ = get_occ
    return mf


def cas_order(mf, open_mo=None):
    '''MO coefficients reordered as [closed | open shell | empty] (each by
    energy) for a CASCI/CASSCF on the open shell of a fractional-occupation
    reference, and the number of closed MOs: zcasci.CASCI(mf, ncas, nelecas,
    ncore=ncore) with mo_coeff=mo.'''
    o, e = mf.mo_occ, mf.mo_energy
    if open_mo is None:
        open_mo = numpy.where((o > 1e-12) & (o < 1 - 1e-12))[0]
    open_mo = numpy.asarray(open_mo)
    closed = [i for i in numpy.argsort(e, kind='stable') if o[i] > 1 - 1e-12 and i not in set(open_mo)]
    empty = [i for i in numpy.argsort(e, kind='stable') if o[i] < 1e-12 and i not in set(open_mo)]
    order = numpy.concatenate([closed, open_mo[numpy.argsort(e[open_mo], kind='stable')], empty]).astype(int)
    return mf.mo_coeff[:, order], len(closed)
