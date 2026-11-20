#!/usr/bin/env python
'''
AVAS: automated construction of CASCI/CASSCF active spaces for a
two-component (j-adapted spinor) reference.
Ref. JCTC 13, 4063 (2017); arXiv:1701.07862

Adapted from pyscf.mcscf.avas.  Differences from the scalar version:

- All overlaps are spinor overlaps (``int1e_ovlp_spinor``); the AO projector
  is complex Hermitian.
- Each spinor orbital holds one electron, so there is no factor of two in the
  electron count and no alpha/beta distinction -- the ``openshell_option``
  machinery of the scalar implementation is not needed.
- Reference AOs are selected by matching ``spinor_labels()`` (e.g. the
  pattern 'Cr 3d' matches all 3d3/2 and 3d5/2 spinors).
- ``ncore`` counts core *spinor* orbitals (= number of core electrons).

For a Kramers-restricted reference the projector eigenvalues come in
degenerate time-reversal pairs, so the selected active space automatically
closes under Kramers symmetry; a warning is issued if the threshold cuts
through a (near-)degenerate pair.

Examples:

>>> from pyscf import gto
>>> from socutils.scf import spinor_hf
>>> from socutils.mcscf import avas, zcasci
>>> mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz')
>>> mf = spinor_hf.SCF(mol).x2camf().run()
>>> ncas, nelecas, mo = avas.avas(mf, ['F 2p', 'H 1s'])
>>> mc = zcasci.CASCI(mf, ncas, nelecas)
>>> mc.kernel(mo)
'''

from functools import reduce
import re
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.lo.iao import reference_mol

THRESHOLD = 0.2
MINAO = 'minao'
WITH_IAO = False
CANONICALIZE = True


def search_spinor_label(mol_or_labels, labels):
    '''Indices of spinor basis functions whose ``spinor_labels()`` entry (or
    entry of an explicit label list) matches any of the given regular
    expression patterns.  Spinor labels look like "0 Cr 3d3/2,-1/2", so a
    scalar-style pattern such as 'Cr 3d' matches both the j=3/2 and j=5/2
    spinors of the shell.'''
    if isinstance(mol_or_labels, (list, tuple, numpy.ndarray)):
        cand = mol_or_labels
    else:
        cand = mol_or_labels.spinor_labels()
    if isinstance(labels, str):
        labels = [labels]
    pats = [re.compile(re.sub(' +', ' ', x.strip(), count=1)) for x in labels]
    baslst = [i for i, s in enumerate(cand)
              if any(re.search(p, s) for p in pats)]
    return numpy.asarray(baslst, dtype=int)


def _warn_kramers_split(log, w, threshold, name, degen_tol=1e-6):
    wsort = numpy.sort(w)
    icut = numpy.searchsorted(wsort, threshold)
    if 0 < icut < wsort.size and wsort[icut] - wsort[icut-1] < degen_tol:
        log.warn('AVAS threshold %g splits a degenerate (Kramers) pair in the '
                 '%s block: eigenvalues %.8g / %.8g. The active space may not '
                 'be closed under time reversal; consider adjusting the '
                 'threshold.', threshold, name, wsort[icut-1], wsort[icut])


def kernel(mf, aolabels, threshold=THRESHOLD, minao=MINAO, with_iao=WITH_IAO,
           canonicalize=CANONICALIZE, ncore=0, verbose=None):
    '''AVAS active space construction on a spinor mean-field object.

    Args:
        mf : a :class:`SpinorSCF` (or derived, e.g. x2camf) object

        aolabels : string or a list of strings
            Reference AO labels, matched against ``spinor_labels()``.

    Kwargs:
        threshold : float
            Truncation threshold of the AO-projector eigenvalues above which
            orbitals are kept in the active space.
        minao : str or (c_ref, labels) tuple
            Reference minimal AO basis for AVAS.  Either a basis name, or a
            tuple of reference orbital coefficients in the molecular spinor
            AO basis plus their labels -- e.g. occupied atomic x2camf spinors
            from :func:`socutils.lo.atomic_ref.atomic_minao` (recommended for
            heavy elements, where the tabulated 'minao' lacks core functions
            and relativistic contractions).
        with_iao : bool
            Whether to use IAOs (built from the occupied orbitals) as the
            reference active AOs instead of raw minao AOs.
        canonicalize : bool
            Diagonalize the Fock matrix within the core, active and virtual
            subspaces.
        ncore : integer
            Number of core spinor orbitals excluded from the AVAS procedure.

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess for
        CASCI/CASSCF (all sizes in spinor orbitals)
    '''
    avas_obj = AVAS(mf, aolabels, threshold, minao, with_iao,
                    canonicalize, ncore, verbose)
    return avas_obj.kernel()


def _kernel(avas_obj):
    mf = avas_obj._scf
    mol = mf.mol
    log = logger.new_logger(avas_obj)
    log.info('\n** AVAS (spinor) **')

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    nao = mol.nao_2c()
    if mo_coeff.shape[0] != nao:
        raise RuntimeError('avas expects a spinor mean-field object '
                           '(mo_coeff of dimension nao_2c); got shape %s'
                           % str(mo_coeff.shape))

    ncore = avas_obj.ncore
    nocc = numpy.count_nonzero(mo_occ != 0)
    ovlp = mol.intor('int1e_ovlp_spinor')
    log.info('  Total number of spinor MOs is equal to  %d', mo_coeff.shape[1])
    log.info('  Number of occupied spinor MOs is equal to  %d', nocc)

    if isinstance(avas_obj.minao, str):
        pmol = reference_mol(mol, avas_obj.minao)
        ref_labels = pmol.spinor_labels()
        s12 = gto.intor_cross('int1e_ovlp_spinor', mol, pmol)
        s2ref = pmol.intor('int1e_ovlp_spinor')
        minao_name = avas_obj.minao
    else:
        # reference AOs given explicitly in the molecular spinor AO basis,
        # e.g. occupied atomic x2camf spinors from lo.atomic_ref.atomic_minao
        c_ref, ref_labels = avas_obj.minao
        s12 = numpy.dot(ovlp, c_ref)
        s2ref = reduce(numpy.dot, (c_ref.conj().T, ovlp, c_ref))
        minao_name = 'user-defined reference orbitals'

    baslst = search_spinor_label(ref_labels, avas_obj.aolabels)
    log.info('reference spinor AO indices for %s %s:\n %s',
             minao_name, avas_obj.aolabels, baslst)
    if len(baslst) == 0:
        raise RuntimeError('AVAS reference AO labels %s match nothing in the '
                           '%s reference' % (avas_obj.aolabels, minao_name))

    if avas_obj.with_iao:
        from socutils.lo.ibo import get_iao_fromS
        c = get_iao_fromS(ovlp, s2ref, s12, mo_coeff[:, ncore:nocc],
                          orthogonalize=False)[:, baslst]
        s2 = reduce(numpy.dot, (c.conj().T, ovlp, c))
        s21 = reduce(numpy.dot, (c.conj().T, ovlp, mo_coeff[:, ncore:]))
    else:
        s2 = s2ref[baslst][:, baslst]
        s21 = numpy.dot(s12[:, baslst].conj().T, mo_coeff[:, ncore:])
    sa = numpy.dot(s21.conj().T, scipy.linalg.solve(s2, s21, assume_a='pos'))

    threshold = avas_obj.threshold
    wocc, u = numpy.linalg.eigh(sa[:(nocc-ncore), :(nocc-ncore)])
    log.info('threshold %s', threshold)
    ncas_occ = int((wocc > threshold).sum())
    nelecas = (nocc - ncore) - int((wocc < threshold).sum())
    mocore = mo_coeff[:, ncore:nocc].dot(u[:, wocc < threshold])
    mocas = mo_coeff[:, ncore:nocc].dot(u[:, wocc >= threshold])
    _warn_kramers_split(log, wocc, threshold, 'occupied')

    wvir, u = numpy.linalg.eigh(sa[(nocc-ncore):, (nocc-ncore):])
    ncas_vir = int((wvir > threshold).sum())
    mocas = numpy.hstack((mocas, mo_coeff[:, nocc:].dot(u[:, wvir >= threshold])))
    movir = mo_coeff[:, nocc:].dot(u[:, wvir < threshold])
    ncas = mocas.shape[1]
    _warn_kramers_split(log, wvir, threshold, 'virtual')

    occ_weights = numpy.hstack([wocc[wocc < threshold], wocc[wocc >= threshold]])
    vir_weights = numpy.hstack([wvir[wvir >= threshold], wvir[wvir < threshold]])

    log.debug('projected occ eig %s', occ_weights)
    log.debug('projected vir eig %s', vir_weights)
    log.info('Active from occupied = %d , eig %s', ncas_occ,
             occ_weights[occ_weights >= threshold])
    log.info('Inactive from occupied = %d', mocore.shape[1])
    log.info('Active from unoccupied = %d , eig %s', ncas_vir,
             vir_weights[vir_weights >= threshold])
    log.info('Inactive from unoccupied = %d', movir.shape[1])
    log.info('Dimensions of active space %d', ncas)
    log.info('# of active electrons %d', nelecas)

    mofreeze = mo_coeff[:, :ncore]
    if avas_obj.canonicalize:
        def trans(c):
            if c.shape[1] == 0:
                return c
            csc = reduce(numpy.dot, (c.conj().T, ovlp, mo_coeff))
            fock = numpy.dot(csc * mo_energy, csc.conj().T)
            e, u = scipy.linalg.eigh(fock)
            return numpy.dot(c, u)
        if ncore > 0:
            mofreeze = trans(mofreeze)
        mocore = trans(mocore)
        mocas = trans(mocas)
        movir = trans(movir)
    mo = numpy.hstack((mofreeze, mocore, mocas, movir))
    return ncas, nelecas, mo, occ_weights, vir_weights


avas = kernel


@lib.with_doc(kernel.__doc__)
class AVAS(lib.StreamObject):
    def __init__(self, mf, aolabels, threshold=THRESHOLD, minao=MINAO,
                 with_iao=WITH_IAO, canonicalize=CANONICALIZE, ncore=0,
                 verbose=None):
        self._scf = mf
        self.aolabels = aolabels
        self.threshold = threshold
        self.minao = minao
        self.with_iao = with_iao
        self.canonicalize = canonicalize
        self.ncore = ncore
        self.stdout = mf.stdout
        self.verbose = verbose or mf.verbose

##################################################
# don't modify the following attributes, they are not input options
        self.ncas = None
        self.nelecas = None
        # Orbitals of the entire space: cores, actives, externals
        self.mo_coeff = None
        # occ_weights and vir_weights to filter active and inactive orbitals
        self.occ_weights = None
        self.vir_weights = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** AVAS flags ********')
        log.info('aolabels = %s', self.aolabels)
        log.info('ncore = %s', self.ncore)
        log.info('minao = %s', self.minao)
        log.info('threshold = %s', self.threshold)
        log.info('with_iao = %s', self.with_iao)
        log.info('canonicalize = %s', self.canonicalize)
        return self

    def kernel(self):
        self.dump_flags()
        self.ncas, self.nelecas, self.mo_coeff, \
                self.occ_weights, self.vir_weights = _kernel(self)
        return self.ncas, self.nelecas, self.mo_coeff
