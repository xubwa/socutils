#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Double-group symmetry for spinor (2c) and Dirac (4c) mean-field methods.

The symmetry logic lives in module functions (:func:`eig`, :func:`get_occ`,
:func:`label_orb_symm`), which only take matrices, coefficients and labels and
can be called from MCSCF/CC/property code.  :class:`DoubleGroupSymm` is a thin
mixin that feeds them from the SCF object; it is combined with the 2c classes
of :mod:`socutils.scf.spinor_hf` and the 4c classes of :mod:`pyscf.scf.dhf`:

    SymAdaptedUHF   = DoubleGroupSymm + spinor_hf.SpinorSCF
    SymAdaptedRHF   = DoubleGroupSymm + spinor_hf.KRHF      (Kramers restricted)
    SymAdaptedDHF   = DoubleGroupSymm + dhf.DHF
    SymAdaptedRDHF  = DoubleGroupSymm + dhf.RDHF            (Kramers restricted)

The molecule must be in its symmetry frame (see
:func:`socutils.symm.to_symm_frame`); the orbitals, their irreps and all
properties refer to that frame.

Kramers restriction with symmetry: for a pair of complex-conjugate irreps only
the first block is diagonalized; the partner block is spanned by the time
reverses of the first block's functions, its matrix is the complex conjugate,
and its orbitals are the time reverses of the first block's orbitals.
Self-conjugate blocks (C1, Ci) are diagonalized with the quaternion solver
zquatev, in the [X | T X] layout of the symmetry-adapted basis.
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import dhf
from pyscf.scf import chkfile

from socutils.symm import double_group as dg
from socutils.symm import spinor_basis
from socutils.scf import spinor_hf


def _kramers_eigh(h, s):
    from socutils.lib import zquatev
    return zquatev.geigh(h, s)


def eig(h, s, so, irrep_id, groupname, kramers=False):
    '''Solve hc = sce block by block in the symmetry-adapted basis ``so``.

    Eigenvalues are grouped by irrep (in the order of ``irrep_id``), not
    sorted.  The returned coefficients carry ``orbsym``.  With ``kramers``,
    the second block of each conjugate pair is obtained by time reversal
    (requires so[partner] = T so[first], as built by
    :func:`socutils.symm.spinor_basis.symm_adapted_basis_spinor`, and a
    time-reversal symmetric ``h``).'''
    irrep_id = list(irrep_id)
    es, cs, orbsym = [], [], []
    solved = {}
    for ir, x in zip(irrep_id, so):
        partner = dg.conj(ir, groupname)
        if kramers and partner != ir and partner in solved:
            e, c = solved[partner]
            c = c.conj()
        else:
            hx = lib.dot(x.conj().T, lib.dot(h, x))
            sx = lib.dot(x.conj().T, lib.dot(s, x))
            if kramers and partner == ir:
                e, c = _kramers_eigh(hx, sx)
            else:
                e, c = scipy.linalg.eigh(hx, sx)
            solved[ir] = (e, c)
        es.append(e)
        cs.append(lib.dot(x, c))
        orbsym.append(numpy.full(e.size, ir))
    e = numpy.hstack(es)
    c = lib.tag_array(numpy.hstack(cs), orbsym=numpy.hstack(orbsym))
    return e, c


def label_orb_symm(mo_coeff, so, irrep_id, thresh=1e-6):
    '''Irrep of each orbital, from the weight of its coefficients in each
    block of the (unitary) symmetry-adapted basis.'''
    weights = numpy.array([(abs(lib.dot(x.conj().T, mo_coeff))**2).sum(axis=0) for x in so])
    best = numpy.argmax(weights, axis=0)
    total = weights.sum(axis=0)
    purity = weights[best, numpy.arange(mo_coeff.shape[1])] / total
    if (1 - purity).max() > thresh:
        raise ValueError('Orbitals are not symmetry adapted (max deviation %.2e)'
                         % (1 - purity).max())
    return numpy.asarray(irrep_id)[best]


def get_occ(mo_energy, orbsym, nelec, irrep_nelec, groupname, mask=None, kramers=False):
    '''Occupation numbers.  ``irrep_nelec`` fixes the electron count of the
    named irreps; the remaining electrons fill the lowest allowed orbitals of
    the other irreps.  ``mask`` marks the allowed orbitals (DHF: the
    positive-energy states).'''
    mo_occ = numpy.zeros(mo_energy.size)
    if mask is None:
        mask = numpy.ones(mo_energy.size, dtype=bool)
    fixed = {dg.irrep_name2id(groupname, k): v for k, v in irrep_nelec.items()}
    if kramers:
        for ir, n in list(fixed.items()):
            partner = dg.conj(ir, groupname)
            if partner == ir:
                if n % 2:
                    raise ValueError('Kramers-restricted occupation of the self-conjugate irrep %s '
                                     'must be even' % dg.irrep_id2name(groupname, ir))
            elif fixed.setdefault(partner, n) != n:
                raise ValueError('Kramers partners %s and %s need the same occupation'
                                 % (dg.irrep_id2name(groupname, ir),
                                    dg.irrep_id2name(groupname, partner)))
    nfix = 0
    rest = mask.copy()
    for ir, n in fixed.items():
        idx = numpy.where(mask & (orbsym == ir))[0]
        if n > idx.size:
            raise ValueError('irrep %s has only %d orbitals for %d electrons'
                             % (dg.irrep_id2name(groupname, ir), idx.size, n))
        order = numpy.argsort(mo_energy[idx].round(9), kind='stable')
        mo_occ[idx[order[:n]]] = 1
        rest[orbsym == ir] = False
        nfix += n
    nfloat = nelec - nfix
    if nfloat < 0:
        raise ValueError('irrep_nelec assigns %d electrons, molecule has %d' % (nfix, nelec))
    idx = numpy.where(rest)[0]
    order = numpy.argsort(mo_energy[idx].round(9), kind='stable')
    if nfloat > idx.size:
        raise ValueError('Not enough orbitals for %d electrons' % nfloat)
    mo_occ[idx[order[:nfloat]]] = 1
    return mo_occ


def irrep_occupation(orbsym, mo_occ, irrep_id):
    return [int(round(mo_occ[orbsym == ir].sum())) for ir in irrep_id]


class DoubleGroupSymm:
    '''Mixin adding double-group symmetry to a spinor or Dirac SCF class.

    Attributes:
        irrep_nelec : dict
            Number of electrons of particular irreps, {'1Eg': 4, ...}.  With
            Kramers restriction the partner irrep gets the same number if it
            is not given.  Irreps not listed are filled by orbital energy.
    '''

    _keys = {'irrep_nelec'}
    _kramers_restricted = False

    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.irrep_nelec = {}

    # hooks; defaults are the 2c case
    def _dg_symm_orb(self):
        return spinor_basis.get_symm_orb(self.mol, '2c')

    def _occ_mask(self, mo_energy):
        return numpy.ones(mo_energy.size, dtype=bool)

    @property
    def doublegroupname(self):
        spinor_basis.get_symm_orb(self.mol)
        return self.mol.doublegroupname

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        mol = self.mol
        log.info('double group %s (point group %s), molecule in its symmetry frame',
                 self.doublegroupname, mol.groupname)
        frame = getattr(mol, 'symm_frame', None)
        if frame is not None:
            log.info('symmetry frame: r = R (r_input - orig), orig = %s, R = %s', frame[0], frame[1])
        log.info('vectors, tensors and m_j labels refer to this frame')
        if self.irrep_nelec:
            log.info('irrep_nelec %s', self.irrep_nelec)
        return self

    def build(self, mol=None):
        if mol is None:
            mol = self.mol
        spinor_basis.check_symm_frame(mol)
        spinor_basis.get_symm_orb(mol)
        names = set(mol.irrep_name_spinor)
        for name in self.irrep_nelec:
            if name not in names:
                raise ValueError('irrep %s not in double group %s (irreps: %s)'
                                 % (name, mol.doublegroupname, mol.irrep_name_spinor))
        return super().build(mol)

    def eig(self, h, s=None, *args, **kwargs):
        if s is None:
            # MO-subspace diagonalization: no symmetry information here
            return super().eig(h, s, *args, **kwargs)
        so, irrep_id = self._dg_symm_orb()
        return eig(h, s, so, irrep_id, self.doublegroupname, kramers=self._kramers_restricted)

    def get_orbsym(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        orbsym = getattr(mo_coeff, 'orbsym', None)
        if orbsym is not None:
            return numpy.asarray(orbsym)
        so, irrep_id = self._dg_symm_orb()
        return label_orb_symm(mo_coeff, so, irrep_id)

    @property
    def orbsym(self):
        return self.get_orbsym(self.mo_coeff)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        orbsym = self.get_orbsym(mo_coeff)
        gname = self.doublegroupname
        mask = self._occ_mask(mo_energy)
        mo_occ = get_occ(mo_energy, orbsym, self.mol.nelectron, self.irrep_nelec, gname,
                         mask=mask, kramers=self._kramers_restricted)
        if self.verbose >= logger.INFO:
            so, irrep_id = self._dg_symm_orb()
            names = [dg.irrep_id2name(gname, ir) for ir in irrep_id]
            occ = mo_occ > 0
            vir = mask & ~occ
            noccs = irrep_occupation(orbsym, mo_occ, irrep_id)
            logger.info(self, 'irrep occupation  ' + ' '.join('%s:%d' % x for x in zip(names, noccs)))
            if vir.any():
                ihomo = numpy.where(occ)[0][numpy.argmax(mo_energy[occ])]
                ilumo = numpy.where(vir)[0][numpy.argmin(mo_energy[vir])]
                logger.info(self, 'HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                            dg.irrep_id2name(gname, orbsym[ihomo]), mo_energy[ihomo],
                            dg.irrep_id2name(gname, orbsym[ilumo]), mo_energy[ilumo])
                if mo_energy[ihomo] > mo_energy[ilumo] + 1e-9:
                    logger.warn(self, 'HOMO %.15g > LUMO %.15g', mo_energy[ihomo], mo_energy[ilumo])
        return mo_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        g = super().get_grad(mo_coeff, mo_occ, fock)
        orbsym = self.get_orbsym(mo_coeff)
        occidx = mo_occ > 0
        viridx = ~occidx
        forbid = orbsym[viridx].reshape(-1, 1) != orbsym[occidx]
        g = numpy.array(g, copy=True)
        if g.size == forbid.size:
            g[forbid.ravel()] = 0
        return g

    def _finalize(self):
        super()._finalize()
        orbsym = self.get_orbsym(self.mo_coeff)
        mask = self._occ_mask(self.mo_energy)
        occ = self.mo_occ > 0
        groups = [numpy.where(~mask)[0], numpy.where(occ)[0], numpy.where(mask & ~occ)[0]]
        order = numpy.hstack([g[numpy.argsort(self.mo_energy[g].round(9), kind='stable')]
                              for g in groups]).astype(int)
        self.mo_energy = self.mo_energy[order]
        self.mo_occ = self.mo_occ[order]
        self.mo_coeff = lib.tag_array(numpy.asarray(self.mo_coeff)[:, order], orbsym=orbsym[order])
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self

    def analyze(self, verbose=None, **kwargs):
        if verbose is None:
            verbose = self.verbose
        log = logger.new_logger(self, verbose)
        gname = self.doublegroupname
        orbsym = self.get_orbsym(self.mo_coeff)
        so, irrep_id = self._dg_symm_orb()
        names = [dg.irrep_id2name(gname, ir) for ir in irrep_id]
        noccs = irrep_occupation(orbsym, self.mo_occ, irrep_id)
        log.note('double group %s', gname)
        log.note('occupancy for each irrep:  ' + ' '.join('%s:%d' % x for x in zip(names, noccs)))
        count = {}
        for k, (e, ir, occ) in enumerate(zip(self.mo_energy, orbsym, self.mo_occ)):
            count[ir] = count.get(ir, 0) + 1
            log.note('MO #%-4d (%s #%d) energy= %.15g occ= %g',
                     k + 1, dg.irrep_id2name(gname, ir), count[ir], e, occ)
        return noccs


class SymAdaptedUHF(DoubleGroupSymm, spinor_hf.SpinorSCF):
    '''Kramers-unrestricted spinor SCF with double-group symmetry.'''


class SymAdaptedRHF(DoubleGroupSymm, spinor_hf.KRHF):
    '''Kramers-restricted spinor SCF with double-group symmetry.'''
    _kramers_restricted = True


class _DiracSymm(DoubleGroupSymm):
    def _dg_symm_orb(self):
        return spinor_basis.get_symm_orb(self.mol, '4c')

    def _occ_mask(self, mo_energy):
        # positive-energy states (cf. pyscf.scf.dhf.DHF.get_occ)
        return mo_energy > -1.999 * lib.param.LIGHT_SPEED**2


class SymAdaptedDHF(_DiracSymm, dhf.DHF):
    '''Dirac-Hartree-Fock with double-group symmetry.'''


class SymAdaptedRDHF(_DiracSymm, dhf.RDHF):
    '''Kramers-restricted Dirac-Hartree-Fock with double-group symmetry.'''
    _kramers_restricted = True
