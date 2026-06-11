#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Relaxed one-particle density matrix and one-electron properties for the
# vvvv-free two-component CCSD (socutils.cc.zccsd_direct.DirectZCCSD and
# socutils.cc.chol_zccsd.DFCCSD).
#
# The (complex, j-spinor) amplitudes solve the same spin-orbital equations as
# pyscf's GCCSD, so the relaxed 1-RDM is assembled with pyscf's gccsd_rdm /
# gccsd_t_rdm (fed the amplitudes directly) and returned in the spinor MO basis,
# or in the spinor-AO basis when ao_repr=True.  Building the 1-RDM needs only
# ovvv/ooov/oovv (no vvvv); ovvv is reconstructed once for the (T) density when
# the backend does not store it.
#
# Validated on a 2c spinor H2O build: the relaxed Tr(gamma h) is invariant under
# a per-orbital complex phase rotation (~5e-11) and the CCSD / CCSD(T) relaxed
# one-electron energies and dipoles agree with non-relativistic RCCSD / RCCSD(T)
# to ~1e-8.
#

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.cc import gccsd_rdm, gccsd_t_rdm

from socutils.cc.eom_zccsd_direct import get_ovvv


def _amps(mycc, t1, t2, l1, l2, eris, with_t):
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None or l2 is None:
        if getattr(mycc, 'l1', None) is None or getattr(mycc, 'l2', None) is None:
            mycc.solve_lambda(t1, t2, eris=eris, with_t=with_t)
        l1, l2 = mycc.l1, mycc.l2
    return t1, t2, l1, l2


def make_rdm1(mycc, t1=None, t2=None, l1=None, l2=None, eris=None,
              ao_repr=False, with_t=False):
    '''Relaxed CCSD (with_t=False) or CCSD(T) (with_t=True) 1-RDM.

    Returned in the spinor MO basis, or in the spinor-AO basis if ao_repr.
    For the (T) density l1/l2 must be the CCSD(T)-Lambda amplitudes
    (solve_lambda(..., with_t=True)); they are solved on demand if absent.
    '''
    t1, t2, l1, l2 = _amps(mycc, t1, t2, l1, l2, eris, with_t)
    if with_t:
        if getattr(eris, 'ovvv', None) is None:
            eris.ovvv = get_ovvv(eris)
        dm1 = gccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris)
    else:
        dm1 = gccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = mo.dot(dm1).dot(mo.conj().T)
    return dm1


def dip_moment(mycc, t1=None, t2=None, l1=None, l2=None, eris=None,
               with_t=False, unit='Debye'):
    '''Relaxed CCSD / CCSD(T) electric dipole moment (spinor AO integrals).'''
    dm_ao = make_rdm1(mycc, t1, t2, l1, l2, eris, ao_repr=True, with_t=with_t)
    mol = mycc.mol
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r_spinor')
    el = -np.einsum('xpq,qp->x', r, dm_ao).real
    nucl = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    dip = nucl + el
    if unit.upper() == 'DEBYE':
        dip *= nist.AU2DEBYE
    return dip
