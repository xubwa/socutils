#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
# Modified from https://github.com/pyscf/properties 
#               pyscf/prop/efg/rhf.py
# CAUTION: The SFX2C-1e EFG in the original implementation is NOT correct.
#          The correct implementation, as well as the response of X2C 
#          transformation, is available in this file.

'''
Electric field gradients for non-/scalar-relativistic methods.
(In testing)
'''

import warnings
import numpy as np
from pyscf import lib, scf
from functools import reduce

warnings.warn('Module EFG is under testing')

def _get_quad_nuc(mol, atm_id):
    mask = np.ones(mol.natm, dtype=bool)
    mask[atm_id] = False  # exclude the contribution from atm_id
    dr = mol.atom_coords()[mask] - mol.atom_coord(atm_id)
    d = np.linalg.norm(dr, axis=1)
    rr = 3*np.einsum('ix,iy->ixy', dr, dr)
    for i in range(3):
        rr[:,i,i] -= d**2
    z = mol.atom_charges()[mask]
    efg_nuc = np.einsum('i,ixy->xy', z/d**5, rr)
    return efg_nuc


def kernel(method, efg_nuc=None, dm=None, Xresp=True):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** EFG for non-relativistic methods (In testing) ********')
    mol = method.mol
    if efg_nuc is None:
        efg_nuc = range(mol.natm)

    if dm is None:
        dm = method.make_rdm1()
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        # UHF density matrix
        dm = dm[0] + dm[1]

    if isinstance(method, scf.hf.SCF):
        with_x2c = getattr(method, 'with_x2c', None)
    else:
        with_x2c = getattr(method._scf, 'with_x2c', None)

    if with_x2c:
        xmol, contr_coeff = with_x2c.get_xmol(mol)
        from socutils.somf import x2c_grad
        t = xmol.intor('int1e_kin')
        s = xmol.intor('int1e_ovlp')
        v = xmol.intor('int1e_nuc')
        w = xmol.intor('int1e_pnucp')
        a, e, xmat, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)

    log.info('\nElectric Field Gradient Tensor Results')
    efg = []
    for i, atm_id in enumerate(efg_nuc):
        if not with_x2c:
            h1 = _get_quadrupole_integrals(mol, atm_id)
        else:
            h1 = np.zeros((3,3,mol.nao,mol.nao), dtype=dm.dtype)
            h1_4c = _get_4csph_quadrupole_integrals(xmol, atm_id)
            for x in range(3):
                for y in range(x+1):
                    if Xresp:
                        h1_tmp = x2c_grad.get_hfw1(a, xmat, st, m4c, h4c, e, r, l, h1_4c[x,y])
                    else:
                        from socutils.somf.eamf import to_2c
                        h1_tmp = to_2c(xmat, r, h1_4c[x,y])
                    h1[x,y] = reduce(np.dot, (contr_coeff.T.conj(), h1_tmp, contr_coeff))
                    h1[y,x] = h1[x,y]
        efg_e = np.einsum('xyij,ji->xy', h1, dm)
        efg_nuc = _get_quad_nuc(mol, atm_id)
        v = efg_nuc - efg_e
        efg.append(v)
        log.info('\nAtom %d' % atm_id)
        log.info('Electronic EFG:')
        print(efg_e.real)
        log.info('Nuclear EFG:')
        print(efg_nuc.real)

        eigs, _ = np.linalg.eigh(v)
        log.info('Total EFG on main axis (Eigenvalues of total EFG tensors):')
        print(eigs)

    return np.asarray(efg).real

def _get_quadrupole_integrals(mol, atm_id):
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Compute the integrals of quadrupole operator
        # (3 \vec{r} \vec{r} - r^2) / r^5
        ipipv = mol.intor('int1e_ipiprinv', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_iprinvip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1ao = h1ao + h1ao.transpose(0,1,3,2)

    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval', coords)
    fc = 4*np.pi/3 * np.einsum('ip,iq->pq', ao, ao)

    h1ao[0,0] += fc
    h1ao[1,1] += fc
    h1ao[2,2] += fc
    return h1ao

def _get_4csph_quadrupole_integrals(xmol, atm_id):
    n2c = xmol.nao
    # The electronic quadrupole operator (3 \vec{r} \vec{r} - r^2) / r^5
    with xmol.with_rinv_origin(xmol.atom_coord(atm_id)):
        ipipv = xmol.intor('int1e_ipiprinv', 9).reshape(3,3,n2c,n2c)
        ipvip = xmol.intor('int1e_iprinvip', 9).reshape(3,3,n2c,n2c)
        h1LL = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1LL = h1LL + h1LL.conj().transpose(0,1,3,2)
        # trace = h1LL[0,0] + h1LL[1,1] + h1LL[2,2]
        # h1LL[0,0] -= trace
        # h1LL[1,1] -= trace
        # h1LL[2,2] -= trace

        ipipv = xmol.intor('int1e_ipipprinvp', 9).reshape(3,3,n2c,n2c)
        ipvip = xmol.intor('int1e_ipprinvpip', 9).reshape(3,3,n2c,n2c)
        h1SS = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1SS = h1SS + h1SS.conj().transpose(0,1,3,2)
        # trace = h1SS[0,0] + h1SS[1,1] + h1SS[2,2]
        # h1SS[0,0] -= trace
        # h1SS[1,1] -= trace
        # h1SS[2,2] -= trace
    
    coord = xmol.atom_coord(atm_id).reshape(1,3)
    aoL = xmol.eval_gto('GTOval', coord)
    aoS = xmol.eval_gto('GTOval_ip', coord)
    fcLL = np.einsum('ip,iq->pq', aoL.conj(), aoL)
    fcSS = np.einsum('dip,diq->pq', aoS.conj(), aoS)

    c = lib.param.LIGHT_SPEED
    int_4c = np.zeros((3, 3, n2c*2, n2c*2), dtype=fcLL.dtype)
    for x in range(3):
        for y in range(3):
            int_4c[x,y,:n2c,:n2c] = h1LL[x,y]
            int_4c[x,y,n2c:,n2c:] = h1SS[x,y] / 4.0 / c**2
            if x == y:
                int_4c[x,y,:n2c,:n2c] += 4*np.pi/3 * fcLL
                int_4c[x,y,n2c:,n2c:] += 4*np.pi/3 * fcSS / 4.0 / c**2
    return int_4c

EFG = kernel

scf.hf.RHF.EFG = scf.uhf.UHF.EFG = scf.rohf.ROHF.EFG = lib.class_as_method(EFG)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = '''
O 0.          0. -0.12390941
H 0. -1.42993701  0.98326612
H 0.  1.42993701  0.98326612
'''
    mol.basis = 'uncccpvdz'
    mol.unit = 'Bohr'
    mol.build()
    mf = scf.RHF(mol).sfx2c1e()
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.EFG(efg_nuc=[0])