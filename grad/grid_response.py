#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Grid response of energy derivatives on a molecular grid that moves with the nuclei (every point of
atom A at r_g = R_A + u_g, the partition weights w_g = vol_g P_A / sum_i P_i differentiated):

    E = sum_g w_g e_g(r_g),
    dE/dR_B = sum_g [ (d w_g / d R_B) e_g + w_g (d e_g / d R_B)|_{fixed point} ]
              + sum_{g of B} w_g grad_r e_g(r_g).

The middle term is what the fixed-grid gradients compute; this module supplies the other two:

* partition_grad(grids, e): sum_g e_g d w_g / d R_B (natm, 3) for the Becke / Stratmann-Scuseria-Frisch
  partitions with the size adjustments the grid was made with (dft.grids.partition_params), by
  lib/dft/becke_deriv.c SOCDFTpartition_grad contracted point by point (no natm x ngrid x 3 array);
* owner_sum(grids, v): sum_{g of A} v_g (natm, 3), v (3, ngrid) the weighted point derivative
  w_g grad_r e_g; by translational invariance grad_r e_g = -sum_B (d e_g / d R_B)|_{fixed point}, so the
  kernels obtain it as minus the per-point sum of their fixed-grid integrand.

The full derivative is translationally invariant: sum_B d w_g / d R_B = 0 and the two point terms
cancel.  References: B. G. Johnson, P. M. W. Gill, J. A. Pople, J. Chem. Phys. 98, 5612 (1993) (the
Becke weight derivatives); R. E. Stratmann, G. E. Scuseria, M. J. Frisch, Chem. Phys. Lett. 257, 213
(1996) (SSF).
'''

import ctypes
import numpy
from pyscf import lib


def _lib():
    from socutils.hessian.rhf_fast import _load_socdft
    libdft = _load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTpartition_grad'):
        return None
    return libdft


def available(grids):
    from socutils.dft import grids as sgrids
    return (_lib() is not None and getattr(grids, 'atm_idx', None) is not None
            and sgrids.partition_params(grids)[0] is not None)


def partition_grad(grids, e, mol=None):
    '''sum_g e_g d w_g / d R_B, (natm, 3): e (ngrid) the energy density at the points (without the weight)'''
    from socutils.dft import grids as sgrids
    if mol is None:
        mol = grids.mol
    libdft = _lib()
    kind, radii = sgrids.partition_params(grids, mol)
    if libdft is None or kind is None:
        raise NotImplementedError('grid response: partition %s' % getattr(grids, 'becke_scheme', None))
    coords = numpy.ascontiguousarray(grids.coords, dtype=numpy.double)
    w = numpy.ascontiguousarray(grids.weights, dtype=numpy.double)
    owner = numpy.ascontiguousarray(grids.atm_idx, dtype=numpy.int32)
    e = numpy.ascontiguousarray(e, dtype=numpy.double)
    assert e.size == w.size == owner.size
    atm = numpy.ascontiguousarray(mol.atom_coords(), dtype=numpy.double)
    out = numpy.zeros((mol.natm, 3))
    pr = lib.c_null_ptr() if radii is None else numpy.ascontiguousarray(radii).ctypes.data_as(ctypes.c_void_p)
    libdft.SOCDFTpartition_grad(
        out.ctypes.data_as(ctypes.c_void_p), coords.ctypes.data_as(ctypes.c_void_p), w.ctypes.data_as(ctypes.c_void_p),
        owner.ctypes.data_as(ctypes.c_void_p), e.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(w.size),
        atm.ctypes.data_as(ctypes.c_void_p), pr, ctypes.c_int(mol.natm), ctypes.c_int(0 if kind == 'becke' else 1))
    return out


def owner_sum(grids, v, natm=None):
    '''sum_{g owned by A} v[:, g], (natm, 3); v (3, ngrid)'''
    if natm is None:
        natm = grids.mol.natm
    owner = numpy.asarray(grids.atm_idx)
    v = numpy.asarray(v).reshape(3, -1)
    keep = owner >= 0
    out = numpy.zeros((natm, 3))
    for x in range(3):
        out[:, x] = numpy.bincount(owner[keep], weights=v[x][keep], minlength=natm)[:natm]
    return out


def cosx_point_terms(k, dm, m, yg=None, tol=None):
    '''(e (ngrid), v (3, ngrid)) of the COSX energy sum_g w_g e_g of socutils.scf.cosx.COSX k at the
    fixed density dm: e_g = sum_ab H_a A_ab F_b (+ 1/2 phi^T Yg phi, the grid-overlap term of the fit),
    H = m^T phi, F = dm phi; v = w_g d e_g / d r_g (lib/dft/cosx_gridresp.c SOCDFTcosx_grid_resp)'''
    from socutils.dft.vxc_fast import _blas_ptrs
    k.build()
    libdft = k._libdft
    mol = k.mol
    atm, bas, env, ao_loc = k._env
    ngrid = k._weights.size
    dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
    m = numpy.ascontiguousarray(m, dtype=numpy.double)
    pg = numpy.empty((3, ngrid))
    eg = numpy.empty(ngrid)
    py = lib.c_null_ptr() if yg is None else numpy.ascontiguousarray(yg).ctypes.data_as(ctypes.c_void_p)
    err = libdft.SOCDFTcosx_grid_resp(
        pg.ctypes.data_as(ctypes.c_void_p), eg.ctypes.data_as(ctypes.c_void_p), dm.ctypes.data_as(ctypes.c_void_p),
        (dm if m is dm else m).ctypes.data_as(ctypes.c_void_p), py, ctypes.c_void_p(k._opt),
        k._coords.ctypes.data_as(ctypes.c_void_p), k._weights.ctypes.data_as(ctypes.c_void_p),
        k._non0.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrid), ctypes.c_int(k.blk),
        ctypes.c_double(k.ao_cutoff), ctypes.c_double(k.tol if tol is None else tol), ctypes.c_double(k.tcut),
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
        *_blas_ptrs())
    if err:
        raise MemoryError('COSX grid response: a kernel buffer could not be allocated')
    return eg, pg


def cosx_response(k, dm, m, yg=None, tol=None):
    '''the grid response of COSX.grad (natm, 3): d/dR of tr(D P K_plain[D]) from the motion of the grid
    points with their atoms and the derivatives of the partition weights, in the integrals, the AOs on the
    grid and the grid overlap S_g of the fit (yg: the matrix COSX.grad contracts with d S_g)'''
    eg, pg = cosx_point_terms(k, dm, m, yg, tol)
    return response(k.grids, eg, pg, k.mol)


def response(grids, e, v, mol=None):
    '''partition_grad(grids, e) + owner_sum(grids, v): the grid response of sum_g w_g e_g whose weighted
    point derivative is v (3, ngrid)'''
    if mol is None:
        mol = grids.mol
    return partition_grad(grids, e, mol) + owner_sum(grids, v, mol.natm)
