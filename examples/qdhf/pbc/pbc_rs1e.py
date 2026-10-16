'''
Range-separated spin-free X2C-1e (rs1e) for periodic systems on top of
pyscf.pbc.x2c.sfx2c1e (k-points, the same uncontracted-basis decoupling).

    h(k) = h_X2C-1e[V_sr](k) + (V_ne(k) - V_sr(k)) + sum_A d_A

* V_sr = -sum_{A,T} Z_A erfc(omega |r - R_A - T|)/|r - R_A - T|: absolutely
  convergent real-space lattice sum (sr_nuclear_kpts), no G = 0 term and no
  neutralising background.  Its pVp is summed the same way.
* V_ne(k) - V_sr(k): the long-range rest of pyscf's Ewald nuclear potential
  (with_df.get_nuc), untransformed like J.
* d_A = h_X2C-1e[V_A] - h_X2C-1e[V_sr,A] - V_lr,A of the isolated atom: puts
  back the picture change of each atom's own long-range nuclear part (a
  one-centre, volume-independent Darwin-like term), so an isolated atom is
  exactly sf-X2C-1e.

pyscf's sfx2c1e decouples the full Ewald potential instead.  That potential is
the potential of the bare nuclei plus a uniform background -rho; X2C-1e
picture-changes the background of the nuclei but not that of the electrons in
J, which leaves -(pi/2c^2) rho per electron.  The difference of the two
Hamiltonians is therefore expected to be  -(pi/2c^2) Z_cell^2 / Omega  per cell
(times ~0.97-1.00 for heavy atoms), plus small multipole terms.
'''

import numpy
from pyscf import gto, lib
from pyscf.x2c import x2c
from pyscf.pbc.x2c import sfx2c1e as pbc_sfx2c1e
from pyscf.pbc.df import df as pbc_df


def _translations(a, rcut):
    b = numpy.linalg.inv(a).T
    nmax = numpy.ceil(rcut * numpy.linalg.norm(b, axis=1)).astype(int)
    n = lib.cartesian_prod([numpy.arange(-m, m + 1) for m in nmax])
    t = n @ a
    t = t[numpy.linalg.norm(t, axis=1) <= rcut]
    return t[numpy.argsort(numpy.linalg.norm(t, axis=1), kind='stable')]


def _shifted(mol, t):
    m = gto.Mole()
    m._atm, m._bas, m._env = mol._atm.copy(), mol._bas.copy(), mol._env.copy()
    for ia in range(mol.natm):
        p = m._atm[ia, gto.PTR_COORD]
        m._env[p:p + 3] += t
    m.cart = mol.cart
    m._built = True
    return m


def _nuclear_shells(coords, ids, zeta):
    fm = gto.Mole()
    fm.atom = [('X%d' % i, r) for i, r in zip(ids, coords)]
    fm.unit = 'Bohr'
    fm.basis = {'X%d' % i: [[0, [zeta, 1.]]] for i in set(ids)}
    fm.verbose = 0
    fm.build(False, False)
    q = (numpy.pi / zeta)**1.5 * (2 * zeta / numpy.pi)**.75   # charge of a libcint s shell
    return fm, q


def sr_nuclear_kpts(cell, kpts, omega, nuc_zeta=1e16, precision=1e-11):
    '''<m|V_sr|n>(k) and <grad m|V_sr|grad n>(k), spherical AO basis of cell
    (which may be decontracted).  omega = 0: full Coulomb (molecule only).'''
    kpts = numpy.reshape(kpts, (-1, 3))
    coords = cell.atom_coords()
    lnp = -numpy.log(precision)
    amin = min(cell.bas_exp(i).min() for i in range(cell.nbas))
    r_ao = numpy.sqrt(lnp / amin)
    periodic = getattr(cell, 'dimension', 0) == 3
    if periodic:
        a = cell.lattice_vectors()
        span = numpy.linalg.norm(coords[:, None] - coords[None], axis=2).max()
        r_sr = (numpy.sqrt(lnp) + 1.) / omega
        t_ao = _translations(a, 2 * r_ao + span)
        t_nuc = _translations(a, r_ao + r_sr + span)
    else:
        r_sr = numpy.inf if not omega else (numpy.sqrt(lnp) + 1.) / omega
        t_ao = t_nuc = numpy.zeros((1, 3))
    Z = cell.atom_charges()
    all_nuc = [(ia, coords[ia] + t) for t in t_nuc for ia in range(cell.natm)]
    nao = cell.nao_nr()
    nk = len(kpts)
    v = numpy.zeros((nk, nao, nao), dtype=complex)
    pvp = numpy.zeros((nk, nao, nao), dtype=complex)
    mol0 = _shifted(cell, numpy.zeros(3))
    nb = mol0.nbas
    for t in t_ao:
        # skip AO image pairs that do not overlap
        if numpy.min(numpy.linalg.norm(coords[:, None] - coords[None] - t, axis=2)) > 2 * r_ao:
            continue
        # nuclei within reach of both ends of the pair
        sel = [(ia, r) for ia, r in all_nuc
               if numpy.min(numpy.linalg.norm(coords - r, axis=1)) <= r_ao + r_sr
               and numpy.min(numpy.linalg.norm(coords + t - r, axis=1)) <= r_ao + r_sr]
        if not sel:
            continue
        fm, q = _nuclear_shells([r for _, r in sel], [ia for ia, _ in sel], nuc_zeta)
        chg = -Z[[ia for ia, _ in sel]] / q
        pm = gto.conc_mol(gto.conc_mol(mol0, _shifted(cell, t)), fm)
        pm.omega = -omega if omega else 0.
        sl = (0, nb, nb, 2 * nb, 2 * nb, pm.nbas)
        vt = numpy.einsum('ijp,p->ij', pm.intor('int3c2e', shls_slice=sl), chg)
        wt = numpy.einsum('ijp,p->ij', pm.intor('int3c2e_pvp1', shls_slice=sl), chg)
        phase = numpy.exp(1j * kpts @ t)
        v += phase[:, None, None] * vt
        pvp += phase[:, None, None] * wt
    return v, pvp


def _x2c_fw(t, v, w, s, c):
    x = x2c._x2c1e_xmatrix(t, v, w, s, c)
    return x2c._get_hcore_fw(t, v, w, s, x, c)


class SpinFreeRS1eHelper(pbc_sfx2c1e.SpinFreeX2CHelper):
    '''pyscf.pbc sfx2c1e helper with the rs1e Hamiltonian (see module notes).'''
    omega = 1.0
    nuc_zeta = 1e16
    self_lr = True
    rs_precision = 1e-11

    def _atomic_d(self, xcell):
        '''sum_A d_A on the uncontracted basis (block diagonal, k independent).'''
        c = lib.param.LIGHT_SPEED
        nao = xcell.nao_nr()
        d = numpy.zeros((nao, nao))
        cache = {}
        for ia, (_, _, p0, p1) in enumerate(xcell.aoslice_by_atom()):
            key = xcell.atom_symbol(ia)
            if key not in cache:
                atom = gto.Mole()
                atom.atom = [(key, (0., 0., 0.))]
                atom.unit = 'Bohr'
                atom.basis = {key: self.cell._basis[key]}
                atom.cart = xcell.cart
                atom.spin = gto.charge(key) % 2
                atom.verbose = 0
                atom.build(False, False)
                if self.xuncontract:
                    # same decontraction as get_xmol (pyscf x2c._uncontract_mol)
                    atom = atom.decontract_basis(aggregate=True)[0]
                t = atom.intor('int1e_kin')
                s = atom.intor('int1e_ovlp')
                vf, wf = (x[0].real for x in sr_nuclear_kpts(atom, numpy.zeros(3), 0., self.nuc_zeta))
                vs, ws = (x[0].real for x in sr_nuclear_kpts(atom, numpy.zeros(3), self.omega, self.nuc_zeta))
                cache[key] = _x2c_fw(t, vf, wf, s, c) - _x2c_fw(t, vs, ws, s, c) - (vf - vs)
            d[p0:p1, p0:p1] = cache[key]
        return d

    def get_hcore(self, cell=None, kpts=None):
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts_lst = numpy.zeros((1, 3))
        else:
            kpts_lst = numpy.reshape(kpts, (-1, 3))
        xcell, contr_coeff = self.get_xmol(cell)
        with_df = pbc_df.DF(xcell)
        c = lib.param.LIGHT_SPEED
        t = xcell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
        s = xcell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts_lst)
        v = lib.asarray(with_df.get_nuc(kpts_lst))
        v_sr, w_sr = sr_nuclear_kpts(xcell, kpts_lst, self.omega, self.nuc_zeta, self.rs_precision)
        d = self._atomic_d(xcell) if self.self_lr else 0.
        h1_kpts = []
        for k in range(len(kpts_lst)):
            vs = v_sr[k] if abs(kpts_lst[k]).max() > 1e-9 else v_sr[k].real
            ws = w_sr[k] if abs(kpts_lst[k]).max() > 1e-9 else w_sr[k].real
            h1 = _x2c_fw(t[k], vs, ws, s[k], c) + (v[k] - vs) + d
            if self.xuncontract and contr_coeff is not None:
                h1 = contr_coeff.T @ h1 @ contr_coeff
            h1_kpts.append(h1)
        if kpts is None or numpy.shape(kpts) == (3,):
            return h1_kpts[0]
        return lib.asarray(h1_kpts)


def sfrs1e(mf, omega=1.0, self_lr=True):
    '''mf.sfx2c1e() with the rs1e helper.'''
    mf = mf.sfx2c1e()
    hlp = SpinFreeRS1eHelper(mf.cell)
    hlp.omega = omega
    hlp.self_lr = self_lr
    mf.with_x2c = hlp
    return mf
