'''
Quadruple precision spin-free x2cmp ('sfmp' of socutils.somf.x2cmp).

    F4c = h4c + J[D4c] - 1/2 K[D4c]          spin-free 4c (modified Dirac) Fock
                                              matrix of the superposition of
                                              atomic SF-DHF densities D4c
    X, R from the positive-energy eigenvectors of (F4c, S4c)
    h   = to_2c(X, R, F4c) - (J[D2c] - 1/2 K[D2c])

D4c and D2c are the spin-free atomic densities of x2camf (the same ones the
production sfmp uses), converted from the spinor to the scalar basis,
symmetrized to exact spherical symmetry and normalized to exactly Z in quad
precision.  The spin-free 4c two-electron integrals are those of
socutils.scf.sfdhf (small component p chi / 2c, dot products of the momenta
on each electron).  The (SS|SS) part is evaluated integral-direct, using that
the atomic densities are atom-block diagonal.

All matrices: scalar decontracted basis xmol of x2c.uncontract_mol,
4c layout [L, S] with S in the pseudo-large representation (metric T/2c^2).
'''

import numpy
from pyscf import gto
from pyscf.data import elements
from pyscf.lib import param
from socutils.qdhf import dhf, x2c, x2c4c, qarray
from socutils.qdhf import integrals as qints
from socutils.qdhf.qarray import QArray, qmp, hmp


# ------------------------------------------------------------ atomic densities
def _x2camf_densities(symb, raw_bas, light_speed):
    import socutils  # noqa: F401  (puts the bundled x2camf on sys.path)
    import x2camf
    z = elements.charge(symb)
    shell = numpy.array([b[0] for b in raw_bas])
    exp_a = numpy.array([b[-1][0] for b in raw_bas])
    ints = x2camf.libx2camf.atm_integrals(0, z, shell[-1] + 1, len(shell), 0, shell, exp_a,
                                          speed_of_light=light_speed, spin_free=True)
    return ints[11], ints[12]


def _spinor_to_scalar(amol, d):
    '''Spin-summed scalar density matrix (spherical AOs of amol) of a spinor
    density matrix.'''
    ca, cb = amol.sph2spinor_coeff()
    da = ca @ d @ ca.conj().T
    db = cb @ d @ cb.conj().T
    return (da + db).real


def _spherical_average(amol, d):
    '''Exactly spherically symmetric version (quad, object array of mpf) of an
    atomic scalar density matrix: delta_ll' delta_mm' d_l(i, j).'''
    n = d.shape[0]
    out = numpy.empty((n, n), dtype=object)
    out[:] = hmp.mpf(0)
    ao_loc = amol.ao_loc_nr()
    for i in range(amol.nbas):
        li = amol.bas_angular(i)
        for j in range(amol.nbas):
            if amol.bas_angular(j) != li:
                continue
            nm = 2 * li + 1
            avg = sum(hmp.mpf(float(d[ao_loc[i]+m, ao_loc[j]+m])) for m in range(nm)) / nm
            for m in range(nm):
                out[ao_loc[i]+m, ao_loc[j]+m] = avg
    return out


def atomic_densities(xmol, light_speed=None):
    '''D4c (2n x 2n, blocks LL LS / SL SS) and D2c (n x n) of xmol: superposition
    of exactly spherical, exactly neutral atomic spin-free densities.'''
    c = dhf.get_light_speed(light_speed)
    n = xmol.nao_nr()
    d4 = numpy.empty((2*n, 2*n), dtype=object)
    d4[:] = hmp.mpf(0)
    d2 = numpy.empty((n, n), dtype=object)
    d2[:] = hmp.mpf(0)
    aoslices = xmol.aoslice_by_atom()
    done = {}
    for ia in range(xmol.natm):
        symb = xmol.atom_pure_symbol(ia)
        if symb not in done:
            raw = gto.mole.uncontracted_basis(xmol._basis[xmol.atom_symbol(ia)])
            z = elements.charge(symb)
            amol = gto.M(atom=[[symb, (0., 0., 0.)]], basis={symb: raw}, spin=z % 2, verbose=0)
            s4, s2 = _x2camf_densities(symb, raw, float(c))
            n2c = amol.nao_2c()
            na = amol.nao_nr()
            blocks = {}
            for key, sl in (('LL', (slice(0, n2c), slice(0, n2c))),
                            ('LS', (slice(0, n2c), slice(n2c, 2*n2c))),
                            ('SL', (slice(n2c, 2*n2c), slice(0, n2c))),
                            ('SS', (slice(n2c, 2*n2c), slice(n2c, 2*n2c)))):
                blocks[key] = _spherical_average(amol, _spinor_to_scalar(amol, s4[sl]))
            blocks['2c'] = _spherical_average(amol, _spinor_to_scalar(amol, s2))
            # exact normalization to Z with quad integrals of the free atom
            qa = qints.QuadIntegrals(amol)
            sa = qa.ovlp_x()
            sy = qa.ovlp_y()
            ta = (sy[0::3, 0::3] + sy[1::3, 1::3] + sy[2::3, 2::3]) * hmp.mpf(0.5)
            dll, dss = QArray.from_mpf(blocks['LL']), QArray.from_mpf(blocks['SS'])
            n4 = qmp.re(qarray.trace_dot(dll, sa)) + \
                qmp.re(qarray.trace_dot(dss, ta)) / (2 * c**2)
            n2 = qmp.re(qarray.trace_dot(QArray.from_mpf(blocks['2c']), sa))
            f4 = hmp.mpf(z) / n4
            f2 = hmp.mpf(z) / n2
            for key in ('LL', 'LS', 'SL', 'SS'):
                blocks[key] = blocks[key] * f4
            blocks['2c'] = blocks['2c'] * f2
            # AO order of amol -> AO order of the atom in xmol, by (l, exponent)
            xa = [ib for ib in range(xmol.nbas) if xmol.bas_atom(ib) == ia]
            key_x = [(xmol.bas_angular(ib), float(xmol.bas_exp(ib)[0])) for ib in xa]
            key_a = [(amol.bas_angular(ib), float(amol.bas_exp(ib)[0])) for ib in range(amol.nbas)]
            assert sorted(key_x) == sorted(key_a) and len(key_x) == len(set(key_x))
            loc_a = amol.ao_loc_nr()
            perm = []
            for k in key_x:
                ib = key_a.index(k)
                perm.extend(range(loc_a[ib], loc_a[ib+1]))
            perm = numpy.array(perm)
            assert len(perm) == na
            done[symb] = {k: v[numpy.ix_(perm, perm)] for k, v in blocks.items()}
        b = done[symb]
        p0, p1 = aoslices[ia][2:]
        sl = slice(p0, p1)
        ss = slice(n + p0, n + p1)
        d4[sl, sl] = b['LL']
        d4[sl, ss] = b['LS']
        d4[ss, sl] = b['SL']
        d4[ss, ss] = b['SS']
        d2[sl, sl] = b['2c']
    return QArray.from_mpf(d4), QArray.from_mpf(d2)


# ------------------------------------------------------------ spin-free 4c
class SpinFree4c:
    '''Spin-free 4c (modified Dirac) Hamiltonian in the scalar basis of mol,
    with the two-electron interaction of socutils.scf.sfdhf.'''

    def __init__(self, mol, light_speed=None):
        c = dhf.get_light_speed(light_speed)
        self.mol = mol
        self.c = c
        qi = self.qi = qints.QuadIntegrals(mol)
        n = self.n = qi.nao
        s = qi.ovlp_x()
        v = qi.nuc_x()
        sy = qi.ovlp_y()
        vy = qi.nuc_y()
        t = (sy[0::3, 0::3] + sy[1::3, 1::3] + sy[2::3, 2::3]) * hmp.mpf(0.5)
        w = vy[0::3, 0::3] + vy[1::3, 1::3] + vy[2::3, 2::3]
        self.s, self.t = s, t
        h4 = QArray.zeros((2*n, 2*n))
        s4 = QArray.zeros((2*n, 2*n))
        L, S = slice(0, n), slice(n, 2*n)
        h4[L, L] = v
        h4[L, S] = t
        h4[S, L] = t
        h4[S, S] = w * (1 / (4 * c**2)) - t
        s4[L, L] = s
        s4[S, S] = t * (1 / (2 * c**2))
        self.hcore, self.ovlp = h4, s4
        # expansion: large = chi, small_a = d_a chi / 2c
        tl = numpy.zeros((2*n, n))
        tl[:n] = numpy.eye(n)
        comps = [(QArray.from_numpy(tl), 'X')]
        for a in range(3):
            ts = numpy.zeros((2*n, 3*n))
            for mu in range(n):
                ts[n+mu, 3*mu+a] = 1.
            comps.append((QArray.from_numpy(ts) * (1 / (2 * c)), 'Y'))
        eri_yyxx = qi.eri_yyxx()
        eri = {('X', 'X'): qi.eri_xxxx(), ('Y', 'X'): eri_yyxx,
               ('X', 'Y'): eri_yyxx.transpose(2, 3, 0, 1)}
        # (SS|SS) is added by _jk_ssss
        self._ham = dhf.Hamiltonian4c(h4, s4, comps, eri, None, c, k_factor=0.5, occ_number=2)
        self._setup_ssss()

    def _setup_ssss(self):
        qi = self.qi
        atoms, ncart = [], []
        for ia, l, es, cs in qints.shell_data(self.mol):
            for ic in range(cs.shape[1]):
                atoms.append(ia)
                ncart.append((l + 2) * (l + 3) // 2)
                if l > 0:
                    atoms.append(ia)
                    ncart.append(l * (l + 1) // 2)
        assert len(atoms) == len(qi.shls_S)
        off = numpy.concatenate([[0], numpy.cumsum(ncart)])
        assert off[-1] == qi.ncart_S
        self._s_atoms = numpy.array(atoms)
        self._s_off = off

    def _jk_ssss(self, dss, batch=6):
        '''Spin-free (SS|SS) J and K (without the (1/2c)^4 factor) of an
        atom-block-diagonal scalar small-component density dss (n x n):
            J_ij = sum_ab sum_kl (d_a i d_a j | d_b k d_b l) D_kl
            K_ij = sum_ab sum_kl (d_a i d_a k | d_b l d_b j) D_kl '''
        qi = self.qi
        n = self.n
        N = qi.ncart_S
        cy = qi.cy
        cya = [cy[a::3] for a in range(3)]          # rows 3 mu + a
        shls = qi.shls_S
        jc = QArray.zeros((N, N))
        kc = [[QArray.zeros((N, N)) for b in range(3)] for a in range(3)]
        for ia in sorted(set(self._s_atoms)):
            sel = numpy.where(self._s_atoms == ia)[0]
            cidx = numpy.concatenate([numpy.arange(self._s_off[i], self._s_off[i+1]) for i in sel])
            nc = len(cidx)
            # cartesian densities restricted to the atom
            rho = None
            rab = [[None] * 3 for _ in range(3)]
            for a in range(3):
                ca = cya[a][:, cidx]
                for b in range(3):
                    cb = cya[b][:, cidx]
                    rab[a][b] = ca.T @ dss @ cb             # R_ab = cy_a^T D cy_b
                rho = rab[a][a] if rho is None else rho + rab[a][a]
            rho_f = rho.reshape(nc * nc, 1)
            r9 = qarray.concatenate([rab[a][b].reshape(1, nc * nc) for a in range(3) for b in range(3)],
                                    axis=0)
            sh_c = shls[sel]
            for p0 in range(0, len(shls), batch):
                sh_p = shls[p0:p0+batch]
                cp = numpy.arange(self._s_off[p0], self._s_off[min(p0+batch, len(shls))])
                npc = len(cp)
                # J: (p q | r s) with r, s on the atom
                e = qi._int2e_cart([sh_p, shls, sh_c, sh_c], 2)
                jc[cp, :] = jc[cp, :] + (e.reshape(npc * N, nc * nc) @ rho_f).reshape(npc, N)
                # K: (p q | r s) with q, r on the atom:  K[p, s] = sum_qr (pq|rs) R[q, r]
                f = qi._int2e_cart([sh_p, sh_c, sh_c, shls], 0)
                fr = f.reshape(npc, nc * nc, N)
                for ip in range(npc):
                    kk = r9 @ fr[ip]                         # (9, N)
                    for a in range(3):
                        for b in range(3):
                            kc[a][b][cp[ip], :] = kc[a][b][cp[ip], :] + kk[3*a+b]
        j = None
        for a in range(3):
            x = cya[a] @ jc @ cya[a].T
            j = x if j is None else j + x
        k = None
        for a in range(3):
            for b in range(3):
                x = cya[a] @ kc[a][b] @ cya[b].T
                k = x if k is None else k + x
        return j, k

    def get_jk(self, d4):
        '''J and K (Fock = h + J - K/2) of a total (spin-summed) 4c density.'''
        vj, vk = self._ham.get_jk(d4)
        n = self.n
        S = slice(n, 2*n)
        j, k = self._jk_ssss(d4[S, S])
        f = 1 / (2 * self.c)**4
        vj[S, S] = vj[S, S] + j * f
        vk[S, S] = vk[S, S] + k * f
        return vj, vk

    def get_jk_full(self, d4):
        '''Reference with the stored (SS|SS) integrals (small systems only).'''
        ham = self._ham
        eri = dict(ham.eri)
        eri[('Y', 'Y')] = self.qi.eri_yyyy()
        h = dhf.Hamiltonian4c(ham.hcore, ham.ovlp, ham.comps, eri, None, self.c, 0.5, 2)
        return h.get_jk(d4)


def get_hcore(xmol, light_speed=None, sf4c=None):
    '''Spin-free x2cmp 2c core Hamiltonian in the scalar decontracted basis.'''
    if sf4c is None:
        sf4c = SpinFree4c(xmol, light_speed)
    d4, d2 = atomic_densities(xmol, light_speed)
    vj, vk = sf4c.get_jk(d4)
    f4 = sf4c.hcore + vj - vk * hmp.mpf(0.5)
    xm, r = x2c4c.decouple(f4.as_complex(), sf4c.ovlp.as_complex())
    h = x2c4c.to_2c(xm, r, f4.as_complex())
    j2, k2 = sf4c.qi.get_jk_direct(d2)
    return QArray(h.re) - (j2 - k2 * hmp.mpf(0.5)), (d4, d2, f4)


class SFMP_RHF(dhf.RHF):
    '''Quad precision RHF with the spin-free x2cmp Hamiltonian, in the
    contracted basis of mol (like the production code).'''

    def __init__(self, mol, light_speed=None):
        dhf.RHF.__init__(self, mol)
        self.sfmp_light_speed = light_speed
        self.init_guess = 'double'

    def build_hamiltonian(self):
        xmol, contr = x2c.uncontract_mol(self.mol)
        h, _ = get_hcore(xmol, self.sfmp_light_speed)
        ham = dhf.build_nr_hamiltonian(self.mol, 'RHF', self.verbose)
        ham.hcore = contr.T @ h @ contr
        self.ham = ham
        return ham
