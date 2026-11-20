#
# One active open shell among several equivalent ones (e.g. the 4f shells of the
# Eu ions of a stoichiometric Eu compound), the others frozen as their
# configuration average.
#
# The reference (fractional occupation of all 4f shells, scf.addons.projected_frac_occ
# over the union of the Eu f functions) has one open space of n_site x 14 spinors.
# localize_open_shell rotates it, inside the open space only (the reference density
# is unchanged), into spinors localised on the active site and on the others, by
# diagonalising the projector onto the active site's basis functions; the Kramers
# pairing is kept with the quaternion eigensolver (zquatev).
#
# FrozenFracSCF wraps the mean-field object for CAHF (mcscf.cahf_fock) and CASCI
# (mcscf.zcasci): the frozen spinors C_f with occupation f are removed from the
# orbital space and enter through
#     h' = h + G(D_f),   D_f = f C_f C_f^+,
#     E_nuc' = E_nuc + tr D_f h + a/2 tr D_f G(D_f),   a = (n-1) m / (n (m-1)),
# i.e. the configuration-averaged energy of the frozen shells and their
# interaction with everything else, exactly the frozen part of the CAHF energy of
# a product of the active-shell ensemble and the frozen-shell ensembles (the
# frozen-frozen pair factor a is that of one shell; for several frozen shells the
# inter-shell pairs carry 1 instead of a, see frozen_constant).  G is the
# G = 0-free kernel of mf.get_jk; the wavefunction constants in mf.energy_nuc()
# count all N electrons and stay as they are.
#
import numpy
from pyscf import lib
from pyscf.lib import logger


def localize_open_shell(s, mo_open, idx_active, nloc):
    '''Rotate the open-shell MOs (Kramers-interleaved columns, as from KRHF /
    addons.cas_order) into nloc spinors on the span of the basis functions
    idx_active and the rest.  Returns (mo_active, mo_rest, weights).'''
    from socutils.lib import zquatev
    n = mo_open.shape[1]
    assert n % 2 == 0 and nloc % 2 == 0
    sf = s[:, idx_active]
    x = numpy.linalg.solve(s[numpy.ix_(idx_active, idx_active)], sf.conj().T @ mo_open)
    p = (sf @ x).conj().T @ mo_open              # <i|P_A|j>
    p = .5 * (p + p.conj().T)
    blk = numpy.concatenate([numpy.arange(0, n, 2), numpy.arange(1, n, 2)])   # interleaved -> [unbarred | barred]
    pb = p[numpy.ix_(blk, blk)]
    m = n // 2
    a, b = pb[:m, :m], pb[m:, :m]
    kerr = max(numpy.abs(pb[m:, m:] - a.conj()).max(), numpy.abs(pb[:m, m:] + b.conj()).max())
    if kerr > 1e-8:
        raise RuntimeError('localize_open_shell: open shell not Kramers-paired (%.1e)' % kerr)
    w, v = zquatev.eigh(-pb, iop=1)              # interleaved Kramers-paired eigenvectors, largest P_A first
    w = -w
    rot = numpy.zeros_like(v)
    rot[blk] = v                                 # back to the interleaved row order of mo_open
    mo = mo_open @ rot
    return mo[:, :nloc], mo[:, nloc:], w


def localize_sites(s, mo_open, idx_sites, nloc):
    '''Split the open space into len(idx_sites) blocks of nloc spinors, block k
    localised on the basis functions idx_sites[k] (localize_open_shell applied
    site after site to what is left).  Returns ([mo_k], [weights_k]).'''
    out, wts, rest = [], [], mo_open
    for k, idx in enumerate(idx_sites):
        if k == len(idx_sites) - 1:
            assert rest.shape[1] == nloc
            out.append(rest)
            wts.append(None)
            break
        a, rest, w = localize_open_shell(s, rest, idx, nloc)
        out.append(a)
        wts.append(w)
    return out, wts


class FrozenFracSCF:
    '''Mean field with frozen fractionally occupied spinors (see module notes).
    Everything not overridden is delegated to the wrapped object.'''

    def __init__(self, mf, mo_frozen, nelec_frozen, shells=1):
        self.__dict__['_mf'] = mf
        if isinstance(mo_frozen, (list, tuple)):
            self.__dict__['blocks'] = list(mo_frozen)
            shells = len(mo_frozen)
            mo_frozen = numpy.hstack(mo_frozen)
        else:
            self.__dict__['blocks'] = None
        m = mo_frozen.shape[1]
        self.__dict__['mo_frozen'] = mo_frozen
        self.__dict__['frac'] = nelec_frozen / m
        nsh, msh = nelec_frozen / shells, m / shells
        self.__dict__['a'] = 0. if nsh <= 1 else (nsh - 1) * msh / (nsh * (msh - 1))
        self.__dict__['shells'] = shells
        self.__dict__['_cache'] = None
        self.__dict__['max_exact_shells'] = 3

    def __getattr__(self, key):
        return getattr(self._mf, key)

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            setattr(self._mf, key, val)

    def _frozen(self):
        if self._cache is None:
            mf, c, f = self._mf, self.mo_frozen, self.frac
            dm = lib.tag_array(f * c @ c.conj().T, mo_coeff=c, mo_occ=numpy.full(c.shape[1], f))
            vj, vk = mf.get_jk(mf.mol, dm)
            g = vj - vk
            h = mf.get_hcore()
            if self.shells == 1:
                e = (numpy.einsum('ij,ji', dm, h) + .5 * self.a * numpy.einsum('ij,ji', dm, g)).real
            else:
                # independent ensembles of each shell: intra-shell pairs carry a,
                # inter-shell pairs 1:  1/2 tr D G(D) - (1-a)/2 sum_s tr D_s G(D_s)
                if self.blocks is None:
                    raise ValueError('several frozen shells: pass the per-shell blocks as a list')
                e = numpy.einsum('ij,ji', dm, h) + .5 * numpy.einsum('ij,ji', dm, g)
                # the self-terms only shift the total energy (not h', the orbitals or the CI levels);
                # with more than max_exact_shells shells (equivalent sites) one is computed and scaled
                exact = len(self.blocks) <= self.max_exact_shells
                for cb in (self.blocks if exact else self.blocks[:1]):
                    ds = lib.tag_array(f * cb @ cb.conj().T, mo_coeff=cb, mo_occ=numpy.full(cb.shape[1], f))
                    vj, vk = mf.get_jk(mf.mol, ds)
                    w = 1 if exact else len(self.blocks)
                    e -= .5 * (1 - self.a) * w * numpy.einsum('ij,ji', ds, vj - vk)
                if not exact:
                    logger.info(mf, 'FrozenFracSCF: %d shells, self-term of the first scaled (equivalent sites assumed)',
                                len(self.blocks))
                e = e.real
            self.__dict__['_cache'] = (h + g, e)
            logger.info(mf, 'FrozenFracSCF: %d frozen spinors x %.4f electrons, a = %.4f, E_frozen = %.10f',
                        c.shape[1], f, self.a, e)
        return self._cache

    def get_hcore(self, *args, **kwargs):
        return self._frozen()[0]

    def energy_nuc(self):
        return self._mf.energy_nuc() + self._frozen()[1]
