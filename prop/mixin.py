#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''One-electron properties as methods of the objects that make a density.

A property object needs two things: a density, and an operator in the
picture of the Hamiltonian.  The density belongs to the method (SCF, CC,
CI, ...); the operator belongs to the reference mean field, which alone
knows whether it is nonrelativistic, X2C or four-component.  The mixins
below put the two together:

* :class:`PropertyMixin` has the properties (dip_moment, expect_int1e), the
  choice of density and the cache; written once.
* The basis of the density and of the operators is declared by the class,
  never guessed from shapes (a spinor density, n2c x n2c, and a
  spin-orbital one, 2 nao x 2 nao, have the same shape): one mixin per
  basis implements ``_prop_expect`` --

  - :class:`ScalarPropertyMixin`: spin-free AO basis (RHF, UHF, ...);
  - :class:`SpinOrbitalPropertyMixin`: GHF AO basis;
  - :class:`SpinorPropertyMixin`: 2c spinor basis, bare or X2C;
  - :class:`FourComponentPropertyMixin`: 4c spinor basis (DHF).

On an X2C reference the operator would have to be picture-changed with the
response of the decoupling (X, R) for every operator component.  Instead the
density is transformed once: the effective 4c densities of the helper
(``with_x2c.get_hfw1_adjoint``, the adjoint of get_hfw1) give
tr(D dh_X2C/dlambda) = tr(G O_4c) for every one-electron operator O_4c.  G is
cached on the object with the density it was made from, and made again when
the density changes (an SCF or CC rerun, another ``relaxed`` or
``x_response``).

The density: a mean field has one.  A correlated method gives the relaxed
density (``make_rdm1_relaxed``), the derivative of its energy; without one
the property raises unless ``relaxed=False`` asks for the expectation value
of the unrelaxed density explicitly.
'''

import numpy
from pyscf.lib import logger
from pyscf.data import nist
from socutils.prop import int1e


def nuclear_dipole(mol, origin=(0., 0., 0.)):
    '''sum_A Z_A (R_A - origin), a.u.'''
    return numpy.einsum('i,ix->x', mol.atom_charges(),
                        mol.atom_coords() - numpy.asarray(origin))


class PropertyMixin:
    '''Properties of the density of the object; see the module docstring.'''

    def _prop_reference(self):
        return getattr(self, '_scf', None) or self

    def _prop_density(self, relaxed=None):
        '''AO density of the object (the basis is that of the mixin).'''
        if getattr(self, '_scf', None) is None:          # a mean field
            return self.make_rdm1()
        if relaxed is None or relaxed:
            make = getattr(self, 'make_rdm1_relaxed', None)
            if make is None:
                raise NotImplementedError(
                    '%s has no relaxed density (make_rdm1_relaxed); pass '
                    'relaxed=False for the expectation value of the unrelaxed '
                    'density' % self.__class__.__name__)
            return make(ao_repr=True)
        return self.make_rdm1(ao_repr=True)

    def _prop_cached(self, tag, dm, make):
        '''Cache of quantities made from the density dm (the effective
        densities): reused while dm is unchanged.'''
        cache = self.__dict__.setdefault('_prop_cache', {})
        hit = cache.get(tag)
        if hit is not None and hit[0].shape == dm.shape and numpy.array_equal(hit[0], dm):
            return hit[1]
        val = make()
        cache[tag] = (numpy.array(dm, copy=True), val)
        return val

    def _prop_expect(self, op, dm, origin, picture_change, x_response):
        '''Electronic expectation value tr(O D) of the operator op (name of
        socutils.prop.int1e), (comp,).  Implemented per basis.'''
        raise NotImplementedError

    def expect_int1e(self, op, dm=None, origin=(0., 0., 0.), relaxed=None,
                     picture_change=True, x_response=True):
        '''tr(O D) of a one-electron operator (a name or spec of
        socutils.prop.int1e), in the picture of the reference: the
        derivative of the energy with respect to a perturbation lambda O for
        x_response=True on an X2C reference.'''
        if dm is None:
            dm = self._prop_density(relaxed)
        return self._prop_expect(op, dm, origin, picture_change, x_response)

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   picture_change=True, x_response=True, origin=(0., 0., 0.),
                   relaxed=None, **kwargs):
        '''Molecular dipole moment sum_A Z_A R_A - tr(r D).

        On an X2C reference the electronic part is the derivative of the X2C
        energy with respect to the field, with the response of the decoupling
        (x_response=True) or the plain picture change [U+ r U]_++
        (x_response=False); picture_change=False takes the bare 2c operator
        (error estimates).  For a correlated method the density is the
        relaxed one unless relaxed=False.'''
        if mol is None:
            mol = self.mol
        log = logger.new_logger(mol, verbose)
        el = self.expect_int1e('r', dm, origin, relaxed, picture_change, x_response)
        mol_dip = nuclear_dipole(mol, origin) - el.real
        if unit.upper() == 'DEBYE':
            mol_dip = mol_dip * nist.AU2DEBYE
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip


def _bare_ll(mol, op, origin):
    '''LL block (the nonrelativistic operator) in mol's spinor basis.'''
    spec = int1e._resolve(op)
    if spec['LS'] is not None:
        raise NotImplementedError('odd (LS/SL) operators need an X2C or 4c reference')
    ll = int1e._blocks(mol, spec, origin)[0]
    return ll


def _spinor_expect(obj, op, dm, origin, picture_change, x_response):
    '''tr(O D) for a 2c spinor density dm (contracted basis).'''
    from socutils.somf.x2cmmf import _X2CMMF
    ref = obj._prop_reference()
    mol = ref.mol
    if isinstance(ref, _X2CMMF) and picture_change:
        return _x2cmmf_expect(obj, ref, op, dm, origin, x_response)
    with_x2c = getattr(ref, 'with_x2c', None)
    if with_x2c is None or not picture_change:
        return numpy.einsum('xij,ji->x', _bare_ll(mol, op, origin), dm)
    if not hasattr(with_x2c, 'get_hfw1_adjoint'):
        # pyscf helpers (and the sNSO helper, which screens only the nuclear
        # spin-orbit integrals of the Hamiltonian): the plain per-operator
        # picture change; x_response=True is not available there
        ints = int1e.prop_int1e(ref, op, x_response=x_response, origin=origin)
        ints = ints.reshape(-1, *ints.shape[-2:])
        return numpy.einsum('xij,ji->x', ints, dm)
    from pyscf.x2c.x2c import _decontract_spinor
    xmol, contr = _decontract_spinor(mol, with_x2c.xuncontract)
    dmx = contr @ dm @ contr.conj().T
    gh = obj._prop_cached(('x2c_effective_density', x_response), dmx,
                          lambda: with_x2c.get_hfw1_adjoint(dmx, x_response=x_response)[0])
    op4c = int1e.assemble_4c(xmol, op, origin=origin)
    return numpy.einsum('xij,ji->x', op4c, gh)


def _x2cmmf_expect(obj, ref, op, dm, origin, x_response):
    '''X2CMMF reference: the plain picture change with the decoupling (X, R)
    of the DHF Fock matrix.  The derivative of the energy would need the
    response of the 4c Fock matrix (the DHF orbital response); without it the
    result is the expectation value of the 4c operator.'''
    if x_response:
        logger.warn(ref, 'X2CMMF: the property is the expectation value with the plain picture '
                    'change (x_response=False); its derivative (gradient) form is ill-defined '
                    'without the response of the DHF Fock matrix.  The difference is small for '
                    'valence properties (dipole) and larger for operators at the nuclei (EFG, '
                    'effective field, contact density): 0.4% for the spin-orbit EFG of Sm3+ in X2C.')
    if ref._x2cmmf_xr is None:
        raise RuntimeError('X2CMMF reference without its decoupling; rebuild it with x2cmmf()')
    from socutils.somf.x2cmp import to_2c_adjoint
    x, r = ref._x2cmmf_xr
    g = obj._prop_cached(('x2cmmf_effective_density',), dm, lambda: to_2c_adjoint(x, r, dm))
    return numpy.einsum('xij,ji->x', int1e.assemble_4c(ref.mol, op, origin=origin), g)


class SpinorPropertyMixin(PropertyMixin):
    '''2c spinor basis: bare spinor integrals, or the X2C picture through the
    effective densities of ``with_x2c``.'''
    def _prop_expect(self, op, dm, origin, picture_change, x_response):
        return _spinor_expect(self, op, dm, origin, picture_change, x_response)


class SpinOrbitalPropertyMixin(PropertyMixin):
    '''GHF AO basis (alpha block, beta block).  The X2C helpers decouple in
    the spinor basis: the density is taken to it, D_sp = U^+ D U with the
    spherical->spinor transformation U = [ca; cb].'''
    def _prop_expect(self, op, dm, origin, picture_change, x_response):
        ref = self._prop_reference()
        mol = ref.mol
        if getattr(ref, 'with_x2c', None) is None or not picture_change:
            spec = int1e._resolve(op)
            name = spec['LL'].replace('_spinor', '')
            with mol.with_common_orig(origin):
                o = mol.intor(name, comp=spec['comp']).reshape(spec['comp'], mol.nao, mol.nao)
            n = mol.nao
            return (numpy.einsum('xij,ji->x', o, dm[:n, :n])
                    + numpy.einsum('xij,ji->x', o, dm[n:, n:]))
        ca, cb = mol.sph2spinor_coeff()
        u = numpy.vstack((ca, cb))
        dm_sp = u.conj().T @ dm @ u
        return _spinor_expect(self, op, dm_sp, origin, picture_change, x_response)


class ScalarPropertyMixin(PropertyMixin):
    '''Spin-free AO basis; (alpha, beta) densities are summed.'''
    def _prop_expect(self, op, dm, origin, picture_change, x_response):
        ref = self._prop_reference()
        mol = ref.mol
        if getattr(ref, 'with_x2c', None) is not None and picture_change:
            raise NotImplementedError('properties of the spin-free X2C reference')
        dm = numpy.asarray(dm)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        spec = int1e._resolve(op)
        name = spec['LL'].replace('_spinor', '')
        with mol.with_common_orig(origin):
            o = mol.intor(name, comp=spec['comp']).reshape(spec['comp'], mol.nao, mol.nao)
        return numpy.einsum('xij,ji->x', o, dm)


class FourComponentPropertyMixin(PropertyMixin):
    '''4c spinor basis (large, small components): O = O_LL + O_SS/(4c^2).'''
    def _prop_expect(self, op, dm, origin, picture_change, x_response):
        mol = self._prop_reference().mol
        return numpy.einsum('xij,ji->x', int1e.assemble_4c(mol, op, origin=origin), dm)

