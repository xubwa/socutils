'''Periodic X2CAMF (spinor_scf): range-separated X2C-1e + atomic mean field
(RangeSeparatedX2CAMFHelper, mf.x2camf()), and the one-centre approximations
of X2CMP (OneCenterX2CMPHelper, mf.x2cmp('x2cmp_1c' or 'x2camf_1c')).

* An atom in a box: the relativistic part E(X2CAMF) - E(NR) equals that of the
  molecular atom, with point nuclei (RSDF zeta 1e14) and with the Gaussian
  nuclear model (cell.nucmod 'G', RSDF zeta = nuclear_zeta(cell)).  The total
  energy carries the periodic NR error of the fit (4.6e-5 Eh for Kr/dyall-v2z,
  extrapolated from L = 22, 28), which cancels in the difference.
* No background term: the relativistic part of the periodic one-electron
  Hamiltonian of an atom in a box is that of the isolated atom (first order,
  fixed density), independent of L.  A full X2C-1e decoupling of the Ewald
  potential would give -(pi/2c^2) Z N / L^3 here.
* For a molecule (HI) rs1e-X2CAMF reproduces the molecular X2CAMF energy to
  ~1.5e-5 Eh at omega = 1 (dyall-v2z; the untransformed long-range potential
  of the other nucleus); the one-centre X2CAMF blocks to ~6e-5 Eh (all
  multi-centre relativistic terms dropped).
'''
import unittest
import warnings
from pyscf import gto, df
from socutils.scf import spinor_hf
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    from socutils.pbc import spinor_scf
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False


def _quiet(obj):
    obj.verbose = 0
    return obj


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    def _atom_in_box(self, nucmod, L=28.):
        # relativistic part E(X2CAMF) - E(NR) of an atom in a box against the
        # molecular atom: the periodic NR error (fit, finite size; 4.6e-5 Eh for
        # Kr/dyall-v2z at L = 22, 28 extrapolated) cancels in the difference
        atm = gto.M(atom='Kr 0 0 0', unit='B', basis='dyall-v2z', verbose=0, nucmod=nucmod)
        aux = df.addons.aug_etb(atm, beta=2.0)
        ref = (_quiet(spinor_hf.SpinorSCF(atm).x2camf().density_fit(auxbasis=aux)).kernel()
               - _quiet(spinor_hf.SpinorSCF(atm).density_fit(auxbasis=aux)).kernel())
        cell = cr.molecule_in_box(atm, L)
        cell.nucmod = nucmod
        cell.build(False, False)
        zeta = spinor_scf.nuclear_zeta(cell)
        r = RSDF(cell, aux, precision=1e-11, nuc_zeta=1e14 if zeta is None else zeta).build()
        e = (_quiet(spinor_scf.SpinorRSDF(cell, r).x2camf()).kernel()
             - _quiet(spinor_scf.SpinorRSDF(cell, r)).kernel())
        return e, ref

    def test_atom_point_nucleus(self):
        e, ref = self._atom_in_box({})
        self.assertLess(abs(e - ref), 1e-6)

    def test_atom_gaussian_nucleus(self):
        e, ref = self._atom_in_box({'Kr': 'G'})
        self.assertLess(abs(e - ref), 1e-6)

    def test_one_centre_error_molecule(self):
        mol = gto.M(atom='I 0 0 0; H 0 0 3.04', unit='B', basis='dyall-v2z', verbose=0)
        aux = df.addons.aug_etb(mol, beta=2.0)
        e_full = _quiet(spinor_hf.SpinorSCF(mol).x2camf().density_fit(auxbasis=aux)).kernel()
        hlp = spinor_scf.OneCenterX2CMPHelper(mol, 'x2camf_1c')
        h1c = mol.intor('int1e_kin_spinor') + mol.intor('int1e_nuc_spinor') + hlp.get_correction()
        mf = _quiet(spinor_hf.SpinorSCF(mol).density_fit(auxbasis=aux))
        mf.get_hcore = lambda *args, **kwargs: h1c
        self.assertLess(abs(mf.kernel() - e_full), 1e-4)

    def test_rs1e_error_molecule(self):
        mol = gto.M(atom='I 0 0 0; H 0 0 3.04', unit='B', basis='dyall-v2z', verbose=0,
                    nucmod={'I': 'G', 'H': 'G'})
        aux = df.addons.aug_etb(mol, beta=2.0)
        e_full = _quiet(spinor_hf.SpinorSCF(mol).x2camf(with_pcc=False).density_fit(auxbasis=aux)).kernel()
        zeta = spinor_scf.nuclear_zeta(mol)
        vne = mol.intor('int1e_nuc')
        for omega, tol in ((1e-3, 1e-7), (1.0, 3e-5)):
            hrs = spinor_scf.RangeSeparatedX2CAMFHelper(mol, omega, zeta).get_hcore(vne)
            mf = _quiet(spinor_hf.SpinorSCF(mol).density_fit(auxbasis=aux))
            mf.get_hcore = lambda *args, hrs=hrs, **kwargs: hrs
            self.assertLess(abs(mf.kernel() - e_full), tol)

    def test_no_background_term(self):
        # first-order relativistic one-electron energy of an atom in a box:
        # Tr D (h - T - V_sr), with V_ne cancelling, equals the isolated atom
        import numpy
        from pyscf.lib import param
        atm = gto.M(atom='Ar 0 0 0', unit='B', basis='dyall-v2z', verbose=0)
        mf = _quiet(spinor_hf.SpinorSCF(atm).x2camf(with_pcc=False))
        mf.kernel()
        dm = mf.make_rdm1()

        def e1(mol):
            hlp = spinor_scf.RangeSeparatedX2CAMFHelper(mol, 1.0, 1e14)
            v = numpy.zeros((mol.nao_nr(), mol.nao_nr()))
            dh = hlp.get_hcore(v)                     # h without the V_ne - V_sr term
            t = numpy.asarray(mol.pbc_intor('int1e_kin', hermi=1)) if getattr(mol, 'dimension', 0) \
                else mol.intor('int1e_kin')
            dh = dh - spinor_hf.sph2spinor(mol, spinor_scf._sph_blockdiag(t + hlp.v_sr))
            return numpy.einsum('ij,ji', dm, dh).real

        e0 = e1(atm)
        z = atm.atom_charge(0)
        for L in (16., 20.):
            cell = cr.molecule_in_box(atm, L)
            full_ewald = -numpy.pi / (2 * param.LIGHT_SPEED**2) * z * z / L**3
            self.assertLess(abs(e1(cell) - e0), 1e-2 * abs(full_ewald))

    def test_flavors(self):
        atm = gto.M(atom='He 0 0 0', unit='B', basis='cc-pvdz', verbose=0)
        cell = cr.molecule_in_box(atm, 8.)
        r = RSDF(cell, df.addons.aug_etb(atm, beta=2.0), nuc_zeta=1e14)
        mf = spinor_scf.SpinorRSDF(cell, r)
        for bad in ('x2cmp', 'sfmp', 'x2camf', 'x2camf_au'):
            with self.assertRaises(ValueError):
                mf.x2cmp(bad)
        self.assertEqual(mf.x2cmp().with_x2c.flavor, 'x2cmp_1c')
        self.assertEqual(mf.x2cmp('x2camf_1c').with_x2c.flavor, 'x2camf_1c')
        self.assertIsInstance(mf.x2camf().with_x2c, spinor_scf.RangeSeparatedX2CAMFHelper)
        # nuclear-model consistency
        with self.assertRaises(ValueError):
            cg = cr.molecule_in_box(atm, 8.)
            cg.nucmod = 'G'
            cg.build(False, False)
            spinor_scf.SpinorRSDF(cg, RSDF(cg, df.addons.aug_etb(atm, beta=2.0), nuc_zeta=1e14))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            spinor_scf.check_nuclear_model(cell, RSDF(cell, None, nuc_zeta=1e12))
            self.assertTrue(any('finite-nucleus' in str(x.message) for x in w))


if __name__ == '__main__':
    unittest.main()
