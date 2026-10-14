'''Periodic CAHF (mcscf.cahf_diis with zcahf.CAHF on spinor_scf) and the
fractional-occupation SCF that seeds it: O atom, 2p^4 over the six 2p
spinors, in a box.  Both extrapolate (A/L^3) to the molecular results with
the same auxiliary basis; the fractional SCF only does so with the exchange
G = 0 term in the form -nu tr(DSDS)/2 (with -nu N/2 it was 5e-2 Eh off).
'''
import io
import contextlib
import unittest
from pyscf import gto, df
from socutils.scf import spinor_hf, addons
from socutils.mcscf import zmcscf, zcahf, cahf_diis
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    from socutils.pbc import spinor_scf
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False


def _cahf(mf, mol):
    mf.verbose = 0
    mf.conv_tol = 1e-11
    addons.projected_frac_occ(mf, addons.spinor_ao_indices(mol, 0, l=1), 6, 4)
    e_frac = mf.kernel()
    mo, ncore = addons.cas_order(mf)
    mc = zmcscf.CASSCF(mf, 6, 4, ncore=ncore)
    mc.verbose = 0
    mc.max_cycle_macro = 50
    mc.fcisolver = zcahf.CAHF(mol)
    with contextlib.redirect_stdout(io.StringIO()):
        cahf_diis.mc_diis(mc, mo, verbose=0, conv_tol=1e-12, conv_tol_grad=1e-7)
    return mc.e_tot, e_frac


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    def test_box_limit_is_the_molecule(self):
        mol = gto.M(atom='O 0 0 0', unit='B', basis='cc-pvdz', verbose=0, spin=2)
        aux = df.addons.aug_etb(mol, beta=2.0)
        ref = _cahf(spinor_hf.KRHF(mol).density_fit(auxbasis=aux), mol)
        res = {}
        for L in (20., 26.):
            cell = cr.molecule_in_box(mol, L)
            r = RSDF(cell, aux, precision=1e-11, nuc_zeta=1e14).build()
            res[L] = _cahf(spinor_scf.periodic(spinor_hf.KRHF(cell), r), cell)
        (a, ea), (b, eb) = sorted(res.items())
        for k in (0, 1):     # CAHF, fractional SCF
            e_inf = (eb[k] * b**3 - ea[k] * a**3) / (b**3 - a**3)
            self.assertLess(abs(e_inf - ref[k]), 5e-5)


if __name__ == '__main__':
    unittest.main()
