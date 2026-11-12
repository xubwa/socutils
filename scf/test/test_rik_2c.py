'''Two-component J / K from a three-index factor (scf.rik_2c): one 4 nao x 4 nao Gram update per
factor block.  Against an explicit contraction of the unpacked factor, against the four-buffer loop
of scf.spinor_hf._DFJHF._get_k_mo (DFK_GRAM = False), for Kramers-paired and general orbital sets,
weighted / negative-sign sets (active natural orbitals), several sets in one pass, DF and CD factors.
'''
import unittest
import numpy
from pyscf import gto, lib
from socutils.scf import spinor_hf, rik_2c


def _ref_k(with_df, c, nao):
    '''sum_P B^P D B^P of D = c c^H in the sph GHF basis, the factor unpacked'''
    b = numpy.vstack([lib.unpack_tril(x) for x in with_df.loop()])
    b = b.reshape(-1, nao, nao)
    d = c @ c.conj().T
    k = numpy.zeros((2 * nao, 2 * nao), dtype=complex)
    for s in range(2):
        for t in range(2):
            dst = d[s * nao:(s + 1) * nao, t * nao:(t + 1) * nao]
            k[s * nao:(s + 1) * nao, t * nao:(t + 1) * nao] = lib.einsum('pij,jk,pkl->il', b, dst, b)
    return k


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz', verbose=0)
        mf = spinor_hf.KRHF(cls.mol).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.verbose = 0
        mf.kernel()
        cls.mf = mf
        cls.nao = cls.mol.nao_nr()
        c2 = numpy.vstack(cls.mol.sph2spinor_coeff())
        cls.cocc = c2 @ mf.mo_coeff[:, mf.mo_occ > 0]
        rng = numpy.random.default_rng(7)
        # a generic (spin-orbit mixed, non-Kramers) set: random complex rotations of occupied + virtuals
        x = rng.normal(size=(2 * cls.nao, 6)) + 1j * rng.normal(size=(2 * cls.nao, 6))
        cls.cgen = (c2 @ mf.mo_coeff) @ numpy.linalg.qr(rng.normal(size=(c2.shape[1], 6)))[0] + .1 * x

    def test_general_vs_explicit(self):
        vj, vk = rik_2c.get_jk(self.mf.with_df, [self.cgen])
        ref = _ref_k(self.mf.with_df, self.cgen, self.nao)
        self.assertLess(abs(vk[0] - ref).max(), 1e-12)
        self.assertEqual(vj, [])

    def test_kramers_vs_explicit(self):
        u = spinor_hf._kramers_unbarred(self.cocc, self.nao)
        self.assertIsNot(u, False)
        vj, vk = rik_2c.get_jk(self.mf.with_df, [u], kramers=[True])
        self.assertLess(abs(vk[0] - _ref_k(self.mf.with_df, self.cocc, self.nao)).max(), 1e-12)
        vj, vk2 = rik_2c.get_jk(self.mf.with_df, [self.cocc])
        self.assertLess(abs(vk[0] - vk2[0]).max(), 1e-12)

    def test_vs_old_loop(self):
        mf = self.mf
        dmt = lib.tag_array(mf.make_rdm1(), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        try:
            spinor_hf.DFK_GRAM = False
            vj0, vk0 = mf.get_jk(self.mol, dmt)
        finally:
            spinor_hf.DFK_GRAM = True
        vj1, vk1 = mf.get_jk(self.mol, dmt)
        self.assertLess(abs(vj1 - vj0).max(), 1e-12)
        self.assertLess(abs(vk1 - vk0).max(), 1e-12)

    def test_sets_one_pass(self):
        '''core (Kramers) + weighted active set + a negative-sign set + J, in one pass'''
        nao = self.nao
        u = spinor_hf._kramers_unbarred(self.cocc[:, :6], nao)
        act = self.cgen * numpy.sqrt(numpy.array([.9, .7, .5, .3, .2, .05]))
        dmj = numpy.random.default_rng(3).normal(size=(nao, nao))
        dmj = dmj + dmj.T
        rows, rik_2c.MAX_ROWS = rik_2c.MAX_ROWS, 8       # several factor blocks
        try:
            vj, vk = rik_2c.get_jk(self.mf.with_df, [u, act, act[:, :2]], [dmj], kramers=[True, False, False],
                                   signs=[1, 1, -1])
        finally:
            rik_2c.MAX_ROWS = rows
        self.assertLess(abs(vk[0] - _ref_k(self.mf.with_df, self.cocc[:, :6], nao)).max(), 1e-12)
        self.assertLess(abs(vk[1] - _ref_k(self.mf.with_df, act, nao)).max(), 1e-12)
        self.assertLess(abs(vk[2] + _ref_k(self.mf.with_df, act[:, :2], nao)).max(), 1e-12)
        vj0 = self.mf.with_df.get_jk(dmj, 1, True, False)[0]
        self.assertLess(abs(vj[0] - vj0).max(), 1e-12)

    def test_cd_factor(self):
        from socutils.cd import cd
        with_df = cd.CD(self.mol, tau=1e-6)
        with_df.verbose = 0
        with_df.build()
        vj, vk = rik_2c.get_jk(with_df, [self.cgen])
        self.assertLess(abs(vk[0] - _ref_k(with_df, self.cgen, self.nao)).max(), 1e-12)

    def test_h5_factor(self):
        from pyscf import df
        with_df = df.DF(self.mol, auxbasis='cc-pvdz-jkfit')
        with_df._cderi_to_save = lib.NamedTemporaryFile().name
        with_df.build()
        self.assertIsInstance(with_df._cderi, str)
        rows, rik_2c.MAX_ROWS = rik_2c.MAX_ROWS, 8
        try:
            vj, vk = rik_2c.get_jk(with_df, [self.cgen])
        finally:
            rik_2c.MAX_ROWS = rows
        self.assertLess(abs(vk[0] - _ref_k(self.mf.with_df, self.cgen, self.nao)).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
