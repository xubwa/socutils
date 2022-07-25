import somf
import numpy, scipy
from pyscf import scf, lib, gto
from pyscf.scf import ghf
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e
from pyscf.gto import mole
import x2camf_hf
class SOMFHelper(sfx2c1e.SpinFreeX2CHelper):
    '''
    '''
    hso = None
    def get_hcore(self, mol=None):
        #hcore = x2c.SpinOrbitalX2CHelper.get_hcore(self, mol)
        sf_hcore = sfx2c1e.SpinFreeX2CHelper.get_hcore(self)
        hcore = scipy.linalg.block_diag(sf_hcore, sf_hcore)
        if self.hso is None:
            mf = scf.sfx2c(scf.RHF(mol)).run()
            dm = mf.make_rdm1()
            self.hso = somf.get_soc_integrals(mf, dm=dm, pc1e='x2c1', pc2e='x2c', unc=True)
            #factor = (0.5 / lib.parameters.LIGHT_SPEED)**2
            #self.hso = factor * somf.get_fso2e_x2c(self.get_xmol()[0], dm)
        s = 1.0 * lib.PauliMatrices
        hso = numpy.einsum('sxy,spq->xpyq', -1.j * s, self.hso)
        #print(numpy.linalg.norm(hso))
        hcore = hcore - hso.reshape(hcore.shape)
        return hcore

def sox2c_ghf(mf):
    print(mf.__class__)
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SOMFHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SOMFHelper):
            mf.with_x2c = SOMFHelper(mf.mol)
            return mf
        else:
            return mf
        
    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__

    class SOX2C_GSCF(x2c._X2C_SCF, mf_class):
        __doc__ = doc + '''
        Attributes for spin-orbital spin-orbit mean field X2C:
            with_x2c : X2C object
        '''
        def __init__(self, mol, *args, **kwargs):
            mf_class.__init__(self, mol, *args, **kwargs)
            self.with_x2c = SOMFHelper(mf.mol)
            self._keys = self._keys.union(['with_x2c'])

        def get_hcore(self, mol=None):
            if mol is None: mol = self.mol
            return self.with_x2c.get_hcore(mol)

    with_x2c = SOMFHelper(mf.mol)
    return mf.view(SOX2C_GSCF).add_keys(with_x2c=with_x2c)

if __name__ == '__main__':
    mol = mole.Mole()
    mol = gto.M(
        verbose = 4,
        atom = [["Br" , (0. , 0.     , 0.)],
                [1  , (0. , 1.59, 0.0)]],
        basis = 'unc-cc-pvtz-dk',
    )
    mf = scf.DHF(mol)
    #e_dhf = mf.kernel()
    #print('E(dhf)=%.12g' %e_dhf) 
    #mf = x2camf.X2CAMF_RHF(mol)
    #e_amf = mf.kernel()
    #print('E(AMF)=%.12g', e_amf)
    gf = scf.GHF(mol)
    egf = gf.kernel()
    g2c = x2camf_hf.x2camf_ghf(gf)
    eso = g2c.kernel()
    spinor2c = x2camf_hf.X2CAMF_RHF(mol)
    e_spinor_so = spinor2c.kernel()
    gsomf = sox2c_ghf(gf)
    esfx2c_so = gsomf.kernel()
    gso1e = gf.x2c1e()
    eso1e = gso1e.kernel()
    x2c_spinor = x2c.RHF(mol)
    e_spinor = x2c_spinor.kernel()
    print('E(NR-GHF) = {:.12g}'.format(egf))
    print('E(X2CAMF, from GHF) = {:.12g}'.format(eso)) 
    print('E(SFX2C-SO) = {:.12g}'.format(esfx2c_so))
    print('E(SO1e, from GHF) = %.12g' % eso1e)
    print('E(SO1e, from spinor) = %.12g' % e_spinor)
