#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
'''
Generalized (spin-orbital, two-component) Kohn-Sham on the socutils GHF, with
the native non-collinear numint (socutils.dft.noncollinear.NumInt2C)::

    from socutils.dft import gks
    mf = gks.GKS(mol, xc='pbe').x2camf()
    mf.collinear = 'mcol'        # or 'ncol' / 'col'
    mf.kernel()
'''

from pyscf.lib import logger
from pyscf.dft import gks
from socutils.scf import ghf
from socutils.dft.noncollinear import NumInt2C


class GKS(gks.GKS, ghf.GHF):
    '''GKS with the X2C spin-orbit shortcuts of socutils.scf.ghf.GHF.'''

    def __init__(self, mol, xc='LDA,VWN'):
        gks.GKS.__init__(self, mol, xc)
        self.with_x2c = None
        self._numint = NumInt2C()

    def dump_flags(self, verbose=None):
        gks.GKS.dump_flags(self, verbose)
        if self._numint.collinear[0] == 'm':
            logger.info(self, 'multi-collinear backend = %s', self._numint.mcol_backend)
        return self

    get_hcore = ghf.GHF.get_hcore

    def to_hf(self):
        mf = ghf.GHF(self.mol)
        mf.with_x2c = self.with_x2c
        return self._transfer_attrs_(mf)
