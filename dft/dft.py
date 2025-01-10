#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
X2C 2-component DFT methods
'''

from socutils.scf import spinor_hf
from pyscf.dft import dks

class SpinorDFT(dks.KohnShamDFT, spinor_hf.SpinorSCF):
    def __init__(self, mol, xc='LDA,VWN'):
        spinor_hf.SpinorSCF.__init__(self, mol)
        dks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        spinor_hf.SpinorSCF.dump_flags(self, verbose)
        dks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert the input mean-field object to an X2C-HF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = self.view(spinor_hf.SpinorSCF)
        mf.converged = False
        return mf

UKS = SpinorDFT

class SymmDFT(dks.KohnShamDFT, spinor_hf.SymmSpinorSCF):
    def __init__(self, mol, xc='LDA,VWN', symmetry=None, occup=None):
        spinor_hf.SymmSpinorSCF.__init__(self, mol, symmetry=symmetry, occup=occup)
        dks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        spinor_hf.SpinorSCF.dump_flags(self, verbose)
        dks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert the input mean-field object to an X2C-HF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = self.view(spinor_hf.SpinorSCF)
        mf.converged = False
        return mf