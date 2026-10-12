'''
Multiwavelet (MRCPP / VAMPyR) tools: closed-shell nonrelativistic
Hartree-Fock in the complete-basis-set limit and the c^-2 / c^-4
coefficients of the sf-X2C-1e picture-change error of multipole operators.

Requires the ``vampyr`` package (github.com/MRChemSoft/vampyr).
'''

from socutils.mw.hf import MWContext, RHF, project_mos
from socutils.mw.picture_change import Multipole, pc_coefficients
