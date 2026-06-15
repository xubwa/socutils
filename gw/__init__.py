#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Relativistic (spinor) GW module for socutils.

Currently provides the one-shot G0W0 approximation with analytic continuation
(:class:`SpinorGWAC`) and direct RPA correlation/total energy
(:class:`SpinorRPA`) on top of a spinor mean-field reference.
'''

from socutils.gw import spinor_gw_ac
from socutils.gw import spinor_rpa
from socutils.gw.spinor_gw_ac import SpinorGWAC
from socutils.gw.spinor_rpa import SpinorRPA


def GW(mf, frozen=None, freq_int='ac'):
    '''Construct a relativistic spinor GW object for a spinor mean-field ``mf``.

    Only analytic continuation (``freq_int='ac'``) is implemented so far.
    '''
    if freq_int.lower() == 'ac':
        return SpinorGWAC(mf, frozen)
    raise NotImplementedError(
        "spinor GW frequency integration '%s' not implemented (only 'ac')" % freq_int)


def RPA(mf, frozen=None):
    '''Construct a relativistic spinor direct-RPA object for ``mf``.'''
    return SpinorRPA(mf, frozen)


dRPA = RPA
