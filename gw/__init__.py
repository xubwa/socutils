#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Relativistic (spinor) GW module for socutils.

Currently provides the one-shot G0W0 approximation with analytic continuation
on top of a spinor mean-field reference (:class:`SpinorSCF` and subclasses).
'''

from socutils.gw import spinor_gw_ac
from socutils.gw.spinor_gw_ac import SpinorGWAC


def GW(mf, frozen=None, freq_int='ac'):
    '''Construct a relativistic spinor GW object for a spinor mean-field ``mf``.

    Only analytic continuation (``freq_int='ac'``) is implemented so far.
    '''
    if freq_int.lower() == 'ac':
        return SpinorGWAC(mf, frozen)
    raise NotImplementedError(
        "spinor GW frequency integration '%s' not implemented (only 'ac')" % freq_int)
