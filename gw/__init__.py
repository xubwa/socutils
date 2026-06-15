#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Relativistic (spinor) GW module for socutils.

.. warning::

   **WORK IN PROGRESS.**  This package is under active development and its API
   and numerical defaults may change.  It currently provides one-shot G0W0 with
   analytic continuation (:class:`SpinorGWAC`) and direct-RPA correlation/total
   energy (:class:`SpinorRPA`) on top of a spinor mean-field reference.  Both
   are validated against the restricted PySCF implementations in the
   non-relativistic limit and run on X2CAMF spin-orbit references, but features
   such as contour-deformation GW, self-consistency, BSE and lower-scaling
   algorithms are not yet implemented.
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
