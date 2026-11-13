#
# Active-space two-electron integrals (tu|vw) without the global DF tensor
# (HANDOFF Open item 1), one call for the CASCI/CAHF drivers:
#
#     eri = eri_active(mf, mo_act)          # (n, n, n, n), chemists' order
#
# 'three_index' (pbc RSDF, lr='gspace'): SR from the SR three-index integrals
# half-transformed to the active pair densities as they are computed,
# B_P,tu, fitted with the SR metric, (tu|vw)_SR = B^T M_SR^-1 B (global SR fit
# now; the local SR fit of Open item 2 replaces W B); LR exact from rho_tu(G).
# The kernel is the one of the lr='gspace' vectors (G = 0 part of the SR kernel
# kept out), i.e. what _CDERIS contracts, so energy_nuc() and the CASCI
# conventions are unchanged.  See RSDF.active_vectors.
#


def eri_active(mf, mo_act, method='three_index'):
    if method == 'three_index':
        df = getattr(mf, 'with_df', None)
        if df is None or not hasattr(df, 'active_eri'):
            raise TypeError('eri_active: three_index needs a periodic RSDF (lr=gspace) mean field')
        return df.active_eri(mo_act)
    raise NotImplementedError('eri_active method %s' % method)
