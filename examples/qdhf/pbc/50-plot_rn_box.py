#!/usr/bin/env python
'''Plot the first-order background Darwin term of Rn in a cubic box against L
(log-log) from 50-box_darwin_heavy.Rn.out, with Eq. (bgdarwin) as a line.'''
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

EH2MEV = 27211.386
# L = 6 bohr is left out: there the overlap with the neighbouring atoms adds
# genuine relativistic contributions (E1 / formula = 1.11)
rows = [l.split() for f in ('50-box_darwin_heavy.Rn.small.out', '50-box_darwin_heavy.Rn.out')
        for l in open(f) if l.startswith('Rn') and float(l.split()[2]) >= 8]
rows.sort(key=lambda r: float(r[2]))
L = numpy.array([float(r[2]) for r in rows])
e_sf = -numpy.array([float(r[4]) for r in rows]) * EH2MEV
e_so = -numpy.array([float(r[5]) for r in rows]) * EH2MEV
c, Z = 137.035999, 86
Lline = numpy.geomspace(4.5, 26, 100)
form = numpy.pi / (2 * c**2) * Z * Z / Lline**3 * EH2MEV
L_au = ((4.078 / 0.529177210903)**3 / 4) ** (1 / 3)

ink, muted, grid, blue, orange = '#0b0b0b', '#898781', '#e1e0d9', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': ink,
                     'xtick.color': muted, 'ytick.color': muted})
fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=200)
ax.loglog(Lline, form, color=muted, lw=1.2, ls='--', label=r'$(\pi/2c^2)\,ZN/L^3$')
ax.loglog(L, e_sf, 'o', ms=5, color=blue, mec='white', mew=0.8, label='SFX2C-1e')
ax.loglog(L, e_so, 's', ms=4.5, color=orange, mec='white', mew=0.8, label='X2C-1e')
ax.axvline(L_au, color=grid, lw=1)
ax.text(L_au * 1.05, 1.3, 'volume per atom\nof fcc Au', color=muted, fontsize=7, va='bottom')
ax.set_xlabel(r'$L$ (bohr)')
ax.set_xticks([5, 10, 20])
ax.set_xticklabels(['5', '10', '20'])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel(r'$-E^{(1)}$ per atom (meV)')
ax.grid(True, which='major', color=grid, lw=0.5)
ax.legend(frameon=False, fontsize=7, loc='upper right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig('50-rn_box.png')
fig.savefig('50-rn_box.pdf')
print('L_Au = %.2f bohr, formula there %.1f meV' % (L_au, numpy.pi / (2 * c**2) * Z * Z / L_au**3 * EH2MEV))
