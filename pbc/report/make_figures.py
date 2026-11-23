#!/usr/bin/env python
'''Figures of pbc/REPORT.md (SVG, written next to this script).  The numbers are those of the report's tables;
their sources (run logs, commits) are given there.    python make_figures.py'''
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']   # fixed categorical order
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': GRID, 'svg.fonttype': 'none',
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def _clean(ax, grid='x'):
    for s in ('top', 'right', 'left' if grid == 'x' else 'bottom'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def build_breakdown():
    '''Eu:YSO 32 atoms, streamed V, A, rank 0 of 8 (4 nodes x 2 ranks x 24 threads, 50 GB per rank).'''
    versions = ['Gram pipeline +\nper-label FT', 'analytic FT,\nnumpy per block', 'cached separable FT,\nOpenMP kernels']
    parts = [('Gram columns, Ginv (xi)', [4171, 4383, 4289]), ('V', [1728, 1727, 1745]), ('FFT', [1658, 1656, 1672]),
             ('aux FT', [2671, 4015, 328]), ('A GEMM', [614, 1183, 1129]), ('waiting for other ranks', [1149, 917, 182]),
             ('Alltoallv', [161, 169, 168])]
    totals = [12366, 14262, 9729]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    left = [0, 0, 0]
    y = [2, 1, 0]
    for (name, vals), col in zip(parts, SERIES):
        ax.barh(y, vals, left=left, height=0.5, color=col, edgecolor=SURFACE, linewidth=1.5, label=name)
        left = [a + b for a, b in zip(left, vals)]
    for yi, tot in zip(y, totals):
        ax.text(tot + 150, yi, '%d s' % tot, va='center', ha='left', color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(versions, color=INK)
    ax.set_xlim(0, 16500)
    ax.set_xlabel('wall time per rank (s)')
    _clean(ax)
    ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.42, 1.0), handlelength=1.2,
              columnspacing=1.2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_build_breakdown.svg'))


def memory_budget():
    '''Eu:YSO 32 atoms, 2 bigmem nodes x 4 ranks x 12 threads: seconds per interpolation point.'''
    parts = ['Gram columns,\nGinv (xi)', 'FFT', 'V', 'A GEMM', 'aux FT', 'Alltoallv']
    a = [0.535, 0.129, 0.132, 0.128, 0.032, 0.014]
    b = [0.192, 0.131, 0.135, 0.042, 0.049, 0.108]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    x = range(len(parts))
    w = 0.34
    ba = ax.bar([i - w / 2 - 0.02 for i in x], a, w, color=SERIES[0], label='50 GB per rank (blocks of 177 points)')
    bb = ax.bar([i + w / 2 + 0.02 for i in x], b, w, color=SERIES[1], label='170 GB per rank (blocks of 1057 points)')
    for bars in (ba, bb):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, '%.3f' % r.get_height(), ha='center',
                    va='bottom', fontsize=7.5, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(parts, color=INK)
    ax.set_ylabel('s per interpolation point')
    ax.set_ylim(0, 0.62)
    _clean(ax, 'y')
    ax.legend(frameon=False, loc='upper right', labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_memory_budget.svg'))


def accuracy():
    '''|E(THC route 1) - E(reference)| of the SCF energy per cell.'''
    rows = [('quartz, 9 atoms (vs exact SR)', 1.4e-5, 0), ('EuOCl, 6 atoms (vs RSDF)', 4.9e-5, 0),
            ('Eu:YSO, 32 atoms (vs RSDF)', 2.6e-4, 0),
            ('LiH 2x2x2, aux auxgen 1e-8 (vs RSDF)', 3.9e-5, 1), ('LiH 2x2x2, aux acCD 1e-3 (vs RSDF)', 1.8e-4, 1),
            ('LiH 2x2x2, aux aug_etb beta 2 (vs RSDF)', 2.5e-3, 1)]
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    n = len(rows)
    for i, (name, err, grp) in enumerate(rows):
        yy = n - 1 - i
        ax.plot([1e-6, err], [yy, yy], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter([err], [yy], s=46, color=SERIES[grp], zorder=3, edgecolor=SURFACE, linewidth=1.5,
                   label=('production auxiliary basis (auxgen, uncontracted)', 'auxiliary basis varied')[grp] if i in (0, 3) else None)
        ax.text(err * 1.25, yy, '%.1e' % err, va='center', ha='left', fontsize=8, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
    ax.set_xscale('log')
    ax.set_xlim(1e-6, 2e-2)
    ax.set_xlabel('|E(THC) - E(reference)| per cell (Eh)')
    _clean(ax)
    ax.legend(ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.3, 1.0), labelcolor=INK, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_accuracy.svg'))


if __name__ == '__main__':
    build_breakdown()
    memory_budget()
    accuracy()
