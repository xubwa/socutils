# Stage-1 raw data

Output of `stage1_pw.py` (JSON; numbers as 40-digit strings).  Naming:
`<system>_<grid>.json`, grids `dm` = med (nb=250), `df`/`ch` = fine (nb=300;
`ch` also has the LL/LS/SS channel split), `dx` = xfine (nb=360), `qm` = quad on med.

| file | system | backend | grid |
|---|---|---|---|
| nr_dm / nr_double_fine / nr_dx | Schroedinger | double | med / fine / xfine |
| z2_dm, z2_dx | Z=2 | double | med, xfine |
| z10_dx, z10_ch | Z=10 | double | xfine (ch: with channels) |
| z30_dm, z30_ch, z30_dx | Z=30 | double | med, fine, xfine |
| z50_ch | Z=50 | double | fine |
| z80_dm, z80_ch | Z=80 | double | med, fine |
| z80_dx_COLLAPSED | Z=80 | double | xfine: variational collapse (1s at -2.899), kept as the record of the failure; not used |
| anchor_c1e3_qm, anchor_c1e5_qm | c/Z = 1370, 137036 | quad | med |
| z30_qm_l20, z80_qm_l20 | Z=30, 80, l=20 only | quad | med |

Speed of light: CODATA 2018, c = 137.035999084.
