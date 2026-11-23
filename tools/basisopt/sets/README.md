# Relativistic contracted basis sets built with `tools/basisopt`

Archive of the sets made on Rockfish (Oct-Nov 2026) with the X2CAMF atomic program
(`basisopt.engine`) and the recontraction tool (`basisopt.contract`), together with the
scripts that generated them and their checks.  The scripts are kept as they were run: their
paths point to the original Rockfish layout (`~/scr4/zmr_runs/basis_opt/...`, DIRAC's
`basis/dyall.*` files; `common/dirac_basis.py` is the DIRAC reader they import as
`../audit/dirac_basis`).

Naming of the contraction kinds:

* `-SF` / `-X2C`: spin-free (SF-X2C-1e / sf-X2CAMF) atomic SCF, one contracted function per
  occupied shell; plain scalar sets.
* `-SOG`: j-adapted, contracted from the X2CAMF (Dirac-Coulomb-Gaunt AMF) atomic SCF; for
  l >= 1 the j = l-1/2 and j = l+1/2 shells have their own contractions.  GENBAS entries carry
  the `n l k ...` information line; the PySCF modules use the kappa format
  `[l, kappa, [exp, c...], ...]` and are only usable as two-component (spinor) bases.

Every GENBAS file is in CFOUR format; every `pyscf/*.py` holds one variable per element in
PySCF's internal format (`genbas.read_pyscf` / `gto.M(basis={'Yb': module.Yb})`).
All sets use a Gaussian nuclear model in the contraction.

| directory | sets | status |
|---|---|---|
| `lanthanides/` | `<Ln>:DYVTZ-SOG-T07`, `<Ln>:DYVTZ-SF-T07` for Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb | delivered (Eu:YSO project), checked |
| `yttrium/` | `Y:DYVDZ-SF`, `Y:DYVDZ-SF-P1`, `Y:DYVDZ-SOG`, `Y:DYVTZ-SF-P1`, `Y:DYVTZ-SF-T05`, `Y:DYVTZ-SF-T07` | delivered, checked |
| `ca_n_h/` | Ca/N/H aug-cc-pV{T,Q,5}Z X2C (SF and SOG); Ca aug-cc-pwCV{T,Q}Z | valence and pwCVTZ delivered; Ca pwCVQZ is a draft |
| `oxygen/` | O cc-pVXZ-X2C(-TP), aug, TP-SOG (X = D-5); pwCV{T,Q}Z-X2C-TP | delivered (O2 ZFS) |
| `europium_dz/` | `EU:DYVDZ-SF`, `EU:DYVDZ-SOG` (neutral 4f7 6s2) | delivered |
| `thorium_uranium/` | O/F -TP-SOG, Th/U DYVnZ-SOG drafts, U aug / core-valence (v3) / EA stopgap (-P1, -P1-6D) | drafts and stopgaps |
| `cfour_extract/` | 121 PySCF modules extracted from the CFOUR x2cso GENBAS (-SO, -SF, -X2C, -DE4 families) | staging, not registered |
| `notes/` | `STATUS.md` (project log, the detailed record of every decision and number), `TASK.md` (project brief) | |

## lanthanides: `<Ln>:DYVTZ-{SOG,SF}-T07` (`scripts/make_ln.py`)

For Ln3+ in a crystal (embedded cluster and periodic X2CAMF CAHF/CASCI, Ln on the Y site of
Y2SiO5).

* primitives: dyall.v3z (30s24p16d11f3g2h); every exponent < 0.07 removed (the periodic
  decontracted overlap/kinetic metric becomes singular otherwise); tight s/p all kept;
* contraction: the trivalent ion, configuration average of 4f^n (n electrons over the 14 4f
  spinors); one contracted function per occupied shell (1s-5s, 2p-5p, 3d-4d, 4f);
* uncontracted: Dyall's 4f_tz combined valence set counted from the smallest remaining
  exponent: s 1,3,4,6; p 1,3,5,6; d 1-5; f 1,3,4,5; g, h all;
* size: SOG 9s12p9d6f3g2h (181 scalar AOs, 304 spinors); SF 9s8p7d5f3g2h (152 AOs).

Checks (`scripts/check_ln.py`, `checks/check_<El>.out`, X2CAMF configuration-average SCF):
truncation changes the energy by < 0.006 mEh, and zeta_4f, <r^-3>_4f and the Slater
integrals F2/F4/F6(4f,4f) by < 2e-5 relative; contraction errors of the 4s-5p, 4d, 4f
spinor energies < 6e-10 Eh.  Not suitable for neutral Ln (no 6s/5d contraction, diffuse
functions removed).

## yttrium (`scripts/make_y*.py`, `polar*.py`)

* `Y:DYVDZ-SF`: dyall.v2z, sf-X2CAMF, 4d1 5s2 average; uncontracted 2nd s, 2nd p, 1st+2nd d
  (Dyall 4d DZ rule), f all; 6s4p4d2f.  Y+ polarizability contraction error -11 % (no 5p).
* `Y:DYVDZ-SF-P1`, `Y:DYVTZ-SF-P1`: plus the most diffuse p uncontracted; Y+ alpha error
  +0.3 % / -0.3 %, Y3+ -3.7 % / -1.8 %.  TZ from dyall.v3z without the 4s4p f 1.115,
  Dyall 4d TZ rule (s 1,3; p 2,3; d 1-3), 3f2g.
* `Y:DYVTZ-SF-T05`, `-T07`: dyall.v3z with exponents < 0.05 / < 0.07 removed, for Y3+ in the
  periodic cell only (Y3+ RHF +2.3 / +2.4 mEh, alpha +1.6 % against the full uncontracted
  v3z; Y+ and neutral Y unusable).
* `Y:DYVDZ-SOG`: j-adapted counterpart of DYVDZ-SF (6s7p6d2f); spin-orbit splittings
  reproduced to 1.0000-1.0007.

## ca_n_h (`scripts/build_ca_val.py`, `build_ca_cv.py`, `write_mods.py`)

* H, N: Dunning aug-cc-pV{T,Q,5}Z primitives recontracted (SF and SOG).
* Ca TZ/QZ: Hill-Peterson aug-cc-pV{T,Q}Z-X2C primitives; Ca 5Z: cc-pVQZ-X2C s/p (HF limit)
  with 4 outer s/p free + Koput-Peterson cc-pV5Z d/f/g/h, aug functions at ratio 2.5.
  Only occupied orbitals contracted: d fully uncontracted; the unoccupied 4p contraction of
  the source replaced by one more free p primitive (Hill-Peterson contract it from the 3P
  state - see notes/STATUS.md).
* Ca aug-cc-pwCVTZ: Hill-Peterson core-correlating exponents pinned into the primitive set,
  neighbouring HF exponents re-optimised with ratios >= 1.6 (HF +0.13 mEh vs valence).
  aug-cc-pwCVQZ (`GENBAS.canh_cv`) is a draft: HF +0.21 mEh, two ratios of 1.5.
* SOG sets are near-linearly dependent in the scalar AO representation (1e-9); use them as
  spinor bases.

## oxygen, europium_dz, thorium_uranium

See `notes/STATUS.md` for the construction, the decisions behind them and the checks.
O: cc-pVXZ-X2C + the ccJ-pVXZ tight p (D 106.2, T 206.76, Q 380.52, 5Z 1173.0), scalar, and
pwCV{T,Q}Z-X2C-TP (Dunning CV functions uncontracted).  Th/U sets are drafts; the U EA sets
are stopgaps (most diffuse p freed, -6D adds g/h), not a final design.

## cfour_extract

`scripts/inventory.py` indexes the relativistic families of the CFOUR x2cso GENBAS,
`dupcheck.py` compares entries that occur twice, `extract.py` writes `modules/` (names:
A -> aug-, P/PC/PWC -> cc-p/cc-pc/cc-pwc, DY -> dy-, suffix lower case; `_X2C` merged into
`-x2c`) and `report.txt` (44 empty -SO information lines filled from `proposed_lines.tsv`,
22 malformed entries skipped, 87 near-dependent entries flagged).  `validate.py` loads every
module through `socutils.gto.basis.register`.  The modules are not registered by default.
