# socutils examples

Runnable examples matching the [documentation](https://xubwa.github.io/socutils/).
They use the canonical `.x2camf()` / `.x2cmp()` driver API.

| File | Topic |
| --- | --- |
| `00-spinor_x2camf.py` | spinor HF with X2CAMF (`spinor_hf.SCF(mol).x2camf()`) |
| `01-spinor_x2cmp.py` | the `x2cmp` flavor and toggling Gaunt/Breit |
| `02-kramers_restricted.py` | Kramers-restricted SCF (`spinor_hf.KRHF`, needs zquatev) |
| `03-symmetry_atom.py` | symmetry-adapted SCF, atom (`symmetry='sph'`) |
| `04-symmetry_linear.py` | symmetry-adapted SCF, linear molecule (must be on z) |
| `05-density_fitting.py` | density fitting via `.density_fit()` |
| `06-ghf_x2camf.py` | GHF (spin-orbital) driver (`ghf.GHF(mol).x2camf()`) |
| `07-somf_helper.py` | constructing the X2C SOC helper directly |
| `08-casci.py` | CASCI on a spinor reference |
| `09-four_component.py` | four-component Dirac-Hartree-Fock |
| `10-casscf.py` | CASSCF orbital optimization (needs zquatev) |

Most examples need the optional `x2camf` package for the spin-orbit integrals;
the Kramers-restricted example additionally needs `zquatev`. See the
[installation guide](https://xubwa.github.io/socutils/install.html).
