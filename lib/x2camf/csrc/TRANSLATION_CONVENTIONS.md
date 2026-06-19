# C translation conventions for the X2CAMF core

The goal is a *faithful, line-by-line* translation of the C++ sources in
`/home/user/x2camf/src` into C99 (with OpenMP), so that the numerical
results agree with the C++/Eigen implementation to machine precision.
Do NOT restructure, "improve" or re-derive any numerics.

## Files

Foundation (already written, study them before translating):
- `x2c_mat.h` / `x2c_mat.c` — matrix layer (`mat_t`, `vmat_t`, `vecd_t`,
  `vvecd_t`, `dvec_t`) and LAPACK-backed `mat_sym_eig`, `mat_inverse`,
  `mat_solve_vec`.
- `x2c_general.h` / `x2c_general.c` — `speedOfLight`, `factorial`,
  `double_factorial`, `CG_wigner_*`, `evaluateChange`,
  `matrix_half(_inverse)`, `eigensolverG`, `X2C_*`, `Rotate_unite_irrep*`,
  `elem_list`, `int2eJK` + `int2e_free`.
- `x2c_int_sph.h` / `x2c_int_sph_basic.c` — `INT_SPH` struct, constructor,
  auxiliary radial integrals, generic radial/angular helpers.
  `x2c_int_sph_basic.c` is the style exemplar: it is the 1:1 translation
  of `src/int_sph_basic.cpp`; diff them to see every convention in action.
- `x2c_dhf_sph.h` — `DHF_SPH` struct + every function signature for the
  DHF translation units.

## Type and expression mapping

| C++ / Eigen                          | C                                            |
|--------------------------------------|----------------------------------------------|
| `MatrixXd A(n,m);`                   | `mat_t A = mat_new(n,m);` (zero-filled)      |
| `MatrixXd::Zero(n,m)`                | `mat_new(n,m)`                               |
| `A(i,j)`                             | `M(A,i,j)`                                   |
| `A.rows()`, `A.cols()`               | `A.rows`, `A.cols`                           |
| `A = B*C;`                           | `mat_assign(&A, mat_mul(B,C));`              |
| `A.transpose()`, `A.adjoint()` (real)| `mat_transpose(A)` (allocates)               |
| `A.inverse()`                        | `mat_inverse(A)` (allocates)                 |
| `A.block(i,j,r,c)`                   | `mat_block(A,i,j,r,c)` (allocates)           |
| `A += s*B`                           | `mat_add_inplace(A,B,s)`                     |
| `SelfAdjointEigenSolver`             | `mat_sym_eig(A, evals, &evecs)` (ascending)  |
| `B.partialPivLu().solve(b)`          | `mat_solve_vec(B, b, x)`                     |
| `VectorXd v(n)` / `::Zero(n)`        | `vecd_t v = vecd_new(n);`, access `v.d[i]`   |
| `vMatrixXd v(n)` / `v.resize(n)`     | `vmat_t v = vmat_new(n);`                    |
| `vVectorXd`                          | `vvecd_t`                                    |
| `std::vector<double>` + `push_back`  | `dvec_t` + `dvec_push`                       |
| `max/min`                            | `MAX`/`MIN` macros from x2c_mat.h            |
| member access `foo_`                 | `self->foo`                                  |
| `cout << ... << endl`                | `printf(...)` with the same message          |
| `exit(99)`                           | keep `exit(99)`                              |
| `assert(...)`                        | omit (the checks duplicate guard clauses)    |

Notes:
- Keep `pow(-1, n)` calls exactly as written; keep all floating-point
  expression structure, parenthesization and evaluation order unchanged.
- Keep all integer arithmetic (including intentional integer division
  like `lp*(lp+1)/2`) in int, exactly as in the C++ source.
- `int n = some_double / int_expr;` translates as
  `int n = (int)(some_double / int_expr);` (C++ implicit conversion
  truncates toward zero — do the same).
- Keep every `#pragma omp parallel for` exactly where it is, and keep the
  same variable scoping (declare loop-local variables inside the loop
  body so the parallel region is correct).
- C99 VLAs are allowed and encouraged where the C++ code uses them
  (e.g. `double tmpJ[size_i*size_i][size_j*size_j];` compiles in C99).
  Heads-up: arrays of `vector<double>` like
  `vector<double> arr[N][a][b];` become VLA arrays of `dvec_t`
  (`dvec_t arr[N][a][b];` followed by `memset(arr, 0, sizeof(arr));`,
  then `dvec_push(&arr[i][j][k], x)`, and `dvec_free` at the end).
- `irrep_list(ir).l` → `self->irrep_list[ir].l`; `shell_list(i).exp_a(j)`
  → `self->shell_list[i].exp_a.d[j]`; `shell_list(i).coeff.rows()` →
  `self->shell_list[i].coeff.rows`.
- `int2eJK` allocation: translate `new double**[n]` chains literally with
  `malloc` (e.g. `h.J = (double****)malloc(n * sizeof(double***));`).
  Keep the same nesting and index layout.
- Returning `vMatrixXd` by value → return `vmat_t` (deep ownership
  transfers to the caller).
- Out-parameters `MatrixXd& fock` that are resized+assigned inside →
  `mat_t* fock`, write with `mat_assign(fock, ...)` or allocate with
  `mat_assign(fock, mat_new(n,n))` then fill via `M(*fock,i,j)`.
- Temporaries: free what you allocate (`mat_free`) once no longer used;
  do this pragmatically — correctness has absolute priority over
  leak-freedom, never free something still in use.
- Timing helpers `countTime/printTime` in DHF code: either implement a
  trivial clock()-based version locally (static) or drop the timing
  *prints* — but keep all other printout behavior gated by `printLevel`
  identical.

## Compile check

Every translated file must compile cleanly:

    cd /home/user/x2camf-c/csrc
    gcc -std=gnu99 -fopenmp -c -Wall -Wextra -I. <file>.c -o /tmp/<file>.o

(Warnings about intentionally unused parameters can be silenced with
`(void)param;` as in x2c_int_sph_basic.c.)
