# Interpolation of the external resolvent sums

Theory of `zmr.uc.engine.chebyshev_weights` / `resolvent_nodes` and of the
contractions that use them.  Implementation notes and the validation runs
are in `performance.md` section 6, the tests in
`zmr/tests/test_interpolation.py`.

## 1. The quantity

For one perturber class the first-order space is a product: an external
label `e` (a set of core and virtual indices with the orbital-energy
combination `delta_e`) and an active function `k` with energy `lam_k` (an
eigenvalue of the Koopmans problem for PC, a sector eigenvalue for UC).  The
Dyall Hamiltonian is diagonal in this basis, and the class contributes

    a[I, J] = - sum_{k, e} conj(u^I_ek) u^J_ek / (lam_k + delta_e - s_J),
    u^J_ek  = sum_X v[e, X] t[J, k, X],

to the second-order effective Hamiltonian; `t[J, k, X] = <b_k|O_X|Psi_J>`
are the projected amplitudes of model state `J`, `v[e, X]` the integrals,
`s_J = E0_J - E_core`.  Summing the external labels first,

    a[I, J] = - sum_k t[I, k, :]^+ W(lam_k - s_J) t[J, k, :],            (1)

    W(d) = sum_e P_e / (d + delta_e),   P_e[X, Y] = conj(v[e, X]) v[e, Y].   (2)

`W` is a matrix in the active operator labels (`n_X x n_X`) that depends on
one real variable `d`.  Everything that is expensive in the doubly external
classes is in `W`: the sum over `e` runs over up to `n_c^2 n_v^2 / 4` labels.

## 2. How many sums are needed

(1) needs `W` at the arguments `d = lam_k - s_J` at which amplitudes are
present.  Their number `N_d` is

* **ccvv**: the active part of the operator is the identity, the functions
  are the model states themselves, `t[J, k]` is non-zero only for `k` in the
  level of `J`.  With state-specific E0 `lam_k = s_J` there, so `d = 0` for
  every state: one sum, the MP2-like expression, whatever the number of
  states.  With a common E0, one argument per level.
* **all other classes**: `k` runs over the whole first-order active space of
  dimension `m` (tens to thousands; it is set by the active space, and with
  the union contraction it grows with the number of states up to the sector
  dimension), and every model level `L` shifts all of them:
  `N_d = m L`.

So outside ccvv the number of sums is large already for one state, and it
grows with the number of levels.  For Nd3+ 4f^3 with 364 states (41
levels) and `m = 91` in cavv, `N_d` is several thousand.

## 3. W as a function of d

Each `P_e` is positive semidefinite of rank one, and all poles of (2) are at
`d = -delta_e <= -delta_min`.  Without intruders the needed arguments lie in
an interval `[lo, hi]` to the right of all poles,

    gap = delta_min + lo > 0,     width = hi - lo,

and there `W(d)` is analytic, positive semidefinite and monotonically
decreasing: a smooth function that is sampled at many points.  This is what
is exploited: `W` is computed at a few nodes and interpolated.

## 4. Interpolation at Chebyshev nodes

Map the interval to `[-1, 1]`, `x = (d - mid) / half`, `mid = (lo + hi) / 2`,
`half = width / 2`, and take as nodes the `n` roots of the Chebyshev
polynomial `T_n`, `x_m = cos((2m + 1) pi / 2n)`.  The interpolant is

    W_n(d) = sum_m l_m(d) W(d_m),                                         (3)

with the Lagrange polynomials `l_m` of the nodes.  Interpolation is linear,
so (3) acts on (2) term by term: `W_n = sum_e P_e p_n[g_e]` with
`g_e(d) = 1 / (d + delta_e)`.  It is enough to know the error for a single
pole.

**Error for a pole.**  Let `omega(x) = prod_m (x - x_m)` and
`g(x) = 1 / (x - p)` with `p` outside the nodes.  The function

    p_n(x) = (1 - omega(x) / omega(p)) / (x - p)

is a polynomial of degree `n - 1` (the numerator vanishes at `x = p`) and
equals `g` at every node (`omega(x_m) = 0`): it is the interpolant.  Hence

    (g - p_n)(x) / g(x) = omega(x) / omega(p),                            (4)

exactly.  For the Chebyshev nodes `omega = T_n / 2^(n-1)`, and the pole of
`g_e` is at `x = -z_e`, `z_e = (delta_e + mid) / half > 1`:

    relative error of term e at d  =  T_n(x) / T_n(-z_e),
    |T_n(x)| <= 1 on the interval,   |T_n(-z)| = cosh(n arccosh z)
                                               = (rho^n + rho^-n) / 2,
    rho = z + sqrt(z^2 - 1).                                              (5)

The bound `eps_e = 1 / T_n(z_e) < 2 rho_e^-n` decreases with `z_e`, so the
nearest pole, `z = (delta_min + mid) / half = 1 + 2 gap / width`, has the
largest error and all other terms are better.  The Chebyshev nodes are the
choice that minimizes `max |omega|` on the interval, i.e. the bound (4)
for a pole at a given distance.

**Error of the matrix and of the energy.**  With `eps = 1 / T_n(z)` every
term obeys `|g_e - p_n[g_e]| <= eps g_e`, and since `P_e >= 0` and `g_e > 0`

    - eps W(d)  <=  W(d) - W_n(d)  <=  eps W(d)                           (6)

in the order of Hermitian matrices.  For a diagonal element of (1) this
gives `|a_II - a_II,n| <= eps |a_II|` class by class (each term of the sum
over `k` is a non-negative quadratic form), and for the off-diagonal ones
`eps sqrt(|a_II| |a_JJ|)`-type bounds by the Cauchy-Schwarz inequality.  The
interpolation error is a relative error `eps` of the second-order energy of
the class, uniformly in the states.

**Number of nodes.**  `n` is chosen so that `2 rho^-n <= 2 exp(-D)`:

    n = ceil(D / ln rho) + 1,    D = INTERP_DIGITS = 40,                  (7)

`eps < 8.5e-18`, below the round-off of the sums themselves.  The
interpolation is therefore not an approximation with a tunable accuracy but
an exact reformulation in floating-point arithmetic.  Limits of (7):

    gap << width:   ln rho ~ 2 sqrt(gap / width),   n ~ 20 sqrt(width / gap),
    gap >> width:   ln rho ~ ln(4 gap / width),     n ~ 40 / ln(4 gap / width).

`n` depends only on the ratio of the spread of the arguments to the
smallest denominator: neither on the number of states nor on the dimension
of the first-order space nor on the number of external labels.  Typical
values: 17 to 30 for width 0.5 to 1 Eh and gap 0.7 to 1.5 Eh (Te2, valence
and (n-1) shell); 96 nodes, the limit `INTERP_MAX_NODES`, are reached at
`gap / width = 0.043`.

**Evaluation.**  (3) is evaluated in barycentric form,

    l_m(d) = [w_m / (d - d_m)] / sum_m' [w_m' / (d - d_m')],
    w_m = (-1)^m sin((2m + 1) pi / 2n),

which is numerically stable for these nodes; the sum of `|l_m|` (Lebesgue
constant) is about `(2 / pi) ln n + 1 < 4`, so round-off in `W(d_m)` is not
amplified.  The weights `l_m(lam_k - s_J)` form an array `(n, m, S)`;
(1) becomes `n` products `t W(d_m)` whose results are combined with these
weights, and the matrices `W(d_m)` of all external blocks of a class are
added before the contraction with `t`.

## 5. Cost

Let `T_int` be the cost of forming the integrals of the class from the
three-index factors and `T_sum` the cost of one sum (2).  The class costs

    exact:          T_int + N_d T_sum
    interpolated:   T_int + n   T_sum

For cavv, per core index: `T_int = N_L n_a n_v^2` (one matrix product with
`N_L` Cholesky vectors), `T_sum = n_a^2 n_v^2`, ratio `n_a / N_L`.  The
exact sums cost `m L n_a / N_L` times the integral formation:

    Te2, 3 states:      36 x 2 x 12 / 2556  ~ 0.3     (interpolation saves little)
    Nd3+, 364 states:   91 x 41 x 14 / 2440 ~ 20      (interpolation is required)

The interpolation matters when `m L n_a / N_L >> 1`: many levels or a large
first-order active space.  For a few states of a small active space the
integral formation dominates either way.  ccvv with state-specific E0 has
`N_d = 1` and is never interpolated: `resolvent_nodes` uses the distinct
arguments themselves whenever they are no more than the nodes.

With four-index blocks kept in memory (`interpolated_block_t`) `T_int` is
absent and the ratio is simply `N_d / n`.

## 6. Where it stops

* **Intruders.**  A pole inside or at the edge of the interval
  (`gap <= 0`): `W` is not analytic on the interval and no polynomial
  interpolation exists.  The contraction falls back to exact denominators
  and the class is reported.
* **Near intruders.**  `gap / width < 0.043` needs more than 96 nodes;
  exact denominators are used.  The interval could be split instead (the
  arguments near the pole exactly, the rest interpolated); not implemented.
* **Few arguments.**  `N_d <= n`: exact sums at the arguments.

## 7. Relation to other schemes

The Laplace transform `1 / D = int exp(-D s) ds` with a quadrature in `s`
(Almlof; Haser) also removes the coupling between the parts of a
denominator, by making it separable in *all* its indices at a finite
quadrature accuracy.  Here the denominator stays exact in the external
labels and only the dependence on the one remaining variable `d`, the
active-space energy argument, is interpolated, to round-off.  The two are
independent and could be combined.

The Cholesky decomposition of the MP2 energy denominators (Koch and
Sanchez de Meras, J. Chem. Phys. 2000; cited from memory, not checked) is
of the same kind as the Laplace transform: the matrix
`1 / (e_a - e_i + e_b - e_j)` is positive definite between `(ai)` and
`(bj)` and a few pivoted Cholesky vectors, independent of the system size,
make the denominator separable.  The object that corresponds to it here is
the Cauchy matrix `1 / (delta_e + d)` between external energies and
active-space arguments.  Its numerical rank grows like `log(width / gap)`,
the number of interpolation nodes like `sqrt(width / gap)`: a pivoted
low-rank factorization, with nodes selected among the arguments that
occur, would need fewer sums than (7) close to intruders and about as many
when gap and width are comparable.  Not implemented.

The scheme of this note was worked out
for this code; whether it has been used elsewhere for multi-state
internally contracted perturbation theory has not been checked in the
literature.
