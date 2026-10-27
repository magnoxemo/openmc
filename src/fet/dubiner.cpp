#include "openmc/fet/dubiner.h"

#include <algorithm>
#include <cmath>

namespace openmc {
namespace fet {

//==============================================================================
// Mode indexing
//==============================================================================

int tet_mode_index(int i, int j, int k)
{
  int m = std::max({i, j, k});
  int base = n_modes_tet(m - 1);
  if (i == m)
    return base + j * (m + 1) + k;
  if (j == m)
    return base + (m + 1) * (m + 1) + i * (m + 1) + k;
  return base + (m + 1) * (m + 1) + m * (m + 1) + i * m + j;
}

std::array<int, 3> tet_mode(int index)
{
  int m = 0;
  while (n_modes_tet(m) <= index)
    ++m;

  int local = index - n_modes_tet(m - 1);
  int n_i = (m + 1) * (m + 1);

  if (local < n_i)
    return {m, local / (m + 1), local % (m + 1)};
  local -= n_i;

  int n_j = m * (m + 1);
  if (local < n_j)
    return {local / (m + 1), m, local % (m + 1)};
  local -= n_j;

  return {local / m, local % m, m};
}

int tri_mode_index(int i, int j)
{
  int m = std::max(i, j);
  int base = n_modes_tri(m - 1);
  return i == m ? base + j : base + (m + 1) + i;
}

std::array<int, 2> tri_mode(int index)
{
  int m = 0;
  while (n_modes_tri(m) <= index)
    ++m;
  int local = index - n_modes_tri(m - 1);
  if (local <= m)
    return {m, local};
  return {local - (m + 1), m};
}

//==============================================================================
// Basis evaluation
//==============================================================================

void scaled_jacobi(
  int n, double alpha, double beta, double z, double t, double* out)
{
  out[0] = 1.0;
  if (n == 0)
    return;
  out[1] = 0.5 * ((alpha + beta + 2.0) * z + (alpha - beta) * t);

  double ab = alpha + beta;
  double d = alpha * alpha - beta * beta;
  for (int k = 1; k < n; ++k) {
    double a = 2.0 * (k + 1) * (k + ab + 1) * (2 * k + ab);
    double b = 2 * k + ab + 1;
    double c = (2 * k + ab + 2) * (2 * k + ab);
    double e = 2.0 * (k + alpha) * (k + beta) * (2 * k + ab + 2);
    out[k + 1] = (b * (c * z + d * t) * out[k] - e * t * t * out[k - 1]) / a;
  }
}

double tet_mode_norm_sq(int i, int j, int k)
{
  return (2 * i + 1) * (j + 1) * (2 * k + 3) / 3.0;
}

double tet_mode_norm(int i, int j, int k)
{
  return std::sqrt(tet_mode_norm_sq(i, j, k));
}

double tri_mode_norm_sq(int i, int j)
{
  return (2 * i + 1) * (j + 1);
}

double tri_mode_norm(int i, int j)
{
  return std::sqrt(tri_mode_norm_sq(i, j));
}

void eval_tet(int order, const double lambda[4], double* out)
{
  // Collapsed coordinates of (A, B, C, D) with D collapsed first:
  //   gamma = 1 - 2 lambda_D, zeta = (2 t2 - t3) / t3, eta = (lambda_B -
  //   lambda_A) / t2 with t2 = lambda_A + lambda_B, t3 = t2 + lambda_C
  constexpr double tiny = 1.0e-14;
  double t2 = lambda[0] + lambda[1];
  double t3 = t2 + lambda[2];
  double gamma = 1.0 - 2.0 * lambda[3];
  double eta = t2 > tiny ? (lambda[1] - lambda[0]) / t2 : 0.0;
  double zeta = t3 > tiny ? (2.0 * t2 - t3) / t3 : 0.0;

  double leg[MAX_ORDER + 1];
  double jac_q[MAX_ORDER + 1];
  double jac_r[MAX_ORDER + 1];
  scaled_jacobi(order, 0.0, 0.0, eta, 1.0, leg);
  scaled_jacobi(order, 0.0, 1.0, zeta, 1.0, jac_q);
  scaled_jacobi(order, 0.0, 2.0, gamma, 1.0, jac_r);

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      for (int k = 0; k <= order; ++k)
        out[tet_mode_index(i, j, k)] = leg[i] * jac_q[j] * jac_r[k];
}

void eval_tri(int order, const double lambda[3], double* out)
{
  constexpr double tiny = 1.0e-14;
  double t2 = lambda[0] + lambda[1];
  double zeta = 1.0 - 2.0 * lambda[2];
  double eta = t2 > tiny ? (lambda[1] - lambda[0]) / t2 : 0.0;

  double leg[MAX_ORDER + 1];
  double jac_q[MAX_ORDER + 1];
  scaled_jacobi(order, 0.0, 0.0, eta, 1.0, leg);
  scaled_jacobi(order, 0.0, 1.0, zeta, 1.0, jac_q);

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      out[tri_mode_index(i, j)] = leg[i] * jac_q[j];
}

} // namespace fet
} // namespace openmc
