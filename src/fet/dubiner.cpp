#include "openmc/fet/dubiner.h"

#include <cmath>

#include "openmc/error.h"

namespace openmc {
namespace fet {

//==============================================================================
// Mode indexing
//==============================================================================

int tet_mode_index(int p, int q, int r)
{
  int k = p + q + r;
  return n_modes_tet(k - 1) + p * (k + 1) - p * (p - 1) / 2 + q;
}

std::array<int, 3> tet_mode(int index)
{
  int k = 0;
  while (n_modes_tet(k) <= index)
    ++k;
  int local = index - n_modes_tet(k - 1);
  for (int p = 0; p <= k; ++p) {
    int n_q = k - p + 1;
    if (local < n_q)
      return {p, local, k - p - local};
    local -= n_q;
  }
  fatal_error("Invalid tetrahedral FET mode index.");
}

int tri_mode_index(int p, int q)
{
  return n_modes_tri(p + q - 1) + p;
}

std::array<int, 2> tri_mode(int index)
{
  int k = 0;
  while (n_modes_tri(k) <= index)
    ++k;
  int p = index - n_modes_tri(k - 1);
  return {p, k - p};
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

void eval_tet(int order, const double lambda[4], double* out)
{
  // Homogeneous variables of the collapsed coordinates:
  //   t2 = (1+zeta)(1+gamma)/4, eta = (lambda_B - lambda_A) / t2
  //   t3 = (1+gamma)/2,         zeta = 2 t2 / t3 - 1
  //   gamma = 1 - 2 lambda_D
  double t2 = lambda[0] + lambda[1];
  double t3 = t2 + lambda[2];
  double gamma = 1.0 - 2.0 * lambda[3];

  // t2^p P_p(eta)
  double leg[MAX_ORDER + 1];
  scaled_jacobi(order, 0.0, 0.0, lambda[1] - lambda[0], t2, leg);

  double jac_q[MAX_ORDER + 1];
  double jac_r[MAX_ORDER + 1];
  for (int p = 0; p <= order; ++p) {
    // t3^q P_q^(0,2p+1)(zeta)
    scaled_jacobi(order - p, 0.0, 2 * p + 1, 2.0 * t2 - t3, t3, jac_q);
    for (int q = 0; q <= order - p; ++q) {
      // P_r^(0,2p+2q+2)(gamma)
      scaled_jacobi(order - p - q, 0.0, 2 * (p + q) + 2, gamma, 1.0, jac_r);
      for (int r = 0; r <= order - p - q; ++r) {
        int k = p + q + r;
        double norm = std::sqrt((2 * p + 1) * (p + q + 1) * (2 * k + 3) / 3.0);
        out[tet_mode_index(p, q, r)] = norm * leg[p] * jac_q[q] * jac_r[r];
      }
    }
  }
}

void eval_tri(int order, const double lambda[3], double* out)
{
  // t2 = (1+zeta)/2, eta = (lambda_B - lambda_A) / t2, zeta = 1 - 2 lambda_C
  double t2 = lambda[0] + lambda[1];
  double zeta = 1.0 - 2.0 * lambda[2];

  // t2^p P_p(eta)
  double leg[MAX_ORDER + 1];
  scaled_jacobi(order, 0.0, 0.0, lambda[1] - lambda[0], t2, leg);

  double jac_q[MAX_ORDER + 1];
  for (int p = 0; p <= order; ++p) {
    // P_q^(0,2p+1)(zeta)
    scaled_jacobi(order - p, 0.0, 2 * p + 1, zeta, 1.0, jac_q);
    for (int q = 0; q <= order - p; ++q) {
      double norm = std::sqrt((2 * p + 1) * (p + q + 1));
      out[tri_mode_index(p, q)] = norm * leg[p] * jac_q[q];
    }
  }
}

} // namespace fet
} // namespace openmc
