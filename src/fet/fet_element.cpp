#include "openmc/fet/fet_element.h"

#include <algorithm>
#include <cmath>

#include "openmc/error.h"
#include "openmc/math_functions.h"

namespace openmc {
namespace fet {

namespace {

// Jacobi parameter beta of each reference axis; alpha is 0 on all of them and
// the weights match the Jacobian of the Duffy collapse
constexpr std::array<int, 3> BETA {0, 1, 2};

} // namespace

//==============================================================================
// FETElement
//==============================================================================

int FETElement::n_modes(int order) const
{
  if (order < 0)
    return 0;
  return (order + 1) * (order + 1) * (order + 1);
}

std::array<int, 3> FETElement::mode(int order, int index) const
{
  const int n = order + 1;
  return {index / (n * n), (index / n) % n, index % n};
}

double FETElement::mode_norm_sq(int order, int index) const
{
  const auto m = mode(order, index);
  double n2 = 1.0;
  for (int d = 0; d < 3; ++d)
    n2 *= (2.0 * m[d] + BETA[d] + 1) / (BETA[d] + 1);
  return n2;
}

int FETElement::eval(int order, Position r, double* psi) const
{
  double xi[3];
  reference_coords(r, xi);

  double jac[3][MAX_ORDER + 1];
  for (int d = 0; d < 3; ++d) {
    xi[d] = std::clamp(xi[d], -1.0, 1.0);
    calc_jacobi(order, 0.0, BETA[d], xi[d], 1.0, jac[d]);
  }

  int m = 0;
  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      for (int k = 0; k <= order; ++k)
        psi[m++] = jac[0][i] * jac[1][j] * jac[2][k];
  return m;
}

//==============================================================================
// TetElement
//==============================================================================

TetElement::TetElement(const vector<Position>& vertices)
{
  if (vertices.size() != 4)
    fatal_error("A tetrahedral element must have 4 vertices.");
  std::copy(vertices.begin(), vertices.end(), v_.begin());

  const Position a = v_[0] - v_[3];
  const Position b = v_[1] - v_[3];
  const Position c = v_[2] - v_[3];
  const Position bc = b.cross(c);
  const Position ca = c.cross(a);
  const Position ab = a.cross(b);
  const double det = a.dot(bc);

  if (std::abs(det) <= 1.0e-14 * a.norm() * b.norm() * c.norm()) {
    fatal_error("Degenerate tetrahedron encountered while building a "
                "functional expansion tally element.");
  }
  measure_ = std::abs(det) / 6.0;

  inv_ = {bc.x / det, bc.y / det, bc.z / det, ca.x / det, ca.y / det,
    ca.z / det, ab.x / det, ab.y / det, ab.z / det};
}

void TetElement::reference_coords(Position r, double* xi) const
{
  const Position d = r - v_[3];
  const double la = inv_[0] * d.x + inv_[1] * d.y + inv_[2] * d.z;
  const double lb = inv_[3] * d.x + inv_[4] * d.y + inv_[5] * d.z;
  const double lc = inv_[6] * d.x + inv_[7] * d.y + inv_[8] * d.z;
  const double ld = 1.0 - la - lb - lc;

  constexpr double tiny = 1.0e-14;
  const double s2 = la + lb;
  const double s3 = s2 + lc;
  xi[0] = s2 > tiny ? (lb - la) / s2 : 0.0;
  xi[1] = s3 > tiny ? (2.0 * s2 - s3) / s3 : 0.0;
  xi[2] = 1.0 - 2.0 * ld;
}

} // namespace fet
} // namespace openmc
