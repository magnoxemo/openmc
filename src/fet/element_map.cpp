#include "openmc/fet/element_map.h"

#include <cmath>

#include "openmc/error.h"

namespace openmc {
namespace fet {

AffineTetMap::AffineTetMap(const std::array<Position, 4>& vertices)
  : origin_ {vertices[3]}
{
  // Columns of the map from barycentric (lambda_A, lambda_B, lambda_C) to
  // physical coordinates relative to vertex D
  Position a = vertices[0] - origin_;
  Position b = vertices[1] - origin_;
  Position c = vertices[2] - origin_;

  // Inverse of the matrix with columns (a, b, c): rows are the cross products
  // divided by the determinant
  Position bc = b.cross(c);
  Position ca = c.cross(a);
  Position ab = a.cross(b);
  double det = a.dot(bc);

  double scale = a.norm() * b.norm() * c.norm();
  if (std::abs(det) <= 1.0e-14 * scale) {
    fatal_error("Degenerate tetrahedron encountered while building a "
                "functional expansion tally map.");
  }

  inv_ = {bc.x / det, bc.y / det, bc.z / det, ca.x / det, ca.y / det,
    ca.z / det, ab.x / det, ab.y / det, ab.z / det};
  volume_ = std::abs(det) / 6.0;
}

void AffineTetMap::barycentric(Position r, double lambda[4]) const
{
  Position d = r - origin_;
  lambda[0] = inv_[0] * d.x + inv_[1] * d.y + inv_[2] * d.z;
  lambda[1] = inv_[3] * d.x + inv_[4] * d.y + inv_[5] * d.z;
  lambda[2] = inv_[6] * d.x + inv_[7] * d.y + inv_[8] * d.z;
  lambda[3] = 1.0 - lambda[0] - lambda[1] - lambda[2];
}

} // namespace fet
} // namespace openmc
