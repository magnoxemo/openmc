#include "openmc/fet/element_map.h"

#include <cmath>

#include "openmc/error.h"

namespace openmc {
namespace fet {

TetElementMapping::TetElementMapping(const vector<Position>& vertices)
  : PhysicalToParametricMapping(vertices)
{
  if (vertices_.size() != 4)
    fatal_error("A tetrahedral element must have 4 vertices.");

  origin_ = vertices_[3];
  const Position a = vertices_[0] - origin_;
  const Position b = vertices_[1] - origin_;
  const Position c = vertices_[2] - origin_;

  const Position bc = b.cross(c);
  const Position ca = c.cross(a);
  const Position ab = a.cross(b);
  const double det = a.dot(bc);

  const double scale = a.norm() * b.norm() * c.norm();
  if (std::abs(det) <= 1.0e-14 * scale) {
    fatal_error("Degenerate tetrahedron encountered while building a "
                "functional expansion tally map.");
  }

  inv_ = {bc.x / det, bc.y / det, bc.z / det, ca.x / det, ca.y / det,
    ca.z / det, ab.x / det, ab.y / det, ab.z / det};
  volume_ = std::abs(det) / 6.0;
}

void TetElementMapping::parametric_coordinates(Position r, double* coords) const
{
  const Position d = r - origin_;
  coords[0] = inv_[0] * d.x + inv_[1] * d.y + inv_[2] * d.z;
  coords[1] = inv_[3] * d.x + inv_[4] * d.y + inv_[5] * d.z;
  coords[2] = inv_[6] * d.x + inv_[7] * d.y + inv_[8] * d.z;
  coords[3] = 1.0 - coords[0] - coords[1] - coords[2];
}

} // namespace fet
} // namespace openmc
