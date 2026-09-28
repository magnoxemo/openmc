//! \file element_map.h
//! \brief Affine maps from physical space to barycentric coordinates of
//! simplex elements.

#ifndef OPENMC_FET_ELEMENT_MAP_H
#define OPENMC_FET_ELEMENT_MAP_H

#include <array>

#include "openmc/position.h"

namespace openmc {
namespace fet {

//==============================================================================
//! Affine map of a linear tetrahedron with vertices (A, B, C, D).
//!
//! The vertex order defines the basis orientation: D is the collapsed vertex
//! of the Duffy transform used by the Dubiner basis. Elements of either
//! orientation are accepted.
//==============================================================================

class AffineTetMap {
public:
  AffineTetMap() = default;

  //! Construct the map from the element's vertices
  //
  //! \param[in] vertices Vertices (A, B, C, D) of the tetrahedron
  explicit AffineTetMap(const std::array<Position, 4>& vertices);

  //! Compute the barycentric coordinates of a point
  //
  //! \param[in] r Position in physical space
  //! \param[out] lambda Barycentric coordinates with respect to (A, B, C, D)
  void barycentric(Position r, double lambda[4]) const;

  //! Volume of the tetrahedron
  double volume() const { return volume_; }

private:
  Position origin_;          //!< Vertex D
  std::array<double, 9> inv_ //!< Row-major inverse of [A-D, B-D, C-D]
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double volume_ {0.0}; //!< Unsigned volume
};

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_ELEMENT_MAP_H
