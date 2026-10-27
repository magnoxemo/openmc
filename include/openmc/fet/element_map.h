//! \file element_map.h
//! \brief Maps from physical space to the parametric coordinates of mesh
//! elements.

#ifndef OPENMC_FET_ELEMENT_MAP_H
#define OPENMC_FET_ELEMENT_MAP_H

#include "openmc/array.h"
#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {
namespace fet {

//==============================================================================
//! Abstract base class for the mapping of a physical position onto the
//! parametric coordinates of an element. Every element shape implements this
//! interface so that the filter does not depend on a particular shape. The
//! same interface can later serve surface tallies (e.g. current).
//==============================================================================

class PhysicalToParametricMapping {
public:
  explicit PhysicalToParametricMapping(vector<Position> vertices)
    : vertices_(std::move(vertices))
  {}
  virtual ~PhysicalToParametricMapping() = default;

  //! Number of parametric coordinates written by parametric_coordinates
  virtual int n_coordinates() const = 0;

  //! Compute the parametric coordinates of a point
  //
  //! \param[in] r Position in physical space
  //! \param[out] coords Array of at least n_coordinates() values
  virtual void parametric_coordinates(Position r, double* coords) const = 0;

  //! Unsigned volume of the element
  double volume() const { return volume_; }

protected:
  vector<Position> vertices_; //!< Vertices of the element
  double volume_ {0.0};       //!< Unsigned volume of the element
};

//==============================================================================
//! Linear tetrahedron. The coordinates are the barycentric coordinates with
//! respect to the vertices (A, B, C, D) in the order given; D is the collapsed
//! vertex of the Duffy transform used by the Dubiner basis. Elements of either
//! orientation are accepted.
//==============================================================================

class TetElementMapping : public PhysicalToParametricMapping {
public:
  explicit TetElementMapping(const vector<Position>& vertices);

  int n_coordinates() const override { return 4; }
  void parametric_coordinates(Position r, double* coords) const override;

private:
  Position origin_;         //!< Vertex D
  std::array<double, 9> inv_ {}; //!< Row-major inverse of [A-D, B-D, C-D]
};

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_ELEMENT_MAP_H
