#ifndef OPENMC_FET_FET_ELEMENT_H
#define OPENMC_FET_FET_ELEMENT_H

#include <array>

#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {
namespace fet {

//! Maximum supported 1D degree of an expansion
constexpr int MAX_ORDER {10};

//! Maximum number of modes of any supported element type and order
constexpr int MAX_MODES {(MAX_ORDER + 1) * (MAX_ORDER + 1) * (MAX_ORDER + 1)};

//==============================================================================
//! One element of a functional expansion tally, holding both its geometry and
//! its expansion functions.
//!
//! A mode (i, j, k) is P_i(xi_0) P_j^(0,1)(xi_1) P_k^(0,2)(xi_2) in the
//! Duffy-collapsed reference coordinates xi of the element, with i, j, k <=
//! order. For an element of order p the modes are numbered lexicographically,
//! index = (i * (p + 1) + j) * (p + 1) + k, so the numbering depends on p.
//==============================================================================

class FETElement {
public:
  virtual ~FETElement() = default;

  //! Number of vertices
  virtual int n_vertices() const = 0;

  //! Name of the element type, written to the statepoint
  virtual const char* type() const = 0;

  //! Name of the measure, written to the statepoint
  virtual const char* measure_name() const = 0;

  //! Reference coordinates xi[0..3) of a position. Not clamped.
  virtual void reference_coords(Position r, double* xi) const = 0;

  //! Vertex i, in the order that defines the reference coordinates
  const Position& vertex(int i) const { return v_[i]; }

  //! Unsigned measure (volume)
  double measure() const { return measure_; }

  //! Number of modes of the given order; 0 for a negative order
  int n_modes(int order) const;

  //! Indices (i, j, k) of a mode of an element of the given order
  std::array<int, 3> mode(int order, int index) const;

  //! Squared normalization N^2 = (2i + 1)(j + 1)(2k + 3) / 3 of a mode.
  //! Multiplying a mode by N gives the basis with integral psi_i psi_j =
  //! measure * delta_ij.
  double mode_norm_sq(int order, int index) const;

  //! Evaluate all unnormalized modes of the given order at a position
  //
  //! The reference coordinates are clamped to [-1, 1], which keeps points that
  //! lie marginally outside the element (roundoff) finite.
  //
  //! \param[in] order Maximum 1D degree, 0 <= order <= MAX_ORDER
  //! \param[in] r Position in the frame of the vertices
  //! \param[out] psi Array of at least n_modes(order) values
  //! \return Number of modes written
  int eval(int order, Position r, double* psi) const;

protected:
  std::array<Position, 4> v_;
  double measure_ {0.0};
};

//==============================================================================
//! Linear tetrahedron; elements of either orientation are accepted.
//!
//! The reference coordinates come from the barycentric coordinates
//! (la, lb, lc, ld) of the position with respect to the vertices (A, B, C, D)
//! through the Duffy collapse, D first:
//!   xi_0 = (lb - la) / s2,   xi_1 = (2 s2 - s3) / s3,   xi_2 = 1 - 2 ld
//! with s2 = la + lb and s3 = la + lb + lc. A coordinate whose denominator
//! vanishes is set to 0.
//==============================================================================

class TetElement final : public FETElement {
public:
  explicit TetElement(const vector<Position>& vertices);

  int n_vertices() const override { return 4; }
  const char* type() const override { return "tet4"; }
  const char* measure_name() const override { return "volume"; }
  void reference_coords(Position r, double* xi) const override;

private:
  std::array<double, 9> inv_ {}; //!< Row-major inverse of [A-D, B-D, C-D]
};

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_FET_ELEMENT_H
