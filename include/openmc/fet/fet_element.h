#ifndef OPENMC_FET_FET_ELEMENT_H
#define OPENMC_FET_FET_ELEMENT_H

#include <array>

#include "openmc/bounding_box.h"
#include "openmc/memory.h"
#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {
namespace fet {

//! Maximum supported 1D degree of an expansion
constexpr int MAX_ORDER {10};

//! Maximum number of reference coordinates of any supported element type
//! As right now I am planning to do spatial FET tally realted stuff
constexpr int MAX_DIM {3};

//! Maximum number of modes of any supported element type and order
constexpr int MAX_MODES {(MAX_ORDER + 1) * (MAX_ORDER + 1) * (MAX_ORDER + 1)};

//! Expansion order along each reference axis. Entries at and beyond the
//! dimension of an element are ignored.
using Orders = std::array<int, MAX_DIM>;

//! Crossing of a line segment with a 2D element
struct SegmentHit {
  double t {0.0};       //!< Segment parameter of the crossing, in (0, 1]
  Position point;       //!< Crossing point
  double cosine {0.0};  //!< Cosine between the segment and the element normal
};

//==============================================================================
//! One element of a functional expansion tally, holding both its geometry and
//! its expansion functions.
//!
//! An element has dim() reference coordinates xi, each in [-1, 1], and an
//! expansion order p_d along each axis d. A mode (m_0, ..., m_{dim-1}) is the
//! tensor product of P_{m_d}^(0, beta_d)(xi_d) with 0 <= m_d <= p_d, where
//! beta_d is the Jacobi parameter of axis d. The subclass picks the betas so
//! that they match the Jacobian of its map to the reference cube, which makes
//! the modes orthogonal on the element. The modes are numbered
//! lexicographically with the last axis fastest, e.g. in 3D
//! index = (i * (p_1 + 1) + j) * (p_2 + 1) + k, so the numbering depends on
//! the orders. The orders are 0 on every axis until set; a negative order on
//! any axis leaves the element without modes.
//==============================================================================

class FETElement {
public:
  virtual ~FETElement() = default;

  //! Number of vertices
  virtual int n_vertices() const = 0;

  //! Vertex i, in the order that defines the reference coordinates
  virtual const Position& vertex(int i) const = 0;

  //! Name of the element type, written to the statepoint
  virtual const char* type() const = 0;

  //! Reference coordinates xi[0..dim()) of a position. Not clamped.
  virtual void reference_coords(Position r, double* xi) const = 0;

  //! Axis-aligned bounding box of the vertices
  BoundingBox bounding_box() const;

  //! Crossing of a line segment with the element
  //
  //! Only 2D elements embedded in 3D (surface elements) can be crossed. The
  //! normal of such an element follows the right-hand rule on its vertex
  //! order. The segment is p0 + t d with t in (0, 1]: a crossing at t = 0 does
  //! not count and one at t = 1 does, so that consecutive segments of a
  //! trajectory never count the same crossing twice. Segments parallel to the
  //! element do not cross it.
  //
  //! \param[in] p0 Start of the segment
  //! \param[in] d Displacement from the start to the end of the segment
  //! \param[out] hit Crossing, if there is one
  //! \return Whether the segment crosses the element
  virtual bool intersect(Position p0, Position d, SegmentHit& hit) const;

  //! Number of reference coordinates
  int dim() const { return dim_; }

  //! Jacobi parameter beta of reference axis d
  int beta(int d) const { return beta_[d]; }

  //! Unsigned volume in dim() dimensions, i.e. area for a 2D element
  double volume() const { return volume_; }

  //! Set the expansion order along one reference axis
  //
  //! \param[in] axis Reference axis, 0 <= axis < dim()
  //! \param[in] order Order along the axis, at most MAX_ORDER; negative
  //!   leaves the element without modes
  void set_order(int axis, int order);

  //! Set the expansion order along every reference axis; entries at and
  //! beyond dim() are ignored
  void set_orders(const Orders& orders);

  //! Expansion order along reference axis d
  int order(int d) const { return orders_[d]; }

  //! Expansion orders along all reference axes
  const Orders& orders() const { return orders_; }

  //! Largest order over the reference axes
  int max_order() const;

  //! Number of modes, prod_d (p_d + 1); 0 if any order is negative
  int n_modes() const;

  //! Per-axis degrees of a mode. Entries at and beyond dim() are 0.
  Orders mode(int index) const;

  //! Squared normalization N^2 = prod_d (2 m_d + beta_d + 1) / (beta_d + 1)
  //! of a mode. Multiplying a mode by N gives the basis with integral
  //! psi_i psi_j = volume * delta_ij.
  double mode_norm_sq(int index) const;

  //! Evaluate all unnormalized modes at a position
  //
  //! The reference coordinates are clamped to [-1, 1], which keeps points that
  //! lie marginally outside the element (roundoff) finite.
  //
  //! \param[in] r Position in the frame of the vertices
  //! \param[out] psi Array of at least n_modes() values
  //! \return Number of modes written
  int eval(Position r, double* psi) const;

protected:
  FETElement(int dim, std::array<int, MAX_DIM> beta) : dim_ {dim}, beta_ {beta}
  {}

  double volume_ {0.0};

private:
  int dim_;
  std::array<int, MAX_DIM> beta_;
  Orders orders_ {}; //!< Expansion order along each reference axis
};

//==============================================================================
//! Linear tetrahedron; elements of either orientation are accepted.
//!
//! The reference coordinates come from the barycentric coordinates
//! (la, lb, lc, ld) of the position with respect to the vertices (A, B, C, D)
//! through the Duffy collapse, D first:
//!   xi_0 = (lb - la) / s2,   xi_1 = (2 s2 - s3) / s3,   xi_2 = 1 - 2 ld
//! with s2 = la + lb and s3 = la + lb + lc. A coordinate whose denominator
//! vanishes is set to 0. The Jacobi parameters are beta = (0, 1, 2).
//==============================================================================

class TetElement final : public FETElement {
public:
  explicit TetElement(const vector<Position>& vertices);

  int n_vertices() const override { return 4; }
  const Position& vertex(int i) const override { return v_[i]; }
  const char* type() const override { return "tet4"; }
  void reference_coords(Position r, double* xi) const override;

private:
  std::array<Position, 4> v_;
  std::array<double, 9> inv_ {}; //!< Row-major inverse of [A-D, B-D, C-D]
};

//==============================================================================
//! Linear triangle embedded in 3D, for surface tallies.
//!
//! A position is projected orthogonally onto the plane of the triangle, and
//! the barycentric coordinates (la, lb, lc) of the projection with respect to
//! the vertices (A, B, C) give the reference coordinates through the Duffy
//! collapse, C first:
//!   xi_0 = (lb - la) / s2,   xi_1 = 1 - 2 lc
//! with s2 = la + lb. xi_0 is set to 0 when s2 vanishes. The Jacobi
//! parameters are beta = (0, 1).
//==============================================================================

class TriElement final : public FETElement {
public:
  explicit TriElement(const vector<Position>& vertices);

  int n_vertices() const override { return 3; }
  const Position& vertex(int i) const override { return v_[i]; }
  const char* type() const override { return "tri3"; }
  void reference_coords(Position r, double* xi) const override;
  bool intersect(Position p0, Position d, SegmentHit& hit) const override;

private:
  std::array<Position, 3> v_;
  Position normal_;               //!< Unit normal (B - A) x (C - A)
  Position e0_;                   //!< A - C
  Position e1_;                   //!< B - C
  std::array<double, 4> ginv_ {}; //!< Row-major inverse of the Gram matrix
};

//! Create the 2D element of a face of a mesh from its vertices, ordered so that
//! the right-hand rule gives the face normal. Throws (fatal_error) when no
//! element type exists for the number of vertices.
std::unique_ptr<FETElement> make_face_element(const vector<Position>& vertices);

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_FET_ELEMENT_H
