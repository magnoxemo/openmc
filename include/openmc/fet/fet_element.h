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

//==============================================================================
//! One element of a functional expansion tally, holding both its geometry and
//! its expansion functions.
//!
//! An element has dim() reference coordinates xi, each in [-1, 1], and an
//! expansion order p_d along each axis d. The basis is defined entirely by the
//! element type: how many modes the orders give, how they are numbered, how
//! they are normalized and how they are evaluated. The orders are 0 on every
//! axis until set; a negative order on any axis leaves the element without
//! modes.
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

  //! Number of reference coordinates
  int dim() const { return dim_; }

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

  //! Number of modes at the current orders
  virtual int n_modes() const = 0;

  //! Per-axis degrees of a mode. Entries at and beyond dim() are 0.
  virtual Orders mode(int index) const = 0;

  //! Squared normalization N^2 of a mode, which multiplies its moment in the
  //! expansion on the element (see the element types)
  virtual double mode_norm_sq(int index) const = 0;

  //! Evaluate the unnormalized functions to score for all modes at a position
  //
  //! The reference coordinates are clamped to [-1, 1], which keeps points that
  //! lie marginally outside the element (roundoff) finite.
  //
  //! \param[in] r Position in the frame of the vertices
  //! \param[out] psi Array of at least n_modes() values
  //! \return Number of modes written
  int eval(Position r, double* psi) const;

  //! Value of the expansion on the element,
  //! f = sum_i N_i^2 moments_i P_i / volume
  //! where P_i are the expansion functions, which need not be the functions
  //! eval() returns. The reference coordinates are clamped as in eval().
  //
  //! \param[in] r Position in the frame of the vertices
  //! \param[in] moments Array of n_modes() values, the integrals over the
  //!   element of f times the functions of eval()
  //! \return f at r, per unit volume; 0 if there are no modes
  double reconstruct(Position r, const double* moments) const;

protected:
  explicit FETElement(int dim) : dim_ {dim} {}

  //! Write the n_modes() values returned by eval() at reference coordinates
  //! xi, which are not clamped
  virtual void eval_modes(const double* xi, double* psi) const = 0;

  //! Write the n_modes() expansion functions used by reconstruct() at reference
  //! coordinates xi, which are not clamped. Override when they differ from the
  //! functions of eval().
  virtual void expansion_modes(const double* xi, double* values) const
  {
    eval_modes(xi, values);
  }

  //! Called after the orders change, for subclasses that cache anything that
  //! depends on them
  virtual void on_orders_changed() {}

  double volume_ {0.0};

private:
  int dim_;
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
//! vanishes is set to 0.
//!
//! A mode (i, j, k) is P_i(xi_0) P_j^(0,1)(xi_1) P_k^(0,2)(xi_2) with
//! 0 <= m_d <= p_d. The Jacobi parameters match the Jacobian of the collapse, so
//! the modes are orthogonal on the element. They are numbered
//! lexicographically with the last axis fastest,
//! index = (i * (p_1 + 1) + j) * (p_2 + 1) + k.
//==============================================================================

class TetElement final : public FETElement {
public:
  explicit TetElement(const vector<Position>& vertices);

  int n_vertices() const override { return 4; }
  const Position& vertex(int i) const override { return v_[i]; }
  const char* type() const override { return "tet4"; }
  void reference_coords(Position r, double* xi) const override;

  int n_modes() const override;
  Orders mode(int index) const override;
  double mode_norm_sq(int index) const override;

protected:
  void eval_modes(const double* xi, double* psi) const override;

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
//! with s2 = la + lb. xi_0 is set to 0 when s2 vanishes.
//!
//! A mode (i, j) is P_i(xi_0) P_j^(0,1)(xi_1) with 0 <= m_d <= p_d. These are
//! orthogonal on the element. They are numbered index = i * (p_1 + 1) + j.
//==============================================================================

class TriElement final : public FETElement {
public:
  explicit TriElement(const vector<Position>& vertices);

  int n_vertices() const override { return 3; }
  const Position& vertex(int i) const override { return v_[i]; }
  const char* type() const override { return "tri3"; }
  void reference_coords(Position r, double* xi) const override;

  int n_modes() const override;
  Orders mode(int index) const override;
  double mode_norm_sq(int index) const override;

protected:
  void eval_modes(const double* xi, double* psi) const override;

private:
  std::array<Position, 3> v_;
  Position e0_;                   //!< A - C
  Position e1_;                   //!< B - C

  //! Row-major inverse of the Gram matrix G = [e0.e0, e0.e1; e0.e1, e1.e1].
  //!It is precomputed because it depends only on the geometry and reference_coords
  //! runs on every scoring event.
  std::array<double, 4> ginv_ {};
};

//==============================================================================
//! Bilinear quadrilateral embedded in 3D, for surface tallies. It must be
//! planar and strictly convex, with the vertices (A, B, C, D) ordered around
//! its boundary.
//!
//! The reference coordinates are those of the bilinear map
//!   r(xi_0, xi_1) = r0 + u xi_0 + v xi_1 + w xi_0 xi_1
//! with the vertex average r0 and
//!   u = (-A + B + C - D) / 4,  v = (-A - B + C + D) / 4,
//!   w = ( A - B + C - D) / 4,
//! so that A is at (-1, -1), B at (1, -1), C at (1, 1) and D at (-1, 1). A
//! position is projected orthogonally onto the plane of the quadrilateral
//! before the map is inverted.
//!
//! The vertices are rotated (keeping their orientation) so that A is a corner
//! with the smallest Jacobian determinant. The determinant then splits into
//! three non-negative parts,
//!   |J| = J_0 + J_1 + J_2
//!       = c0 + j1 (1 + xi_0) + j2 (1 + xi_1),
//! and there is one set of modes (i, j) per part, with 0 <= m_d <= p_d,
//!   set 0: P_i(xi_0)       P_j(xi_1)
//!   set 1: P_i^(0,1)(xi_0) P_j(xi_1)
//!   set 2: P_i(xi_0)       P_j^(0,1)(xi_1)
//! Each set is orthogonal with respect to its own part J_k. The sets are stored
//! one after the other, index = k * (p_0 + 1) * (p_1 + 1) + i * (p_1 + 1) + j.
//!
//! eval() returns the modes weighted by J_k / |J|, so that scoring them over the
//! element gives the moments a = integral of f P J_k over the reference square.
//! reconstruct() uses the polynomials P without that weight,
//!   f = sum over all modes of N^2 a P / area,
//! which is exact if f is a polynomial within the orders. All three sets exist
//! even where a part vanishes (j1 = j2 = 0 for a parallelogram).
//==============================================================================

class QuadElement final : public FETElement {
public:
  explicit QuadElement(const vector<Position>& vertices);

  int n_vertices() const override { return 4; }
  const Position& vertex(int i) const override { return vertices_[i]; }
  const char* type() const override { return "quad4"; }
  void reference_coords(Position r, double* xi) const override;

  int n_modes() const override;
  Orders mode(int index) const override;
  double mode_norm_sq(int index) const override;

protected:
  void eval_modes(const double* xi, double* psi) const override;
  void expansion_modes(const double* xi, double* values) const override;

private:
  std::array<Position, 4> vertices_;

  // Coefficients of the bilinear map
  Position center_; //!< r0
  Position u_;
  Position v_;
  Position w_;

  // Parts of the Jacobian determinant
  double c0_;
  double j1_;
  double j2_;

  // With x = r - r0 and p ^ q = n . (p x q) for the unit normal n, xi_0 solves
  // k2 xi_0^2 + k1 xi_0 - x ^ v = 0, where k2 = u ^ w and k1 = u ^ v - x ^ w.
  // The parts that depend only on the geometry are precomputed.
  double k2_;
  double k1_center_; //!< u ^ v, the value of k1 at r0
  Position k1_grad_; //!< n x w, so that k1 = k1_center + x . k1_grad
  Position k0_grad_; //!< v x n, so that x ^ v = x . k0_grad
};

//! Create the 2D element of a face of a mesh from its vertices, ordered so that
//! the right-hand rule gives the face normal. Throws (fatal_error) when no
//! element type exists for the number of vertices.
std::unique_ptr<FETElement> make_face_element(const vector<Position>& vertices);

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_FET_ELEMENT_H
