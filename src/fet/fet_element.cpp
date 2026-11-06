#include "openmc/fet/fet_element.h"

#include <algorithm>
#include <cmath>

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/math_functions.h"

namespace openmc {
namespace fet {

//==============================================================================
// FETElement
//==============================================================================

void FETElement::set_order(int axis, int order)
{
  if (axis < 0 || axis >= dim_) {
    fatal_error(fmt::format(
      "Invalid reference axis {} of a {}-dimensional FET element.", axis, dim_));
  }
  if (order > MAX_ORDER) {
    fatal_error(fmt::format(
      "FET order {} exceeds the maximum supported order {}.", order, MAX_ORDER));
  }
  orders_[axis] = order;
}

void FETElement::set_orders(const Orders& orders)
{
  for (int d = 0; d < dim_; ++d)
    set_order(d, orders[d]);
}

int FETElement::max_order() const
{
  return *std::max_element(orders_.begin(), orders_.begin() + dim_);
}

int FETElement::n_modes() const
{
  int n = 1;
  for (int d = 0; d < dim_; ++d) {
    if (orders_[d] < 0)
      return 0;
    n *= orders_[d] + 1;
  }
  return n;
}

BoundingBox FETElement::bounding_box() const
{
  BoundingBox box = BoundingBox::inverted();
  for (int i = 0; i < n_vertices(); ++i)
    box.expand(vertex(i));
  return box;
}

bool FETElement::intersect(Position p0, Position d, SegmentHit& hit) const
{
  fatal_error(fmt::format("Segments cannot cross a {} FET element; only 2D "
                          "(surface) elements support crossings.",
    type()));
}

Orders FETElement::mode(int index) const
{
  Orders m {};
  for (int d = dim_ - 1; d >= 0; --d) {
    const int n = orders_[d] + 1;
    m[d] = index % n;
    index /= n;
  }
  return m;
}

double FETElement::mode_norm_sq(int index) const
{
  const auto m = mode(index);
  double n2 = 1.0;
  for (int d = 0; d < dim_; ++d)
    n2 *= (2.0 * m[d] + beta_[d] + 1) / (beta_[d] + 1);
  return n2;
}

int FETElement::eval(Position r, double* psi) const
{
  if (n_modes() == 0)
    return 0;

  double xi[MAX_DIM];
  reference_coords(r, xi);

  // Build the tensor product one axis at a time, in place. Going backwards
  // over the existing entries never overwrites an entry before it is read,
  // since entry m expands into entries m * n .. m * n + n - 1 >= m.
  double jac[MAX_ORDER + 1];
  psi[0] = 1.0;
  int size = 1;
  for (int d = 0; d < dim_; ++d) {
    const int n = orders_[d] + 1;
    calc_jacobi(
      orders_[d], 0.0, beta_[d], std::clamp(xi[d], -1.0, 1.0), 1.0, jac);
    for (int m = size - 1; m >= 0; --m) {
      const double v = psi[m];
      for (int k = n - 1; k >= 0; --k)
        psi[m * n + k] = v * jac[k];
    }
    size *= n;
  }
  return size;
}

//==============================================================================
// TetElement
//==============================================================================

TetElement::TetElement(const vector<Position>& vertices)
  : FETElement(3, {0, 1, 2})
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
  volume_ = std::abs(det) / 6.0;

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

//==============================================================================
// TriElement
//==============================================================================

TriElement::TriElement(const vector<Position>& vertices)
  : FETElement(2, {0, 1, 0})
{
  if (vertices.size() != 3)
    fatal_error("A triangular element must have 3 vertices.");
  std::copy(vertices.begin(), vertices.end(), v_.begin());

  e0_ = v_[0] - v_[2];
  e1_ = v_[1] - v_[2];
  const double area2 = e0_.cross(e1_).norm();

  if (area2 <= 1.0e-14 * e0_.norm() * e1_.norm()) {
    fatal_error("Degenerate triangle encountered while building a "
                "functional expansion tally element.");
  }
  volume_ = area2 / 2.0;
  normal_ = e0_.cross(e1_) / area2;

  // Gram matrix G = [e0.e0, e0.e1; e0.e1, e1.e1]; det G = |e0 x e1|^2
  const double g00 = e0_.dot(e0_);
  const double g01 = e0_.dot(e1_);
  const double g11 = e1_.dot(e1_);
  const double det = area2 * area2;
  ginv_ = {g11 / det, -g01 / det, -g01 / det, g00 / det};
}

void TriElement::reference_coords(Position r, double* xi) const
{
  // Least-squares barycentric coordinates, i.e. those of the orthogonal
  // projection of r onto the plane of the triangle
  const Position d = r - v_[2];
  const double p0 = e0_.dot(d);
  const double p1 = e1_.dot(d);
  const double la = ginv_[0] * p0 + ginv_[1] * p1;
  const double lb = ginv_[2] * p0 + ginv_[3] * p1;
  const double lc = 1.0 - la - lb;

  constexpr double tiny = 1.0e-14;
  const double s2 = la + lb;
  xi[0] = s2 > tiny ? (lb - la) / s2 : 0.0;
  xi[1] = 1.0 - 2.0 * lc;
}

bool TriElement::intersect(Position p0, Position d, SegmentHit& hit) const
{
  // Moller-Trumbore on the triangle C + u e0 + v e1 (u, v the barycentric
  // coordinates of A and B)
  constexpr double bary_tol = 1.0e-10;
  const double d_norm = d.norm();
  const Position pv = d.cross(e1_);
  const double det = e0_.dot(pv);
  if (d_norm == 0.0 || std::abs(det) <= 1.0e-12 * d_norm * 2.0 * volume_)
    return false;
  const double inv = 1.0 / det;

  const Position tv = p0 - v_[2];
  const double u = tv.dot(pv) * inv;
  if (u < -bary_tol || u > 1.0 + bary_tol)
    return false;
  const Position qv = tv.cross(e0_);
  const double v = d.dot(qv) * inv;
  if (v < -bary_tol || u + v > 1.0 + bary_tol)
    return false;
  const double t = e1_.dot(qv) * inv;
  if (t <= 0.0 || t > 1.0)
    return false;

  hit.t = t;
  hit.point = p0 + t * d;
  hit.cosine = d.dot(normal_) / d_norm;
  return true;
}

//==============================================================================
// Factories
//==============================================================================

unique_ptr<FETElement> make_face_element(const vector<Position>& vertices)
{
  switch (vertices.size()) {
  case 3:
    return make_unique<TriElement>(vertices);
  default:
    fatal_error(fmt::format("There is no FET element for a face with {} "
                            "vertices.",
      vertices.size()));
  }
}

} // namespace fet
} // namespace openmc
