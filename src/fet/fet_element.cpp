#include "openmc/fet/fet_element.h"

#include <algorithm>
#include <cmath>

#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"

namespace openmc {
namespace fet {

namespace {

//==============================================================================
// Tensor products of Jacobi polynomials P_{m_d}^(0, beta_d)(xi_d), for the
// elements whose basis is one. Modes are numbered lexicographically with the
// last axis fastest.
//==============================================================================

int tensor_n_modes(int dim, const Orders& orders)
{
  int n = 1;
  for (int d = 0; d < dim; ++d) {
    if (orders[d] < 0)
      return 0;
    n *= orders[d] + 1;
  }
  return n;
}

Orders tensor_mode(int dim, const Orders& orders, int index)
{
  Orders m {};
  for (int d = dim - 1; d >= 0; --d) {
    const int n = orders[d] + 1;
    m[d] = index % n;
    index /= n;
  }
  return m;
}

//! Product over the axes of (2 m_d + beta_d + 1) / (beta_d + 1)
double tensor_norm_sq(int dim, const Orders& beta, const Orders& m)
{
  double n2 = 1.0;
  for (int d = 0; d < dim; ++d)
    n2 *= (2.0 * m[d] + beta[d] + 1) / (beta[d] + 1);
  return n2;
}

//! Evaluate all modes at xi, clamped to [-1, 1]
void tensor_eval(int dim, const Orders& orders, const Orders& beta,
  const double* xi, double* psi)
{
  // Build the tensor product one axis at a time, in place. Going backwards
  // over the existing entries never overwrites an entry before it is read,
  // since entry m expands into entries m * n .. m * n + n - 1 >= m.
  double jac[MAX_ORDER + 1];
  psi[0] = 1.0;
  int size = 1;
  for (int d = 0; d < dim; ++d) {
    const int n = orders[d] + 1;
    calc_jacobi(
      orders[d], 0.0, beta[d], std::clamp(xi[d], -1.0, 1.0), 1.0, jac);
    for (int m = size - 1; m >= 0; --m) {
      const double v = psi[m];
      for (int k = n - 1; k >= 0; --k)
        psi[m * n + k] = v * jac[k];
    }
    size *= n;
  }
}

// Jacobi parameters of each element type
constexpr Orders TET_BETA {0, 1, 2};
constexpr Orders TRI_BETA {0, 1, 0};
constexpr int QUAD_SETS {3};
constexpr std::array<Orders, QUAD_SETS> QUAD_BETA {
  {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};

} // namespace

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
  on_orders_changed();
}

void FETElement::set_orders(const Orders& orders)
{
  for (int d = 0; d < dim_; ++d) {
    if (orders[d] > MAX_ORDER) {
      fatal_error(fmt::format("FET order {} exceeds the maximum supported "
                              "order {}.",
        orders[d], MAX_ORDER));
    }
  }
  for (int d = 0; d < dim_; ++d)
    orders_[d] = orders[d];
  on_orders_changed();
}

int FETElement::max_order() const
{
  return *std::max_element(orders_.begin(), orders_.begin() + dim_);
}

BoundingBox FETElement::bounding_box() const
{
  BoundingBox box = BoundingBox::inverted();
  for (int i = 0; i < n_vertices(); ++i)
    box.expand(vertex(i));
  return box;
}

int FETElement::eval(Position r, double* psi) const
{
  const int n = n_modes();
  if (n == 0)
    return 0;

  double xi[MAX_DIM];
  reference_coords(r, xi);
  eval_modes(xi, psi);
  return n;
}

std::string FETElement::mode_label(int index) const
{
  const Orders m = mode(index);
  std::string degrees = std::to_string(m[0]);
  for (int d = 1; d < dim_; ++d)
    degrees += fmt::format(", {}", m[d]);
  return fmt::format("FET mode ({})", degrees);
}

double FETElement::reconstruct(Position r, const double* moments) const
{
  const int n = n_modes();
  if (n == 0)
    return 0.0;

  double xi[MAX_DIM];
  reference_coords(r, xi);
  double values[MAX_MODES];
  expansion_modes(xi, values);

  double f = 0.0;
  for (int i = 0; i < n; ++i)
    f += mode_norm_sq(i) * moments[i] * values[i];
  return f / volume_;
}

//==============================================================================
// TetElement
//==============================================================================

TetElement::TetElement(const vector<Position>& vertices)
  : FETElement(3)
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

int TetElement::n_modes() const
{
  return tensor_n_modes(dim(), orders());
}

Orders TetElement::mode(int index) const
{
  return tensor_mode(dim(), orders(), index);
}

double TetElement::mode_norm_sq(int index) const
{
  return tensor_norm_sq(dim(), TET_BETA, mode(index));
}

void TetElement::eval_modes(const double* xi, double* psi) const
{
  tensor_eval(dim(), orders(), TET_BETA, xi, psi);
}

//==============================================================================
// TriElement
//==============================================================================

TriElement::TriElement(const vector<Position>& vertices)
  : FETElement(2)
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

int TriElement::n_modes() const
{
  return tensor_n_modes(dim(), orders());
}

Orders TriElement::mode(int index) const
{
  return tensor_mode(dim(), orders(), index);
}

double TriElement::mode_norm_sq(int index) const
{
  return tensor_norm_sq(dim(), TRI_BETA, mode(index));
}

void TriElement::eval_modes(const double* xi, double* psi) const
{
  tensor_eval(dim(), orders(), TRI_BETA, xi, psi);
}

//==============================================================================
// QuadElement
//==============================================================================

QuadElement::QuadElement(const vector<Position>& vertices)
  : FETElement(2)
{
  if (vertices.size() != 4)
    fatal_error("A quadrilateral element must have 4 vertices.");
  std::copy(vertices.begin(), vertices.end(), vertices_.begin());

  // Rotate the vertices so that the first one has the smallest corner
  // determinant. Ties within roundoff keep the earliest vertex, so that the
  // choice does not depend on the roundoff.
  const Position diag =
    (vertices_[2] - vertices_[0]).cross(vertices_[3] - vertices_[1]);
  std::array<double, 4> corner;
  for (int i = 0; i < 4; ++i) {
    const Position next = vertices_[(i + 1) % 4] - vertices_[i];
    const Position prev = vertices_[(i + 3) % 4] - vertices_[i];
    corner[i] = diag.dot(next.cross(prev));
  }
  const double smallest = *std::min_element(corner.begin(), corner.end());
  const double tie = smallest + TINY_BIT* diag.dot(diag);
  int first = 0;
  while (first < 3 && corner[first] > tie)
    ++first;
  std::rotate(
    vertices_.begin(), vertices_.begin() + first, vertices_.end());

  const Position a = vertices_[0];
  const Position b = vertices_[1];
  const Position c = vertices_[2];
  const Position d = vertices_[3];

  center_ = 0.25 * (a + b + c + d);
  u_ = 0.25 * (b + c - a - d);
  v_ = 0.25 * (c + d - a - b);
  w_ = 0.25 * (a - b + c - d);

  const Position uv = u_.cross(v_);
  const double uv_norm = uv.norm();
  if (uv_norm <= 1.0e-14 * u_.norm() * v_.norm()) {
    fatal_error("Degenerate quadrilateral encountered while building a "
                "functional expansion tally element.");
  }
  const Position normal = uv / uv_norm;

  // Every vertex is a distance |w . n| away from the plane through r0
  if (std::abs(w_.dot(normal)) > TINY_BIT * (u_.norm() + v_.norm())) {
    fatal_error("Non-planar quadrilateral encountered while building a "
                "functional expansion tally element.");
  }

  k2_ = normal.dot(u_.cross(w_));
  k1_center_ = uv_norm;
  k1_grad_ = normal.cross(w_);
  k0_grad_ = v_.cross(normal);

  // The Jacobian determinant u ^ v + xi_0 (u ^ w) + xi_1 (w ^ v) is linear, so
  // it is positive on the whole element if it is positive at the corners
  const double wv = normal.dot(w_.cross(v_));
  if (uv_norm - std::abs(k2_) - std::abs(wv) <= 1.0e-10 * uv_norm) {
    fatal_error("A quadrilateral functional expansion tally element must be "
                "strictly convex, with its vertices ordered around its "
                "boundary.");
  }

  // A has the smallest determinant, so j1 and j2 are negative only by roundoff
  j1_ = std::max(k2_, 0.0);
  j2_ = std::max(wv, 0.0);
  c0_ = uv_norm - j1_ - j2_;

  // The determinant averages to u ^ v over the reference square of area 4
  volume_ = 4.0 * uv_norm;
}

void QuadElement::reference_coords(Position r, double* xi) const
{
  const Position x = r - center_;
  const double k1 = k1_center_ + x.dot(k1_grad_);
  const double x_v = x.dot(k0_grad_);

  // The derivative of the quadratic at the root inside the element equals the
  // Jacobian determinant, so it is the root with the positive square root. This
  // form of it does not cancel, since k1 > 0 inside a convex element, and also
  // holds in the linear case k2 = 0. The discriminant can only be negative far
  // outside the element.
  const double disc = std::max(k1 * k1 + 4.0 * k2_ * x_v, 0.0);
  xi[0] = 2.0 * x_v / (k1 + std::sqrt(disc));
  // x - u xi_0 = xi_1 (v + w xi_0)
  const Position t = v_ + xi[0] * w_;
  xi[1] = (x - xi[0] * u_).dot(t) / t.dot(t);
}

int QuadElement::n_modes() const
{
  return QUAD_SETS * tensor_n_modes(dim(), orders());
}

Orders QuadElement::mode(int index) const
{
  return tensor_mode(dim(), orders(), index % tensor_n_modes(dim(), orders()));
}

double QuadElement::mode_norm_sq(int index) const
{
  const int set = index / tensor_n_modes(dim(), orders());
  return tensor_norm_sq(dim(), QUAD_BETA[set], mode(index));
}

std::string QuadElement::mode_label(int index) const
{
  const int set = index / tensor_n_modes(dim(), orders());
  return fmt::format("{}, set {}", FETElement::mode_label(index), set);
}

void QuadElement::eval_modes(const double* xi, double* psi) const
{
  // Clamped like the polynomials, which keeps every part non-negative
  const double part_1 = j1_ * (1.0 + std::clamp(xi[0], -1.0, 1.0));
  const double part_2 = j2_ * (1.0 + std::clamp(xi[1], -1.0, 1.0));
  const double det = c0_ + part_1 + part_2;
  const std::array<double, QUAD_SETS> weight {
    c0_ / det, part_1 / det, part_2 / det};

  expansion_modes(xi, psi);
  const int n = tensor_n_modes(dim(), orders());
  for (int set = 0; set < QUAD_SETS; ++set) {
    for (int m = 0; m < n; ++m)
      psi[set * n + m] *= weight[set];
  }
}

void QuadElement::expansion_modes(const double* xi, double* values) const
{
  const int n = tensor_n_modes(dim(), orders());
  for (int set = 0; set < QUAD_SETS; ++set)
    tensor_eval(dim(), orders(), QUAD_BETA[set], xi, values + set * n);
}

//==============================================================================
// Factories
//==============================================================================

std::unique_ptr<FETElement> make_face_element(const vector<Position>& vertices)
{
  switch (vertices.size()) {
  case 3:
    return std::make_unique<TriElement>(vertices);
  case 4:
    return std::make_unique<QuadElement>(vertices);
  default:
    fatal_error(fmt::format("There is no FET element for a face with {} "
                            "vertices.",
      vertices.size()));
  }
}

} // namespace fet
} // namespace openmc
