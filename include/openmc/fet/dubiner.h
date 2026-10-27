//! \file dubiner.h
//! \brief Orthogonal tensor-product Legendre/Jacobi bases on collapsed simplex
//! coordinates for functional expansion tallies on unstructured meshes.
//
//! A tetrahedral mode (i, j, k) is P_i(eta) P_j^(0,1)(zeta) P_k^(0,2)(gamma),
//! where (eta, zeta, gamma) are the Duffy-collapsed coordinates of the
//! barycentric coordinates (A, B, C, D) and the Jacobi weights match the
//! Jacobian of the collapse, (1 + zeta)(1 + gamma)^2. A triangular mode (i, j)
//! is P_i(eta) P_j^(0,1)(zeta). These are not polynomials in the physical
//! coordinates.

#ifndef OPENMC_FET_DUBINER_H
#define OPENMC_FET_DUBINER_H

#include <array>

namespace openmc {
namespace fet {

//==============================================================================
// Constants
//==============================================================================

//! Maximum supported 1D degree of a simplex expansion
constexpr int MAX_ORDER {10};

//! Number of tetrahedral modes with i, j, k <= order
constexpr int n_modes_tet(int order)
{
  return order < 0 ? 0 : (order + 1) * (order + 1) * (order + 1);
}

//! Number of triangular modes with i, j <= order
constexpr int n_modes_tri(int order)
{
  return order < 0 ? 0 : (order + 1) * (order + 1);
}

//! Maximum number of tetrahedral modes for any supported order
constexpr int MAX_MODES_TET {n_modes_tet(MAX_ORDER)};

//! Maximum number of triangular modes for any supported order
constexpr int MAX_MODES_TRI {n_modes_tri(MAX_ORDER)};

//==============================================================================
// Mode indexing
//
// Modes are ordered in shells of m = max(i, j, k). The first n_modes_tet(N)
// modes are exactly the order-N basis, so changing the order of an element only
// appends or drops trailing coefficients. Within shell m the order is: i = m
// (j, then k), then i < m with j = m (i, then k), then i, j < m with k = m (i,
// then j).
//==============================================================================

//! Index of tetrahedral mode (i, j, k) in the shell ordering
int tet_mode_index(int i, int j, int k);

//! Indices (i, j, k) of the tetrahedral mode at a shell-ordered index
std::array<int, 3> tet_mode(int index);

//! Index of triangular mode (i, j) in the shell ordering
int tri_mode_index(int i, int j);

//! Indices (i, j) of the triangular mode at a shell-ordered index
std::array<int, 2> tri_mode(int index);

//==============================================================================
// Basis evaluation
//==============================================================================

//! Squared normalization factor of tetrahedral mode (i, j, k)
//
//! Equal to (2i + 1)(j + 1)(2k + 3)/3, so that the mean of psi^2 over any
//! tetrahedron is 1/N^2. Multiplying the mode by N gives the basis with
//! integral_T psi_i psi_j dV = |T| delta_ij, in which mode 0 is identically one.
//! The modes returned by eval_tet are not normalized, to keep the square roots
//! out of transport. Intended for post-processing.
double tet_mode_norm_sq(int i, int j, int k);

//! Normalization factor of tetrahedral mode (i, j, k), sqrt of the above
double tet_mode_norm(int i, int j, int k);

//! Squared normalization factor (2i + 1)(j + 1) of triangular mode (i, j)
double tri_mode_norm_sq(int i, int j);

//! Normalization factor of triangular mode (i, j), sqrt of the above
double tri_mode_norm(int i, int j);

//! Evaluate all tetrahedral modes with i, j, k <= order, unnormalized
//
//! Where the collapse is singular (A and B coincide with the collapsed edge, or
//! A, B and C with the collapsed vertex) the affected coordinate is set to 0.
//
//! \param[in] order Maximum 1D degree, 0 <= order <= MAX_ORDER
//! \param[in] lambda Barycentric coordinates with respect to (A, B, C, D)
//! \param[out] out Array of at least n_modes_tet(order) values in shell order
void eval_tet(int order, const double lambda[4], double* out);

//! Evaluate all triangular modes with i, j <= order, unnormalized, with vertex
//! C collapsed
//
//! \param[in] order Maximum 1D degree, 0 <= order <= MAX_ORDER
//! \param[in] lambda Barycentric coordinates with respect to (A, B, C)
//! \param[out] out Array of at least n_modes_tri(order) values in shell order
void eval_tri(int order, const double lambda[3], double* out);

//! Evaluate t^k P_k^(alpha,beta)(z/t) for k = 0..n
//
//! This is the homogenized Jacobi recurrence; it is a polynomial in (z, t) and
//! stays finite as t -> 0. With t = 1 it reduces to the ordinary Jacobi
//! polynomials P_k^(alpha,beta)(z).
//
//! \param[in] n Maximum degree
//! \param[in] alpha Jacobi parameter alpha > -1
//! \param[in] beta Jacobi parameter beta > -1
//! \param[in] z Numerator of the argument
//! \param[in] t Scaling (denominator of the argument)
//! \param[out] out Array of at least n + 1 values
void scaled_jacobi(
  int n, double alpha, double beta, double z, double t, double* out);

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_DUBINER_H
