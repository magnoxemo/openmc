//! \file dubiner.h
//! \brief Orthogonal Dubiner (Koornwinder) bases on simplices for functional
//! expansion tallies on unstructured meshes.

#ifndef OPENMC_FET_DUBINER_H
#define OPENMC_FET_DUBINER_H

#include <array>

namespace openmc {
namespace fet {

//==============================================================================
// Constants
//==============================================================================

//! Maximum supported total polynomial degree of a simplex expansion
constexpr int MAX_ORDER {10};

//! Number of tetrahedral modes of total degree <= order
constexpr int n_modes_tet(int order)
{
  return order < 0 ? 0 : (order + 1) * (order + 2) * (order + 3) / 6;
}

//! Number of triangular modes of total degree <= order
constexpr int n_modes_tri(int order)
{
  return order < 0 ? 0 : (order + 1) * (order + 2) / 2;
}

//! Maximum number of tetrahedral modes for any supported order
constexpr int MAX_MODES_TET {n_modes_tet(MAX_ORDER)};

//! Maximum number of triangular modes for any supported order
constexpr int MAX_MODES_TRI {n_modes_tri(MAX_ORDER)};

//==============================================================================
// Mode indexing
//
// Modes are ordered hierarchically: first by total degree k, then by p and
// then q within a degree. The first n_modes_tet(N) modes are therefore exactly
// the order-N basis, so changing the order of an element only appends or drops
// trailing coefficients.
//==============================================================================

//! Index of tetrahedral mode (p, q, r) in the hierarchical ordering
int tet_mode_index(int p, int q, int r);

//! Indices (p, q, r) of the tetrahedral mode at a hierarchical index
std::array<int, 3> tet_mode(int index);

//! Index of triangular mode (p, q) in the hierarchical ordering
int tri_mode_index(int p, int q);

//! Indices (p, q) of the triangular mode at a hierarchical index
std::array<int, 2> tri_mode(int index);

//==============================================================================
// Basis evaluation
//==============================================================================

//! Evaluate all tetrahedral modes of total degree <= order
//
//! \param[in] order Maximum total degree, 0 <= order <= MAX_ORDER
//! \param[in] lambda Barycentric coordinates with respect to (A, B, C, D)
//! \param[out] out Array of at least n_modes_tet(order) values in
//!   hierarchical order
void eval_tet(int order, const double lambda[4], double* out);

//! Evaluate all triangular modes of total degree <= order
//
//! Normalized so that integral_T psi_i psi_j dA = |T| delta_ij with vertex C
//! collapsed.
//
//! \param[in] order Maximum total degree, 0 <= order <= MAX_ORDER
//! \param[in] lambda Barycentric coordinates with respect to (A, B, C)
//! \param[out] out Array of at least n_modes_tri(order) values in
//!   hierarchical order
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
