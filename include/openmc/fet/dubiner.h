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
//
// The collapsed (Duffy) coordinates of a tetrahedron with vertices A, B, C, D
// (D collapsed) are related to the barycentric coordinates by
//
//   lambda_A = (1+gamma)(1+zeta)(1-eta)/8,  lambda_B =
//   (1+gamma)(1+zeta)(1+eta)/8 lambda_C = (1+gamma)(1-zeta)/4,         lambda_D
//   = (1-gamma)/2
//
// and the Dubiner mode (p, q, r) is
//
//   P_p(eta) ((1+zeta)/2)^p P_q^(0,2p+1)(zeta)
//     ((1+gamma)/2)^(p+q) P_r^(0,2p+2q+2)(gamma)
//
// which is a polynomial of total degree p+q+r in the physical coordinates. It
// is evaluated here directly from barycentric coordinates using scaled Jacobi
// recurrences, which avoids the 0/0 of the collapsed coordinates at the
// collapsed vertex and edge. Modes are scaled so that over any tetrahedron T
//
//   integral_T psi_i psi_j dV = |T| delta_ij
//
// making mode 0 identically one and the element average of f equal to its
// zeroth coefficient.
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
