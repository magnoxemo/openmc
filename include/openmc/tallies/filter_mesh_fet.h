#ifndef OPENMC_TALLIES_FILTER_MESH_FET_H
#define OPENMC_TALLIES_FILTER_MESH_FET_H

#include <cstdint>

#include "openmc/fet/element_map.h"
#include "openmc/tallies/filter_mesh.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Element-wise functional expansion tally (FET) on an unstructured mesh of
//! linear tetrahedra.
//!
//! Each element e carries an orthogonal Dubiner expansion of total degree
//! order(e); a negative order excludes the element from the tally. The filter
//! has one bin per (element, mode) pair, stored element-major: the bins of
//! element e are [offset(e), offset(e+1)) and follow the hierarchical mode
//! ordering of fet::tet_mode_index. The weight of each bin is the value of
//! the basis function at the event location, so the expansion coefficient of
//! mode i on element e is
//!
//!   a_i = (tally in bin offset(e) + i) / V_e
//!
//! and the reconstructed distribution is f(r) = sum_i a_i psi_i(r). Only
//! analog and collision estimators are supported.
//!
//! The per-element orders determine the number of bins and must therefore be
//! set before any tally using this filter allocates its results.
//==============================================================================

class MeshFETFilter : public MeshFilter {
public:
  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "meshfet"; }
  FilterType type() const override { return FilterType::MESH_FET; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  void set_mesh(int32_t mesh) override;

  //! Set the same expansion order on every element
  void set_order(int order);

  //! Set the expansion order of each element; negative excludes an element
  void set_orders(const vector<int>& orders);

  const vector<int>& orders() const { return orders_; }

  int order(int element) const { return orders_[element]; }

  //! First filter bin belonging to an element
  int offset(int element) const { return offsets_[element]; }

  int max_order() const { return max_order_; }

  //! Number of mesh elements covered by the filter
  int n_elements() const { return static_cast<int>(maps_.size()); }

  //! Affine map of an element
  const fet::AffineTetMap& element_map(int element) const
  {
    return maps_[element];
  }

  //! Element containing a filter bin
  int element_from_bin(int bin) const;

  //! Unsigned volume of an element as seen by the expansion
  double volume(int element) const { return maps_[element].volume(); }

  //----------------------------------------------------------------------------
  // Evaluation
  //
  // The basis orientation of an element is defined by the vertex order of the
  // mesh that built this filter, which may differ between mesh interfaces.
  // Coupled codes must therefore evaluate the expansion through these methods
  // rather than rebuilding the basis from their own connectivity.

  //! Apply the filter's translation and rotation to a global position
  Position mesh_position(Position r) const;

  //! Evaluate the basis functions of an element
  //
  //! \param[in] element Element index (mesh bin)
  //! \param[in] r Position in global coordinates
  //! \param[out] psi Array of at least fet::n_modes_tet(order(element)) values
  //! \return Number of basis functions written
  int evaluate_basis(int element, Position r, double* psi) const;

  //! Reconstruct the expanded distribution of an element
  //
  //! \param[in] element Element index (mesh bin)
  //! \param[in] r Position in global coordinates
  //! \param[in] values Tally values of the element's bins, i.e. the entries
  //!   [offset(element), offset(element + 1)) of the filter
  //! \return Distribution per unit volume at r
  double reconstruct(int element, Position r, const double* values) const;

private:
  //! Recompute bin offsets from the element orders
  void update_bins();

  //! Evaluate the basis of an element at a position in the mesh frame
  int basis_in_mesh_frame(int element, Position r, double* psi) const;

  //----------------------------------------------------------------------------
  // Data members

  vector<fet::AffineTetMap> maps_; //!< Affine map of each element
  vector<int> orders_;             //!< Expansion order of each element
  vector<int> offsets_;            //!< First bin of each element (size n+1)
  int max_order_ {-1};             //!< Maximum order over all elements
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_MESH_FET_H
