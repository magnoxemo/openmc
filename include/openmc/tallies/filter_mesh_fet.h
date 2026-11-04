#ifndef OPENMC_TALLIES_FILTER_MESH_FET_H
#define OPENMC_TALLIES_FILTER_MESH_FET_H

#include <cstdint>

#include "openmc/fet/fet_element.h"
#include "openmc/memory.h"
#include "openmc/tallies/filter_mesh.h"
#include "openmc/vector.h"

namespace openmc {

class UnstructuredMesh;

//==============================================================================
//! Element-wise functional expansion tally (FET) on an unstructured mesh of
//! linear tetrahedra.
//!
//! Each element e carries the orthogonal expansion of a fet::FETElement, with
//! an expansion order along each of its reference axes (see
//! fet::FETElement::set_order); a negative order excludes the element from the
//! tally. The filter has one bin per (element, mode) pair, stored
//! element-major: the bins of element e are [offset(e), offset(e+1)) in the
//! mode order of the element. The weight of each bin is the value of the
//! unnormalized mode at the event location, so that no square roots are
//! evaluated during transport. With the squared mode norm N_i^2, the
//! reconstructed distribution on element e is
//!
//!   f(r) = sum_i N_i^2 * (tally in bin offset(e) + i) * psi_i(r) / volume_e
//!
//! Only analog and collision estimators are supported.
//!
//! The per-element orders determine the number of bins and must therefore be
//! set before any tally using this filter allocates its results. After
//! changing the orders of elements directly, call update_bins().
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

  //! Set the same expansion order on every axis of every element
  void set_order(int order);

  //! Set the expansion order of each element, the same on all of its axes;
  //! negative excludes an element
  void set_orders(const vector<int>& orders);

  //! Recompute the bin offsets from the element orders. Call after changing
  //! the orders of elements through element().
  void update_bins();

  //! First filter bin belonging to an element
  int offset(int element) const { return offsets_[element]; }

  //! Largest order over all elements and axes
  int max_order() const { return max_order_; }

  //! Number of elements covered by the filter
  int n_elements() const { return static_cast<int>(elements_.size()); }

  const fet::FETElement& element(int element) const
  {
    return *elements_[element];
  }

  //! Element, e.g. to change its orders; call update_bins() afterwards
  fet::FETElement& element(int element) { return *elements_[element]; }

  //! Element containing a filter bin
  int element_from_bin(int bin) const;

  //----------------------------------------------------------------------------
  // Evaluation
  //
  // The orientation of the modes of an element is defined by the vertex order
  // of the mesh that built this filter, which may differ between mesh
  // interfaces. Coupled codes must therefore evaluate the expansion through
  // these methods rather than rebuilding the basis from their own connectivity.

  //! Apply the filter's translation and rotation to a global position
  Position mesh_position(Position r) const;

  //! Evaluate the modes of an element
  //
  //! \param[in] element Element index
  //! \param[in] r Position in global coordinates
  //! \param[out] psi Array of at least element(element).n_modes() values
  //! \return Number of modes written
  int evaluate_basis(int element, Position r, double* psi) const;

  //! Reconstruct the expanded distribution of an element
  //
  //! \param[in] element Element index
  //! \param[in] r Position in global coordinates
  //! \param[in] values Tally values of the element's bins, i.e. the entries
  //!   [offset(element), offset(element + 1)) of the filter
  //! \return Distribution per unit volume (area for 2D elements) at r
  double reconstruct(int element, Position r, const double* values) const;

protected:
  //! Build elements_ from the (unstructured) mesh
  virtual void build_elements(const UnstructuredMesh& mesh);

  //! Evaluate the modes of an element at a position in the mesh frame
  int basis_in_mesh_frame(int element, Position r, double* psi) const;

  //----------------------------------------------------------------------------
  // Data members

  vector<unique_ptr<fet::FETElement>> elements_; //!< Element of each index
  vector<int> offsets_; //!< First bin of each element (size n+1)
  int max_order_ {-1};  //!< Maximum order over all elements and axes
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_MESH_FET_H
