#ifndef OPENMC_FILTER_SURFACE_FET_H
#define OPENMC_FILTER_SURFACE_FET_H

#include <memory>
#include <vector>

#include "filter.h"
#include "openmc/fet/fet_element.h"
#include "openmc/tallies/filter_surface.hpp"

namespace openmc {

// maybe inheriting from the mesh_fet_filter will make things easier but
// not now. I am mainly intending this to be a dagmc surface filter

class DAGSurface;

class SurfaceFETFilter : public SurfaceFETFilter {
public:
  ~SurfaceFETFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "surface_fet"; }
  FilterType type() const override { return FilterType::SURFACE_FET; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  void set_surfaces(span<int32_t> surfaces);

  //! Set the same expansion order on every axis of every element
  void set_order(unsigned int order);

  //! Set the expansion order of each element, the same on all of its axes;
  //! negative excludes an element
  void set_orders(const vector<unsigned int>& orders);

  //! Recompute the bin offsets from the element orders. Call after changing
  //! the orders of elements through element().
  void update_bins();

  //! First filter bin belonging to an element
  int offset(int surface_element_index) const
  {
    return offsets_[surface_element_index];
  }

  //! Largest order over all elements and axes
  int max_order() const { return max_order_; }

  //! Number of elements covered by the filter
  int n_elements() const { return static_cast<int>(elements_.size()); }

  const fet::FETElement& element(int surface_element_index) const
  {
    return *elements_[surface_element_index];
  }

  //! Element, e.g. to change its orders; call update_bins()
  fet::FETElement& element(int surface_element_index)
  {
    return *elements_[surface_element_index];
  }

  //! Element containing a filter bin
  int element_from_bin(int bin) const;

  //! Evaluate the modes of an element
  //
  //! \param[in] element Element index
  //! \param[in] r Position in global coordinates
  //! \param[out] psi Array of at least element(element).n_modes() values
  //! \return Number of modes written
  int evaluate_basis(int surface_element_index, Position r, double* psi) const;

  //! Reconstruct the expanded distribution of an element
  //! \param[in] surface_element_index Element index
  //! \param[in] r Position in global coordinates
  //! \param[in] values Tally values of the element's bins, i.e. the entries
  //!   [offset(element), offset(element + 1)) of the filter
  //! \return Distribution per unit volume (area for 2D elements) at r
  double reconstruct(int element, Position r, const double* values) const;

private:
  // builds surface elements by querying the dagmc surface
  // this one will live in the set_surfaces
  virtual void build_surface_elements(const DAGSurface& dg_surface);

  std::vector<std::unique_ptr<openmc::fet::TriElement>> elements_;
  std::vector<std::unique_ptr<fet::FETElement>>
    elements_;               //!< Element of each index
  std::vector<int> offsets_; //!< First bin of each element (size n+1)
  int max_order_ {-1};       //!< Maximum order over all elements and axes
}
} // namespace openmc

#endif // OPENMC_FILTER_SURFACE_FET_H
