#ifndef OPENMC_TALLIES_FILTER_SURFACE_FET_H
#define OPENMC_TALLIES_FILTER_SURFACE_FET_H

#include <cstdint>
#include <unordered_map>

#include "openmc/fet/fet_element.h"
#include "openmc/memory.h"
#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Facet-wise functional expansion tally (FET) on XDG surface meshes.
//!
//! Every facet of every filtered surface is a fet::FETElement (linear
//! triangles for now) with its own expansion order along each reference axis.
//! The filter has one bin per (facet, mode) pair, stored facet-major: the bins
//! of facet e are [offset(e), offset(e+1)), in the mode order of the element.
//! Facets are numbered surface after surface, in the order of the surfaces of
//! the filter; the facets of a surface are in the order of the XDG mesh
//! manager. The weight of a bin is the value of the unnormalized mode at the
//! crossing point, as for MeshFETFilter, so with the squared mode norm N_i^2
//! the reconstructed distribution on facet e is
//!
//!   f(r) = sum_i N_i^2 * (tally in bin offset(e) + i) * psi_i(r) / area_e
//!
//! The filter only locates the facet and weights the bins. The scoring of
//! surface tallies (net current or surface-crossing flux, including the sign of
//! the crossing) is done by the surface tally machinery.
//!
//! The facets are built when the surfaces are set, which resets all orders to
//! the order given by set_order() (0 until set). After changing the orders of
//! facets directly through element(), call update_bins().
//==============================================================================

class SurfaceFETFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "surfacefet"; }
  FilterType type() const override { return FilterType::SURFACE_FET; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  //! Set the surfaces of the filter and build the facets of each of them
  //
  //! \param[in] surfaces Indices into model::surfaces; each must be an XDG
  //!   surface mesh surface
  void set_surfaces(span<int32_t> surfaces);

  //! Set the same expansion order on every axis of every facet; negative
  //! excludes the facets. Also the order of facets built later.
  void set_order(int order);

  //! Recompute the bin offsets from the facet orders. Call after changing the
  //! orders of facets through element().
  void update_bins();

  //! First filter bin belonging to a facet
  int offset(int facet) const { return offsets_[facet]; }

  //! Largest order over all facets and axes
  int max_order() const { return max_order_; }

  //! Number of facets covered by the filter
  int n_elements() const { return static_cast<int>(elements_.size()); }

  const fet::FETElement& element(int facet) const { return *elements_[facet]; }

  //! Facet element, e.g. to change its orders; call update_bins() afterwards
  fet::FETElement& element(int facet) { return *elements_[facet]; }

  //! Facet containing a filter bin
  int element_from_bin(int bin) const;

  //! Index into the surfaces of the filter of the surface owning a facet
  int surface_slot_from_element(int facet) const;

private:
  //! Append the facets of one surface to elements_
  void build_facets(int32_t surface_index);


  //----------------------------------------------------------------------------
  // Data members

  //! Indices of the filtered surfaces in model::surfaces
  vector<int32_t> surfaces_;

  //! Map from a surface index to its position in surfaces_
  std::unordered_map<int32_t, int> slot_map_;

  //! Map from the XDG ID of a facet to its index in elements_
  std::unordered_map<int32_t, int> facet_map_;

  //! First facet of each surface of surfaces_ (size n_surfaces + 1)
  vector<int> surface_first_;

  vector<unique_ptr<fet::FETElement>> elements_; //!< Facet of each index
  vector<int> offsets_; //!< First bin of each facet (size n+1)
  int max_order_ {-1};  //!< Maximum order over all facets and axes
  int order_ {0};       //!< Order given to newly built facets
};

} // namespace openmc

#endif // OPENMC_TALLIES_FILTER_SURFACE_FET_H
