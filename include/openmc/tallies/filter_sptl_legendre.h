#ifndef OPENMC_TALLIES_FILTER_SPTL_LEGENDRE_H
#define OPENMC_TALLIES_FILTER_SPTL_LEGENDRE_H

#include <string>

#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

enum class LegendreAxis { x = 0, y = 1, z = 2 };

//==============================================================================
//! Multi-dimensional Functional Expansion Tally (FET) filter using Legendre
//! polynomials.
//!
//! Each active axis maps the particle's spatial coordinate to [-1, 1] and
//! evaluates Legendre polynomials P_0 … P_n.  Bins are the tensor product
//! of the active axes in x → y → z order; weights are the corresponding
//! products of Legendre values, enabling the collision estimator to directly
//! tally the expansion moments.
//!
//! 1-D usage: add a single axis.
//! 2-D / 3-D usage: add two or three axes in any order — they are always
//! sorted into x → y → z order internally.
//==============================================================================

class SpatialLegendreFilter : public Filter {
public:
  //! Per-axis configuration.
  struct AxisDef {
    LegendreAxis axis;
    int order;  //!< Maximum Legendre polynomial order (>= 0).
    double min; //!< Physical coordinate lower bound.
    double max; //!< Physical coordinate upper bound.
  };

  ~SpatialLegendreFilter() = default;

  //----------------------------------------------------------------------------
  // Virtual interface
  //----------------------------------------------------------------------------

  std::string type_str() const override { return "spatiallegendre"; }
  FilterType type() const override { return FilterType::SPATIAL_LEGENDRE; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------

  //! Register an axis.  Axes are stored in x -> y -> z order regardless of
  //! call order.  Each axis may be added at most once.
  void add_axis(LegendreAxis axis, int order, double min, double max);

  //----------------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------------

  int n_axes() const { return static_cast<int>(axes_.size()); }
  const vector<AxisDef>& axes() const { return axes_; }
  const AxisDef& axis(int d) const { return axes_[d]; }

private:
  //----------------------------------------------------------------------------
  // Helpers
  //----------------------------------------------------------------------------

  //! Recompute n_bins_ as prod(order_d + 1) over active axes.
  void update_n_bins();

  //! Convert a flat bin index to per-axis polynomial indices (xyz-major).
  vector<int> decode_bin(int bin) const;

  //----------------------------------------------------------------------------
  // Data
  //----------------------------------------------------------------------------

  vector<AxisDef> axes_; //!< Active axes, always in x -> y -> z order.
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_SPTL_LEGENDRE_H