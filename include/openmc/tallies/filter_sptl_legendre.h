#ifndef OPENMC_TALLIES_FILTER_SPTL_LEGENDRE_H
#define OPENMC_TALLIES_FILTER_SPTL_LEGENDRE_H

#include <string>

#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

    enum class LegendreAxis { x, y, z };

//==============================================================================
//! Gives Legendre moments of the particle's normalized position along one or
//! more axes.
//!
//! Single-axis mode (original behaviour): one axis, order, min, max.
//! Multi-dimensional FET mode: up to three axes, each with its own order,
//! min, and max.  Bins are the tensor product of the per-axis polynomials;
//! weights are the corresponding products of Legendre values.  Both the
//! COLLISION and TRACKLENGTH estimators are supported.
//==============================================================================

    class SpatialLegendreFilter : public Filter {
    public:
        ~SpatialLegendreFilter() = default;

        std::string type_str() const override { return "spatiallegendre"; }
        FilterType type() const override { return FilterType::SPATIAL_LEGENDRE; }

        void from_xml(pugi::xml_node node) override;
        void get_all_bins(const Particle& p, TallyEstimator estimator,
                          FilterMatch& match) const override;
        void to_statepoint(hid_t filter_group) const override;
        std::string text_label(int bin) const override;

        // Single-axis backward-compatible accessors
        int order() const { return axes_[0].order; }
        void set_order(int order);
        LegendreAxis axis() const { return axes_[0].axis; }
        void set_axis(LegendreAxis axis);
        double min() const { return axes_[0].min; }
        double max() const { return axes_[0].max; }
        void set_minmax(double min, double max);

        // Multi-dimensional FET
        int n_axes() const { return static_cast<int>(axes_.size()); }
        void add_axis(LegendreAxis axis, int order, double min, double max);

    private:
        struct AxisDef {
            LegendreAxis axis;
            int          order;
            double       min;
            double       max;
        };

        void update_n_bins();
        vector<int> decode_bin(int bin) const;

        vector<AxisDef> axes_;
    };

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_SPTL_LEGENDRE_H