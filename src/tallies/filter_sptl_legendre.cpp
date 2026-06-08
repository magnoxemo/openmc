#include "openmc/tallies/filter_sptl_legendre.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
//                                 local helpers
//==============================================================================

namespace {

const char* axis_name(LegendreAxis a)
{
  switch (a) {
  case LegendreAxis::x:
    return "x";
  case LegendreAxis::y:
    return "y";
  default:
    return "z";
  }
}

double axis_coord(const Particle& p, LegendreAxis a)
{
  switch (a) {
  case LegendreAxis::x:
    return p.r().x;
  case LegendreAxis::y:
    return p.r().y;
  default:
    return p.r().z;
  }
}

LegendreAxis axis_from_char(char c)
{
  switch (c) {
  case 'x':
    return LegendreAxis::x;
  case 'y':
    return LegendreAxis::y;
  case 'z':
    return LegendreAxis::z;
  default:
    throw std::invalid_argument {
      fmt::format("Invalid FET axis '{}'; must be 'x', 'y', or 'z'.", c)};
  }
}

} // namespace

void SpatialLegendreFilter::add_axis(
  LegendreAxis axis, int order, double min, double max)
{
  if (order < 0)
    throw std::invalid_argument {"Legendre order must be >= 0."};
  if (max <= min)
    throw std::invalid_argument {"FET axis max must be greater than min."};
  for (const auto& ad : axes_)
    if (ad.axis == axis)
      throw std::invalid_argument {
        fmt::format("Axis '{}' has already been added to this FET filter.",
          axis_name(axis))};

  axes_.push_back({axis, order, min, max});

  // For reconstruction sanity enforce x -> y -> z storage order so bin layout
  // is always xyz-major
  std::sort(axes_.begin(), axes_.end(), [](const AxisDef& a, const AxisDef& b) {
    return static_cast<int>(a.axis) < static_cast<int>(b.axis);
  });

  update_n_bins();
}

void SpatialLegendreFilter::from_xml(pugi::xml_node node)
{
  // Axes are written as child elements <x>, <y>, <z> each containing
  // <order>, <min>, <max>.  Read them in xyz order — add_axis will sort
  // internally but iterating in order avoids any ambiguity.
  static constexpr LegendreAxis axis_vals[3] = {
    LegendreAxis::x, LegendreAxis::y, LegendreAxis::z};
  static constexpr const char* axis_tags[3] = {"x", "y", "z"};

  bool found_any = false;
  for (int d = 0; d < 3; ++d) {
    auto child = node.child(axis_tags[d]);
    if (!child)
      continue;
    found_any = true;
    add_axis(axis_vals[d], std::stoi(get_node_value(child, "order")),
      std::stod(get_node_value(child, "min")),
      std::stod(get_node_value(child, "max")));
  }

  if (!found_any)
    throw std::invalid_argument {
      "SpatialLegendreFilter requires at least one axis child element "
      "(<x>, <y>, or <z>) with <order>, <min>, and <max>."};
}

void SpatialLegendreFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // Evaluate Legendre basis for each active axis.
  // Return immediately if the particle is outside any axis domain.
  vector<vector<double>> legendre_weights(axes_.size());
  for (std::size_t d = 0; d < axes_.size(); ++d) {
    const auto& ax = axes_[d];
    const double coord = axis_coord(p, ax.axis);

    if (coord < ax.min || coord > ax.max)
      return;

    const double xi = 2.0 * (coord - ax.min) / (ax.max - ax.min) - 1.0;
    legendre_weights[d].resize(ax.order + 1);
    calc_pn_c(ax.order, xi, legendre_weights[d].data());
  }

  // Enumerate all tensor-product bins.
  // Weight = product of per-axis Legendre values at the decoded polynomial
  // indices.  The bin ordering is xyz-major because axes_ is xyz-sorted.
  for (int bin = 0; bin < n_bins_; ++bin) {
    const auto idx = decode_bin(bin);
    double weight = 1.0;
    for (std::size_t d = 0; d < axes_.size(); ++d)
      weight *= legendre_weights[d][idx[d]];
    match.bins_.push_back(bin);
    match.weights_.push_back(weight);
  }
}

void SpatialLegendreFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);

  // Write axis metadata as parallel arrays so postprocessors can read the
  // tensor shape directly: co effs.reshape(tensor_shape) gives a xyz-indexed
  // array with no further reordering needed.
  const std::size_t n = axes_.size();
  vector<std::string> labels(n);
  vector<int> orders(n);
  vector<double> mins(n), maxs(n), tensor_shape(n);

  for (std::size_t d = 0; d < n; ++d) {
    labels[d] = axis_name(axes_[d].axis);
    orders[d] = axes_[d].order;
    mins[d] = axes_[d].min;
    maxs[d] = axes_[d].max;
    tensor_shape[d] = axes_[d].order + 1;
  }

  write_dataset(filter_group, "axes", labels);
  write_dataset(filter_group, "orders", orders);
  write_dataset(filter_group, "mins", mins);
  write_dataset(filter_group, "maxs", maxs);
  write_dataset(filter_group, "tensor_shape", tensor_shape);
}

std::string SpatialLegendreFilter::text_label(int bin) const
{
  const auto idx = decode_bin(bin);
  std::string label = "FET";
  for (std::size_t d = 0; d < axes_.size(); ++d)
    label += fmt::format(" P{}({})", idx[d], axis_name(axes_[d].axis));
  return label;
}

void SpatialLegendreFilter::update_n_bins()
{
  int total = 1;
  for (const auto& ad : axes_)
    total *= ad.order + 1;
  n_bins_ = total;
}

std::vector<int> SpatialLegendreFilter::decode_bin(int bin) const
{
  // Row-major (xyz-major) decode: peel off the fastest-varying axis last.
  vector<int> idx(axes_.size());
  int remainder = bin;
  for (int d = static_cast<int>(axes_.size()) - 1; d >= 0; --d) {
    idx[d] = remainder % (axes_[d].order + 1);
    remainder /= axes_[d].order + 1;
  }
  return idx;
}

//==============================================================================
// C-API functions
//==============================================================================

std::pair<int, SpatialLegendreFilter*> check_sptl_legendre_filter(int32_t index)
{
  // Make sure this is a valid index to an allocated filter.
  int err = verify_filter(index);
  if (err) {
    return {err, nullptr};
  }

  // Get a pointer to the filter and downcast.
  const auto& filt_base = model::tally_filters[index].get();
  auto* filt = dynamic_cast<SpatialLegendreFilter*>(filt_base);

  // Check the filter type.
  if (!filt) {
    set_errmsg("Not a spatial Legendre filter.");
    err = OPENMC_E_INVALID_TYPE;
  }
  return {err, filt};
}

extern "C" int openmc_spatial_legendre_filter_add_axis(
  int32_t index, int axis, int order, double min, double max)
{
  auto [err, filter] = check_sptl_legendre_filter(index);
  if (err)
    return err;
  try {
    filter->add_axis(static_cast<LegendreAxis>(axis), order, min, max);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_INVALID_ARGUMENT;
  }
  return 0;
}

extern "C" int openmc_spatial_legendre_filter_get_n_axes(int32_t index, int* n)
{
  auto [err, filter] = check_sptl_legendre_filter(index);
  if (err)
    return err;
  *n = filter->n_axes();
  return 0;
}

extern "C" int openmc_spatial_legendre_filter_get_axis(
  int32_t index, int d, int* axis, int* order, double* min, double* max)
{
  auto [err, filter] = check_sptl_legendre_filter(index);
  if (err)
    return err;
  if (d < 0 || d >= filter->n_axes()) {
    set_errmsg(
      fmt::format("Axis index {} out of range [0, {}).", d, filter->n_axes()));
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  const auto& ad = filter->axis(d);
  *axis = static_cast<int>(ad.axis);
  *order = ad.order;
  *min = ad.min;
  *max = ad.max;
  return 0;
}

} // namespace openmc