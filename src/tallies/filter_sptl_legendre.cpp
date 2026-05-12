#include "openmc/tallies/filter_sptl_legendre.h"

#include <stdexcept>
#include <utility>

#include <fmt/core.h>

#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/xml_interface.h"

namespace openmc {

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

} // anonymous namespace

void SpatialLegendreFilter::from_xml(pugi::xml_node node)
{
  bool has_axis_children =
    node.child("x") || node.child("y") || node.child("z");

  if (has_axis_children) {
    static constexpr LegendreAxis axis_vals[3] = {
      LegendreAxis::x, LegendreAxis::y, LegendreAxis::z};
    static constexpr const char* axis_tags[3] = {"x", "y", "z"};
    for (int d = 0; d < 3; ++d) {
      auto child = node.child(axis_tags[d]);
      if (!child)
        continue;
      add_axis(axis_vals[d], std::stoi(get_node_value(child, "order")),
        std::stod(get_node_value(child, "min")),
        std::stod(get_node_value(child, "max")));
    }
  } else {
    this->set_order(std::stoi(get_node_value(node, "order")));
    auto axis_str = get_node_value(node, "axis");
    switch (axis_str[0]) {
    case 'x':
      this->set_axis(LegendreAxis::x);
      break;
    case 'y':
      this->set_axis(LegendreAxis::y);
      break;
    case 'z':
      this->set_axis(LegendreAxis::z);
      break;
    default:
      throw std::runtime_error {
        "Axis for SpatialLegendreFilter must be 'x', 'y', or 'z'"};
    }
    this->set_minmax(std::stod(get_node_value(node, "min")),
      std::stod(get_node_value(node, "max")));
  }
}

void SpatialLegendreFilter::set_order(int order)
{
  if (order < 0)
    throw std::invalid_argument {"Legendre order must be non-negative."};
  if (axes_.empty())
    axes_.push_back({LegendreAxis::x, order, -1.0, 1.0});
  else
    axes_[0].order = order;
  update_n_bins();
}

void SpatialLegendreFilter::set_axis(LegendreAxis axis)
{
  if (axes_.empty())
    axes_.push_back({axis, 0, -1.0, 1.0});
  else
    axes_[0].axis = axis;
}

void SpatialLegendreFilter::set_minmax(double min, double max)
{
  if (max <= min)
    throw std::invalid_argument {
      "Maximum value must be greater than minimum value"};
  if (axes_.empty())
    axes_.push_back({LegendreAxis::x, 0, min, max});
  else {
    axes_[0].min = min;
    axes_[0].max = max;
  }
}

void SpatialLegendreFilter::add_axis(
  LegendreAxis axis, int order, double min, double max)
{
  if (order < 0)
    throw std::invalid_argument {"Legendre order must be non-negative."};
  if (max <= min)
    throw std::invalid_argument {
      "Maximum value must be greater than minimum value"};
  for (const auto& ad : axes_)
    if (ad.axis == axis)
      throw std::invalid_argument {fmt::format(
        "Axis '{}' has already been added to this filter.", axis_name(axis))};
  axes_.push_back({axis, order, min, max});
  update_n_bins();
}

void SpatialLegendreFilter::get_all_bins(
  const Particle& p, TallyEstimator /*estimator*/, FilterMatch& match) const
{
  vector<vector<double>> wgts(axes_.size());
  for (std::size_t d = 0; d < axes_.size(); ++d) {
    const auto& ad = axes_[d];
    double coord = axis_coord(p, ad.axis);
    if (coord < ad.min || coord > ad.max)
      return;
    double xi = 2.0 * (coord - ad.min) / (ad.max - ad.min) - 1.0;
    wgts[d].resize(ad.order + 1);
    calc_pn_c(ad.order, xi, wgts[d].data());
  }
  for (int bin = 0; bin < n_bins_; ++bin) {
    auto idx = decode_bin(bin);
    double weight = 1.0;
    for (std::size_t d = 0; d < axes_.size(); ++d)
      weight *= wgts[d][idx[d]];
    match.bins_.push_back(bin);
    match.weights_.push_back(weight);
  }
}

void SpatialLegendreFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  if (axes_.size() == 1) {
    write_dataset(filter_group, "order", axes_[0].order);
    write_dataset(filter_group, "axis", axis_name(axes_[0].axis));
    write_dataset(filter_group, "min", axes_[0].min);
    write_dataset(filter_group, "max", axes_[0].max);
  } else {
    for (const auto& ad : axes_) {
      hid_t g = create_group(filter_group, axis_name(ad.axis));
      write_dataset(g, "order", ad.order);
      write_dataset(g, "min", ad.min);
      write_dataset(g, "max", ad.max);
      close_group(g);
    }
  }
}

std::string SpatialLegendreFilter::text_label(int bin) const
{
  auto idx = decode_bin(bin);
  std::string label = "Legendre expansion";
  for (std::size_t d = 0; d < axes_.size(); ++d)
    label += fmt::format(", {} axis, P{}", axis_name(axes_[d].axis), idx[d]);
  return label;
}

void SpatialLegendreFilter::update_n_bins()
{
  int total = 1;
  for (const auto& ad : axes_)
    total *= ad.order + 1;
  n_bins_ = total;
}

vector<int> SpatialLegendreFilter::decode_bin(int bin) const
{
  vector<int> idx(axes_.size());
  int remainder = bin;
  for (int d = static_cast<int>(axes_.size()) - 1; d >= 0; --d) {
    idx[d] = remainder % (axes_[d].order + 1);
    remainder /= axes_[d].order + 1;
  }
  return idx;
}

// C-API — all original functions unchanged; add_axis variant added at end.

std::pair<int, SpatialLegendreFilter*> check_sptl_legendre_filter(int32_t index)
{
  int err = verify_filter(index);
  if (err)
    return {err, nullptr};
  auto* filt =
    dynamic_cast<SpatialLegendreFilter*>(model::tally_filters[index].get());
  if (!filt) {
    set_errmsg("Not a spatial Legendre filter.");
    err = OPENMC_E_INVALID_TYPE;
  }
  return {err, filt};
}

extern "C" int openmc_spatial_legendre_filter_get_order(
  int32_t index, int* order)
{
  auto [e, f] = check_sptl_legendre_filter(index);
  if (e)
    return e;
  *order = f->order();
  return 0;
}

extern "C" int openmc_spatial_legendre_filter_get_params(
  int32_t index, int* axis, double* min, double* max)
{
  auto [e, f] = check_sptl_legendre_filter(index);
  if (e)
    return e;
  *axis = static_cast<int>(f->axis());
  *min = f->min();
  *max = f->max();
  return 0;
}

extern "C" int openmc_spatial_legendre_filter_set_order(
  int32_t index, int order)
{
  auto [e, f] = check_sptl_legendre_filter(index);
  if (e)
    return e;
  f->set_order(order);
  return 0;
}

extern "C" int openmc_spatial_legendre_filter_set_params(
  int32_t index, const int* axis, const double* min, const double* max)
{
  auto [e, f] = check_sptl_legendre_filter(index);
  if (e)
    return e;
  if (axis)
    f->set_axis(static_cast<LegendreAxis>(*axis));
  if (min && max)
    f->set_minmax(*min, *max);
  return 0;
}

extern "C" int openmc_spatial_legendre_filter_add_axis(
  int32_t index, int axis, int order, double min, double max)
{
  auto [err, filt] = check_sptl_legendre_filter(index);
  if (err)
    return err;
  try {
    filt->add_axis(static_cast<LegendreAxis>(axis), order, min, max);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_INVALID_ARGUMENT;
  }
  return 0;
}

} // namespace openmc