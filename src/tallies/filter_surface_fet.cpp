#include "openmc/tallies/filter_surface_fet.h"

#include <algorithm>
#include <limits>

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/position.h"
#include "openmc/surface.h"
#include "openmc/tensor.h"
#include "openmc/xml_interface.h"

#ifdef OPENMC_XDG_ENABLED
#include "openmc/xdg.h"
#endif

namespace openmc {

void SurfaceFETFilter::from_xml(pugi::xml_node node)
{
  auto surfaces = get_node_array<int32_t>(node, "bins");

  // Convert surface IDs to indices of the global surfaces vector.
  for (auto& s : surfaces) {
    auto search = model::surface_map.find(s);
    if (search == model::surface_map.end()) {
      fatal_error(fmt::format(
        "Could not find surface {} specified on tally filter.", s));
    }
    s = search->second;
  }

  if (!check_for_node(node, "order")) {
    fatal_error(fmt::format(
      "An 'order' element is required on surface FET filter {}.", id()));
  }

  // The order must be known before the facets are built
  order_ = std::stoi(get_node_value(node, "order"));
  this->set_surfaces(surfaces);
}

void SurfaceFETFilter::set_surfaces(span<int32_t> surfaces)
{
  surfaces_.clear();
  slot_map_.clear();
  surface_first_.clear();
  facet_map_.clear();
  elements_.clear();

  surfaces_.reserve(surfaces.size());
  for (const auto index : surfaces) {
    if (index < 0 || index >= static_cast<int32_t>(model::surfaces.size())) {
      fatal_error(fmt::format(
        "Surface index {} on surface FET filter {} is out of range.", index,
        id()));
    }
    if (slot_map_.count(index)) {
      fatal_error(fmt::format("Surface {} appears more than once on surface "
                              "FET filter {}.",
        model::surfaces[index]->id_, id()));
    }
    slot_map_[index] = static_cast<int>(surfaces_.size());
    surfaces_.push_back(index);

    surface_first_.push_back(static_cast<int>(elements_.size()));
    build_facets(index);
  }
  surface_first_.push_back(static_cast<int>(elements_.size()));

  update_bins();
}

void SurfaceFETFilter::build_facets(int32_t surface_index)
{
#ifdef OPENMC_XDG_ENABLED
  const auto* surface =
    dynamic_cast<const XDGSurface*>(model::surfaces[surface_index].get());
  if (!surface) {
    fatal_error(fmt::format("Surface {} on surface FET filter {} is not an XDG "
                            "surface mesh surface.",
      model::surfaces[surface_index]->id_, id()));
  }

  const auto& mesh_manager = *surface->xdg_ptr()->mesh_manager();
  const auto faces = mesh_manager.get_surface_faces(surface->xdg_id());
  elements_.reserve(elements_.size() + faces.size());

  for (const auto face : faces) {
    const auto coords = mesh_manager.face_vertex_coordinates(face);
    if (coords.size() != 3) {
      fatal_error(fmt::format(
        "Surface {} on surface FET filter {} has a facet with {} vertices; "
        "only triangular facets are supported.",
        model::surfaces[surface_index]->id_, id(), coords.size()));
    }

    vector<Position> vertices;
    vertices.reserve(coords.size());
    for (const auto& v : coords)
      vertices.emplace_back(v.x, v.y, v.z);

    facet_map_[face] = static_cast<int>(elements_.size());
    auto element = fet::make_face_element(vertices);
    for (int axis = 0; axis < element->dim(); ++axis)
      element->set_order(axis, order_);
    elements_.push_back(std::move(element));
  }
#else
  fatal_error("Surface FET filters require OpenMC to be built with XDG "
              "(OPENMC_USE_XDG=ON).");
#endif
}

void SurfaceFETFilter::set_order(int order)
{
  order_ = order;
  for (auto& element : elements_) {
    for (int axis = 0; axis < element->dim(); ++axis)
      element->set_order(axis, order);
  }
  update_bins();
}

void SurfaceFETFilter::update_bins()
{
  offsets_.resize(elements_.size() + 1);
  offsets_[0] = 0;
  max_order_ = -1;
  int64_t total = 0;
  for (std::size_t e = 0; e < elements_.size(); ++e) {
    total += elements_[e]->n_modes();
    if (total > std::numeric_limits<int>::max()) {
      fatal_error(fmt::format(
        "Surface FET filter {} has too many bins for a single tally.", id()));
    }
    offsets_[e + 1] = static_cast<int>(total);
    max_order_ = std::max(max_order_, elements_[e]->max_order());
  }
  n_bins_ = offsets_.back();
}

void SurfaceFETFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  auto search = slot_map_.find(p.surface_index());
  if (search == slot_map_.end())
    return;

  const int slot = search->second;

  // The facet crossed by the particle is the last one hit by the XDG ray that
  // found the boundary
  int e = -1;
#ifdef OPENMC_XDG_ENABLED
  const auto& history = p.xdg_prev_elements();
  if (!history.empty()) {
    auto it = facet_map_.find(history.back());
    if (it != facet_map_.end() && it->second >= surface_first_[slot] &&
        it->second < surface_first_[slot + 1])
      e = it->second;
  }
#endif

  if (e < 0) {
    fatal_error(fmt::format("Surface FET filter {} could not find the facet "
                            "crossed at surface {}; the XDG hit history does "
                            "not hold a facet of that surface.",
      id(), model::surfaces[surfaces_[slot]]->id_));
  }
  if (elements_[e]->n_modes() == 0)
    return;

  double psi[fet::MAX_MODES];
  int n = elements_[e]->eval(p.r(), psi);
  int first = offsets_[e];
  for (int i = 0; i < n; ++i) {
    match.bins_.push_back(first + i);
    match.weights_.push_back(psi[i]);
  }
}

int SurfaceFETFilter::element_from_bin(int bin) const
{
  auto it = std::upper_bound(offsets_.begin(), offsets_.end(), bin);
  return static_cast<int>(it - offsets_.begin()) - 1;
}

int SurfaceFETFilter::surface_slot_from_element(int facet) const
{
  auto it =
    std::upper_bound(surface_first_.begin(), surface_first_.end(), facet);
  return static_cast<int>(it - surface_first_.begin()) - 1;
}

void SurfaceFETFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);

  vector<int32_t> surface_ids;
  for (const auto s : surfaces_)
    surface_ids.push_back(model::surfaces[s]->id_);
  write_dataset(filter_group, "bins", surface_ids);

  write_dataset(filter_group, "basis", "legendre_jacobi_duffy");
  write_dataset(filter_group, "normalized", 0);

  // First facet of each surface (size n_surfaces + 1) and first filter bin of
  // each facet (size n_facets + 1). Excluded facets own no bins.
  write_dataset(filter_group, "surface_first", surface_first_);
  write_dataset(filter_group, "offsets", offsets_);

  if (elements_.empty())
    return;
  const fet::FETElement& shape = *elements_[0];

  // Exact vertices of each facet in cm. The last vertex is the one collapsed
  // by the Duffy transform.
  size_t n_elem = elements_.size();
  size_t n_vert = shape.n_vertices();
  size_t dim = shape.dim();
  tensor::Tensor<double> vertices({n_elem, n_vert, size_t(3)});
  tensor::Tensor<double> volume({n_elem});
  tensor::Tensor<int> orders({n_elem, dim});
  for (size_t e = 0; e < n_elem; ++e) {
    for (size_t d = 0; d < dim; ++d)
      orders(e, d) = elements_[e]->order(d);
    for (size_t i = 0; i < n_vert; ++i) {
      const Position& v = elements_[e]->vertex(i);
      vertices(e, i, 0) = v.x;
      vertices(e, i, 1) = v.y;
      vertices(e, i, 2) = v.z;
    }
    volume(e) = elements_[e]->volume();
  }
  write_dataset(filter_group, "vertices_ordered", vertices);
  write_dataset(filter_group, "volume", volume);
  write_dataset(filter_group, "orders", orders);
  write_attribute(filter_group, "fet_version", 2);
  write_attribute(filter_group, "element_type", shape.type());
}

std::string SurfaceFETFilter::text_label(int bin) const
{
  int e = element_from_bin(bin);
  int slot = surface_slot_from_element(e);
  const auto& element = *elements_[e];
  auto mode = element.mode(bin - offsets_[e]);
  return fmt::format("Surface {}, facet {}, FET mode ({}, {})",
    model::surfaces[surfaces_[slot]]->id_, e - surface_first_[slot], mode[0],
    mode[1]);
}

} // namespace openmc
