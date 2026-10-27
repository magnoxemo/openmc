#include "openmc/tallies/filter_mesh_fet.h"

#include <algorithm>
#include <limits>

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/fet/dubiner.h"
#include "openmc/mesh.h"
#include "openmc/xml_interface.h"

namespace openmc {

void MeshFETFilter::from_xml(pugi::xml_node node)
{
  MeshFilter::from_xml(node);

  if (check_for_node(node, "orders")) {
    set_orders(get_node_array<int>(node, "orders"));
  } else if (check_for_node(node, "order")) {
    set_order(std::stoi(get_node_value(node, "order")));
  } else {
    fatal_error(fmt::format(
      "An 'order' or 'orders' element is required on mesh FET filter {}.",
      id()));
  }
}

void MeshFETFilter::set_mesh(int32_t mesh)
{
  MeshFilter::set_mesh(mesh);

  const auto* umesh =
    dynamic_cast<const UnstructuredMesh*>(model::meshes[mesh_].get());
  if (!umesh) {
    fatal_error(fmt::format("Mesh {} on mesh FET filter {} is not an "
                            "unstructured mesh.",
      model::meshes[mesh_]->id_, id()));
  }

  // Build the mapping of each element. Only linear tetrahedra are supported for
  // now.
  int n_elem = umesh->n_bins();
  maps_.clear();
  maps_.reserve(n_elem);
  for (int e = 0; e < n_elem; ++e) {
    auto conn = umesh->connectivity(e);
    if (conn.size() != 4) {
      fatal_error(fmt::format("Element {} of mesh {} has {} vertices; mesh FET "
                              "filters only support linear tetrahedra.",
        e, umesh->id_, conn.size()));
    }
    std::vector<Position> vertices;
    vertices.reserve(4); // Maybe I need to get waya from that at some point.
    for (int i = 0; i < 4; ++i)
      vertices.push_back(umesh->vertex(conn[i]));
    // TODO:
    // I will come back and implement this for other elemental mapping as well.
    maps_.push_back(make_unique<fet::TetElementMapping>(vertices));
  }

  // Keep existing orders if they still match the mesh
  if (orders_.size() != maps_.size())
    orders_.assign(maps_.size(), 0);
  update_bins();
}

void MeshFETFilter::set_order(int order)
{
  set_orders(vector<int>(maps_.size(), order));
}

void MeshFETFilter::set_orders(const vector<int>& orders)
{
  if (orders.size() != maps_.size()) {
    fatal_error(fmt::format("Mesh FET filter {} received {} element orders "
                            "but its mesh has {} elements.",
      id(), orders.size(), maps_.size()));
  }
  for (int order : orders) {
    if (order > fet::MAX_ORDER) {
      fatal_error(fmt::format("Mesh FET filter {} requested order {}; the "
                              "maximum supported order is {}.",
        id(), order, fet::MAX_ORDER));
    }
  }
  orders_ = orders;
  update_bins();
}

void MeshFETFilter::update_bins()
{
  offsets_.resize(orders_.size() + 1);
  offsets_[0] = 0;
  max_order_ = -1;
  int64_t total = 0;
  for (std::size_t e = 0; e < orders_.size(); ++e) {
    total += fet::n_modes_tet(orders_[e]);
    if (total > std::numeric_limits<int>::max()) {
      fatal_error(fmt::format(
        "Mesh FET filter {} has too many bins for a single tally.", id()));
    }
    offsets_[e + 1] = static_cast<int>(total);
    max_order_ = std::max(max_order_, orders_[e]);
  }
  n_bins_ = offsets_.back();
}

void MeshFETFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  if (estimator == TallyEstimator::TRACKLENGTH) {
    fatal_error("Mesh FET filters do not support the track-length estimator; "
                "use a collision or analog estimator.");
  }

  Position r = mesh_position(p.r());
  int e = model::meshes[mesh_]->get_bin(r);
  if (e < 0 || orders_[e] < 0)
    return;

  double psi[fet::MAX_MODES_TET];
  int n = basis_in_mesh_frame(e, r, psi);
  int first = offsets_[e];
  for (int i = 0; i < n; ++i) {
    match.bins_.push_back(first + i);
    match.weights_.push_back(psi[i]);
  }
}

Position MeshFETFilter::mesh_position(Position r) const
{
  if (translated_)
    r -= translation();
  if (!rotation_.empty())
    r = r.rotate(rotation_);
  return r;
}

int MeshFETFilter::basis_in_mesh_frame(
  int element, Position r, double* psi) const
{
  int order = orders_[element];
  if (order < 0)
    return 0;
  double lambda[4];
  maps_[element]->parametric_coordinates(r, lambda);
  fet::eval_tet(order, lambda, psi);
  return fet::n_modes_tet(order);
}

int MeshFETFilter::evaluate_basis(int element, Position r, double* psi) const
{
  return basis_in_mesh_frame(element, mesh_position(r), psi);
}

double MeshFETFilter::reconstruct(
  int element, Position r, const double* values) const
{
  double psi[fet::MAX_MODES_TET];
  int n = evaluate_basis(element, r, psi);
  // The tallied modes are unnormalized. With squared normalization N_i^2, the
  // coefficient is a_i = N_i * tally_i / V and the mode is N_i * psi_i.
  double f = 0.0;
  for (int i = 0; i < n; ++i) {
    auto m = fet::tet_mode(i);
    f += fet::tet_mode_norm_sq(m[0], m[1], m[2]) * values[i] * psi[i];
  }
  return f / maps_[element]->volume();
}

void MeshFETFilter::to_statepoint(hid_t filter_group) const
{
  MeshFilter::to_statepoint(filter_group);
  write_dataset(filter_group, "basis", "legendre_jacobi");
  write_dataset(filter_group, "normalized", 0);
  write_dataset(filter_group, "orders", orders_);
}

int MeshFETFilter::element_from_bin(int bin) const
{
  auto it = std::upper_bound(offsets_.begin(), offsets_.end(), bin);
  return static_cast<int>(it - offsets_.begin()) - 1;
}

std::string MeshFETFilter::text_label(int bin) const
{
  int e = element_from_bin(bin);
  auto mode = fet::tet_mode(bin - offsets_[e]);
  return fmt::format("{}, FET mode ({}, {}, {})", MeshFilter::text_label(e),
    mode[0], mode[1], mode[2]);
}

} // namespace openmc
