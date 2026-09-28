#include "openmc/tallies/filter_mesh_fet.h"

#include <algorithm>
#include <limits>

#include <fmt/core.h>

#include "openmc/capi.h"
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

  // Build the affine map of each element. Only linear tetrahedra are
  // supported, which makes every map affine and the basis exactly orthogonal.
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
    std::array<Position, 4> vertices;
    for (int i = 0; i < 4; ++i)
      vertices[i] = umesh->vertex(conn[i]);
    maps_.emplace_back(vertices);
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
  maps_[element].barycentric(r, lambda);
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
  double f = 0.0;
  for (int i = 0; i < n; ++i)
    f += values[i] * psi[i];
  return f / maps_[element].volume();
}

void MeshFETFilter::to_statepoint(hid_t filter_group) const
{
  MeshFilter::to_statepoint(filter_group);
  write_dataset(filter_group, "basis", "dubiner");
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

//==============================================================================
// C-API functions
//==============================================================================

//! Look up a mesh FET filter by index, setting an error message on failure
static int get_mesh_fet_filter(int32_t index, MeshFETFilter*& filt)
{
  if (int err = verify_filter(index))
    return err;
  filt = dynamic_cast<MeshFETFilter*>(model::tally_filters[index].get());
  if (!filt) {
    set_errmsg("Tried to access orders of a non-mesh-FET filter.");
    return OPENMC_E_INVALID_TYPE;
  }
  return 0;
}

extern "C" int openmc_meshfet_filter_get_orders(
  int32_t index, const int** orders, size_t* n)
{
  MeshFETFilter* filt;
  if (int err = get_mesh_fet_filter(index, filt))
    return err;
  *orders = filt->orders().data();
  *n = filt->orders().size();
  return 0;
}

extern "C" int openmc_meshfet_filter_evaluate(
  int32_t index, int32_t element, const double xyz[3], double* psi, int* n)
{
  MeshFETFilter* filt;
  if (int err = get_mesh_fet_filter(index, filt))
    return err;
  if (element < 0 || element >= filt->n_elements()) {
    set_errmsg("Element index is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *n = filt->evaluate_basis(element, {xyz[0], xyz[1], xyz[2]}, psi);
  return 0;
}

extern "C" int openmc_meshfet_filter_set_orders(
  int32_t index, const int* orders, size_t n)
{
  MeshFETFilter* filt;
  if (int err = get_mesh_fet_filter(index, filt))
    return err;
  if (n != static_cast<size_t>(filt->n_elements())) {
    set_errmsg("Number of orders does not match the number of mesh elements.");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  filt->set_orders(vector<int>(orders, orders + n));
  return 0;
}

} // namespace openmc
