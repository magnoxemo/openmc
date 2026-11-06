#include "openmc/tallies/filter_mesh_fet.h"

#include <algorithm>
#include <limits>

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/mesh.h"
#include "openmc/tensor.h"
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

  // Keep existing orders if they still match the mesh
  vector<fet::Orders> orders;
  for (const auto& elem : elements_)
    orders.push_back(elem->orders());

  build_elements(*umesh);

  if (orders.size() == elements_.size()) {
    for (std::size_t e = 0; e < elements_.size(); ++e)
      elements_[e]->set_orders(orders[e]);
  }
  update_bins();
}

void MeshFETFilter::build_elements(const UnstructuredMesh& umesh)
{
  int n_elem = umesh.n_bins();
  elements_.clear();
  elements_.reserve(n_elem);
  for (int e = 0; e < n_elem; ++e) {
    auto conn = umesh.connectivity(e);
    if (conn.size() != 4) {
      fatal_error(fmt::format("Element {} of mesh {} has {} vertices; mesh FET "
                              "filters only support linear tetrahedra.",
        e, umesh.id_, conn.size()));
    }
    vector<Position> vertices;
    for (int i = 0; i < 4; ++i)
      vertices.push_back(umesh.vertex(conn[i]));

    elements_.push_back(make_unique<fet::TetElement>(vertices));
  }
}

void MeshFETFilter::set_order(int order)
{
  set_orders(vector<int>(elements_.size(), order));
}

void MeshFETFilter::set_orders(const vector<int>& orders)
{
  if (orders.size() != elements_.size()) {
    fatal_error(fmt::format("Mesh FET filter {} received {} element orders "
                            "but its mesh has {} elements.",
      id(), orders.size(), elements_.size()));
  }
  for (int order : orders) {
    if (order > fet::MAX_ORDER) {
      fatal_error(fmt::format("Mesh FET filter {} requested order {}; the "
                              "maximum supported order is {}.",
        id(), order, fet::MAX_ORDER));
    }
  }
  for (std::size_t e = 0; e < elements_.size(); ++e)
    elements_[e]->set_orders({orders[e], orders[e], orders[e]});
  update_bins();
}

void MeshFETFilter::update_bins()
{
  offsets_.resize(elements_.size() + 1);
  offsets_[0] = 0;
  max_order_ = -1;
  int64_t total = 0;
  for (std::size_t e = 0; e < elements_.size(); ++e) {
    total += elements_[e]->n_modes();
    if (total > std::numeric_limits<int>::max()) {
      fatal_error(fmt::format(
        "Mesh FET filter {} has too many bins for a single tally.", id()));
    }
    offsets_[e + 1] = static_cast<int>(total);
    max_order_ = std::max(max_order_, elements_[e]->max_order());
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
  if (e < 0)
    return;

  double psi[fet::MAX_MODES];
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
  return elements_[element]->eval(r, psi);
}

int MeshFETFilter::evaluate_basis(int element, Position r, double* psi) const
{
  return basis_in_mesh_frame(element, mesh_position(r), psi);
}

double MeshFETFilter::reconstruct(
  int element, Position r, const double* values) const
{
  double psi[fet::MAX_MODES];
  int n = evaluate_basis(element, r, psi);
  double f = 0.0;
  for (int i = 0; i < n; ++i)
    f += elements_[element]->mode_norm_sq(i) * values[i] * psi[i];
  return f / elements_[element]->volume();
}

void MeshFETFilter::to_statepoint(hid_t filter_group) const
{
  MeshFilter::to_statepoint(filter_group);
  write_dataset(filter_group, "basis", "legendre_jacobi_duffy");
  write_dataset(filter_group, "normalized", 0);

  // First filter bin of each element (size n_elements + 1). Excluded elements
  // (negative order) own no bins.
  write_dataset(filter_group, "offsets", offsets_);

  if (elements_.empty())
    return;
  const fet::FETElement& shape = *elements_[0];

  // Exact vertices of each element, in the mesh frame (before the filter
  // translation/rotation is applied) and in cm. The last vertex is the one
  // collapsed by the Duffy transform.
  size_t n_elem = elements_.size();
  size_t n_vert = shape.n_vertices();
  tensor::Tensor<double> vertices({n_elem, n_vert, size_t(3)});
  size_t dim = shape.dim();
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

int MeshFETFilter::element_from_bin(int bin) const
{
  auto it = std::upper_bound(offsets_.begin(), offsets_.end(), bin);
  return static_cast<int>(it - offsets_.begin()) - 1;
}

std::string MeshFETFilter::text_label(int bin) const
{
  int e = element_from_bin(bin);
  const auto& elem = *elements_[e];
  auto mode = elem.mode(bin - offsets_[e]);
  if (elem.dim() == 2) {
    return fmt::format(
      "{}, FET mode ({}, {})", MeshFilter::text_label(e), mode[0], mode[1]);
  }
  return fmt::format("{}, FET mode ({}, {}, {})", MeshFilter::text_label(e),
    mode[0], mode[1], mode[2]);
}

} // namespace openmc
