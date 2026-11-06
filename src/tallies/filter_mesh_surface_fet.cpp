#include "openmc/tallies/filter_mesh_surface_fet.h"

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/particle.h"
#include "openmc/tensor.h"
#include "openmc/xml_interface.h"

namespace openmc {

void MeshSurfaceFETFilter::from_xml(pugi::xml_node node)
{
  if (!check_for_node(node, "sideset")) {
    fatal_error(fmt::format(
      "A 'sideset' element is required on mesh surface FET filter {}.", id()));
  }
  // Read before the mesh, whose assignment builds the faces
  sideset_ = std::stoi(get_node_value(node, "sideset"));

  if (check_for_node(node, "direction")) {
    std::string dir = get_node_value(node, "direction", true, true);
    if (dir == "net") {
      direction_ = SurfaceCurrentDirection::NET;
    } else if (dir == "outward") {
      direction_ = SurfaceCurrentDirection::OUTWARD;
    } else if (dir == "inward") {
      direction_ = SurfaceCurrentDirection::INWARD;
    } else {
      fatal_error(fmt::format("Invalid direction '{}' on mesh surface FET "
                              "filter {}; use 'net', 'outward' or 'inward'.",
        dir, id()));
    }
  }

  MeshFETFilter::from_xml(node);
}

void MeshSurfaceFETFilter::set_mesh(int32_t mesh)
{
  MeshFETFilter::set_mesh(mesh);
  mesh_set_ = true;
}

void MeshSurfaceFETFilter::set_sideset(int sideset)
{
  sideset_ = sideset;
  if (mesh_set_)
    set_mesh(mesh_);
}

void MeshSurfaceFETFilter::build_elements(const UnstructuredMesh& umesh)
{
  elements_.clear();
  faces_.clear();
  if (sideset_ < 0)
    return;

  faces_ = umesh.sideset_faces(sideset_);

  elements_.reserve(faces_.size());
  for (const auto& face : faces_)
    elements_.push_back(fet::make_face_element(face.vertices));
  grid_.build(elements_);
}

void MeshSurfaceFETFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // The segment travelled since the previous event, in the mesh frame
  Position r0 = mesh_position(p.r_last_current());
  Position r1 = mesh_position(p.r());

  vector<fet::ElementGrid::Hit> hits;
  grid_.intersect(elements_, r0, r1, hits);

  double psi[fet::MAX_MODES];
  for (const auto& hit : hits) {
    double sign = hit.hit.cosine >= 0.0 ? 1.0 : -1.0;
    double weight;
    switch (direction_) {
    case SurfaceCurrentDirection::NET:
      weight = sign;
      break;
    case SurfaceCurrentDirection::OUTWARD:
      if (sign < 0.0)
        continue;
      weight = 1.0;
      break;
    default:
      if (sign > 0.0)
        continue;
      weight = 1.0;
      break;
    }

    int n = basis_in_mesh_frame(hit.element, hit.hit.point, psi);
    int first = offsets_[hit.element];
    for (int i = 0; i < n; ++i) {
      match.bins_.push_back(first + i);
      match.weights_.push_back(weight * psi[i]);
    }
  }
}

void MeshSurfaceFETFilter::to_statepoint(hid_t filter_group) const
{
  MeshFETFilter::to_statepoint(filter_group);
  write_dataset(filter_group, "sideset", sideset_);

  const char* dir = direction_ == SurfaceCurrentDirection::NET       ? "net"
                    : direction_ == SurfaceCurrentDirection::OUTWARD ? "outward"
                                                                     : "inward";
  write_dataset(filter_group, "direction", dir);

  tensor::Tensor<int> owners({faces_.size(), size_t(2)});
  for (size_t f = 0; f < faces_.size(); ++f) {
    owners(f, 0) = faces_[f].element;
    owners(f, 1) = faces_[f].side;
  }
  write_dataset(filter_group, "face_elements_sides", owners);
}

std::string MeshSurfaceFETFilter::text_label(int bin) const
{
  int f = element_from_bin(bin);
  auto mode = elements_[f]->mode(bin - offsets_[f]);
  return fmt::format(
    "Sideset {} face {}, FET mode ({}, {})", sideset_, f, mode[0], mode[1]);
}

} // namespace openmc
