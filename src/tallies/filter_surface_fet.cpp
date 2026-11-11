#include <cstdint>


#include "openmc/tallies/filter_surface_fet.h"

#ifdef OPENMC_DAGMC_ENABLED

#include <fmt/core.h>

#include "openmc/dagmc.h"
#include "openmc/error.h"
#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {

void SurfaceFETFilter::build_surface_elements(const DAGSurface& dag_surface)
{
  moab::DagMC* dagmc_ptr = dag_surface.dagmc_ptr();
  moab::Interface* mbi = dagmc_ptr->moab_instance();

  // Mesh set of this surface and the facets it contains
  const moab::EntityHandle surf = dag_surface.mesh_handle();
  moab::Range facets;
  moab::ErrorCode rval = mbi->get_entities_by_type(surf, moab::MBTRI, facets);
  if (rval != moab::MB_SUCCESS) {
    fatal_error(fmt::format("Could not get the facets of DAGMC surface {}.",
      dag_surface.dag_index()));
  }

  // Append one element per facet. The vertex order of a MOAB triangle follows
  // the DAGMC facet orientation, so the right-hand rule gives the facet normal.
  elements_.reserve(elements_.size() + facets.size());
  for (const moab::EntityHandle facet : facets) {
    const moab::EntityHandle* conn = nullptr;
    int n_nodes = 0;
    rval = mbi->get_connectivity(facet, conn, n_nodes);
    if (rval != moab::MB_SUCCESS || n_nodes != 3)
      fatal_error(
        fmt::format("Facet of DAGMC surface {} is not a valid triangle.",
          dag_surface.dag_index()));

    double coords[9];
    rval = mbi->get_coords(conn, 3, coords);
    if (rval != moab::MB_SUCCESS) {
      fatal_error(fmt::format(
        "Could not get vertex coordinates of a facet of DAGMC surface {}.",
        dag_surface.dag_index()));
    }

    vector<Position> vertices;
    vertices.reserve(3);
    for (int i = 0; i < 3; ++i) {
      vertices.emplace_back(
        coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
    }
    elements_.push_back(fet::make_face_element(vertices));
  }
}

void SurfaceFETFilter::set_surfaces(span<int32_t> surfaces)
{

  SurfaceFilter::set_surfaces(surfaces);

  // now we can preprocess our data
  // now I need to get the dagmc surface index
  for (const auto surface_index : surfaces) {
    // how to get the surface object from the model??

    const auto* dagmc_surface =
      dynamic_cast<const DAGSurface*>(model::surfaces[surface_index].get())
      // maybe something like this?

      if (dagmc_surface)
        fatal_error("Surface {} is not a dagmc surface", surface_index);

    // maybe no the best route
    vector<fet::Orders> orders;
    for (const auto& elem : elements_)
      orders.push_back(elem->orders());

    build_surface_elements(*dagmc_surface);

    // this should be a function
    if (orders.size() == elements_.size()) {
      for (std::size_t e = 0; e < elements_.size(); ++e)
        elements_[e]->set_orders(orders[e]);
    }
    update_bins();
  }
}

void SurfaceFETFilter::update_bins() {}

} // namespace openmc

#endif // OPENMC_DAGMC_ENABLED
