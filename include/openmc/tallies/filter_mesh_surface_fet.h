#ifndef OPENMC_TALLIES_FILTER_MESH_SURFACE_FET_H
#define OPENMC_TALLIES_FILTER_MESH_SURFACE_FET_H

#include "openmc/fet/element_grid.h"
#include "openmc/mesh.h"
#include "openmc/tallies/filter_mesh_fet.h"

namespace openmc {

//==============================================================================
//! Which crossings of a sideset face a MeshSurfaceFETFilter counts
//==============================================================================

enum class SurfaceCurrentDirection {
  NET,     //!< Crossings along the face normal count +1, against it -1
  OUTWARD, //!< Only crossings along the face normal, counted +1
  INWARD   //!< Only crossings against the face normal, counted +1
};

//==============================================================================
//! Functional expansion tally (FET) of the current through a sideset of an
//! unstructured mesh.
//!
//! The faces of the sideset are the elements of the filter; each is a 2D
//! fet::FETElement (see fet::make_face_element) with its own expansion orders,
//! and the bins, weights and reconstruction work as in MeshFETFilter with faces
//! in place of volume elements. The normal of a face points out of the mesh
//! element that owns it.
//!
//! The filter is scored at the same events as a mesh surface filter and sees
//! the whole flight segment since the previous event. It finds the crossings of
//! the segment with the faces of the sideset itself, so the sideset need not
//! coincide with any surface of the geometry. A crossing contributes its
//! direction weight (see SurfaceCurrentDirection) times the unnormalized mode
//! value at the crossing point. Tallies using this filter must score only
//! 'current'.
//==============================================================================

class MeshSurfaceFETFilter : public MeshFETFilter {
public:
  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "meshsurfacefet"; }
  FilterType type() const override { return FilterType::MESH_SURFACE_FET; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  void set_mesh(int32_t mesh) override;

  //! Set the sideset; the faces are rebuilt, and the orders of the faces are
  //! kept if the number of faces does not change
  void set_sideset(int sideset);

  int sideset() const { return sideset_; }

  void set_direction(SurfaceCurrentDirection direction)
  {
    direction_ = direction;
  }

  SurfaceCurrentDirection direction() const { return direction_; }

  //! Mesh description of a face (owning element and side)
  const UnstructuredMesh::SidesetFace& face(int face) const
  {
    return faces_[face];
  }

protected:
  void build_elements(const UnstructuredMesh& mesh) override;

private:
  int sideset_ {-1};
  SurfaceCurrentDirection direction_ {SurfaceCurrentDirection::NET};
  bool mesh_set_ {false};
  vector<UnstructuredMesh::SidesetFace> faces_;
  fet::ElementGrid grid_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_MESH_SURFACE_FET_H
