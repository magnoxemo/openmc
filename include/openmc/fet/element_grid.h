#ifndef OPENMC_FET_ELEMENT_GRID_H
#define OPENMC_FET_ELEMENT_GRID_H

#include <array>

#include "openmc/bounding_box.h"
#include "openmc/fet/fet_element.h"
#include "openmc/memory.h"
#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {
namespace fet {

//==============================================================================
//! Uniform grid over a set of 2D FET elements that finds where a line segment
//! crosses them.
//!
//! The elements are registered in every grid cell their (padded) bounding box
//! overlaps. A segment is clipped to the bounding box of the whole set, which
//! rejects most segments immediately when the set is small compared to the
//! domain, and the remaining ones are walked through the grid cell by cell and
//! tested with FETElement::intersect. The grid stores only boxes and indices;
//! the elements are passed to each call and must be the ones it was built from.
//==============================================================================

class ElementGrid {
public:
  using Elements = vector<unique_ptr<FETElement>>;

  //! One crossing of a segment with an element
  struct Hit {
    int element;    //!< Index of the element
    SegmentHit hit; //!< Where and how the element was crossed
  };

  //! Build the grid; element i keeps index i
  void build(const Elements& elements);

  //! Append all crossings of the segment p0 -> p1, in order of increasing t
  //
  //! A crossing through the shared edge or vertex of several elements is
  //! reported once, for the lowest element index. See FETElement::intersect
  //! for the treatment of segment endpoints.
  //
  //! \param[in] elements The elements the grid was built from
  //! \param[in] p0 Start of the segment
  //! \param[in] p1 End of the segment
  //! \param[in,out] hits Hits, appended to
  void intersect(const Elements& elements, Position p0, Position p1,
    vector<Hit>& hits) const;

  //! Bounding box of all elements
  const BoundingBox& bounding_box() const { return bbox_; }

private:
  //! Cell index along one axis of a coordinate, clamped to the grid
  int cell_index(double x, int axis) const;

  int n_elements_ {0};
  BoundingBox bbox_;
  vector<BoundingBox> boxes_;      //!< Bounding box of each element
  std::array<int, 3> n_ {1, 1, 1}; //!< Cells per axis
  Position cell_size_;
  vector<int> cell_start_; //!< CSR offsets into cell_elems_, size cells + 1
  vector<int> cell_elems_; //!< Element indices of each cell
};

} // namespace fet
} // namespace openmc

#endif // OPENMC_FET_ELEMENT_GRID_H
