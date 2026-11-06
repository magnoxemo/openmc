#include "openmc/fet/element_grid.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace openmc {
namespace fet {

namespace {

//! Largest number of grid cells along one axis
constexpr int MAX_CELLS_PER_AXIS {256};

//! Largest total number of grid cells
constexpr int64_t MAX_CELLS {4000000};

//! Crossings closer than this along the segment (cm) are the same crossing
constexpr double SAME_HIT_DISTANCE {1.0e-10};

} // namespace

void ElementGrid::build(const Elements& elements)
{
  n_elements_ = static_cast<int>(elements.size());
  boxes_.clear();
  boxes_.reserve(elements.size());
  bbox_ = BoundingBox::inverted();
  for (const auto& elem : elements) {
    boxes_.push_back(elem->bounding_box());
    bbox_.expand(boxes_.back().min);
    bbox_.expand(boxes_.back().max);
  }

  n_ = {1, 1, 1};
  cell_start_.assign(2, 0);
  cell_elems_.clear();
  if (elements.empty())
    return;

  // Pad the box so that elements lying in a coordinate plane have thickness
  Position extent = bbox_.max - bbox_.min;
  double max_extent = std::max({extent.x, extent.y, extent.z});
  double pad = std::max(1.0e-9 * max_extent, 1.0e-12);
  bbox_.min -= pad;
  bbox_.max += pad;
  extent = bbox_.max - bbox_.min;

  // Choose the cell size so that there is about one element per cell, treating
  // axes that the elements do not span as a single layer
  std::array<bool, 3> flat;
  int n_free = 0;
  double free_volume = 1.0;
  for (int i = 0; i < 3; ++i) {
    flat[i] = extent[i] < 1.0e-6 * max_extent + 4.0 * pad;
    if (!flat[i]) {
      ++n_free;
      free_volume *= extent[i];
    }
  }
  double h =
    n_free > 0
      ? std::pow(free_volume / static_cast<double>(n_elements_), 1.0 / n_free)
      : 1.0;
  int64_t total = 1;
  for (int i = 0; i < 3; ++i) {
    int n = 1;
    if (!flat[i] && h > 0.0) {
      n = static_cast<int>(
        std::clamp(std::ceil(extent[i] / h), 1.0, double(MAX_CELLS_PER_AXIS)));
    }
    n_[i] = n;
    total *= n;
  }
  while (total > MAX_CELLS) {
    int axis = std::max_element(n_.begin(), n_.end()) - n_.begin();
    total /= n_[axis];
    n_[axis] = std::max(1, n_[axis] / 2);
    total *= n_[axis];
  }
  for (int i = 0; i < 3; ++i)
    cell_size_[i] = extent[i] / n_[i];

  // Register every element in the cells its padded bounding box overlaps
  auto cell_range = [&](int i, std::array<int, 3>& lo, std::array<int, 3>& hi) {
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = cell_index(boxes_[i].min[axis] - pad, axis);
      hi[axis] = cell_index(boxes_[i].max[axis] + pad, axis);
    }
  };
  auto cell_of = [&](int x, int y, int z) {
    return (static_cast<std::size_t>(z) * n_[1] + y) * n_[0] + x;
  };

  std::array<int, 3> lo, hi;
  cell_start_.assign(static_cast<std::size_t>(total) + 1, 0);
  for (int i = 0; i < n_elements_; ++i) {
    cell_range(i, lo, hi);
    for (int z = lo[2]; z <= hi[2]; ++z)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int x = lo[0]; x <= hi[0]; ++x)
          ++cell_start_[cell_of(x, y, z) + 1];
  }
  for (std::size_t c = 0; c < static_cast<std::size_t>(total); ++c)
    cell_start_[c + 1] += cell_start_[c];

  cell_elems_.resize(cell_start_.back());
  vector<int> fill(cell_start_.begin(), cell_start_.end() - 1);
  for (int i = 0; i < n_elements_; ++i) {
    cell_range(i, lo, hi);
    for (int z = lo[2]; z <= hi[2]; ++z)
      for (int y = lo[1]; y <= hi[1]; ++y)
        for (int x = lo[0]; x <= hi[0]; ++x)
          cell_elems_[fill[cell_of(x, y, z)]++] = i;
  }
}

int ElementGrid::cell_index(double x, int axis) const
{
  int i =
    static_cast<int>(std::floor((x - bbox_.min[axis]) / cell_size_[axis]));
  return std::clamp(i, 0, n_[axis] - 1);
}

void ElementGrid::intersect(
  const Elements& elements, Position p0, Position p1, vector<Hit>& hits) const
{
  if (n_elements_ == 0)
    return;
  const Position d = p1 - p0;
  const double d_norm = d.norm();
  if (d_norm == 0.0)
    return;

  // Clip the segment to the bounding box of all elements
  double t0 = 0.0;
  double t1 = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (d[axis] == 0.0) {
      if (p0[axis] < bbox_.min[axis] || p0[axis] > bbox_.max[axis])
        return;
      continue;
    }
    double ta = (bbox_.min[axis] - p0[axis]) / d[axis];
    double tb = (bbox_.max[axis] - p0[axis]) / d[axis];
    if (ta > tb)
      std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    if (t0 > t1)
      return;
  }

  // Walk the grid cells along the clipped segment, gathering candidates
  const Position start = p0 + t0 * d;
  std::array<int, 3> ijk;
  std::array<int, 3> step;
  std::array<double, 3> t_next;
  std::array<double, 3> t_delta;
  constexpr double inf = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    ijk[axis] = cell_index(start[axis], axis);
    if (d[axis] > 0.0) {
      step[axis] = 1;
      t_next[axis] =
        (bbox_.min[axis] + (ijk[axis] + 1) * cell_size_[axis] - p0[axis]) /
        d[axis];
      t_delta[axis] = cell_size_[axis] / d[axis];
    } else if (d[axis] < 0.0) {
      step[axis] = -1;
      t_next[axis] =
        (bbox_.min[axis] + ijk[axis] * cell_size_[axis] - p0[axis]) / d[axis];
      t_delta[axis] = -cell_size_[axis] / d[axis];
    } else {
      step[axis] = 0;
      t_next[axis] = inf;
      t_delta[axis] = inf;
    }
  }

  vector<int> candidates;
  const int max_steps = n_[0] + n_[1] + n_[2] + 3;
  for (int s = 0; s < max_steps; ++s) {
    std::size_t cell =
      (static_cast<std::size_t>(ijk[2]) * n_[1] + ijk[1]) * n_[0] + ijk[0];
    candidates.insert(candidates.end(), cell_elems_.begin() + cell_start_[cell],
      cell_elems_.begin() + cell_start_[cell + 1]);

    int axis = 0;
    if (t_next[1] < t_next[axis])
      axis = 1;
    if (t_next[2] < t_next[axis])
      axis = 2;
    if (t_next[axis] > t1)
      break;
    ijk[axis] += step[axis];
    if (ijk[axis] < 0 || ijk[axis] >= n_[axis])
      break;
    t_next[axis] += t_delta[axis];
  }

  std::sort(candidates.begin(), candidates.end());
  candidates.erase(
    std::unique(candidates.begin(), candidates.end()), candidates.end());

  std::size_t first = hits.size();
  Hit hit;
  for (int i : candidates) {
    if (elements[i]->intersect(p0, d, hit.hit)) {
      hit.element = i;
      hits.push_back(hit);
    }
  }

  // Order by t and merge crossings of the same point (shared edges/vertices),
  // keeping the lowest element index
  std::sort(hits.begin() + first, hits.end(), [](const Hit& a, const Hit& b) {
    return a.hit.t < b.hit.t || (a.hit.t == b.hit.t && a.element < b.element);
  });
  std::size_t out = first;
  for (std::size_t i = first; i < hits.size();) {
    std::size_t best = i;
    std::size_t j = i + 1;
    while (j < hits.size() &&
           (hits[j].hit.t - hits[j - 1].hit.t) * d_norm < SAME_HIT_DISTANCE) {
      if (hits[j].element < hits[best].element)
        best = j;
      ++j;
    }
    hits[out++] = hits[best];
    i = j;
  }
  hits.resize(out);
}

} // namespace fet
} // namespace openmc
