//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-758885
//
// All rights reserved.
//
// This file is part of Comb.
//
// For details, see https://github.com/LLNL/Comb
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#ifndef _BOX3D_HPP
#define _BOX3D_HPP

#include "config.hpp"

#include "memory.hpp"
#include "utils.hpp"
#include "MeshInfo.hpp"

struct IdxTemplate
{
  enum struct location: IdxT
  { min =-1
  , mid = 0
  , max = 1
  };
  location idx;
  IdxT offset;
  constexpr IdxTemplate(location idx_, IdxT offset_) : idx(idx_), offset(offset_) {}
};

struct Box3d
{
  static Box3d make_ghost_box(MeshInfo const& info)
  {
    IdxT local_ghost_min[3] { 0
                            , 0
                            , 0 };
    IdxT local_ghost_max[3] { info.len[0]
                            , info.len[1]
                            , info.len[2] };
    return Box3d{info, local_ghost_min, local_ghost_max};
  }

  static Box3d make_owned_box(MeshInfo const& info)
  {
    IdxT local_own_min[3] { info.global_own_min[0] - info.global_offset[0]
                          , info.global_own_min[1] - info.global_offset[1]
                          , info.global_own_min[2] - info.global_offset[2] };
    IdxT local_own_max[3] { info.global_own_max[0] - info.global_offset[0]
                          , info.global_own_max[1] - info.global_offset[1]
                          , info.global_own_max[2] - info.global_offset[2] };
    return Box3d{info, local_own_min, local_own_max};
  }

  MeshInfo info;
  IdxT min[3];
  IdxT sizes[3];
  Box3d(MeshInfo const& info_, const IdxT local_min[], const IdxT local_max[])
    : info(info_)
    , min{ local_min[0]
         , local_min[1]
         , local_min[2] }
    , sizes{ local_max[0] - local_min[0]
           , local_max[1] - local_min[1]
           , local_max[2] - local_min[2] }
  {
    //FPRINTF(stdout, "Box3d i %d %d j %d %d k %d %d\n", min[0], max[0], min[1], max[1], min[2], max[2]);
    //assert((imax-imin)*(jmax-jmin)*(kmax-kmin) <= 13*3*3);
  }

  void print(const char* name) const
  {
    FPRINTF(stdout, "Box3d %32s local (%i %i %i)-(%i %i %i) info (%i %i %i)-(%i %i %i) global (%i %i %i)-(%i %i %i)\n",
                     name,
                     min[0], min[1], min[2], min[0]+sizes[0], min[1]+sizes[1], min[2]+sizes[2],
                     info.min[0], info.min[1], info.min[2], info.max[0], info.max[1], info.max[2],
                     info.global_min[0], info.global_min[1], info.global_min[2], info.global_max[0], info.global_max[1], info.global_max[2] );
  }

  void correct_periodicity()
  {
    // nothing to do here, only have local information
    // correct info periodicity
    info.correct_periodicity();
  }

  // get the intersection between two boxes from the same GlobalMeshInfo
  // returns a box with indices in the local index space of this->info
  Box3d intersect(Box3d const& other) const
  {
    assert(info.global == other.info.global);
    // find min, max in global coords
    IdxT omin[3] { std::max(min[0] + info.global_offset[0], other.min[0] + other.info.global_offset[0])
                 , std::max(min[1] + info.global_offset[1], other.min[1] + other.info.global_offset[1])
                 , std::max(min[2] + info.global_offset[2], other.min[2] + other.info.global_offset[2]) };
    IdxT omax[3] { std::min(min[0] + sizes[0] + info.global_offset[0], other.min[0] + other.sizes[0] + other.info.global_offset[0])
                 , std::min(min[1] + sizes[1] + info.global_offset[1], other.min[1] + other.sizes[1] + other.info.global_offset[1])
                 , std::min(min[2] + sizes[2] + info.global_offset[2], other.min[2] + other.sizes[2] + other.info.global_offset[2]) };

    for (IdxT dim = 0; dim < 3; ++dim) {
      /*
      // make work with periodicity, may not be necessary
      if (info.global.periodic[dim]) {
        // correct for periodicity
        // move other.min[dim] so in the range [ min[dim], min[dim]+info.global.sizes[dim] )
        IdxT this_min = min[dim] + info.global_offset[dim];
        IdxT other_min = other.min[dim] + other.info.global_offset[dim];
        IdxT diff_min = other_min - this_min;

        IdxT quot = diff_min / info.global.sizes[dim];
        other_min -= (quot+(diff_min < 0 ? 1 : 0)) * info.global.sizes[dim];

        IdxT this_remaining_size = sizes[dim] - (other_min - this_min);

        omin[dim] = other_min;
        omax[dim] = other_min + std::min(this_remaining_size, other.sizes[dim]);
      }


      assert(0 <= omin[dim] - min[dim] && omin[dim] - min[dim] < info.global.sizes[dim]);
      assert(omax[dim] <= min[0] + sizes[0] + info.global_offset[0]); // this happens when omax would wrap around to the beginning of this
      */

      // switch back to local coords
      omin[dim] -= info.global_offset[dim];
      omax[dim] -= info.global_offset[dim];

      // correct max that are less than min to min (size 0 no overlap)
      if (omax[dim] < omin[dim]) omax[dim] = omin[dim];
    }

    return Box3d{ info, omin, omax };
  }

  size_t size() const
  {
    return sizes[0] * sizes[1] * sizes[2] ;
  }

  bool operator==(Box3d const& other) const
  {
    return info == other.info &&
           min[0] == other.min[0] &&
           min[1] == other.min[1] &&
           min[2] == other.min[2] &&
           sizes[0] == other.sizes[0] &&
           sizes[1] == other.sizes[1] &&
           sizes[2] == other.sizes[2] ;
  }

  bool operator<(Box3d const& other) const
  {
    if (this == &other) return false;
    int cmp = compare_totalsize(other);
    if (cmp != 0) return cmp < 0;
    for (IdxT dim = 0; dim < 3; ++dim) {
      cmp = compare_size(dim, other);
      if (cmp != 0) return cmp < 0;
    }
    for (IdxT dim = 0; dim < 3; ++dim) {
      cmp = compare_min(dim, other);
      if (cmp != 0) return cmp < 0;
    }
    return info < other.info;
  }

  int compare_totalsize(Box3d const& other) const
  {
    if (size() < other.size()) {
      return -1;
    } else if (size() == other.size()) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_size(IdxT dim, Box3d const& other) const
  {
    if (sizes[dim] < other.sizes[dim]) {
      return -1;
    } else if (sizes[dim] == other.sizes[dim]) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_min(IdxT dim, Box3d const& other) const
  {
    if (min[dim] < other.min[dim]) {
      return -1;
    } else if (min[dim] == other.min[dim]) {
      return 0;
    } else {
      return 1;
    }
  }

  bool equivalent(Box3d const& other) const
  {
    // check coordinates are equivalent
    return min[0] == other.min[0] &&
           min[1] == other.min[1] &&
           min[2] == other.min[2] &&
           sizes[0] == other.sizes[0] &&
           sizes[1] == other.sizes[1] &&
           sizes[2] == other.sizes[2] ;
  }

  bool global_equivalent(Box3d const& other) const
  {
    IdxT global_min[3] { min[0] + info.global_offset[0]
                       , min[1] + info.global_offset[1]
                       , min[2] + info.global_offset[2] };

    IdxT other_min[3] { other.min[0] + other.info.global_offset[0]
                      , other.min[1] + other.info.global_offset[1]
                      , other.min[2] + other.info.global_offset[2] };

    bool equiv = true;

    // check sizes are the same
    for(IdxT dim = 0; dim < 3; ++dim) {
      equiv = equiv && (sizes[dim] == other.sizes[dim]);
    }

    // check indices are equivalent
    for(IdxT dim = 0; dim < 3; ++dim) {

      if (info.global.periodic[dim]) {
        global_min[dim] = global_min[dim] % info.global.sizes[dim];
        if (global_min[dim] < 0) global_min[dim] += info.global.sizes[dim];
      }

      if (other.info.global.periodic[dim]) {
        other_min[dim] = other_min[dim] % other.info.global.sizes[dim];
        if (other_min[dim] < 0) other_min[dim] += other.info.global.sizes[dim];
      }

      equiv = equiv && (global_min[dim] == other_min[dim]);
    }

    return equiv;
  }

  template < typename policy >
  void set_indices(policy const& pol, LidxT* index_list) const
  {
    IdxT imin = min[0];
    IdxT jmin = min[1];
    IdxT kmin = min[2];
    IdxT imax = min[0] + sizes[0];
    IdxT jmax = min[1] + sizes[1];
    IdxT kmax = min[2] + sizes[2];
    for_all_3d(pol, kmin, kmax, jmin, jmax, imin, imax, make_set_idxr_idxr(detail::indexer_kji{info.len[0]*info.len[1], info.len[0]}, index_list, detail::indexer_idx{}));
    //for(IdxT idx = 0; idx < (imax-imin)*(jmax-jmin)*(kmax-kmin); ++idx) {
    //  FPRINTF(stdout, "indices[%i] = %i\n", idx, index_list[idx]);
    //  assert(0 <= index_list[idx] && index_list[idx] < (imax-imin)*(jmax-jmin)*(kmax-kmin));
    //}
    synchronize(pol);
  }
};

struct Box3dTemplate
{
  using IT = IdxTemplate;
  using ITL = IdxTemplate::location;
  static Box3d get_connection_inner_box(MeshInfo const& info, const int neighbor_coords[])
  {
    int bounds[3] { neighbor_coords[0] - info.global_coords[0]
                  , neighbor_coords[1] - info.global_coords[1]
                  , neighbor_coords[2] - info.global_coords[2] };

    ITL locs[3] { (bounds[0] == -1) ? ITL::min : ( (bounds[0] == 1) ? ITL::max : ITL::mid )
                , (bounds[1] == -1) ? ITL::min : ( (bounds[1] == 1) ? ITL::max : ITL::mid )
                , (bounds[2] == -1) ? ITL::min : ( (bounds[2] == 1) ? ITL::max : ITL::mid ) };

    return Box3dTemplate::make_Box3dTemplate_inner(info.ghost_width, locs).make_box(info);
  }

  static Box3d get_connection_ghost_box(MeshInfo const& info, const int neighbor_coords[])
  {
    int bounds[3] { neighbor_coords[0] - info.global_coords[0]
                  , neighbor_coords[1] - info.global_coords[1]
                  , neighbor_coords[2] - info.global_coords[2] };

    ITL locs[3] { (bounds[0] == -1) ? ITL::min : ( (bounds[0] == 1) ? ITL::max : ITL::mid )
                , (bounds[1] == -1) ? ITL::min : ( (bounds[1] == 1) ? ITL::max : ITL::mid )
                , (bounds[2] == -1) ? ITL::min : ( (bounds[2] == 1) ? ITL::max : ITL::mid ) };

    return Box3dTemplate::make_Box3dTemplate_ghost(info.ghost_width, locs).make_box(info);
  }

  static Box3dTemplate make_Box3dTemplate_inner(IdxT width, ITL bounds[])
  {
    IT tmin[3] { IT{(bounds[0] == ITL::mid) ? ITL::min : bounds[0], (bounds[0] == ITL::max) ? -width : 0 }
               , IT{(bounds[1] == ITL::mid) ? ITL::min : bounds[1], (bounds[1] == ITL::max) ? -width : 0 }
               , IT{(bounds[2] == ITL::mid) ? ITL::min : bounds[2], (bounds[2] == ITL::max) ? -width : 0 } };
    IT tmax[3] { IT{(bounds[0] == ITL::mid) ? ITL::max : bounds[0], (bounds[0] == ITL::min) ?  width : 0 }
               , IT{(bounds[1] == ITL::mid) ? ITL::max : bounds[1], (bounds[1] == ITL::min) ?  width : 0 }
               , IT{(bounds[2] == ITL::mid) ? ITL::max : bounds[2], (bounds[2] == ITL::min) ?  width : 0 } };
    return Box3dTemplate{ tmin, tmax };
  }
  static Box3dTemplate make_Box3dTemplate_ghost(IdxT width, IT::location bounds[])
  {
    IT tmin[3] { IT{(bounds[0] == ITL::mid) ? ITL::min : bounds[0], (bounds[0] == ITL::min) ? -width : 0 }
               , IT{(bounds[1] == ITL::mid) ? ITL::min : bounds[1], (bounds[1] == ITL::min) ? -width : 0 }
               , IT{(bounds[2] == ITL::mid) ? ITL::min : bounds[2], (bounds[2] == ITL::min) ? -width : 0 } };
    IT tmax[3] { IT{(bounds[0] == ITL::mid) ? ITL::max : bounds[0], (bounds[0] == ITL::max) ?  width : 0 }
               , IT{(bounds[1] == ITL::mid) ? ITL::max : bounds[1], (bounds[1] == ITL::max) ?  width : 0 }
               , IT{(bounds[2] == ITL::mid) ? ITL::max : bounds[2], (bounds[2] == ITL::max) ?  width : 0 } };
    return Box3dTemplate{ tmin, tmax };
  }

  IdxTemplate tmin[3];
  IdxTemplate tmax[3];
  constexpr Box3dTemplate(IdxTemplate const tmin_[], IdxTemplate const tmax_[])
    : tmin{tmin_[0], tmin_[1], tmin_[2]}
    , tmax{tmax_[0], tmax_[1], tmax_[2]}
  {
  }

  Box3d make_box(MeshInfo const& info)
  {
    IdxT min[3] { idx(0, info, tmin[0])
                , idx(1, info, tmin[1])
                , idx(2, info, tmin[2]) };
    IdxT max[3] { idx(0, info, tmax[0])
                , idx(1, info, tmax[1])
                , idx(2, info, tmax[2]) };
    return Box3d{info, min, max};
  }

private:

  IdxT idx(IdxT dim, MeshInfo const& info, IdxTemplate it)
  {
    IdxT idx = 0;
    switch (it.idx) {
      case ITL::min:
        idx = info.min[0]; break;
      case ITL::max:
        idx = info.max[0]; break;
      default:
        assert(0); break;
    }
    return idx + it.offset;
  }
};

#endif // _BOX3D_HPP

