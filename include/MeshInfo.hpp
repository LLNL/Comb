//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by Jason Burmark, burmark1@llnl.gov
// LLNL-CODE-758885
//
// All rights reserved.
//
// This file is part of Comb.
//
// For details, see https://github.com/LLNL/Comb
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#ifndef _MESHINFO_HPP
#define _MESHINFO_HPP

#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <set>

struct GlobalMeshInfo {
  IdxT sizes[3];
  IdxT totalsize;
  int divisions[3];
  int periodic[3];
  IdxT ghost_width;
  IdxT* division_indices[3];

  GlobalMeshInfo(IdxT sizes_[], IdxT num_divisions, int divisions_[], int periodic_[], IdxT ghost_width_)
    : sizes{sizes_[0], sizes_[1], sizes_[2]}
    , totalsize{sizes_[0] * sizes_[1] * sizes_[2]}
    , divisions{divisions_[0], divisions_[1], divisions_[2]}
    , periodic{periodic_[0] ? 1 : 0, periodic_[1] ? 1 : 0, periodic_[2] ? 1 : 0}
    , ghost_width(ghost_width_)
    , division_indices{nullptr, nullptr, nullptr}
  {
    set_divisions(num_divisions);
  }

  void set_divisions(IdxT num_divisions)
  {
    if (divisions[0] == 0 && divisions[1] == 0 && divisions[2] == 0) {
      // decide how to cut up mesh

      IdxT P = num_divisions;
      IdxT sqrtP = sqrt(P);

      // get prime factors of P
      std::multiset<IdxT> prime_factors;
      {
        // get list of possible prime factors (excluding P)
        std::vector<IdxT> primes({2});
        for(IdxT p = 3; p < sqrtP; p += 2) {
          IdxT primes_size = primes.size();
          IdxT i = 0;
          for (; i < primes_size; ++i) {
            if (p % primes[i] == 0) break; // not prime
          }
          if (i == primes_size) {
            primes.push_back(p);
          }
        }

        IdxT P_tmp = P;
        IdxT pi = 0;
        IdxT primes_size = primes.size();
        while(pi != primes_size) {
          if (P_tmp % primes[pi] == 0) {
            // found prime factor
            prime_factors.insert(primes[pi]);
            P_tmp /= primes[pi];
          } else {
            ++pi;
          }
        }
        if (P_tmp != 1) {
          prime_factors.insert(P_tmp);
        }
      }

      double I = std::cbrt(sizes[0] * sizes[1] * sizes[2]) / P;
      IdxT divisions[3] {1, 1, 1};

      while(std::begin(prime_factors) != std::end(prime_factors)) {

        double best_relative_remainder_dist = std::numeric_limits<double>::max();
        auto best_factor = std::end(prime_factors);
        IdxT best_dim = 3;

        double ideal[3] {sizes[0] / (divisions[0] * I),
                         sizes[1] / (divisions[1] * I),
                         sizes[2] / (divisions[2] * I)};

        auto end = std::end(prime_factors);
        for(auto k = std::begin(prime_factors); k != end; ++k) {
          IdxT factor = *k;
          for(IdxT dim = 2; dim >= 0; --dim) {
            double remainder = std::fmod(ideal[dim], factor);
            double remainder_dist = std::min(remainder, factor - remainder);
            double relative_remainder_dist = remainder_dist / factor;
            if (relative_remainder_dist < best_relative_remainder_dist) {
              best_relative_remainder_dist = relative_remainder_dist;
              best_factor = k;
              best_dim = dim;
            }
          }
        }

        assert(best_factor != end && best_dim < 3);

        divisions[best_dim] *= *best_factor;

        prime_factors.erase(best_factor);
      }

      divisions[0] = divisions[0];
      divisions[1] = divisions[1];
      divisions[2] = divisions[2];
    }

    assert(sizes[0] >= divisions[0] && sizes[1] >= divisions[1] && sizes[2] >= divisions[2]);

    for (IdxT dim = 0; dim < 3; ++dim) {
      division_indices[dim] = new IdxT[divisions[dim]+1];

      for (IdxT div = 0; div < divisions[dim]+1; ++div) {
        division_indices[dim][div] = div * (sizes[dim] / divisions[dim]) + std::min(div, sizes[dim] % divisions[dim]);
      }
      assert(division_indices[dim][0] == 0);
      assert(division_indices[dim][divisions[dim]] == sizes[dim]);
    }
  }

  IdxT division_index(IdxT dim, IdxT coord) const
  {
    // correct for periodicity when looking up values
    IdxT idx_offset = 0;
    if (periodic[dim]) {
      IdxT mult = coord / divisions[dim];
      coord = coord % divisions[dim];
      if (coord < 0) {
        mult -= 1;
        coord += divisions[dim];
      }
      idx_offset = mult * sizes[dim];
    }
    assert(0 <= coord && coord <= divisions[dim]);
    return division_indices[dim][coord] + idx_offset;
  }

  bool operator==(GlobalMeshInfo const& other) const
  {
    if (this == &other) return true;
    bool equal = false;
    equal = equal && (ghost_width == other.ghost_width);
    for (IdxT dim = 0; dim < 3; ++dim) {
      equal = compare_size(dim, other) == 0;
    }
    return equal;
  }

  bool operator<(GlobalMeshInfo const& other) const
  {
    if (this == &other) return false;
    int cmp = compare_totalsize(other);
    if (cmp != 0) return cmp < 0;
    for (IdxT dim = 0; dim < 3; ++dim) {
      cmp = compare_size(dim, other);
      if (cmp != 0) return cmp < 0;
    }
    cmp = compare_ghostwidth(other);
    if (cmp != 0) return cmp < 0;
    for (IdxT dim = 0; dim < 3; ++dim) {
      cmp = compare_periodic(dim, other);
      if (cmp != 0) return cmp < 0;
    }
    return false;
  }

  int compare_totalsize(GlobalMeshInfo const& other) const
  {
    if (totalsize < other.totalsize) {
      return -1;
    } else if (totalsize == other.totalsize) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_size(IdxT dim, GlobalMeshInfo const& other) const
  {
    if (sizes[dim] < other.sizes[dim]) {
      return -1;
    } else if (sizes[dim] == other.sizes[dim]) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_ghostwidth(GlobalMeshInfo const& other) const
  {
    if (ghost_width < other.ghost_width) {
      return -1;
    } else if (ghost_width == other.ghost_width) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_periodic(IdxT dim, GlobalMeshInfo const& other) const
  {
    if (periodic[dim] < other.periodic[dim]) {
      return -1;
    } else if (periodic[dim] == other.periodic[dim]) {
      return 0;
    } else {
      return 1;
    }
  }


  ~GlobalMeshInfo()
  {
    if (division_indices[0]) delete[] division_indices[0];
    if (division_indices[1]) delete[] division_indices[1];
    if (division_indices[2]) delete[] division_indices[2];
  }
};

struct MeshInfo {

  // when over a periodic boundary will return a meshinfo that lies outside the global mesh
  static MeshInfo get_local(GlobalMeshInfo const& global, const int arg_coords[])
  {
    IdxT global_min[3]{ global.division_index(0, arg_coords[0]  )
                      , global.division_index(1, arg_coords[1]  )
                      , global.division_index(2, arg_coords[2]  ) };
    IdxT global_max[3]{ global.division_index(0, arg_coords[0]+1)
                      , global.division_index(1, arg_coords[1]+1)
                      , global.division_index(2, arg_coords[2]+1) };
    return MeshInfo{global, global_min, global_max, global.ghost_width, arg_coords};
  }

  GlobalMeshInfo const& global;
  IdxT min[3];
  IdxT max[3];
  IdxT size[3];
  IdxT totalsize;
  IdxT len[3];
  IdxT totallen;
  IdxT stride[3]; // stride[0] = 1
  IdxT ghost_width;
  IdxT global_min[3];
  IdxT global_max[3];
  IdxT global_offset[3];
  IdxT global_own_min[3];
  IdxT global_own_max[3];
  IdxT global_coords[3];

  MeshInfo(GlobalMeshInfo const& global_,
           const IdxT global_min_[], const IdxT global_max_[],
           IdxT ghost_width_, const int global_coords_[])
    : global(global_)
    , min{ ghost_width_
         , ghost_width_
         , ghost_width_ }
    , max{ global_max_[0] - global_min_[0] + ghost_width_
         , global_max_[1] - global_min_[1] + ghost_width_
         , global_max_[2] - global_min_[2] + ghost_width_ }
    , size{ global_max_[0] - global_min_[0]
          , global_max_[1] - global_min_[1]
          , global_max_[2] - global_min_[2] }
    , totalsize{ (global_max_[0] - global_min_[0])
               * (global_max_[1] - global_min_[1])
               * (global_max_[2] - global_min_[2]) }
    , len{ global_max_[0] - global_min_[0] + 2*ghost_width_
         , global_max_[1] - global_min_[1] + 2*ghost_width_
         , global_max_[2] - global_min_[2] + 2*ghost_width_ }
    , totallen{ (global_max_[0] - global_min_[0] + 2*ghost_width_)
              * (global_max_[1] - global_min_[1] + 2*ghost_width_)
              * (global_max_[2] - global_min_[2] + 2*ghost_width_) }
    , stride{ 1
            , global_max_[0] - global_min_[0] + 2*ghost_width_
            , (global_max_[0] - global_min_[0] + 2*ghost_width_) * (global_max_[1] - global_min_[1] + 2*ghost_width_) }
    , ghost_width(ghost_width_)
    , global_min{ global_min_[0]
                , global_min_[1]
                , global_min_[2] }
    , global_max{ global_max_[0]
                , global_max_[1]
                , global_max_[2] }
    , global_offset{ global_min_[0] - ghost_width_
                   , global_min_[1] - ghost_width_
                   , global_min_[2] - ghost_width_ }
    , global_own_min{ global_min_[0] - ((global_min_[0] == 0                && !global_.periodic[0]) ? ghost_width_ : 0)
                    , global_min_[1] - ((global_min_[1] == 0                && !global_.periodic[1]) ? ghost_width_ : 0)
                    , global_min_[2] - ((global_min_[2] == 0                && !global_.periodic[2]) ? ghost_width_ : 0) }
    , global_own_max{ global_max_[0] + ((global_max_[0] == global_.sizes[0] && !global_.periodic[0]) ? ghost_width_ : 0)
                    , global_max_[1] + ((global_max_[1] == global_.sizes[1] && !global_.periodic[1]) ? ghost_width_ : 0)
                    , global_max_[2] + ((global_max_[2] == global_.sizes[2] && !global_.periodic[2]) ? ghost_width_ : 0) }
    , global_coords{ global_coords_[0]
                   , global_coords_[1]
                   , global_coords_[2] }
  {
    assert(size[0] >= ghost_width && size[1] >= ghost_width && size[2] >= ghost_width);
  }

  void correct_periodicity()
  {
    for (IdxT dim = 0; dim < 3; ++dim) {

      IdxT idx_offset = 0;
      if (global.periodic[dim]) {
        IdxT mult = global_coords[dim] / global.divisions[dim];
        global_coords[dim] = global_coords[dim] % global.divisions[dim];
        if (global_coords[dim] < 0) {
          mult -= 1;
          global_coords[dim] += global.divisions[dim];
        }
        idx_offset = mult * global.sizes[dim];
      }
      assert(0 <= global_coords[dim] && global_coords[dim] < global.divisions[dim]);

      global_min[dim] -= idx_offset;
      global_max[dim] -= idx_offset;
      global_offset[dim] -= idx_offset;
      global_own_min[dim] -= idx_offset;
      global_own_max[dim] -= idx_offset;
    }
  }

  bool operator==(MeshInfo const& other) const
  {
    return this == &other || (
           global == other.global &&
           global_min[0] == other.global_min[0] &&
           global_min[1] == other.global_min[1] &&
           global_min[2] == other.global_min[2] &&
           global_max[0] == other.global_max[0] &&
           global_max[1] == other.global_max[1] &&
           global_max[2] == other.global_max[2] );
  }

  bool operator<(MeshInfo const& other) const
  {
    if (this == &other) return false;
    int cmp = compare_global(other);
    if (cmp != 0) return cmp < 0;
    cmp = compare_totalsize(other);
    if (cmp != 0) return cmp < 0;
    for (IdxT dim = 0; dim < 3; ++dim) {
      cmp = compare_size(dim, other);
      if (cmp != 0) return cmp < 0;
    }
    for (IdxT dim = 0; dim < 3; ++dim) {
      cmp = compare_globalmin(dim, other);
      if (cmp != 0) return cmp < 0;
    }
    return false;
  }

  int compare_global(MeshInfo const& other) const
  {
    if (global < other.global) {
      return -1;
    } else if (global == other.global) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_totalsize(MeshInfo const& other) const
  {
    if (totalsize < other.totalsize) {
      return -1;
    } else if (totalsize == other.totalsize) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_size(IdxT dim, MeshInfo const& other) const
  {
    if (size[dim] < other.size[dim]) {
      return -1;
    } else if (size[dim] == other.size[dim]) {
      return 0;
    } else {
      return 1;
    }
  }

  int compare_globalmin(IdxT dim, MeshInfo const& other) const
  {
    if (global_min[dim] < other.global_min[dim]) {
      return -1;
    } else if (global_min[dim] == other.global_min[dim]) {
      return 0;
    } else {
      return 1;
    }
  }

  ~MeshInfo()
  {

  }
};


#endif // _MESHINFO_HPP

