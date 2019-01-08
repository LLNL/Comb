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

#ifndef _MESHDATA_HPP
#define _MESHDATA_HPP

#include "config.hpp"

#include "memory.hpp"
#include "MeshInfo.hpp"

struct MeshData
{
  Allocator& aloc;
  MeshInfo const& info;
  DataT* ptr;

  MeshData(MeshInfo const& meshinfo, Allocator& aloc_)
    : aloc(aloc_)
    , info(meshinfo)
    , ptr(nullptr)
  {

  }

  void allocate()
  {
    if (ptr == nullptr) {
      ptr = (DataT*)aloc.allocate(info.totallen*sizeof(DataT));
    }
  }

  bool operator==(MeshData const& other) const
  {
    return aloc.name() == other.aloc.name() &&
           info == other.info &&
           ptr == other.ptr;
  }

  DataT* data() const
  {
    return ptr;
  }

  void deallocate()
  {
    if (ptr != nullptr) {
      aloc.deallocate(ptr);
      ptr = nullptr;
    }
  }

  ~MeshData()
  {
    deallocate();
  }
};

#endif // _MESHDATA_HPP

