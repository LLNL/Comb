
#ifndef _MESH_CUH
#define _MESH_CUH

#include "memory.cuh"

struct MeshInfo {
  IdxT imin;
  IdxT jmin;
  IdxT kmin;
  IdxT imax;
  IdxT jmax;
  IdxT kmax;
  IdxT isize;
  IdxT jsize;
  IdxT ksize;
  IdxT ijsize;
  IdxT iksize;
  IdxT jksize;
  IdxT ijksize;
  IdxT ilen;
  IdxT jlen;
  IdxT klen;
  IdxT ijlen;
  IdxT iklen;
  IdxT jklen;
  IdxT ijklen;
  IdxT ghost_width;

  MeshInfo(IdxT isize_, IdxT jsize_, IdxT ksize_, IdxT ghost_width_)
    : imin(ghost_width_)
    , jmin(ghost_width_)
    , kmin(ghost_width_)
    , imax(imin + isize_)
    , jmax(jmin + jsize_)
    , kmax(kmin + ksize_)
    , isize(imax - imin)
    , jsize(jmax - jmin)
    , ksize(kmax - kmin)
    , ijsize(isize*jsize)
    , iksize(isize*ksize)
    , jksize(jsize*ksize)
    , ijksize(isize*jsize*ksize)
    , ilen(imax + ghost_width_)
    , jlen(jmax + ghost_width_)
    , klen(kmax + ghost_width_)
    , ijlen(ilen*jlen)
    , iklen(ilen*klen)
    , jklen(jlen*klen)
    , ijklen(ilen*jlen*klen)
    , ghost_width(ghost_width_)
  {

  }

  ~MeshInfo()
  {

  }
};

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
      ptr = (DataT*)aloc.allocate(info.ijklen*sizeof(DataT));
    }
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

#endif // _MESH_CUH

