
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

  MeshInfo(IdxT isize_, IdxT jsize_, IdxT ksize_)
    : imin(1)
    , jmin(1)
    , kmin(1)
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
    , ilen(imax + 1)
    , jlen(jmax + 1)
    , klen(kmax + 1)
    , ijlen(ilen*jlen)
    , iklen(ilen*klen)
    , jklen(jlen*klen)
    , ijklen(ilen*jlen*klen)
  {

  }

  ~MeshInfo()
  {

  }
};

struct MeshData
{
  Allocator& m_alloc;
  MeshInfo const& m_meshinfo;
  DataT* m_data;
  
  MeshData(MeshInfo const& meshinfo, Allocator& alloc)
    : m_alloc(alloc)
    , m_meshinfo(meshinfo)
    , m_data(nullptr)
  {

  }

  void allocate()
  {
    if (m_data == nullptr) {
      m_data = (DataT*)m_alloc.allocate(m_meshinfo.ijklen*sizeof(DataT));
    }
  }

  DataT* data()
  {
    return m_data;
  }

  void deallocate()
  {
    if (m_data != nullptr) {
      m_alloc.deallocate(m_data);
      m_data = nullptr;
    }
  }

  ~MeshData()
  {
    deallocate();
  }
};

#endif // _MESH_CUH

