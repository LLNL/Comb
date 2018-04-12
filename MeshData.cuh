
#ifndef _MESHDATA_CUH
#define _MESHDATA_CUH

#include "memory.cuh"
#include "MeshInfo.cuh"

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

#endif // _MESHDATA_CUH

