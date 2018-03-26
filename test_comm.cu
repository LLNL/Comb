#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <chrono>

#include <cuda.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <mpi.h>

#include <basic_mempool.hpp>

#define HOST __host__
#define DEVICE __device__

#define cudaCheck(...) \
  if (__VA_ARGS__ != cudaSuccess) { \
    fprintf(stderr, "Error performaing " #__VA_ARGS__ " %s:%i\n", __FILE__, __LINE__); fflush(stderr); \
    assert(0); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  }

using IdxT = int;
using DataT = double;

inline int cuda_get_device() {
  static int d = -1;
  if (d == -1) {
    cudaCheck(cudaGetDevice(&d));
  }
  return d;
}

template < typename alloc >
using mempool = RAJA::basic_mempool::MemPool<alloc>;

  struct cuda_host_pinned_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFreeHost(ptr));
    }
  };

  struct cuda_device_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMalloc(&ptr, nbytes));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_read_mostly_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetReadMostly, 0);
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_host_preferred_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_host_preferred_device_accessed_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, cuda_get_device()));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

  struct cuda_managed_device_preferred_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cuda_get_device());
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };
  
  struct cuda_managed_device_preferred_host_accessed_allocator {
    void* malloc(size_t nbytes) {
      void* ptr = nullptr;
      cudaCheck(cudaMallocManaged(&ptr, nbytes));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, cuda_get_device()));
      cudaCheck(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
      return ptr;
    }
    void free(void* ptr) {
      cudaCheck(cudaFree(ptr));
    }
  };

struct Allocator
{
  virtual const char* name() = 0;
  virtual void* allocate(size_t) = 0;
  virtual void deallocate(void*) = 0;
};

struct HostAllocator : Allocator
{
  virtual const char* name() { return "Host"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = malloc(nbytes);
    //printf("allocated %p nbytes %zu\n", ptr, nbytes); fflush(stdout);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    //printf("deallocating %p\n", ptr); fflush(stdout);
    free(ptr);
  }
};

HostAllocator host_alloc;

struct HostPinnedAllocator : Allocator
{
  virtual const char* name() { return "HostPinned"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_host_pinned_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_host_pinned_allocator>::getInstance().free(ptr);
  }
};

HostPinnedAllocator hostpinned_alloc;

struct DeviceAllocator : Allocator
{
  virtual const char* name() { return "Device"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_device_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_device_allocator>::getInstance().free(ptr);
  }
};

DeviceAllocator device_alloc;

struct ManagedAllocator : Allocator
{
  virtual const char* name() { return "Managed"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_allocator>::getInstance().free(ptr);
  }
};

ManagedAllocator managed_alloc;

struct ManagedHostPreferredAllocator : Allocator
{
  virtual const char* name() { return "ManagedHostPreferred"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_host_preferred_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_host_preferred_allocator>::getInstance().free(ptr);
  }
};

ManagedHostPreferredAllocator managed_host_preferred_alloc;

struct ManagedHostPreferredDeviceAccessedAllocator : Allocator
{
  virtual const char* name() { return "ManagedHostPreferredDeviceAccessed"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_host_preferred_device_accessed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_host_preferred_device_accessed_allocator>::getInstance().free(ptr);
  }
};

ManagedHostPreferredDeviceAccessedAllocator managed_host_preferred_device_accessed_alloc;

struct ManagedDevicePreferredAllocator : Allocator
{
  virtual const char* name() { return "ManagedDevicePreferred"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_device_preferred_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_device_preferred_allocator>::getInstance().free(ptr);
  }
};

ManagedDevicePreferredAllocator managed_device_preferred_alloc;

struct ManagedDevicePreferredhostAccessedAllocator : Allocator
{
  virtual const char* name() { return "ManagedDevicePreferredHostAccessed"; }
  virtual void* allocate(size_t nbytes)
  {
    void* ptr = mempool<cuda_managed_device_preferred_host_accessed_allocator>::getInstance().malloc<char>(nbytes);
    return ptr;
  }
  virtual void deallocate(void* ptr)
  {
    mempool<cuda_managed_device_preferred_host_accessed_allocator>::getInstance().free(ptr);
  }
};

ManagedDevicePreferredHostAccessedAllocator managed_device_preferred_host_accessed_alloc;




struct seq_pol { static const bool async = false; };
struct omp_pol { static const bool async = false; };
struct cuda_pol { static const bool async = true; };

template < typename body_type >
void for_all(seq_pol const&, IdxT begin, IdxT end, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin; i0 < end; ++i0) {
    body(i0, i++);
  }
}

template < typename body_type >
void for_all(omp_pol const&, IdxT begin, IdxT end, body_type&& body)
{
  const IdxT len = end - begin;
#pragma omp parallel for
  for(IdxT i = 0; i < len; ++i) {
    body(i + begin, i);
  }
}

template < typename body_type >
__global__
void cuda_for_all(IdxT begin, IdxT len, body_type body)
{
  const IdxT i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    body(i + begin, i);
  }
}

template < typename body_type >
void for_all(cuda_pol const&, IdxT begin, IdxT end, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len = end - begin;

  const IdxT threads = 256;
  const IdxT blocks = (len + threads - 1) / threads;

  void* func = (void*)&cuda_for_all<decayed_body_type>;
  dim3 gridDim(blocks);
  dim3 blockDim(threads);
  void* args[]{&begin, &len, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = 0;
  
  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}


template < typename body_type >
void for_all_2d(seq_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      body(i0, i1, i++);
    }
  }
}

template < typename body_type >
void for_all_2d(omp_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  const IdxT len0 = end0 - begin0;
  const IdxT len1 = end1 - begin1;
#pragma omp parallel for collapse(2)
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }
}

template < typename body_type >
__global__
void cuda_for_all_2d(IdxT begin0, IdxT len0, IdxT begin1, IdxT len1, body_type body)
{
  const IdxT i0 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }
}

template < typename body_type >
void for_all_2d(cuda_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;

  const IdxT threads0 = 8;
  const IdxT threads1 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;

  void* func = (void*)&cuda_for_all_2d<decayed_body_type>;
  dim3 gridDim(blocks1, blocks0, 1);
  dim3 blockDim(threads1, threads0, 1);
  void* args[]{&begin0, &len0, &begin1, &len1, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = 0;
  
  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}


template < typename body_type >
void for_all_3d(seq_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  IdxT i = 0;
  for(IdxT i0 = begin0; i0 < end0; ++i0) {
    for(IdxT i1 = begin1; i1 < end1; ++i1) {
      for(IdxT i2 = begin2; i2 < end2; ++i2) {
        body(i0, i1, i2, i++);
      }
    }
  }
}

template < typename body_type >
void for_all_3d(omp_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  const IdxT len0 = end0 - begin0;
  const IdxT len1 = end1 - begin1;
  const IdxT len2 = end2 - begin2;
  const IdxT len12 = len1 * len2;
#pragma omp parallel for collapse(3)
  for(IdxT i0 = 0; i0 < len0; ++i0) {
    for(IdxT i1 = 0; i1 < len1; ++i1) {
      for(IdxT i2 = 0; i2 < len2; ++i2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }
}

template < typename body_type >
__global__
void cuda_for_all_3d(IdxT begin0, IdxT len0, IdxT begin1, IdxT len1, IdxT begin2, IdxT len2, IdxT len12, body_type body)
{
  const IdxT i0 = blockIdx.z;
  const IdxT i1 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i2 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      if (i2 < len2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }
}

template < typename body_type >
void for_all_3d(cuda_pol const&, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;
  IdxT len2 = end2 - begin2;
  IdxT len12 = len1 * len2;

  const IdxT threads0 = 1;
  const IdxT threads1 = 8;
  const IdxT threads2 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;
  const IdxT blocks2 = (len2 + threads2 - 1) / threads2;

  void* func =(void*)&cuda_for_all_3d<decayed_body_type>;
  dim3 gridDim(blocks2, blocks1, blocks0);
  dim3 blockDim(threads2, threads1, threads0);
  void* args[]{&begin0, &len0, &begin1, &len1, &begin2, &len2, &len12, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = 0;
  
  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}



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

struct CommBuffers
{
  Allocator& m_corner;
  Allocator& m_edge;
  Allocator& m_face;

  bool m_allocated;

  DataT* corner_imin_jmin_kmin;
  DataT* corner_imax_jmin_kmin;
  DataT* corner_imin_jmax_kmin;
  DataT* corner_imax_jmax_kmin;
  DataT* corner_imin_jmin_kmax;
  DataT* corner_imax_jmin_kmax;
  DataT* corner_imin_jmax_kmax;
  DataT* corner_imax_jmax_kmax;

  DataT* edge_i_jmin_kmin;
  DataT* edge_i_jmax_kmin;
  DataT* edge_i_jmin_kmax;
  DataT* edge_i_jmax_kmax;
  DataT* edge_imin_j_kmin;
  DataT* edge_imax_j_kmin;
  DataT* edge_imin_j_kmax;
  DataT* edge_imax_j_kmax;
  DataT* edge_imin_jmin_k;
  DataT* edge_imax_jmin_k;
  DataT* edge_imin_jmax_k;
  DataT* edge_imax_jmax_k;

  DataT* face_i_j_kmin;
  DataT* face_i_j_kmax;
  DataT* face_i_jmin_k;
  DataT* face_i_jmax_k;
  DataT* face_imin_j_k;
  DataT* face_imax_j_k;

  CommBuffers(Allocator& corner, Allocator& edge, Allocator& face)
    : m_corner(corner)
    , m_edge(edge)
    , m_face(face)
    , m_allocated(false)
  {

  }

  void allocate(MeshInfo const& mesh)
  {
    if (m_allocated) deallocate();
    //printf("allocateing buffers\n"); fflush(stdout);
    corner_imin_jmin_kmin = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imax_jmin_kmin = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imin_jmax_kmin = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imax_jmax_kmin = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imin_jmin_kmax = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imax_jmin_kmax = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imin_jmax_kmax = (DataT*)m_corner.allocate(1*sizeof(DataT));
    corner_imax_jmax_kmax = (DataT*)m_corner.allocate(1*sizeof(DataT));
    edge_i_jmin_kmin = (DataT*)m_edge.allocate(mesh.isize*sizeof(DataT)); 
    edge_i_jmax_kmin = (DataT*)m_edge.allocate(mesh.isize*sizeof(DataT)); 
    edge_i_jmin_kmax = (DataT*)m_edge.allocate(mesh.isize*sizeof(DataT)); 
    edge_i_jmax_kmax = (DataT*)m_edge.allocate(mesh.isize*sizeof(DataT)); 
    edge_imin_j_kmin = (DataT*)m_edge.allocate(mesh.jsize*sizeof(DataT)); 
    edge_imax_j_kmin = (DataT*)m_edge.allocate(mesh.jsize*sizeof(DataT)); 
    edge_imin_j_kmax = (DataT*)m_edge.allocate(mesh.jsize*sizeof(DataT)); 
    edge_imax_j_kmax = (DataT*)m_edge.allocate(mesh.jsize*sizeof(DataT)); 
    edge_imin_jmin_k = (DataT*)m_edge.allocate(mesh.ksize*sizeof(DataT)); 
    edge_imax_jmin_k = (DataT*)m_edge.allocate(mesh.ksize*sizeof(DataT)); 
    edge_imin_jmax_k = (DataT*)m_edge.allocate(mesh.ksize*sizeof(DataT));
    edge_imax_jmax_k = (DataT*)m_edge.allocate(mesh.ksize*sizeof(DataT));
    face_i_j_kmin = (DataT*)m_edge.allocate(mesh.ijsize*sizeof(DataT)); 
    face_i_j_kmax = (DataT*)m_edge.allocate(mesh.ijsize*sizeof(DataT)); 
    face_i_jmin_k = (DataT*)m_edge.allocate(mesh.iksize*sizeof(DataT)); 
    face_i_jmax_k = (DataT*)m_edge.allocate(mesh.iksize*sizeof(DataT)); 
    face_imin_j_k = (DataT*)m_edge.allocate(mesh.jksize*sizeof(DataT)); 
    face_imax_j_k = (DataT*)m_edge.allocate(mesh.jksize*sizeof(DataT));
    m_allocated = true;
  }

  void deallocate()
  {
    if (m_allocated) {
      //printf("deallocating buffers\n"); fflush(stdout);
      m_corner.deallocate(corner_imin_jmin_kmin);
      m_corner.deallocate(corner_imax_jmin_kmin);
      m_corner.deallocate(corner_imin_jmax_kmin);
      m_corner.deallocate(corner_imax_jmax_kmin);
      m_corner.deallocate(corner_imin_jmin_kmax);
      m_corner.deallocate(corner_imax_jmin_kmax);
      m_corner.deallocate(corner_imin_jmax_kmax);
      m_corner.deallocate(corner_imax_jmax_kmax);
      m_edge.deallocate(edge_i_jmin_kmin); 
      m_edge.deallocate(edge_i_jmax_kmin); 
      m_edge.deallocate(edge_i_jmin_kmax); 
      m_edge.deallocate(edge_i_jmax_kmax); 
      m_edge.deallocate(edge_imin_j_kmin); 
      m_edge.deallocate(edge_imax_j_kmin); 
      m_edge.deallocate(edge_imin_j_kmax); 
      m_edge.deallocate(edge_imax_j_kmax); 
      m_edge.deallocate(edge_imin_jmin_k); 
      m_edge.deallocate(edge_imax_jmin_k); 
      m_edge.deallocate(edge_imin_jmax_k); 
      m_edge.deallocate(edge_imax_jmax_k); 
      m_face.deallocate(face_i_j_kmin); 
      m_face.deallocate(face_i_j_kmax); 
      m_face.deallocate(face_i_jmin_k); 
      m_face.deallocate(face_i_jmax_k); 
      m_face.deallocate(face_imin_j_k); 
      m_face.deallocate(face_imax_j_k);
      m_allocated = false;
    }
  }
 
  ~CommBuffers()
  {
    deallocate();
  }
};

struct indexer_ji {
  IdxT ilen, koff;
  indexer_ji(IdxT ilen_, IdxT koff_) : ilen(ilen_), koff(koff_) {}
  HOST DEVICE IdxT operator()(IdxT j, IdxT i, IdxT) const { return i + j * ilen + koff; }
};
struct indexer_ki {
  IdxT ijlen, joff;
  indexer_ki(IdxT ijlen_, IdxT joff_) : ijlen(ijlen_), joff(joff_) {}
  HOST DEVICE IdxT operator()(IdxT k, IdxT i, IdxT) const { return i + joff + k * ijlen; }
};
struct indexer_kj {
  IdxT ijlen, ilen, ioff;
  indexer_kj(IdxT ijlen_, IdxT ilen_, IdxT ioff_) : ijlen(ijlen_), ilen(ilen_), ioff(ioff_) {}
  HOST DEVICE IdxT operator()(IdxT k, IdxT j, IdxT) const { return ioff + j * ilen + k * ijlen; }
};

struct indexer_i {
  IdxT off;
  indexer_i(IdxT off_) : off(off_) {}
  HOST DEVICE IdxT operator()(IdxT i, IdxT) const { return i + off; }
};
struct indexer_j {
  IdxT ilen, off;
  indexer_j(IdxT ilen_, IdxT off_) : ilen(ilen_), off(off_) {}
  HOST DEVICE IdxT operator()(IdxT j, IdxT) const { return j * ilen + off; }
};
struct indexer_k {
  IdxT ijlen, off;
  indexer_k(IdxT ijlen_, IdxT off_) : ijlen(ijlen_), off(off_) {}
  HOST DEVICE IdxT operator()(IdxT k, IdxT) const { return k * ijlen + off; }
};

struct indexer_ {
  IdxT off;
  indexer_(IdxT off_) : off(off_) {}
  HOST DEVICE IdxT operator()(IdxT, IdxT) const { return off; }
};

struct indexer_idx {
  indexer_idx() {}
  HOST DEVICE IdxT operator()(IdxT, IdxT idx) const { return idx; }
  HOST DEVICE IdxT operator()(IdxT, IdxT, IdxT idx) const { return idx; }
  HOST DEVICE IdxT operator()(IdxT, IdxT, IdxT, IdxT idx) const { return idx; }
};

template < typename T_src, typename I_src, typename T_dst, typename I_dst >
struct copy_idxr_idxr {
  T_src* ptr_src;
  T_dst* ptr_dst;
  I_src idxr_src;
  I_dst idxr_dst;
  copy_idxr_idxr(T_src* const& ptr_src_, I_src const& idxr_src_, T_dst* const& ptr_dst_, I_dst const& idxr_dst_) : ptr_src(ptr_src_), ptr_dst(ptr_dst_), idxr_src(idxr_src_), idxr_dst(idxr_dst_) {}
  template < typename ... Ts >
  HOST DEVICE void operator()(Ts... args) const { ptr_dst[idxr_dst(args...)] = ptr_src[idxr_src(args...)]; }
};
template < typename T_src, typename I_src, typename T_dst, typename I_dst >
copy_idxr_idxr<T_src, I_src, T_dst, I_dst> make_copy_idxr_idxr(T_src* const& ptr_src, I_src const& idxr_src, T_dst* const& ptr_dst, I_dst const& idxr_dst) {
  return copy_idxr_idxr<T_src, I_src, T_dst, I_dst>(ptr_src, idxr_src, ptr_dst, idxr_dst);
}

template < typename policy, typename indexer >
void pack_face( policy const& pol, DataT* data, DataT* buffer, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, indexer&& idxr)
{
  for_all_2d(pol, begin0, end0,
                  begin1, end1,
                  make_copy_idxr_idxr(data, idxr, buffer, indexer_idx{}));
}
template < typename policy, typename indexer >
void pack_edge( policy const& pol, DataT* data, DataT* buffer, IdxT begin, IdxT end, indexer&& idxr)
{
  for_all(pol, begin, end,
               make_copy_idxr_idxr(data, idxr, buffer, indexer_idx{}));
}
template < typename policy, typename indexer >
void pack_corner( policy const& pol, DataT* data, DataT* buffer, indexer&& idxr)
{
  for_all(pol, 0, 1,
               make_copy_idxr_idxr(data, idxr, buffer, indexer_idx{}));
}

template < typename policy, typename indexer >
void unpack_face( policy const& pol, DataT* data, DataT* buffer, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, indexer&& idxr)
{
  for_all_2d(pol, begin0, end0,
                  begin1, end1,
                  make_copy_idxr_idxr(buffer, indexer_idx{}, data, idxr));
}
template < typename policy, typename indexer >
void unpack_edge( policy const& pol, DataT* data, DataT* buffer, IdxT begin, IdxT end, indexer&& idxr)
{
  for_all(pol, begin, end,
               make_copy_idxr_idxr(buffer, indexer_idx{}, data, idxr));
}
template < typename policy, typename indexer >
void unpack_corner( policy const& pol, DataT* data, DataT* buffer, indexer&& idxr)
{
  for_all(pol, 0, 1,
               make_copy_idxr_idxr(buffer, indexer_idx{}, data, idxr));
}



template < typename policy_corner, typename policy_edge, typename policy_face >
struct Comm
{
  Allocator& m_corner_alloc;
  Allocator& m_edge_alloc;
  Allocator& m_face_alloc;

  MeshData& m_meshdata;

  CommBuffers m_send;
  CommBuffers m_recv;

  Comm(MeshData& meshdata, Allocator& corner_alloc, Allocator& edge_alloc, Allocator& face_alloc)
    : m_corner_alloc(corner_alloc)
    , m_edge_alloc(edge_alloc)
    , m_face_alloc(face_alloc)
    , m_meshdata(meshdata)
    , m_send(m_corner_alloc, m_edge_alloc, m_face_alloc)
    , m_recv(m_corner_alloc, m_edge_alloc, m_face_alloc)
  {

  }

  void postRecv()
  {
    //printf("posting receives\n"); fflush(stdout);
    m_recv.allocate(m_meshdata.m_meshinfo);
  }

  void postSend()
  {
    //printf("posting sends\n"); fflush(stdout);
    m_send.allocate(m_meshdata.m_meshinfo);
    
    IdxT imin = m_meshdata.m_meshinfo.imin;
    IdxT jmin = m_meshdata.m_meshinfo.jmin;
    IdxT kmin = m_meshdata.m_meshinfo.kmin;
    IdxT imax = m_meshdata.m_meshinfo.imax;
    IdxT jmax = m_meshdata.m_meshinfo.jmax;
    IdxT kmax = m_meshdata.m_meshinfo.kmax;
    IdxT ilen = m_meshdata.m_meshinfo.ilen;
    IdxT ijlen = m_meshdata.m_meshinfo.ijlen;

    // faces
    pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_j_kmin, jmin, jmax, imin, imax, indexer_ji(ilen,  kmin * ijlen));
    pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_j_kmax, jmin, jmax, imin, imax, indexer_ji(ilen,  (kmax-1) * ijlen));
    pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_jmin_k, kmin, kmax, imin, imax, indexer_ki(ijlen, jmin * ilen));
    pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_jmax_k, kmin, kmax, imin, imax, indexer_ki(ijlen, (jmax-1) * ilen));
    pack_face(policy_face{}, m_meshdata.data(), m_send.face_imin_j_k, kmin, kmax, jmin, jmax, indexer_kj(ijlen, ilen, imin));
    pack_face(policy_face{}, m_meshdata.data(), m_send.face_imax_j_k, kmin, kmax, jmin, jmax, indexer_kj(ijlen, ilen, imax-1));

    // edges
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmin_kmin, imin, imax, indexer_i(jmin * ilen     + kmin * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmax_kmin, imin, imax, indexer_i((jmax-1) * ilen + kmin * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmin_kmax, imin, imax, indexer_i(jmin * ilen     + (kmax-1) * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmax_kmax, imin, imax, indexer_i((jmax-1) * ilen + (kmax-1) * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_j_kmin, jmin, jmax, indexer_j(ilen, imin     + kmin * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_j_kmin, jmin, jmax, indexer_j(ilen, (imax-1) + kmin * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_j_kmax, jmin, jmax, indexer_j(ilen, imin     + (kmax-1) * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_j_kmax, jmin, jmax, indexer_j(ilen, (imax-1) + (kmax-1) * ijlen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_jmin_k, kmin, kmax, indexer_k(ijlen, imin     + jmin * ilen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_jmin_k, kmin, kmax, indexer_k(ijlen, (imax-1) + jmin * ilen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_jmax_k, kmin, kmax, indexer_k(ijlen, imin     + (jmax-1) * ilen));
    pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_jmax_k, kmin, kmax, indexer_k(ijlen, (imax-1) + (jmax-1) * ilen));

    // corners
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmin_kmin, indexer_(imin     + jmin * ilen     + kmin * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmin_kmin, indexer_((imax-1) + jmin * ilen     + kmin * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmax_kmin, indexer_(imin     + (jmax-1) * ilen + kmin * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmax_kmin, indexer_((imax-1) + (jmax-1) * ilen + kmin * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmin_kmax, indexer_(imin     + jmin * ilen     + (kmax-1) * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmin_kmax, indexer_((imax-1) + jmin * ilen     + (kmax-1) * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmax_kmax, indexer_(imin     + (jmax-1) * ilen + (kmax-1) * ijlen));
    pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmax_kmax, indexer_((imax-1) + (jmax-1) * ilen + (kmax-1) * ijlen));
  }

  void waitRecv()
  {
    //printf("waiting receives\n"); fflush(stdout);
    IdxT imin = m_meshdata.m_meshinfo.imin;
    IdxT jmin = m_meshdata.m_meshinfo.jmin;
    IdxT kmin = m_meshdata.m_meshinfo.kmin;
    IdxT imax = m_meshdata.m_meshinfo.imax;
    IdxT jmax = m_meshdata.m_meshinfo.jmax;
    IdxT kmax = m_meshdata.m_meshinfo.kmax;
    IdxT ilen = m_meshdata.m_meshinfo.ilen;
    IdxT ijlen = m_meshdata.m_meshinfo.ijlen;

    // faces
    unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_j_kmin, jmin, jmax, imin, imax, indexer_ji(ilen,  (kmin-1) * ijlen));
    unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_j_kmax, jmin, jmax, imin, imax, indexer_ji(ilen,  kmax * ijlen));
    unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_jmin_k, kmin, kmax, imin, imax, indexer_ki(ijlen, (jmin-1) * ilen));
    unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_jmax_k, kmin, kmax, imin, imax, indexer_ki(ijlen, jmax * ilen));
    unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_imin_j_k, kmin, kmax, jmin, jmax, indexer_kj(ijlen, ilen, imin-1));
    unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_imax_j_k, kmin, kmax, jmin, jmax, indexer_kj(ijlen, ilen, imax));

    // edges
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmin_kmin, imin, imax, indexer_i((jmin-1) * ilen + (kmin-1) * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmax_kmin, imin, imax, indexer_i(jmax * ilen     + (kmin-1) * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmin_kmax, imin, imax, indexer_i((jmin-1) * ilen + kmax * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmax_kmax, imin, imax, indexer_i(jmax * ilen     + kmax * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_j_kmin, jmin, jmax, indexer_j(ilen, (imin-1) + (kmin-1) * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_j_kmin, jmin, jmax, indexer_j(ilen, imax     + (kmin-1) * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_j_kmax, jmin, jmax, indexer_j(ilen, (imin-1) + kmax * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_j_kmax, jmin, jmax, indexer_j(ilen, imax     + kmax * ijlen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_jmin_k, kmin, kmax, indexer_k(ijlen, (imin-1) + (jmin-1) * ilen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_jmin_k, kmin, kmax, indexer_k(ijlen, imax     + (jmin-1) * ilen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_jmax_k, kmin, kmax, indexer_k(ijlen, (imin-1) + jmax * ilen));
    unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_jmax_k, kmin, kmax, indexer_k(ijlen, imax     + jmax * ilen));

    // corners
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmin_kmin, indexer_((imin-1) + (jmin-1) * ilen + (kmin-1) * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmin_kmin, indexer_(imax     + (jmin-1) * ilen + (kmin-1) * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmax_kmin, indexer_((imin-1) + jmax * ilen     + (kmin-1) * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmax_kmin, indexer_(imax     + jmax * ilen     + (kmin-1) * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmin_kmax, indexer_((imin-1) + (jmin-1) * ilen + kmax * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmin_kmax, indexer_(imax     + (jmin-1) * ilen + kmax * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmax_kmax, indexer_((imin-1) + jmax * ilen     + kmax * ijlen));
    unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmax_kmax, indexer_(imax     + jmax * ilen     + kmax * ijlen));
    
    m_recv.deallocate();
  }

  void waitSend()
  {
    //printf("posting sends\n"); fflush(stdout);
    m_send.deallocate();
  }

  ~Comm()
  {
  }
};

struct Timer {

  struct TimePoint {
    using type = std::chrono::high_resolution_clock::time_point;
    type tp;
  };

  TimePoint* times;
  const char** names;
  IdxT idx;
  IdxT size;

  Timer(IdxT size_)
    : times(nullptr)
    , names(nullptr)
    , idx(0)
    , size(0)
  {
    resize(size_);
  }

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  void start(const char* str) {
    if (idx < size) {
      record_time();
      names[idx] = str;
      ++idx;
    }
  }

  void restart(const char* str) {
    if (idx < size) {
      start(str);
    } else if (idx == size) {
      stop();
    }
  }

  void stop() {
    if (idx <= size) {
      record_time();
      if (idx < size) names[idx] = nullptr;
      ++idx;
    }
  }

  void print(const char* prefix = "") {
    for (IdxT i = 1; i < idx; ++i) {
      if (names[i-1] != nullptr) {
        std::chrono::duration<double> time_s = times[i].tp - times[i-1].tp;
        printf("%s%s: %.9f s\n", prefix, names[i-1], time_s.count());
      }
    }
    fflush(stdout);
  }

  void resize(IdxT size_) {
    clean();
    times = (TimePoint*)malloc((size_+1)*sizeof(TimePoint));
    names = (const char**)malloc(size_*sizeof(const char*));
    size = size_;
  }

  void clear() {
    for (IdxT i = 0; i < idx; ++i) {
      times[i].~TimePoint();
    }
    idx = 0;
  }

  void clean() {
    clear();
    if (times != nullptr) { free(times); times = nullptr; }
    if (names != nullptr) { free(names); names = nullptr; }
  }

  ~Timer() { clean(); }
private:
  void record_time() {
      new(&times[idx]) TimePoint{std::chrono::high_resolution_clock::now()};
  }
};

struct Range {
  static const uint32_t green    = 0x0000FF00;
  static const uint32_t red      = 0x00FF0000;
  static const uint32_t blue     = 0x000000FF;
  static const uint32_t yellow   = 0x00FFFF00;
  static const uint32_t cyan     = 0x0000FFFF;
  static const uint32_t magenta  = 0x00FF00FF;
  static const uint32_t orange   = 0x00FFA500;
  static const uint32_t pink     = 0x00FF69B4;

  const char* name;
  nvtxRangeId_t id;

  Range(const char* name_, uint32_t color)
    : name(nullptr)
  {
    start(name_, color);
  }

  void start(const char* name_, uint32_t color) {
    if (name_ != nullptr) {
      nvtxEventAttributes_t eventAttrib = {0};
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = color;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = name_;
      id = nvtxRangeStartEx(&eventAttrib);
      name = name_;
    }
  }

  void stop()
  {
    if(name != nullptr) {
      nvtxRangeEnd(id);
      name = nullptr;
    }
  }

  void restart(const char* name_, uint32_t color) {
    stop();
    start(name_, color);
  }

  ~Range()
  {
    stop();
  }
};

      struct set_n1 {
         DataT* data;
         set_n1(DataT* data_) : data(data_) {}
         HOST DEVICE
         void operator()(IdxT i, IdxT) const {
           IdxT zone = i;
           //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
           data[zone] = -1.0;
         }
      };

      struct set_1 {
         IdxT ilen, ijlen;
         DataT* data;
         set_1(IdxT ilen_, IdxT ijlen_, DataT* data_) : ilen(ilen_), ijlen(ijlen_), data(data_) {}
         HOST DEVICE
         void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
           IdxT zone = i + j * ilen + k * ijlen;
           //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
           data[zone] = 1.0;
         }
      };

      struct reset_1 {
         IdxT ilen, ijlen;
         DataT* data;
         IdxT imin, jmin, kmin;
         IdxT imax, jmax, kmax;
         reset_1(IdxT ilen_, IdxT ijlen_, DataT* data_, IdxT imin_, IdxT jmin_, IdxT kmin_, IdxT imax_, IdxT jmax_, IdxT kmax_)
           : ilen(ilen_), ijlen(ijlen_), data(data_)
           , imin(imin_), jmin(jmin_), kmin(kmin_)
           , imax(imax_), jmax(jmax_), kmax(kmax_)
         {}
         HOST DEVICE
         void operator()(IdxT k, IdxT j, IdxT i, IdxT idx) const {
           IdxT zone = i + j * ilen + k * ijlen;
           DataT expected, found, next;
           if (k >= kmin && k < kmax &&
               j >= jmin && j < jmax &&
               i >= imin && i < imax) {
             expected = 1.0; found = data[zone]; next = 1.0;
           } else {
             expected = 0.0; found = data[zone]; next = -1.0;
           }
           //if (found != expected) printf("zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
           //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
           data[zone] = next;
         }
      };

template < typename pol_loop, typename pol_corner, typename pol_edge, typename pol_face >
void do_cycles(const char* name, MeshInfo& mesh, IdxT ncycles, Allocator& aloc_mesh, Allocator& aloc_corner, Allocator& aloc_edge, Allocator& aloc_face, Timer& tm)
{
    tm.clear();

    printf("Starting test %s\n", name); fflush(stdout);

    Range r0(name, Range::orange);

    tm.start("start-up");

    MeshData var(mesh, aloc_mesh);
    Comm<pol_corner, pol_edge, pol_face> comm(var, aloc_corner, aloc_edge, aloc_face);

    {
      var.allocate();
      DataT* data = var.data();
      IdxT ijklen = mesh.ijklen;

      for_all(pol_loop{}, 0, ijklen,
                          set_n1(data));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

    }

    tm.stop();

    for(IdxT cycle = 0; cycle < ncycles; cycle++) {

      Range r1("cycle", Range::yellow);

      IdxT imin = mesh.imin;
      IdxT jmin = mesh.jmin;
      IdxT kmin = mesh.kmin;
      IdxT imax = mesh.imax;
      IdxT jmax = mesh.jmax;
      IdxT kmax = mesh.kmax;
      IdxT ilen = mesh.ilen;
      IdxT jlen = mesh.jlen;
      IdxT klen = mesh.klen;
      IdxT ijlen = mesh.ijlen;
      
      DataT* data = var.data();
      
      Range r2("pre-comm", Range::red);
      tm.start("pre-comm");

      for_all_3d(pol_loop{}, kmin, kmax,
                             jmin, jmax,
                             imin, imax,
                             set_1(ilen, ijlen, data));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.restart("post-recv", Range::pink);
      tm.start("post-recv");
      
      comm.postRecv();

      tm.stop();
      r2.restart("post-send", Range::pink);
      tm.start("post-send");

      comm.postSend();

      if (pol_corner::async || pol_edge::async || pol_face::async) {cudaCheck(cudaDeviceSynchronize());}

/*      for_all_3d(pol_loop{}, 0, klen,
                            0, jlen,
                            0, ilen,
                            [=] (IdxT k, IdxT j, IdxT i, IdxT idx) {
        IdxT zone = i + j * ilen + k * ijlen;
        DataT expected, found, next;
        if (k >= kmin && k < kmax &&
            j >= jmin && j < jmax &&
            i >= imin && i < imax) {
          expected = 1.0; found = data[zone]; next = 1.0;
        } else {
          expected = -1.0; found = data[zone]; next = -1.0;
        }
        if (found != expected) printf("zone %i(%i %i %i) = %f expected %f\n", zone, i, j, k, found, expected);
        //printf("%p[%i] = %f\n", data, zone, 1.0); fflush(stdout);
        data[zone] = next;
      });
*/

      tm.stop();
      r2.restart("wait-send", Range::pink);
      tm.start("wait-send");

      comm.waitSend();

      if (pol_corner::async || pol_edge::async || pol_face::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.restart("wait-recv", Range::pink);
      tm.start("wait-recv");

      comm.waitRecv();

      tm.stop();
      r2.restart("post-comm", Range::red);
      tm.start("post-comm");

      for_all_3d(pol_loop{}, 0, klen,
                             0, jlen,
                             0, ilen,
                             reset_1(ilen, ijlen, data, imin, jmin, kmin, imax, jmax, kmax));

      if (pol_loop::async) {cudaCheck(cudaDeviceSynchronize());}

      tm.stop();
      r2.stop();

    }

    tm.print();
    tm.clear();
}
 

int main(int argc, char** argv)
{
  int required = MPI_THREAD_SINGLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, required, &provided);

  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  if (required != provided) {
    fprintf(stderr, "Didn't receive MPI thread support required %i provided %i.\n", required, provided); fflush(stderr);
    MPI_Abort(mpi_comm, 1);
  }

  int comm_rank = -1;
  MPI_Comm_rank(mpi_comm, &comm_rank);
  int comm_size = 0;
  MPI_Comm_size(mpi_comm, &comm_size);

  if (comm_rank == 0) {
    printf("Started rank %i of %i\n", comm_rank, comm_size); fflush(stdout);
  }

  cudaCheck(cudaDeviceSynchronize());  

  IdxT isize = 0;
  IdxT jsize = 0;
  IdxT ksize = 0;

  if (argc == 1) {
    isize = 100;
    jsize = 100;
    ksize = 100;
  } else if (argc == 2) {
    isize = static_cast<IdxT>(atoll(argv[1]));
    jsize = isize;
    ksize = isize;
  } else if (argc == 4) {
    isize = static_cast<IdxT>(atoll(argv[1]));
    jsize = static_cast<IdxT>(atoll(argv[2]));
    ksize = static_cast<IdxT>(atoll(argv[3]));
  } else {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid arguments.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (isize <= 0 || jsize <= 0 || ksize <= 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Invalid size arguments.\n"); fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MeshInfo mesh(isize, jsize, ksize);
    
  if (comm_rank == 0) {
    printf("Mesh info\n");
    printf("%i %i %i\n", mesh.isize, mesh.jsize, mesh.ksize);
    printf("ij %i ik %i jk %i\n", mesh.ijsize, mesh.iksize, mesh.jksize);
    printf("ijk %i\n", mesh.ijksize);
    printf("i %8i %8i %8i %8i\n", 0, mesh.imin, mesh.imax, mesh.ilen);
    printf("j %8i %8i %8i %8i\n", 0, mesh.jmin, mesh.jmax, mesh.jlen);
    printf("k %8i %8i %8i %8i\n", 0, mesh.kmin, mesh.kmax, mesh.klen);
    printf("ij %i ik %i jk %i\n", mesh.ijlen, mesh.iklen, mesh.jklen);
    printf("ijk %i\n", mesh.ijklen);
    fflush(stdout);
  }
  
  IdxT ncycles = 5;

  Timer tm(1024);

  // warm-up memory pools
  {
    printf("Starting up memory pools\n"); fflush(stdout);

    Range r("Memmory pool init", Range::green);

    void* var0;
    void* var1;
 
    tm.start("host");

    var0 = host_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = host_alloc.allocate(mesh.ijksize*sizeof(DataT));
    host_alloc.deallocate(var0);
    host_alloc.deallocate(var1);

    tm.restart("host_pinned");

    var0 = hostpinned_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = hostpinned_alloc.allocate(mesh.ijksize*sizeof(DataT));
    hostpinned_alloc.deallocate(var0);
    hostpinned_alloc.deallocate(var1);

    tm.restart("device");

    var0 = device_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = device_alloc.allocate(mesh.ijksize*sizeof(DataT));
    device_alloc.deallocate(var0);
    device_alloc.deallocate(var1);

    tm.restart("managed");

    var0 = managed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    var1 = managed_alloc.allocate(mesh.ijksize*sizeof(DataT));
    managed_alloc.deallocate(var0);
    managed_alloc.deallocate(var1);

    tm.stop();

    tm.print();
    tm.clear();

  }

  // host allocated
  do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>("Host_seq", mesh, ncycles, host_alloc, host_alloc, host_alloc, host_alloc, tm);
 
  // host pinned allocated
  do_cycles<seq_pol, seq_pol, seq_pol, seq_pol>("hostpinned_seq", mesh, ncycles, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  do_cycles<cuda_pol, seq_pol, seq_pol, seq_pol>("hostpinned_cuda_seq", mesh, ncycles, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>("hostpinned_cuda", mesh, ncycles, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  // device allocated
  do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>("device_hostpinned_cuda", mesh, ncycles, device_alloc, hostpinned_alloc, hostpinned_alloc, hostpinned_alloc, tm);

  do_cycles<cuda_pol, cuda_pol, cuda_pol, cuda_pol>("device_cuda", mesh, ncycles, device_alloc, device_alloc, device_alloc, device_alloc, tm);

  MPI_Finalize();
  return 0;
}

