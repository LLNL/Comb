#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>

#ifndef _COMM_CUH
#define _COMM_CUH

#include "memory.cuh"
#include "for_all.cuh"
#include "mesh.cuh"

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

namespace detail {

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
    detail::pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_j_kmin, jmin, jmax, imin, imax, detail::indexer_ji(ilen,  kmin * ijlen));
    detail::pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_j_kmax, jmin, jmax, imin, imax, detail::indexer_ji(ilen,  (kmax-1) * ijlen));
    detail::pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_jmin_k, kmin, kmax, imin, imax, detail::indexer_ki(ijlen, jmin * ilen));
    detail::pack_face(policy_face{}, m_meshdata.data(), m_send.face_i_jmax_k, kmin, kmax, imin, imax, detail::indexer_ki(ijlen, (jmax-1) * ilen));
    detail::pack_face(policy_face{}, m_meshdata.data(), m_send.face_imin_j_k, kmin, kmax, jmin, jmax, detail::indexer_kj(ijlen, ilen, imin));
    detail::pack_face(policy_face{}, m_meshdata.data(), m_send.face_imax_j_k, kmin, kmax, jmin, jmax, detail::indexer_kj(ijlen, ilen, imax-1));

    // edges
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmin_kmin, imin, imax, detail::indexer_i(jmin * ilen     + kmin * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmax_kmin, imin, imax, detail::indexer_i((jmax-1) * ilen + kmin * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmin_kmax, imin, imax, detail::indexer_i(jmin * ilen     + (kmax-1) * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_i_jmax_kmax, imin, imax, detail::indexer_i((jmax-1) * ilen + (kmax-1) * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_j_kmin, jmin, jmax, detail::indexer_j(ilen, imin     + kmin * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_j_kmin, jmin, jmax, detail::indexer_j(ilen, (imax-1) + kmin * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_j_kmax, jmin, jmax, detail::indexer_j(ilen, imin     + (kmax-1) * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_j_kmax, jmin, jmax, detail::indexer_j(ilen, (imax-1) + (kmax-1) * ijlen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_jmin_k, kmin, kmax, detail::indexer_k(ijlen, imin     + jmin * ilen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_jmin_k, kmin, kmax, detail::indexer_k(ijlen, (imax-1) + jmin * ilen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imin_jmax_k, kmin, kmax, detail::indexer_k(ijlen, imin     + (jmax-1) * ilen));
    detail::pack_edge(policy_edge{}, m_meshdata.data(), m_send.edge_imax_jmax_k, kmin, kmax, detail::indexer_k(ijlen, (imax-1) + (jmax-1) * ilen));

    // corners
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmin_kmin, detail::indexer_(imin     + jmin * ilen     + kmin * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmin_kmin, detail::indexer_((imax-1) + jmin * ilen     + kmin * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmax_kmin, detail::indexer_(imin     + (jmax-1) * ilen + kmin * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmax_kmin, detail::indexer_((imax-1) + (jmax-1) * ilen + kmin * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmin_kmax, detail::indexer_(imin     + jmin * ilen     + (kmax-1) * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmin_kmax, detail::indexer_((imax-1) + jmin * ilen     + (kmax-1) * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imin_jmax_kmax, detail::indexer_(imin     + (jmax-1) * ilen + (kmax-1) * ijlen));
    detail::pack_corner(policy_corner{}, m_meshdata.data(), m_send.corner_imax_jmax_kmax, detail::indexer_((imax-1) + (jmax-1) * ilen + (kmax-1) * ijlen));
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
    detail::unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_j_kmin, jmin, jmax, imin, imax, detail::indexer_ji(ilen,  (kmin-1) * ijlen));
    detail::unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_j_kmax, jmin, jmax, imin, imax, detail::indexer_ji(ilen,  kmax * ijlen));
    detail::unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_jmin_k, kmin, kmax, imin, imax, detail::indexer_ki(ijlen, (jmin-1) * ilen));
    detail::unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_i_jmax_k, kmin, kmax, imin, imax, detail::indexer_ki(ijlen, jmax * ilen));
    detail::unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_imin_j_k, kmin, kmax, jmin, jmax, detail::indexer_kj(ijlen, ilen, imin-1));
    detail::unpack_face(policy_face{}, m_meshdata.data(), m_recv.face_imax_j_k, kmin, kmax, jmin, jmax, detail::indexer_kj(ijlen, ilen, imax));

    // edges
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmin_kmin, imin, imax, detail::indexer_i((jmin-1) * ilen + (kmin-1) * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmax_kmin, imin, imax, detail::indexer_i(jmax * ilen     + (kmin-1) * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmin_kmax, imin, imax, detail::indexer_i((jmin-1) * ilen + kmax * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_i_jmax_kmax, imin, imax, detail::indexer_i(jmax * ilen     + kmax * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_j_kmin, jmin, jmax, detail::indexer_j(ilen, (imin-1) + (kmin-1) * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_j_kmin, jmin, jmax, detail::indexer_j(ilen, imax     + (kmin-1) * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_j_kmax, jmin, jmax, detail::indexer_j(ilen, (imin-1) + kmax * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_j_kmax, jmin, jmax, detail::indexer_j(ilen, imax     + kmax * ijlen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_jmin_k, kmin, kmax, detail::indexer_k(ijlen, (imin-1) + (jmin-1) * ilen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_jmin_k, kmin, kmax, detail::indexer_k(ijlen, imax     + (jmin-1) * ilen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imin_jmax_k, kmin, kmax, detail::indexer_k(ijlen, (imin-1) + jmax * ilen));
    detail::unpack_edge(policy_edge{}, m_meshdata.data(), m_recv.edge_imax_jmax_k, kmin, kmax, detail::indexer_k(ijlen, imax     + jmax * ilen));

    // corners
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmin_kmin, detail::indexer_((imin-1) + (jmin-1) * ilen + (kmin-1) * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmin_kmin, detail::indexer_(imax     + (jmin-1) * ilen + (kmin-1) * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmax_kmin, detail::indexer_((imin-1) + jmax * ilen     + (kmin-1) * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmax_kmin, detail::indexer_(imax     + jmax * ilen     + (kmin-1) * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmin_kmax, detail::indexer_((imin-1) + (jmin-1) * ilen + kmax * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmin_kmax, detail::indexer_(imax     + (jmin-1) * ilen + kmax * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imin_jmax_kmax, detail::indexer_((imin-1) + jmax * ilen     + kmax * ijlen));
    detail::unpack_corner(policy_corner{}, m_meshdata.data(), m_recv.corner_imax_jmax_kmax, detail::indexer_(imax     + jmax * ilen     + kmax * ijlen));
    
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

#endif // _COMM_CUH

