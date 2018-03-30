
#ifndef _COMM_CUH
#define _COMM_CUH

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <list>
#include <vector>
#include <utility>

#include <mpi.h>

#include "memory.cuh"
#include "for_all.cuh"
#include "mesh.cuh"


struct CartComm
{
  MPI_Comm comm;
  int rank;
  int size;
  int coords[3];
  int cuts[3];
  int periodic[3];
  explicit CartComm()
    : comm(MPI_COMM_NULL)
    , rank(-1)
    , size(0)
    , coords{-1, -1, -1}
    , cuts{1, 1, 1}
    , periodic{0, 0, 0}
  {
  }
  
  void create()
  {
    MPI_Cart_create(MPI_COMM_WORLD, 3, cuts, periodic, 1, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, 3, coords);
  }
  
  int get_rank(const int arg_coords[])
  {
    int output_rank = -1;
    int input_coords[3] {-1, -1, -1};
    for(IdxT dim = 0; dim < 3; ++dim) {
      if ((0 <= arg_coords[dim] && arg_coords[dim] < cuts[dim]) || periodic[dim]) {
        input_coords[dim] = arg_coords[dim] % cuts[dim];
        if (input_coords[dim] < 0) input_coords[dim] += cuts[dim];
      }
    }
    if (input_coords[0] != -1 && input_coords[1] != -1 && input_coords[2] != -1) {
      MPI_Cart_rank(comm, input_coords, &output_rank);
    }
    return output_rank;
  }
  
  void disconnect()
  {
    MPI_Comm_disconnect(&comm);
  }
};

struct CommInfo
{
  int rank;
  int size;
  
  CartComm cart;
  
  bool mock_communication;
  
  enum struct method : IdxT
  { any
  , some
  , all };
  
  static const char* method_str(method m)
  {
    const char* str = "unknown";
    switch (m) {
      case method::any:  str = "a";    break;
      case method::some: str = "some"; break;
      case method::all:  str = "all";  break;
    }
    return str;
  }
  
  method send_method;
  method recv_method;
  
  CommInfo()
    : rank(-1)
    , size(0)
    , cart()
    , mock_communication(false)
    , send_method(method::all)
    , recv_method(method::all)
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }
    
  void barrier()
  {
    if (cart.comm != MPI_COMM_NULL) {
      MPI_Barrier(cart.comm);
    } else {
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  
  template < typename ... Ts >
  void print_any(const char* fmt, Ts&&... args)
  {
    fprintf(stdout, fmt, std::forward<Ts>(args)...); fflush(stdout);
  }
  
  template < typename ... Ts >
  void print_master(const char* fmt, Ts&&... args)
  {
    if (rank == 0) {
      print_any(fmt, std::forward<Ts>(args)...);
    }
  }
  
  template < typename ... Ts >
  void warn_any(const char* fmt, Ts&&... args)
  {
    fprintf(stderr, fmt, std::forward<Ts>(args)...); fflush(stderr);
  }
  
  template < typename ... Ts >
  void warn_master(const char* fmt, Ts&&... args)
  {
    if (rank == 0) {
      warn_any(fmt, std::forward<Ts>(args)...);
    }
  }
  
  template < typename ... Ts >
  void abort_any(const char* fmt, Ts&&... args)
  {
    warn_any(fmt, std::forward<Ts>(args)...);
    abort();
  }
  
  template < typename ... Ts >
  void abort_master(const char* fmt, Ts&&... args)
  {
    warn_master(fmt, std::forward<Ts>(args)...);
    abort();
  }
  
  void abort()
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
};

namespace detail {

struct indexer_kji {
  IdxT ijlen, ilen;
  indexer_kji(IdxT ijlen_, IdxT ilen_) : ijlen(ijlen_), ilen(ilen_) {}
  HOST DEVICE IdxT operator()(IdxT k, IdxT j, IdxT i, IdxT) const { return i + j * ilen + k * ijlen; }
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

struct indexer_list_idx {
  LidxT* indices;
  indexer_list_idx(LidxT* indices_) : indices(indices_) {}
  HOST DEVICE IdxT operator()(IdxT, IdxT idx) const { return indices[idx]; }
  HOST DEVICE IdxT operator()(IdxT, IdxT, IdxT idx) const { return indices[idx]; }
  HOST DEVICE IdxT operator()(IdxT, IdxT, IdxT, IdxT idx) const { return indices[idx]; }
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

template < typename I_src, typename T_dst, typename I_dst >
struct set_idxr_idxr {
  T_dst* ptr_dst;
  I_src idxr_src;
  I_dst idxr_dst;
  set_idxr_idxr(I_src const& idxr_src_, T_dst* const& ptr_dst_, I_dst const& idxr_dst_) : idxr_src(idxr_src_), ptr_dst(ptr_dst_), idxr_dst(idxr_dst_) {}
  template < typename ... Ts >
  HOST DEVICE void operator()(Ts... args) const { ptr_dst[idxr_dst(args...)] = idxr_src(args...); }
};

template < typename I_src, typename T_dst, typename I_dst >
set_idxr_idxr<I_src, T_dst, I_dst> make_set_idxr_idxr(I_src const& idxr_src, T_dst* const& ptr_dst, I_dst const& idxr_dst) {
  return set_idxr_idxr<I_src, T_dst, I_dst>(idxr_src, ptr_dst, idxr_dst);
}

} // namespace detail


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
  IdxT imin, jmin, kmin, imax, jmax, kmax;
  MeshData const& mesh;
  Box3d(MeshData const& mesh_,
        IdxT imin_, IdxT imax_,
        IdxT jmin_, IdxT jmax_,
        IdxT kmin_, IdxT kmax_)
    : imin(imin_)
    , jmin(jmin_)
    , kmin(kmin_)
    , imax(imax_)
    , jmax(jmax_)
    , kmax(kmax_)
    , mesh(mesh_)
  {
    // printf("Box3d i %d %d j %d %d k %d %d\n", imin, imax, jmin, jmax, kmin, kmax);
  }
  size_t size() const
  {
    return (imax - imin) * (jmax - jmin) * (kmax - kmin);
  }
  template < typename policy >
  LidxT* get_indices(policy const& p) const
  {
    LidxT* index_list = (LidxT*)mesh.aloc.allocate(size()*sizeof(LidxT));
    for_all_3d(p, imin, imax, jmin, jmax, kmin, kmax, make_set_idxr_idxr(detail::indexer_kji{mesh.info.ijlen, mesh.info.ilen}, index_list, detail::indexer_idx{}));
    return index_list;
  }
  void deallocate_indices(LidxT* ptr) const
  {
    mesh.aloc.deallocate(ptr);
  }
};

struct Box3dTemplate
{
  using IT = IdxTemplate;
  using ITL = IdxTemplate::location;
  static constexpr Box3dTemplate make_Box3dTemplate_inner(IdxT width, ITL bounds[])
  {
    return Box3dTemplate{ IT{(bounds[0] == ITL::mid) ? ITL::min : bounds[0], (bounds[0] == ITL::max) ? -width : 0 }
                        , IT{(bounds[0] == ITL::mid) ? ITL::max : bounds[0], (bounds[0] == ITL::min) ?  width : 0 }
                        , IT{(bounds[1] == ITL::mid) ? ITL::min : bounds[1], (bounds[1] == ITL::max) ? -width : 0 }
                        , IT{(bounds[1] == ITL::mid) ? ITL::max : bounds[1], (bounds[1] == ITL::min) ?  width : 0 }
                        , IT{(bounds[2] == ITL::mid) ? ITL::min : bounds[2], (bounds[2] == ITL::max) ? -width : 0 }
                        , IT{(bounds[2] == ITL::mid) ? ITL::max : bounds[2], (bounds[2] == ITL::min) ?  width : 0 } };
  }
  static constexpr Box3dTemplate make_Box3dTemplate_ghost(IdxT width, IT::location bounds[])
  {
    return Box3dTemplate{ IT{(bounds[0] == ITL::mid) ? ITL::min : bounds[0], (bounds[0] == ITL::min) ? -width : 0 }
                        , IT{(bounds[0] == ITL::mid) ? ITL::max : bounds[0], (bounds[0] == ITL::max) ?  width : 0 }
                        , IT{(bounds[1] == ITL::mid) ? ITL::min : bounds[1], (bounds[1] == ITL::min) ? -width : 0 }
                        , IT{(bounds[1] == ITL::mid) ? ITL::max : bounds[1], (bounds[1] == ITL::max) ?  width : 0 }
                        , IT{(bounds[2] == ITL::mid) ? ITL::min : bounds[2], (bounds[2] == ITL::min) ? -width : 0 }
                        , IT{(bounds[2] == ITL::mid) ? ITL::max : bounds[2], (bounds[2] == ITL::max) ?  width : 0 } };
  }

  IdxTemplate imin, jmin, kmin, imax, jmax, kmax;
  constexpr Box3dTemplate(IdxTemplate imin_, IdxTemplate imax_,
                          IdxTemplate jmin_, IdxTemplate jmax_,
                          IdxTemplate kmin_, IdxTemplate kmax_)
    : imin(imin_)
    , jmin(jmin_)
    , kmin(kmin_)
    , imax(imax_)
    , jmax(jmax_)
    , kmax(kmax_)
  {
  }
  Box3d make_box(MeshData const& mesh)
  {
    return Box3d{mesh, I_idx(mesh.info, imin), I_idx(mesh.info, imax),
                       J_idx(mesh.info, jmin), J_idx(mesh.info, jmax),
                       K_idx(mesh.info, kmin), K_idx(mesh.info, kmax)};
  }
private:
  IdxT I_idx(MeshInfo const& info, IdxTemplate it)
  {
    IdxT idx = 0;
    switch (it.idx) {
      case ITL::min:
        idx = info.imin; break;
      case ITL::max:
        idx = info.imax; break;
      default:
        assert(0); break;
    }
    return idx + it.offset;
  }
  IdxT J_idx(MeshInfo const& info, IdxTemplate it)
  {
    IdxT idx = 0;
    switch (it.idx) {
      case ITL::min:
        idx = info.jmin; break;
      case ITL::max:
        idx = info.jmax; break;
      default:
        assert(0); break;
    }
    return idx + it.offset;
  }
  IdxT K_idx(MeshInfo const& info, IdxTemplate it)
  {
    IdxT idx = 0;
    switch (it.idx) {
      case ITL::min:
        idx = info.kmin; break;
      case ITL::max:
        idx = info.kmax; break;
      default:
        assert(0); break;
    }
    return idx + it.offset;
  }
};

template < typename policy0, typename policy1, typename policy2 >
struct Message
{
  Allocator& buf_aloc;
  int dest_rank;
  IdxT pol;
  DataT* m_buf;
  size_t m_size;
  
  using list_item_type = std::pair<Box3d, LidxT*>;
  std::list<list_item_type> boxes;

  Message(int dest_rank_, Allocator& buf_aloc_, IdxT pol_)
    : buf_aloc(buf_aloc_)
    , dest_rank(dest_rank_)
    , pol(pol_)
    , m_buf(nullptr)
    , m_size(0)
  {

  }
  
  DataT* buffer()
  {
    return m_buf;
  }
  
  size_t size() const
  {
    return m_size;
  }
  
  void add(Box3d const& box)
  {
    LidxT* indices = nullptr;
    switch (pol) {
      case 0:
        indices = box.get_indices(policy0{}); break;
      case 1:
        indices = box.get_indices(policy1{}); break;
      case 2:
        indices = box.get_indices(policy2{}); break;
      default:
        assert(0); break;
    }
    boxes.push_back(list_item_type{box, indices});
    m_size += box.size();
  }
  
  void pack()
  {
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(boxes);
    for (auto i = std::begin(boxes); i != end; ++i) {
      IdxT len = i->first.size();
      LidxT* indices = i->second;
      DataT* data = i->first.mesh.data();
      switch (pol) {
        case 0:
          for_all(policy0{}, 0, len, make_copy_idxr_idxr(data, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{})); break;
        case 1:
          for_all(policy1{}, 0, len, make_copy_idxr_idxr(data, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{})); break;
        case 2:
          for_all(policy2{}, 0, len, make_copy_idxr_idxr(data, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{})); break;
        default:
          assert(0); break;
      }
      buf += len;
    }
  }
  
  void unpack()
  {
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(boxes);
    for (auto i = std::begin(boxes); i != end; ++i) {
      IdxT len = i->first.size();
      LidxT* indices = i->second;
      DataT* data = i->first.mesh.data();
      switch (pol) {
        case 0:
          for_all(policy0{}, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, data, detail::indexer_list_idx{indices})); break;
        case 1:
          for_all(policy1{}, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, data, detail::indexer_list_idx{indices})); break;
        case 2:
          for_all(policy2{}, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, data, detail::indexer_list_idx{indices})); break;
        default:
          assert(0); break;
      }
      buf += len;
    }
  }

  void allocate()
  {
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(size()*sizeof(DataT));
    }
  }

  void deallocate()
  {
    if (m_buf != nullptr) {
      buf_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }
 
  ~Message()
  {
    deallocate();
    auto end = std::end(boxes);
    for (auto i = std::begin(boxes); i != end; ++i) {
      i->first.deallocate_indices(i->second);
    }
  }
};

template < typename policy_face, typename policy_edge, typename policy_corner >
struct Comm
{
  Allocator& face_aloc;
  Allocator& edge_aloc;
  Allocator& corner_aloc;
  
  CommInfo comminfo;
  
  using message_type = Message<policy_face, policy_edge, policy_corner>;
  std::vector<message_type> m_sends;
  std::vector<message_type> m_recvs;
  
  IdxT num_face_neighbors;
  IdxT num_edge_neighbors;
  IdxT num_corner_neighbors;
  
  Comm(CommInfo const& comminfo_, Allocator& face_aloc_, Allocator& edge_aloc_, Allocator& corner_aloc_)
    : face_aloc(face_aloc_)
    , edge_aloc(edge_aloc_)
    , corner_aloc(corner_aloc_)
    , comminfo(comminfo_)
    , num_face_neighbors(0)
    , num_edge_neighbors(0)
    , num_corner_neighbors(0)
  {
    // initialize messages and neighbor information
    // The msg param in for_*_neighbors depends on 
    // num_face_neighbors, num_edge_neighbors, num_corners_neighbors
    // but should be correct in this initialization ordering
    for_face_neighbors( [&](IdxT msg, const int neighbor_coords[]) {
      num_face_neighbors++;
      add_message(neighbor_coords, face_aloc, 0);
    });
    
    for_edge_neighbors( [&](IdxT msg, const int neighbor_coords[]) {
      num_edge_neighbors++;
      add_message(neighbor_coords, edge_aloc, 1);
    });
    
    for_corner_neighbors( [&](IdxT msg, const int neighbor_coords[]) {
      num_corner_neighbors++;
      add_message(neighbor_coords, corner_aloc, 2);
    });
  }
  
  void add_message(const int coords[], Allocator& aloc, IdxT policy)
  {
    int neighbor_rank = comminfo.cart.get_rank(coords);
    assert(0 <= neighbor_rank);
    m_sends.emplace_back(neighbor_rank, face_aloc, 0);
    m_recvs.emplace_back(neighbor_rank, face_aloc, 0);
  }
  
  template < typename loop_body >
  void for_face_neighbors(loop_body&& body)
  {
    IdxT msg = 0;
    bool active[3] {false, false, false};
    for (IdxT dim = 2; dim >= 0; --dim) {
      active[dim] = true;
      IdxT off[1];
      for (off[0] = -1; off[0] <= 1; off[0] += 2) {
        IdxT off_idx = 0;
        int neighbor_coords[3] { comminfo.cart.coords[0], comminfo.cart.coords[1], comminfo.cart.coords[2] } ;
        if (active[0]) { neighbor_coords[0] += off[off_idx++]; }
        if (active[1]) { neighbor_coords[1] += off[off_idx++]; }
        if (active[2]) { neighbor_coords[2] += off[off_idx++]; }
        if (((0 <= neighbor_coords[0] && neighbor_coords[0] < comminfo.cart.cuts[0]) || comminfo.cart.periodic[0]) && 
            ((0 <= neighbor_coords[1] && neighbor_coords[1] < comminfo.cart.cuts[1]) || comminfo.cart.periodic[1]) && 
            ((0 <= neighbor_coords[2] && neighbor_coords[2] < comminfo.cart.cuts[2]) || comminfo.cart.periodic[2]) ) {
          body(msg++, neighbor_coords);
        }
      }
      active[dim] = false;
    }
  }
  
  template < typename loop_body >
  void for_edge_neighbors(loop_body&& body)
  {
    IdxT msg = num_face_neighbors;
    bool active[3] {true, true, true};
    for (IdxT dim = 0; dim < 3; ++dim) {
      active[dim] = false;
      IdxT off[2];
      for (off[1] = -1; off[1] <= 1; off[1] += 2) {
        for (off[0] = -1; off[0] <= 1; off[0] += 2) {
          IdxT off_idx = 0;
          int neighbor_coords[3] { comminfo.cart.coords[0], comminfo.cart.coords[1], comminfo.cart.coords[2] } ;
          if (active[0]) { neighbor_coords[0] += off[off_idx++]; }
          if (active[1]) { neighbor_coords[1] += off[off_idx++]; }
          if (active[2]) { neighbor_coords[2] += off[off_idx++]; }
          if (((0 <= neighbor_coords[0] && neighbor_coords[0] < comminfo.cart.cuts[0]) || comminfo.cart.periodic[0]) && 
              ((0 <= neighbor_coords[1] && neighbor_coords[1] < comminfo.cart.cuts[1]) || comminfo.cart.periodic[1]) && 
              ((0 <= neighbor_coords[2] && neighbor_coords[2] < comminfo.cart.cuts[2]) || comminfo.cart.periodic[2]) ) {
            body(msg++, neighbor_coords);
          }
        }
      }
      active[dim] = true;
    }
  }
  
  template < typename loop_body >
  void for_corner_neighbors(loop_body&& body)
  {
    IdxT msg = num_face_neighbors + num_edge_neighbors;
    bool active[3] {true, true, true};
    IdxT off[3];
    for (off[2] = -1; off[2] <= 1; off[2] += 2) {
      for (off[1] = -1; off[1] <= 1; off[1] += 2) {
        for (off[0] = -1; off[0] <= 1; off[0] += 2) {
          IdxT off_idx = 0;
          int neighbor_coords[3] { comminfo.cart.coords[0], comminfo.cart.coords[1], comminfo.cart.coords[2] } ;
          if (active[0]) { neighbor_coords[0] += off[off_idx++]; }
          if (active[1]) { neighbor_coords[1] += off[off_idx++]; }
          if (active[2]) { neighbor_coords[2] += off[off_idx++]; }
          if (((0 <= neighbor_coords[0] && neighbor_coords[0] < comminfo.cart.cuts[0]) || comminfo.cart.periodic[0]) && 
              ((0 <= neighbor_coords[1] && neighbor_coords[1] < comminfo.cart.cuts[1]) || comminfo.cart.periodic[1]) && 
              ((0 <= neighbor_coords[2] && neighbor_coords[2] < comminfo.cart.cuts[2]) || comminfo.cart.periodic[2]) ) {
            body(msg++, neighbor_coords);
          }
        }
      }
    }
  }
  
  void add_box(IdxT msg, MeshData& mesh, const IdxT bounds[])
  {
    using IT = IdxTemplate;
    using ITL = IdxTemplate::location;
    ITL locs[3] { (bounds[0] == -1) ? ITL::min : ( (bounds[0] == 1) ? ITL::max : ITL::mid )
                , (bounds[1] == -1) ? ITL::min : ( (bounds[1] == 1) ? ITL::max : ITL::mid )
                , (bounds[2] == -1) ? ITL::min : ( (bounds[2] == 1) ? ITL::max : ITL::mid ) };
    assert(msg < m_sends.size());
    m_sends[msg].add(Box3dTemplate::make_Box3dTemplate_inner(mesh.info.ghost_width, locs).make_box(mesh));
    assert(msg < m_recvs.size());
    m_recvs[msg].add(Box3dTemplate::make_Box3dTemplate_ghost(mesh.info.ghost_width, locs).make_box(mesh));
  }
  
  void add_var(MeshData& mesh)
  {
    // edges
    for_face_neighbors( [&](IdxT msg, const int neighbor_coords[]) {
      int bounds[3] { neighbor_coords[0] - comminfo.cart.coords[0]
                    , neighbor_coords[1] - comminfo.cart.coords[1]
                    , neighbor_coords[2] - comminfo.cart.coords[2] };
      add_box(msg, mesh, bounds);
    });
    
    for_edge_neighbors( [&](IdxT msg, const int neighbor_coords[]) {
      int bounds[3] { neighbor_coords[0] - comminfo.cart.coords[0]
                    , neighbor_coords[1] - comminfo.cart.coords[1]
                    , neighbor_coords[2] - comminfo.cart.coords[2] };
      add_box(msg, mesh, bounds);
    });
    
    for_corner_neighbors( [&](IdxT msg, const int neighbor_coords[]) {
      int bounds[3] { neighbor_coords[0] - comminfo.cart.coords[0]
                    , neighbor_coords[1] - comminfo.cart.coords[1]
                    , neighbor_coords[2] - comminfo.cart.coords[2] };
      add_box(msg, mesh, bounds);
    });
  }

  void postRecv()
  {
    //printf("posting receives\n"); fflush(stdout);
    auto end = std::end(m_recvs);
    for (auto i = std::begin(m_recvs); i != end; ++i) {
      i->allocate();
    }
  }

  void postSend()
  {
    //printf("posting sends\n"); fflush(stdout);
    auto end = std::end(m_sends);
    for (auto i = std::begin(m_sends); i != end; ++i) {
      i->allocate();
      i->pack();
      // do send
    }
  }

  void waitRecv()
  {
    //printf("waiting receives\n"); fflush(stdout);
    auto end = std::end(m_recvs);
    for (auto i = std::begin(m_recvs); i != end; ++i) {
      // do recv
      i->unpack();
      i->deallocate();
    }
  }

  void waitSend()
  {
    //printf("posting sends\n"); fflush(stdout);
    auto end = std::end(m_sends);
    for (auto i = std::begin(m_sends); i != end; ++i) {
      i->deallocate();
    }
  }

  ~Comm()
  {
  }
};

#endif // _COMM_CUH

