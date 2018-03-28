
#ifndef _COMM_CUH
#define _COMM_CUH

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <list>
#include <vector>
#include <utility>

#include "memory.cuh"
#include "for_all.cuh"
#include "mesh.cuh"

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
  static const IdxT min = 0;
  static const IdxT max = 1;
  IdxT idx, offset;
  constexpr IdxTemplate(IdxT idx_, IdxT offset_) : idx(idx_), offset(offset_) {}
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
#define Box3dTemplate_all       IT{IT::min, 0},     IT{IT::max, 0}
#define Box3dTemplate_min       IT{IT::min, 0},     IT{IT::min, width}
#define Box3dTemplate_max       IT{IT::max,-width}, IT{IT::max, 0}
  static constexpr Box3dTemplate get_i_j_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_all,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_i_j_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_all,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_i_jmin_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_min,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_i_jmax_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_max,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_imin_j_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_all,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_imax_j_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_all,
                         Box3dTemplate_all};
  }
  
  static constexpr Box3dTemplate get_i_jmin_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_min,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_i_jmax_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_max,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_i_jmin_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_min,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_i_jmax_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_max,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_imin_j_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_all,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_imax_j_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_all,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_imin_j_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_all,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_imax_j_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_all,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_imin_jmin_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_min,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_imax_jmin_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_min,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_imin_jmax_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_max,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_imax_jmax_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_max,
                         Box3dTemplate_all};
  }
  
  static constexpr Box3dTemplate get_imin_jmin_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_min,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_imax_jmin_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_min,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_imin_jmax_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_max,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_imax_jmax_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_max,
                         Box3dTemplate_min};
  }
  static constexpr Box3dTemplate get_imin_jmin_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_min,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_imax_jmin_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_min,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_imin_jmax_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_min,
                         Box3dTemplate_max,
                         Box3dTemplate_max};
  }
  static constexpr Box3dTemplate get_imax_jmax_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_max,
                         Box3dTemplate_max,
                         Box3dTemplate_max};
  }
  
#undef Box3dTemplate_min
#undef Box3dTemplate_max

#define Box3dTemplate_ghost_min IT{IT::min,-width}, IT{IT::min, 0}
#define Box3dTemplate_ghost_max IT{IT::max, 0}, IT{IT::max, width}

  static constexpr Box3dTemplate get_ghost_i_j_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_all,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_i_j_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_all,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_i_jmin_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_ghost_i_jmax_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_ghost_imin_j_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_all,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_ghost_imax_j_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_all,
                         Box3dTemplate_all};
  }
  
  static constexpr Box3dTemplate get_ghost_i_jmin_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_i_jmax_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_i_jmin_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_i_jmax_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_all,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_imin_j_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_all,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_imax_j_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_all,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_imin_j_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_all,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_imax_j_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_all,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_imin_jmin_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_ghost_imax_jmin_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_ghost_imin_jmax_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_all};
  }
  static constexpr Box3dTemplate get_ghost_imax_jmax_k(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_all};
  }
  
  static constexpr Box3dTemplate get_ghost_imin_jmin_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_imax_jmin_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_imin_jmax_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_imax_jmax_kmin(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_min};
  }
  static constexpr Box3dTemplate get_ghost_imin_jmin_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_imax_jmin_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_imin_jmax_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_min,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_max};
  }
  static constexpr Box3dTemplate get_ghost_imax_jmax_kmax(IdxT width)
  {
    return Box3dTemplate{Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_max,
                         Box3dTemplate_ghost_max};
  }
  
#undef Box3dTemplate_all
#undef Box3dTemplate_ghost_min
#undef Box3dTemplate_ghost_max

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
      case IdxTemplate::min:
        idx = info.imin; break;
      case IdxTemplate::max:
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
      case IdxTemplate::min:
        idx = info.jmin; break;
      case IdxTemplate::max:
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
      case IdxTemplate::min:
        idx = info.kmin; break;
      case IdxTemplate::max:
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
  IdxT pol;
  DataT* m_buf;
  size_t m_size;
  
  using list_item_type = std::pair<Box3d, LidxT*>;
  std::list<list_item_type> boxes;

  Message(Allocator& buf_aloc_, IdxT pol_)
    : buf_aloc(buf_aloc_)
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
  
  using message_type = Message<policy_face, policy_edge, policy_corner>;
  std::vector<message_type> m_sends;
  std::vector<message_type> m_recvs;
  
  Comm(Allocator& face_aloc_, Allocator& edge_aloc_, Allocator& corner_aloc_)
    : face_aloc(face_aloc_)
    , edge_aloc(edge_aloc_)
    , corner_aloc(corner_aloc_)
  {
    m_sends.reserve(26);
    m_recvs.reserve(26);
    for(IdxT face_neighbor = 0; face_neighbor < 6; ++face_neighbor) {
      m_sends.emplace_back(face_aloc, 0);
      m_recvs.emplace_back(face_aloc, 0);
    }
    for(IdxT edge_neighbor = 0; edge_neighbor < 12; ++edge_neighbor) {
      m_sends.emplace_back(edge_aloc, 1);
      m_recvs.emplace_back(edge_aloc, 1);
    }
    for(IdxT corner_neighbor = 0; corner_neighbor < 8; ++corner_neighbor) {
      m_sends.emplace_back(corner_aloc, 2);
      m_recvs.emplace_back(corner_aloc, 2);
    }
  }
  
  void add_var(MeshData& mesh)
  {
    {
      IdxT idx = 0;
    
      // faces
      m_sends[idx++].add(Box3dTemplate::get_i_j_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_i_j_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_i_jmin_k(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_i_jmax_k(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_j_k(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_j_k(mesh.info.ghost_width).make_box(mesh));

      // edges
      m_sends[idx++].add(Box3dTemplate::get_i_jmin_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_i_jmax_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_i_jmin_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_i_jmax_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_j_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_j_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_j_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_j_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_jmin_k(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_jmin_k(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_jmax_k(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_jmax_k(mesh.info.ghost_width).make_box(mesh));
      
      // corners
      m_sends[idx++].add(Box3dTemplate::get_imin_jmin_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_jmin_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_jmax_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_jmax_kmin(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_jmin_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_jmin_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imin_jmax_kmax(mesh.info.ghost_width).make_box(mesh));
      m_sends[idx++].add(Box3dTemplate::get_imax_jmax_kmax(mesh.info.ghost_width).make_box(mesh));
    }
    
    {
      IdxT idx = 0;
    
      // faces
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_j_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_j_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_jmin_k(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_jmax_k(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_j_k(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_j_k(mesh.info.ghost_width).make_box(mesh));

      // edges
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_jmin_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_jmax_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_jmin_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_i_jmax_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_j_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_j_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_j_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_j_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_jmin_k(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_jmin_k(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_jmax_k(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_jmax_k(mesh.info.ghost_width).make_box(mesh));
      
      // corners
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_jmin_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_jmin_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_jmax_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_jmax_kmin(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_jmin_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_jmin_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imin_jmax_kmax(mesh.info.ghost_width).make_box(mesh));
      m_recvs[idx++].add(Box3dTemplate::get_ghost_imax_jmax_kmax(mesh.info.ghost_width).make_box(mesh));
    }
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

