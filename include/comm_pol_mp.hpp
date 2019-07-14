//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#ifndef _COMM_POL_MP_HPP
#define _COMM_POL_MP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MP

#include <exception>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <map>

#include "for_all.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"
#include "utils_mp.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct MpRequest
{
  int status;
  struct mp* comm;
  int partner_rank;
  ContextEnum context_type;
  union context_union {
    int invalid;
    CPUContext cpu;
    CudaContext cuda;
    context_union() : invalid(-1) {}
    ~context_union() {}
  } context;

  MpRequest()
    : status(0)
    , comm(nullptr)
    , partner_rank(-1)
    , context_type(ContextEnum::invalid)
    , context()
  {

  }

  MpRequest(MpRequest const& other)
    : status(other.status)
    , comm(other.comm)
    , partner_rank(other.partner_rank)
    , context_type(ContextEnum::invalid)
    , context()
  {
    copy_context(other.context_type, other.context);
  }

  MpRequest& operator=(MpRequest const& other)
  {
    status = other.status;
    comm = other.comm;
    partner_rank = other.partner_rank;
    copy_context(other.context_type, other.context);
    return *this;
  }

  ~MpRequest()
  {
    destroy_context();
  }

  void setContext(CPUContext const& con)
  {
    if (context_type == ContextEnum::cpu) {
      context.cpu = con;
    } else {
      destroy_context();
      new(&context.cpu) CPUContext(con);
      context_type = ContextEnum::cpu;
    }
  }

  void setContext(CudaContext const& con)
  {
    if (context_type == ContextEnum::cuda) {
      context.cuda = con;
    } else {
      destroy_context();
      new(&context.cuda) CudaContext(con);
      context_type = ContextEnum::cuda;
    }
  }

private:
  void copy_context(ContextEnum const& other_type, context_union const& other_context)
  {
    switch (other_type) {
      case (ContextEnum::invalid):
      {
        // do nothing
      } break;
      case (ContextEnum::cpu):
      {
        setContext(other_context.cpu);
      } break;
      case (ContextEnum::cuda):
      {
        setContext(other_context.cuda);
      } break;
    }
  }

  void destroy_context()
  {
    switch (context_type) {
      case (ContextEnum::invalid):
      {
        // do nothing
      } break;
      case (ContextEnum::cpu):
      {
        context.cpu.~CPUContext();
      } break;
      case (ContextEnum::cuda):
      {
        context.cuda.~CudaContext();
      } break;
    }
    context_type = ContextEnum::invalid;
  }
};

struct mp_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = false;
  static const char* get_name() { return "mp"; }
  using communicator_type = struct mp*;
  static inline communicator_type communicator_create(MPI_Comm comm) { return detail::mp::init(comm); }
  static inline void communicator_destroy(communicator_type g) { detail::mp::term(g); }
  using send_request_type = MpRequest;
  static inline send_request_type send_request_null() { return send_request_type{}; }
  using recv_request_type = MpRequest;
  static inline recv_request_type recv_request_null() { return recv_request_type{}; }
  using send_status_type = int;
  static inline send_status_type send_status_null() { return 0; }
  using recv_status_type = int;
  static inline recv_status_type recv_status_null() { return 0; }
};

template < >
struct CommContext<mp_pol> : CudaContext
{
  using base = CudaContext;
  CommContext()
    : base()
  { }
  CommContext(base const& b)
    : base(b)
  { }
};


inline void connect_ranks(mp_pol const&,
                          mp_pol::communicator_type comm,
                          std::vector<int> const& send_ranks,
                          std::vector<int> const& recv_ranks)
{
  std::set<int> ranks;
  for (int rank : send_ranks) {
    if (ranks.find(rank) == ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : recv_ranks) {
    if (ranks.find(rank) == ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : ranks) {
    detail::mp::connect_propose(comm, rank);
  }
  for (int rank : ranks) {
    detail::mp::connect_accept(comm, rank);
  }
}

inline void disconnect_ranks(mp_pol const&,
                             mp_pol::communicator_type comm,
                             std::vector<int> const& send_ranks,
                             std::vector<int> const& recv_ranks)
{
  std::set<int> ranks;
  for (int rank : send_ranks) {
    if (ranks.find(rank) != ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : recv_ranks) {
    if (ranks.find(rank) != ranks.end()) {
      ranks.insert(rank);
    }
  }
  for (int rank : ranks) {
    detail::mp::disconnect(comm, rank);
  }
}

struct mp_mempool
{
  using communicator_type = typename mp_pol::communicator_type;
  struct ibv_ptr
  {
    struct ibv_mr* mr = nullptr;
    size_t offset = 0;
    void* ptr = nullptr;
  };

  ibv_ptr allocate(struct mp* g, COMB::Allocator& aloc_in, size_t size)
  {
    assert(g == this->g);
    ibv_ptr ptr{};
    if (size > 0) {
      size = std::max(size, sizeof(std::max_align_t));

      auto iter = m_allocators.find(&aloc_in);
      if (iter != m_allocators.end()) {
        COMB::Allocator& aloc = *iter->first;
        used_ptr_map&    used_ptrs = iter->second.used;
        unused_ptr_map&  unused_ptrs = iter->second.unused;

        ptr_info info{};

        auto unused_iter = unused_ptrs.find(size);
        if (unused_iter != unused_ptrs.end()) {
          // found an existing unused ptr
          info = unused_iter->second;
          unused_ptrs.erase(unused_iter);
          used_ptrs.emplace(info.ptr.ptr, info);
        } else {
          // allocate a new pointer for this size
          info.size = size;
          info.ptr.ptr = aloc.allocate(info.size);
          info.ptr.mr = detail::mp::register_region(g, info.ptr.ptr, info.size);
          info.ptr.offset = 0;
          used_ptrs.emplace(info.ptr.ptr, info);
        }

        ptr = info.ptr;
      } else {
        throw std::invalid_argument("unknown allocator passed to mp_mempool::allocate");
      }
    }
    return ptr;
  }

  void deallocate(struct mp* g, COMB::Allocator& aloc_in, ibv_ptr ptr)
  {
    assert(g == this->g);
    if (ptr.ptr != nullptr) {

      auto iter = m_allocators.find(&aloc_in);
      if (iter != m_allocators.end()) {
        COMB::Allocator& aloc = *iter->first;
        used_ptr_map&    used_ptrs = iter->second.used;
        unused_ptr_map&  unused_ptrs = iter->second.unused;

        auto used_iter = used_ptrs.find(ptr.ptr);
        if (used_iter != used_ptrs.end()) {
          // found an existing used ptr
          ptr_info info = used_iter->second;
          used_ptrs.erase(used_iter);
          unused_ptrs.emplace(info.size, info);
        } else {
          // unknown or unused pointer
          throw std::invalid_argument("unknown or unused pointer passed to mp_mempool::deallocate");
        }
      } else {
        throw std::invalid_argument("unknown allocator passed to mp_mempool::deallocate");
      }
    }
  }

  void add_allocator(struct mp* g, COMB::Allocator& aloc)
  {
    if (this->g == nullptr) {
      this->g = g;
    }
    assert(g == this->g);
    if (m_allocators.find(&aloc) == m_allocators.end()) {
      // new allocator
      m_allocators.emplace(&aloc, ptr_map{});
    }
  }

  void remove_allocators(struct mp* g)
  {
    assert(g == this->g);
    bool error = false;
    auto iter = m_allocators.begin();
    while (iter != m_allocators.end()) {
      COMB::Allocator& aloc = *iter->first;
      used_ptr_map&    used_ptrs = iter->second.used;
      unused_ptr_map&  unused_ptrs = iter->second.unused;

      auto inner_iter = unused_ptrs.begin();
      while (inner_iter != unused_ptrs.end()) {
        ptr_info& info = inner_iter->second;

        detail::mp::deregister_region(this->g, info.ptr.mr);
        aloc.deallocate(info.ptr.ptr);
        inner_iter = unused_ptrs.erase(inner_iter);
      }

      if (used_ptrs.empty()) {
        iter = m_allocators.erase(iter);
      } else {
        ++iter;
        error = true;
      }
    }

    if (error) throw std::logic_error("can not remove Allocator with used ptr");
    this->g = nullptr;
  }

private:
  struct ptr_info
  {
    ibv_ptr ptr{};
    size_t size = 0;
  };
  using used_ptr_map = std::unordered_map<void*, ptr_info>;
  using unused_ptr_map = std::multimap<size_t, ptr_info>;
  struct ptr_map
  {
    used_ptr_map used{};
    unused_ptr_map unused{};
  };

  struct mp* g = nullptr;
  std::unordered_map<COMB::Allocator*, ptr_map> m_allocators;
};


template < >
struct Message<mp_pol> : detail::MessageBase
{
  using base = detail::MessageBase;

  using policy_comm = mp_pol;
  using communicator_type = typename policy_comm::communicator_type;
  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;
  using send_status_type  = typename policy_comm::send_status_type;
  using recv_status_type  = typename policy_comm::recv_status_type;

  using region_type = mp_mempool::ibv_ptr;
  static inline mp_mempool& get_mempool()
  {
    static mp_mempool mempool;
    return mempool;
  }

  static void setup_mempool(communicator_type comm,
                            COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    get_mempool().add_allocator(comm, many_aloc);
    get_mempool().add_allocator(comm, few_aloc);
  }

  static void teardown_mempool(communicator_type comm)
  {
    get_mempool().remove_allocators(comm);
  }


  Message(Kind _kind, int partner_rank, int tag, bool have_many)
    : base(_kind, partner_rank, tag, have_many)
    , m_region()
  { }

  ~Message()
  { }


  template < typename context >
  void pack(context& con, communicator_type comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
      con.for_all(0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      buf += len;
    }
  }

  template < typename context >
  void unpack(context& con, communicator_type comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
      con.for_all(0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      buf += len;
    }
  }

private:
  void start_Isend(CPUContext const&, communicator_type comm)
  {
    detail::mp::isend(comm, partner_rank(), m_region.mr, m_region.offset, nbytes());
  }

  void start_Isend(CudaContext const& con, communicator_type comm)
  {
    detail::mp::stream_send(comm, partner_rank(), con.stream(), m_region.mr, m_region.offset, nbytes());
  }

public:
  template < typename context >
  void Isend(context& con, communicator_type comm, send_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());

    start_Isend(con, comm);
    request->status = 1;
    request->comm = comm;
    request->partner_rank = partner_rank();
    request->setContext(con);
  }

  template < typename context >
  void Irecv(context& con, communicator_type comm, recv_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());

    detail::mp::receive(comm, partner_rank(), m_region.mr, m_region.offset, nbytes());
    request->status = -1;
    request->comm = comm;
    request->partner_rank = partner_rank();
    request->setContext(con);
  }


  template < typename context >
  void allocate(context&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    if (m_buf == nullptr) {
      m_region = get_mempool().allocate(comm, buf_aloc, nbytes());
      m_buf = (DataT*)m_region.ptr;
    }
  }

private:
  void wait_send(CPUContext const&, communicator_type comm)
  {
    // already done
  }

  void wait_send(CudaContext const& con, communicator_type comm)
  {
    detail::mp::wait_send_complete(comm, partner_rank());
  }

  void wait_recv(CPUContext const&, communicator_type comm)
  {
    // already done
  }

  void wait_recv(CudaContext const& con, communicator_type comm)
  {
    detail::mp::wait_receive_complete(comm, partner_rank());
  }

public:
  template < typename context >
  void deallocate(context& con, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "mp_pol does not support mpi_type_pol");
    if (m_buf != nullptr) {

      if (m_kind == Kind::send) {
        wait_send(con, comm);
      } else if (m_kind == Kind::recv) {
        wait_recv(con, comm);
      }
      get_mempool().deallocate(comm, buf_aloc, m_region);
      m_region = region_type{};
      m_buf = nullptr;
    }
  }


private:
  static bool start_wait_send(send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      detail::mp::stream_wait_send_complete(request.comm, request.partner_rank, request.context.cuda.stream());
      return true;
    } else if (request.context_type == ContextEnum::cpu) {
      detail::mp::cpu_ack_isend(request.comm, request.partner_rank);
      return detail::mp::is_send_complete(request.comm, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

  static bool test_waiting_send(send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      assert(0);
    } else if (request.context_type == ContextEnum::cpu) {
      return detail::mp::is_send_complete(request.comm, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

public:
  static int test_send_any(int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != 3) {
        if (requests[i].status == 1) {
          // haven't seen this request yet, start it
          if (start_wait_send(requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            requests[i].status = 2;
            continue;
          }
        } else if (requests[i].status == 2) {
          // have seen this one before, test it
          if (test_waiting_send(requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            continue;
          }
        } else {
          assert(0);
        }
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int wait_send_any(int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    int ready = -1;
    do {
      ready = test_send_any(count, requests, statuses);
    } while (ready == -1);
    return ready;
  }

  static int test_send_some(int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != 3) {
        if (requests[i].status == 1) {
          // haven't seen this request yet, start it
          if (start_wait_send(requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            requests[i].status = 2;
            continue;
          }
        } else if (requests[i].status == 2) {
          // have seen this one before, test it
          if (test_waiting_send(requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            continue;
          }
        } else {
          assert(0);
        }
        statuses[i] = 1;
        indices[done++] = i;
      }
    }
    return done;
  }

  static int wait_send_some(int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    int done = 0;
    do {
      done = test_send_some(count, requests, indices, statuses);
    } while (done == 0);
    return done;
  }

  static bool test_send_all(int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != 3) {
        if (requests[i].status == 1) {
          // haven't seen this request yet, start it
          if (start_wait_send(requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            requests[i].status = 2;
            continue;
          }
        } else if (requests[i].status == 2) {
          // have seen this one before, test it
          if (test_waiting_send(requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            continue;
          }
        } else {
          assert(0);
        }
        statuses[i] = 1;
      }
      done++;
    }
    return done == count;
  }

  static void wait_send_all(int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    bool done = false;
    do {
      done = test_send_all(count, requests, statuses);
    } while (!done);
  }


private:
  static bool start_wait_recv(recv_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      detail::mp::stream_wait_recv_complete(request.comm, request.partner_rank, request.context.cuda.stream());
      return true;
    } else if (request.context_type == ContextEnum::cpu) {
      detail::mp::cpu_ack_recv(request.comm, request.partner_rank);
      return detail::mp::is_receive_complete(request.comm, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

  static bool test_waiting_recv(recv_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      assert(0);
    } else if (request.context_type == ContextEnum::cpu) {
      return detail::mp::is_receive_complete(request.comm, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

public:
  static int test_recv_any(int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != -3) {
        if (requests[i].status == -1) {
          // haven't seen this request yet, start it
          if (start_wait_recv(requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            requests[i].status = -2;
            continue;
          }
        } else if (requests[i].status == -2) {
          // have seen this one before, test it
          if (test_waiting_recv(requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            continue;
          }
        } else {
          assert(0);
        }
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int wait_recv_any(int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    int ready = -1;
    do {
      ready = test_recv_any(count, requests, statuses);
    } while (ready == -1);
    return ready;
  }

  static int test_recv_some(int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != -3) {
        if (requests[i].status == -1) {
          // haven't seen this request yet, start it
          if (start_wait_recv(requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            requests[i].status = -2;
            continue;
          }
        } else if (requests[i].status == -2) {
          // have seen this one before, test it
          if (test_waiting_recv(requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            continue;
          }
        } else {
          assert(0);
        }
        statuses[i] = 1;
        indices[done++] = i;
      }
    }
    return done;
  }

  static int wait_recv_some(int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    int done = 0;
    do {
      done = test_recv_some(count, requests, indices, statuses);
    } while (done == 0);
    return done;
  }

  static bool test_recv_all(int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != -3) {
        if (requests[i].status == -1) {
          // haven't seen this request yet, start it
          if (start_wait_recv(requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            requests[i].status = -2;
            continue;
          }
        } else if (requests[i].status == -2) {
          // have seen this one before, test it
          if (test_waiting_recv(requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            continue;
          }
        } else {
          assert(0);
        }
        statuses[i] = 1;
      }
      done++;
    }
    return done == count;
  }

  static void wait_recv_all(int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    bool done = false;
    do {
      done = test_recv_all(count, requests, statuses);
    } while (!done);
  }

private:
  region_type m_region;
};

#endif // COMB_ENABLE_MP

#endif // _COMM_POL_MP_HPP
