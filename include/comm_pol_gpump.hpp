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

#ifndef _COMM_POL_GPUMP_HPP
#define _COMM_POL_GPUMP_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_GPUMP

#include <exception>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <map>

#include "for_all.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"
#include "utils_gpump.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct GpumpRequest
{
  int status;
  struct gpump* g;
  int partner_rank;
  ContextEnum context_type;
  union context_union {
    int invalid;
    CPUContext cpu;
    CudaContext cuda;
    context_union() : invalid(-1) {}
    ~context_union() {}
  } context;

  GpumpRequest()
    : status(0)
    , g(nullptr)
    , partner_rank(-1)
    , context_type(ContextEnum::invalid)
    , context()
  {

  }

  GpumpRequest(GpumpRequest const& other)
    : status(other.status)
    , g(other.g)
    , partner_rank(other.partner_rank)
    , context_type(ContextEnum::invalid)
    , context()
  {
    copy_context(other.context_type, other.context);
  }

  GpumpRequest& operator=(GpumpRequest const& other)
  {
    status = other.status;
    g = other.g;
    partner_rank = other.partner_rank;
    copy_context(other.context_type, other.context);
    return *this;
  }

  ~GpumpRequest()
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

struct gpump_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = false;
  static const char* get_name() { return "gpump"; }
  using send_request_type = GpumpRequest;
  using recv_request_type = GpumpRequest;
  using send_status_type = int;
  using recv_status_type = int;
};

template < >
struct CommContext<gpump_pol> : CudaContext
{
  using base = CudaContext;

  using pol = gpump_pol;

  using send_request_type = typename pol::send_request_type;
  using recv_request_type = typename pol::recv_request_type;
  using send_status_type = typename pol::send_status_type;
  using recv_status_type = typename pol::recv_status_type;

  struct gpump* g;

  CommContext()
    : base()
    , g(nullptr)
  { }

  CommContext(base const& b)
    : base(b)
    , g(nullptr)
  { }

  CommContext(CommContext const& a_, MPI_Comm comm_)
    : base(a_)
    , g(detail::gpump::init(comm_))
  { }

  ~CommContext()
  {
    if (g != nullptr) {
      detail::gpump::term(g); g = nullptr;
    }
  }

  void ensure_waitable()
  {

  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  send_request_type send_request_null() { return send_request_type{}; }
  recv_request_type recv_request_null() { return recv_request_type{}; }
  send_status_type send_status_null() { return 0; }
  recv_status_type recv_status_null() { return 0; }


  void connect_ranks(std::vector<int> const& send_ranks,
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
      detail::gpump::connect_propose(g, rank);
    }
    for (int rank : ranks) {
      detail::gpump::connect_accept(g, rank);
    }
  }

  void disconnect_ranks(std::vector<int> const& send_ranks,
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
      detail::gpump::disconnect(g, rank);
    }
  }
};

struct gpump_mempool
{
  struct ibv_ptr
  {
    struct ibv_mr* mr = nullptr;
    size_t offset = 0;
    void* ptr = nullptr;
  };

  ibv_ptr allocate(struct gpump* g, COMB::Allocator& aloc_in, size_t size)
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
          info.ptr.mr = detail::gpump::register_region(g, info.ptr.ptr, info.size);
          info.ptr.offset = 0;
          used_ptrs.emplace(info.ptr.ptr, info);
        }

        ptr = info.ptr;
      } else {
        throw std::invalid_argument("unknown allocator passed to gpump_mempool::allocate");
      }
    }
    return ptr;
  }

  void deallocate(struct gpump* g, COMB::Allocator& aloc_in, ibv_ptr ptr)
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
          throw std::invalid_argument("unknown or unused pointer passed to gpump_mempool::deallocate");
        }
      } else {
        throw std::invalid_argument("unknown allocator passed to gpump_mempool::deallocate");
      }
    }
  }

  void add_allocator(struct gpump* g, COMB::Allocator& aloc)
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

  void remove_allocators(struct gpump* g)
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

        detail::gpump::deregister_region(this->g, info.ptr.mr);
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

  struct gpump* g = nullptr;
  std::unordered_map<COMB::Allocator*, ptr_map> m_allocators;
};


template < >
struct Message<gpump_pol> : detail::MessageBase
{
  using base = detail::MessageBase;

  using policy_comm = gpump_pol;
  using communicator_type = CommContext<policy_comm>;
  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;
  using send_status_type  = typename policy_comm::send_status_type;
  using recv_status_type  = typename policy_comm::recv_status_type;

  using region_type = gpump_mempool::ibv_ptr;
  static inline gpump_mempool& get_mempool()
  {
    static gpump_mempool mempool;
    return mempool;
  }

  static void setup_mempool(communicator_type& con_comm,
                            COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    get_mempool().add_allocator(con_comm.g, many_aloc);
    get_mempool().add_allocator(con_comm.g, few_aloc);
  }

  static void teardown_mempool(communicator_type& con_comm)
  {
    get_mempool().remove_allocators(con_comm.g);
  }


  Message(Kind _kind, int partner_rank, int tag, bool have_many)
    : base(_kind, partner_rank, tag, have_many)
    , m_region()
  { }

  ~Message()
  { }


  template < typename context >
  void pack(context& con, communicator_type& con_comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FGPRINTF(FileGroup::proc, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
      con.for_all(0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      buf += len;
    }
  }

  template < typename context >
  void unpack(context& con, communicator_type& con_comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FGPRINTF(FileGroup::proc, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
      con.for_all(0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      buf += len;
    }
  }

private:
  void start_Isend(CPUContext const&, communicator_type& con_comm)
  {
    detail::gpump::isend(con_comm.g, partner_rank(), m_region.mr, m_region.offset, nbytes());
  }

  void start_Isend(CudaContext const& con, communicator_type& con_comm)
  {
    detail::gpump::stream_send(con_comm.g, partner_rank(), con.stream(), m_region.mr, m_region.offset, nbytes());
  }

public:

  template < typename context >
  void Isend(context& con, communicator_type& con_comm, send_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FGPRINTF(FileGroup::proc, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());

    start_Isend(con, con_comm);
    request->status = 1;
    request->g = con_comm.g;
    request->partner_rank = partner_rank();
    request->setContext(con);
  }

private:
  static void cork_Isends(CPUContext const&, communicator_type& con_comm)
  {
    COMB::ignore_unused(con_comm);
  }

  static void cork_Isends(CudaContext const&, communicator_type& con_comm)
  {
    detail::gpump::cork(con_comm.g);
  }

  static void uncork_Isends(CPUContext const&, communicator_type& con_comm)
  {
    COMB::ignore_unused(con_comm);
  }

  static void uncork_Isends(CudaContext const&, communicator_type& con_comm)
  {
    detail::gpump::uncork(con_comm.g, con_comm.stream());
  }

public:
  template < typename context >
  static void start_Isends(context& con, communicator_type& con_comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FGPRINTF(FileGroup::proc, "start_Isends\n");

    cork_Isends(con, con_comm);
  }

  template < typename context >
  static void finish_Isends(context& con, communicator_type& con_comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FGPRINTF(FileGroup::proc, "finish_Isends\n");

    uncork_Isends(con, con_comm);
  }

  template < typename context >
  void Irecv(context& con, communicator_type& con_comm, recv_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FGPRINTF(FileGroup::proc, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());

    detail::gpump::receive(con_comm.g, partner_rank(), m_region.mr, m_region.offset, nbytes());
    request->status = -1;
    request->g = con_comm.g;
    request->partner_rank = partner_rank();
    request->setContext(con);
  }


  template < typename context >
  void allocate(context&, communicator_type& con_comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    if (m_buf == nullptr) {
      m_region = get_mempool().allocate(con_comm.g, buf_aloc, nbytes());
      m_buf = (DataT*)m_region.ptr;
    }
  }

private:
  void wait_send(CPUContext const&, communicator_type& con_comm)
  {
    // already done
  }

  void wait_send(CudaContext const& con, communicator_type& con_comm)
  {
    detail::gpump::wait_send_complete(con_comm.g, partner_rank());
  }

  void wait_recv(CPUContext const&, communicator_type& con_comm)
  {
    // already done
  }

  void wait_recv(CudaContext const& con, communicator_type& con_comm)
  {
    detail::gpump::wait_receive_complete(con_comm.g, partner_rank());
  }

public:
  template < typename context >
  void deallocate(context& con, communicator_type& con_comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    if (m_buf != nullptr) {

      if (m_kind == Kind::send) {
        wait_send(con, con_comm);
      } else if (m_kind == Kind::recv) {
        wait_recv(con, con_comm);
      }
      get_mempool().deallocate(con_comm.g, buf_aloc, m_region);
      m_region = region_type{};
      m_buf = nullptr;
    }
  }


private:
  static bool start_wait_send(communicator_type&,
                              send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      detail::gpump::stream_wait_send_complete(request.g, request.partner_rank, request.context.cuda.stream());
      return true;
    } else if (request.context_type == ContextEnum::cpu) {
      detail::gpump::cpu_ack_isend(request.g, request.partner_rank);
      return detail::gpump::is_send_complete(request.g, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

  static bool test_waiting_send(communicator_type&,
                                send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      assert(0);
    } else if (request.context_type == ContextEnum::cpu) {
      return detail::gpump::is_send_complete(request.g, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

public:
  static int test_send_any(communicator_type& con_comm,
                           int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != 3) {
        if (requests[i].status == 1) {
          // haven't seen this request yet, start it
          if (start_wait_send(con_comm, requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            requests[i].status = 2;
            continue;
          }
        } else if (requests[i].status == 2) {
          // have seen this one before, test it
          if (test_waiting_send(con_comm, requests[i])) {
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

  static int wait_send_any(communicator_type& con_comm,
                           int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    int ready = -1;
    do {
      ready = test_send_any(con_comm, count, requests, statuses);
    } while (ready == -1);
    return ready;
  }

  static int test_send_some(communicator_type& con_comm,
                            int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != 3) {
        if (requests[i].status == 1) {
          // haven't seen this request yet, start it
          if (start_wait_send(con_comm, requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            requests[i].status = 2;
            continue;
          }
        } else if (requests[i].status == 2) {
          // have seen this one before, test it
          if (test_waiting_send(con_comm, requests[i])) {
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

  static int wait_send_some(communicator_type& con_comm,
                            int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    int done = 0;
    do {
      done = test_send_some(con_comm, count, requests, indices, statuses);
    } while (done == 0);
    return done;
  }

  static bool test_send_all(communicator_type& con_comm,
                            int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != 3) {
        if (requests[i].status == 1) {
          // haven't seen this request yet, start it
          if (start_wait_send(con_comm, requests[i])) {
            // done
            requests[i].status = 3;
          } else {
            // test again later
            requests[i].status = 2;
            continue;
          }
        } else if (requests[i].status == 2) {
          // have seen this one before, test it
          if (test_waiting_send(con_comm, requests[i])) {
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

  static void wait_send_all(communicator_type& con_comm,
                            int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    bool done = false;
    do {
      done = test_send_all(con_comm, count, requests, statuses);
    } while (!done);
  }


private:
  static bool start_wait_recv(communicator_type&,
                              recv_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      detail::gpump::stream_wait_recv_complete(request.g, request.partner_rank, request.context.cuda.stream());
      return true;
    } else if (request.context_type == ContextEnum::cpu) {
      detail::gpump::cpu_ack_recv(request.g, request.partner_rank);
      return detail::gpump::is_receive_complete(request.g, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

  static bool test_waiting_recv(communicator_type&,
                                recv_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      assert(0);
    } else if (request.context_type == ContextEnum::cpu) {
      return detail::gpump::is_receive_complete(request.g, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

public:
  static int test_recv_any(communicator_type& con_comm,
                           int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != -3) {
        if (requests[i].status == -1) {
          // haven't seen this request yet, start it
          if (start_wait_recv(con_comm, requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            requests[i].status = -2;
            continue;
          }
        } else if (requests[i].status == -2) {
          // have seen this one before, test it
          if (test_waiting_recv(con_comm, requests[i])) {
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

  static int wait_recv_any(communicator_type& con_comm,
                           int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    int ready = -1;
    do {
      ready = test_recv_any(con_comm, count, requests, statuses);
    } while (ready == -1);
    return ready;
  }

  static int test_recv_some(communicator_type& con_comm,
                            int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != -3) {
        if (requests[i].status == -1) {
          // haven't seen this request yet, start it
          if (start_wait_recv(con_comm, requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            requests[i].status = -2;
            continue;
          }
        } else if (requests[i].status == -2) {
          // have seen this one before, test it
          if (test_waiting_recv(con_comm, requests[i])) {
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

  static int wait_recv_some(communicator_type& con_comm,
                            int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    int done = 0;
    do {
      done = test_recv_some(con_comm, count, requests, indices, statuses);
    } while (done == 0);
    return done;
  }

  static bool test_recv_all(communicator_type& con_comm,
                            int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i].status != -3) {
        if (requests[i].status == -1) {
          // haven't seen this request yet, start it
          if (start_wait_recv(con_comm, requests[i])) {
            // done
            requests[i].status = -3;
          } else {
            // test again later
            requests[i].status = -2;
            continue;
          }
        } else if (requests[i].status == -2) {
          // have seen this one before, test it
          if (test_waiting_recv(con_comm, requests[i])) {
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

  static void wait_recv_all(communicator_type& con_comm,
                            int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    bool done = false;
    do {
      done = test_recv_all(con_comm, count, requests, statuses);
    } while (!done);
  }

private:
  region_type m_region;
};

#endif // COMB_ENABLE_GPUMP

#endif // _COMM_POL_GPUMP_HPP
