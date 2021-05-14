//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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
#include <unordered_set>
#include <map>

#include "exec.hpp"
#include "comm_utils_gpump.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

namespace detail {

namespace gpump {

struct Request
{
  int status;
  struct gpump* g;
  int partner_rank;
  ContextEnum context_type;
  bool completed;
  union context_union {
    int invalid;
    CPUContext cpu;
    MPIContext mpi;
    CudaContext cuda;
    context_union() : invalid(-1) {}
    ~context_union() {}
  } context;

  Request()
    : status(0)
    , g(nullptr)
    , partner_rank(-1)
    , context_type(ContextEnum::invalid)
    , context()
    , completed(true)
  {

  }

  Request(Request const& other)
    : status(other.status)
    , g(other.g)
    , partner_rank(other.partner_rank)
    , context_type(ContextEnum::invalid)
    , context()
    , completed(other.completed)
  {
    copy_context(other.context_type, other.context);
  }

  Request& operator=(Request const& other)
  {
    status = other.status;
    g = other.g;
    partner_rank = other.partner_rank;
    copy_context(other.context_type, other.context);
    completed = other.completed;
    return *this;
  }

  ~Request()
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

  void setContext(MPIContext const& con)
  {
    if (context_type == ContextEnum::mpi) {
      context.mpi = con;
    } else {
      destroy_context();
      new(&context.mpi) MPIContext(con);
      context_type = ContextEnum::mpi;
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
      case (ContextEnum::mpi):
      {
        setContext(other_context.mpi);
      } break;
      case (ContextEnum::cuda):
      {
        setContext(other_context.cuda);
      } break;
      default:
      {
        assert(0);
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
      case (ContextEnum::mpi):
      {
        context.mpi.~MPIContext();
      } break;
      case (ContextEnum::cuda):
      {
        context.cuda.~CudaContext();
      } break;
      default:
      {
        assert(0);
      } break;
    }
    context_type = ContextEnum::invalid;
  }
};

struct mempool
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
        throw std::invalid_argument("unknown allocator passed to detail::gpump::mempool::allocate");
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
          throw std::invalid_argument("unknown or unused pointer passed to detail::gpump::mempool::deallocate");
        }
      } else {
        throw std::invalid_argument("unknown allocator passed to detail::gpump::mempool::deallocate");
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

} // namespace gpump

} // namespace detail

struct gpump_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = false;
  static const char* get_name() { return "gpump"; }
  using send_request_type = detail::gpump::Request*;
  using recv_request_type = detail::gpump::Request*;
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

  send_request_type send_request_null() { return nullptr; }
  recv_request_type recv_request_null() { return nullptr; }
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


  inline detail::gpump::mempool& get_mempool()
  {
    static detail::gpump::mempool mempool;
    return mempool;
  }

  void setup_mempool(COMB::Allocator& many_aloc,
                     COMB::Allocator& few_aloc)
  {
    get_mempool().add_allocator(this->g, many_aloc);
    get_mempool().add_allocator(this->g, few_aloc);
  }

  void teardown_mempool()
  {
    get_mempool().remove_allocators(this->g);
  }


  struct message_request_type
  {
    using region_type = detail::gpump::mempool::ibv_ptr;

    detail::MessageBase::Kind kind;
    region_type region;
    detail::gpump::Request request;

    message_request_type(detail::MessageBase::Kind kind_)
      : kind(kind_)
    { }
  };

  inline std::unordered_set<message_request_type*>& get_message_request_map()
  {
    static std::unordered_set<message_request_type*> messages;
    return messages;
  }

  // returns when msg has completed
  void wait_request(message_request_type* msg_request)
  {
    // loop over all messages to check if they are done
    // this allows all messages to make progress
    while (1) {
      for (message_request_type* other_msg_request : get_message_request_map()) {

        detail::MessageBase::Kind other_kind = other_msg_request->kind;
        detail::gpump::Request& other_request = other_msg_request->request;

        if (!other_request.completed && other_kind == msg_request->kind) {
          if (other_kind == detail::MessageBase::Kind::send) {
            other_request.completed = detail::gpump::is_send_complete(other_request.g, other_request.partner_rank);
          } else if (other_kind == detail::MessageBase::Kind::recv) {
            other_request.completed = detail::gpump::is_receive_complete(other_request.g, other_request.partner_rank);
          } else {
            assert(0 && (other_kind == detail::MessageBase::Kind::send || other_kind == detail::MessageBase::Kind::recv));
          }
        }

        if (other_msg_request == msg_request && other_request.completed) return;
      }
    }
  }
};


namespace detail {

template < >
struct Message<MessageBase::Kind::send, gpump_pol>
  : MessageInterface<MessageBase::Kind::send, gpump_pol>
{
  using base = MessageInterface<MessageBase::Kind::send, gpump_pol>;

  using policy_comm = typename base::policy_comm;
  using communicator_type = typename base::communicator_type;
  using request_type      = typename base::request_type;
  using status_type       = typename base::status_type;

  // use the base class constructor
  using base::base;


  static int test_send_any(communicator_type& con_comm,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      int status = handle_send_request(con_comm, requests[i]);
      if (status == 3) {
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int wait_send_any(communicator_type& con_comm,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    int ready = -1;
    do {
      ready = test_send_any(con_comm, count, requests, statuses);
    } while (ready == -1);
    return ready;
  }

  static int test_send_some(communicator_type& con_comm,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    int done = 0;
    if (count > 0) {
      bool new_requests = (requests[0]->status == 1);
      if (new_requests) {
        // detail::gpump::cork(con_comm.g);
      }
      for (int i = 0; i < count; ++i) {
        int status = handle_send_request(con_comm, requests[i]);
        if (status == 3) {
          statuses[i] = 1;
          indices[done++] = i;
        }
      }
      if (new_requests) {
        // detail::gpump::uncork(con_comm.g, con_comm.stream_launch());
      }
    }
    return done;
  }

  static int wait_send_some(communicator_type& con_comm,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    int done = 0;
    do {
      done = test_send_some(con_comm, count, requests, indices, statuses);
    } while (done == 0);
    return done;
  }

  static bool test_send_all(communicator_type& con_comm,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    int done = 0;
    if (count > 0) {
      bool new_requests = (requests[0]->status == 1);
      if (new_requests) {
        // detail::gpump::cork(con_comm.g);
      }
      for (int i = 0; i < count; ++i) {
        int status = handle_send_request(con_comm, requests[i]);
        if (status == 3) {
          statuses[i] = 1;
        }
        if (status == 3 || status == 4) {
          done++;
        }
      }
      if (new_requests) {
        // detail::gpump::uncork(con_comm.g, con_comm.stream_launch());
      }
    }
    return done == count;
  }

  static void wait_send_all(communicator_type& con_comm,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    bool done = false;
    do {
      done = test_send_all(con_comm, count, requests, statuses);
    } while (!done);
  }

private:
  static bool start_wait_send(communicator_type&,
                              request_type& request)
  {
    assert(!request->completed);
    bool done = false;
    if (request->context_type == ContextEnum::cuda) {
      detail::gpump::stream_wait_send_complete(request->g, request->partner_rank, request->context.cuda.stream_launch());
      done = true;
    } else if (request->context_type == ContextEnum::cpu) {
      detail::gpump::cpu_ack_isend(request->g, request->partner_rank);
    } else {
      assert(0 && (request->context_type == ContextEnum::cuda || request->context_type == ContextEnum::cpu));
    }
    return done;
  }

  static bool test_waiting_send(communicator_type&,
                                request_type& request)
  {
    assert(!request->completed);
    bool done = false;
    if (request->context_type == ContextEnum::cuda) {
      done = detail::gpump::is_send_complete(request->g, request->partner_rank);
      request->completed = done;
      // do one test to get things moving, then allow something else to be enqueued
      done = true;
    } else if (request->context_type == ContextEnum::cpu) {
      done = detail::gpump::is_send_complete(request->g, request->partner_rank);
      request->completed = done;
    } else {
      assert(0 && (request->context_type == ContextEnum::cuda || request->context_type == ContextEnum::cpu));
    }
    return done;
  }

  // possible status values
  // 0 - not ready to wait, not sent
  // 1 - ready to wait, first wait
  // 2 - ready to wait, waited before
  // 3 - ready, first ready
  // 4 - ready, ready before
  static int handle_send_request(communicator_type& con_comm,
                                 request_type& request)
  {
    if (request->status == 0) {
      // not sent
      assert(0 && (request->status != 0));
    } else if (request->status == 1) {
      // sent, start waiting
      if (start_wait_send(con_comm, request)) {
        // done
        request->status = 3;
      } else {
        // wait again later
        request->status = 2;
      }
    } else if (request->status == 2) {
      // still waiting, keep waiting
      if (test_waiting_send(con_comm, request)) {
        // done
        request->status = 3;
      }
    } else if (request->status == 3) {
      // already done
      request->status = 4;
    } else if (request->status == 4) {
      // still done
    } else {
      assert(0 && (0 <= request->status && request->status <= 4));
    }
    return request->status;
  }
};


template < >
struct Message<MessageBase::Kind::recv, gpump_pol>
  : MessageInterface<MessageBase::Kind::recv, gpump_pol>
{
  using base = MessageInterface<MessageBase::Kind::recv, gpump_pol>;

  using policy_comm = typename base::policy_comm;
  using communicator_type = typename base::communicator_type;
  using request_type      = typename base::request_type;
  using status_type       = typename base::status_type;

  // use the base class constructor
  using base::base;


  static int test_recv_any(communicator_type& con_comm,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      int status = handle_recv_request(con_comm, requests[i]);
      if (status == -3) {
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int wait_recv_any(communicator_type& con_comm,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    int ready = -1;
    do {
      ready = test_recv_any(con_comm, count, requests, statuses);
    } while (ready == -1);
    return ready;
  }

  static int test_recv_some(communicator_type& con_comm,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    int done = 0;
    if (count > 0) {
      bool new_requests = (requests[0]->status == -1);
      if (new_requests) {
        // detail::gpump::cork(con_comm.g);
      }
      for (int i = 0; i < count; ++i) {
        int status = handle_recv_request(con_comm, requests[i]);
        if (status == -3) {
          statuses[i] = 1;
          indices[done++] = i;
        }
      }
      if (new_requests) {
        // detail::gpump::uncork(con_comm.g, con_comm.stream_launch());
      }
    }
    return done;
  }

  static int wait_recv_some(communicator_type& con_comm,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    int done = 0;
    do {
      done = test_recv_some(con_comm, count, requests, indices, statuses);
    } while (done == 0);
    return done;
  }

  static bool test_recv_all(communicator_type& con_comm,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    int done = 0;
    if (count > 0) {
      bool new_requests = (requests[0]->status == -1);
      if (new_requests) {
        // detail::gpump::cork(con_comm.g);
      }
      for (int i = 0; i < count; ++i) {
        int status = handle_recv_request(con_comm, requests[i]);
        if (status == -3) {
          statuses[i] = 1;
        }
        if (status == -3 || status == -4) {
          done++;
        }
      }
      if (new_requests) {
        // detail::gpump::uncork(con_comm.g, con_comm.stream_launch());
      }
    }
    return done == count;
  }

  static void wait_recv_all(communicator_type& con_comm,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    bool done = false;
    do {
      done = test_recv_all(con_comm, count, requests, statuses);
    } while (!done);
  }

private:
  static bool start_wait_recv(communicator_type&,
                              request_type& request)
  {
    assert(!request->completed);
    bool done = false;
    if (request->context_type == ContextEnum::cuda) {
      detail::gpump::stream_wait_recv_complete(request->g, request->partner_rank, request->context.cuda.stream_launch());
      done = true;
    } else if (request->context_type == ContextEnum::cpu) {
      detail::gpump::cpu_ack_recv(request->g, request->partner_rank);
    } else {
      assert(0 && (request->context_type == ContextEnum::cuda || request->context_type == ContextEnum::cpu));
    }
    return done;
  }

  static bool test_waiting_recv(communicator_type&,
                                request_type& request)
  {
    assert(!request->completed);
    bool done = false;
    if (request->context_type == ContextEnum::cuda) {
      done = detail::gpump::is_receive_complete(request->g, request->partner_rank);
      request->completed = done;
      // do one test to get things moving, then allow the packs to be enqueued
      done = true;
    } else if (request->context_type == ContextEnum::cpu) {
      done = detail::gpump::is_receive_complete(request->g, request->partner_rank);
      request->completed = done;
    } else {
      assert(0 && (request->context_type == ContextEnum::cuda || request->context_type == ContextEnum::cpu));
    }
    return done;
  }

  // possible status values
  //  0 - not ready to wait, not received
  // -1 - ready to wait, first wait
  // -2 - ready to wait, waited before
  // -3 - ready, first ready
  // -4 - ready, ready before
  static int handle_recv_request(communicator_type& con_comm,
                                 request_type& request)
  {
    if (request->status == 0) {
      // not received
      assert(0 && (request->status != 0));
    } else if (request->status == -1) {
      // received, start waiting
      if (start_wait_recv(con_comm, request)) {
        // done
        request->status = -3;
      } else {
        // wait again later
        request->status = -2;
      }
    } else if (request->status == -2) {
      // still waiting, keep waiting
      if (test_waiting_recv(con_comm, request)) {
        // done
        request->status = -3;
      }
    } else if (request->status == -3) {
      // already done
      request->status = -4;
    } else if (request->status == -4) {
      // still done
    } else {
      assert(0 && (-4 <= request->status && request->status <= 0));
    }
    return request->status;
  }
};


template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::send, gpump_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::send, gpump_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::send, gpump_pol, exec_policy>;

  using policy_comm       = typename base::policy_comm;
  using communicator_type = typename base::communicator_type;
  using message_type      = typename base::message_type;
  using request_type      = typename base::request_type;
  using status_type       = typename base::status_type;

  using message_item_type = typename base::message_item_type;
  using context_type      = typename base::context_type;
  using event_type        = typename base::event_type;
  using group_type        = typename base::group_type;
  using component_type    = typename base::component_type;

  using message_request_type = typename communicator_type::message_request_type;
  using region_type = typename message_request_type::region_type;

  std::vector<message_request_type> m_msg_requests;

  // use the base class constructor
  using base::base;


  void finalize()
  {
    // call base finalize
    base::finalize();

    // allocate m_msg_requests
    m_msg_requests.resize(this->messages.size(), message_request_type{MessageBase::Kind::send});
  }


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      message_request_type& msg_request = m_msg_requests[msg->idx];
      msg_request.region = con_comm.get_mempool().allocate(con_comm.g, this->m_aloc, nbytes);
      msg->buf = msg_request.region.ptr;
      con_comm.get_message_request_map().emplace(&msg_request);
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.allocate(con, this->m_variables, this->m_items.size());
    }
  }

  void pack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con_comm);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    if (!comb_allow_pack_loop_fusion()) {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        const IdxT msg_idx = msg->idx;
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg_idx].start_component(this->m_groups[len-1], this->m_components[msg_idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          for (DataT const* src : this->m_variables) {
            // LOGPRINTF("%p pack %p = %p[%p] nitems %d\n", this, buf, src, indices, nitems);
            this->m_contexts[msg_idx].for_all(nitems, make_copy_idxr_idxr(src, detail::indexer_list_i{indices},
                                               static_cast<DataT*>(static_cast<void*>(buf)), detail::indexer_i{}));
            buf += nbytes;
          }
        }
        if (async == detail::Async::no) {
          this->m_contexts[msg_idx].finish_component(this->m_groups[len-1], this->m_components[msg_idx]);
        } else {
          this->m_contexts[msg_idx].finish_component_recordEvent(this->m_groups[len-1], this->m_components[msg_idx], this->m_events[msg->idx]);
        }
      }
    }
    else if (false && async == detail::Async::no) { // not sure how to know when individual contexts are in different streams
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          this->m_fuser.enqueue(con, (DataT*)buf, item->indices, item->size);
          buf += item->nbytes * this->m_variables.size();
          assert(static_cast<IdxT>(item->size*sizeof(DataT)) == item->nbytes);
        }
      }
      this->m_fuser.exec(con);
    } else {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        const IdxT msg_idx = msg->idx;
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg_idx].start_component(this->m_groups[len-1], this->m_components[msg_idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          this->m_fuser.enqueue(this->m_contexts[msg_idx], (DataT*)buf, item->indices, item->size);
          buf += item->nbytes * this->m_variables.size();
          assert(static_cast<IdxT>(item->size*sizeof(DataT)) == item->nbytes);
        }
        this->m_fuser.exec(this->m_contexts[msg_idx]);
        this->m_contexts[msg_idx].finish_component_recordEvent(this->m_groups[len-1], this->m_components[msg_idx], this->m_events[msg_idx]);
      }
    }
    con.finish_group(this->m_groups[len-1]);
  }

  IdxT wait_pack_complete(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    // LOGPRINTF("wait_pack_complete\n");

    // gpump isends use message packing context and don't need synchronization
    COMB::ignore_unused(con, con_comm, msgs, async);
    return len;
  }

  static void start_Isends(context_type& con, communicator_type& con_comm)
  {
    // LOGPRINTF("start_Isends\n");
    cork_Isends(con, con_comm);
  }

  void Isend(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/, request_type* requests)
  {
    if (len <= 0) return;
    start_Isends(con, con_comm);
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      const IdxT nbytes = msg->nbytes() * this->m_variables.size();

      // LOGPRINTF("%p Isend %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
      message_request_type& msg_request = m_msg_requests[msg->idx];
      start_Isend(con, con_comm, partner_rank, nbytes, msg_request);
      msg_request.request.status = 1;
      msg_request.request.g = con_comm.g;
      msg_request.request.partner_rank = partner_rank;
      msg_request.request.setContext(con);
      msg_request.request.completed = false;
      requests[i] = &msg_request.request;
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // LOGPRINTF("finish_Isends\n");
    uncork_Isends(con, con_comm);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf != nullptr);

      message_request_type& msg_request = m_msg_requests[msg->idx];
      finish_send(con_comm, msg_request);
      con_comm.get_mempool().deallocate(con_comm.g, this->m_aloc, msg_request.region);
      msg_request.region = region_type{};
      msg->buf = nullptr;
      con_comm.get_message_request_map().erase(&msg_request);
    }

    // TODO: worry about host reusing this memory before device synchronized
    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.deallocate(con);
    }
  }

private:
  void start_Isend(CPUContext&, communicator_type& con_comm, int partner_rank, IdxT nbytes, message_request_type& msg_request)
  {
    detail::gpump::isend(con_comm.g, partner_rank, msg_request.region.mr, msg_request.region.offset, nbytes);
  }

  void start_Isend(CudaContext& con, communicator_type& con_comm, int partner_rank, IdxT nbytes, message_request_type& msg_request)
  {
    detail::gpump::stream_send(con_comm.g, partner_rank, con.stream_launch(), msg_request.region.mr, msg_request.region.offset, nbytes);
  }

  static void cork_Isends(CPUContext&, communicator_type& con_comm)
  {
    COMB::ignore_unused(con_comm);
  }

  static void cork_Isends(CudaContext&, communicator_type& con_comm)
  {
    detail::gpump::cork(con_comm.g);
  }

  static void uncork_Isends(CPUContext&, communicator_type& con_comm)
  {
    COMB::ignore_unused(con_comm);
  }

  static void uncork_Isends(CudaContext&, communicator_type& con_comm)
  {
    detail::gpump::uncork(con_comm.g, con_comm.stream_launch());
  }

  void finish_send(communicator_type& con_comm, message_request_type& msg_request)
  {
    if (!msg_request.request.completed) {
      msg_request.request.completed = detail::gpump::is_send_complete(msg_request.request.g, msg_request.request.partner_rank);

      if (!msg_request.request.completed) {
        con_comm.wait_request(&msg_request);
      }
    }
  }
};

template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::recv, gpump_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::recv, gpump_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::recv, gpump_pol, exec_policy>;

  using policy_comm       = typename base::policy_comm;
  using communicator_type = typename base::communicator_type;
  using message_type      = typename base::message_type;
  using request_type      = typename base::request_type;
  using status_type       = typename base::status_type;

  using message_item_type = typename base::message_item_type;
  using context_type      = typename base::context_type;
  using event_type        = typename base::event_type;
  using group_type        = typename base::group_type;
  using component_type    = typename base::component_type;

  using message_request_type = typename communicator_type::message_request_type;
  using region_type = typename message_request_type::region_type;

  std::vector<message_request_type> m_msg_requests;

  // use the base class constructor
  using base::base;


  void finalize()
  {
    // call base finalize
    base::finalize();

    // allocate m_msg_requests
    m_msg_requests.resize(this->messages.size(), message_request_type{MessageBase::Kind::recv});
  }


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      message_request_type& msg_request = m_msg_requests[msg->idx];
      msg_request.region = con_comm.get_mempool().allocate(con_comm.g, this->m_aloc, nbytes);
      msg->buf = msg_request.region.ptr;
      con_comm.get_message_request_map().emplace(&msg_request);
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.allocate(con, this->m_variables, this->m_items.size());
    }
  }

  void Irecv(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      const IdxT nbytes = msg->nbytes() * this->m_variables.size();
      // LOGPRINTF("%p Irecv %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
      message_request_type& msg_request = m_msg_requests[msg->idx];
      detail::gpump::receive(con_comm.g, partner_rank, msg_request.region.mr, msg_request.region.offset, nbytes);
      msg_request.request.status = -1;
      msg_request.request.g = con_comm.g;
      msg_request.request.partner_rank = partner_rank;
      msg_request.request.setContext(con);
      msg_request.request.completed = false;
      requests[i] = &msg_request.request;
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con_comm);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    if (!comb_allow_pack_loop_fusion()) {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char const* buf = static_cast<char const*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          for (DataT* dst : this->m_variables) {
            // LOGPRINTF("%p unpack %p[%p] = %p nitems %d\n", this, dst, indices, buf, nitems);
            this->m_contexts[msg->idx].for_all(nitems, make_copy_idxr_idxr(static_cast<DataT const*>(static_cast<void const*>(buf)), detail::indexer_i{},
                                               dst, detail::indexer_list_i{indices}));
            buf += nbytes;
          }
        }
        this->m_contexts[msg->idx].finish_component(this->m_groups[len-1], this->m_components[msg->idx]);
      }
    }
    else if (false) { // not sure how to know when individual contexts are in different streams
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char const* buf = static_cast<char const*>(msg->buf);
        assert(buf != nullptr);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          this->m_fuser.enqueue(con, (DataT const*)buf, item->indices, item->size);
          buf += item->nbytes * this->m_variables.size();
          assert(static_cast<IdxT>(item->size*sizeof(DataT)) == item->nbytes);
        }
      }
      this->m_fuser.exec(con);
    } else {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char const* buf = static_cast<char const*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          this->m_fuser.enqueue(this->m_contexts[msg->idx], (DataT const*)buf, item->indices, item->size);
          buf += item->nbytes * this->m_variables.size();
          assert(static_cast<IdxT>(item->size*sizeof(DataT)) == item->nbytes);
        }
        this->m_fuser.exec(this->m_contexts[msg->idx]);
        this->m_contexts[msg->idx].finish_component(this->m_groups[len-1], this->m_components[msg->idx]);
      }
    }
    con.finish_group(this->m_groups[len-1]);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf != nullptr);

      message_request_type& msg_request = m_msg_requests[msg->idx];
      finish_recv(con_comm, msg_request);
      con_comm.get_mempool().deallocate(con_comm.g, this->m_aloc, msg_request.region);
      msg_request.region = region_type{};
      msg->buf = nullptr;
      con_comm.get_message_request_map().erase(&msg_request);
    }

    // TODO: worry about host reusing this memory before device synchronized
    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.deallocate(con);
    }
  }

private:
  void finish_recv(communicator_type& con_comm, message_request_type& msg_request)
  {
    if (!msg_request.request.completed) {
      msg_request.request.completed = detail::gpump::is_receive_complete(msg_request.request.g, msg_request.request.partner_rank);

      if (!msg_request.request.completed) {
        con_comm.wait_request(&msg_request);
      }
    }
  }
};


template < >
struct MessageGroup<MessageBase::Kind::send, gpump_pol, mpi_type_pol>
{
  // unimplemented
};

template < >
struct MessageGroup<MessageBase::Kind::recv, gpump_pol, mpi_type_pol>
{
  // unimplemented
};

} // namespace detail

#endif // COMB_ENABLE_GPUMP

#endif // _COMM_POL_GPUMP_HPP
