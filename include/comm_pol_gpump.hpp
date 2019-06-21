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


#include "for_all.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"
#include "utils_gpump.hpp"
#include "MessageBase.hpp"

struct GpumpRequest
{
  int status;
  struct gpump* comm;
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
    , comm(nullptr)
    , partner_rank(-1)
    , context_type(ContextEnum::invalid)
    , context()
  {

  }

  GpumpRequest(GpumpRequest const& other)
    : status(other.status)
    , comm(other.comm)
    , partner_rank(other.partner_rank)
    , context_type(ContextEnum::invalid)
    , context()
  {
    copy_context(other.context_type, other.context);
  }

  GpumpRequest& operator=(GpumpRequest const& other)
  {
    status = other.status;
    comm = other.comm;
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
  static const char* get_name() { return "gpump"; }
  using communicator_type = struct gpump*;
  static inline communicator_type communicator_create(MPI_Comm comm) { return detail::gpump::init(comm); }
  static inline void communicator_destroy(communicator_type g) { detail::gpump::term(g); }
  using send_request_type = GpumpRequest;
  static inline send_request_type send_request_null() { return send_request_type{}; }
  using recv_request_type = GpumpRequest;
  static inline recv_request_type recv_request_null() { return recv_request_type{}; }
  using send_status_type = int;
  static inline send_status_type send_status_null() { return 0; }
  using recv_status_type = int;
  static inline recv_status_type recv_status_null() { return 0; }
};


inline void connect_ranks(gpump_pol const&,
                          gpump_pol::communicator_type comm,
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
    detail::gpump::connect_propose(comm, rank);
  }
  for (int rank : ranks) {
    detail::gpump::connect_accept(comm, rank);
  }
}

inline void disconnect_ranks(gpump_pol const&,
                             gpump_pol::communicator_type comm,
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
    detail::gpump::disconnect(comm, rank);
  }
}


template < >
struct Message<gpump_pol> : detail::MessageBase
{
  using base = detail::MessageBase;

  using policy_comm = gpump_pol;
  using communicator_type = typename policy_comm::communicator_type;
  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;
  using send_status_type  = typename policy_comm::send_status_type;
  using recv_status_type  = typename policy_comm::recv_status_type;


  Message(Kind _kind, int partner_rank, int tag, bool have_many)
    : base(_kind, partner_rank, tag, have_many)
    , m_region(nullptr)
  { }

  ~Message()
  { }


  template < typename context >
  void pack(context const& con, communicator_type comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
      for_all(con, 0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      buf += len;
    }
  }

  template < typename context >
  void unpack(context const& con, communicator_type comm)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
      for_all(con, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      buf += len;
    }
  }

private:
  void start_Isend(CPUContext const&, communicator_type comm)
  {
    detail::gpump::isend(comm, partner_rank(), m_region, 0, nbytes());
    detail::gpump::cpu_ack_isend(comm, partner_rank());
  }

  void start_Isend(CudaContext const& con, communicator_type comm)
  {
    detail::gpump::stream_send(comm, partner_rank(), con.stream(), m_region, 0, nbytes());
    detail::gpump::stream_wait_send_complete(comm, partner_rank(), con.stream());
  }

public:
  template < typename context >
  void Isend(context const& con, communicator_type comm, send_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());

    start_Isend(con, comm);
    request->status = 1;
    request->comm = comm;
    request->partner_rank = partner_rank();
    request->setContext(con);
  }

  template < typename context >
  void Irecv(context const& con, communicator_type comm, recv_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());

    detail::gpump::receive(comm, partner_rank(), m_region, 0, nbytes());
    request->status = -1;
    request->comm = comm;
    request->partner_rank = partner_rank();
    request->setContext(con);
  }


  template < typename context >
  void allocate(context const&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(nbytes());
      m_region = detail::gpump::register_region(comm, m_buf, nbytes());
    }
  }

private:
  void wait_recv(CPUContext const&, communicator_type comm)
  {
    // already done
  }

  void wait_recv(CudaContext const& con, communicator_type comm)
  {
    detail::gpump::wait_receive_complete(comm, partner_rank());
  }

public:
  template < typename context >
  void deallocate(context const& con, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "gpump_pol does not support mpi_type_pol");
    if (m_buf != nullptr) {

      if (m_kind == Kind::send) {
        // already done
      } else if (m_kind == Kind::recv) {
        wait_recv(con, comm);
      }
      detail::gpump::deregister_region(comm, m_region);
      m_region = nullptr;
      buf_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }


private:
  static bool start_wait_send(send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      return detail::gpump::is_send_complete(request.comm, request.partner_rank);
    } else if (request.context_type == ContextEnum::cpu) {
      return detail::gpump::is_send_complete(request.comm, request.partner_rank);
    } else {
      assert(0);
    }
    return false;
  }

  static bool test_waiting_send(send_request_type& request)
  {
    if (request.context_type == ContextEnum::cuda) {
      return detail::gpump::is_send_complete(request.comm, request.partner_rank);
    } else if (request.context_type == ContextEnum::cpu) {
      return detail::gpump::is_send_complete(request.comm, request.partner_rank);
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
      detail::gpump::stream_wait_recv_complete(request.comm, request.partner_rank, request.context.cuda.stream());
      return true;
    } else if (request.context_type == ContextEnum::cpu) {
      detail::gpump::cpu_ack_recv(request.comm, request.partner_rank);
      return detail::gpump::is_receive_complete(request.comm, request.partner_rank);
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
      return detail::gpump::is_receive_complete(request.comm, request.partner_rank);
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
  struct ibv_mr* m_region;
};

#endif // COMB_ENABLE_GPUMP

#endif // _COMM_POL_GPUMP_HPP
