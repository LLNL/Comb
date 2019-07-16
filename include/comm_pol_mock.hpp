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

#ifndef _COMM_POL_MOCK_HPP
#define _COMM_POL_MOCK_HPP

#include "config.hpp"

#include "for_all.hpp"
#include "utils.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct mock_pol {
  // static const bool async = false;
  static const bool mock = true;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = true;
  static const char* get_name() { return "mock"; }
  using communicator_type = int;
  static inline communicator_type communicator_create(MPI_Comm) { return communicator_type{0}; }
  static inline void communicator_destroy(communicator_type) { }
  using send_request_type = int;
  static inline send_request_type send_request_null() { return 0; }
  using recv_request_type = int;
  static inline recv_request_type recv_request_null() { return 0; }
  using send_status_type = int;
  static inline send_status_type send_status_null() { return 0; }
  using recv_status_type = int;
  static inline recv_status_type recv_status_null() { return 0; }
};

template < >
struct CommContext<mock_pol> : MPIContext
{
  using base = MPIContext;
  CommContext()
    : base()
  { }
  CommContext(base const& b)
    : base(b)
  { }
};


inline void connect_ranks(mock_pol const&,
                          mock_pol::communicator_type comm,
                          std::vector<int> const& send_ranks,
                          std::vector<int> const& recv_ranks)
{
  COMB::ignore_unused(comm, send_ranks, recv_ranks);
}

inline void disconnect_ranks(mock_pol const&,
                             mock_pol::communicator_type comm,
                             std::vector<int> const& send_ranks,
                             std::vector<int> const& recv_ranks)
{
  COMB::ignore_unused(comm, send_ranks, recv_ranks);
}


template < >
struct Message<mock_pol> : detail::MessageBase
{
  using base = detail::MessageBase;

  using policy_comm = mock_pol;
  using communicator_type = typename policy_comm::communicator_type;
  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;
  using send_status_type  = typename policy_comm::send_status_type;
  using recv_status_type  = typename policy_comm::recv_status_type;

  static void setup_mempool(communicator_type comm,
                            COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    COMB::ignore_unused(comm, many_aloc, few_aloc);
  }

  static void teardown_mempool(communicator_type comm)
  {
    COMB::ignore_unused(comm);
  }


  // use the base class constructor
  using base::base;

  ~Message()
  { }


  template < typename context >
  void pack(context& con, communicator_type)
  {
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

  void pack(ExecContext<mpi_type_pol>&, communicator_type /*comm*/)
  {
    if (items.size() == 1) {
      m_nbytes = sizeof(DataT)*items.front().size;
    } else {
      DataT* buf = m_buf;
      assert(buf != nullptr);
      IdxT buf_max_nbytes = max_nbytes();
      int pos = 0;
      auto end = std::end(items);
      for (auto i = std::begin(items); i != end; ++i) {
        DataT const* src = i->data;
        MPI_Datatype mpi_type = i->mpi_type;
        // FPRINTF(stdout, "%p pack %p[%i] = %p\n", this, buf, pos, src);
        detail::MPI::Pack(src, 1, mpi_type, buf, buf_max_nbytes, &pos, /*comm*/ MPI_COMM_WORLD);
      }
      // set nbytes to actual value
      m_nbytes = pos;
    }
  }

  template < typename context >
  void unpack(context& con, communicator_type)
  {
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

  void unpack(ExecContext<mpi_type_pol>&, communicator_type /*comm*/)
  {
    if (items.size() == 1) {
      // nothing to do
    } else {
      DataT const* buf = m_buf;
      assert(buf != nullptr);
      IdxT buf_max_nbytes = max_nbytes();
      int pos = 0;
      auto end = std::end(items);
      for (auto i = std::begin(items); i != end; ++i) {
        DataT* dst = i->data;
        MPI_Datatype mpi_type = i->mpi_type;
        // FPRINTF(stdout, "%p unpack %p = %p[%i]\n", this, dst, buf, pos);
        detail::MPI::Unpack(buf, buf_max_nbytes, &pos, dst, 1, mpi_type, /*comm*/ MPI_COMM_WORLD);
      }
    }
  }


  template < typename context >
  void Isend(context&, communicator_type, send_request_type* request)
  {
    // FPRINTF(stdout, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    *request = 1;
  }

  void Isend(ExecContext<mpi_type_pol>&, communicator_type, send_request_type* request)
  {

    if (items.size() == 1) {
      // DataT const* src = items.front().data;
      // MPI_Datatype mpi_type = items.front().mpi_type;
      // FPRINTF(stdout, "%p Isend %p to %i tag %i\n", this, src, partner_rank(), tag());
      *request = 1;
    } else {
      // FPRINTF(stdout, "%p Isend %p nbytes %i to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
      *request = 1;
    }
  }

  template < typename context >
  static void start_Isends(context& con, communicator_type comm)
  {
    // FPRINTF(stdout, "start_Isends\n");
    COMB::ignore_unused(con, comm);
  }

  template < typename context >
  static void finish_Isends(context& con, communicator_type comm)
  {
    // FPRINTF(stdout, "finish_Isends\n");
    COMB::ignore_unused(con, comm);
  }

  template < typename context >
  void Irecv(context&, communicator_type, recv_request_type* request)
  {
    // FPRINTF(stdout, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    *request = -1;
  }

  void Irecv(ExecContext<mpi_type_pol>&, communicator_type, recv_request_type* request)
  {
    if (items.size() == 1) {
      // DataT* dst = items.front().data;
      // MPI_Datatype mpi_type = items.front().mpi_type;
      // FPRINTF(stdout, "%p Irecv %p to %i tag %i\n", this, dst, partner_rank(), tag());
      *request = -1;
    } else {
      // FPRINTF(stdout, "%p Irecv %p maxnbytes %i to %i tag %i\n", this, dst, max_nbytes(), partner_rank(), tag());
      *request = -1;
    }
  }


  template < typename context >
  void allocate(context&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    COMB::ignore_unused(comm);
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(nbytes());
    }
  }

  void allocate(ExecContext<mpi_type_pol>&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    COMB::ignore_unused(comm);
    if (m_buf == nullptr) {
      if (items.size() == 1) {
        // no buffer needed
      } else {
        m_buf = (DataT*)buf_aloc.allocate(max_nbytes());
      }
    }
  }

  template < typename context >
  void deallocate(context&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    COMB::ignore_unused(comm);
    if (m_buf != nullptr) {
      buf_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }


  static int wait_send_any(int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int test_send_any(int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int wait_send_some(int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
        indices[done++] = i;
      }
    }
    return done;
  }

  static int test_send_some(int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
        indices[done++] = i;
      }
    }
    return done;
  }

  static void wait_send_all(int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
      }
    }
  }

  static bool test_send_all(int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
      }
    }
    return true;
  }


  static int wait_recv_any(int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int test_recv_any(int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
        return i;
      }
    }
    return -1;
  }

  static int wait_recv_some(int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
        indices[done++] = i;
      }
    }
    return done;
  }

  static int test_recv_some(int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    int done = 0;
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
        indices[done++] = i;
      }
    }
    return done;
  }

  static void wait_recv_all(int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
      }
    }
  }

  static bool test_recv_all(int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
      }
    }
    return true;
  }
};

#endif // _COMM_POL_MOCK_HPP
