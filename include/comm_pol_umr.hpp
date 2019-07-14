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

#ifndef _COMM_POL_UMR_HPP
#define _COMM_POL_UMR_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_UMR

#include "for_all.hpp"
#include "utils.hpp"
#include "utils_umr.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct umr_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = false;
  static const char* get_name() { return "umr"; }
  using communicator_type = UMR_Comm;
  static inline communicator_type communicator_create(UMR_Comm comm) { return comm; }
  static inline void communicator_destroy(communicator_type) { }
  using send_request_type = UMR_Request;
  static inline send_request_type send_request_null() { return UMR_REQUEST_NULL; }
  using recv_request_type = UMR_Request;
  static inline recv_request_type recv_request_null() { return UMR_REQUEST_NULL; }
  using send_status_type = UMR_Status;
  static inline send_status_type send_status_null() { return send_status_type{}; }
  using recv_status_type = UMR_Status;
  static inline recv_status_type recv_status_null() { return recv_status_type{}; }
};

template < >
struct CommContext<umr_pol> : MPIContext
{
  using base = MPIContext;
  CommContext()
    : base()
  { }
  CommContext(base const& b)
    : base(b)
  { }
};


inline void connect_ranks(umr_pol const&,
                          umr_pol::communicator_type comm,
                          std::vector<int> const& send_ranks,
                          std::vector<int> const& recv_ranks)
{
  COMB::ignore_unused(comm, send_ranks, recv_ranks);
}

inline void disconnect_ranks(umr_pol const&,
                             umr_pol::communicator_type comm,
                             std::vector<int> const& send_ranks,
                             std::vector<int> const& recv_ranks)
{
  COMB::ignore_unused(comm, send_ranks, recv_ranks);
}


template < >
struct Message<umr_pol> : detail::MessageBase
{
  using base = detail::MessageBase;

  using policy_comm = umr_pol;
  using communicator_type = typename policy_comm::communicator_type;
  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;
  using send_status_type  = typename policy_comm::send_status_type;
  using recv_status_type  = typename policy_comm::recv_status_type;

  static void setup_mempool(COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    COMB::ignore_unused(many_aloc, few_aloc);
  }

  static void teardown_mempool()
  {

  }


  // use the base class constructor
  using base::base;

  ~Message()
  { }


  template < typename context >
  void pack(context& con, communicator_type)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "umr_pol does not support mpi_type_pol");
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
  void unpack(context& con, communicator_type)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "umr_pol does not support mpi_type_pol");
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


  template < typename context >
  void Isend(context&, communicator_type comm, send_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "umr_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    detail::UMR::Isend(buffer(), nbytes(), UMR_BYTE, partner_rank(), tag(), comm, request);
  }

  template < typename context >
  void Irecv(context&, communicator_type comm, recv_request_type* request)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "umr_pol does not support mpi_type_pol");
    // FPRINTF(stdout, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    detail::UMR::Irecv(buffer(), nbytes(), UMR_BYTE, partner_rank(), tag(), comm, request);
  }


  template < typename context >
  void allocate(context&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "umr_pol does not support mpi_type_pol");
    COMB::ignore_unused(comm);
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(nbytes());
    }
  }

  template < typename context >
  void deallocate(context&, communicator_type comm, COMB::Allocator& buf_aloc)
  {
    static_assert(!std::is_same<context, ExecContext<mpi_type_pol>>::value, "umr_pol does not support mpi_type_pol");
    COMB::ignore_unused(comm);
    if (m_buf != nullptr) {
      buf_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }


  static int wait_send_any(int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    return detail::UMR::Waitany(count, requests, statuses);
  }

  static int test_send_any(int count, send_request_type* requests,
                           send_status_type* statuses)
  {
    return detail::UMR::Testany(count, requests, statuses);
  }

  static int wait_send_some(int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    return detail::UMR::Waitsome(count, requests, indices, statuses);
  }

  static int test_send_some(int count, send_request_type* requests,
                            int* indices, send_status_type* statuses)
  {
    return detail::UMR::Testsome(count, requests, indices, statuses);
  }

  static void wait_send_all(int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    detail::UMR::Waitall(count, requests, statuses);
  }

  static bool test_send_all(int count, send_request_type* requests,
                            send_status_type* statuses)
  {
    return detail::UMR::Testall(count, requests, statuses);
  }


  static int wait_recv_any(int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    return detail::UMR::Waitany(count, requests, statuses);
  }

  static int test_recv_any(int count, recv_request_type* requests,
                           recv_status_type* statuses)
  {
    return detail::UMR::Testany(count, requests, statuses);
  }

  static int wait_recv_some(int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    return detail::UMR::Waitsome(count, requests, indices, statuses);
  }

  static int test_recv_some(int count, recv_request_type* requests,
                            int* indices, recv_status_type* statuses)
  {
    return detail::UMR::Testsome(count, requests, indices, statuses);
  }

  static void wait_recv_all(int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    detail::UMR::Waitall(count, requests, statuses);
  }

  static bool test_recv_all(int count, recv_request_type* requests,
                            recv_status_type* statuses)
  {
    return detail::UMR::Testall(count, requests, statuses);
  }
};

#endif // COMB_ENABLE_UMR

#endif // _COMM_POL_UMR_HPP
