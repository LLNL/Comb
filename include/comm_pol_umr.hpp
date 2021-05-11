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

#ifndef _COMM_POL_UMR_HPP
#define _COMM_POL_UMR_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_UMR

#include "exec.hpp"
#include "comm_utils_umr.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct umr_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = false;
  static const char* get_name() { return "umr"; }
  using send_request_type = UMR_Request;
  using recv_request_type = UMR_Request;
  using send_status_type = UMR_Status;
  using recv_status_type = UMR_Status;
};

template < >
struct CommContext<umr_pol> : MPIContext
{
  using base = MPIContext;

  using pol = umr_pol;

  using send_request_type = typename pol::send_request_type;
  using recv_request_type = typename pol::recv_request_type;
  using send_status_type = typename pol::send_status_type;
  using recv_status_type = typename pol::recv_status_type;

  MPI_Comm comm = MPI_COMM_NULL;

  CommContext()
    : base()
  { }

  CommContext(base const& b)
    : base(b)
  { }

  CommContext(CommContext const& a_, MPI_Comm comm_)
    : base(a_)
    , comm(comm_)
  { }

  void ensure_waitable()
  {

  }

  template < typename context >
  void waitOn(context& con)
  {
    con.ensure_waitable();
    base::waitOn(con);
  }

  send_request_type send_request_null() { return UMR_REQUEST_NULL; }
  recv_request_type recv_request_null() { return UMR_REQUEST_NULL; }
  send_status_type send_status_null() { return send_status_type{}; }
  recv_status_type recv_status_null() { return recv_status_type{}; }

  void connect_ranks(std::vector<int> const& send_ranks,
                     std::vector<int> const& recv_ranks)
  {
    COMB::ignore_unused(comm, send_ranks, recv_ranks);
  }

  void disconnect_ranks(std::vector<int> const& send_ranks,
                        std::vector<int> const& recv_ranks)
  {
    COMB::ignore_unused(comm, send_ranks, recv_ranks);
  }


  void setup_mempool(COMB::Allocator& many_aloc,
                     COMB::Allocator& few_aloc)
  {
    COMB::ignore_unused(many_aloc, few_aloc);
  }

  void teardown_mempool()
  {
  }
};




namespace detail {

template < >
struct Message<MessageBase::Kind::send, umr_pol>
  : MessageInterface<MessageBase::Kind::send, umr_pol>
{
  using base = MessageInterface<MessageBase::Kind::send, umr_pol>;

  using policy_comm = typename base::policy_comm;
  using communicator_type = typename base::communicator_type;
  using request_type      = typename base::request_type;
  using status_type       = typename base::status_type;

  // use the base class constructor
  using base::base;


  static int wait_send_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    return detail::UMR::Waitany(count, requests, statuses);
  }

  static int test_send_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    return detail::UMR::Testany(count, requests, statuses);
  }

  static int wait_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::UMR::Waitsome(count, requests, indices, statuses);
  }

  static int test_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::UMR::Testsome(count, requests, indices, statuses);
  }

  static void wait_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    detail::UMR::Waitall(count, requests, statuses);
  }

  static bool test_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    return detail::UMR::Testall(count, requests, statuses);
  }
};


template < >
struct Message<MessageBase::Kind::recv, umr_pol>
  : MessageInterface<MessageBase::Kind::recv, umr_pol>
{
  using base = MessageInterface<MessageBase::Kind::recv, umr_pol>;

  using policy_comm = typename base::policy_comm;
  using communicator_type = typename base::communicator_type;
  using request_type      = typename base::request_type;
  using status_type       = typename base::status_type;

  // use the base class constructor
  using base::base;


  static int wait_recv_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    return detail::UMR::Waitany(count, requests, statuses);
  }

  static int test_recv_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    return detail::UMR::Testany(count, requests, statuses);
  }

  static int wait_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::UMR::Waitsome(count, requests, indices, statuses);
  }

  static int test_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::UMR::Testsome(count, requests, indices, statuses);
  }

  static void wait_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    detail::UMR::Waitall(count, requests, statuses);
  }

  static bool test_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    return detail::UMR::Testall(count, requests, statuses);
  }
};


template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::send, umr_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::send, umr_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::send, umr_pol, exec_policy>;

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

  // use the base class constructor
  using base::base;


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      msg->buf = this->m_aloc.allocate(nbytes);
    }
  }

  void pack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con_comm);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
      for (const MessageItemBase* msg_item : msg->message_items) {
        const message_item_type* item = static_cast<const message_item_type*>(msg_item);
        const IdxT len = item->size;
        const IdxT nbytes = item->nbytes;
        LidxT const* indices = item->indices;
        for (DataT const* src : this->m_variables) {
          // FGPRINTF(FileGroup::proc, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
          this->m_contexts[msg->idx].for_all(0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices},
                                             static_cast<DataT*>(static_cast<void*>(buf)), detail::indexer_idx{}));
          buf += nbytes;
        }
      }
      if (async == detail::Async::no) {
        this->m_contexts[msg->idx].finish_component(this->m_groups[len-1], this->m_components[msg->idx]);
      } else {
        this->m_contexts[msg->idx].finish_component_recordEvent(this->m_groups[len-1], this->m_components[msg->idx], this->m_events[msg->idx]);
      }
    }
    con.finish_group(this->m_groups[len-1]);
  }

  IdxT wait_pack_complete(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    // FGPRINTF(FileGroup::proc, "wait_pack_complete\n");
    if (len <= 0) return 0;
    if (async == detail::Async::no) {
      con_comm.waitOn(con);
    } else {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        if (!this->m_contexts[msg->idx].queryEvent(this->m_events[msg->idx])) {
          return i;
        }
      }
    }
    return len;
  }

  static void start_Isends(context_type& con, communicator_type& con_comm)
  {
    // FGPRINTF(FileGroup::proc, "start_Isends\n");
    COMB::ignore_unused(con, con_comm);
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
      const int tag = msg->msg_tag;
      const IdxT nbytes = msg->nbytes() * this->m_variables.size();
      // FGPRINTF(FileGroup::proc, "%p Isend %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
      detail::UMR::Isend(buf, nbytes, UMR_BYTE,
                         partner_rank, tag, con_comm.comm, &requests[i]);
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // FGPRINTF(FileGroup::proc, "finish_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf != nullptr);

      this->m_aloc.deallocate(msg->buf);

      msg->buf = nullptr;
    }
  }
};

template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::recv, umr_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::recv, umr_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::recv, umr_pol, exec_policy>;

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

  // use the base class constructor
  using base::base;


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      msg->buf = this->m_aloc.allocate(nbytes);
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
      const int tag = msg->msg_tag;
      const IdxT nbytes = msg->nbytes() * this->m_variables.size();
      // FGPRINTF(FileGroup::proc, "%p Irecv %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
      detail::UMR::Irecv(buf, nbytes, UMR_BYTE,
                         partner_rank, tag, con_comm.comm, &requests[i]);
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con_comm);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
      for (const MessageItemBase* msg_item : msg->message_items) {
        const message_item_type* item = static_cast<const message_item_type*>(msg_item);
        const IdxT len = item->size;
        const IdxT nbytes = item->nbytes;
        LidxT const* indices = item->indices;
        for (DataT* dst : this->m_variables) {
          // FGPRINTF(FileGroup::proc, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
          this->m_contexts[msg->idx].for_all(0, len, make_copy_idxr_idxr(static_cast<DataT*>(static_cast<void*>(buf)), detail::indexer_idx{},
                                             dst, detail::indexer_list_idx{indices}));
          buf += nbytes;
        }
      }
      this->m_contexts[msg->idx].finish_component(this->m_groups[len-1], this->m_components[msg->idx]);
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

      this->m_aloc.deallocate(msg->buf);

      msg->buf = nullptr;
    }
  }
};


template < >
struct MessageGroup<MessageBase::Kind::send, umr_pol, mpi_type_pol>
{
  // unimplemented
};

template < >
struct MessageGroup<MessageBase::Kind::recv, umr_pol, mpi_type_pol>
{
  // unimplemented
};

} // namespace detail

#endif // COMB_ENABLE_UMR

#endif // _COMM_POL_UMR_HPP
