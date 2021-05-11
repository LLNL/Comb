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

#ifndef _COMM_POL_MPI_HPP
#define _COMM_POL_MPI_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_MPI

#include "exec.hpp"
#include "comm_utils_mpi.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct mpi_pol {
  // static const bool async = false;
  static const bool mock = false;
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = true;
  static const char* get_name() { return "mpi"; }
  using send_request_type = MPI_Request;
  using recv_request_type = MPI_Request;
  using send_status_type = MPI_Status;
  using recv_status_type = MPI_Status;
};

template < >
struct CommContext<mpi_pol> : MPIContext
{
  using base = MPIContext;

  using pol = mpi_pol;

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

  send_request_type send_request_null() { return MPI_REQUEST_NULL; }
  recv_request_type recv_request_null() { return MPI_REQUEST_NULL; }
  send_status_type send_status_null() { return send_status_type{}; }
  recv_status_type recv_status_null() { return recv_status_type{}; }

  void connect_ranks(std::vector<int> const& send_ranks,
                     std::vector<int> const& recv_ranks)
  {
    COMB::ignore_unused(send_ranks, recv_ranks);
  }

  void disconnect_ranks(std::vector<int> const& send_ranks,
                        std::vector<int> const& recv_ranks)
  {
    COMB::ignore_unused(send_ranks, recv_ranks);
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
struct Message<MessageBase::Kind::send, mpi_pol>
  : MessageInterface<MessageBase::Kind::send, mpi_pol>
{
  using base = MessageInterface<MessageBase::Kind::send, mpi_pol>;

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
    return detail::MPI::Waitany(count, requests, statuses);
  }

  static int test_send_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    return detail::MPI::Testany(count, requests, statuses);
  }

  static int wait_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::MPI::Waitsome(count, requests, indices, statuses);
  }

  static int test_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::MPI::Testsome(count, requests, indices, statuses);
  }

  static void wait_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    detail::MPI::Waitall(count, requests, statuses);
  }

  static bool test_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    return detail::MPI::Testall(count, requests, statuses);
  }
};


template < >
struct Message<MessageBase::Kind::recv, mpi_pol>
  : MessageInterface<MessageBase::Kind::recv, mpi_pol>
{
  using base = MessageInterface<MessageBase::Kind::recv, mpi_pol>;

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
    return detail::MPI::Waitany(count, requests, statuses);
  }

  static int test_recv_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    return detail::MPI::Testany(count, requests, statuses);
  }

  static int wait_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::MPI::Waitsome(count, requests, indices, statuses);
  }

  static int test_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    return detail::MPI::Testsome(count, requests, indices, statuses);
  }

  static void wait_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    detail::MPI::Waitall(count, requests, statuses);
  }

  static bool test_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    return detail::MPI::Testall(count, requests, statuses);
  }
};


template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::send, mpi_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::send, mpi_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::send, mpi_pol, exec_policy>;

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

  // vars for fused loops
  DataT const** m_srcs = nullptr;

  DataT**       m_bufs = nullptr;
  LidxT const** m_idxs = nullptr;
  IdxT*         m_lens = nullptr;
  IdxT m_pos = 0;

  // use the base class constructor
  using base::base;


  void finalize()
  {
    // call base finalize
    base::finalize();
  }


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

    if (comb_allow_pack_loop_fusion() && m_srcs == nullptr) {

      // allocate per variable vars
      IdxT num_vars = this->m_variables.size();
      m_srcs = (DataT const**)con.util_aloc.allocate(num_vars*sizeof(DataT const*));

      // variable vars initialized here
      for (IdxT i = 0; i < num_vars; ++i) {
        m_srcs[i] = this->m_variables[i];
      }

      // allocate per item vars
      IdxT num_items = this->m_items.size();
      m_bufs = (DataT**)      con.util_aloc.allocate(num_items*sizeof(DataT*));
      m_idxs = (LidxT const**)con.util_aloc.allocate(num_items*sizeof(LidxT const*));
      m_lens = (IdxT*)        con.util_aloc.allocate(num_items*sizeof(IdxT));

      // item vars initialized in pack
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
    }
    else if (async == detail::Async::no) {
      IdxT num_vars = this->m_variables.size();
      DataT const** srcs = m_srcs;
      DataT**       bufs = m_bufs + m_pos;
      LidxT const** idxs = m_idxs + m_pos;
      IdxT*         lens = m_lens + m_pos;
      IdxT total_items = 0;
      IdxT num_fused = 0;
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          bufs[num_fused] = (DataT*)buf;
          idxs[num_fused] = indices;
          lens[num_fused] = nitems;
          total_items += nitems;
          num_fused += 1;
          buf += nbytes * num_vars;
          assert(static_cast<IdxT>(nitems*sizeof(DataT)) == nbytes);
        }
      }
      // FGPRINTF(FileGroup::proc, "%p pack %p = %p[%p] nitems %d\n", this, buf, src, indices, nitems);
      IdxT avg_items = (total_items + num_fused - 1) / num_fused;
      con.fused(num_fused, num_vars, avg_items, fused_packer(srcs, bufs, idxs, lens));
      m_pos += num_fused;
    } else {
      IdxT num_vars = this->m_variables.size();
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        DataT const** srcs = m_srcs;
        DataT**       bufs = m_bufs + m_pos;
        LidxT const** idxs = m_idxs + m_pos;
        IdxT*         lens = m_lens + m_pos;
        IdxT total_items = 0;
        IdxT num_fused = 0;
        this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          bufs[num_fused] = (DataT*)buf;
          idxs[num_fused] = indices;
          lens[num_fused] = nitems;
          total_items += nitems;
          num_fused += 1;
          buf += nbytes * num_vars;
          assert(static_cast<IdxT>(nitems*sizeof(DataT)) == nbytes);
        }
        // FGPRINTF(FileGroup::proc, "%p pack %p = %p[%p] nitems %d\n", this, buf, src, indices, nitems);
        IdxT avg_items = (total_items + num_fused - 1) / num_fused;
        this->m_contexts[msg->idx].fused(num_fused, num_vars, avg_items, fused_packer(srcs, bufs, idxs, lens));
        m_pos += num_fused;
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
      detail::MPI::Isend(buf, nbytes, MPI_BYTE,
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

    if (comb_allow_pack_loop_fusion() && m_srcs != nullptr && m_pos == static_cast<IdxT>(this->m_items.size())) {

      // deallocate per variable vars
      con.util_aloc.deallocate(m_srcs); m_srcs = nullptr;

      // deallocate per item vars
      con.util_aloc.deallocate(m_bufs); m_bufs = nullptr;
      con.util_aloc.deallocate(m_idxs); m_idxs = nullptr;
      con.util_aloc.deallocate(m_lens); m_lens = nullptr;

      // reset pos
      m_pos = 0;
    }
  }
};

template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::recv, mpi_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::recv, mpi_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::recv, mpi_pol, exec_policy>;

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

  // fused loop vars
  DataT**       m_dsts = nullptr;

  DataT const** m_bufs = nullptr;
  LidxT const** m_idxs = nullptr;
  IdxT*         m_lens = nullptr;
  IdxT m_pos = 0;

  // use the base class constructor
  using base::base;


  void finalize()
  {
    // call base finalize
    base::finalize();
  }

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

    if (comb_allow_pack_loop_fusion() && m_dsts == nullptr) {

      // allocate per variable vars
      IdxT num_vars = this->m_variables.size();
      m_dsts = (DataT**)con.util_aloc.allocate(num_vars*sizeof(DataT*));

      // variable vars initialized here
      for (IdxT i = 0; i < num_vars; ++i) {
        m_dsts[i] = this->m_variables[i];
      }

      // allocate per item vars
      IdxT num_items = this->m_items.size();
      m_bufs = (DataT const**)con.util_aloc.allocate(num_items*sizeof(DataT const*));
      m_idxs = (LidxT const**)con.util_aloc.allocate(num_items*sizeof(LidxT const*));
      m_lens = (IdxT*)        con.util_aloc.allocate(num_items*sizeof(IdxT));

      // item vars initialized in pack
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
      detail::MPI::Irecv(buf, nbytes, MPI_BYTE,
                         partner_rank, tag, con_comm.comm, &requests[i]);
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
    }
    else {
      IdxT num_vars = this->m_variables.size();
      DataT**       dsts = m_dsts;
      DataT const** bufs = m_bufs + m_pos;
      LidxT const** idxs = m_idxs + m_pos;
      IdxT*         lens = m_lens + m_pos;
      IdxT total_items = 0;
      IdxT num_fused = 0;
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          bufs[num_fused] = (DataT const*)buf;
          idxs[num_fused] = indices;
          lens[num_fused] = nitems;
          total_items += nitems;
          num_fused += 1;
          buf += nbytes * num_vars;
          assert(static_cast<IdxT>(nitems*sizeof(DataT)) == nbytes);
        }
      }
      // FGPRINTF(FileGroup::proc, "%p pack %p = %p[%p] nitems %d\n", this, buf, dst, indices, nitems);
      IdxT avg_items = (total_items + num_fused - 1) / num_fused;
      con.fused(num_fused, num_vars, avg_items, fused_unpacker(dsts, bufs, idxs, lens));
      m_pos += num_fused;
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

    if (comb_allow_pack_loop_fusion() && m_dsts != nullptr && m_pos == static_cast<IdxT>(this->m_items.size())) {

      // deallocate per variable vars
      con.util_aloc.deallocate(m_dsts); m_dsts = nullptr;

      // deallocate per item vars
      con.util_aloc.deallocate(m_bufs); m_bufs = nullptr;
      con.util_aloc.deallocate(m_idxs); m_idxs = nullptr;
      con.util_aloc.deallocate(m_lens); m_lens = nullptr;

      // reset pos
      m_pos = 0;
    }
  }
};


template < >
struct MessageGroup<MessageBase::Kind::send, mpi_pol, mpi_type_pol>
  : detail::MessageGroupInterface<MessageBase::Kind::send, mpi_pol, mpi_type_pol>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::send, mpi_pol, mpi_type_pol>;

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


  void finalize()
  {
    // call base finalize
    base::finalize();
  }


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // no buffer needed
      } else {
        IdxT nbytes = msg->nbytes() * this->m_variables.size();

        msg->buf = this->m_aloc.allocate(nbytes);
      }
    }
  }

  void pack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // pack via data type in send, previously used to reset message nbytes
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        int pos = 0;
        const IdxT nbytes = msg->nbytes() * this->m_variables.size();
        for (MessageItemBase* msg_item : msg->message_items) {
          message_item_type* item = static_cast<message_item_type*>(msg_item);
          const IdxT len = 1;
          MPI_Datatype mpi_type = item->mpi_type;
          int old_pos = pos;
          for (DataT const* src : this->m_variables) {
            // FGPRINTF(FileGroup::proc, "%p pack %p[%i] = %p\n", this, buf, pos, src);
            detail::MPI::Pack(src, len, mpi_type,
                              buf, nbytes, &pos, con_comm.comm);
          }
          item->packed_nbytes = pos - old_pos;
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
      const int partner_rank = msg->partner_rank;
      const int tag = msg->msg_tag;
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        const DataT* src = this->m_variables.front();
        const IdxT len = 1;
        const message_item_type* item = static_cast<const message_item_type*>(msg->message_items.front());
        MPI_Datatype mpi_type = item->mpi_type;
        // FGPRINTF(FileGroup::proc, "%p Isend %p to %i tag %i\n", this, src, partner_rank, tag);
        detail::MPI::Isend(src, len, mpi_type,
                           partner_rank, tag, con_comm.comm, &requests[i]);
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        int packed_nbytes = 0;
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          packed_nbytes += item->packed_nbytes;
        }
        // FGPRINTF(FileGroup::proc, "%p Isend %p nbytes %i to %i tag %i\n", this, buf, packed_nbytes, partner_rank, tag);
        detail::MPI::Isend(buf, packed_nbytes, MPI_PACKED,
                           partner_rank, tag, con_comm.comm, &requests[i]);
      }
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
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // buf not allocated
        assert(msg->buf == nullptr);
      } else {
        assert(msg->buf != nullptr);
        this->m_aloc.deallocate(msg->buf);
        msg->buf = nullptr;
      }
    }
  }
};

template < >
struct MessageGroup<MessageBase::Kind::recv, mpi_pol, mpi_type_pol>
  : detail::MessageGroupInterface<MessageBase::Kind::recv, mpi_pol, mpi_type_pol>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::recv, mpi_pol, mpi_type_pol>;

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


  void finalize()
  {
    // call base finalize
    base::finalize();
  }


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // no buffer needed
      } else {
        IdxT nbytes = msg->nbytes() * this->m_variables.size();

        msg->buf = this->m_aloc.allocate(nbytes);
      }
    }
  }

  void Irecv(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      const int partner_rank = msg->partner_rank;
      const int tag = msg->msg_tag;
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        DataT* dst = m_variables.front();
        assert(dst != nullptr);
        IdxT len = 1;
        const message_item_type* item = static_cast<const message_item_type*>(msg->message_items.front());
        MPI_Datatype mpi_type = item->mpi_type;
        // FGPRINTF(FileGroup::proc, "%p Irecv %p to %i tag %i\n", this, dst, partner_rank, tag);
        detail::MPI::Irecv(dst, len, mpi_type,
                           partner_rank, tag, con_comm.comm, &requests[i]);
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        const IdxT nbytes = msg->nbytes() * this->m_variables.size();
        // FGPRINTF(FileGroup::proc, "%p Irecv %p maxnbytes %i to %i tag %i\n", this, dst, nbytes, partner_rank, tag);
        detail::MPI::Irecv(buf, nbytes, MPI_PACKED,
                           partner_rank, tag, con_comm.comm, &requests[i]);
      }
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con_comm);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      this->m_contexts[msg->idx].start_component(this->m_groups[len-1], this->m_components[msg->idx]);
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // nothing to do
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        const IdxT nbytes = msg->nbytes() * this->m_variables.size();
        int pos = 0;
        for (MessageItemBase* msg_item : msg->message_items) {
          message_item_type* item = static_cast<message_item_type*>(msg_item);
          const IdxT len = 1;
          MPI_Datatype mpi_type = item->mpi_type;
          int old_pos = pos;
          for (DataT* dst : this->m_variables) {
            // FGPRINTF(FileGroup::proc, "%p unpack %p = %p[%i]\n", this, dst, buf, pos);
            detail::MPI::Unpack(buf, nbytes, &pos,
                                dst, len, mpi_type, con_comm.comm);
          }
          item->packed_nbytes = pos - old_pos;
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
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // buf not allocated
        assert(msg->buf == nullptr);
      } else {
        assert(msg->buf != nullptr);
        this->m_aloc.deallocate(msg->buf);
        msg->buf = nullptr;
      }
    }
  }
};

} // namespace detail

#endif

#endif // _COMM_POL_MPI_HPP
