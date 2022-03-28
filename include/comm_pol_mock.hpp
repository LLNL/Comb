//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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

#include "exec.hpp"
#include "comm_utils_mpi.hpp"
#include "MessageBase.hpp"
#include "ExecContext.hpp"

struct mock_pol {
  // static const bool async = false;
  static const bool mock = true;
#ifdef COMB_ENABLE_MPI
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = true;
#endif
  static const bool persistent = false;
  static const char* get_name() { return "mock"; }
  using send_request_type = int;
  using recv_request_type = int;
  using send_status_type = int;
  using recv_status_type = int;
};


#ifdef COMB_ENABLE_MPI
#define COMB_MOCK_BASE MPIContext
#else
#define COMB_MOCK_BASE CPUContext
#endif

template < >
struct CommContext<mock_pol> : COMB_MOCK_BASE
{
  using base = COMB_MOCK_BASE;

  using pol = mock_pol;

  using send_request_type = typename pol::send_request_type;
  using recv_request_type = typename pol::recv_request_type;
  using send_status_type = typename pol::send_status_type;
  using recv_status_type = typename pol::recv_status_type;

#ifdef COMB_ENABLE_MPI
  MPI_Comm comm = MPI_COMM_NULL;
#endif

  CommContext()
    : base()
  { }

  CommContext(base const& b)
    : base(b)
  { }

  CommContext(CommContext const& a_
#ifdef COMB_ENABLE_MPI
             ,MPI_Comm comm_
#endif
              )
    : base(a_)
#ifdef COMB_ENABLE_MPI
    , comm(comm_)
#endif
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

  send_request_type send_request_null() { return 0; }
  recv_request_type recv_request_null() { return 0; }
  send_status_type send_status_null() { return 0; }
  recv_status_type recv_status_null() { return 0; }

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
struct Message<MessageBase::Kind::send, mock_pol>
  : MessageInterface<MessageBase::Kind::send, mock_pol>
{
  using base = MessageInterface<MessageBase::Kind::send, mock_pol>;

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

  static int test_send_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
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

  static int wait_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
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

  static int test_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
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

  static void wait_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != 2) {
        assert(requests[i] == 1);
        requests[i] = 2;
        statuses[i] = 1;
      }
    }
  }

  static bool test_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
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
};


template < >
struct Message<MessageBase::Kind::recv, mock_pol>
  : MessageInterface<MessageBase::Kind::recv, mock_pol>
{
  using base = MessageInterface<MessageBase::Kind::recv, mock_pol>;

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

  static int test_recv_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
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

  static int wait_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
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

  static int test_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
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

  static void wait_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    for (int i = 0; i < count; ++i) {
      if (requests[i] != -2) {
        assert(requests[i] == -1);
        requests[i] = -2;
        statuses[i] = 1;
      }
    }
  }

  static bool test_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
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

template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::send, mock_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::send, mock_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::send, mock_pol, exec_policy>;

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

  void setup(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm, msgs, len, requests);
  }

  void cleanup(communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con_comm, msgs, len, requests);
  }

  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      msg->buf = this->m_aloc.allocate(nbytes);
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
          this->m_contexts[msg_idx].finish_component_recordEvent(this->m_groups[len-1], this->m_components[msg_idx], this->m_events[msg_idx]);
        }
      }
    }
    else if (async == detail::Async::no) {
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
    if (len <= 0) return 0;
    if (async == detail::Async::no) {
      con_comm.waitOn(con);
    } else {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        const IdxT msg_idx = msg->idx;
        if (!this->m_contexts[msg_idx].queryEvent(this->m_events[msg_idx])) {
          return i;
        }
      }
    }
    return len;
  }

  static void start_Isends(context_type& con, communicator_type& con_comm)
  {
    // LOGPRINTF("start_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void Isend(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async, request_type* requests)
  {
    COMB::ignore_unused(async);
    if (len <= 0) return;
    start_Isends(con, con_comm);
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      // const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      for (const MessageItemBase* msg_item : msg->message_items) {
        const IdxT nbytes = msg_item->nbytes * this->m_variables.size();
        // LOGPRINTF("%p Isend %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
        buf += nbytes;
      }
      requests[i] = 1;
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // LOGPRINTF("finish_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf != nullptr);

      this->m_aloc.deallocate(msg->buf);

      msg->buf = nullptr;
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.deallocate(con);
    }
  }
};

template < typename exec_policy >
struct MessageGroup<MessageBase::Kind::recv, mock_pol, exec_policy>
  : detail::MessageGroupInterface<MessageBase::Kind::recv, mock_pol, exec_policy>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::recv, mock_pol, exec_policy>;

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

  void setup(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm, msgs, len, requests);
  }

  void cleanup(communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con_comm, msgs, len, requests);
  }

  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      msg->buf = this->m_aloc.allocate(nbytes);
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.allocate(con, this->m_variables, this->m_items.size());
    }
  }

  void Irecv(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm, async);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      // const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      // const IdxT nbytes = msg->nbytes() * this->m_variables.size();
      // LOGPRINTF("%p Irecv %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
      requests[i] = -1;
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con_comm, async);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    if (!comb_allow_pack_loop_fusion()) {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        const IdxT msg_idx = msg->idx;
        char const* buf = static_cast<char const*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg_idx].start_component(this->m_groups[len-1], this->m_components[msg_idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          for (DataT* dst : this->m_variables) {
            // LOGPRINTF("%p unpack %p[%p] = %p nitems %d\n", this, dst, indices, buf, nitems);
            this->m_contexts[msg_idx].for_all(nitems, make_copy_idxr_idxr(static_cast<DataT const*>(static_cast<void const*>(buf)), detail::indexer_i{},
                                               dst, detail::indexer_list_i{indices}));
            buf += nbytes;
          }
        }
        this->m_contexts[msg_idx].finish_component(this->m_groups[len-1], this->m_components[msg_idx]);
      }
    }
    else {
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
    }
    con.finish_group(this->m_groups[len-1]);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf != nullptr);

      this->m_aloc.deallocate(msg->buf);

      msg->buf = nullptr;
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.deallocate(con);
    }
  }
};


#ifdef COMB_ENABLE_MPI

template < >
struct MessageGroup<MessageBase::Kind::send, mock_pol, mpi_type_pol>
  : detail::MessageGroupInterface<MessageBase::Kind::send, mock_pol, mpi_type_pol>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::send, mock_pol, mpi_type_pol>;

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

  void setup(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm, msgs, len, requests);
  }

  void cleanup(communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con_comm, msgs, len, requests);
  }

  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
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
          const IdxT nitems = 1;
          MPI_Datatype mpi_type = item->mpi_type;
          int old_pos = pos;
          for (DataT const* src : this->m_variables) {
            // LOGPRINTF("%p pack %p[%i] = %p\n", this, buf, pos, src);
            detail::MPI::Pack(src, nitems, mpi_type,
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
    // LOGPRINTF("wait_pack_complete\n");
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
    // LOGPRINTF("start_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void Isend(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async, request_type* requests)
  {
    COMB::ignore_unused(async);
    if (len <= 0) return;
    start_Isends(con, con_comm);
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      // const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        // const DataT* src = this->m_variables.front();
        // const IdxT nitems = 1;
        // const message_item_type* item = static_cast<const message_item_type*>(msg->message_items.front());
        // MPI_Datatype mpi_type = item->mpi_type;
        // LOGPRINTF("%p Isend %p to %i tag %i\n", this, src, partner_rank, tag);
        requests[i] = 1;
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        // int packed_nbytes = 0;
        // for (const MessageItemBase* msg_item : msg->message_items) {
        //   packed_nbytes += item->packed_nbytes;
        // }
        // LOGPRINTF("%p Isend %p nbytes %i to %i tag %i\n", this, buf, packed_nbytes, partner_rank, tag);
        requests[i] = 1;
      }
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // LOGPRINTF("finish_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
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
struct MessageGroup<MessageBase::Kind::recv, mock_pol, mpi_type_pol>
  : detail::MessageGroupInterface<MessageBase::Kind::recv, mock_pol, mpi_type_pol>
{
  using base = detail::MessageGroupInterface<MessageBase::Kind::recv, mock_pol, mpi_type_pol>;

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

  void setup(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm, msgs, len, requests);
  }

  void cleanup(communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con_comm, msgs, len, requests);
  }

  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
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

  void Irecv(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm, async);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      // const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      if (msg->message_items.size() == 1 && this->m_variables.size() == 1) {
        DataT* dst = m_variables.front();
        assert(dst != nullptr);
        // IdxT nitems = 1;
        // const message_item_type* item = static_cast<const message_item_type*>(msg->message_items.front());
        // MPI_Datatype mpi_type = item.mpi_type;
        // LOGPRINTF("%p Irecv %p to %i tag %i\n", this, dst, partner_rank, tag);
        requests[i] = -1;
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        // const IdxT nbytes = msg->nbytes() * this->m_variables.size();
        // LOGPRINTF("%p Irecv %p maxnbytes %i to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
        requests[i] = -1;
      }
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con_comm, async);
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
          const IdxT nitems = 1;
          MPI_Datatype mpi_type = item->mpi_type;
          int old_pos = pos;
          for (DataT* dst : this->m_variables) {
            // LOGPRINTF("%p unpack %p = %p[%i]\n", this, dst, buf, pos);
            detail::MPI::Unpack(buf, nbytes, &pos,
                                dst, nitems, mpi_type, con_comm.comm);
          }
          item->packed_nbytes = pos - old_pos;
        }
      }
      this->m_contexts[msg->idx].finish_component(this->m_groups[len-1], this->m_components[msg->idx]);
    }
    con.finish_group(this->m_groups[len-1]);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con, con_comm, async);
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

#endif

} // namespace detail

#endif // _COMM_POL_MOCK_HPP
