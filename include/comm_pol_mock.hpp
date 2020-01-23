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
#ifdef COMB_ENABLE_MPI
  // compile mpi_type packing/unpacking tests for this comm policy
  static const bool use_mpi_type = true;
#endif
  static const char* get_name() { return "mock"; }
  using send_request_type = int;
  using recv_request_type = int;
  using send_status_type = int;
  using recv_status_type = int;
};

template < >
struct CommContext<mock_pol> : CPUContext
{
  using base = CPUContext;

  using pol = mock_pol;

  using send_request_type = typename pol::send_request_type;
  using recv_request_type = typename pol::recv_request_type;
  using send_status_type = typename pol::send_status_type;
  using recv_status_type = typename pol::recv_status_type;

  CommContext()
    : base()
  { }

  CommContext(base const& b)
    : base(b)
  { }

  CommContext(CommContext const& a_
#ifdef COMB_ENABLE_MPI
             ,MPI_Comm
#endif
              )
    : base(a_)
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


  static void setup_mempool(communicator_type& con_comm,
                            COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    COMB::ignore_unused(con_comm, many_aloc, few_aloc);
  }

  static void teardown_mempool(communicator_type& con_comm)
  {
    COMB::ignore_unused(con_comm);
  }


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


  static void setup_mempool(communicator_type& con_comm,
                            COMB::Allocator& many_aloc,
                            COMB::Allocator& few_aloc)
  {
    COMB::ignore_unused(con_comm, many_aloc, few_aloc);
  }

  static void teardown_mempool(communicator_type& con_comm)
  {
    COMB::ignore_unused(con_comm);
  }


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


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len)
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

  void Isend(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
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
        // FGPRINTF(FileGroup::proc, "%p Isend %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
        buf += nbytes;
      }
      requests[i] = 1;
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // FGPRINTF(FileGroup::proc, "finish_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len)
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


  void allocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len)
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

  void Irecv(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      // const int partner_rank = msg->partner_rank;
      // const int tag = msg->msg_tag;
      // const IdxT nbytes = msg->nbytes() * this->m_variables.size();
      // FGPRINTF(FileGroup::proc, "%p Irecv %p nbytes %d to %i tag %i\n", this, buf, nbytes, partner_rank, tag);
      requests[i] = -1;
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len)
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

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len)
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


#ifdef COMB_ENABLE_MPI

// mpi_type_pol specialization of MessageGroup<mock_pol, ?>
template < >
struct MessageGroup<mock_pol, mpi_type_pol>
  : detail::MessageGroupInterface<mock_pol, mpi_type_pol>
{
  using base = detail::MessageGroupInterface<MessageGroup<mock_pol, mpi_type_pol>>;

  // use the base class constructor
  using base::base;


  void allocate(context_type& con, communicator_type& con_comm)
  {
    COMB::ignore_unused(con, con_comm);
    if (m_buf == nullptr) {
      if (m_variables.size() == 1) {
        // no buffer needed
      } else {
        IdxT nbytes = 0;
        for (message_info_type const& msg : messages) {
          for (const MessageItemBase* msg_item : msg.message_items) {
            nbytes += msg_item->nbytes * m_variables.size();
          }
        }
        m_buf = m_aloc.allocate(nbytes);
      }
    }
  }

  void pack(context_type& con, communicator_type& con_comm)
  {
    COMB::ignore_unused(con);
    con.start_group(m_group);
    if (m_variables.size() == 1) {
      // pack via data type in send, previously used to reset message nbytes
    } else {
      char* buf = static_cast<char*>(m_buf);
      assert(buf != nullptr);
      for (message_info_type const& msg : messages) {
        for (const MessageItemBase* msg_item : msg.message_items) {
          const message_item_type* item = static_cast<item>(msg_item);
          const IdxT len = item->size;
          const IdxT nbytes = item->nbytes * m_variables.size();
          MPI_Datatype mpi_type = item->mpi_type;
          int pos = 0;
          for (const DataT* src : m_variables) {
            // FGPRINTF(FileGroup::proc, "%p pack %p[%i] = %p\n", this, buf, pos, src);
            detail::MPI::Pack(src, len, mpi_type,
                              buf, nbytes, &pos, con_comm.comm);
          }
          item->packed_nbytes = pos;
          buf += nbytes;
        }
      }
    }
    con.finish_group(m_group);
  }

  static void wait_pack_complete(context_type& con, communicator_type& con_comm)
  {
    // FGPRINTF(FileGroup::proc, "wait_pack_complete\n");
    con_comm.waitOn(con);
  }

  static void start_Isends(context_type& con, communicator_type& con_comm)
  {
    // FGPRINTF(FileGroup::proc, "start_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void Isend(context_type& con, communicator_type& con_comm, request_type* request)
  {
    wait_pack_complete(con, con_comm);
    start_Isends(con, con_comm);
    if (m_variables.size() == 1) {
      const DataT* src = m_variables.front();
      assert(src != nullptr);
      for (message_info_type const& msg : messages) {
        const int partner_rank = msg.partner_rank;
        const int tag = msg.msg_tag;
        for (const MessageItemBase* msg_item : msg.message_items) {
          const message_item_type* item = static_cast<item>(msg_item);
          const IdxT len = item->size;
          MPI_Datatype mpi_type = item->mpi_type;
          // FGPRINTF(FileGroup::proc, "%p Isend %p to %i tag %i\n", this, src, partner_rank, tag);
          *request++ = 1;
        }
      }
    } else {
      const char* buf = static_cast<const char*>(m_buf);
      assert(buf != nullptr);
      for (message_info_type const& msg : messages) {
        const int partner_rank = msg.partner_rank;
        const int tag = msg.msg_tag;
        for (auto const& item : m_items) {
          const IdxT max_nbytes = item.nbytes * m_variables.size();
          const int packed_nbytes = item.packed_nbytes;
          // FGPRINTF(FileGroup::proc, "%p Isend %p nbytes %i to %i tag %i\n", this, buf, packed_nbytes, partner_rank, tag);
          *request++ = 1;
          buf += max_nbytes;
        }
      }
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // FGPRINTF(FileGroup::proc, "finish_Isends\n");
    COMB::ignore_unused(con, con_comm);
  }

  void Irecv(context_type& con, communicator_type& con_comm, recv_request_type* request)
  {
    COMB::ignore_unused(con);
    if (m_variables.size() == 1) {
      DataT* dst = m_variables.front();
      assert(dst != nullptr);
      for (message_info_type const& msg : messages) {
        const int partner_rank = msg.partner_rank;
        const int tag = msg.msg_tag;
        for (auto const& item : m_items) {
          MPI_Datatype mpi_type = item.mpi_type;
          const IdxT len = item.size;
          // FGPRINTF(FileGroup::proc, "%p Irecv %p to %i tag %i\n", this, dst, partner_rank(), tag());
          *request++ = -1;
        }
      }
    } else {
      const char* buf = static_cast<const char*>(m_buf);
      assert(buf != nullptr);
      for (message_info_type const& msg : messages) {
        const int partner_rank = msg.partner_rank;
        const int tag = msg.msg_tag;
        for (auto const& item : m_items) {
          const IdxT max_nbytes = item.nbytes * m_variables.size();
          // FGPRINTF(FileGroup::proc, "%p Irecv %p maxnbytes %i to %i tag %i\n", this, dst, max_nbytes(), partner_rank(), tag());
          *request++ = -1;
          buf += max_nbytes;
        }
      }
    }
  }

  void unpack(context_type& con, communicator_type& con_comm)
  {
    COMB::ignore_unused(con);
    con.start_group(m_group);
    if (m_variables.size() == 1) {
      // nothing to do
    } else {
      const char* buf = static_cast<const char*>(m_buf);
      assert(buf != nullptr);
      for (message_info_type const& msg : messages) {
        for (auto const& item : m_items) {
          MPI_Datatype mpi_type = item.mpi_type;
          const IdxT len = item.size;
          const IdxT max_nbytes = item.nbytes * m_variables.size();
          int pos = 0;
          for (DataT* dst : m_variables) {
            // FGPRINTF(FileGroup::proc, "%p unpack %p = %p[%i]\n", this, dst, buf, pos);
            detail::MPI::Unpack(buf, max_nbytes, &pos,
                                dst, len, mpi_type, con_comm.comm);
          }
          buf += max_nbytes;
        }
      }
    }
    con.finish_group(m_group);
  }

  void deallocate(context_type& con, communicator_type& con_comm, COMB::Allocator& buf_aloc)
  {
    COMB::ignore_unused(con, con_comm);
    if (m_buf != nullptr) {
      m_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }
};

#endif

} // namespace detail

#endif // _COMM_POL_MOCK_HPP
