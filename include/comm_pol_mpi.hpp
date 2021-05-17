//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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
    LOGPRINTF("Message<mpi>::wait_send_any count %d requests %p statuses %p\n", count, requests, statuses);
    int ret = detail::MPI::Waitany(count, requests, statuses);
    LOGPRINTF("Message<mpi>::wait_send_any return %d\n", ret);
    return std::move(ret);
  }

  static int test_send_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::test_send_any count %d requests %p statuses %p\n", count, requests, statuses);
    int ret = detail::MPI::Testany(count, requests, statuses);
    LOGPRINTF("Message<mpi>::test_send_any return %d\n", ret);
    return std::move(ret);
  }

  static int wait_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::wait_send_some count %d requests %p indices %p statuses %p\n", count, requests, indices, statuses);
    int ret = detail::MPI::Waitsome(count, requests, indices, statuses);
    LOGPRINTF("Message<mpi>::wait_send_some return %d\n", ret);
    return std::move(ret);
  }

  static int test_send_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::test_send_some count %d requests %p indices %p statuses %p\n", count, requests, indices, statuses);
    int ret = detail::MPI::Testsome(count, requests, indices, statuses);
    LOGPRINTF("Message<mpi>::test_send_some return %d\n", ret);
    return std::move(ret);
  }

  static void wait_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::wait_send_all count %d requests %p statuses %p\n", count, requests, statuses);
    detail::MPI::Waitall(count, requests, statuses);
    LOGPRINTF("Message<mpi>::wait_send_all return\n");
  }

  static bool test_send_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::test_send_all count %d requests %p statuses %p\n", count, requests, statuses);
    bool ret = detail::MPI::Testall(count, requests, statuses);
    LOGPRINTF("Message<mpi>::test_send_all return %s\n", ret ? "true" : "false");
    return std::move(ret);
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
    LOGPRINTF("Message<mpi>::wait_recv_any count %d requests %p statuses %p\n", count, requests, statuses);
    int ret = detail::MPI::Waitany(count, requests, statuses);
    LOGPRINTF("Message<mpi>::wait_recv_any return %d\n", ret);
    return std::move(ret);
  }

  static int test_recv_any(communicator_type&,
                           int count, request_type* requests,
                           status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::test_recv_any count %d requests %p statuses %p\n", count, requests, statuses);
    int ret = detail::MPI::Testany(count, requests, statuses);
    LOGPRINTF("Message<mpi>::test_recv_any return %d\n", ret);
    return std::move(ret);
  }

  static int wait_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::wait_recv_some count %d requests %p indices %p statuses %p\n", count, requests, indices, statuses);
    int ret = detail::MPI::Waitsome(count, requests, indices, statuses);
    LOGPRINTF("Message<mpi>::wait_recv_some return %d\n", ret);
    return std::move(ret);
  }

  static int test_recv_some(communicator_type&,
                            int count, request_type* requests,
                            int* indices, status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::test_recv_some count %d requests %p indices %p statuses %p\n", count, requests, indices, statuses);
    int ret = detail::MPI::Testsome(count, requests, indices, statuses);
    LOGPRINTF("Message<mpi>::test_recv_some return %d\n", ret);
    return std::move(ret);
  }

  static void wait_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::wait_recv_all count %d requests %p statuses %p\n", count, requests, statuses);
    detail::MPI::Waitall(count, requests, statuses);
    LOGPRINTF("Message<mpi>::wait_recv_all return\n");
  }

  static bool test_recv_all(communicator_type&,
                            int count, request_type* requests,
                            status_type* statuses)
  {
    LOGPRINTF("Message<mpi>::test_recv_all count %d requests %p statuses %p\n", count, requests, statuses);
    bool ret = detail::MPI::Testall(count, requests, statuses);
    LOGPRINTF("Message<mpi>::test_recv_all return %s\n", ret ? "true" : "false");
    return std::move(ret);
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
    LOGPRINTF("%p send allocate msgs %p len %d\n", this, msgs, len);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      msg->buf = this->m_aloc.allocate(nbytes);
      LOGPRINTF("%p send allocate msg %p buf %p nbytes %d\n", this, msg, msg->buf, nbytes);
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.allocate(con, this->m_variables, this->m_items.size());
    }
  }

  void pack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async async)
  {
    COMB::ignore_unused(con_comm);
    LOGPRINTF("%p send pack con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    if (!comb_allow_pack_loop_fusion()) {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        LOGPRINTF("%p send pack msg %p buf %p\n", this, msg, msg->buf);
        const IdxT msg_idx = msg->idx;
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg_idx].start_component(this->m_groups[len-1], this->m_components[msg_idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          LOGPRINTF("%p send pack con %p item %p buf %p = srcs[indices %p] nitems %d\n", this, &this->m_contexts[msg_idx], item, buf, item->indices, item->size);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          for (DataT const* src : this->m_variables) {
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
        LOGPRINTF("%p send pack msg %p buf %p\n", this, msg, msg->buf);
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          LOGPRINTF("%p send pack con %p item %p buf %p = srcs[indices %p] nitems %d\n", this, &con, item, buf, item->indices, item->size);
          this->m_fuser.enqueue(con, (DataT*)buf, item->indices, item->size);
          buf += item->nbytes * this->m_variables.size();
          assert(static_cast<IdxT>(item->size*sizeof(DataT)) == item->nbytes);
        }
      }
      this->m_fuser.exec(con);
    } else {
      IdxT num_vars = this->m_variables.size();
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        LOGPRINTF("%p send pack msg %p buf %p\n", this, msg, msg->buf);
        const IdxT msg_idx = msg->idx;
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg_idx].start_component(this->m_groups[len-1], this->m_components[msg_idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          LOGPRINTF("%p send pack con %p item %p buf %p = srcs[indices %p] nitems %d\n", this, &this->m_contexts[msg_idx], item, buf, item->indices, item->size);
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
    LOGPRINTF("%p send wait_pack_complete con %p msgs %p len %d\n", this, &con, msgs, len);
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
    LOGPRINTF("send start_Isends con %p\n", &con);
    COMB::ignore_unused(con, con_comm);
  }

  void Isend(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/, request_type* requests)
  {
    LOGPRINTF("%p send Isend con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    start_Isends(con, con_comm);
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      LOGPRINTF("%p send Isend msg %p buf %p nbytes %d to %i tag %i\n",
                this, msg, msg->buf, msg->nbytes() * this->m_variables.size(), msg->partner_rank, msg->msg_tag);
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      const int partner_rank = msg->partner_rank;
      const int tag = msg->msg_tag;
      const IdxT msg_nbytes = msg->nbytes() * this->m_variables.size();

      // char const* print_buf = buf;
      // for (const MessageItemBase* msg_item : msg->message_items) {
      //   const message_item_type* item = static_cast<const message_item_type*>(msg_item);
      //   const IdxT nitems = item->size;
      //   const IdxT nbytes = item->nbytes;
      //   LidxT const* indices = item->indices;
      //   for (DataT const* src : this->m_variables) {
      //     if (nitems*sizeof(DataT) == nbytes) {
      //       LOGPRINTF("  buf %p nitems %d nbytes %d [ ", print_buf, nitems, nbytes);
      //       for (IdxT idx = 0; idx < nitems; ++idx) {
      //         LOGPRINTF("%f ", (double)((DataT const*)print_buf)[idx]);
      //       }
      //       LOGPRINTF("] src %p[indices %p]\n", src, indices);
      //     }
      //     print_buf += nbytes;
      //   }
      // }

      detail::MPI::Isend(buf, msg_nbytes, MPI_BYTE,
                         partner_rank, tag, con_comm.comm, &requests[i]);
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    LOGPRINTF("send finish_Isends con %p\n", &con);
    COMB::ignore_unused(con, con_comm);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    LOGPRINTF("%p send deallocate con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      LOGPRINTF("%p send deallocate msg %p buf %p\n", this, msg, msg->buf);
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
    LOGPRINTF("%p recv allocate con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      assert(msg->buf == nullptr);

      IdxT nbytes = msg->nbytes() * this->m_variables.size();

      msg->buf = this->m_aloc.allocate(nbytes);
      LOGPRINTF("%p recv allocate msg %p buf %p nbytes %d\n",
                                this, msg, msg->buf, msg->nbytes() * this->m_variables.size());
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.allocate(con, this->m_variables, this->m_items.size());
    }
  }

  void Irecv(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/, request_type* requests)
  {
    COMB::ignore_unused(con, con_comm);
    LOGPRINTF("%p recv Irecv con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      const message_type* msg = msgs[i];
      LOGPRINTF("%p recv Irecv msg %p buf %p nbytes %d to %d tag %d\n",
                                this, msg, msg->buf, msg->nbytes() * this->m_variables.size(), msg->partner_rank, msg->msg_tag);
      char* buf = static_cast<char*>(msg->buf);
      assert(buf != nullptr);
      const int partner_rank = msg->partner_rank;
      const int tag = msg->msg_tag;
      const IdxT nbytes = msg->nbytes() * this->m_variables.size();
      detail::MPI::Irecv(buf, nbytes, MPI_BYTE,
                         partner_rank, tag, con_comm.comm, &requests[i]);
    }
  }

  void unpack(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con_comm);
    LOGPRINTF("%p recv unpack con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    con.start_group(this->m_groups[len-1]);
    if (!comb_allow_pack_loop_fusion()) {
      for (IdxT i = 0; i < len; ++i) {
        const message_type* msg = msgs[i];
        LOGPRINTF("%p recv unpack msg %p buf %p\n", this, msg, msg->buf);
        const IdxT msg_idx = msg->idx;
        char const* buf = static_cast<char const*>(msg->buf);
        assert(buf != nullptr);
        this->m_contexts[msg_idx].start_component(this->m_groups[len-1], this->m_components[msg_idx]);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          LOGPRINTF("%p recv unpack con %p item %p dsts[indices %p] = buf %p[i] nitems %d\n", this, &this->m_contexts[msg_idx], item, item->indices, buf, item->size);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;
          for (DataT* dst : this->m_variables) {

            // if (nitems*sizeof(DataT) == nbytes) {
            //   LOGPRINTF("  buf %p nitems %d nbytes %d [ ", buf, nitems, nbytes);
            //   for (IdxT idx = 0; idx < nitems; ++idx) {
            //     LOGPRINTF("%f ", (double)((DataT const*)buf)[idx]);
            //   }
            //   LOGPRINTF("] dst %p[indices %p]\n", dst, indices);
            // }

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
        LOGPRINTF("%p recv unpack msg %p buf %p\n", this, msg, msg->buf);
        char const* buf = static_cast<char const*>(msg->buf);
        assert(buf != nullptr);
        for (const MessageItemBase* msg_item : msg->message_items) {
          const message_item_type* item = static_cast<const message_item_type*>(msg_item);
          LOGPRINTF("%p recv unpack con %p item %p dsts[indices %p] = buf %p[i] nitems %d\n", this, &con, item, item->indices, buf, item->size);
          const IdxT nitems = item->size;
          const IdxT nbytes = item->nbytes;
          LidxT const* indices = item->indices;

          // for (DataT* dst : this->m_variables) {
          //   char const* print_buf = buf;
          //   if (nitems*sizeof(DataT) == nbytes) {
          //     LOGPRINTF("  buf %p nitems %d nbytes %d [ ", print_buf, nitems, nbytes);
          //     for (IdxT idx = 0; idx < nitems; ++idx) {
          //       LOGPRINTF("%f ", (double)((DataT const*)print_buf)[idx]);
          //     }
          //     LOGPRINTF("] dst %p[indices %p]\n", dst, indices);
          //   }
          //   print_buf += nbytes;
          // }

          this->m_fuser.enqueue(con, (DataT const*)buf, indices, nitems);
          buf += nbytes * this->m_variables.size();
          assert(static_cast<IdxT>(nitems*sizeof(DataT)) == nbytes);
        }
      }
      this->m_fuser.exec(con);
    }
    con.finish_group(this->m_groups[len-1]);
  }

  void deallocate(context_type& con, communicator_type& con_comm, message_type** msgs, IdxT len, detail::Async /*async*/)
  {
    COMB::ignore_unused(con, con_comm);
    LOGPRINTF("%p recv deallocate con %p msgs %p len %d\n", this, &con, msgs, len);
    if (len <= 0) return;
    for (IdxT i = 0; i < len; ++i) {
      message_type* msg = msgs[i];
      LOGPRINTF("%p recv deallocate msg %p buf %p\n", this, msg, msg->buf);
      assert(msg->buf != nullptr);

      this->m_aloc.deallocate(msg->buf);

      msg->buf = nullptr;
    }

    if (comb_allow_pack_loop_fusion()) {
      this->m_fuser.deallocate(con);
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
            // LOGPRINTF("%p pack %p[%i] = %p\n", this, buf, pos, src);
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
        // LOGPRINTF("%p Isend %p to %i tag %i\n", this, src, partner_rank, tag);
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
        // LOGPRINTF("%p Isend %p nbytes %i to %i tag %i\n", this, buf, packed_nbytes, partner_rank, tag);
        detail::MPI::Isend(buf, packed_nbytes, MPI_PACKED,
                           partner_rank, tag, con_comm.comm, &requests[i]);
      }
    }
    finish_Isends(con, con_comm);
  }

  static void finish_Isends(context_type& con, communicator_type& con_comm)
  {
    // LOGPRINTF("finish_Isends\n");
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
        // LOGPRINTF("%p Irecv %p to %i tag %i\n", this, dst, partner_rank, tag);
        detail::MPI::Irecv(dst, len, mpi_type,
                           partner_rank, tag, con_comm.comm, &requests[i]);
      } else {
        char* buf = static_cast<char*>(msg->buf);
        assert(buf != nullptr);
        const IdxT nbytes = msg->nbytes() * this->m_variables.size();
        // LOGPRINTF("%p Irecv %p maxnbytes %i to %i tag %i\n", this, dst, nbytes, partner_rank, tag);
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
            // LOGPRINTF("%p unpack %p = %p[%i]\n", this, dst, buf, pos);
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
