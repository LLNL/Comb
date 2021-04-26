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

#ifndef _MESSAGE_HPP
#define _MESSAGE_HPP

#include "config.hpp"

#include <type_traits>
#include <list>
#include <utility>

#include "memory.hpp"
#include "exec_for_all.hpp"
#include "exec_utils.hpp"
#include "comm_utils_mpi.hpp"

namespace detail {

enum struct Async {
  no
 ,yes
};

struct MessageBase
{
  enum struct Kind {
    send
   ,recv
  };

  Kind m_kind;
};


struct MessageItemBase
{
  IdxT size;
  IdxT nbytes;

  MessageItemBase(IdxT _size, IdxT _nbytes)
    : size(_size)
    , nbytes(_nbytes)
  { }

  MessageItemBase(MessageItemBase const&) = delete;
  MessageItemBase& operator=(MessageItemBase const&) = delete;

  MessageItemBase(MessageItemBase && o)
    : size(detail::exchange(o.size, 0))
    , nbytes(detail::exchange(o.nbytes, 0))
  { }
  MessageItemBase& operator=(MessageItemBase &&) = delete;
};

template < typename exec_policy >
struct MessageItem : MessageItemBase
{
  LidxT* indices;
  COMB::Allocator& m_aloc;

  MessageItem(IdxT _size, IdxT _nbytes, LidxT* _indices, COMB::Allocator& _aloc)
    : MessageItemBase(_size, _nbytes)
    , indices(_indices)
    , m_aloc(_aloc)
  { }

  MessageItem(MessageItem const&) = delete;
  MessageItem& operator=(MessageItem const&) = delete;

  MessageItem(MessageItem && o)
    : MessageItemBase(std::move(o))
    , indices(detail::exchange(o.indices, nullptr))
    , m_aloc(o.m_aloc)
  { }
  MessageItem& operator=(MessageItem &&) = delete;

  ~MessageItem()
  {
    if (indices) {
      m_aloc.deallocate(indices); indices = nullptr;
    }
  }
};

#ifdef COMB_ENABLE_MPI

template < >
struct MessageItem<mpi_type_pol> : MessageItemBase
{
  MPI_Datatype mpi_type;
  int packed_nbytes;

  MessageItem(IdxT _size, IdxT _nbytes, MPI_Datatype _mpi_type)
    : MessageItemBase(_size, _nbytes)
    , mpi_type(_mpi_type)
    , packed_nbytes(0)
  { }

  MessageItem(MessageItem const&) = delete;
  MessageItem& operator=(MessageItem const&) = delete;

  MessageItem(MessageItem && o)
    : MessageItemBase(std::move(o))
    , mpi_type(detail::exchange(o.mpi_type, MPI_DATATYPE_NULL))
    , packed_nbytes(detail::exchange(o.packed_nbytes, 0))
  { }
  MessageItem& operator=(MessageItem &&) = delete;

  ~MessageItem()
  {
    if (mpi_type != MPI_DATATYPE_NULL) {
      detail::MPI::Type_free(&mpi_type); mpi_type = MPI_DATATYPE_NULL;
    }
  }
};

#endif


template < MessageBase::Kind kind, typename comm_policy >
struct Message;

template < MessageBase::Kind kind, typename comm_policy, typename exec_policy >
struct MessageGroup;


template < MessageBase::Kind kind, typename comm_policy >
struct MessageInterface
{
  using policy_comm       = comm_policy;
  using communicator_type = CommContext<policy_comm>;
  using request_type      = typename std::conditional<kind == MessageBase::Kind::send,
                                typename policy_comm::send_request_type,
                                typename policy_comm::recv_request_type>::type;
  using status_type       = typename std::conditional<kind == MessageBase::Kind::send,
                                typename policy_comm::send_status_type,
                                typename policy_comm::recv_status_type>::type;

  IdxT idx;
  int partner_rank;
  int msg_tag;

  void* buf = nullptr;

  request_type* request = nullptr;
  status_type*  status = nullptr;

  std::vector<MessageItemBase*> message_items;

  MessageInterface(IdxT _idx, int _partner_rank, int _tag)
    : idx(_idx)
    , partner_rank(_partner_rank)
    , msg_tag(_tag)
  {

  }

  void set_request(request_type* _request)
  {
    request = _request;
  }

  void set_status(status_type* _status)
  {
    status = _status;
  }

  void add_item(MessageItemBase& item)
  {
    message_items.push_back(&item);
  }

  IdxT nbytes() const
  {
    IdxT msg_nbytes = 0;
    for (const MessageItemBase* item : message_items) {
      msg_nbytes += item->nbytes;
    }
    return msg_nbytes;
  }
};

template < MessageBase::Kind kind, typename comm_policy, typename exec_policy >
struct MessageGroupInterface
{
  using policy_comm       = comm_policy;
  using communicator_type = CommContext<policy_comm>;
  using message_type      = Message<kind, comm_policy>;
  using request_type      = typename std::conditional<kind == MessageBase::Kind::send,
                                typename policy_comm::send_request_type,
                                typename policy_comm::recv_request_type>::type;
  using status_type       = typename std::conditional<kind == MessageBase::Kind::send,
                                typename policy_comm::send_status_type,
                                typename policy_comm::recv_status_type>::type;

  using message_item_type = MessageItem<exec_policy>;
  using context_type      = ExecContext<exec_policy>;
  using event_type        = typename exec_policy::event_type;
  using group_type        = typename exec_policy::group_type;
  using component_type    = typename exec_policy::component_type;

  std::vector<message_type> messages;
  std::vector<context_type> m_contexts;
  std::vector<event_type> m_events;
  std::vector<component_type> m_components;
  std::vector<group_type> m_groups;

  std::vector<DataT*> m_variables;

  std::vector<int> m_item_partner_ranks;
  std::vector<message_item_type> m_items;

  COMB::Allocator& m_aloc;


  MessageGroupInterface(COMB::Allocator& aloc_)
    : m_aloc(aloc_)
  {

  }

  void add_message(context_type& con, int partner_rank, int tag)
  {
    messages.emplace_back(messages.size(), partner_rank, tag);
    m_contexts.emplace_back( con );
    m_events.emplace_back( m_contexts.back().createEvent() );
    m_components.emplace_back( m_contexts.back().create_component() );
    m_groups.emplace_back( m_contexts.back().create_group() );
  }

  void add_variable(DataT *data)
  {
    m_variables.emplace_back(data);
  }

  void add_message_item(int partner_rank, message_item_type&& item)
  {
    // emplace_back invalidates iterators
    m_item_partner_ranks.emplace_back(partner_rank);
    m_items.emplace_back(std::move(item));
  }

  void finalize()
  {
    // add items to messages
    IdxT numItems = m_items.size();
    for (IdxT i = 0; i < numItems; ++i) {

      int partner_rank = m_item_partner_ranks[i];
      message_item_type& item = m_items[i];

      bool found = false;
      for (message_type& msg : messages) {
        if (msg.partner_rank == partner_rank) {
          msg.add_item(item);
          found = true;
          break;
        }
      }
      assert(found);
    }
    m_item_partner_ranks.clear();
  }

  ~MessageGroupInterface()
  {
    IdxT numMessages = messages.size();
    for(IdxT i = 0; i < numMessages; i++) {
      m_contexts[i].destroyEvent(m_events[i]);
    }
    for(IdxT i = 0; i < numMessages; i++) {
      m_contexts[i].destroy_component(m_components[i]);
    }
    for(IdxT i = 0; i < numMessages; i++) {
      m_contexts[i].destroy_group(m_groups[i]);
    }
  }
};

} // namespace detail


#endif // _MESSAGE_HPP

