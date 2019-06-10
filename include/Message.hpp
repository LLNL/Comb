//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "for_all.hpp"
#include "utils.hpp"

template < typename policy_comm_ >
struct Message
{
  using policy_comm = policy_comm_;

  int m_partner_rank;
  int m_msg_tag;
  DataT* m_buf;
  IdxT m_size;
  IdxT m_max_nbytes;
  IdxT m_nbytes;
  bool m_have_many;

  struct list_item_type
  {
    DataT* data;
    LidxT* indices;
    COMB::Allocator& aloc;
    IdxT size;
    MPI_Datatype mpi_type;
    IdxT mpi_pack_max_nbytes;
    list_item_type(DataT* data_, LidxT* indices_, COMB::Allocator& aloc_, IdxT size_,
                   MPI_Datatype mpi_type_, IdxT mpi_pack_max_nbytes_)
     : data(data_), indices(indices_), aloc(aloc_), size(size_),
       mpi_type(mpi_type_), mpi_pack_max_nbytes(mpi_pack_max_nbytes_)
    { }
  };

  std::list<list_item_type> items;

  Message(int partner_rank, int tag, bool have_many)
    : m_partner_rank(partner_rank)
    , m_msg_tag(tag)
    , m_buf(nullptr)
    , m_max_nbytes(0)
    , m_nbytes(0)
    , m_have_many(have_many)
  {

  }

  int partner_rank()
  {
    return m_partner_rank;
  }

  int tag()
  {
    return m_msg_tag;
  }

  DataT* buffer()
  {
    return m_buf;
  }

  IdxT size() const
  {
    return m_size;
  }

  IdxT max_nbytes() const
  {
    return m_max_nbytes;
  }

  IdxT nbytes() const
  {
    return m_nbytes;
  }

  bool have_many() const
  {
    return m_have_many;
  }

  void add(DataT* data, LidxT* indices, COMB::Allocator& aloc, IdxT size, MPI_Datatype mpi_type, IdxT mpi_pack_max_nbytes)
  {
    items.emplace_back(data, indices, aloc, size, mpi_type, mpi_pack_max_nbytes);
    if (items.back().mpi_type != MPI_DATATYPE_NULL) {
      m_max_nbytes += mpi_pack_max_nbytes;
      m_nbytes += mpi_pack_max_nbytes;
    } else {
      m_max_nbytes += sizeof(DataT)*size;
      m_nbytes += sizeof(DataT)*size;
    }
    m_size += size;
  }

  template < typename policy >
  void pack(policy const& pol, typename policy_comm::communicator_type)
  {
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p pack %p = %p[%p] len %d\n", this, buf, src, indices, len);
      for_all(pol, 0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      buf += len;
    }
  }

  void pack(mpi_type_pol const&, typename policy_comm::communicator_type comm)
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
        detail::MPI::Pack(src, 1, mpi_type, buf, buf_max_nbytes, &pos, comm);
      }
      // set nbytes to actual value
      m_nbytes = pos;
    }
  }

  template < typename policy >
  void unpack(policy const& pol, typename policy_comm::communicator_type)
  {
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      // FPRINTF(stdout, "%p unpack %p[%p] = %p len %d\n", this, dst, indices, buf, len);
      for_all(pol, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      buf += len;
    }
  }

  void unpack(mpi_type_pol const&, typename policy_comm::communicator_type comm)
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
        detail::MPI::Unpack(buf, buf_max_nbytes, &pos, dst, 1, mpi_type, comm);
      }
    }
  }

  template < typename policy >
  void Isend(policy const&, typename policy_comm::communicator_type comm, typename policy_comm::send_request_type* request)
  {
    // FPRINTF(stdout, "%p Isend %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    start_send(policy_comm{}, buffer(), nbytes(), MPI_BYTE, partner_rank(), tag(), comm, request);
  }

  void Isend(mpi_type_pol const&, typename policy_comm::communicator_type comm, typename policy_comm::send_request_type* request)
  {
    if (items.size() == 1) {
      DataT const* src = items.front().data;
      MPI_Datatype mpi_type = items.front().mpi_type;
      // FPRINTF(stdout, "%p Isend %p to %i tag %i\n", this, src, partner_rank(), tag());
      start_send(policy_comm{}, src, 1, mpi_type, partner_rank(), tag(), comm, request);
    } else {
      // FPRINTF(stdout, "%p Isend %p nbytes %i to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
      start_send(policy_comm{}, buffer(), nbytes(), MPI_PACKED, partner_rank(), tag(), comm, request);
    }
  }

  template < typename policy >
  void Irecv(policy const&, typename policy_comm::communicator_type comm, typename policy_comm::recv_request_type* request)
  {
    // FPRINTF(stdout, "%p Irecv %p nbytes %d to %i tag %i\n", this, buffer(), nbytes(), partner_rank(), tag());
    start_recv(policy_comm{}, buffer(), nbytes(), MPI_BYTE, partner_rank(), tag(), comm, request);
  }

  void Irecv(mpi_type_pol const&, typename policy_comm::communicator_type comm, typename policy_comm::recv_request_type* request)
  {
    if (items.size() == 1) {
      DataT* dst = items.front().data;
      MPI_Datatype mpi_type = items.front().mpi_type;
      // FPRINTF(stdout, "%p Irecv %p to %i tag %i\n", this, dst, partner_rank(), tag());
      start_recv(policy_comm{}, dst, 1, mpi_type, partner_rank(), tag(), comm, request);
    } else {
      // FPRINTF(stdout, "%p Irecv %p maxnbytes %i to %i tag %i\n", this, dst, max_nbytes(), partner_rank(), tag());
      start_recv(policy_comm{}, buffer(), max_nbytes(), MPI_PACKED, partner_rank(), tag(), comm, request);
    }
  }

  template < typename policy >
  void allocate(policy const&, COMB::Allocator& buf_aloc)
  {
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(nbytes());
    }
  }

  void allocate(mpi_type_pol const&, COMB::Allocator& buf_aloc)
  {
    if (m_buf == nullptr) {
      if (items.size() == 1) {
        // no buffer needed
      } else {
        m_buf = (DataT*)buf_aloc.allocate(max_nbytes());
      }
    }
  }

  template < typename policy >
  void deallocate(policy const&, COMB::Allocator& buf_aloc)
  {
    if (m_buf != nullptr) {
      buf_aloc.deallocate(m_buf);
      m_buf = nullptr;
    }
  }

  void destroy()
  {
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      if (i->indices) {
        i->aloc.deallocate(i->indices); i->indices = nullptr;
      }
      if (i->mpi_type != MPI_DATATYPE_NULL) {
        detail::MPI::Type_free(&i->mpi_type); i->mpi_type = MPI_DATATYPE_NULL;
      }
    }
    items.clear();
  }

  ~Message()
  {
  }
};

#endif // _MESSAGE_HPP

