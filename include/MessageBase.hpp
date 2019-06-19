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

#ifndef _MESSAGE_HPP
#define _MESSAGE_HPP

#include "config.hpp"

#include <type_traits>
#include <list>
#include <utility>

#include "memory.hpp"
#include "for_all.hpp"
#include "utils.hpp"

namespace detail {

struct MessageBase
{
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

  MessageBase(int partner_rank, int tag, bool have_many)
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

  ~MessageBase()
  {
  }
};

} // namespace detail


template < typename policy_comm_ >
struct Message;


#endif // _MESSAGE_HPP

