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

#ifndef _COMM_HPP
#define _COMM_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdarg>
#include <type_traits>
#include <list>
#include <vector>
#include <map>
#include <utility>

#include <mpi.h>

#include "memory.hpp"
#include "for_all.hpp"
#include "utils.hpp"

struct CartRank
{
  int rank;
  int coords[3];

  CartRank() : rank(-1), coords{-1, -1, -1} {}
  CartRank(int rank_, int coords_[]) : rank(rank_), coords{coords_[0], coords_[1], coords_[2]} {}

  CartRank& setup(int rank_, MPI_Comm cartcomm)
  {
    rank = rank_;
    detail::MPI::Cart_coords(cartcomm, rank, 3, coords);
    return *this;
  }
};

struct CartComm : CartRank
{
  MPI_Comm comm;
  int size;
  int divisions[3];
  int periodic[3];

  explicit CartComm()
    : CartRank()
    , comm(MPI_COMM_NULL)
    , size(0)
    , divisions{0, 0, 0}
    , periodic{0, 0, 0}
  {
  }

  CartComm(CartComm const& other)
    : CartRank(other)
    , comm(other.comm != MPI_COMM_NULL ? detail::MPI::Comm_dup(other.comm) : MPI_COMM_NULL)
    , size(other.size)
    , divisions{other.divisions[0], other.divisions[1], other.divisions[2]}
    , periodic{other.periodic[0], other.periodic[1], other.periodic[2]}
  {
  }

  // don't allow assignment
  CartComm& operator=(CartComm const&) = delete;

  void create(const int divisions_[], const int periodic_[])
  {
    divisions[0] = divisions_[0];
    divisions[1] = divisions_[1];
    divisions[2] = divisions_[2];

    periodic[0] = periodic_[0];
    periodic[1] = periodic_[1];
    periodic[2] = periodic_[2];

    comm = detail::MPI::Cart_create(MPI_COMM_WORLD, 3, divisions, periodic, 1);
    size = detail::MPI::Comm_size(comm);
    setup(detail::MPI::Comm_rank(comm), comm);
  }

  int get_rank(const int arg_coords[]) const
  {
    int output_rank = -1;
    int input_coords[3] {-1, -1, -1};
    for(IdxT dim = 0; dim < 3; ++dim) {
      input_coords[dim] = arg_coords[dim];
      if (periodic[dim]) {
        input_coords[dim] = input_coords[dim] % divisions[dim];
        if (input_coords[dim] < 0) input_coords[dim] += divisions[dim];
      }
      assert(0 <= input_coords[dim] && input_coords[dim] < divisions[dim]);
    }
    output_rank = detail::MPI::Cart_rank(comm, input_coords);
    return output_rank;
  }

  ~CartComm()
  {
    if (comm != MPI_COMM_NULL) {
      detail::MPI::Comm_free(&comm);
    }
  }
};

struct CommInfo
{
  int rank;
  int size;

  CartComm cart;

  bool mock_communication;

  IdxT cutoff;

  enum struct method : IdxT
  { waitany
  , testany
  , waitsome
  , testsome
  , waitall
  , testall };

  static const char* method_str(method m)
  {
    const char* str = "unknown";
    switch (m) {
      case method::waitany:  str = "wait_any";  break;
      case method::testany:  str = "test_any";  break;
      case method::waitsome: str = "wait_some"; break;
      case method::testsome: str = "test_some"; break;
      case method::waitall:  str = "wait_all";  break;
      case method::testall:  str = "test_all";  break;
    }
    return str;
  }

  method post_send_method;
  method post_recv_method;
  method wait_send_method;
  method wait_recv_method;

  CommInfo()
    : rank(-1)
    , size(0)
    , cart()
    , mock_communication(false)
    , cutoff(200)
    , post_send_method(method::waitall)
    , post_recv_method(method::waitall)
    , wait_send_method(method::waitall)
    , wait_recv_method(method::waitall)
  {
    rank = detail::MPI::Comm_rank(MPI_COMM_WORLD);
    size = detail::MPI::Comm_size(MPI_COMM_WORLD);
  }

  void barrier()
  {
    if (cart.comm != MPI_COMM_NULL) {
      detail::MPI::Barrier(cart.comm);
    } else {
      detail::MPI::Barrier(MPI_COMM_WORLD);
    }
  }

  void set_name(const char* name)
  {
    if (cart.comm != MPI_COMM_NULL) {
      detail::MPI::Comm_set_name(cart.comm, name);
    }
  }

  void print_any(const char* fmt, ...)
  {
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
  }

  void print_master(const char* fmt, ...)
  {
    if (rank == 0) {
      va_list args;
      va_start(args, fmt);
      vfprintf(stdout, fmt, args);
      va_end(args);
    }
  }

  void warn_any(const char* fmt, ...)
  {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
  }

  void warn_master(const char* fmt, ...)
  {
    if (rank == 0) {
      va_list args;
      va_start(args, fmt);
      vfprintf(stderr, fmt, args);
      va_end(args);
    }
  }

  void abort_any(const char* fmt, ...)
  {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
  }

  void abort_master(const char* fmt, ...)
  {
    if (rank == 0) {
      va_list args;
      va_start(args, fmt);
      vfprintf(stderr, fmt, args);
      va_end(args);
    }
    abort();
  }

  void abort()
  {
    detail::MPI::Abort(MPI_COMM_WORLD, 1);
  }
};

struct Message
{
  int m_partner_rank;
  int m_msg_tag;
  DataT* m_buf;
  IdxT m_size;
  bool m_have_many;

  struct list_item_type
  {
    DataT* data;
    LidxT* indices;
    Allocator& aloc;
    IdxT size;
    list_item_type(DataT* data_, LidxT* indices_, Allocator& aloc_, IdxT size_)
     : data(data_), indices(indices_), aloc(aloc_), size(size_)
    { }
  };

  std::list<list_item_type> items;

  Message(int partner_rank, int tag, bool have_many)
    : m_partner_rank(partner_rank)
    , m_msg_tag(tag)
    , m_buf(nullptr)
    , m_size(0)
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

  IdxT nbytes() const
  {
    return sizeof(DataT)*m_size;
  }

  bool have_many() const
  {
    return m_have_many;
  }

  void add(DataT* data, LidxT* indices, Allocator& aloc, IdxT size)
  {
    items.emplace_front(data, indices, aloc, size);
    m_size += size;
  }

  template < typename policy >
  void pack(policy&& pol)
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

  template < typename policy >
  void unpack(policy&& pol)
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

  void allocate(Allocator& buf_aloc)
  {
    if (m_buf == nullptr) {
      m_buf = (DataT*)buf_aloc.allocate(nbytes());
    }
  }

  void deallocate(Allocator& buf_aloc)
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
      i->aloc.deallocate(i->indices); i->indices = nullptr;
    }
    items.clear();
  }

  ~Message()
  {
  }
};

template < typename policy_many_, typename policy_few_ >
struct Comm
{
  using policy_many = policy_many_;
  using policy_few  = policy_few_;

  Allocator& many_aloc;
  Allocator& few_aloc;

  CommInfo comminfo;

  using message_type = Message;
  std::vector<message_type> m_sends;
  std::vector<message_type> m_recvs;

  std::vector<MPI_Request> m_send_requests;
  std::vector<MPI_Request> m_recv_requests;

  std::vector<typename policy_many::event_type> m_many_events;
  std::vector<typename policy_few::event_type> m_few_events;

  Comm(CommInfo const& comminfo_, Allocator& many_aloc_, Allocator& few_aloc_)
    : many_aloc(many_aloc_)
    , few_aloc(few_aloc_)
    , comminfo(comminfo_)
  {
    // if policies are the same set cutoff to 0 (always use policy_many) to simplify algorithms
    if (std::is_same<policy_many, policy_few>::value) {
      comminfo.cutoff = 0;
    }
  }


  void postRecv()
  {
    //FPRINTF(stdout, "posting receives\n");

    m_recv_requests.resize(m_recvs.size(), MPI_REQUEST_NULL);

    switch (comminfo.post_recv_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].allocate(many_aloc);
          } else {
            m_recvs[i].allocate(few_aloc);
          }

          if (!comminfo.mock_communication) {
            detail::MPI::Irecv( m_recvs[i].buffer(), m_recvs[i].nbytes(),
                                m_recvs[i].partner_rank(), m_recvs[i].tag(),
                                comminfo.cart.comm, &m_recv_requests[i] );
          }
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].allocate(many_aloc);
          } else {
            m_recvs[i].allocate(few_aloc);
          }

          if (!comminfo.mock_communication) {
            detail::MPI::Irecv( m_recvs[i].buffer(), m_recvs[i].nbytes(),
                                m_recvs[i].partner_rank(), m_recvs[i].tag(),
                                comminfo.cart.comm, &m_recv_requests[i] );
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].allocate(many_aloc);
          } else {
            m_recvs[i].allocate(few_aloc);
          }
        }
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (!comminfo.mock_communication) {
            detail::MPI::Irecv( m_recvs[i].buffer(), m_recvs[i].nbytes(),
                                m_recvs[i].partner_rank(), m_recvs[i].tag(),
                                comminfo.cart.comm, &m_recv_requests[i] );
          }
        }
      } break;
      default:
      {
        assert(0);
      } break;
    }
  }



  void postSend()
  {
    //FPRINTF(stdout, "posting sends\n");

    m_send_requests.resize(m_recvs.size(), MPI_REQUEST_NULL);

    switch (comminfo.post_send_method) {
      case CommInfo::method::waitany:
      {
        IdxT num_sends = m_sends.size();
        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].allocate(many_aloc);
            m_sends[i].pack(policy_many{});
            synchronize(policy_many{});
          } else {
            m_sends[i].allocate(few_aloc);
            m_sends[i].pack(policy_few{});
            synchronize(policy_few{});
          }

          if (!comminfo.mock_communication) {
            detail::MPI::Isend( m_sends[i].buffer(), m_sends[i].nbytes(),
                               m_sends[i].partner_rank(), m_sends[i].tag(),
                               comminfo.cart.comm, &m_send_requests[i] );
          }
        }
      } break;
      case CommInfo::method::testany:
      {
        IdxT num_sends = m_sends.size();
        bool have_many = false;
        bool have_few = false;

        // allocate
        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].allocate(many_aloc);
            have_many = true;
          } else {
            m_sends[i].allocate(few_aloc);
            have_few = true;
          }
        }

        // pack and send
        if (have_many && have_few) {
          persistent_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_launch(policy_many{});
        } else if (have_few) {
          persistent_launch(policy_few{});
        }

        bool post_pack_complete = false;
        IdxT pack_send = 0;
        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (post_many_send < num_sends || post_few_send < num_sends) {

          // pack and record events
          if (pack_send < num_sends) {

            if (m_sends[pack_send].have_many()) {
              m_sends[pack_send].pack(policy_many{});
              recordEvent(policy_many{}, m_many_events[pack_send]);
            } else {
              m_sends[pack_send].pack(policy_few{});
              recordEvent(policy_few{}, m_few_events[pack_send]);
            }

            ++pack_send;

          } else if (!post_pack_complete) {

            if (have_many && have_few) {
              batch_launch(policy_few{}, policy_many{});
            } else if (have_many) {
              batch_launch(policy_many{});
            } else if (have_few) {
              batch_launch(policy_few{});
            }

            // stop persistent kernel
            if (have_many && have_few) {
              persistent_stop(policy_few{}, policy_many{});
            } else if (have_many) {
              persistent_stop(policy_many{});
            } else if (have_few) {
              persistent_stop(policy_few{});
            }

            post_pack_complete = true;
          }

          while (post_many_send < pack_send) {

            if (m_sends[post_many_send].have_many()) {

              if (queryEvent(policy_many{}, m_many_events[post_many_send])) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[post_many_send].buffer(), m_sends[post_many_send].nbytes(),
                                      m_sends[post_many_send].partner_rank(), m_sends[post_many_send].tag(),
                                      comminfo.cart.comm, &m_send_requests[post_many_send] );
                }

                ++post_many_send;

              } else {
                break;
              }
            } else {

              ++post_many_send;
            }
          }

          while (post_few_send < pack_send) {

            if (!m_sends[post_few_send].have_many()) {

              if (queryEvent(policy_few{}, m_few_events[post_few_send])) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[post_few_send].buffer(), m_sends[post_few_send].nbytes(),
                                      m_sends[post_few_send].partner_rank(), m_sends[post_few_send].tag(),
                                      comminfo.cart.comm, &m_send_requests[post_few_send] );
                }

                ++post_few_send;

              } else {
                break;
              }

            } else {

              ++post_few_send;
            }
          }

        }
      } break;
      case CommInfo::method::waitsome:
      {
        IdxT num_sends = m_sends.size();

        {
          // have many case, send after all packed
          bool found_many = false;

          for (IdxT i = 0; i < num_sends; ++i) {

            if (m_sends[i].have_many()) {
              m_sends[i].allocate(many_aloc);
              m_sends[i].pack(policy_many{});
              found_many = true;
            }
          }

          if (found_many) {

            synchronize(policy_many{});

            for (IdxT i = 0; i < num_sends; ++i) {

              if (m_sends[i].have_many()) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[i].buffer(), m_sends[i].nbytes(),
                                      m_sends[i].partner_rank(), m_sends[i].tag(),
                                      comminfo.cart.comm, &m_send_requests[i] );
                }
              }
            }
          }
        }

        {
          // have_few case, send immediately
          for (IdxT i = 0; i < num_sends; ++i) {

            if (!m_sends[i].have_many()) {
              m_sends[i].allocate(few_aloc);
              m_sends[i].pack(policy_few{});

              synchronize(policy_few{});

              if (!comminfo.mock_communication) {
                detail::MPI::Isend( m_sends[i].buffer(), m_sends[i].nbytes(),
                                    m_sends[i].partner_rank(), m_sends[i].tag(),
                                    comminfo.cart.comm, &m_send_requests[i] );
              }
            }
          }
        }
      } break;
      case CommInfo::method::testsome:
      {
        IdxT num_sends = m_sends.size();

        bool have_many = false;
        bool have_few = false;

        // allocate
        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].allocate(many_aloc);
            have_many = true;
          } else {
            m_sends[i].allocate(few_aloc);
            have_few = true;
          }
        }

        IdxT pack_many_send = 0;
        IdxT pack_few_send = 0;
        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        if (have_many) {

          persistent_launch(policy_many{});

          while (pack_many_send < num_sends) {

            if (m_sends[pack_many_send].have_many()) {

              m_sends[pack_many_send].pack(policy_many{});

              recordEvent(policy_many{}, m_many_events[pack_many_send]);

            }

            ++pack_many_send;

            // post sends if possible
            while (post_many_send < pack_many_send) {

              if (m_sends[post_many_send].have_many()) {

                if (queryEvent(policy_many{}, m_many_events[post_many_send])) {

                  if (!comminfo.mock_communication) {
                    detail::MPI::Isend( m_sends[post_many_send].buffer(), m_sends[post_many_send].nbytes(),
                                        m_sends[post_many_send].partner_rank(), m_sends[post_many_send].tag(),
                                        comminfo.cart.comm, &m_send_requests[post_many_send] );
                  }

                  ++post_many_send;

                } else {

                  break;
                }
              } else {

                ++post_many_send;
              }
            }
          }

          batch_launch(policy_many{});
          persistent_stop(policy_many{});
        } else {
          pack_many_send = num_sends;
          post_many_send = num_sends;
        }

        if (have_few) {

          persistent_launch(policy_few{});

          while (pack_few_send < num_sends) {

            if (!m_sends[pack_few_send].have_many()) {

              m_sends[pack_few_send].pack(policy_few{});

              recordEvent(policy_few{}, m_few_events[pack_few_send]);

            }

            ++pack_few_send;

            // post more sends if possible
            while (post_many_send < pack_many_send) {

              if (m_sends[post_many_send].have_many()) {

                if (queryEvent(policy_many{}, m_many_events[post_many_send])) {

                  if (!comminfo.mock_communication) {
                    detail::MPI::Isend( m_sends[post_many_send].buffer(), m_sends[post_many_send].nbytes(),
                                        m_sends[post_many_send].partner_rank(), m_sends[post_many_send].tag(),
                                        comminfo.cart.comm, &m_send_requests[post_many_send] );
                  }

                  ++post_many_send;

                } else {

                  break;
                }
              } else {

                ++post_many_send;
              }
            }

            // post sends if possible
            while (post_few_send < pack_few_send) {

              if (!m_sends[post_few_send].have_many()) {

                if (queryEvent(policy_few{}, m_few_events[post_few_send])) {

                  if (!comminfo.mock_communication) {
                    detail::MPI::Isend( m_sends[post_few_send].buffer(), m_sends[post_few_send].nbytes(),
                                        m_sends[post_few_send].partner_rank(), m_sends[post_few_send].tag(),
                                        comminfo.cart.comm, &m_send_requests[post_few_send] );
                  }

                  ++post_few_send;

                } else {

                  break;
                }
              } else {

                ++post_few_send;
              }
            }
          }

          batch_launch(policy_few{});
          persistent_stop(policy_few{});
        } else {
          pack_few_send = num_sends;
          post_few_send = num_sends;
        }

        // finish posting sends
        while (post_many_send < num_sends || post_few_send < num_sends) {

          while (post_many_send < pack_many_send) {

            if (m_sends[post_many_send].have_many()) {

              if (queryEvent(policy_many{}, m_many_events[post_many_send])) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[post_many_send].buffer(), m_sends[post_many_send].nbytes(),
                                      m_sends[post_many_send].partner_rank(), m_sends[post_many_send].tag(),
                                      comminfo.cart.comm, &m_send_requests[post_many_send] );
                }

                ++post_many_send;

              } else {

                break;
              }
            } else {

              ++post_many_send;
            }
          }

          while (post_few_send < pack_few_send) {

            if (!m_sends[post_few_send].have_many()) {

              if (queryEvent(policy_few{}, m_few_events[post_few_send])) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[post_few_send].buffer(), m_sends[post_few_send].nbytes(),
                                      m_sends[post_few_send].partner_rank(), m_sends[post_few_send].tag(),
                                      comminfo.cart.comm, &m_send_requests[post_few_send] );
                }

                ++post_few_send;

              } else {

                break;
              }
            } else {

              ++post_few_send;
            }
          }

        }
      } break;
      case CommInfo::method::waitall:
      {
        IdxT num_sends = m_sends.size();
        bool have_many = false;
        bool have_few = false;

        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].allocate(many_aloc);
            m_sends[i].pack(policy_many{});
            have_many = true;
          } else {
            m_sends[i].allocate(few_aloc);
            m_sends[i].pack(policy_few{});
            have_few = true;
          }
        }

        if (have_many && have_few) {
          synchronize(policy_few{}, policy_many{});
        } else if (have_many) {
          synchronize(policy_many{});
        } else if (have_few) {
          synchronize(policy_few{});
        }

        for (IdxT i = 0; i < num_sends; ++i) {

          if (!comminfo.mock_communication) {
            detail::MPI::Isend( m_sends[i].buffer(), m_sends[i].nbytes(),
                               m_sends[i].partner_rank(), m_sends[i].tag(),
                               comminfo.cart.comm, &m_send_requests[i] );
          }
        }
      } break;
      case CommInfo::method::testall:
      {
        IdxT num_sends = m_sends.size();
        bool have_many = false;
        bool have_few = false;

        // allocate
        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].allocate(many_aloc);
            have_many = true;
          } else {
            m_sends[i].allocate(few_aloc);
            have_few = true;
          }
        }

        // pack and send
        if (have_many && have_few) {
          persistent_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_launch(policy_many{});
        } else if (have_few) {
          persistent_launch(policy_few{});
        }

        IdxT pack_send = 0;
        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (pack_send < num_sends) {

          // pack and record events
          if (m_sends[pack_send].have_many()) {
            m_sends[pack_send].pack(policy_many{});
            recordEvent(policy_many{}, m_many_events[pack_send]);
          } else {
            m_sends[pack_send].pack(policy_few{});
            recordEvent(policy_few{}, m_few_events[pack_send]);
          }

          ++pack_send;
        }

        if (have_many && have_few) {
          batch_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          batch_launch(policy_many{});
        } else if (have_few) {
          batch_launch(policy_few{});
        }

        // stop persistent kernel
        if (have_many && have_few) {
          persistent_stop(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_stop(policy_many{});
        } else if (have_few) {
          persistent_stop(policy_few{});
        }

        // post all sends
        while (post_many_send < num_sends || post_few_send < num_sends) {

          while (post_many_send < num_sends) {

            if (m_sends[post_many_send].have_many()) {

              if (queryEvent(policy_many{}, m_many_events[post_many_send])) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[post_many_send].buffer(), m_sends[post_many_send].nbytes(),
                                      m_sends[post_many_send].partner_rank(), m_sends[post_many_send].tag(),
                                      comminfo.cart.comm, &m_send_requests[post_many_send] );
                }

                ++post_many_send;

              } else {
                break;
              }
            } else {

              ++post_many_send;
            }
          }

          while (post_few_send < num_sends) {

            if (!m_sends[post_few_send].have_many()) {

              if (queryEvent(policy_few{}, m_few_events[post_few_send])) {

                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[post_few_send].buffer(), m_sends[post_few_send].nbytes(),
                                      m_sends[post_few_send].partner_rank(), m_sends[post_few_send].tag(),
                                      comminfo.cart.comm, &m_send_requests[post_few_send] );
                }

                ++post_few_send;

              } else {
                break;
              }

            } else {

              ++post_few_send;
            }
          }

        }
      } break;
      default:
      {
       assert(0);
      } break;
    }
  }



  void waitRecv()
  {
    //FPRINTF(stdout, "waiting receives\n");

    bool have_many = false;
    bool have_few = false;

    switch (comminfo.wait_recv_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_recvs = m_recvs.size();

        for (IdxT i = 0; i < num_recvs; ++i) {
          if (m_recvs[i].have_many()) {
            have_many = true;
          } else {
            have_few = true;
          }
        }

        if (have_many && have_few) {
          persistent_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_launch(policy_many{});
        } else if (have_few) {
          persistent_launch(policy_few{});
        }

        MPI_Status status;

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          IdxT idx = num_done;
          if (!comminfo.mock_communication) {
            if (comminfo.wait_recv_method == CommInfo::method::waitany) {
              idx = detail::MPI::Waitany( num_recvs, &m_recv_requests[0], &status);
            } else {
              idx = -1;
              while(idx < 0 || idx >= num_recvs) {
                idx = detail::MPI::Testany( num_recvs, &m_recv_requests[0], &status);
              }
            }
          }

          if (m_recvs[idx].have_many()) {
            m_recvs[idx].unpack(policy_many{});
            m_recvs[idx].deallocate(many_aloc);
            batch_launch(policy_many{});
          } else {
            m_recvs[idx].unpack(policy_few{});
            m_recvs[idx].deallocate(few_aloc);
            batch_launch(policy_few{});
          }

          num_done += 1;

        }

        // if (have_many && have_few) {
        //   batch_launch(policy_few{}, policy_many{});
        // } else if (have_many) {
        //   batch_launch(policy_many{});
        // } else if (have_few) {
        //   batch_launch(policy_few{});
        // }

        if (have_many && have_few) {
          persistent_stop(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_stop(policy_many{});
        } else if (have_few) {
          persistent_stop(policy_few{});
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_recvs = m_recvs.size();

        for (IdxT i = 0; i < num_recvs; ++i) {
          if (m_recvs[i].have_many()) {
            have_many = true;
          } else {
            have_few = true;
          }
        }

        if (have_many && have_few) {
          persistent_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_launch(policy_many{});
        } else if (have_few) {
          persistent_launch(policy_few{});
        }

        std::vector<MPI_Status> recv_statuses(m_recv_requests.size());
        std::vector<int> indices(m_recv_requests.size(), -1);

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          IdxT num = num_recvs;
          if (!comminfo.mock_communication) {
            if (comminfo.wait_recv_method == CommInfo::method::waitsome) {
              num = detail::MPI::Waitsome( num_recvs, &m_recv_requests[0], &indices[0], &recv_statuses[0]);
            } else {
              while( 0 == (num = detail::MPI::Testsome( num_recvs, &m_recv_requests[0], &indices[0], &recv_statuses[0])) );
            }
          } else {
            for (IdxT i = 0; i < num; ++i) {
              indices[i] = num_done + i;
            }
          }

          bool inner_have_many = false;
          bool inner_have_few = false;

          for (IdxT i = 0; i < num; ++i) {

            if (m_recvs[indices[i]].have_many()) {

              m_recvs[indices[i]].unpack(policy_many{});
              m_recvs[indices[i]].deallocate(many_aloc);

              inner_have_many = true;

              num_done += 1;
            }
          }

          if (inner_have_many) {
            batch_launch(policy_many{});
          }

          for (IdxT i = 0; i < num; ++i) {

            if (!m_recvs[indices[i]].have_many()) {

              m_recvs[indices[i]].unpack(policy_few{});
              m_recvs[indices[i]].deallocate(few_aloc);

              inner_have_few = true;

              num_done += 1;
            }
          }

          if (inner_have_few) {
            batch_launch(policy_few{});
          }
        }

        if (have_many && have_few) {
          persistent_stop(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_stop(policy_many{});
        } else if (have_few) {
          persistent_stop(policy_few{});
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_recvs = m_recvs.size();

        for (IdxT i = 0; i < num_recvs; ++i) {
          if (m_recvs[i].have_many()) {
            have_many = true;
          } else {
            have_few = true;
          }
        }

        if (have_many && have_few) {
          persistent_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_launch(policy_many{});
        } else if (have_few) {
          persistent_launch(policy_few{});
        }

        std::vector<MPI_Status> recv_statuses(m_recv_requests.size());

        if (!comminfo.mock_communication) {
          if (comminfo.wait_recv_method == CommInfo::method::waitall) {
            detail::MPI::Waitall( num_recvs, &m_recv_requests[0], &recv_statuses[0]);
          } else {
            while (!detail::MPI::Testall( num_recvs, &m_recv_requests[0], &recv_statuses[0]));
          }
        }

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          if (m_recvs[num_done].have_many()) {
            m_recvs[num_done].unpack(policy_many{});
            m_recvs[num_done].deallocate(many_aloc);
            have_many = true;
          } else {
            m_recvs[num_done].unpack(policy_few{});
            m_recvs[num_done].deallocate(few_aloc);
            have_few = true;
          }

          num_done += 1;
        }

        if (have_many && have_few) {
          batch_launch(policy_few{}, policy_many{});
        } else if (have_many) {
          batch_launch(policy_many{});
        } else if (have_few) {
          batch_launch(policy_few{});
        }

        if (have_many && have_few) {
          persistent_stop(policy_few{}, policy_many{});
        } else if (have_many) {
          persistent_stop(policy_many{});
        } else if (have_few) {
          persistent_stop(policy_few{});
        }
      } break;
      default:
      {
        assert(0);
      } break;
    }

    m_recv_requests.clear();

    if (have_many && have_few) {
      synchronize(policy_few{}, policy_many{});
    } else if (have_many) {
      synchronize(policy_many{});
    } else if (have_few) {
      synchronize(policy_few{});
    }
  }



  void waitSend()
  {
    //FPRINTF(stdout, "posting sends\n");

    switch (comminfo.wait_send_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_sends = m_sends.size();
        IdxT num_done = 0;

        MPI_Status status;

        while (num_done < num_sends) {

          IdxT idx = num_done;
          if (!comminfo.mock_communication) {
            if (comminfo.wait_send_method == CommInfo::method::waitany) {
              idx = detail::MPI::Waitany( num_sends, &m_send_requests[0], &status);
            } else {
              idx = -1;
              while(idx < 0 || idx >= num_sends) {
                idx = detail::MPI::Testany( num_sends, &m_send_requests[0], &status);
              }
            }
          }

          if (m_sends[idx].have_many()) {
            m_sends[idx].deallocate(many_aloc);
          } else {
            m_sends[idx].deallocate(few_aloc);
          }

          num_done += 1;

        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_sends = m_sends.size();
        IdxT num_done = 0;

        std::vector<MPI_Status> send_statuses(m_send_requests.size());
        std::vector<int> indices(m_send_requests.size(), -1);

        while (num_done < num_sends) {

          IdxT num = num_sends;
          if (!comminfo.mock_communication) {
            if (comminfo.wait_send_method == CommInfo::method::waitsome) {
              num = detail::MPI::Waitsome( num_sends, &m_send_requests[0], &indices[0], &send_statuses[0]);
            } else {
              num = detail::MPI::Testsome( num_sends, &m_send_requests[0], &indices[0], &send_statuses[0]);
            }
          } else {
            for (IdxT i = 0; i < num; ++i) {
              indices[i] = num_done + i;
            }
          }

          for (IdxT i = 0; i < num; ++i) {

            IdxT idx = indices[i];
            if (m_sends[idx].have_many()) {
              m_sends[idx].deallocate(many_aloc);
            } else {
              m_sends[idx].deallocate(few_aloc);
            }

            num_done += 1;

          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_sends = m_sends.size();
        IdxT num_done = 0;

        std::vector<MPI_Status> send_statuses(m_send_requests.size());

        if (!comminfo.mock_communication) {
          if (comminfo.wait_send_method == CommInfo::method::waitall) {
            detail::MPI::Waitall( num_sends, &m_send_requests[0], &send_statuses[0]);
          } else {
            while(!detail::MPI::Testall( num_sends, &m_send_requests[0], &send_statuses[0]));
          }
        }

        while (num_done < num_sends) {

          IdxT idx = num_done;
          if (m_sends[idx].have_many()) {
            m_sends[idx].deallocate(many_aloc);
          } else {
            m_sends[idx].deallocate(few_aloc);
          }

          num_done += 1;
        }
      } break;
      default:
      {
        assert(0);
      } break;
    }

    m_send_requests.clear();
  }

  ~Comm()
  {
    size_t num_events = m_many_events.size();
    for(size_t i = 0; i != num_events; ++i) {
      destroyEvent(policy_many{}, m_many_events[i]);
    }
    num_events = m_few_events.size();
    for(size_t i = 0; i != num_events; ++i) {
      destroyEvent(policy_few{}, m_few_events[i]);
    }
    for(message_type& msg : m_sends) {
      msg.destroy();
    }
    for(message_type& msg : m_recvs) {
      msg.destroy();
    }
  }
};

#endif // _COMM_HPP

