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

#ifndef _COMM_HPP
#define _COMM_HPP

#include "config.hpp"

#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <list>
#include <vector>
#include <map>
#include <utility>

#ifdef COMB_ENABLE_MPI
#include <mpi.h>
#endif

#include "print.hpp"
#include "memory.hpp"
#include "exec_utils.hpp"
#include "exec.hpp"

#include "MessageBase.hpp"


struct CartRank
{
  int rank;
  int coords[3];

  CartRank() : rank(-1), coords{-1, -1, -1} {}
  CartRank(int rank_, int coords_[]) : rank(rank_), coords{coords_[0], coords_[1], coords_[2]} {}

  CartRank& setup(int rank_
#ifdef COMB_ENABLE_MPI
                 ,MPI_Comm cartcomm
#endif
                  )
  {
    rank = rank_;
#ifdef COMB_ENABLE_MPI
    detail::MPI::Cart_coords(cartcomm, rank, 3, coords);
#else
    coords[0] = 0; coords[1] = 0; coords[2] = 0;
#endif
    return *this;
  }
};

struct CartComm : CartRank
{
#ifdef COMB_ENABLE_MPI
  MPI_Comm comm;
#endif
  int size;
  int divisions[3];
  int periodic[3];

  explicit CartComm()
    : CartRank()
#ifdef COMB_ENABLE_MPI
    , comm(MPI_COMM_NULL)
#endif
    , size(0)
    , divisions{0, 0, 0}
    , periodic{0, 0, 0}
  {
  }

  CartComm(CartComm const& other)
    : CartRank(other)
#ifdef COMB_ENABLE_MPI
    , comm(other.comm != MPI_COMM_NULL ? detail::MPI::Comm_dup(other.comm) : MPI_COMM_NULL)
#endif
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

#ifdef COMB_ENABLE_MPI
    comm = detail::MPI::Cart_create(MPI_COMM_WORLD, 3, divisions, periodic, 1);
    size = detail::MPI::Comm_size(comm);
    setup(detail::MPI::Comm_rank(comm), comm);
#else
    size = 1;
    setup(0);
#endif
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
#ifdef COMB_ENABLE_MPI
    output_rank = detail::MPI::Cart_rank(comm, input_coords);
#else
    output_rank = 0;
#endif
    return output_rank;
  }

  ~CartComm()
  {
#ifdef COMB_ENABLE_MPI
    if (comm != MPI_COMM_NULL) {
      detail::MPI::Comm_free(&comm);
    }
#endif
  }
};

struct CommInfo
{
  int rank;
  int size;

  CartComm cart;

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
    , cutoff(200)
    , post_send_method(method::waitall)
    , post_recv_method(method::waitall)
    , wait_send_method(method::waitall)
    , wait_recv_method(method::waitall)
  {
#ifdef COMB_ENABLE_MPI
    rank = detail::MPI::Comm_rank(MPI_COMM_WORLD);
    size = detail::MPI::Comm_size(MPI_COMM_WORLD);
#else
    rank = 0;
    size = 1;
#endif
  }

  void barrier()
  {
#ifdef COMB_ENABLE_MPI
    if (cart.comm != MPI_COMM_NULL) {
      detail::MPI::Barrier(cart.comm);
    } else {
      detail::MPI::Barrier(MPI_COMM_WORLD);
    }
#endif
  }

  void set_name(const char* name)
  {
#ifdef COMB_ENABLE_MPI
    if (cart.comm != MPI_COMM_NULL) {
      detail::MPI::Comm_set_name(cart.comm, name);
    }
#else
    COMB::ignore_unused(name);
#endif
  }

  void abort()
  {
#ifdef COMB_ENABLE_MPI
    detail::MPI::Abort(MPI_COMM_WORLD, 1);
#else
    std::abort();
#endif
  }
};

template < typename policy_many_, typename policy_few_, typename policy_comm_ >
struct Comm
{
  using policy_many = policy_many_;
  using policy_few  = policy_few_;
  using policy_comm  = policy_comm_;

#ifdef COMB_ENABLE_MPI
  static constexpr bool pol_many_is_mpi_type = std::is_same<policy_many, mpi_type_pol>::value;
  static constexpr bool pol_few_is_mpi_type  = std::is_same<policy_few,  mpi_type_pol>::value;
  static constexpr bool use_mpi_type = pol_many_is_mpi_type && pol_few_is_mpi_type;

  // check policies are consistent
  static_assert(pol_many_is_mpi_type == pol_few_is_mpi_type,
      "pol_many and pol_few must both be mpi_type_pol if either is mpi_type_pol");
#endif

  COMB::Allocator& mesh_aloc;
  COMB::Allocator& many_aloc;
  COMB::Allocator& few_aloc;

  CommInfo& comminfo;

  CommContext<policy_comm>& con_comm;

  CommInfo::method post_recv_method;
  CommInfo::method post_send_method;
  CommInfo::method wait_recv_method;
  CommInfo::method wait_send_method;

  using send_message_type = detail::Message<detail::MessageBase::Kind::send, policy_comm>;
  using recv_message_type = detail::Message<detail::MessageBase::Kind::recv, policy_comm>;

  using send_message_group_many_type = detail::MessageGroup<detail::MessageBase::Kind::send, policy_comm, policy_many>;
  using send_message_group_few_type  = detail::MessageGroup<detail::MessageBase::Kind::send, policy_comm, policy_few>;

  using recv_message_group_many_type = detail::MessageGroup<detail::MessageBase::Kind::recv, policy_comm, policy_many>;
  using recv_message_group_few_type  = detail::MessageGroup<detail::MessageBase::Kind::recv, policy_comm, policy_few>;

  using send_request_type = typename policy_comm::send_request_type;
  using recv_request_type = typename policy_comm::recv_request_type;


  struct send_message_vars_s
  {
    send_message_vars_s(COMB::Allocator& many_aloc_, COMB::Allocator& few_aloc_)
      : message_group_many(many_aloc_)
      , message_group_few(few_aloc_)
    { }

    send_message_group_many_type message_group_many;
    send_message_group_few_type message_group_few;
    std::vector<send_message_type*> messages;
    std::vector<send_request_type> requests;
  };

  send_message_vars_s m_sends;


  struct recv_message_vars_s
  {
    recv_message_vars_s(COMB::Allocator& many_aloc_, COMB::Allocator& few_aloc_)
      : message_group_many(many_aloc_)
      , message_group_few(few_aloc_)
    { }

    recv_message_group_many_type message_group_many;
    recv_message_group_few_type message_group_few;
    std::vector<recv_message_type*> messages;
    std::vector<recv_request_type> requests;
  };

  recv_message_vars_s m_recvs;


  Comm(CommContext<policy_comm>& con_comm_, CommInfo& comminfo_,
       COMB::Allocator& mesh_aloc_, COMB::Allocator& many_aloc_, COMB::Allocator& few_aloc_)
    : mesh_aloc(mesh_aloc_)
    , many_aloc(many_aloc_)
    , few_aloc(few_aloc_)
    , comminfo(comminfo_)
    , con_comm(con_comm_)
    , post_recv_method(comminfo_.post_recv_method)
    , post_send_method(comminfo_.post_send_method)
    , wait_recv_method(comminfo_.wait_recv_method)
    , wait_send_method(comminfo_.wait_send_method)
    , m_sends(many_aloc_, few_aloc_)
    , m_recvs(many_aloc_, few_aloc_)
  { }

  void finish_populating(ExecContext<policy_many>& con_many, ExecContext<policy_few>& con_few)
  {
    COMB::ignore_unused(con_many, con_few);
    //FGPRINTF(FileGroup::proc, "finish populating comm\n");

    std::vector<int> send_ranks;
    std::vector<int> recv_ranks;
    for (send_message_type& msg : m_sends.message_group_many.messages) {
      send_ranks.emplace_back(msg.partner_rank);
    }
    for (send_message_type& msg : m_sends.message_group_few.messages) {
      send_ranks.emplace_back(msg.partner_rank);
    }
    for (recv_message_type& msg : m_recvs.message_group_many.messages) {
      recv_ranks.emplace_back(msg.partner_rank);
    }
    for (recv_message_type& msg : m_recvs.message_group_few.messages) {
      recv_ranks.emplace_back(msg.partner_rank);
    }
    con_comm.connect_ranks(send_ranks, recv_ranks);

    con_comm.setup_mempool(many_aloc, few_aloc);
  }

  ~Comm()
  {
    con_comm.teardown_mempool();

    std::vector<int> send_ranks;
    std::vector<int> recv_ranks;
    for (send_message_type& msg : m_sends.message_group_many.messages) {
      send_ranks.emplace_back(msg.partner_rank);
    }
    for (send_message_type& msg : m_sends.message_group_few.messages) {
      send_ranks.emplace_back(msg.partner_rank);
    }
    for (recv_message_type& msg : m_recvs.message_group_many.messages) {
      recv_ranks.emplace_back(msg.partner_rank);
    }
    for (recv_message_type& msg : m_recvs.message_group_few.messages) {
      recv_ranks.emplace_back(msg.partner_rank);
    }
    con_comm.disconnect_ranks(send_ranks, recv_ranks);
  }

  bool mock_communication() const
  {
    return policy_comm::mock;
  }

  void barrier()
  {
    comminfo.barrier();
  }

  void postRecv(ExecContext<policy_many>& con_many, ExecContext<policy_few>& con_few)
  {
    COMB::ignore_unused(con_many, con_few);
    //FGPRINTF(FileGroup::proc, "posting receives\n");

    IdxT num_many = m_recvs.message_group_many.messages.size();
    IdxT num_few = m_recvs.message_group_few.messages.size();

    IdxT num_recvs = num_many + num_few;

    m_recvs.messages.resize(num_recvs, nullptr);
    m_recvs.requests.resize(num_recvs, con_comm.recv_request_null());

    recv_message_type** messages_many = &m_recvs.messages[0];
    recv_message_type** messages_few  = &m_recvs.messages[num_many];

    recv_request_type* requests_many = &m_recvs.requests[0];
    recv_request_type* requests_few  = &m_recvs.requests[num_many];

    switch (post_recv_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_recvs.message_group_many.messages[i_many];
          m_recvs.message_group_many.allocate(con_many, con_comm, &messages_many[i_many], 1);
          m_recvs.message_group_many.Irecv(con_many, con_comm, &messages_many[i_many], 1, &requests_many[i_many]);
        }

        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_recvs.message_group_few.messages[i_few];
          m_recvs.message_group_few.allocate(con_few, con_comm, &messages_few[i_few], 1);
          m_recvs.message_group_few.Irecv(con_few, con_comm, &messages_few[i_few], 1, &requests_few[i_few]);
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_recvs.message_group_many.messages[i_many];
        }
        m_recvs.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_recvs.message_group_many.Irecv(con_many, con_comm, &messages_many[0], num_many, &requests_many[0]);

        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_recvs.message_group_few.messages[i_few];
        }
        m_recvs.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);
        m_recvs.message_group_few.Irecv(con_few, con_comm, &messages_few[0], num_few, &requests_few[0]);
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_recvs.message_group_many.messages[i_many];
        }
        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_recvs.message_group_few.messages[i_few];
        }

        m_recvs.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_recvs.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);

        m_recvs.message_group_many.Irecv(con_many, con_comm, &messages_many[0], num_many, &requests_many[0]);
        m_recvs.message_group_few.Irecv(con_few, con_comm, &messages_few[0], num_few, &requests_few[0]);
      } break;
      default:
      {
        assert(0);
      } break;
    }
  }



  void postSend(ExecContext<policy_many>& con_many, ExecContext<policy_few>& con_few)
  {
    COMB::ignore_unused(con_many, con_few);
    //FGPRINTF(FileGroup::proc, "posting sends\n");

    IdxT num_many = m_sends.message_group_many.messages.size();
    IdxT num_few = m_sends.message_group_few.messages.size();

    IdxT num_sends = num_many + num_few;

    m_sends.messages.resize(num_sends, nullptr);
    m_sends.requests.resize(num_sends, con_comm.send_request_null());

    send_message_type** messages_many = &m_sends.messages[0];
    send_message_type** messages_few  = &m_sends.messages[num_many];

    send_request_type* requests_many = &m_sends.requests[0];
    send_request_type* requests_few  = &m_sends.requests[num_many];

    switch (post_send_method) {
      case CommInfo::method::waitany:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_sends.message_group_many.messages[i_many];
          m_sends.message_group_many.allocate(con_many, con_comm, &messages_many[i_many], 1);
          m_sends.message_group_many.pack(con_many, con_comm, &messages_many[i_many], 1, detail::Async::no);
          m_sends.message_group_many.wait_pack_complete(con_many, con_comm, &messages_many[i_many], 1, detail::Async::no);
          m_sends.message_group_many.Isend(con_many, con_comm, &messages_many[i_many], 1, &requests_many[i_many]);
        }

        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_sends.message_group_few.messages[i_few];
          m_sends.message_group_few.allocate(con_few, con_comm, &messages_few[i_few], 1);
          m_sends.message_group_few.pack(con_few, con_comm, &messages_few[i_few], 1, detail::Async::no);
          m_sends.message_group_few.wait_pack_complete(con_few, con_comm, &messages_few[i_few], 1, detail::Async::no);
          m_sends.message_group_few.Isend(con_few, con_comm, &messages_few[i_few], 1, &requests_few[i_few]);
        }
      } break;
      case CommInfo::method::testany:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_sends.message_group_many.messages[i_many];
        }
        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_sends.message_group_few.messages[i_few];
        }

        m_sends.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_sends.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);

        IdxT num_many = m_sends.message_group_many.messages.size();
        IdxT num_few = m_sends.message_group_few.messages.size();

        IdxT pack_many_send = 0;
        IdxT pack_few_send = 0;

        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (post_many_send < num_many || post_few_send < num_few) {

          // pack and record events
          if (pack_many_send < num_many) {

            m_sends.message_group_many.pack(con_many, con_comm, &messages_many[pack_many_send], 1, detail::Async::yes);
            ++pack_many_send;

          } else
          if (pack_few_send < num_few) {

            m_sends.message_group_few.pack(con_few, con_comm, &messages_few[pack_few_send], 1, detail::Async::yes);
            ++pack_few_send;
          }

          int next_post_many_send = post_many_send;

          // have_many query events
          if (next_post_many_send < pack_many_send) {

            next_post_many_send += m_sends.message_group_many.wait_pack_complete(con_many, con_comm, &messages_many[next_post_many_send], pack_many_send-next_post_many_send, detail::Async::yes);
          }

          // have_many isends
          if (post_many_send < next_post_many_send) {

            m_sends.message_group_many.Isend(con_many, con_comm, &messages_many[post_many_send], next_post_many_send-post_many_send, &requests_many[post_many_send]);
            post_many_send = next_post_many_send;
          }

          int next_post_few_send = post_few_send;

          // have_few query events
          if (next_post_few_send < pack_few_send) {

            next_post_few_send += m_sends.message_group_few.wait_pack_complete(con_few, con_comm, &messages_few[next_post_few_send], pack_few_send-next_post_few_send, detail::Async::yes);
          }

          // have_few isends
          if (post_few_send < next_post_few_send) {

            m_sends.message_group_few.Isend(con_few, con_comm, &messages_few[post_few_send], next_post_few_send-post_few_send, &requests_few[post_few_send]);
            post_few_send = next_post_few_send;
          }
        }
      } break;
      case CommInfo::method::waitsome:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_sends.message_group_many.messages[i_many];
        }
        m_sends.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_sends.message_group_many.pack(con_many, con_comm, &messages_many[0], num_many, detail::Async::no);
        m_sends.message_group_many.wait_pack_complete(con_many, con_comm, &messages_many[0], num_many, detail::Async::no);
        m_sends.message_group_many.Isend(con_many, con_comm, &messages_many[0], num_many, &requests_many[0]);

        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_sends.message_group_few.messages[i_few];
        }
        m_sends.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);
        m_sends.message_group_few.pack(con_few, con_comm, &messages_few[0], num_few, detail::Async::no);
        m_sends.message_group_few.wait_pack_complete(con_few, con_comm, &messages_few[0], num_few, detail::Async::no);
        m_sends.message_group_few.Isend(con_few, con_comm, &messages_few[0], num_few, &requests_few[0]);
      } break;
      case CommInfo::method::testsome:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_sends.message_group_many.messages[i_many];
        }
        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_sends.message_group_few.messages[i_few];
        }

        m_sends.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_sends.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);

        IdxT num_many = m_sends.message_group_many.messages.size();
        IdxT num_few = m_sends.message_group_few.messages.size();

        IdxT pack_many_send = 0;
        IdxT pack_few_send = 0;

        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (post_many_send < num_many || post_few_send < num_few) {

          // pack and record events
          if (pack_many_send < num_many) {

            m_sends.message_group_many.pack(con_many, con_comm, &messages_many[pack_many_send], num_many-pack_many_send, detail::Async::yes);
            pack_many_send = num_many;

          } else
          if (pack_few_send < num_few) {

            m_sends.message_group_few.pack(con_few, con_comm, &messages_few[pack_few_send], num_few-pack_few_send, detail::Async::yes);
            pack_few_send = num_few;
          }

          int next_post_many_send = post_many_send;

          // have_many query events
          if (next_post_many_send < pack_many_send) {

            next_post_many_send += m_sends.message_group_many.wait_pack_complete(con_many, con_comm, &messages_many[next_post_many_send], pack_many_send-next_post_many_send, detail::Async::yes);
          }

          // have_many isends
          if (post_many_send < next_post_many_send) {

            m_sends.message_group_many.Isend(con_many, con_comm, &messages_many[post_many_send], next_post_many_send-post_many_send, &requests_many[post_many_send]);
            post_many_send = next_post_many_send;
          }

          int next_post_few_send = post_few_send;

          // have_few query events
          if (next_post_few_send < pack_few_send) {

            next_post_few_send += m_sends.message_group_few.wait_pack_complete(con_few, con_comm, &messages_few[next_post_few_send], pack_few_send-next_post_few_send, detail::Async::yes);
          }

          // have_few isends
          if (post_few_send < next_post_few_send) {

            m_sends.message_group_few.Isend(con_few, con_comm, &messages_few[post_few_send], next_post_few_send-post_few_send, &requests_few[post_few_send]);
            post_few_send = next_post_few_send;
          }
        }
      } break;
      case CommInfo::method::waitall:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_sends.message_group_many.messages[i_many];
        }
        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_sends.message_group_few.messages[i_few];
        }

        m_sends.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_sends.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);

        m_sends.message_group_many.pack(con_many, con_comm, &messages_many[0], num_many, detail::Async::no);
        m_sends.message_group_few.pack(con_few, con_comm, &messages_few[0], num_few, detail::Async::no);

        m_sends.message_group_many.wait_pack_complete(con_many, con_comm, &messages_many[0], num_many, detail::Async::no);
        m_sends.message_group_few.wait_pack_complete(con_few, con_comm, &messages_few[0], num_few, detail::Async::no);

        m_sends.message_group_many.Isend(con_many, con_comm, &messages_many[0], num_many, &requests_many[0]);
        m_sends.message_group_few.Isend(con_few, con_comm, &messages_few[0], num_few, &requests_few[0]);
      } break;
      case CommInfo::method::testall:
      {
        for (IdxT i_many = 0; i_many < num_many; i_many++) {
          messages_many[i_many] = &m_sends.message_group_many.messages[i_many];
        }
        for (IdxT i_few = 0; i_few < num_few; i_few++) {
          messages_few[i_few] = &m_sends.message_group_few.messages[i_few];
        }

        m_sends.message_group_many.allocate(con_many, con_comm, &messages_many[0], num_many);
        m_sends.message_group_few.allocate(con_few, con_comm, &messages_few[0], num_few);

        IdxT num_many = m_sends.message_group_many.messages.size();
        IdxT num_few = m_sends.message_group_few.messages.size();

        IdxT pack_many_send = 0;
        IdxT pack_few_send = 0;

        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (post_many_send < num_many || post_few_send < num_few) {

          // pack and record events
          if (pack_many_send < num_many) {

            m_sends.message_group_many.pack(con_many, con_comm, &messages_many[pack_many_send], num_many-pack_many_send, detail::Async::yes);
            pack_many_send = num_many;

          }
          if (pack_few_send < num_few) {

            m_sends.message_group_few.pack(con_few, con_comm, &messages_few[pack_few_send], num_few-pack_few_send, detail::Async::yes);
            pack_few_send = num_few;
          }

          int next_post_many_send = post_many_send;

          // have_many query events
          if (next_post_many_send < pack_many_send) {

            next_post_many_send += m_sends.message_group_many.wait_pack_complete(con_many, con_comm, &messages_many[next_post_many_send], pack_many_send-next_post_many_send, detail::Async::yes);
          }

          // have_many isends
          if (post_many_send < next_post_many_send) {

            m_sends.message_group_many.Isend(con_many, con_comm, &messages_many[post_many_send], next_post_many_send-post_many_send, &requests_many[post_many_send]);
            post_many_send = next_post_many_send;
          }

          int next_post_few_send = post_few_send;

          // have_few query events
          if (next_post_few_send < pack_few_send) {

            next_post_few_send += m_sends.message_group_few.wait_pack_complete(con_few, con_comm, &messages_few[next_post_few_send], pack_few_send-next_post_few_send, detail::Async::yes);
          }

          // have_few isends
          if (post_few_send < next_post_few_send) {

            m_sends.message_group_few.Isend(con_few, con_comm, &messages_few[post_few_send], next_post_few_send-post_few_send, &requests_few[post_few_send]);
            post_few_send = next_post_few_send;
          }
        }
      } break;
      default:
      {
       assert(0);
      } break;
    }
  }



  void waitRecv(ExecContext<policy_many>& con_many, ExecContext<policy_few>& con_few)
  {
    //FGPRINTF(FileGroup::proc, "waiting receives\n");

    IdxT num_many = m_recvs.message_group_many.messages.size();
    IdxT num_few = m_recvs.message_group_few.messages.size();

    IdxT num_recvs = num_many + num_few;

    recv_message_type** messages = &m_recvs.messages[0];

    recv_message_type** messages_many = &m_recvs.messages[0];
    recv_message_type** messages_few  = &m_recvs.messages[num_many];

    recv_request_type* requests = &m_recvs.requests[0];

    using recv_status_type = typename policy_comm::recv_status_type;
    std::vector<recv_status_type> recv_statuses(num_recvs, con_comm.recv_status_null());

    switch (wait_recv_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_done = 0;
        while (num_done < num_recvs) {

          IdxT idx = num_recvs;
          if (wait_recv_method == CommInfo::method::waitany) {
            idx = recv_message_type::wait_recv_any(con_comm, num_recvs, &requests[0], &recv_statuses[0]);
          } else {
            while(idx < 0 || idx >= num_recvs) {
              idx = recv_message_type::test_recv_any(con_comm, num_recvs, &requests[0], &recv_statuses[0]);
            }
          }
              assert(0 <= idx && idx < num_recvs);
              assert(requests > (recv_request_type*)0x1);

          if (idx < num_many) {
            m_recvs.message_group_many.unpack(con_many, con_comm, &messages[idx], 1);
              assert(0 <= idx && idx < num_recvs);
              assert(requests > (recv_request_type*)0x1);
            m_recvs.message_group_many.deallocate(con_many, con_comm, &messages[idx], 1);
              assert(0 <= idx && idx < num_recvs);
              assert(requests > (recv_request_type*)0x1);
          } else if (idx < num_recvs) {
            m_recvs.message_group_few.unpack(con_few, con_comm, &messages[idx], 1);
              assert(0 <= idx && idx < num_recvs);
              assert(requests > (recv_request_type*)0x1);
            m_recvs.message_group_few.deallocate(con_few, con_comm, &messages[idx], 1);
              assert(0 <= idx && idx < num_recvs);
              assert(requests > (recv_request_type*)0x1);
          } else {
            assert(0 <= idx && idx < num_recvs);
          }

          num_done += 1;

        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        std::vector<int> indices(num_recvs, -1);
        std::vector<recv_message_type*> recvd_messages(num_recvs, nullptr);

        recv_message_type** recvd_messages_many = &recvd_messages[0];
        recv_message_type** recvd_messages_few = &recvd_messages[num_many];

        int recvd_num_many = 0;
        int recvd_num_few = 0;

        while (recvd_num_many < num_many || recvd_num_few < num_few) {

          IdxT num_recvd = 0;
          if (wait_recv_method == CommInfo::method::waitsome) {
            num_recvd = recv_message_type::wait_recv_some(con_comm, num_recvs, &requests[0], &indices[0], &recv_statuses[0]);
          } else {
            num_recvd = recv_message_type::test_recv_some(con_comm, num_recvs, &requests[0], &indices[0], &recv_statuses[0]);
          }

          int next_recvd_num_many = recvd_num_many;
          int next_recvd_num_few = recvd_num_few;

          // put received messages is lists
          for (IdxT i = 0; i < num_recvd; ++i) {
            if (indices[i] < num_many) {
              recvd_messages_many[next_recvd_num_many++] = messages[indices[i]];
            } else if (indices[i] < num_recvs) {
              recvd_messages_few[next_recvd_num_few++] = messages[indices[i]];
            } else {
              assert(0 <= indices[i] && indices[i] < num_recvs);
            }
          }

          if (recvd_num_many < next_recvd_num_many) {
            m_recvs.message_group_many.unpack(con_many, con_comm, &recvd_messages_many[recvd_num_many], next_recvd_num_many-recvd_num_many);
            m_recvs.message_group_many.deallocate(con_many, con_comm, &recvd_messages_many[recvd_num_many], next_recvd_num_many-recvd_num_many);
            recvd_num_many = next_recvd_num_many;
          }

          if (recvd_num_few < next_recvd_num_few) {
            m_recvs.message_group_few.unpack(con_few, con_comm, &recvd_messages_few[recvd_num_few], next_recvd_num_few-recvd_num_few);
            m_recvs.message_group_few.deallocate(con_few, con_comm, &recvd_messages_few[recvd_num_few], next_recvd_num_few-recvd_num_few);
            recvd_num_few = next_recvd_num_few;
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        if (wait_recv_method == CommInfo::method::waitall) {
          recv_message_type::wait_recv_all(con_comm, num_recvs, &requests[0], &recv_statuses[0]);
        } else {
          while (!recv_message_type::test_recv_all(con_comm, num_recvs, &requests[0], &recv_statuses[0]));
        }

        m_recvs.message_group_many.unpack(con_many, con_comm, &messages_many[0], num_many);
        m_recvs.message_group_few.unpack(con_few, con_comm, &messages_few[0], num_few);

        m_recvs.message_group_many.deallocate(con_many, con_comm, &messages_many[0], num_many);
        m_recvs.message_group_few.deallocate(con_few, con_comm, &messages_few[0], num_few);
      } break;
      default:
      {
        assert(0);
      } break;
    }

    m_recvs.messages.clear();
    m_recvs.requests.clear();

    if (num_few > 0) {
      con_few.synchronize();
    }
    if (num_many > 0) {
      con_many.synchronize();
    }
  }



  void waitSend(ExecContext<policy_many>& con_many, ExecContext<policy_few>& con_few)
  {
    COMB::ignore_unused(con_many, con_few);
    //FGPRINTF(FileGroup::proc, "posting sends\n");

    IdxT num_many = m_sends.message_group_many.messages.size();
    IdxT num_few = m_sends.message_group_few.messages.size();

    IdxT num_sends = num_many + num_few;

    m_sends.messages.resize(num_sends, nullptr);
    m_sends.requests.resize(num_sends, con_comm.send_request_null());

    send_message_type** messages = &m_sends.messages[0];

    send_message_type** messages_many = &m_sends.messages[0];
    send_message_type** messages_few  = &m_sends.messages[num_many];

    send_request_type* requests = &m_sends.requests[0];

    using send_status_type = typename policy_comm::send_status_type;
    std::vector<send_status_type> send_statuses(num_sends, con_comm.send_status_null());

    switch (wait_send_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_done = 0;
        while (num_done < num_sends) {

          IdxT idx = num_sends;
          if (wait_send_method == CommInfo::method::waitany) {
            idx = send_message_type::wait_send_any(con_comm, num_sends, &requests[0], &send_statuses[0]);
          } else {
            while(idx < 0 || idx >= num_sends) {
              idx = send_message_type::test_send_any(con_comm, num_sends, &requests[0], &send_statuses[0]);
            }
          }

          if (idx < num_many) {
            m_sends.message_group_many.deallocate(con_many, con_comm, &messages[idx], 1);
          } else if (idx < num_sends) {
            m_sends.message_group_few.deallocate(con_few, con_comm, &messages[idx], 1);
          } else {
            assert(0 <= idx || idx < num_sends);
          }

          num_done += 1;
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        std::vector<int> indices(num_sends, -1);
        std::vector<send_message_type*> sent_messages(num_sends, nullptr);

        send_message_type** sent_messages_many = &sent_messages[0];
        send_message_type** sent_messages_few = &sent_messages[num_many];

        int sent_num_many = 0;
        int sent_num_few = 0;

        while (sent_num_many < num_many || sent_num_few < num_few) {

          IdxT num_sent = 0;
          if (wait_send_method == CommInfo::method::waitsome) {
            num_sent = send_message_type::wait_send_some(con_comm, num_sends, &requests[0], &indices[0], &send_statuses[0]);
          } else {
            num_sent = send_message_type::test_send_some(con_comm, num_sends, &requests[0], &indices[0], &send_statuses[0]);
          }

          int next_sent_num_many = sent_num_many;
          int next_sent_num_few = sent_num_few;

          // put received messages is lists
          for (IdxT i = 0; i < num_sent; ++i) {
            if (indices[i] < num_many) {
              sent_messages_many[next_sent_num_many++] = messages[indices[i]];
            } else if (indices[i] < num_sends) {
              sent_messages_few[next_sent_num_few++] = messages[indices[i]];
            } else {
              assert(0 <= indices[i] && indices[i] < num_sends);
            }
          }

          if (sent_num_many < next_sent_num_many) {
            m_sends.message_group_many.deallocate(con_many, con_comm, &sent_messages_many[sent_num_many], next_sent_num_many-sent_num_many);
            sent_num_many = next_sent_num_many;
          }
          if (sent_num_few < next_sent_num_few) {
            m_sends.message_group_few.deallocate(con_few, con_comm, &sent_messages_few[sent_num_few], next_sent_num_few-sent_num_few);
            sent_num_few = next_sent_num_few;
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        if (wait_send_method == CommInfo::method::waitall) {
          send_message_type::wait_send_all(con_comm, num_sends, &requests[0], &send_statuses[0]);
        } else {
          while(!send_message_type::test_send_all(con_comm, num_sends, &requests[0], &send_statuses[0]));
        }

        m_sends.message_group_many.deallocate(con_many, con_comm, &messages_many[0], num_many);
        m_sends.message_group_few.deallocate(con_few, con_comm, &messages_few[0], num_few);
      } break;
      default:
      {
        assert(0);
      } break;
    }

    m_sends.messages.clear();
    m_sends.requests.clear();
  }
};

namespace COMB {

struct CommunicatorsAvailable
{
  bool mock = false;
  bool mpi = false;
  bool gdsync = false;
  bool gpump = false;
  bool mp = false;
  bool umr = false;
};

} // namespace COMB

#endif // _COMM_HPP

