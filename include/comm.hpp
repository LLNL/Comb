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

#include "MessageBase.hpp"


enum struct FileGroup
{ out_any    // stdout, any proc
, out_master // stdout, master only
, err_any    // stderr, any proc
, err_master // stderr, master only
, proc       // per process file, any proc
, summary    // per run summary file, master only
, all        // out_master, proc, summary
};

extern FILE* comb_out_file;
extern FILE* comb_err_file;
extern FILE* comb_proc_file;
extern FILE* comb_summary_file;

extern void comb_setup_files(int rank);
extern void comb_teardown_files();

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

  void print(FileGroup fg, const char* fmt, ...);

  void abort()
  {
    detail::MPI::Abort(MPI_COMM_WORLD, 1);
  }
};

template < typename policy_many_, typename policy_few_, typename policy_comm_ >
struct Comm
{
  using policy_many = policy_many_;
  using policy_few  = policy_few_;
  using policy_comm  = policy_comm_;

  static constexpr bool pol_many_is_mpi_type = std::is_same<policy_many, mpi_type_pol>::value;
  static constexpr bool pol_few_is_mpi_type  = std::is_same<policy_few,  mpi_type_pol>::value;
  static constexpr bool use_mpi_type = pol_many_is_mpi_type && pol_few_is_mpi_type;

  // check policies are consistent
  static_assert(pol_many_is_mpi_type == pol_few_is_mpi_type,
      "pol_many and pol_few must both be mpi_type_pol if either is mpi_type_pol");

  COMB::Allocator& mesh_aloc;
  COMB::Allocator& many_aloc;
  COMB::Allocator& few_aloc;

  CommInfo& comminfo;

  CommInfo::method post_recv_method;
  CommInfo::method post_send_method;
  CommInfo::method wait_recv_method;
  CommInfo::method wait_send_method;

  typename policy_comm::communicator_type communicator;

  using message_type = Message<policy_comm>;
  std::vector<message_type> m_sends;
  std::vector<typename policy_comm::send_request_type> m_send_requests;

  std::vector<typename policy_many::event_type> m_send_events_many;
  std::vector<typename policy_few::event_type>  m_send_events_few;

  std::vector<ExecContext<policy_many>> m_send_contexts_many;
  std::vector<ExecContext<policy_few>>  m_send_contexts_few;

  std::vector<message_type> m_recvs;
  std::vector<typename policy_comm::recv_request_type> m_recv_requests;

  std::vector<ExecContext<policy_many>> m_recv_contexts_many;
  std::vector<ExecContext<policy_few>>  m_recv_contexts_few;

  Comm(CommInfo& comminfo_, COMB::Allocator& mesh_aloc_, COMB::Allocator& many_aloc_, COMB::Allocator& few_aloc_)
    : mesh_aloc(mesh_aloc_)
    , many_aloc(many_aloc_)
    , few_aloc(few_aloc_)
    , comminfo(comminfo_)
    , post_recv_method(comminfo_.post_recv_method)
    , post_send_method(comminfo_.post_send_method)
    , wait_recv_method(comminfo_.wait_recv_method)
    , wait_send_method(comminfo_.wait_send_method)
    , communicator(policy_comm::communicator_create(comminfo_.cart.comm))
  { }

  void finish_populating(ExecContext<policy_many> const& con_many, ExecContext<policy_few> const& con_few)
  {
    size_t num_sends = m_sends.size();
    for(size_t i = 0; i != num_sends; ++i) {
      m_send_contexts_many.push_back( con_many );
      m_send_contexts_few.push_back( con_few );
      m_send_events_many.push_back( createEvent(m_send_contexts_many.back()) );
      m_send_events_few.push_back( createEvent(m_send_contexts_few.back()) );
    }

    size_t num_recvs = m_recvs.size();
    for(size_t i = 0; i != num_recvs; ++i) {
      m_recv_contexts_many.push_back( con_many );
      m_recv_contexts_few.push_back( con_few );
    }

    std::vector<int> send_ranks;
    std::vector<int> recv_ranks;
    for(message_type& msg : m_sends) {
      send_ranks.push_back(msg.partner_rank());
    }
    for(message_type& msg : m_recvs) {
      recv_ranks.push_back(msg.partner_rank());
    }
    connect_ranks(policy_comm{}, communicator, send_ranks, recv_ranks);
  }

  void depopulate()
  {
    size_t num_events = m_send_events_many.size();
    for(size_t i = 0; i != num_events; ++i) {
      destroyEvent(m_send_contexts_many[i], m_send_events_many[i]);
    }
    num_events = m_send_events_few.size();
    for(size_t i = 0; i != num_events; ++i) {
      destroyEvent(m_send_contexts_few[i], m_send_events_few[i]);
    }

    std::vector<int> send_ranks;
    std::vector<int> recv_ranks;
    for(message_type& msg : m_sends) {
      send_ranks.push_back(msg.partner_rank());
    }
    for(message_type& msg : m_recvs) {
      recv_ranks.push_back(msg.partner_rank());
    }
    disconnect_ranks(policy_comm{}, communicator, send_ranks, recv_ranks);

    for(message_type& msg : m_sends) {
      msg.destroy();
    }
    for(message_type& msg : m_recvs) {
      msg.destroy();
    }
  }

  ~Comm()
  {
    policy_comm::communicator_destroy(communicator);
  }

  bool mock_communication() const
  {
    return policy_comm::mock;
  }

  void barrier()
  {
    comminfo.barrier();
  }

  void postRecv(ExecContext<policy_many> const& con_many, ExecContext<policy_few> const& con_few)
  {
    COMB::ignore_unused(con_many, con_few);
    //FPRINTF(stdout, "posting receives\n");

    m_recv_requests.resize(m_recvs.size(), policy_comm::recv_request_null());

    switch (post_recv_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].allocate(m_recv_contexts_many[i], communicator, many_aloc);
            m_recvs[i].Irecv(m_recv_contexts_many[i], communicator, &m_recv_requests[i]);
          } else {
            m_recvs[i].allocate(m_recv_contexts_few[i], communicator, few_aloc);
            m_recvs[i].Irecv(m_recv_contexts_few[i], communicator, &m_recv_requests[i]);
          }
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].allocate(m_recv_contexts_many[i], communicator, many_aloc);
            m_recvs[i].Irecv(m_recv_contexts_many[i], communicator, &m_recv_requests[i]);
          } else {
            m_recvs[i].allocate(m_recv_contexts_few[i], communicator, few_aloc);
            m_recvs[i].Irecv(m_recv_contexts_few[i], communicator, &m_recv_requests[i]);
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].allocate(m_recv_contexts_many[i], communicator, many_aloc);
          } else {
            m_recvs[i].allocate(m_recv_contexts_few[i], communicator, few_aloc);
          }
        }
        for (IdxT i = 0; i < num_recvs; ++i) {

          if (m_recvs[i].have_many()) {
            m_recvs[i].Irecv(m_recv_contexts_many[i], communicator, &m_recv_requests[i]);
          } else {
            m_recvs[i].Irecv(m_recv_contexts_few[i], communicator, &m_recv_requests[i]);
          }
        }
      } break;
      default:
      {
        assert(0);
      } break;
    }
  }



  void postSend(ExecContext<policy_many> const& con_many, ExecContext<policy_few> const& con_few)
  {
    //FPRINTF(stdout, "posting sends\n");

    m_send_requests.resize(m_sends.size(), policy_comm::send_request_null());

    switch (post_send_method) {
      case CommInfo::method::waitany:
      {
        IdxT num_sends = m_sends.size();
        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].allocate(m_send_contexts_many[i], communicator, many_aloc);
            m_sends[i].pack(m_send_contexts_many[i], communicator);
            synchronize(m_send_contexts_many[i]);
            m_sends[i].Isend(m_send_contexts_many[i], communicator, &m_send_requests[i]);
          } else {
            m_sends[i].allocate(m_send_contexts_few[i], communicator, few_aloc);
            m_sends[i].pack(m_send_contexts_few[i], communicator);
            synchronize(m_send_contexts_few[i]);
            m_sends[i].Isend(m_send_contexts_few[i], communicator, &m_send_requests[i]);
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
            m_sends[i].allocate(m_send_contexts_many[i], communicator, many_aloc);
            have_many = true;
          } else {
            m_sends[i].allocate(m_send_contexts_few[i], communicator, few_aloc);
            have_few = true;
          }
        }

        // pack and send
        if (have_many && have_few) {
          persistent_launch(con_few); persistent_launch(con_many);
        } else if (have_many) {
          persistent_launch(con_many);
        } else if (have_few) {
          persistent_launch(con_few);
        }

        bool post_pack_complete = false;
        IdxT pack_send = 0;
        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (post_many_send < num_sends || post_few_send < num_sends) {

          // pack and record events
          if (pack_send < num_sends) {

            if (m_sends[pack_send].have_many()) {
              m_sends[pack_send].pack(m_send_contexts_many[pack_send], communicator);
              recordEvent(m_send_contexts_many[pack_send], m_send_events_many[pack_send]);
            } else {
              m_sends[pack_send].pack(m_send_contexts_few[pack_send], communicator);
              recordEvent(m_send_contexts_few[pack_send], m_send_events_few[pack_send]);
            }

            ++pack_send;

          } else if (!post_pack_complete) {

            if (have_many && have_few) {
              batch_launch(con_few); batch_launch(con_many);
            } else if (have_many) {
              batch_launch(con_many);
            } else if (have_few) {
              batch_launch(con_few);
            }

            // stop persistent kernel
            if (have_many && have_few) {
              persistent_stop(con_few); persistent_stop(con_many);
            } else if (have_many) {
              persistent_stop(con_many);
            } else if (have_few) {
              persistent_stop(con_few);
            }

            post_pack_complete = true;
          }

          while (post_many_send < pack_send) {

            if (m_sends[post_many_send].have_many()) {

              if (queryEvent(m_send_contexts_many[post_many_send], m_send_events_many[post_many_send])) {

                m_sends[post_many_send].Isend(m_send_contexts_many[post_many_send], communicator, &m_send_requests[post_many_send]);

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

              if (queryEvent(m_send_contexts_few[post_few_send], m_send_events_few[post_few_send])) {

                m_sends[post_few_send].Isend(m_send_contexts_few[post_few_send], communicator, &m_send_requests[post_few_send]);

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
              m_sends[i].allocate(m_send_contexts_many[i], communicator, many_aloc);
              m_sends[i].pack(m_send_contexts_many[i], communicator);
              found_many = true;
            }
          }

          if (found_many) {

            synchronize(con_many);

            for (IdxT i = 0; i < num_sends; ++i) {

              if (m_sends[i].have_many()) {

                m_sends[i].Isend(m_send_contexts_many[i], communicator, &m_send_requests[i]);
              }
            }
          }
        }

        {
          // have_few case, send immediately
          for (IdxT i = 0; i < num_sends; ++i) {

            if (!m_sends[i].have_many()) {
              m_sends[i].allocate(m_send_contexts_few[i], communicator, few_aloc);
              m_sends[i].pack(m_send_contexts_few[i], communicator);

              synchronize(m_send_contexts_few[i]);

              m_sends[i].Isend(m_send_contexts_few[i], communicator, &m_send_requests[i]);
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
            m_sends[i].allocate(m_send_contexts_many[i], communicator, many_aloc);
            have_many = true;
          } else {
            m_sends[i].allocate(m_send_contexts_few[i], communicator, few_aloc);
            have_few = true;
          }
        }

        IdxT pack_many_send = 0;
        IdxT pack_few_send = 0;
        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        if (have_many) {

          persistent_launch(con_many);

          while (pack_many_send < num_sends) {

            if (m_sends[pack_many_send].have_many()) {

              m_sends[pack_many_send].pack(m_send_contexts_many[pack_many_send], communicator);

              recordEvent(m_send_contexts_many[pack_many_send], m_send_events_many[pack_many_send]);

            }

            ++pack_many_send;

            // post sends if possible
            while (post_many_send < pack_many_send) {

              if (m_sends[post_many_send].have_many()) {

                if (queryEvent(m_send_contexts_many[post_many_send], m_send_events_many[post_many_send])) {

                  m_sends[post_many_send].Isend(m_send_contexts_many[post_many_send], communicator, &m_send_requests[post_many_send]);

                  ++post_many_send;

                } else {

                  break;
                }
              } else {

                ++post_many_send;
              }
            }
          }

          batch_launch(con_many);
          persistent_stop(con_many);
        } else {
          pack_many_send = num_sends;
          post_many_send = num_sends;
        }

        if (have_few) {

          persistent_launch(con_few);

          while (pack_few_send < num_sends) {

            if (!m_sends[pack_few_send].have_many()) {

              m_sends[pack_few_send].pack(m_send_contexts_few[pack_few_send], communicator);

              recordEvent(m_send_contexts_few[pack_few_send], m_send_events_few[pack_few_send]);

            }

            ++pack_few_send;

            // post more sends if possible
            while (post_many_send < pack_many_send) {

              if (m_sends[post_many_send].have_many()) {

                if (queryEvent(m_send_contexts_many[post_many_send], m_send_events_many[post_many_send])) {

                  m_sends[post_many_send].Isend(m_send_contexts_many[post_many_send], communicator, &m_send_requests[post_many_send]);

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

                if (queryEvent(m_send_contexts_few[post_few_send], m_send_events_few[post_few_send])) {

                  m_sends[post_few_send].Isend(m_send_contexts_few[post_few_send], communicator, &m_send_requests[post_few_send]);

                  ++post_few_send;

                } else {

                  break;
                }
              } else {

                ++post_few_send;
              }
            }
          }

          batch_launch(con_few);
          persistent_stop(con_few);
        } else {
          pack_few_send = num_sends;
          post_few_send = num_sends;
        }

        // finish posting sends
        while (post_many_send < num_sends || post_few_send < num_sends) {

          while (post_many_send < pack_many_send) {

            if (m_sends[post_many_send].have_many()) {

              if (queryEvent(m_send_contexts_many[post_many_send], m_send_events_many[post_many_send])) {

                m_sends[post_many_send].Isend(m_send_contexts_many[post_many_send], communicator, &m_send_requests[post_many_send]);

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

              if (queryEvent(m_send_contexts_few[post_few_send], m_send_events_few[post_few_send])) {

                m_sends[post_few_send].Isend(m_send_contexts_few[post_few_send], communicator, &m_send_requests[post_few_send]);

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
            m_sends[i].allocate(m_send_contexts_many[i], communicator, many_aloc);
            m_sends[i].pack(m_send_contexts_many[i], communicator);
            have_many = true;
          } else {
            m_sends[i].allocate(m_send_contexts_few[i], communicator, few_aloc);
            m_sends[i].pack(m_send_contexts_few[i], communicator);
            have_few = true;
          }
        }

        if (have_many && have_few) {
          synchronize(con_few); synchronize(con_many);
        } else if (have_many) {
          synchronize(con_many);
        } else if (have_few) {
          synchronize(con_few);
        }

        for (IdxT i = 0; i < num_sends; ++i) {

          if (m_sends[i].have_many()) {
            m_sends[i].Isend(m_send_contexts_many[i], communicator, &m_send_requests[i]);
          } else {
            m_sends[i].Isend(m_send_contexts_few[i], communicator, &m_send_requests[i]);
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
            m_sends[i].allocate(m_send_contexts_many[i], communicator, many_aloc);
            have_many = true;
          } else {
            m_sends[i].allocate(m_send_contexts_few[i], communicator, few_aloc);
            have_few = true;
          }
        }

        // pack and send
        if (have_many && have_few) {
          persistent_launch(con_few); persistent_launch(con_many);
        } else if (have_many) {
          persistent_launch(con_many);
        } else if (have_few) {
          persistent_launch(con_few);
        }

        IdxT pack_send = 0;
        IdxT post_many_send = 0;
        IdxT post_few_send = 0;

        while (pack_send < num_sends) {

          // pack and record events
          if (m_sends[pack_send].have_many()) {
            m_sends[pack_send].pack(m_send_contexts_many[pack_send], communicator);
            recordEvent(m_send_contexts_many[pack_send], m_send_events_many[pack_send]);
          } else {
            m_sends[pack_send].pack(m_send_contexts_few[pack_send], communicator);
            recordEvent(m_send_contexts_few[pack_send], m_send_events_few[pack_send]);
          }

          ++pack_send;
        }

        if (have_many && have_few) {
          batch_launch(con_few); batch_launch(con_many);
        } else if (have_many) {
          batch_launch(con_many);
        } else if (have_few) {
          batch_launch(con_few);
        }

        // stop persistent kernel
        if (have_many && have_few) {
          persistent_stop(con_few); persistent_stop(con_many);
        } else if (have_many) {
          persistent_stop(con_many);
        } else if (have_few) {
          persistent_stop(con_few);
        }

        // post all sends
        while (post_many_send < num_sends || post_few_send < num_sends) {

          while (post_many_send < num_sends) {

            if (m_sends[post_many_send].have_many()) {

              if (queryEvent(m_send_contexts_many[post_many_send], m_send_events_many[post_many_send])) {

                m_sends[post_many_send].Isend(m_send_contexts_many[post_many_send], communicator, &m_send_requests[post_many_send]);

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

              if (queryEvent(m_send_contexts_few[post_few_send], m_send_events_few[post_few_send])) {

                m_sends[post_few_send].Isend(m_send_contexts_few[post_few_send], communicator, &m_send_requests[post_few_send]);

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



  void waitRecv(ExecContext<policy_many> const& con_many, ExecContext<policy_few> const& con_few)
  {
    //FPRINTF(stdout, "waiting receives\n");

    bool have_many = false;
    bool have_few = false;

    switch (wait_recv_method) {
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
          persistent_launch(con_few); persistent_launch(con_many);
        } else if (have_many) {
          persistent_launch(con_many);
        } else if (have_few) {
          persistent_launch(con_few);
        }

        typename policy_comm::recv_status_type status = policy_comm::recv_status_null();

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          IdxT idx = num_done;
          if (wait_recv_method == CommInfo::method::waitany) {
            idx = message_type::wait_recv_any(num_recvs, &m_recv_requests[0], &status);
          } else {
            idx = -1;
            while(idx < 0 || idx >= num_recvs) {
              idx = message_type::test_recv_any(num_recvs, &m_recv_requests[0], &status);
            }
          }

          if (m_recvs[idx].have_many()) {
            m_recvs[idx].unpack(m_recv_contexts_many[idx], communicator);
            m_recvs[idx].deallocate(m_recv_contexts_many[idx], communicator, many_aloc);
            batch_launch(m_recv_contexts_many[idx]);
          } else {
            m_recvs[idx].unpack(m_recv_contexts_few[idx], communicator);
            m_recvs[idx].deallocate(m_recv_contexts_few[idx], communicator, few_aloc);
            batch_launch(m_recv_contexts_few[idx]);
          }

          num_done += 1;

        }

        // if (have_many && have_few) {
        //   batch_launch(m_recv_contexts_few[i], m_recv_contexts_many[i]);
        // } else if (have_many) {
        //   batch_launch(m_recv_contexts_many[i]);
        // } else if (have_few) {
        //   batch_launch(m_recv_contexts_few[i]);
        // }

        if (have_many && have_few) {
          persistent_stop(con_few); persistent_stop(con_many);
        } else if (have_many) {
          persistent_stop(con_many);
        } else if (have_few) {
          persistent_stop(con_few);
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
          persistent_launch(con_few); persistent_launch(con_many);
        } else if (have_many) {
          persistent_launch(con_many);
        } else if (have_few) {
          persistent_launch(con_few);
        }

        std::vector<typename policy_comm::recv_status_type> recv_statuses(m_recv_requests.size(), policy_comm::recv_status_null());
        std::vector<int> indices(m_recv_requests.size(), -1);

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          IdxT num_recvd = num_recvs;
          if (wait_recv_method == CommInfo::method::waitsome) {
            num_recvd = message_type::wait_recv_some(num_recvs, &m_recv_requests[0], &indices[0], &recv_statuses[0]);
          } else {
            while( 0 == (num_recvd = message_type::test_recv_some(num_recvs, &m_recv_requests[0], &indices[0], &recv_statuses[0])) );
          }

          bool inner_have_many = false;
          bool inner_have_few = false;

          for (IdxT i = 0; i < num_recvd; ++i) {

            if (m_recvs[indices[i]].have_many()) {

              m_recvs[indices[i]].unpack(m_recv_contexts_many[indices[i]], communicator);
              m_recvs[indices[i]].deallocate(m_recv_contexts_many[indices[i]], communicator, many_aloc);

              inner_have_many = true;

              num_done += 1;
            }
          }

          if (inner_have_many) {
            batch_launch(con_many);
          }

          for (IdxT i = 0; i < num_recvd; ++i) {

            if (!m_recvs[indices[i]].have_many()) {

              m_recvs[indices[i]].unpack(m_recv_contexts_few[indices[i]], communicator);
              m_recvs[indices[i]].deallocate(m_recv_contexts_few[indices[i]], communicator, few_aloc);

              inner_have_few = true;

              num_done += 1;
            }
          }

          if (inner_have_few) {
            batch_launch(con_few);
          }
        }

        if (have_many && have_few) {
          persistent_stop(con_few); persistent_stop(con_many);
        } else if (have_many) {
          persistent_stop(con_many);
        } else if (have_few) {
          persistent_stop(con_few);
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
          persistent_launch(con_few); persistent_launch(con_many);
        } else if (have_many) {
          persistent_launch(con_many);
        } else if (have_few) {
          persistent_launch(con_few);
        }

        std::vector<typename policy_comm::recv_status_type> recv_statuses(m_recv_requests.size(), policy_comm::recv_status_null());

        if (wait_recv_method == CommInfo::method::waitall) {
          message_type::wait_recv_all(num_recvs, &m_recv_requests[0], &recv_statuses[0]);
        } else {
          while (!message_type::test_recv_all(num_recvs, &m_recv_requests[0], &recv_statuses[0]));
        }

        IdxT num_done = 0;
        while (num_done < num_recvs) {

          if (m_recvs[num_done].have_many()) {
            m_recvs[num_done].unpack(m_recv_contexts_many[num_done], communicator);
            m_recvs[num_done].deallocate(m_recv_contexts_many[num_done], communicator, many_aloc);
            have_many = true;
          } else {
            m_recvs[num_done].unpack(m_recv_contexts_few[num_done], communicator);
            m_recvs[num_done].deallocate(m_recv_contexts_few[num_done], communicator, few_aloc);
            have_few = true;
          }

          num_done += 1;
        }

        if (have_many && have_few) {
          batch_launch(con_few); batch_launch(con_many);
        } else if (have_many) {
          batch_launch(con_many);
        } else if (have_few) {
          batch_launch(con_few);
        }

        if (have_many && have_few) {
          persistent_stop(con_few); persistent_stop(con_many);
        } else if (have_many) {
          persistent_stop(con_many);
        } else if (have_few) {
          persistent_stop(con_few);
        }
      } break;
      default:
      {
        assert(0);
      } break;
    }

    m_recv_requests.clear();

    if (have_many && have_few) {
      synchronize(con_few); synchronize(con_many);
    } else if (have_many) {
      synchronize(con_many);
    } else if (have_few) {
      synchronize(con_few);
    }
  }



  void waitSend(ExecContext<policy_many> const& con_many, ExecContext<policy_few> const& con_few)
  {
    COMB::ignore_unused(con_many, con_few);
    //FPRINTF(stdout, "posting sends\n");

    switch (wait_send_method) {
      case CommInfo::method::waitany:
      case CommInfo::method::testany:
      {
        IdxT num_sends = m_sends.size();
        IdxT num_done = 0;

        typename policy_comm::send_status_type status = policy_comm::send_status_null();

        while (num_done < num_sends) {

          IdxT idx = num_done;
          if (wait_send_method == CommInfo::method::waitany) {
            idx = message_type::wait_send_any(num_sends, &m_send_requests[0], &status);
          } else {
            idx = -1;
            while(idx < 0 || idx >= num_sends) {
              idx = message_type::test_send_any(num_sends, &m_send_requests[0], &status);
            }
          }

          if (m_sends[idx].have_many()) {
            m_sends[idx].deallocate(m_send_contexts_many[idx], communicator, many_aloc);
          } else {
            m_sends[idx].deallocate(m_send_contexts_few[idx], communicator, few_aloc);
          }

          num_done += 1;

        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_sends = m_sends.size();
        IdxT num_done = 0;

        std::vector<typename policy_comm::send_status_type> send_statuses(m_send_requests.size(), policy_comm::send_status_null());
        std::vector<int> indices(m_send_requests.size(), -1);

        while (num_done < num_sends) {

          IdxT num_sent = num_sends;
          if (wait_send_method == CommInfo::method::waitsome) {
            num_sent = message_type::wait_send_some(num_sends, &m_send_requests[0], &indices[0], &send_statuses[0]);
          } else {
            num_sent = message_type::test_send_some(num_sends, &m_send_requests[0], &indices[0], &send_statuses[0]);
          }

          for (IdxT i = 0; i < num_sent; ++i) {

            IdxT idx = indices[i];
            if (m_sends[idx].have_many()) {
              m_sends[idx].deallocate(m_send_contexts_many[idx], communicator, many_aloc);
            } else {
              m_sends[idx].deallocate(m_send_contexts_few[idx], communicator, few_aloc);
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

        std::vector<typename policy_comm::send_status_type> send_statuses(m_send_requests.size(), policy_comm::send_status_null());

        if (wait_send_method == CommInfo::method::waitall) {
          message_type::wait_send_all(num_sends, &m_send_requests[0], &send_statuses[0]);
        } else {
          while(!message_type::test_send_all(num_sends, &m_send_requests[0], &send_statuses[0]));
        }

        while (num_done < num_sends) {

          IdxT idx = num_done;
          if (m_sends[idx].have_many()) {
            m_sends[idx].deallocate(m_send_contexts_many[idx], communicator, many_aloc);
          } else {
            m_sends[idx].deallocate(m_send_contexts_few[idx], communicator, few_aloc);
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
};

namespace COMB {

struct CommunicatorsAvailable
{
  bool mock = false;
  bool mpi = false;
  bool gpump = false;
  bool mp = false;
  bool umr = false;
};

} // namespace COMB

#endif // _COMM_HPP

