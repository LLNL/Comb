
#ifndef _COMM_CUH
#define _COMM_CUH

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <list>
#include <vector>
#include <map>
#include <utility>

#include <mpi.h>

#include "memory.cuh"
#include "for_all.cuh"
#include "utils.cuh"

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
      if ((0 <= arg_coords[dim] && arg_coords[dim] < divisions[dim]) || periodic[dim]) {
        input_coords[dim] = arg_coords[dim] % divisions[dim];
        if (input_coords[dim] < 0) input_coords[dim] += divisions[dim];
      }
    }
    if (input_coords[0] != -1 && input_coords[1] != -1 && input_coords[2] != -1) {
      output_rank = detail::MPI::Cart_rank(comm, input_coords);
    }
    return output_rank;
  }
  
  void disconnect()
  {
    detail::MPI::Comm_disconnect(&comm);
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
    , post_send_method(method::waitall)
    , post_recv_method(method::waitall)
    , wait_send_method(method::waitall)
    , wait_recv_method(method::waitall)
    , cutoff(200)
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
  
  template < typename ... Ts >
  void print_any(const char* fmt, Ts&&... args)
  {
    FPRINTF(stdout, fmt, std::forward<Ts>(args)...);
  }
  
  template < typename ... Ts >
  void print_master(const char* fmt, Ts&&... args)
  {
    if (rank == 0) {
      print_any(fmt, std::forward<Ts>(args)...);
    }
  }
  
  template < typename ... Ts >
  void warn_any(const char* fmt, Ts&&... args)
  {
    FPRINTF(stderr, fmt, std::forward<Ts>(args)...);
  }
  
  template < typename ... Ts >
  void warn_master(const char* fmt, Ts&&... args)
  {
    if (rank == 0) {
      warn_any(fmt, std::forward<Ts>(args)...);
    }
  }
  
  template < typename ... Ts >
  void abort_any(const char* fmt, Ts&&... args)
  {
    warn_any(fmt, std::forward<Ts>(args)...);
    abort();
  }
  
  template < typename ... Ts >
  void abort_master(const char* fmt, Ts&&... args)
  {
    warn_master(fmt, std::forward<Ts>(args)...);
    abort();
  }
  
  void abort()
  {
    detail::MPI::Abort(MPI_COMM_WORLD, 1);
  }
};

template < typename policy_many, typename policy_few >
struct Message
{
  using pol_many = policy_many;
  using pol_few = policy_few;
  Allocator& buf_aloc_many;
  Allocator& buf_aloc_few;
  int m_dest_rank;
  int msg_tag;
  IdxT cutoff;
  DataT* m_buf;
  IdxT m_size;
  bool have_many;
  
  
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

  Message(int dest_rank_, int tag, Allocator& buf_aloc_many_, Allocator& buf_aloc_few_, IdxT cutoff_)
    : buf_aloc_many(buf_aloc_many_)
    , buf_aloc_few(buf_aloc_few_)
    , m_dest_rank(dest_rank_)
    , msg_tag(tag)
    , cutoff(cutoff_)
    , m_buf(nullptr)
    , m_size(0)
    , have_many(false)
  {

  }
  
  int dest_rank()
  {
    return m_dest_rank;
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
  
  void add(DataT* data, LidxT* indices, Allocator& aloc, IdxT size)
  {
    items.emplace_front(data, indices, aloc, size);
    m_size += size;
    
    have_many = ( (m_size + items.size() - 1) / items.size() ) > cutoff ;
  }
  
  void pack()
  {
    DataT* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT const* src = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      if (have_many) {
        for_all(policy_many{}, 0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      } else {
        for_all(policy_few{},  0, len, make_copy_idxr_idxr(src, detail::indexer_list_idx{indices}, buf, detail::indexer_idx{}));
      }
      buf += len;
    }
  }
  
  void unpack()
  {
    DataT const* buf = m_buf;
    assert(buf != nullptr);
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      DataT* dst = i->data;
      LidxT const* indices = i->indices;
      IdxT len = i->size;
      if (have_many) {
        for_all(policy_many{}, 0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      } else {
        for_all(policy_few{},  0, len, make_copy_idxr_idxr(buf, detail::indexer_idx{}, dst, detail::indexer_list_idx{indices}));
      }
      buf += len;
    }
  }

  void allocate()
  {
    if (m_buf == nullptr) {
      if (have_many) {
        m_buf = (DataT*)buf_aloc_many.allocate(size()*sizeof(DataT));
      } else {
        m_buf = (DataT*)buf_aloc_few.allocate(size()*sizeof(DataT));
      }
    }
  }

  void deallocate()
  {
    if (m_buf != nullptr) {
      if (have_many) {
        buf_aloc_many.deallocate(m_buf);
      } else {
        buf_aloc_few.deallocate(m_buf);
      }
      m_buf = nullptr;
    }
  }
 
  ~Message()
  {
    deallocate();
    auto end = std::end(items);
    for (auto i = std::begin(items); i != end; ++i) {
      i->aloc.deallocate(i->indices); i->indices = nullptr;
    }
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
  
  using message_type = Message<policy_many, policy_few>;
  std::vector<message_type> m_sends;
  std::vector<message_type> m_recvs;
  
  std::vector<MPI_Request> m_send_requests;
  std::vector<MPI_Request> m_recv_requests;
  
  Comm(CommInfo const& comminfo_, Allocator& many_aloc_, Allocator& few_aloc_)
    : many_aloc(many_aloc_)
    , few_aloc(few_aloc_)
    , comminfo(comminfo_)
  {
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
        
          m_recvs[i].allocate();
          
          if (!comminfo.mock_communication) {
            detail::MPI::Irecv( m_recvs[i].buffer(), m_recvs[i].nbytes(),
                                m_recvs[i].dest_rank(), m_recvs[i].dest_rank(),
                                comminfo.cart.comm, &m_recv_requests[i] );
          }
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {
        
          m_recvs[i].allocate();
          
          if (!comminfo.mock_communication) {
            detail::MPI::Irecv( m_recvs[i].buffer(), m_recvs[i].nbytes(),
                               m_recvs[i].dest_rank(), m_recvs[i].dest_rank(),
                               comminfo.cart.comm, &m_recv_requests[i] );
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_recvs = m_recvs.size();
        for (IdxT i = 0; i < num_recvs; ++i) {
        
          m_recvs[i].allocate();
        }
        for (IdxT i = 0; i < num_recvs; ++i) {
        
          if (!comminfo.mock_communication) {
            detail::MPI::Irecv( m_recvs[i].buffer(), m_recvs[i].nbytes(),
                               m_recvs[i].dest_rank(), m_recvs[i].dest_rank(),
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
      case CommInfo::method::testany:
      {
        IdxT num_sends = m_sends.size();
        for (IdxT i = 0; i < num_sends; ++i) {
        
          m_sends[i].allocate();
          m_sends[i].pack();
          
          if (m_sends[i].have_many) {
            synchronize(policy_many{});
          } else {
            synchronize(policy_few{});
          }
          
          if (!comminfo.mock_communication) {
            detail::MPI::Isend( m_sends[i].buffer(), m_sends[i].nbytes(),
                               m_sends[i].dest_rank(), m_sends[i].dest_rank(),
                               comminfo.cart.comm, &m_send_requests[i] );
          }
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_sends = m_sends.size();
        
        for (int val = 0; val < 2; ++val) {
        
          bool expected = val;
          bool found_expected = false;
        
          for (IdxT i = 0; i < num_sends; ++i) {
        
            if (m_sends[i].have_many == expected) {
              m_sends[i].allocate();
              m_sends[i].pack();
              found_expected = true;
            }
          }
        
          if (found_expected) {
        
            if (expected) {
              synchronize(policy_many{});
            } else {
              synchronize(policy_few{});
            }
            
            for (IdxT i = 0; i < num_sends; ++i) {
        
              if (m_sends[i].have_many == expected) {
          
                if (!comminfo.mock_communication) {
                  detail::MPI::Isend( m_sends[i].buffer(), m_sends[i].nbytes(),
                                     m_sends[i].dest_rank(), m_sends[i].dest_rank(),
                                     comminfo.cart.comm, &m_send_requests[i] );
                }
              }
            }
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_sends = m_sends.size();
        bool have_many = false;
        bool have_few = false;
        
        for (IdxT i = 0; i < num_sends; ++i) {
        
          m_sends[i].allocate();
          m_sends[i].pack();
          
          if (m_sends[i].have_many) {
            have_many = true;
          } else {
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
                               m_sends[i].dest_rank(), m_sends[i].dest_rank(),
                               comminfo.cart.comm, &m_send_requests[i] );
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
        IdxT num_done = 0;
          
        MPI_Status status;
          
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
            
          m_recvs[idx].unpack();
          m_recvs[idx].deallocate();
          
          if (m_recvs[idx].have_many) {
            have_many = true;
          } else {
            have_few = true;
          }
            
          num_done += 1;
            
        }
      } break;
      case CommInfo::method::waitsome:
      case CommInfo::method::testsome:
      {
        IdxT num_recvs = m_recvs.size();
        IdxT num_done = 0;
          
        std::vector<MPI_Status> recv_statuses(m_recv_requests.size());
        std::vector<int> indices(m_recv_requests.size(), -1);
          
        while (num_done < num_recvs) {
          
          IdxT num = num_recvs;
          if (!comminfo.mock_communication) {
            if (comminfo.wait_recv_method == CommInfo::method::waitsome) {
              num = detail::MPI::Waitsome( num_recvs, &m_recv_requests[0], &indices[0], &recv_statuses[0]);
            } else {
              num = detail::MPI::Testsome( num_recvs, &m_recv_requests[0], &indices[0], &recv_statuses[0]);
            }
          } else {
            for (IdxT i = 0; i < num; ++i) {
              indices[i] = num_done + i;
            }
          }
          
          for (IdxT i = 0; i < num; ++i) {
              
            m_recvs[indices[i]].unpack();
            m_recvs[indices[i]].deallocate();
          
            if (m_recvs[indices[i]].have_many) {
              have_many = true;
            } else {
              have_few = true;
            }
            
            num_done += 1;
            
          }
        }
      } break;
      case CommInfo::method::waitall:
      case CommInfo::method::testall:
      {
        IdxT num_recvs = m_recvs.size();
        IdxT num_done = 0;
          
        std::vector<MPI_Status> recv_statuses(m_recv_requests.size());
        
        if (!comminfo.mock_communication) {
          if (comminfo.wait_recv_method == CommInfo::method::waitall) {
            detail::MPI::Waitall( num_recvs, &m_recv_requests[0], &recv_statuses[0]);
          } else {
            while (!detail::MPI::Testall( num_recvs, &m_recv_requests[0], &recv_statuses[0]));
          }
        }
        
        while (num_done < num_recvs) {
              
          m_recvs[num_done].unpack();
          m_recvs[num_done].deallocate();
          
          if (m_recvs[num_done].have_many) {
            have_many = true;
          } else {
            have_few = true;
          }
           
          num_done += 1;
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
          
          m_sends[idx].deallocate();
           
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
              
            m_sends[indices[i]].deallocate();
            
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
          
          m_sends[num_done].deallocate();
          
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
  }
};

#endif // _COMM_CUH

