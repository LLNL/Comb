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

#ifndef _COMM_FACTORY_HPP
#define _COMM_FACTORY_HPP

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#include "memory.hpp"
#include "for_all.hpp"
#include "utils.hpp"
#include "MeshInfo.hpp"
#include "MeshData.hpp"
#include "Box3d.hpp"
#include "comm.hpp"


template < typename loop_body >
void for_face_connections(IdxT& connection_idx, MeshInfo meshinfo, loop_body&& body)
{
  bool active[3] {false, false, false};
  for (IdxT dim = 2; dim >= 0; --dim) {
    active[dim] = true;
    IdxT off[1];
    for (off[0] = -1; off[0] <= 1; off[0] += 2) {
      IdxT off_idx = 0;
      int neighbor_coords[3] { meshinfo.global_coords[0], meshinfo.global_coords[1], meshinfo.global_coords[2] } ;
      if (active[0]) { neighbor_coords[0] += off[off_idx++]; }
      if (active[1]) { neighbor_coords[1] += off[off_idx++]; }
      if (active[2]) { neighbor_coords[2] += off[off_idx++]; }
      if (((0 <= neighbor_coords[0] && neighbor_coords[0] < meshinfo.global.divisions[0]) || meshinfo.global.periodic[0]) &&
          ((0 <= neighbor_coords[1] && neighbor_coords[1] < meshinfo.global.divisions[1]) || meshinfo.global.periodic[1]) &&
          ((0 <= neighbor_coords[2] && neighbor_coords[2] < meshinfo.global.divisions[2]) || meshinfo.global.periodic[2]) ) {

        body(connection_idx++, neighbor_coords);
      }
    }
    active[dim] = false;
  }
}

template < typename loop_body >
void for_edge_connections(IdxT& connection_idx, MeshInfo meshinfo, loop_body&& body)
{
  bool active[3] {true, true, true};
  for (IdxT dim = 0; dim < 3; ++dim) {
    active[dim] = false;
    IdxT off[2];
    for (off[1] = -1; off[1] <= 1; off[1] += 2) {
      for (off[0] = -1; off[0] <= 1; off[0] += 2) {
        IdxT off_idx = 0;
        int neighbor_coords[3] { meshinfo.global_coords[0], meshinfo.global_coords[1], meshinfo.global_coords[2] } ;
        if (active[0]) { neighbor_coords[0] += off[off_idx++]; }
        if (active[1]) { neighbor_coords[1] += off[off_idx++]; }
        if (active[2]) { neighbor_coords[2] += off[off_idx++]; }
        if (((0 <= neighbor_coords[0] && neighbor_coords[0] < meshinfo.global.divisions[0]) || meshinfo.global.periodic[0]) &&
            ((0 <= neighbor_coords[1] && neighbor_coords[1] < meshinfo.global.divisions[1]) || meshinfo.global.periodic[1]) &&
            ((0 <= neighbor_coords[2] && neighbor_coords[2] < meshinfo.global.divisions[2]) || meshinfo.global.periodic[2]) ) {

          body(connection_idx++, neighbor_coords);
        }
      }
    }
    active[dim] = true;
  }
}

template < typename loop_body >
void for_corner_connections(IdxT& connection_idx, MeshInfo meshinfo, loop_body&& body)
{
  bool active[3] {true, true, true};
  IdxT off[3];
  for (off[2] = -1; off[2] <= 1; off[2] += 2) {
    for (off[1] = -1; off[1] <= 1; off[1] += 2) {
      for (off[0] = -1; off[0] <= 1; off[0] += 2) {
        IdxT off_idx = 0;
        int neighbor_coords[3] { meshinfo.global_coords[0], meshinfo.global_coords[1], meshinfo.global_coords[2] } ;
        if (active[0]) { neighbor_coords[0] += off[off_idx++]; }
        if (active[1]) { neighbor_coords[1] += off[off_idx++]; }
        if (active[2]) { neighbor_coords[2] += off[off_idx++]; }
        if (((0 <= neighbor_coords[0] && neighbor_coords[0] < meshinfo.global.divisions[0]) || meshinfo.global.periodic[0]) &&
            ((0 <= neighbor_coords[1] && neighbor_coords[1] < meshinfo.global.divisions[1]) || meshinfo.global.periodic[1]) &&
            ((0 <= neighbor_coords[2] && neighbor_coords[2] < meshinfo.global.divisions[2]) || meshinfo.global.periodic[2]) ) {

          body(connection_idx++, neighbor_coords);
        }
      }
    }
  }
}

template < typename loop_body >
void for_connections(MeshInfo meshinfo, loop_body&& body)
{
  IdxT connection_idx = 0;
  for_face_connections(connection_idx, meshinfo, body);
  for_edge_connections(connection_idx, meshinfo, body);
  for_corner_connections(connection_idx, meshinfo, body);
}

struct CommFactory
{
  static inline bool& allow_per_message_pack_fusing()
  {
    static bool allow = true;
    return allow;
  }

  CommInfo const& comminfo;

  using msg_map_type  = std::map<Box3d, Box3d>; // maps recv boxes to send boxes
  using data_map_type = std::map<MeshInfo, std::list<MeshData const*>>; // maps meshinfo to lists of meshdata
  using rank_map_type = std::map<MeshInfo, int>; // maps meshinfo to the rank that owns it

  // map from recv boxes (in the recv meshinfo's indices)
  //   to send boxes (in the send meshinfo's indices)
  msg_map_type msg_map;

  data_map_type data_map;

  rank_map_type rank_map;

  CommFactory(CommInfo const& comminfo_)
    : comminfo(comminfo_)
  { }

  void add_mesh(MeshInfo const& meshinfo)
  {
    Box3d self_own            = Box3d::make_owned_box(meshinfo);
    Box3d self_potential_recv = Box3d::make_ghost_box(meshinfo);

    // self_own.print("self_own");
    // self_potential_recv.print("self_potential_recv");

    {
      int self_rank = comminfo.cart.get_rank(meshinfo.global_coords);

      // add this meshinfo to rank_map
      auto iter = rank_map.find(meshinfo);
      if (iter == rank_map.end()) {
        auto res = rank_map.insert(
            typename rank_map_type::value_type{meshinfo, self_rank});
        assert(res.second);
        iter = res.first;
      }
      assert(iter->first == meshinfo);
      assert(iter->second == self_rank);
    }

    // go though neighbors adding to msg_map[recv_box] = neighbor_send_box, msg_map[neighbor_recv_box] = send_box
    for_connections(meshinfo, [&](IdxT cnct, const int neighbor_coords[]) {
      COMB::ignore_unused(cnct);

      MeshInfo neighbor_info = MeshInfo::get_local(meshinfo.global, neighbor_coords);

      int neighbor_rank = comminfo.cart.get_rank(neighbor_info.global_coords);

      Box3d neighbor_own            = Box3d::make_owned_box(neighbor_info);
      Box3d neighbor_potential_recv = Box3d::make_ghost_box(neighbor_info);

      // neighbor_own.print("neighbor_own");
      // neighbor_potential_recv.print("neighbor_potential_recv");

      Box3d self_recv     = self_potential_recv.intersect(neighbor_own);
      Box3d neighbor_recv = neighbor_potential_recv.intersect(self_own);

      // self_recv.print("self_recv");
      // neighbor_recv.print("neighbor_recv");

      Box3d self_send     = self_own.intersect(neighbor_recv);
      Box3d neighbor_send = neighbor_own.intersect(self_recv);

      // self_send.print("self_send");
      // neighbor_send.print("neighbor_send");

      assert(self_recv.size() == neighbor_send.size());
      assert(self_send.size() == neighbor_recv.size());

      assert(self_recv.info == meshinfo);
      assert(self_send.info == meshinfo);

      assert(neighbor_recv.info == neighbor_info);
      assert(neighbor_send.info == neighbor_info);

      neighbor_recv.correct_periodicity();
      neighbor_send.correct_periodicity();

      // neighbor_recv.print("neighbor_recv_corrected");
      // neighbor_send.print("neighbor_send_corrected");

      {
        int neighbor_recv_rank = comminfo.cart.get_rank(neighbor_recv.info.global_coords);
        assert(neighbor_rank == neighbor_recv_rank);

        // add this meshinfo to rank_map
        auto iter = rank_map.find(neighbor_recv.info);
        if (iter == rank_map.end()) {
          auto res = rank_map.insert(
              typename rank_map_type::value_type{neighbor_recv.info, neighbor_rank});
          assert(res.second);
          iter = res.first;
        }
        assert(iter->first == neighbor_recv.info);
        assert(iter->second == neighbor_rank);
      }

      {
        int neighbor_send_rank = comminfo.cart.get_rank(neighbor_send.info.global_coords);
        assert(neighbor_rank == neighbor_send_rank);

        // add this meshinfo to rank_map
        auto iter = rank_map.find(neighbor_send.info);
        if (iter == rank_map.end()) {
          auto res = rank_map.insert(
              typename rank_map_type::value_type{neighbor_send.info, neighbor_rank});
          assert(res.second);
          iter = res.first;
        }
        assert(iter->first == neighbor_send.info);
        assert(iter->second == neighbor_rank);
      }


      auto my_recv_to_sends_iter = msg_map.find(self_recv);
      if (my_recv_to_sends_iter == msg_map.end()) {
        auto res = msg_map.insert(
            typename msg_map_type::value_type{self_recv, neighbor_send});
        assert(res.second);
        my_recv_to_sends_iter = res.first;
      }
      assert(my_recv_to_sends_iter->first  == self_recv);
      assert(my_recv_to_sends_iter->second == neighbor_send);

      auto neighbor_recv_to_sends_iter = msg_map.find(neighbor_recv);
      if (neighbor_recv_to_sends_iter == msg_map.end()) {
        auto res = msg_map.insert(
            typename msg_map_type::value_type{neighbor_recv, self_send});
        assert(res.second);
        neighbor_recv_to_sends_iter = res.first;
      }
      assert(neighbor_recv_to_sends_iter->first == neighbor_recv);
      assert(neighbor_recv_to_sends_iter->second == self_send);
    });
  }

  void add_var(MeshData const& meshdata)
  {
    add_mesh(meshdata.info);

    // add this meshinfo to data_map if not found
    auto iter = data_map.find(meshdata.info);
    if (iter == data_map.end()) {
      auto res = data_map.insert(
          typename data_map_type::value_type{meshdata.info, typename data_map_type::mapped_type{}});
      assert(res.second);
      iter = res.first;
    }
    // add meshdata to meshinfo
    iter->second.push_back(&meshdata);
  }


  struct message_info_data_type
  {
    MeshData const* data;
    std::list<Box3d> boxes;

    message_info_data_type(MeshData const* data_)
      : data(data_)
    { }

    void add_box(Box3d const& msg_box)
    {
      boxes.emplace_back(msg_box);
    }

    IdxT total_size() const
    {
      IdxT mysize = 0;
      for (Box3d const& box : boxes) {
        mysize += box.size();
      }
      return mysize;
    }
  };

  struct message_info_type
  {
    int partner_rank;
    int msg_tag;
    std::list<message_info_data_type> data_items;

    message_info_type(int partner_rank_, int msg_tag_)
      : partner_rank(partner_rank_), msg_tag(msg_tag_)
    { }

    message_info_data_type& add_data(MeshData const* msg_data)
    {
      for (message_info_data_type& data : data_items) {
        if (data.data == msg_data) {
          // found data item
          return data;
        }
      }
      // add data item
      data_items.emplace_back(msg_data);
      return data_items.back();
    }

    // get average of total size of boxes, all data (rounded up)
    IdxT average_data_size() const
    {
      IdxT mysize = 0;
      IdxT num_data = 0;
      for (message_info_data_type const& data : data_items) {
        for (Box3d const& box : data.boxes) {
          mysize += box.size();
        }
        num_data += 1;
      }
      return (mysize + num_data - 1) / num_data;
    }

    // get average of all size of boxes, all data (rounded up)
    IdxT average_item_size() const
    {
      IdxT mysize = 0;
      IdxT num_boxes = 0;
      for (message_info_data_type const& data : data_items) {
        for (Box3d const& box : data.boxes) {
          mysize += box.size();
          num_boxes += 1;
        }
      }
      return (mysize + num_boxes - 1) / num_boxes;
    }

    // get total size with boxes, all data
    IdxT total_size() const
    {
      IdxT mysize = 0;
      for (message_info_data_type const& data : data_items) {
        for (Box3d const& box : data.boxes) {
          mysize += box.size();
        }
      }
      return mysize;
    }
  };

  using msg_info_map_type = std::map<int, message_info_type>;

  template < typename comm_type >
  void populate(comm_type& comm,
                ExecContext<typename comm_type::policy_many>& con_many,
                ExecContext<typename comm_type::policy_few>& con_few) const
  {

    // map from partner rank to message indices
    msg_info_map_type recv_msg_info_map;
    // map from partner rank to message indices
    msg_info_map_type send_msg_info_map;

    // populate the msg_info_maps
    auto recv_send_end = msg_map.end();
    for (auto recv_send_iter = msg_map.begin(); recv_send_iter != recv_send_end; ++recv_send_iter) {

      Box3d const& recv_box = recv_send_iter->first;
      Box3d const& send_box = recv_send_iter->second;

      // recv_box.print("recv_box");
      // send_box.print("send_box");

      auto recv_rank_iter = rank_map.find(recv_box.info);
      assert(recv_rank_iter != rank_map.end());
      int recv_rank = recv_rank_iter->second;

      auto send_rank_iter = rank_map.find(send_box.info);
      assert(send_rank_iter != rank_map.end());
      int send_rank = send_rank_iter->second;

      // TODO: check for out of bounds tag
      int msg_tag = recv_rank;

      populate_msg_info_map(recv_msg_info_map, recv_box, send_rank, msg_tag);
      populate_msg_info_map(send_msg_info_map, send_box, recv_rank, msg_tag);
    }

    // use the msg_info_maps to populate messages in comm
    populate_comm(comm, con_many, con_few, detail::MessageBase::Kind::recv, comm.m_recvs, recv_msg_info_map);
    populate_comm(comm, con_many, con_few, detail::MessageBase::Kind::send, comm.m_sends, send_msg_info_map);

    comm.finish_populating(con_many, con_few);
  }

  ~CommFactory()
  {
  }

private:
  struct msg_info_type {
    LidxT* indices = nullptr;
    IdxT len = 0;
#ifdef COMB_ENABLE_MPI
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    IdxT mpi_pack_nbytes = 0;
#endif
  };

  void populate_msg_info_map(msg_info_map_type& msg_info_map, Box3d const& msg_box, int partner_rank, int msg_tag) const
  {
    auto msg_data_list_iter = data_map.find(msg_box.info);
    if (msg_data_list_iter != data_map.end()) {
      std::list<MeshData const*> const& msg_data_list = msg_data_list_iter->second;
      if (!msg_data_list.empty()) {

        auto msg_info_iter = msg_info_map.find(partner_rank);
        if (msg_info_iter == msg_info_map.end()) {
          // didn't find existing entry, add new
          auto ret = msg_info_map.emplace(std::make_pair(partner_rank, message_info_type{partner_rank, msg_tag}));
          assert(ret.second);
          msg_info_iter = ret.first;
        }
        message_info_type& msginfo = msg_info_iter->second;

        auto msg_data_end = msg_data_list.end();
        for (auto msg_data_iter = msg_data_list.begin(); msg_data_iter != msg_data_end; ++msg_data_iter) {
          MeshData const* msg_data = *msg_data_iter;

          message_info_data_type& data_info = msginfo.add_data(msg_data);

          data_info.add_box(msg_box);
        }
      }
    }
  }

  template < typename context >
  bool msg_info_items_combineable(context&) const
  {
    return allow_per_message_pack_fusing();
  }

#ifdef COMB_ENABLE_MPI
  bool msg_info_items_combineable(ExecContext<mpi_type_pol>&) const
  {
    return false;
  }
#endif

  template < typename context >
  std::list<msg_info_type> populate_msg_info(context& con, bool combineable
#ifdef COMB_ENABLE_MPI
                                            ,MPI_Comm
#endif
                                            ,message_info_data_type const& data_item) const
  {
    MeshData const* msg_data = data_item.data;

    std::list<msg_info_type> msg_info_list;

    IdxT offset = 0;

    if (combineable) {
      msg_info_list.emplace_back();
      msg_info_type& msg_info = msg_info_list.back();

      msg_info.len = data_item.total_size();
      msg_info.indices = (LidxT*)msg_data->aloc.allocate(sizeof(LidxT)*msg_info.len);
    }

    for (Box3d const& msg_box : data_item.boxes) {

      if (!combineable) {

        // add new item
        msg_info_list.emplace_back();
        msg_info_type& msg_info = msg_info_list.back();

        // fill item data
        msg_info.len = msg_box.size();
        msg_info.indices = (LidxT*)msg_data->aloc.allocate(sizeof(LidxT)*msg_info.len);
        msg_box.set_indices(con, msg_info.indices);

      } else {
        msg_info_type& msg_info = msg_info_list.back();

        // append data
        msg_box.set_indices(con, msg_info.indices + offset);
        offset += msg_box.size();
      }
    }

    return msg_info_list;
  }

#ifdef COMB_ENABLE_MPI
  std::list<msg_info_type> populate_msg_info(ExecContext<mpi_type_pol>&, bool combineable
                                            ,MPI_Comm comm
                                            ,message_info_data_type const& data_item) const
  {
    std::list<msg_info_type> msg_info_list;

    if (combineable) {
      assert(0);
    }

    for (Box3d const& msg_box : data_item.boxes) {

      if (!combineable) {

        // add new item
        msg_info_list.emplace_back();
        msg_info_type& msg_info = msg_info_list.back();

        // fill item data
        msg_info.len = msg_box.size();
        msg_info.mpi_type = msg_box.get_type_subarray();
        detail::MPI::Type_commit(&msg_info.mpi_type);
        msg_info.mpi_pack_nbytes = detail::MPI::Pack_size(1, msg_info.mpi_type, comm);

      } else {
        assert(0);
      }
    }

    return msg_info_list;
  }
#endif

  template < typename comm_type, typename msg_list_type >
  void populate_comm(comm_type& comm,
                     ExecContext<typename comm_type::policy_many>& con_many,
                     ExecContext<typename comm_type::policy_few>& con_few,
                     detail::MessageBase::Kind kind, msg_list_type& msg_list,
                     msg_info_map_type& msg_info_map) const
  {
    auto lambda = [&](message_info_type& msginfo) {

      // TODO: handle many,few separately
      bool combineable = msg_info_items_combineable(con_many) &&
                         msg_info_items_combineable(con_few);

      bool have_many = combineable
                         ? msginfo.average_data_size() >= comm.comminfo.cutoff
                         : msginfo.average_item_size() >= comm.comminfo.cutoff;

      // add a new message to the message list
      msg_list.emplace_back(kind, msginfo.partner_rank, msginfo.msg_tag, have_many);
      typename comm_type::message_type& msg = msg_list.back();

      for (message_info_data_type const& data_item : msginfo.data_items) {
        MeshData const* msg_data = data_item.data;

        std::list<msg_info_type> msg_info_list;

        if (have_many) {
          msg_info_list = populate_msg_info(con_many, combineable
#ifdef COMB_ENABLE_MPI
                                           ,comm.comminfo.cart.comm
#endif
                                           ,data_item);
        } else {
          msg_info_list = populate_msg_info(con_few, combineable
#ifdef COMB_ENABLE_MPI
                                           ,comm.comminfo.cart.comm
#endif
                                           ,data_item);
        }

        for (msg_info_type& msg_info : msg_info_list) {
          msg.add(msg_data->data(), msg_info.indices, msg_data->aloc, msg_info.len
#ifdef COMB_ENABLE_MPI
                 ,msg_info.mpi_type, msg_info.mpi_pack_nbytes
#endif
                 );
        }
      }
    };

    int myrank = comm.comminfo.rank;

    auto msg_info_begin  = msg_info_map.begin();
    auto msg_info_middle = msg_info_map.lower_bound(myrank);
    auto msg_info_end    = msg_info_map.end();

    // myrank = 4;
    // partner ranks = [2, 3, 7, 9];
    // send/recv order = [7, 9, 2, 3];

    // reorder messages so that each rank sends/recvs messages in order
    // starting with the lowest ranked partner whose rank is greater than its rank
    for (auto msg_info_iter = msg_info_middle; msg_info_iter != msg_info_end;    ++msg_info_iter) {
      lambda(msg_info_iter->second);
    }
    // then wrapping back around to its lowest ranked partner
    for (auto msg_info_iter = msg_info_begin;  msg_info_iter != msg_info_middle; ++msg_info_iter) {
      lambda(msg_info_iter->second);
    }
  }
};

#endif // _COMM_FACTORY_HPP

