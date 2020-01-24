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
  CommFactory(CommInfo const& comminfo_)
    : comminfo(comminfo_)
  { }

  ~CommFactory()
  {
  }

  void add_var(MeshData const& meshdata)
  {
    // add this meshinfo to data_map if not found
    auto iter = data_map.find(meshdata.info);
    if (iter == data_map.end()) {
      auto res = data_map.emplace(meshdata.info, typename data_map_type::mapped_type{});
      assert(res.second);
      iter = res.first;

      // add messages for this meshinfo
      add_mesh(meshdata.info);
    }

    // add meshdata for meshinfo in data_map
    iter->second.emplace_back(&meshdata);
  }

  template < typename comm_type >
  void populate(comm_type& comm,
                ExecContext<typename comm_type::policy_many>& con_many,
                ExecContext<typename comm_type::policy_few>& con_few) const
  {

    // list of MeshInfo to map from partner rank to message boxes
    mesh_info_map_type recv_mesh_info_map;
    // list of MeshInfo to map from partner rank to message boxes
    mesh_info_map_type send_mesh_info_map;

    // populate the msg_info_maps with message boxes in the right order
    for (auto const& recv_send : msg_map) {

      Box3d const& recv_box = recv_send.first;
      Box3d const& send_box = recv_send.second;

      // recv_box.print("recv_box");
      // send_box.print("send_box");

      int recv_rank = rank_map.at(recv_box.info);
      int send_rank = rank_map.at(send_box.info);

      // TODO: check for out of bounds tag
      int msg_tag = recv_rank;

      populate_mesh_info_map(recv_mesh_info_map, recv_box, send_rank, msg_tag);
      populate_mesh_info_map(send_mesh_info_map, send_box, recv_rank, msg_tag);
    }

    // use the msg_info_maps to populate messages in comm
    populate_comm(comm, con_many, con_few, comm.m_recvs, recv_mesh_info_map);
    populate_comm(comm, con_many, con_few, comm.m_sends, send_mesh_info_map);

    comm.finish_populating(con_many, con_few);
  }

private:
  struct message_info_data_type
  {
    std::list<Box3d> boxes;

    void add_box(Box3d const& msg_box)
    {
      boxes.emplace_back(msg_box);
    }

    IdxT num_boxes() const
    {
      return boxes.size();
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
    message_info_data_type data_items;

    message_info_type(int partner_rank_, int msg_tag_)
      : partner_rank(partner_rank_), msg_tag(msg_tag_)
    { }
  };

  using msg_info_map_type  = std::map<int, message_info_type>;
  using mesh_info_map_type = std::map<MeshInfo, msg_info_map_type>;

  using msg_map_type  = std::map<Box3d, Box3d>; // maps recv boxes to send boxes
  using data_map_type = std::map<MeshInfo, std::list<MeshData const*>>; // maps meshinfo to lists of meshdata
  using rank_map_type = std::map<MeshInfo, int>; // maps meshinfo to the rank that owns it

  CommInfo const& comminfo;

  // map from recv boxes (in the recv meshinfo's indices)
  //   to send boxes (in the send meshinfo's indices)
  msg_map_type msg_map;

  data_map_type data_map;

  rank_map_type rank_map;

  void add_mesh(MeshInfo const& meshinfo)
  {
    Box3d self_own            = Box3d::make_owned_box(meshinfo);
    Box3d self_potential_recv = Box3d::make_ghost_box(meshinfo);

    // self_own.print("self_own");
    // self_potential_recv.print("self_potential_recv");

    int self_rank = comminfo.cart.get_rank(meshinfo.global_coords);

    // add this meshinfo to rank_map if not already added
    auto iter = rank_map.find(meshinfo);
    if (iter == rank_map.end()) {
      auto res = rank_map.emplace(meshinfo, self_rank);
      assert(res.second);
      iter = res.first;

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
            auto res = rank_map.emplace(neighbor_recv.info, neighbor_rank);
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
            auto res = rank_map.emplace(neighbor_send.info, neighbor_rank);
            assert(res.second);
            iter = res.first;
          }
          assert(iter->first == neighbor_send.info);
          assert(iter->second == neighbor_rank);
        }


        auto my_recv_to_sends_iter = msg_map.find(self_recv);
        if (my_recv_to_sends_iter == msg_map.end()) {
          auto res = msg_map.emplace(self_recv, neighbor_send);
          assert(res.second);
          my_recv_to_sends_iter = res.first;
        }
        assert(my_recv_to_sends_iter->first  == self_recv);
        assert(my_recv_to_sends_iter->second == neighbor_send);

        auto neighbor_recv_to_sends_iter = msg_map.find(neighbor_recv);
        if (neighbor_recv_to_sends_iter == msg_map.end()) {
          auto res = msg_map.emplace(neighbor_recv, self_send);
          assert(res.second);
          neighbor_recv_to_sends_iter = res.first;
        }
        assert(neighbor_recv_to_sends_iter->first == neighbor_recv);
        assert(neighbor_recv_to_sends_iter->second == self_send);
      });
    }
  }

  void populate_mesh_info_map(mesh_info_map_type& mesh_info_map, Box3d const& msg_box, int partner_rank, int msg_tag) const
  {
    auto msg_data_list_iter = data_map.find(msg_box.info);
    if (msg_data_list_iter != data_map.end()) {
      std::list<MeshData const*> const& msg_data_list = msg_data_list_iter->second;
      if (!msg_data_list.empty()) {

        auto mesh_info_iter = mesh_info_map.find(msg_box.info);
        if (mesh_info_iter == mesh_info_map.end()) {
          // didn't find existing entry, add new
          auto ret = mesh_info_map.emplace(msg_box.info, std::map<int, message_info_type>{});
          assert(ret.second);
          mesh_info_iter = ret.first;
        }
        msg_info_map_type& msg_info_map = mesh_info_iter->second;

        auto msg_info_iter = msg_info_map.find(partner_rank);
        if (msg_info_iter == msg_info_map.end()) {
          // didn't find existing entry, add new
          auto ret = msg_info_map.emplace(partner_rank, message_info_type{partner_rank, msg_tag});
          assert(ret.second);
          msg_info_iter = ret.first;
        }
        message_info_type& msginfo = msg_info_iter->second;

        msginfo.data_items.add_box(msg_box);
      }
    }
  }

  template < typename context >
  bool msg_info_items_combineable(context&) const
  {
    return comb_allow_per_message_pack_fusing();
  }

#ifdef COMB_ENABLE_MPI
  bool msg_info_items_combineable(ExecContext<mpi_type_pol>&) const
  {
    return false;
  }
#endif

  template < typename comm_type, typename exec_policy, typename msg_group_type >
  void populate_msg_info(
      comm_type& comm,
      ExecContext<exec_policy>& con,
      msg_group_type& msg_group,
      int partner_rank,
      bool combineable,
      message_info_data_type const& data_item,
      COMB::Allocator& mesh_aloc) const
  {
    COMB::ignore_unused(comm);
    using message_item_type = detail::MessageItem<exec_policy>;

    IdxT combined_size = 0;
    IdxT combined_nbytes = 0;
    LidxT* combined_indices = nullptr;
    IdxT combined_offset = 0;

    if (combineable) {
      combined_size = data_item.total_size();
      combined_nbytes = sizeof(DataT)*combined_size; // data nbytes
      combined_indices = (LidxT*)mesh_aloc.allocate(sizeof(LidxT)*combined_size);
    }

    for (Box3d const& msg_box : data_item.boxes) {

      if (!combineable) {

        // fill item data
        IdxT size = msg_box.size();
        IdxT nbytes = sizeof(DataT)*size; // data nbytes
        LidxT* indices = (LidxT*)mesh_aloc.allocate(sizeof(LidxT)*size);
        msg_box.set_indices(con, indices);

        msg_group.add_message_item(
            partner_rank,
            message_item_type{size, nbytes, indices, mesh_aloc});

      } else {

        // append data
        msg_box.set_indices(con, combined_indices + combined_offset);
        combined_offset += msg_box.size();
      }
    }

    if (combineable) {
      msg_group.add_message_item(
          partner_rank,
          message_item_type{combined_size, combined_nbytes, combined_indices, mesh_aloc});
    }
  }

#ifdef COMB_ENABLE_MPI
  template < typename comm_type, typename msg_group_type >
  void populate_msg_info(
      comm_type& comm,
      ExecContext<mpi_type_pol>& con,
      msg_group_type& msg_group,
      int partner_rank,
      bool combineable,
      message_info_data_type const& data_item,
      COMB::Allocator& msg_aloc) const
  {
    COMB::ignore_unused(con, msg_aloc);
    using message_item_type = detail::MessageItem<mpi_type_pol>;

    assert(!combineable);

    for (Box3d const& msg_box : data_item.boxes) {

      // fill item data
      IdxT size = msg_box.size();
      MPI_Datatype mpi_type = msg_box.get_type_subarray();
      detail::MPI::Type_commit(&mpi_type);
      IdxT nbytes = detail::MPI::Pack_size(1, mpi_type, comm.con_comm.comm);

      msg_group.add_message_item(
          partner_rank,
          message_item_type{size, nbytes, mpi_type});
    }
  }
#endif

  struct msg_extra_info
  {
    IdxT size = 0;
    IdxT num_items = 0;
    bool combineable = true;
    bool have_many = false;
  };

  template < typename comm_type, typename msg_vars_type >
  void populate_comm(comm_type& comm,
                     ExecContext<typename comm_type::policy_many>& con_many,
                     ExecContext<typename comm_type::policy_few>& con_few,
                     msg_vars_type& msg_list,
                     mesh_info_map_type& mesh_info_map) const
  {
    // calculate combineable and have_many for each message
    std::map<int, msg_extra_info> msg_extras;

    for (auto& mesh_msg_item : mesh_info_map) {

      msg_info_map_type& msg_info_map = mesh_msg_item.second;

      for (auto msg_info_item : msg_info_map) {

        message_info_type& msginfo = msg_info_item.second;

        auto extras_iter = msg_extras.find(msginfo.partner_rank);
        if (extras_iter == msg_extras.end()) {
          auto res = msg_extras.emplace(msginfo.partner_rank, msg_extra_info{});
          assert(res.second);
          extras_iter = res.first;
        }
        msg_extra_info& extras = extras_iter->second;


        // TODO: handle many,few separately
        extras.combineable = extras.combineable &&
                             msg_info_items_combineable(con_many) &&
                             msg_info_items_combineable(con_few);

        extras.size += msginfo.data_items.total_size();
        extras.num_items += msginfo.data_items.num_boxes();
        IdxT avg_size = (extras.size + extras.num_items - 1) / extras.num_items;

        extras.have_many = extras.combineable
                             ? extras.size >= comm.comminfo.cutoff
                             : avg_size >= comm.comminfo.cutoff;
      }
    }

    int myrank = comm.comminfo.rank;

    // the MeshGroup coding can only handle one kind of MeshInfo
    assert(mesh_info_map.size() <= 1);

    for (auto& mesh_msg_item : mesh_info_map) {

      MeshInfo const& meshinfo = mesh_msg_item.first;
      msg_info_map_type& msg_info_map = mesh_msg_item.second;

      std::list<MeshData const*> const& msg_data_list = data_map.at(meshinfo);

      // skip this MeshInfo if it isn't used
      if (msg_data_list.size() == 0) continue;

      // add variables for this MeshInfo
      for (MeshData const* msg_data : msg_data_list) {
        DataT *data = msg_data->data();
        msg_list.message_group_many.add_variable(data);
        msg_list.message_group_few.add_variable(data);
      }

      // get allocator for mesh for use with indices
      COMB::Allocator& mesh_aloc = msg_data_list.front()->aloc;

      // add message and each box per message to the comm
      auto lambda = [&](message_info_type& msginfo) {

        msg_extra_info& extras = msg_extras.at(msginfo.partner_rank);

        bool combineable = extras.combineable;
        bool have_many   = extras.have_many;

        // add a new message to the message group
        if (have_many) {
          msg_list.message_group_many.add_message(con_many, msginfo.partner_rank, msginfo.msg_tag);
          populate_msg_info(comm, con_many, msg_list.message_group_many, msginfo.partner_rank, combineable, msginfo.data_items, mesh_aloc);
        } else {
          msg_list.message_group_few.add_message(con_few, msginfo.partner_rank, msginfo.msg_tag);
          populate_msg_info(comm, con_few, msg_list.message_group_few, msginfo.partner_rank, combineable, msginfo.data_items, mesh_aloc);
        }
      };

      // order messages (myrank-end), [begin-myrank)
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

      msg_list.message_group_many.finalize();
      msg_list.message_group_few.finalize();

    }
  }
};

#endif // _COMM_FACTORY_HPP

