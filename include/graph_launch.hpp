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

#ifndef _GRAPH_LAUNCH_HPP
#define _GRAPH_LAUNCH_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA_GRAPH

#include "utils.hpp"
#include "utils_cuda.hpp"
#include <list>

namespace cuda {

namespace graph_launch {

namespace detail {

struct Graph
{
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  cudaEvent_t event;
  bool launched;
  bool event_recorded;
  int num_events;

  Graph()
    : launched(false)
    , event_recorded(false)
    , num_events(0)
  {
    cudaCheck(cudaGraphCreate(&graph, 0));
  }

  // no copy construction
  Graph(Graph const&) = delete;
  Graph(Graph &&) = delete;

  // no copy assignment
  Graph& operator=(Graph const&) = delete;
  Graph& operator=(Graph &&) = delete;

  void createRecordEvent(cudaStream_t stream)
  {
    if (!event_recorded) {
      cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      cudaCheck(cudaEventRecord(event, stream));
      event_recorded = true;
    }
  }

  void add_event(cudaStream_t stream)
  {
    ++num_events;
    if (launched) {
      createRecordEvent(stream);
    }
  }

  bool query_event()
  {
    assert(num_events > 0);
    if (!launched) return false;
    assert(event_recorded);
    return cudaCheckReady(cudaEventQuery(event));
  }

  void wait_event()
  {
    assert(num_events > 0);
    assert(launched);
    assert(event_recorded);
    cudaCheck(cudaEventSynchronize(event));
  }

  void launch(cudaStream_t stream)
  {
    if (!launched) {
      cudaGraphNode_t errorNode;
      constexpr size_t bufferSize = 1024;
      char logBuffer[bufferSize] = "";
      cudaCheck(cudaGraphInstantiate(&graphExec, graph, &errorNode, logBuffer, bufferSize));

      cudaCheck(cudaGraphLaunch(graphExec, stream));

      if (num_events > 0) {
        createRecordEvent(stream);
      }

      launched = true;
    }
  }

  void remove_event()
  {
    --num_events;
  }

  ~Graph()
  {
    if (event_recorded) {
      cudaCheck(cudaEventDestroy(event));
    }
    if (launched) {
      cudaCheck(cudaGraphExecDestroy(graphExec));
    }
    cudaCheck(cudaGraphDestroy(graph));
  }
};

inline std::list<Graph>& getGraphs()
{
  static std::list<Graph> graphList{};
  return graphList;
}

struct Event
{
  std::list<Graph>::iterator graph_iter;
};

inline std::list<Event>& getEventList()
{
  static std::list<Event> list{};
  return list;
}

inline void enqueue(std::list<Graph>& graphs, cudaKernelNodeParams& params)
{
  if (graphs.empty()) {
    // list is empty
    graphs.emplace_front();
  } else if (graphs.front().launched) {
    if (graphs.front().num_events == 0) {
      // erase previous graph if it has no events
      graphs.erase(graphs.begin());
    }
    // graph is already launched, make another
    graphs.emplace_front();
  }
  Graph& graph = graphs.front();

  const cudaGraphNode_t* dependencies = nullptr;
  int num_dependencies = 0;

  cudaGraphNode_t node;
  cudaCheck(cudaGraphAddKernelNode(&node, graph.graph, dependencies, num_dependencies, &params));
}


extern void launch(Graph& graph, cudaStream_t stream);

} // namespace detail

using event_type = typename std::list<detail::Event>::iterator;

inline event_type createEvent()
{
  // create event pointing to invalid graph iterator
  detail::getEventList().emplace_front(detail::Event{detail::getGraphs().end()});
  return detail::getEventList().begin();
}

inline void recordEvent(event_type event, cudaStream_t stream = 0)
{
  if (!detail::getGraphs().empty()) {
    event->graph_iter = detail::getGraphs().begin();
    event->graph_iter->add_event(stream);
  }
}

inline bool queryEvent(event_type event)
{
  if (event->graph_iter == detail::getGraphs().end()) return true;
  return event->graph_iter->query_event();
}

inline void waitEvent(event_type event)
{
  if (event->graph_iter == detail::getGraphs().end()) return;
  event->graph_iter->wait_event();
}

inline void destroyEvent(event_type event)
{
  if (event->graph_iter == detail::getGraphs().end()) return;
  event->graph_iter->remove_event();
  if (event->graph_iter->num_events == 0) {
    detail::getGraphs().erase(event->graph_iter);
  }
  detail::getEventList().erase(event);
}


extern void force_launch(cudaStream_t stream = 0);
extern void synchronize(cudaStream_t stream = 0);

template < typename body_type >
__global__
void graph_for_all(IdxT begin, IdxT len, body_type body)
{
  const IdxT i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    body(i + begin, i);
  }
}

template <typename body_type>
inline void for_all(int begin, int end, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  int len = end - begin;

  const IdxT threads = 256;
  const IdxT blocks = (len + threads - 1) / threads;

  cudaKernelNodeParams params;
  params.func = (void*)&graph_for_all<decayed_body_type>;
  params.gridDim = dim3{(unsigned int)blocks};
  params.blockDim = dim3{(unsigned int)threads};
  params.sharedMemBytes = 0;
  void* args[]{&begin, &len, &body};
  params.kernelParams = args;
  params.extra = nullptr;

  detail::enqueue(detail::getGraphs(), params);
}

template < typename body_type >
__global__
void graph_for_all_2d(IdxT begin0, IdxT len0, IdxT begin1, IdxT len1, body_type body)
{
  const IdxT i0 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      IdxT i = i0 * len1 + i1;
      body(i0 + begin0, i1 + begin1, i);
    }
  }
}

template < typename body_type >
inline void for_all_2d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;

  const IdxT threads0 = 8;
  const IdxT threads1 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;

  cudaKernelNodeParams params;
  params.func = (void*)&graph_for_all_2d<decayed_body_type>;
  params.gridDim = dim3{(unsigned int)blocks1, (unsigned int)blocks0, (unsigned int)1};
  params.blockDim = dim3{(unsigned int)threads1, (unsigned int)threads0, (unsigned int)1};
  params.sharedMemBytes = 0;
  void* args[]{&begin0, &len0, &begin1, &len1, &body};
  params.kernelParams = args;
  params.extra = nullptr;

  detail::enqueue(detail::getGraphs(), params);
}

template < typename body_type >
__global__
void graph_for_all_3d(IdxT begin0, IdxT len0, IdxT begin1, IdxT len1, IdxT begin2, IdxT len2, IdxT len12, body_type body)
{
  const IdxT i0 = blockIdx.z;
  const IdxT i1 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i2 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      if (i2 < len2) {
        IdxT i = i0 * len12 + i1 * len2 + i2;
        body(i0 + begin0, i1 + begin1, i2 + begin2, i);
      }
    }
  }
}

template < typename body_type >
inline void for_all_3d(IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;
  IdxT len2 = end2 - begin2;
  IdxT len12 = len1 * len2;

  const IdxT threads0 = 1;
  const IdxT threads1 = 8;
  const IdxT threads2 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;
  const IdxT blocks2 = (len2 + threads2 - 1) / threads2;

  cudaKernelNodeParams params;
  params.func =(void*)&graph_for_all_3d<decayed_body_type>;
  params.gridDim = dim3{(unsigned int)blocks2, (unsigned int)blocks1, (unsigned int)blocks0};
  params.blockDim = dim3{(unsigned int)threads2, (unsigned int)threads1, (unsigned int)threads0};
  params.sharedMemBytes = 0;
  void* args[]{&begin0, &len0, &begin1, &len1, &begin2, &len2, &len12, &body};
  params.kernelParams = args;
  params.extra = nullptr;

  detail::enqueue(detail::getGraphs(), params);
}

} // namespace graph_launch

} // namespace cuda

#endif

#endif // _GRAPH_LAUNCH_HPP

