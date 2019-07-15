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

#ifndef _GRAPH_LAUNCH_HPP
#define _GRAPH_LAUNCH_HPP

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA_GRAPH

#include "utils.hpp"
#include "utils_cuda.hpp"
#include <vector>
#include <assert.h>

namespace cuda {

namespace graph_launch {

namespace detail {

struct Graph
{
  struct Event
  {
    Graph* graph = nullptr;
  };

  Graph()
    : m_launched(false)
    , m_event_recorded(false)
    , m_instantiated_num_nodes(false)
    , m_num_nodes(0)
    , m_ref(1)
    , m_num_events(0)
  {
    cudaCheck(cudaGraphCreate(&m_graph, 0));
  }

  // no copy construction
  Graph(Graph const&) = delete;
  Graph(Graph &&) = delete;

  // no copy assignment
  Graph& operator=(Graph const&) = delete;
  Graph& operator=(Graph &&) = delete;

  int inc()
  {
    return ++m_ref;
  }

  int dec()
  {
    return --m_ref;
  }

  void add_event(Event* event, cudaStream_t stream)
  {
    assert(event != nullptr);
    assert(event->graph == this);
    inc();
    m_num_events++;
    assert(m_num_events > 0);
    if (m_launched) {
      createRecordEvent(stream);
    }
  }

  bool query_event(Event* event)
  {
    assert(event != nullptr);
    assert(event->graph == this);
    assert(m_num_events > 0);
    if (!m_launched) return false;
    assert(m_event_recorded);
    return cudaCheckReady(cudaEventQuery(m_event));
  }

  void wait_event(Event* event)
  {
    assert(event != nullptr);
    assert(event->graph == this);
    assert(m_num_events > 0);
    assert(m_launched);
    assert(m_event_recorded);
    cudaCheck(cudaEventSynchronize(m_event));
  }

  int remove_event(Event* event)
  {
    assert(event != nullptr);
    assert(event->graph == this);
    assert(m_num_events > 0);
    m_num_events--;
    return dec();
  }

  void enqueue(cudaKernelNodeParams& params)
  {
    assert(!m_launched);
    if (m_num_nodes < m_nodes.size()) {
      if (m_num_nodes < m_instantiated_num_nodes) {
        cudaCheck(cudaGraphExecKernelNodeSetParams(m_graphExec, m_nodes[m_num_nodes], &params));
      }
      cudaCheck(cudaGraphKernelNodeSetParams(m_nodes[m_num_nodes], &params));
    } else {
      assert(m_num_nodes == m_nodes.size());
      m_nodes.emplace_back();
      const cudaGraphNode_t* dependencies = nullptr;
      int num_dependencies = 0;
      cudaCheck(cudaGraphAddKernelNode(&m_nodes[m_num_nodes], m_graph, dependencies, num_dependencies, &params));
    }
    m_num_nodes++;
  }

  void launch(cudaStream_t stream)
  {
    // NVTX_RANGE_COLOR(NVTX_CYAN)
    if (!m_launched) {
      if (m_instantiated_num_nodes != m_num_nodes) {
        if (m_instantiated_num_nodes > 0) {
          cudaCheck(cudaGraphExecDestroy(m_graphExec));
          for (int i = m_num_nodes; i < m_nodes.size(); ++i) {
            cudaCheck(cudaGraphDestroyNode(m_nodes[i]));
          }
          m_nodes.resize(m_num_nodes);
          m_instantiated_num_nodes = 0;
        }
        cudaGraphNode_t errorNode;
        constexpr size_t bufferSize = 1024;
        char logBuffer[bufferSize] = "";
        cudaCheck(cudaGraphInstantiate(&m_graphExec, m_graph, &errorNode, logBuffer, bufferSize));
        m_instantiated_num_nodes = m_num_nodes;
      }

      cudaCheck(cudaGraphLaunch(m_graphExec, stream));

      if (m_num_events > 0) {
        createRecordEvent(stream);
      }

      m_launched = true;
    }
  }

  void reuse()
  {
    assert(m_num_events == 0);
    m_launched = false;
    m_num_nodes = 0;
  }

  bool launchable() const
  {
    return !m_launched && m_num_nodes > 0;
  }

  ~Graph()
  {
    assert(m_ref == 0);
    assert(m_num_events == 0);
    if (m_event_recorded) {
      cudaCheck(cudaEventDestroy(m_event));
    }
    if (m_instantiated_num_nodes > 0) {
      cudaCheck(cudaGraphExecDestroy(m_graphExec));
    }
    for (int i = 0; i < m_nodes.size(); ++i) {
      cudaCheck(cudaGraphDestroyNode(m_nodes[i]));
    }
    m_nodes.clear();
    cudaCheck(cudaGraphDestroy(m_graph));
  }
private:
  std::vector<cudaGraphNode_t> m_nodes;
  cudaGraph_t m_graph;
  cudaGraphExec_t m_graphExec;
  cudaEvent_t m_event;

  bool m_launched;
  bool m_event_recorded;
  int m_instantiated_num_nodes;
  int m_num_nodes;
  int m_ref;
  int m_num_events;

  void createRecordEvent(cudaStream_t stream)
  {
    if (!m_event_recorded) {
      cudaCheck(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
      cudaCheck(cudaEventRecord(m_event, stream));
      m_event_recorded = true;
    }
  }
};

} // namespace detail

struct component
{
  void* data = nullptr;
};

struct group
{
  detail::Graph* graph = nullptr;
};

inline group create_group()
{
  // graph starts at ref count 1
  group g{};
  g.graph = new detail::Graph{};
  return g;
}

inline group& get_active_group()
{
  static group g = create_group();
  return g;
}

inline void destroy_group(group g)
{
  if (g.graph && g.graph->dec() <= 0) {
    delete g.graph;
  }
}

inline void new_active_group()
{
  destroy_group(get_active_group());
  get_active_group() = create_group();
}

inline void set_active_group(group g)
{
  if (g.graph != get_active_group().graph) {
    destroy_group(get_active_group());
    get_active_group() = g;
    get_active_group().graph->inc();
  }
  get_active_group().graph->reuse();
}

using event_type = detail::Graph::Event*;

inline event_type createEvent()
{
  // create event pointing to invalid graph iterator
  return new detail::Graph::Event();
}

inline void recordEvent(event_type event, cudaStream_t stream = 0)
{
  assert(event->graph == nullptr || event->graph == get_active_group().graph);
  event->graph = get_active_group().graph;
  event->graph->add_event(event, stream);
}

inline bool queryEvent(event_type event)
{
  if (event->graph == nullptr) return true;
  return event->graph->query_event(event);
}

inline void waitEvent(event_type event)
{
  if (event->graph == nullptr) return;
  event->graph->wait_event(event);
}

inline void destroyEvent(event_type event)
{
  if (event->graph == nullptr) return;
  if (event->graph->remove_event(event) <= 0) {
    delete event->graph; event->graph = nullptr;
  }
  delete event;
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

  get_active_group().graph->enqueue(params);
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

  get_active_group().graph->enqueue(params);
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

  get_active_group().graph->enqueue(params);
}

} // namespace graph_launch

} // namespace cuda

#endif

#endif // _GRAPH_LAUNCH_HPP

