//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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

// cuda 10.1 and higher have graph kernel node set params
#if CUDART_VERSION >= 10010
#define COMB_HAVE_CUDA_GRAPH_KERNEL_NODE_SET_PARAMS
#endif

// cuda 10.2 and higher have graph update
#if CUDART_VERSION >= 10020
#define COMB_HAVE_CUDA_GRAPH_UPDATE
#endif

#include "ExecContext.hpp"
#include "exec_utils.hpp"
#include "exec_utils_cuda.hpp"
#include <vector>
#include <assert.h>

// enable empty nodes at the begin and end of each graph instead of allowing disconnected components
// #define COMB_GRAPH_BEGIN_END_NODES

// enable launching kernels immediately instead of creating graphs
// #define COMB_GRAPH_KERNEL_LAUNCH
// enable launching kernels immediately into per component streams
// #define COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS

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
    , m_event_created(false)
    , m_instantiated_num_nodes(false)
    , m_num_nodes(0)
    , m_ref(1)
    , m_num_events(0)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).Graph\n", this );
    createGraph();
  }

  // no copy construction
  Graph(Graph const&) = delete;
  Graph(Graph &&) = delete;

  // no copy assignment
  Graph& operator=(Graph const&) = delete;
  Graph& operator=(Graph &&) = delete;

  int inc()
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).inc %i\n", this, m_ref+1 );
    return ++m_ref;
  }

  int dec()
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).dec %i\n", this, m_ref );
    return --m_ref;
  }

  void add_event(Event* event, cudaStream_t stream)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).add_event(%p) stream(%p)\n", this, event, (void*)stream );
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
    // LOGPRINTF("cuda::graph_launch::Graph(%p).query_event(%p)\n", this, event );
    assert(event != nullptr);
    assert(event->graph == this);
    assert(m_num_events > 0);
    if (!m_launched) return false;
    assert(m_event_recorded);
    // LOGPRINTF("cuda::graph_launch::Graph(%p).query_event(%p) cudaEventQuery(%p)\n", this, event, &m_event );
    bool done = cudaCheckReady(cudaEventQuery(m_event));
    // LOGPRINTF("cuda::graph_launch::Graph(%p).query_event(%p) cudaEventQuery(%p) -> %s\n", this, event, &m_event, (done ? "true" : "false"));
    return done;
  }

  void wait_event(Event* event)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).wait_event(%p) %p\n", this, event );
    assert(event != nullptr);
    assert(event->graph == this);
    assert(m_num_events > 0);
    assert(m_launched);
    assert(m_event_recorded);
    // LOGPRINTF("cuda::graph_launch::Graph(%p).wait_event(%p) cudaEventSynchronize(%p)\n", this, event, &m_event );
    cudaCheck(cudaEventSynchronize(m_event));
    // LOGPRINTF("cuda::graph_launch::Graph(%p).wait_event(%p) cudaEventSynchronize(%p) -> done\n", this, event, &m_event );
  }

  int remove_event(Event* event)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).remove_event(%p)\n", this, event );
    assert(event != nullptr);
    assert(event->graph == this);
    assert(m_num_events > 0);
    m_num_events--;
    return dec();
  }

  void enqueue(cudaKernelNodeParams& params)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).enqueue %p grid(%i,%i,%i) block(%i,%i,%i) shmem(%zu) params(%p) extra(%p)\n", this, params.func, (int)params.gridDim.x, (int)params.gridDim.y, (int)params.gridDim.z, (int)params.blockDim.x, (int)params.blockDim.y, (int)params.blockDim.z, (size_t)params.sharedMemBytes, params.kernelParams, params.extra );
#ifndef COMB_GRAPH_KERNEL_LAUNCH
    assert(!m_launched);
    if (m_num_nodes < m_nodes.size()) {
      if (m_num_nodes < m_instantiated_num_nodes) {
#if defined(COMB_HAVE_CUDA_GRAPH_KERNEL_NODE_SET_PARAMS) && !defined(COMB_HAVE_CUDA_GRAPH_UPDATE)
        // LOGPRINTF("cuda::graph_launch::Graph(%p).enqueue cudaGraphExecKernelNodeSetParams(%p, %p)\n", this, &m_graphExec, &m_nodes[m_num_nodes] );
        cudaCheck(cudaGraphExecKernelNodeSetParams(m_graphExec, m_nodes[m_num_nodes], &params));
#endif
      } else {
        assert(0 && (m_num_nodes < m_instantiated_num_nodes));
      }
#ifdef COMB_HAVE_CUDA_GRAPH_UPDATE
      // LOGPRINTF("cuda::graph_launch::Graph(%p).enqueue cudaGraphKernelNodeSetParams(%p, %p)\n", this, &m_graph, &m_nodes[m_num_nodes] );
      cudaCheck(cudaGraphKernelNodeSetParams(m_nodes[m_num_nodes], &params));
#endif
    } else {
      assert(m_num_nodes == m_nodes.size());
      m_nodes.emplace_back();
#ifdef COMB_GRAPH_BEGIN_END_NODES
      const cudaGraphNode_t* dependencies = &m_node_begin;
      int num_dependencies = 1;
#else
      const cudaGraphNode_t* dependencies = nullptr;
      int num_dependencies = 0;
#endif
      // LOGPRINTF("cuda::graph_launch::Graph(%p).enqueue cudaGraphAddKernelNode(%p)\n", this, &m_graph );
      cudaCheck(cudaGraphAddKernelNode(&m_nodes[m_num_nodes], m_graph, dependencies, num_dependencies, &params));
      // LOGPRINTF("cuda::graph_launch::Graph(%p).enqueue cudaGraphAddKernelNode(%p) -> %p\n", this, &m_graph, &m_nodes[m_num_nodes] );
    }
#endif
    m_num_nodes++;
  }

  void update(cudaStream_t stream)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).update\n", this );
    // NVTX_RANGE_COLOR(NVTX_CYAN)
    if (!m_launched) {
      if (m_instantiated_num_nodes != m_num_nodes) {
        if (m_instantiated_num_nodes > 0) {
#ifndef COMB_GRAPH_KERNEL_LAUNCH
          // LOGPRINTF("cuda::graph_launch::Graph(%p).update cudaGraphExecDestroy(%p)\n", this, &m_graphExec );
          cudaCheck(cudaGraphExecDestroy(m_graphExec));
#ifdef COMB_GRAPH_BEGIN_END_NODES
          cudaCheck(cudaGraphDestroyNode(m_node_end));
#endif
          for (int i = m_num_nodes; i < m_nodes.size(); ++i) {
            // LOGPRINTF("cuda::graph_launch::Graph(%p).update cudaGraphDestroyNode(%p)\n", this, &m_nodes[i] );
            cudaCheck(cudaGraphDestroyNode(m_nodes[i]));
          }
          m_nodes.resize(m_num_nodes);
#endif
          m_instantiated_num_nodes = 0;
        }
#ifndef COMB_GRAPH_KERNEL_LAUNCH
#ifdef COMB_GRAPH_BEGIN_END_NODES
        // add end node depending on all kernel nodes
        cudaCheck(cudaGraphAddEmptyNode(&m_node_end, m_graph, &m_nodes[0], m_num_nodes));
#endif
        // LOGPRINTF("cuda::graph_launch::Graph(%p).update cudaGraphInstantiate(%p)\n", this, &m_graph );
        cudaGraphNode_t errorNode;
        constexpr size_t bufferSize = 1024;
        char logBuffer[bufferSize] = "";
        cudaCheck(cudaGraphInstantiate(&m_graphExec, m_graph, &errorNode, logBuffer, bufferSize));
#endif
        m_instantiated_num_nodes = m_num_nodes;
        // LOGPRINTF("cuda::graph_launch::Graph(%p).update cudaGraphInstantiate(%p) -> %p\n", this, &m_graph, &m_graphExec );
      }
#ifndef COMB_GRAPH_KERNEL_LAUNCH
#ifdef COMB_HAVE_CUDA_GRAPH_UPDATE
      else if (m_instantiated_num_nodes > 0) {
        // LOGPRINTF("cuda::graph_launch::Graph(%p).update cudaGraphExecUpdate(%p, %p)\n", this, &m_graph, &m_graphExec );
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdateResult result;
        cudaCheck(cudaGraphExecUpdate(m_graphExec, m_graph, &errorNode, &result));
      }
#endif
#endif
    }
  }

  void launch(cudaStream_t stream)
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).launch\n", this );
    // NVTX_RANGE_COLOR(NVTX_CYAN)
    if (!m_launched) {

#ifndef COMB_GRAPH_KERNEL_LAUNCH
      // LOGPRINTF("cuda::graph_launch::Graph(%p).launch cudaGraphLaunch(%p) stream(%p)\n", this, &m_graphExec, (void*)stream );
      cudaCheck(cudaGraphLaunch(m_graphExec, stream));
#endif

      if (m_num_events > 0) {
        createRecordEvent(stream);
      }

      m_launched = true;
    }
  }

  void reuse()
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).reuse\n", this );
    assert(m_num_events == 0);
    m_event_recorded = false;
    m_launched = false;
    m_num_nodes = 0;
#if !(defined(COMB_HAVE_CUDA_GRAPH_KERNEL_NODE_SET_PARAMS) || defined(COMB_HAVE_CUDA_GRAPH_UPDATE))
    destroyGraph();
    createGraph();
#endif
  }

  bool launchable() const
  {
    return !m_launched && m_num_nodes > 0;
  }

  ~Graph()
  {
    // LOGPRINTF("cuda::graph_launch::Graph(%p).~Graph\n", this );
    assert(m_ref == 0);
    assert(m_num_events == 0);
    if (m_event_created) {
      // LOGPRINTF("cuda::graph_launch::Graph(%p).~Graph cudaEventDestroy(%p)\n", this, &m_event );
      cudaCheck(cudaEventDestroy(m_event));
    }
    destroyGraph();
  }
private:
#ifndef COMB_GRAPH_KERNEL_LAUNCH
  std::vector<cudaGraphNode_t> m_nodes;
#ifdef COMB_GRAPH_BEGIN_END_NODES
  cudaGraphNode_t m_node_begin;
  cudaGraphNode_t m_node_end;
#endif
  cudaGraph_t m_graph;
  cudaGraphExec_t m_graphExec;
#endif
  cudaEvent_t m_event;

  bool m_launched;
  bool m_event_recorded;
  bool m_event_created;
  int m_instantiated_num_nodes;
  int m_num_nodes;
  int m_ref;
  int m_num_events;

  void createRecordEvent(cudaStream_t stream)
  {
    if (!m_event_created) {
      // LOGPRINTF("cuda::graph_launch::Graph(%p).createRecordEvent cudaEventCreateWithFlags()\n", this );
      cudaCheck(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
      // LOGPRINTF("cuda::graph_launch::Graph(%p).createRecordEvent cudaEventCreateWithFlags() -> %p\n", this, &m_event );
      m_event_created = true;
    }
    if (!m_event_recorded) {
      // LOGPRINTF("cuda::graph_launch::Graph(%p).createRecordEvent cudaEventRecord(%p) stream(%p)\n", this, &m_event, (void*)stream );
      cudaCheck(cudaEventRecord(m_event, stream));
      m_event_recorded = true;
    }
  }

  void createGraph()
  {
#ifndef COMB_GRAPH_KERNEL_LAUNCH
    // LOGPRINTF("cuda::graph_launch::Graph(%p).createGraph cudaGraphCreate()\n", this );
    cudaCheck(cudaGraphCreate(&m_graph, 0));
    // LOGPRINTF("cuda::graph_launch::Graph(%p).createGraph cudaGraphCreate() -> %p\n", this, &m_graph );
#ifdef COMB_GRAPH_BEGIN_END_NODES
    // insert begin node
    cudaCheck(cudaGraphAddEmptyNode(&m_node_begin, m_graph, nullptr, 0));
#endif
#endif
  }

  void destroyGraph()
  {
#ifndef COMB_GRAPH_KERNEL_LAUNCH
    if (m_instantiated_num_nodes > 0) {
      // LOGPRINTF("cuda::graph_launch::Graph(%p).destroyGraph cudaGraphExecDestroy(%p)\n", this, &m_graphExec );
      cudaCheck(cudaGraphExecDestroy(m_graphExec));
    }
#ifdef COMB_GRAPH_BEGIN_END_NODES
    if (m_nodes.size() > 0) {
      cudaCheck(cudaGraphDestroyNode(m_node_begin));
    }
    if (m_instantiated_num_nodes > 0) {
      cudaCheck(cudaGraphDestroyNode(m_node_end));
    }
#endif
    for (int i = 0; i < m_nodes.size(); ++i) {
      // LOGPRINTF("cuda::graph_launch::Graph(%p).destroyGraph cudaGraphDestroyNode(%p)\n", this, &m_nodes[i] );
      cudaCheck(cudaGraphDestroyNode(m_nodes[i]));
    }
    m_nodes.clear();
    m_num_nodes = 0;
    m_instantiated_num_nodes = 0;
    m_launched = false;
    // LOGPRINTF("cuda::graph_launch::Graph(%p).destroyGraph cudaGraphDestroy(%p)\n", this, &m_graph );
    cudaCheck(cudaGraphDestroy(m_graph));
#endif
  }
};

} // namespace detail

struct component
{
#ifdef COMB_GRAPH_KERNEL_LAUNCH_COMPONENT_STREAMS
  ::CudaContext m_con;
#else
  void* data = nullptr;
#endif
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

inline void destroy_group(group g)
{
  if (g.graph && g.graph->dec() <= 0) {
    delete g.graph;
  }
}

inline group& get_group()
{
  static group g{};
  return g;
}

inline group& get_active_group()
{
  group& g = get_group();
  if (g.graph == nullptr) {
    g = create_group();
  }
  return g;
}

inline void new_active_group()
{
  destroy_group(get_group());
  get_group().graph = nullptr;
}

inline void set_active_group(group g)
{
  if (g.graph != get_group().graph) {
    destroy_group(get_group());
    get_group() = g;
    if (get_group().graph != nullptr) {
      get_group().graph->inc();
    }
  }
  if (get_group().graph != nullptr) {
    get_group().graph->reuse();
  }
}

using event_type = detail::Graph::Event*;

inline event_type createEvent()
{
  // create event pointing to invalid graph iterator
  return new detail::Graph::Event();
}

inline void recordEvent(event_type& event, cudaStream_t stream = 0)
{
  assert(get_group().graph != nullptr);
  assert(event->graph == nullptr || event->graph == get_group().graph);
  event->graph = get_group().graph;
  if (event->graph) {
    event->graph->add_event(event, stream);
  }
}

inline bool queryEvent(event_type& event)
{
  if (event->graph == nullptr) return true;
  bool done = event->graph->query_event(event);
  if (done) {
    if (event->graph->remove_event(event) <= 0) {
      delete event->graph;
    }
    event->graph = nullptr;
  }
  return done;
}

inline void waitEvent(event_type& event)
{
  if (event->graph == nullptr) return;
  event->graph->wait_event(event);
  if (event->graph->remove_event(event) <= 0) {
    delete event->graph;
  }
  event->graph = nullptr;
}

inline void destroyEvent(event_type& event)
{
  if (event->graph == nullptr) return;
  if (event->graph->remove_event(event) <= 0) {
    delete event->graph;
  }
  event->graph = nullptr;
  delete event;
}

// Ensure the current graph launched (actually launches graph)
inline void force_launch(cudaStream_t stream = 0)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (get_group().graph != nullptr && get_group().graph->launchable()) {
      get_group().graph->update(stream);
      get_group().graph->launch(stream);
      new_active_group();
   }
}

// Wait for graph to finish running
inline void synchronize(cudaStream_t stream = 0)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   force_launch(stream);

   // perform synchronization
   cudaCheck(cudaStreamSynchronize(stream));
}

template < typename body_type >
__global__
void graph_for_all(IdxT len, body_type body)
{
  const IdxT i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    body(i);
  }
}

template <typename body_type>
inline void for_all(int len, body_type&& body
#ifdef COMB_GRAPH_KERNEL_LAUNCH
    , cudaStream_t stream
#endif
    )
{
  using decayed_body_type = typename std::decay<body_type>::type;

  const IdxT threads = 256;
  const IdxT blocks = (len + threads - 1) / threads;

  cudaKernelNodeParams params;
  params.func = (void*)&graph_for_all<decayed_body_type>;
  params.gridDim = dim3{(unsigned int)blocks};
  params.blockDim = dim3{(unsigned int)threads};
  params.sharedMemBytes = 0;
  void* args[]{&len, &body};
  params.kernelParams = args;
  params.extra = nullptr;

  get_active_group().graph->enqueue(params);

#ifdef COMB_GRAPH_KERNEL_LAUNCH
  cudaCheck(cudaLaunchKernel(params.func, params.gridDim, params.blockDim, params.kernelParams, params.sharedMemBytes, stream));
#endif
}

template < typename body_type >
__global__
void graph_for_all_2d(IdxT len0, IdxT len1, body_type body)
{
  const IdxT i0 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      body(i0, i1);
    }
  }
}

template < typename body_type >
inline void for_all_2d(IdxT len0, IdxT len1, body_type&& body
#ifdef COMB_GRAPH_KERNEL_LAUNCH
    , cudaStream_t stream
#endif
    )
{
  using decayed_body_type = typename std::decay<body_type>::type;

  const IdxT threads0 = 8;
  const IdxT threads1 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;

  cudaKernelNodeParams params;
  params.func = (void*)&graph_for_all_2d<decayed_body_type>;
  params.gridDim = dim3{(unsigned int)blocks1, (unsigned int)blocks0, (unsigned int)1};
  params.blockDim = dim3{(unsigned int)threads1, (unsigned int)threads0, (unsigned int)1};
  params.sharedMemBytes = 0;
  void* args[]{&len0, &len1, &body};
  params.kernelParams = args;
  params.extra = nullptr;

  get_active_group().graph->enqueue(params);

#ifdef COMB_GRAPH_KERNEL_LAUNCH
  cudaCheck(cudaLaunchKernel(params.func, params.gridDim, params.blockDim, params.kernelParams, params.sharedMemBytes, stream));
#endif
}

template < typename body_type >
__global__
void graph_for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type body)
{
  const IdxT i0 = blockIdx.z;
  const IdxT i1 = threadIdx.y + blockIdx.y * blockDim.y;
  const IdxT i2 = threadIdx.x + blockIdx.x * blockDim.x;
  if (i0 < len0) {
    if (i1 < len1) {
      if (i2 < len2) {
        body(i0, i1, i2);
      }
    }
  }
}

template < typename body_type >
inline void for_all_3d(IdxT len0, IdxT len1, IdxT len2, body_type&& body
#ifdef COMB_GRAPH_KERNEL_LAUNCH
    , cudaStream_t stream
#endif
    )
{
  using decayed_body_type = typename std::decay<body_type>::type;

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
  void* args[]{&len0, &len1, &len2, &body};
  params.kernelParams = args;
  params.extra = nullptr;

  get_active_group().graph->enqueue(params);

#ifdef COMB_GRAPH_KERNEL_LAUNCH
  cudaCheck(cudaLaunchKernel(params.func, params.gridDim, params.blockDim, params.kernelParams, params.sharedMemBytes, stream));
#endif
}

} // namespace graph_launch

} // namespace cuda

#endif

#endif // _GRAPH_LAUNCH_HPP

