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

#ifndef _POL_CUDA_GRAPH_HPP
#define _POL_CUDA_GRAPH_HPP

#ifdef COMB_ENABLE_CUDA_GRAPH
#include <cuda.h>

struct cuda_graph_pol {
  static const bool async = true;
  static const char* get_name() { return "cudaGraph"; }
  using event_type = cudaEvent_t;
};

template < >
struct ExecContext<cuda_graph_pol>
{
  cudaStream_t stream = 0;
};

inline bool operator==(ExecContext<cuda_graph_pol> const& lhs, ExecContext<cuda_graph_pol> const& rhs)
{
  return lhs.stream == rhs.stream;
}

inline void synchronize(ExecContext<cuda_graph_pol> const& con)
{
  // called after batch_launch
  // ensure previous work complete
  cudaCheck(cudaStreamSynchronize(con.stream));
}

inline void persistent_launch(ExecContext<cuda_graph_pol> const&)
{
  // called before anything launched
  // start a graph
}

inline void batch_launch(ExecContext<cuda_graph_pol> const&)
{
  // called after some things are launched/enqueued
  // finish and launch graph
}

inline void persistent_stop(ExecContext<cuda_graph_pol> const&)
{
  // called after everything launched/enqueued
  // cleanup
}

inline typename cuda_graph_pol::event_type createEvent(ExecContext<cuda_graph_pol> const&)
{
  cudaEvent_t event;
  cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  return event;
}

inline void recordEvent(ExecContext<cuda_graph_pol> const& con, typename cuda_graph_pol::event_type event)
{
  cudaCheck(cudaEventRecord(event, con.stream));
}

inline bool queryEvent(ExecContext<cuda_graph_pol> const&, typename cuda_graph_pol::event_type event)
{
  return cudaCheckReady(cudaEventQuery(event));
}

inline void waitEvent(ExecContext<cuda_graph_pol> const&, typename cuda_graph_pol::event_type event)
{
  cudaCheck(cudaEventSynchronize(event));
}

inline void destroyEvent(ExecContext<cuda_graph_pol> const&, typename cuda_graph_pol::event_type event)
{
  cudaCheck(cudaEventDestroy(event));
}

template < typename body_type >
inline void for_all(ExecContext<cuda_graph_pol> const& con, IdxT begin, IdxT end, body_type&& body)
{
  COMB::ignore_unused(con);
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len = end - begin;

  const IdxT threads = 256;
  const IdxT blocks = (len + threads - 1) / threads;

  void* func = (void*)&cuda_for_all<decayed_body_type>;
  dim3 gridDim(blocks);
  dim3 blockDim(threads);
  void* args[]{&begin, &len, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = con.stream;

  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
  //synchronize(con);
}

template < typename body_type >
inline void for_all_2d(ExecContext<cuda_graph_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, body_type&& body)
{
  COMB::ignore_unused(con);
  using decayed_body_type = typename std::decay<body_type>::type;

  IdxT len0 = end0 - begin0;
  IdxT len1 = end1 - begin1;

  const IdxT threads0 = 8;
  const IdxT threads1 = 32;
  const IdxT blocks0 = (len0 + threads0 - 1) / threads0;
  const IdxT blocks1 = (len1 + threads1 - 1) / threads1;

  void* func = (void*)&cuda_for_all_2d<decayed_body_type>;
  dim3 gridDim(blocks1, blocks0, 1);
  dim3 blockDim(threads1, threads0, 1);
  void* args[]{&begin0, &len0, &begin1, &len1, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = con.stream;

  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
  //synchronize(con);
}

template < typename body_type >
inline void for_all_3d(ExecContext<cuda_graph_pol> const& con, IdxT begin0, IdxT end0, IdxT begin1, IdxT end1, IdxT begin2, IdxT end2, body_type&& body)
{
  COMB::ignore_unused(con);
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

  void* func =(void*)&cuda_for_all_3d<decayed_body_type>;
  dim3 gridDim(blocks2, blocks1, blocks0);
  dim3 blockDim(threads2, threads1, threads0);
  void* args[]{&begin0, &len0, &begin1, &len1, &begin2, &len2, &len12, &body};
  size_t sharedMem = 0;
  cudaStream_t stream = con.stream;

  cudaCheck(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
  //synchronize(con);
}

#endif // COMB_ENABLE_CUDA_GRAPH

#endif // _POL_CUDA_GRAPH_HPP
