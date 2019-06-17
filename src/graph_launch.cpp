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

#include "config.hpp"

#ifdef COMB_ENABLE_CUDA_GRAPH

#include "graph_launch.hpp"

namespace cuda {

namespace graph_launch {

namespace detail {

// Launches a batch kernel and cycles to next buffer
void launch(Graph& graph, cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   graph.launch(stream);
}

} // namespace detail

// Ensure the current batch launched (actually launches batch)
void force_launch(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (!detail::getGraphs().empty() && !detail::getGraphs().front().launched) {
      detail::launch(detail::getGraphs().front(), stream);
   }
}

// Wait for all batches to finish running
void synchronize(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   force_launch(stream);

   // perform synchronization
   cudaCheck(cudaStreamSynchronize(stream));
}

} // namespace graph_launch

} // namespace cuda

#endif
