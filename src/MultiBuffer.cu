//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by Jason Burmark, burmark1@llnl.gov
// LLNL-CODE-758885
//
// All rights reserved.
//
// This file is part of Comb.
//
// For details, see https://github.com/LLNL/Comb
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "MultiBuffer.cuh"

namespace detail {

MultiBuffer::MultiBuffer()
  : m_info_first( nullptr )
  , m_info_cur( nullptr )
  , m_info_arr( nullptr )
  , m_buffer(nullptr)
{
  cudaCheck(cudaMallocManaged(&m_buffer, total_capacity, cudaMemAttachGlobal));
  cudaCheck(cudaMemAdvise(m_buffer, total_capacity, cudaMemAdviseSetPreferredLocation, ::detail::cuda::get_device()));
  cudaCheck(cudaMemAdvise(m_buffer, total_capacity, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
  cudaCheck(cudaMemPrefetchAsync(m_buffer, total_capacity, cudaCpuDeviceId, cudaStream_t{0}));
  cudaCheck(cudaMemPrefetchAsync(m_buffer, total_capacity, ::detail::cuda::get_device(), cudaStream_t{0}));


  size_t num_info = total_capacity / buffer_capacity;

  m_info_first = m_info_cur = m_info_arr = new internal_info[num_info];

  for (size_t i = 0; i < num_info; ++i) {
    m_info_arr[i].next = &m_info_arr[(i+1) % num_info];
    m_info_arr[i].buffer_device = &m_buffer[i];
    m_info_arr[i].buffer_device->init();
    m_info_arr[i].buffer_pos = 0;
  };
}

// Do not free cuda memory as this may be destroyed after the cuda context
// is destroyed
MultiBuffer::~MultiBuffer()
{
  delete[] m_info_arr;
  // cudaCheck(cudaFree(m_buffer));
}

} // namespace detail

