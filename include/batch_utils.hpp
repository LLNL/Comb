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

#ifndef _BATCH_UTILS_HPP
#define _BATCH_UTILS_HPP

#ifdef __CUDACC__

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <atomic>
#include <type_traits>

#include <cuda.h>
#include <cooperative_groups.h>
#include <mpi.h>

#include "utils.hpp"


inline bool& get_batch_always_grid_sync()
{
  static bool batch_always_grid_sync = true;
  return batch_always_grid_sync;
}

namespace detail {

template < typename T, size_t align>
struct aligned_sizeof {
   static const size_t value = sizeof(T) + ((sizeof(T) % align != 0) ? (align - (sizeof(T) % align)) : 0);
};

using device_wrapper_ptr = const volatile void*(*)(const volatile void*);

using batch_event_type = unsigned long long;
using batch_event_type_ptr = volatile batch_event_type*;

constexpr batch_event_type event_init_val = 0;
constexpr batch_event_type event_complete_val = 1;

#define comb_detail_fnc_null_val  ((device_wrapper_ptr)0)
#define comb_detail_fnc_event_val ((device_wrapper_ptr)1)

template < typename len_type, typename read_type >
struct DynamicBuffer
{
  read_type* buf;
  len_type len;
  DynamicBuffer(len_type len_, read_type* buf_)
   : len(len_)
   , buf(buf_)
  { }
};

template < typename len_type, len_type total_bytes, typename read_type >
struct alignas(std::max_align_t) StaticBuffer
{
  template < typename dyn_read_type >
  using dynamic_buffer_type = DynamicBuffer<len_type, dyn_read_type>;

  static const len_type capacity = (total_bytes - aligned_sizeof<len_type, ((alignof(read_type) > alignof(len_type)) ? alignof(read_type) : alignof(len_type)) >::value ) / sizeof(read_type);
  static const len_type capacity_bytes = capacity * sizeof(read_type);

  read_type buf[capacity];
  len_type len; // in number of read_type that may be read

  StaticBuffer() = default;
  StaticBuffer(StaticBuffer const&) = delete;
  StaticBuffer& operator=(StaticBuffer const&) = delete;

  // initial setup
  void init()
  {
    len = 0;
  }

  // attempt to reinit, fails if reads not done on reading threads/device
  bool reinit() volatile
  {
    // check if len has been reset to 0 by reading thread/device
    if (len == 0) {
      // ensure reads complete and writes not started before reinitializing buffer
      std::atomic_thread_fence(std::memory_order_seq_cst);
      len = 0;
      return true;
    }
    return false;
  }

  // read data from another buffer into this buffer
  // returns true if read new data, false otherwise
  bool read(StaticBuffer const volatile* src)
  {
    len_type old_len = len;
    len_type new_len = src->len;
    if (new_len != old_len) {
      // ensure writes up to new_len visible before reading
      std::atomic_thread_fence(std::memory_order_acquire);
      for (len_type i = old_len; i < new_len; ++i) {
        buf[i] = src->buf[i];
      }
      len = new_len;
    }
    return new_len != old_len;
  }

  // does nothing to this StaticBuffer
  // resets src StaticBuffer to a writeable state
  // programmer must ensure all threads/devices are finished using src before calling this
  void done_reading(StaticBuffer volatile* src)
  {
    src->len = 0;
  }

  template < typename read_type0, typename... read_types >
  len_type write_bytes(dynamic_buffer_type<read_type0> const& src0, dynamic_buffer_type<read_types> const&... srcs)
  {
    return sizeof(read_type)*write_len_helper(src0, srcs...);
  }

  // writes data from multiple dynamic buffers into this buffer
  // returns true if data written, false otherwise
  // All data written will be read by the next successful read from another buffer
  template < typename read_type0, typename... read_types >
  bool write(dynamic_buffer_type<read_type0> const& src0, dynamic_buffer_type<read_types> const&... srcs)
  {
    len_type old_len = len;
    len_type new_len = old_len + write_len_helper(src0, srcs...);
    if (new_len <= capacity) {
      write_helper(old_len, src0, srcs...);
      // ensure recent writes visible before chaning len
      std::atomic_thread_fence(std::memory_order_release);
      len = new_len;
      return true;
    } else {
      return false;
    }
  }
private:
  len_type write_len_helper()
  {
    return 0;
  }
  template < typename read_type0, typename... read_types >
  len_type write_len_helper(dynamic_buffer_type<read_type0> const& src0, dynamic_buffer_type<read_types> const&... srcs)
  {
    len_type write_bytes = sizeof(read_type0)*src0.len;
    len_type write_len   = write_bytes / sizeof(read_type);

    assert(write_len * sizeof(read_type) == write_bytes);

    return write_len + write_len_helper(srcs...);
  }

  void write_helper(len_type)
  {
  }
  template < typename read_type0, typename... read_types >
  void write_helper(len_type cur_len, dynamic_buffer_type<read_type0> const& src0, dynamic_buffer_type<read_types> const&... srcs)
  {
    len_type write_len = write_len_helper(src0);

    read_type*       dst = &buf[cur_len];
    const read_type* src = (const read_type*)&src0.buf[0];

    assert(reinterpret_cast<unsigned long long>(src) % alignof(read_type) == 0);
    assert(reinterpret_cast<unsigned long long>(src) % alignof(read_type0) == 0);
    assert(reinterpret_cast<unsigned long long>(dst) % alignof(read_type0) == 0);

    for (len_type i = 0; i < write_len; ++i) {
      dst[i] = src[i];
    }

    write_helper(cur_len + write_len, srcs...);
  }
};

// must be in shared memory, accessed by whole thread block
template < typename len_type, len_type total_bytes, typename read_type >
struct SharedStaticBuffer : StaticBuffer<len_type, total_bytes, read_type>
{
  using buffer_type = StaticBuffer<len_type, total_bytes, read_type>;

  SharedStaticBuffer() = default;
  SharedStaticBuffer(SharedStaticBuffer const&) = delete;
  SharedStaticBuffer& operator=(SharedStaticBuffer const&) = delete;

  // do first time initialization
  DEVICE
  void init() volatile
  {
    if (threadIdx.x == 0) {
      this->len = 0;
    }
    __syncthreads();
  }

  // reinitialize after finisheding reading from another buffer
  DEVICE
  bool reinit() volatile
  {
    if (threadIdx.x == 0) {
      this->len = 0;
    }
    __syncthreads();
    return true;
  }

  // read data from another buffer into this shared buffer
  // returns true if read new data, false otherwise
  DEVICE
  bool read(buffer_type const volatile* src) volatile
  {
               len_type old_len = this->len;
    __shared__ len_type volatile new_len;
    if (threadIdx.x == 0) {
      new_len = src->len;
    }
    __syncthreads();
    if (new_len != old_len) {
      // ensure writes up to new_len visible before reading (to each thread in block)
      __threadfence_system();
      for(len_type i = threadIdx.x; i < new_len; i += blockDim.x) {
        if (old_len <= i) {
          this->buf[i] = src->buf[i];
        }
      }
      if (threadIdx.x == 0) {
        this->len = new_len;
      }
      __syncthreads();
    }
    return new_len != old_len;
  }

  DEVICE
  const volatile void* buffer() const volatile
  {
    return &this->buf[this->len];
  }

  // does nothing to this SharedStaticBuffer
  // resets src StaticBuffer to a writeable state
  // programmer must ensure all grids/blocks/threads are finished using src before calling this
  DEVICE
  void done_reading(buffer_type volatile* src) volatile
  {
    if (threadIdx.x == 0) {
      src->len = 0;
      // ensure reset len visible to other devices
      __threadfence_system();
    }
  }
};


// the wrapper function takes a pointer to the kernel body
// and returns a pointer to the next device_wrapper_ptr
template<typename kernel_type>
__device__ const volatile void* device_wrapper_fnc(const volatile void* ptr)
{
  const kernel_type* kernel_ptr = (const kernel_type*)ptr;
  kernel_ptr->operator()();

  return (const volatile void*)(((const volatile char*)ptr) + ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value);
}

// cuda global function that writes the device wrapper function pointer
// for the template type to the pointer provided.
template<typename kernel_type>
__global__ void write_device_wrapper_ptr(device_wrapper_ptr* out)
{
   *out = &device_wrapper_fnc<kernel_type>;
}

// Function to allocate a permanent pinned memory buffer
inline device_wrapper_ptr* get_pinned_device_wrapper_ptr_buf()
{
  static device_wrapper_ptr* ptr = nullptr;
  if (ptr == nullptr) {
     cudaCheck(cudaHostAlloc(&ptr, sizeof(device_wrapper_ptr), cudaHostAllocDefault));
  }
  return ptr;
}

// Function that gets and caches the device wrapper function pointer
// for the templated type. A pointer to a device function can only be taken
// in device code, so this launches a kernel to get the pointer. It then holds
// onto the pointer so a kernel doesn't have to be launched to get the
// function pointer the next time.
template<typename kernel_type>
inline device_wrapper_ptr get_device_wrapper_ptr()
{
  static_assert(alignof(kernel_type) <= sizeof(device_wrapper_ptr),
           "kernel_type has excessive alignment requirements");
  static device_wrapper_ptr ptr = nullptr;
  if (ptr == nullptr) {
     device_wrapper_ptr* pinned_buf = get_pinned_device_wrapper_ptr_buf();

     cudaStream_t stream = 0;
     void* func = (void*)&write_device_wrapper_ptr<kernel_type>;
     void* args[] = {&pinned_buf};
     cudaCheck(cudaLaunchKernel(func, 1, 1, args, 0, stream));
     cudaCheck(cudaStreamSynchronize(stream));
     ptr = *pinned_buf;
  }
  return ptr;
}

// A class to hold the lambda body and iteration bounds.
// This class also calls the lambda with appropriate arguments.
template <typename kernel_type>
class kernel_holder_B_N {
public:
  kernel_holder_B_N(kernel_type const& kernel, int begin, int n)
     :  m_kernel(kernel),
        m_begin(begin),
        m_n(n)
  {

  }

  DEVICE
  void operator()() const
  {
     // create local copies to avoid avoid changing the canonical version in
     // shared memory.
     kernel_type kernel = m_kernel;
     const int begin = m_begin;
     const int n = m_n;
     const int stride = blockDim.x * gridDim.x;
     for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
        kernel(i + begin, i);
     }
  }

private:
  kernel_type m_kernel;
  int m_begin;
  int m_n;
};


DEVICE
inline const volatile device_wrapper_ptr* event_complete(const volatile device_wrapper_ptr* buf_ptr)
{
  const volatile batch_event_type_ptr* event_ptr_ptr = (const volatile batch_event_type_ptr*)buf_ptr;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    **event_ptr_ptr = event_complete_val;
    __threadfence_system();
  }
  return (const volatile device_wrapper_ptr*)( event_ptr_ptr + 1 );
}


/*
Implements a kernel that runs functions and events in buffer arg_buf.
Reads arg_buf by creating a shared shared_buffer_type to copy parts of the buffer into shared memory.
Moves through the shared buffer running functions and executing events.
Reads more from arg_buf as necessary, and when complete switches to another, stopping when gets nullptr.
Implements the following diagram.

                    Start      Decide3 ---> Stop
                      |       /    /\
                     \/      \/     |
                    Init  ReInit  BufFinalize
                      |   |         /\
                     \/  \/          |
                    BufRead      GridSync
                     | /\          /\
                    \/  |           |
EventFinish <-----> Decide1 --> EventStart
 /\                      |        /\
  |                     \/         |
GridSync   GridSync -> Compute     |
 /\               /\     |         |
  |                \    \/         |
EventStart <------- Decide2 -------|
                      | /\
                     \/  |
                    BufRead

This diagram represents an algorithm that always
  does a GridSync between Computes
      Compute ...> GridSync ...> Compute
  does an EventStart and GridSync between a Compute and an EventFinish
      Compute ...> EventStart ...> GridSync ...> EventFinish
  does a GridSync between  BufRead and BufFinalize
      BufRead ...> GridSync ...> BufFinalize
  does exactly one GridSync per Compute between Init or Reinit and BufFinalize if one or more Compute
    and does exactly one GridSync between Init or Reinit and BufFinalize if no Compute
      Init|Reinit ...> numberof(GridSync) == max(numberof(Compute), 1) ...> BufFinalize

*/
template < typename shared_buffer_type >
__global__
void block_read_device(typename shared_buffer_type::buffer_type volatile* arg_buf)
{
  // Start:
  __shared__ shared_buffer_type volatile shr_buf;
  __shared__ typename shared_buffer_type::buffer_type volatile* volatile dev_buf;

  // Init:
  if (threadIdx.x == 0) {
    dev_buf = arg_buf;
  }
  shr_buf.init();
  /*
  if (threadIdx.x+1 == blockDim.x) {
    // inform host that device is done reading this block
    FPRINTF(stdout, "block %6i thread %6i Device Buffer get %p\n", blockIdx.x, threadIdx.x, dev_buf);
  }
  */

  pre_loop:

  // Decide3:
  while (dev_buf != nullptr) {

    const volatile void* shr_ptr = shr_buf.buffer();

    device_wrapper_ptr fnc = nullptr;

    goto synced_loop_read;

    // execute functions in read buffer
    while (1) {

      while (shr_ptr != shr_buf.buffer()) {

        fnc = ((const volatile device_wrapper_ptr*)shr_ptr)[0];
        /*
        if (threadIdx.x+1 == blockDim.x) {
          FPRINTF(stdout, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
        }
        */
        // Decide2:
        if (fnc == comb_detail_fnc_event_val) {

          // EventStart: ensure memory writes in all threads readable everywhere
          __threadfence_system();

          // GridSync: synchronize before complete event
          { cooperative_groups::this_grid().sync(); }

          // complete event located at shr_buf
          shr_ptr = event_complete(&((const volatile device_wrapper_ptr*)shr_ptr)[1]);

          while (1) {

            while (shr_ptr != shr_buf.buffer()) {

              fnc = ((const volatile device_wrapper_ptr*)shr_ptr)[0];
              /*
              if (threadIdx.x+1 == blockDim.x) {
                FPRINTF(stdout, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
              }
              */
              // Decide1:
              if (fnc == comb_detail_fnc_event_val) {

                // repeat event, no need to synchronize
                // complete event located at shr_buf
                shr_ptr = event_complete(&((const volatile device_wrapper_ptr*)shr_ptr)[1]);

              } else if (fnc != comb_detail_fnc_null_val) {

                goto do_fnc;

              } else {

                goto finish_buf;

              }

            }

            synced_loop_read:

            // BufRead: read in part of device buffer into shared buffer
            shr_buf.read(dev_buf);

          }

        } else if (fnc != comb_detail_fnc_null_val) {
          /*
          if (threadIdx.x+1 == blockDim.x) {
            FPRINTF(stdout, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
          }
          */
          // GridSync: synchronize before run function
          { cooperative_groups::this_grid().sync(); }

          do_fnc:

          // Compute: run function in this buffer
          shr_ptr = fnc(&((const volatile device_wrapper_ptr*)shr_ptr)[1]);

        } else {

          finish_buf:

          {
            // Reinit: finish of this buffer, get next buffer
            typename shared_buffer_type::buffer_type volatile* old_buf;
            if (threadIdx.x == 0) {
              // use thread 0 to avoid race condition when reading and writing dev_buf
              old_buf = dev_buf;
              dev_buf = (typename shared_buffer_type::buffer_type volatile*)((const volatile device_wrapper_ptr*)shr_ptr)[1];
            }
            /*
            if (threadIdx.x+1 == blockDim.x) {
              FPRINTF(stdout, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
            }
            */
            // EventStart: ensure memory writes in all threads readable everywhere
            __threadfence_system();

            // GridSync: ensure all blocks have read the end of buffer, next buffer
            { cooperative_groups::this_grid().sync(); }

            // BufFinalize:
            if (blockIdx.x == 0) {
              // inform host that device is done reading this block
              shr_buf.done_reading(old_buf);
            }
            // Reinit:
            shr_buf.reinit();
            /*
            if (threadIdx.x+1 == blockDim.x) {
              // inform host that device is done reading this block
              FPRINTF(stdout, "block %6i thread %6i Device Buffer read %p\n", blockIdx.x, threadIdx.x, dev_buf);
            }

            if (threadIdx.x+1 == blockDim.x) {
              FPRINTF(stdout, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
            }

            { cooperative_groups::this_grid().sync(); }
            */
          }

          goto pre_loop;

        }

      }

      // BufRead: read in part of device buffer into shared buffer
      shr_buf.read(dev_buf);

    }

  }

  // Stop:
}
///
template < typename shared_buffer_type >
__global__
void block_read_device_few_grid_sync(typename shared_buffer_type::buffer_type volatile* arg_buf)
{
  // Start:
  __shared__ shared_buffer_type volatile shr_buf;
  __shared__ typename shared_buffer_type::buffer_type volatile* volatile dev_buf;

  // Init:
  if (threadIdx.x == 0) {
    dev_buf = arg_buf;
  }
  shr_buf.init();
  /*
  if (threadIdx.x+1 == blockDim.x) {
    // inform host that device is done reading this block
    FPRINTF(stdout, "block %6i thread %6i Device Buffer get %p\n", blockIdx.x, threadIdx.x, dev_buf);
  }
  */

  pre_loop:

  // Decide3:
  while (dev_buf != nullptr) {

    const volatile void* shr_ptr = shr_buf.buffer();

    device_wrapper_ptr fnc = nullptr;

    goto synced_loop_read;

    // execute functions in read buffer
    while (1) {

      while (shr_ptr != shr_buf.buffer()) {

        fnc = ((const volatile device_wrapper_ptr*)shr_ptr)[0];
        /*
        if (threadIdx.x+1 == blockDim.x) {
          FPRINTF(stdout, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
        }
        */
        // Decide2:
        if (fnc == comb_detail_fnc_event_val) {

          // EventStart: ensure memory writes in all threads readable everywhere
          __threadfence_system();

          // GridSync: synchronize before complete event
          { cooperative_groups::this_grid().sync(); }

          // complete event located at shr_buf
          shr_ptr = event_complete(&((const volatile device_wrapper_ptr*)shr_ptr)[1]);

          while (1) {

            while (shr_ptr != shr_buf.buffer()) {

              fnc = ((const volatile device_wrapper_ptr*)shr_ptr)[0];
              /*
              if (threadIdx.x+1 == blockDim.x) {
                FPRINTF(stdout, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
              }
              */
              // Decide1:
              if (fnc == comb_detail_fnc_event_val) {

                // repeat event, no need to synchronize
                // complete event located at shr_buf
                shr_ptr = event_complete(&((const volatile device_wrapper_ptr*)shr_ptr)[1]);

              } else if (fnc != comb_detail_fnc_null_val) {

                goto do_fnc;

              } else {

                goto finish_buf;

              }

            }

            synced_loop_read:

            // BufRead: read in part of device buffer into shared buffer
            shr_buf.read(dev_buf);

          }

        } else if (fnc != comb_detail_fnc_null_val) {
          /*
          if (threadIdx.x+1 == blockDim.x) {
            FPRINTF(stdout, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
          }
          */
          // GridSync: synchronize before run function
          // { cooperative_groups::this_grid().sync(); }

          do_fnc:

          // Compute: run function in this buffer
          shr_ptr = fnc(&((const volatile device_wrapper_ptr*)shr_ptr)[1]);

        } else {

          finish_buf:

          {
            // Reinit: finish of this buffer, get next buffer
            typename shared_buffer_type::buffer_type volatile* old_buf;
            if (threadIdx.x == 0) {
              // use thread 0 to avoid race condition when reading and writing dev_buf
              old_buf = dev_buf;
              dev_buf = (typename shared_buffer_type::buffer_type volatile*)((const volatile device_wrapper_ptr*)shr_ptr)[1];
            }
            /*
            if (threadIdx.x+1 == blockDim.x) {
              FPRINTF(stdout, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
            }
            */
            // EventStart: ensure memory writes in all threads readable everywhere
            __threadfence_system();

            // GridSync: ensure all blocks have read the end of buffer, next buffer
            { cooperative_groups::this_grid().sync(); }

            // BufFinalize:
            if (blockIdx.x == 0) {
              // inform host that device is done reading this block
              shr_buf.done_reading(old_buf);
            }
            // Reinit:
            shr_buf.reinit();
            /*
            if (threadIdx.x+1 == blockDim.x) {
              // inform host that device is done reading this block
              FPRINTF(stdout, "block %6i thread %6i Device Buffer read %p\n", blockIdx.x, threadIdx.x, dev_buf);
            }

            if (threadIdx.x+1 == blockDim.x) {
              FPRINTF(stdout, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
            }

            { cooperative_groups::this_grid().sync(); }
            */
          }

          goto pre_loop;

        }

      }

      // BufRead: read in part of device buffer into shared buffer
      shr_buf.read(dev_buf);

    }

  }

  // Stop:
}

} // namespace detail

#endif

#endif // _BATCH_UTILS_HPP

