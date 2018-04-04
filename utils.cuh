
#ifndef _UTILS_CUH
#define _UTILS_CUH

#include <cassert>
#include <atomic>
#include <type_traits>

#include <cuda.h>
#include <cooperative_groups.h>
#include <mpi.h>

#define HOST __host__
#define DEVICE __device__

#define cudaCheck(...) \
  if (__VA_ARGS__ != cudaSuccess) { \
    fprintf(stderr, "Error performing " #__VA_ARGS__ " %s:%i\n", __FILE__, __LINE__); fflush(stderr); \
    /* assert(0); */ \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  }

namespace detail {

namespace cuda {

inline int get_device() {
  static int d = -1;
  if (d == -1) {
    cudaCheck(cudaGetDevice(&d));
  }
  return d;
}

inline int get_num_sm() {
   static int num_sm = -1;
   if (num_sm == -1) {
      cudaCheck(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, get_device()));
   }
   return num_sm;
}


inline int get_arch() {
   static int cuda_arch = -1;
   if (cuda_arch == -1) {
      int major, minor;
      cudaCheck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, get_device()));
      cudaCheck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, get_device()));
      cuda_arch = 100*major + 10*minor;
   }
   return cuda_arch;
}

} // namespace cuda

template < typename T, size_t align>
struct aligned_sizeof {
   static const size_t value = sizeof(T) + ((sizeof(T) % align != 0) ? (align - (sizeof(T) % align)) : 0);
};

template < typename T, typename ... types >
struct Count;

template < typename T >
struct Count<T> {
  static const size_t value = 0;
};

template < typename T, typename ... types >
struct Count<T, T, types...> {
  static const size_t value = 1 + Count<T, types...>::value;
};

template < typename T, typename T0, typename ... types >
struct Count<T, T0, types...> {
  static const size_t value = Count<T, types...>::value;
};

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
struct StaticBuffer
{
  template < typename dyn_read_type >
  using dynamic_buffer_type = DynamicBuffer<len_type, dyn_read_type>;
  constexpr len_type capacity = (total_bytes - aligned_sizeof<sizeof(len_type), ((alignof(read_type) > alignof(len_type)) ? alignof(read_type) : alignof(len_type))>) / sizeof(read_type);
  constexpr len_type capacity_bytes = capacity * sizeof(read_type);
  static_assert(sizeof(StaticBuffer) == total_bytes, "Static Buffer of total_bytes could not be made");
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
      // ensure reads complete before reinitializing buffer
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
  template < >
  len_type write_len_helper()
  {
    return 0;
  }
  template < typename... read_type0, typename... read_types >
  len_type write_len_helper(dynamic_buffer_type<read_type0> const& src0, dynamic_buffer_type<read_types> const&... srcs)
  {
    len_type write_bytes = sizeof(read_type0)*src0.len;
    len_type write_len   = write_bytes / sizeof(read_type);
    
    assert(write_len * sizeof(read_type) == write_bytes);
    
    return write_len + write_len_helper(srcs...);
  }

  template < >
  void write_helper(len_type)
  {
  }
  template < typename... read_type0, typename... read_types >
  void write_helper(len_type cur_len, dynamic_buffer_type<read_type0> const& src0, dynamic_buffer_type<read_types> const&... srcs)
  {
    len_type write_len = write_len_helper(src0);
    
    read_type*       dst = &buf[cur_len];
    const read_type* src = (const read_type*)&src0.buf[0];
    
    assert(reinterpret_cast<unsigned long long>(src) % alignof(read_type) == 0);
    
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
  SharedStaticBuffer(StaticBuffer const&) = delete;
  SharedStaticBuffer& operator=(StaticBuffer const&) = delete;
  
  // do first time initialization
  DEVICE
  void init()
  {
    if (cooperative_groups::this_thread_block().thread_rank() == 0) {
      len = 0;
    }
  }
  
  // reinitialize after finisheding reading from another buffer
  DEVICE
  bool reinit()
  {
    if (cooperative_groups::this_thread_block().thread_rank() == 0) {
      len = 0;
    }
    return true;
  }
  
  // read data from another buffer into this shared buffer
  // returns true if read new data, false otherwise
  DEVICE
  bool read(buffer_type const volatile* src)
  {
               len_type old_len = len;
    __shared__ len_type new_len;
    if (cooperative_groups::this_thread_block().thread_rank() == 0) {
      new_len = src->len;
    }
    cooperative_groups::this_thread_block().sync();
    if (new_len != old_len) {
      // ensure writes up to new_len visible before reading (to each thread in block)
      __thread_fence_system();
      for(len_type i = cooperative_groups::this_thread_block().thread_rank(); i < new_len; i += cooperative_groups::this_thread_block().size()) {
        if (old_len <= i) {
          buf[i] = src->buf[i];
        }
      }
      if (cooperative_groups::this_thread_block().thread_rank() == 0) {
        len = new_len;
      }
      cooperative_groups::this_thread_block().sync();
    }
    return new_len != old_len;
  }
  
  const void* buffer() const
  {
    return &buf[len];
  }
  
  // does nothing to this SharedStaticBuffer
  // resets src StaticBuffer to a writeable state
  // programmer must ensure all grids/blocks/threads are finished using src before calling this
  DEVICE
  void done_reading(buffer_type volatile* src)
  {
    if (cooperative_groups::this_thread_block().thread_rank() == 0) {
      src->len = 0;
      // ensure reset len visible to other devices
      __thread_fence_system();
    }
  }
};


using device_wrapper_ptr = const char*(*)(const char*);

// the wrapper function takes a pointer to the kernel body
// and returns a pointer to the next device_wrapper_ptr
template<typename kernel_type>
__device__ const void* device_wrapper_fnc(const void* ptr)
{
   const kernel_type* kernel_ptr = (const kernel_type*)ptr;
   kernel_ptr->operator()();

   return (const void*)(((const char*)ptr) + ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value);
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

   __device__
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


// This class handles writing a circularly linked list of pinned buffers.
// Multiple buffers allows multiple kernels to be enqueued on the device
// at the same time.
// Each buffer may be reused once it's associated batch kernel has completed.
// This is achieved by waiting on a flag set by the batch kernel associated
// with the next buffer be set.
class MultiBuffer {
public:
   static const size_t total_capacity = (64ull*1024ull);
   static const size_t buffer_capacity = (4ull*1024ull);
   
   using buffer_read_type = int;
   using buffer_size_type = int;
   using shared_buffer_type = SharedStaticBuffer<buffer_size_type, buffer_capacity, buffer_read_type>;
   using buffer_type = shared_buffer_type::buffer_type;
   
   static const size_t useable_capacity = buffer_type::capacity_bytes - 2*sizeof(device_wrapper_ptr);

   MultiBuffer();

   // pack a function pointer, associated arguments
   // returns false if unable to pack
   template < typename kernel_type >
   bool pack(device_wrapper_ptr fnc, kernel_type* data)
   {
      static_assert(sizeof(device_wrapper_ptr) + ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value <= useable_capacity, "Type could never fit into useable capacity");
      
      buffer_size_type nbytes = m_info_cur->buffer_device.write_bytes( buffer_type::dynamic_buffer_type<device_wrapper_ptr>(1, &fnc)
                                                                     , buffer_type::dynamic_buffer_type<kernel_type>(1, data) );
      
      buffer_size_type aligned_nbytes = nbytes
      if (aligned_nbytes % sizeof(device_wrapper_ptr) != 0) {
         aligned_nbytes += sizeof(device_wrapper_ptr) - (aligned_nbytes % sizeof(device_wrapper_ptr));
      }
      assert(aligned_nbytes <= useable_capacity);
      
      if (!cur_buffer_writeable()) {
         return false;
      }
      
      if (m_info_cur->buffer_pos + aligned_nbytes <= useable_capacity) {
      
         m_info_cur->buffer_pos += aligned_nbytes;
         char extra_bytes[sizeof(device_wrapper_ptr)];
         bool success = m_info_cur->buffer_device->write( buffer_type::dynamic_buffer_type<device_wrapper_ptr>(1, &fnc)
                                                        , buffer_type::dynamic_buffer_type<kernel_type>(1, data)
                                                        , buffer_type::dynamic_buffer_type<char>(aligned_nbytes - nbytes, &extra_bytes[0]) );
         assert(success);
         return true;
      } else if (next_buffer_empty()) {
      
         continue_and_next_buffer();
         
         m_info_cur->buffer_pos += aligned_nbytes;
         char extra_bytes[sizeof(device_wrapper_ptr)];
         bool success = m_info_cur->buffer_device->write( buffer_type::dynamic_buffer_type<device_wrapper_ptr>(1, &fnc)
                                                        , buffer_type::dynamic_buffer_type<kernel_type>(1, data)
                                                        , buffer_type::dynamic_buffer_type<char>(aligned_nbytes - nbytes, &extra_bytes[0]) );
         assert(success);
         return true;
      }
      return false;
   }

   buffer_type* done_packing()
   {
      return stop_and_next_buffer();
   }
   
   buffer_type* get_buffer()
   {
      return m_info_first->buffer_device;
   }
   
   bool cur_buffer_empty()
   {
      if (m_info_cur->buffer_pos == 0) {
         return true;
      } else if (m_info_cur->next->buffer_pos == buffer_capacity) {
         if (m_info_cur->buffer_device->reinit()) {
           m_info_cur->buffer_pos = 0;
           return true;
         }
      }
      return false;
   }

   ~MultiBuffer();

   MultiBuffer(MultiBuffer const&) = delete;
   MultiBuffer(MultiBuffer &&) = delete;
   MultiBuffer& operator=(MultiBuffer const&) = delete;
   MultiBuffer& operator=(MultiBuffer &&) = delete;

private:
   bool cur_buffer_writeable()
   {
      if (m_info_cur->buffer_pos <= useable_capacity) {
         return true;
      } else if (m_info_cur->buffer_pos == buffer_capacity) {
         if (m_info_cur->buffer_device->reinit()) {
           m_info_cur->buffer_pos = 0;
           return true;
         }
      }
      return false;
   }
   
   bool next_buffer_empty()
   {
      if (m_info_cur->next->buffer_pos == 0) {
         return true;
      } else if (m_info_cur->next->buffer_pos == buffer_capacity) {
         if (m_info_cur->next->buffer_device->reinit()) {
           m_info_cur->next->buffer_pos = 0;
           return success;
         }
      }
      return false;
   }
   
   buffer_type* stop_and_next_buffer()
   {
      void* ptrs[2] {nullptr, nullptr};
      bool wrote = m_info_cur->buffer_device->write( buffer_type::dynamic_buffer_type<void*>(2, &ptrs[0]) );
      assert(wrote);
      m_info_cur = m_info_cur->next;
      buffer_type* buffer = get_buffer();
      m_info_first = m_info_cur;
      return buffer;
   }
   
   void continue_and_next_buffer()
   {
      void* ptrs[2] {nullptr, (void*)m_info_cur->next->buffer_device};
      bool wrote = m_info_cur->buffer_device->write( buffer_type::dynamic_buffer_type<void*>(2, &ptrs[0]) );
      assert(wrote);
      m_info_cur = m_info_cur->next;
   }

   struct internal_info {
      // pointer for circularly linked list
      internal_info* next;

      buffer_type* buffer_device;
      // if buffer_pos == 0 then unused and synched
      // if 0 < buffer_pos <= useable_capacity then used, still cur, and not synched
      // if buffer_pos == buffer_capacity then used, not cur, and not synched
      int buffer_pos;
   };

   // current buffer
   internal_info* m_info_first;
   internal_info* m_info_cur;
   
   internal_info* m_info_arr;
   void* m_buffer;
   
   void print_state(const char* func)
   {
      // printf("MultiBuffer(%p) %20s\tpos %4i max_n %7i\n", m_info->buffer_host, func, m_info->buffer_pos);
   }
};


MultiBuffer::MultiBuffer()
  : m_info( nullptr )
  , m_info_arr( nullptr )
  , m_buffer(nullptr)
{
  cudaCheck(cudaMallocManaged(&m_buffer, total_capacity, cudaMemAttachGlobal));
  cudaCheck(cudaMemAdvise(m_buffer, total_capacity, cudaMemAdviseSetPreferredLocation, ::detail::cuda::get_device()));
  cudaCheck(cudaMemAdvise(m_buffer, total_capacity, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
  cudaCheck(cudaMemPrefetchAsync(m_buffer, total_capacity, ::detail::cuda::get_device(), cudaCpuDeviceId));
  cudaCheck(cudaMemPrefetchAsync(m_buffer, total_capacity, ::detail::cuda::get_device(), cudaStream_t{0}));
  
  size_t num_info = total_capacity / buffer_capacity;
  
  m_info_first = m_info_cur = m_info_arr = new internal_info[num_info];
  
  for (size_t i = 0; i < num_info; ++i) {
    m_info_arr[i].next = &info[(i+1) % num_info];
    m_info_arr[i].buffer_device = &static_cast<char*>(m_buffer)[i * buffer_capacity];
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


template < typename shared_buffer_type >
__global__
void block_read_device(shared_buffer_type::buffer_type volatile* arg_buf)
{
   __shared__ shared_buffer_type shr_buf;
   __shared__ shared_buffer_type::buffer_type volatile* dev_buf;
   
   shr_buf.init();
   
   if (cooperative_groups::this_thread_block().thread_rank() == 0) {
     dev_buf = arg_buf;
   }
   cooperative_groups::this_thread_block().sync();
   
   pre_loop:
   
   while (dev_buf != nullptr) {
     
     const void* shr_ptr = shr_buf.buffer();
     
     // loop until get first 
     while (1) {
     
       // read in part of device buffer into shared buffer
       shr_buf.read(dev_buf);
       
       if (shr_ptr != shr_buf.buffer()) {
         
         device_wrapper_ptr fnc = ((const device_wrapper_ptr*)shr_ptr)[0];
       
         if (fnc != nullptr) {
         
           // run function in this buffer
           shr_ptr = fnc(&((const device_wrapper_ptr*)shr_ptr)[1]);
           break;
           
         } else {
           // end of this buffer, get next buffer
         
           // ensure all blocks have read the end of buffer, next buffer
           cooperative_groups::this_grid().sync();
       
           if (cooperative_groups::this_thread_block().group_index().x == 0) {
             // inform host that device is done reading this block
             shr_buf.done_reading(dev_buf);
           }
           dev_buf = ((shared_buffer_type::buffer_type volatile**)&((const device_wrapper_ptr*)shr_ptr)[1])[0];
           shr_buf.reinit();
           
           goto pre_loop;
           
         }
       }
       
     }
   
     // execute functions in read buffer
     while (1) {
     
       // read in part of device buffer into shared buffer
       shr_buf.read(dev_buf);
     
       if (shr_ptr != shr_buf.buffer()) {
       
         device_wrapper_ptr fnc = ((const device_wrapper_ptr*)shr_ptr)[0];
       
         if (fnc != nullptr) {
           // synchronize before run function
           cooperative_groups::this_grid().sync();
         
           // run function in this buffer
           shr_ptr = fnc(&((const device_wrapper_ptr*)shr_ptr)[1]);
           
         } else {
           // end of this buffer, get next buffer
         
           // ensure all blocks have read the end of buffer, next buffer
           cooperative_groups::this_grid().sync();
       
           if (cooperative_groups::this_thread_block().group_index().x == 0) {
             // inform host that device is done reading this block
             shr_buf.done_reading(dev_buf);
           }
           dev_buf = ((shared_buffer_type::buffer_type volatile**)&((const device_wrapper_ptr*)shr_ptr)[1])[0];
           shr_buf.reinit();
           
           goto pre_loop;
         
         }
       
       }

     }
     
   }
}

} // namespace detail

#endif // _UTILS_CUH

