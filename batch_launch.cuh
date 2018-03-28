
#ifndef _BATCH_LAUNCH_CUH
#define _BATCH_LAUNCH_CUH

#include <type_traits>

#include "utils.cuh"

namespace cuda {

namespace batch_launch {

namespace detail {

using device_wrapper_ptr = const char*(*)(const char*);

// the wrapper function takes a pointer to the kernel body
// and returns a pointer to the next device_wrapper_ptr
template<typename kernel_type>
__device__ const char* device_wrapper_fnc(const char* ptr)
{
   kernel_type* kernel_ptr = (kernel_type*)ptr;
   kernel_ptr->operator()();

   return ptr + ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value;
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
   void operator()()
   {
      // create local copies to avoid avoid changing the canonical version in
      // shared memory.
      kernel_type kernel = m_kernel;
      const int begin = m_begin;
      const int n = m_n;
      const int stride = blockDim.x * gridDim.x;
      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
         kernel(i + begin);
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
   static const size_t buffer_capacity = 4ull*1024ull;

   static MultiBuffer& getInstance()
   {
      static MultiBuffer buf;
      return buf;
   }

   MultiBuffer();

   size_t size()
   {
      return m_info->buffer_pos;
   }

   size_t unused_size()
   {
      return buffer_capacity - m_info->buffer_pos;
   }

   const char* buf()
   {
      return m_info->buffer_host;
   }

   volatile bool* done()
   {
      return m_info->buffer_done;
   }

   volatile char* device_cache()
   {
      return m_device_cache;
   }

   volatile unsigned* device_state()
   {
      return m_device_state;
   }

   int max_n()
   {
      return m_info->max_n;
   }

   int max_n(int val)
   {
      if (val > m_info->max_n) {
         m_info->max_n = val;
      }
      return m_info->max_n;
   }


   void pack(const void* src, size_t nbytes)
   {
      assert(m_info->buffer_pos + nbytes <= buffer_capacity);
      memcpy(&m_info->buffer_host[m_info->buffer_pos], src, nbytes);
      m_info->buffer_pos += nbytes;
   }

   void set_launched(cudaStream_t s)
   {
      m_info->buffer_pos = buffer_capacity;
      m_info->max_n = 0;
      m_info = m_info->next;
   }

   // ensure reads of your buffer are done
   // do so by waiting for the next launch to have started
   void wait_until_writeable()
   {
      // wait for next to be done to ensure your kernel is complete
      if (m_info->buffer_pos == buffer_capacity) {
         if (!m_info->next->buffer_done[0]) {
            while (!m_info->next->buffer_done[0]);
         }
         m_info->buffer_done[0] = false;
         m_info->buffer_pos = 0;
      }
   }

   void inform_synced()
   {
      internal_info* info = m_info;
      do {

         info->buffer_done[0] = false;
         info->buffer_pos = 0;

         info = info->next;
      } while (info != m_info);
   }

   ~MultiBuffer();

   MultiBuffer(MultiBuffer const&) = delete;
   MultiBuffer(MultiBuffer &&) = delete;
   MultiBuffer& operator=(MultiBuffer const&) = delete;
   MultiBuffer& operator=(MultiBuffer &&) = delete;

private:
   struct internal_info {
      // pointer for circularly linked list
      internal_info* next;

      char* buffer_host;
      volatile bool* buffer_done;
      // if buffer_pos == buffer_capacity then launched but not synched
      int buffer_pos;
      // if max_n > 0 work then work enqueued but not launched
      int max_n;
   };

   internal_info* m_info;
   char* m_device_cache;
   unsigned* m_device_state;
   internal_info m_info_arr[8];

   void print_state(const char* func)
   {
      printf("MultiBuffer(%p) %20s\tpos %4i max_n %7i\n", m_info->buffer_host, func, m_info->buffer_pos, m_info->max_n);
   }
};


extern MultiBuffer* mb_instance;


extern void launch(cudaStream_t stream = 0);

// Enqueue a kernel in the current buffer.
template <typename kernel_type_in>
inline void enqueue(int begin, int n, kernel_type_in&& kernel_in, cudaStream_t stream = 0)
{
   using kernel_type = kernel_holder_B_N<typename std::remove_reference<kernel_type_in>::type>;

   size_t enqueue_size = sizeof(device_wrapper_ptr) + ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value;

   if ( mb_instance->unused_size() < enqueue_size + sizeof(device_wrapper_ptr) ) {
      // Not enough room, launch current buffer (switches to next buffer)
      if (mb_instance->max_n() > 0) {
         launch(stream);
      }
      // Wait for next buffer to be writeable
      mb_instance->wait_until_writeable();
   }

   // record n associated with this kernel to get proper launch parameters
   mb_instance->max_n(n);

   // write device wrapper function pointer to pinned buffer
   device_wrapper_ptr wrapper_ptr = get_device_wrapper_ptr<kernel_type>();
   mb_instance->pack(&wrapper_ptr, sizeof(device_wrapper_ptr));

   // Copy kernel into kernel holder and write to pinned buffer
   kernel_type kernel{kernel_in, begin, n};
   mb_instance->pack(&kernel, sizeof(kernel_type));

   // correct alignment by padding if necessary
   size_t size_diff = ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value - sizeof(kernel_type);
   if (size_diff > 0) {
      char arr[size_diff];
      mb_instance->pack(&arr[0], size_diff);
   }
}

} // namespace detail


extern void force_launch(cudaStream_t stream = 0);
extern void synchronize(cudaStream_t stream = 0);

template <typename kernel_type_in>
inline void for_all(int begin, int end, kernel_type_in&& kernel_in, cudaStream_t stream = 0 )
{
   if (begin < end) {
      if (detail::mb_instance == nullptr) {
         detail::mb_instance = new detail::MultiBuffer();
      }
      detail::enqueue(begin, end - begin, std::forward<kernel_type_in>(kernel_in), stream);
   }
}

} // namespace batch_launch

} // namespace cuda

#endif // _BATCH_LAUNCH_CUH

