
#include "persistent_launch.cuh"

#include <cooperative_groups.h>

namespace cuda {

namespace persistent_launch {

namespace detail {


MultiBuffer::MultiBuffer()
   :  m_info( &m_info_arr[0] ),
      m_device_cache( nullptr ),
      m_device_state( nullptr ),
      m_info_arr{ {&m_info_arr[1], nullptr, nullptr, 0, 0},
                  {&m_info_arr[2], nullptr, nullptr, 0, 0},
                  {&m_info_arr[3], nullptr, nullptr, 0, 0},
                  {&m_info_arr[4], nullptr, nullptr, 0, 0},
                  {&m_info_arr[5], nullptr, nullptr, 0, 0},
                  {&m_info_arr[6], nullptr, nullptr, 0, 0},
                  {&m_info_arr[7], nullptr, nullptr, 0, 0},
                  {&m_info_arr[0], nullptr, nullptr, 0, 0} }
{
   cudaCheck(cudaMalloc(&m_device_cache, buffer_capacity + sizeof(unsigned)));
   cudaCheck(cudaMemset(m_device_cache, 0, buffer_capacity + sizeof(unsigned)));
   m_device_state = (unsigned*)&m_device_cache[buffer_capacity];

   internal_info* info = m_info;
   do {

      cudaCheck(cudaHostAlloc(&info->buffer_host, buffer_capacity + sizeof(bool), cudaHostAllocDefault));
      memset(info->buffer_host, 0, buffer_capacity + sizeof(bool));
      info->buffer_done = (bool*)&info->buffer_host[buffer_capacity];

      info = info->next;
   } while (info != m_info);
}

// Do not free cuda memory as this may be destroyed after the cuda context
// is destroyed
MultiBuffer::~MultiBuffer()
{
   // internal_info* info = m_info;
   // do {

   //    cudaCheck(cudaFreeHost(info->buffer_host));

   //    info = info->next;
   // } while (info != m_info);

   // cudaCheck(cudaFree(m_device_cache));
}


struct device_arg_wrapper {
   const char* pinned_ptr;
   // volatile bool* pinned_done; // = &pinned_ptr[MultiBuffer::buffer_capacity];
   int size;
};

__device__ volatile unsigned device_state = 0;
__device__ volatile char device_cache[MultiBuffer::buffer_capacity];


// the global function takes a pointer to a constant memory buffer
// the buffer consists of (device_wrapper_ptr, kernel),
//                        (device_wrapper_ptr, kernel),
//                         ... ,
//                        nullptr
// the global function's responsibility is to iterate through the buffer until
// it finds the nullptr calling the wrapper functions it finds along the way.
__global__
void all_blocks_read_pinned(const device_arg_wrapper arg)
{
   __shared__ char buf[MultiBuffer::buffer_capacity];

   // Read pinned memory into shared memory buffer
   for ( int i = threadIdx.x; i < arg.size; i+=blockDim.x)
      buf[i] = arg.pinned_ptr[i];
   __syncthreads();

   if (threadIdx.x == 0) {
      // signal reads complete to host
      ((volatile bool*)&arg.pinned_ptr[MultiBuffer::buffer_capacity])[0] = true;
      __threadfence_system();
   }

   const char* ptr = &buf[0];

   device_wrapper_ptr fnc = *((device_wrapper_ptr*)ptr);

   if (fnc == nullptr) return;

   while(1) {

      ptr = fnc(ptr + sizeof(device_wrapper_ptr));

      fnc = *((device_wrapper_ptr*)ptr);

      if (fnc == nullptr) {
         break;
      }

      cooperative_groups::this_grid().sync();

   }
}

// the global function takes a pointer to a constant memory buffer
// the buffer consists of (device_wrapper_ptr, kernel),
//                        (device_wrapper_ptr, kernel),
//                         ... ,
//                        nullptr
// the global function's responsibility is to iterate through the buffer until
// it finds the nullptr calling the wrapper functions it finds along the way.
// The first block reads the data from pinned memory and stores it in a
// device global memory buffer.
// Other blocks spin on device_state until it becomes ready.
// device_state  0 uninitialized
//               1 being populated
//            >= 2 ready
__global__
void first_block_read_pinned(const device_arg_wrapper arg)
{
   __shared__ char buf[MultiBuffer::buffer_capacity];

   // first block copies pinned memory into device memory buffer
   // Other blocks wait, then read device memory buffer
   {
      int state = 1;

      if (threadIdx.x == 0) {
         // read state of device cache, changing it to 1 if it was 0
         state = atomicCAS((unsigned*)&device_state, 0, 1);
         if (state == 1) {
            // wait until the buffer is ready
            while (device_state == 1);
         }
      }

      // one block will have a thread with state == 0
      state = __syncthreads_or(state == 0);

      if (state) {
         // one block initializes the device cache
         for ( int i = threadIdx.x; i < arg.size; i+=blockDim.x) {
            device_cache[i] = buf[i] = arg.pinned_ptr[i];
         }
         // ensure initialization of device cache visible to other blocks
         __threadfence();
         __syncthreads();
         if (threadIdx.x == 0) {
            // signal device cache ready (increment state to 2)
            atomicInc((unsigned*)&device_state, gridDim.x);
            // signal reads complete to host
            ((volatile bool*)&arg.pinned_ptr[MultiBuffer::buffer_capacity])[0] = true;
            __threadfence_system();
         }
      } else {
         // other blocks read from the device cache
         __threadfence();
         for ( int i = threadIdx.x; i < arg.size; i+=blockDim.x) {
            buf[i] = device_cache[i];
         }
         __syncthreads();
         if (threadIdx.x == 0) {
            // ensure state reset to 0
            atomicInc((unsigned*)&device_state, gridDim.x);
         }
      }

   }

   const char* ptr = &buf[0];

   device_wrapper_ptr fnc = *((device_wrapper_ptr*)ptr);

   if (fnc != nullptr) {

      while(1) {

         ptr = fnc(ptr + sizeof(device_wrapper_ptr));

         fnc = *((device_wrapper_ptr*)ptr);

         if (fnc == nullptr) {
            break;
         }

         cooperative_groups::this_grid().sync();

      }
   }

}

MultiBuffer* mb_instance = nullptr;


// Launches a batch kernel and cycles to next buffer
void launch(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (mb_instance->max_n() > 0) {

      // write nullptr into current buffer
      assert( mb_instance->unused_size() >= sizeof(device_wrapper_ptr) );

      device_wrapper_ptr wrapper_ptr = nullptr;
      mb_instance->pack(&wrapper_ptr, sizeof(device_wrapper_ptr));

      int blocks_cutoff = 1;
      // TODO decide cutoff below which all blocks read pinned memory directly
      // if (::detail::cuda::get_cuda_arch() >= 600) {
      //    // pascal or newer
      //    blocks_cutoff = 1;
      // } else {
      //    blocks_cutoff = 1;
      // }

      int blocksize = 1024;
      // TODO decide blocksize in a smart way

      int num_blocks = (mb_instance->max_n()+(blocksize-1))/blocksize;
      if (num_blocks > ::detail::cuda::get_num_sm()) {
         num_blocks = ::detail::cuda::get_num_sm();
      }

      // wrap args in a struct to avoid multiple implicit calls to cudaSetupArgument
      device_arg_wrapper arg {
               mb_instance->buf(),
               (int)mb_instance->size() };

      void* args[] = { (void*)&arg };

      void* func = NULL;

      if (num_blocks < blocks_cutoff) {
         // don't use device cache
         // all_blocks_read_pinned<<<num_blocks,blocksize,mb_instance->size(),stream>>>
         //       ( arg );

         func = (void*)&all_blocks_read_pinned;
      } else {
         // use device cache
         // first_block_read_pinned<<<num_blocks,blocksize,mb_instance->size(),stream>>>
         //       ( arg );

         func = (void*)&first_block_read_pinned;
      }
      cudaCheck(cudaLaunchCooperativeKernel(func, num_blocks, blocksize,
                                            args, 0, stream));

      // inform multibuffer of launch (switch to next buffer)
      mb_instance->set_launched(stream);
   }
}


} // namespace detail

// Launch the current batch
void force_launch(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (detail::mb_instance != nullptr) {
      if (detail::mb_instance->max_n() > 0) {
         detail::launch(stream);
      }
   }
}

// Wait for all batches to finish running
void synchronize(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (detail::mb_instance != nullptr) {
      if (detail::mb_instance->max_n() > 0) {
         detail::launch(stream);
      }

      // perform synchronization
      cudaCheck(cudaDeviceSynchronize());
      detail::mb_instance->inform_synced();
   }
}

} // namespace persistent_launch

} // namespace cuda

