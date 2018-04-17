
#include "persistent_launch.cuh"

#include <cooperative_groups.h>

namespace cuda {

namespace persistent_launch {

namespace detail {

// Launches a batch kernel and cycles to next buffer
void launch(::detail::MultiBuffer& mb, cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (!getLaunched()) {

      // ensure other thread/device done reading buffer before launch
      while(!mb.cur_buffer_empty());
      // get empty unused buffer
      ::detail::MultiBuffer::buffer_type* device_buffer = mb.get_buffer();

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

      int num_blocks = ::detail::cuda::get_num_sm();

      void* func = NULL;
      void* args[] = { (void*)&device_buffer };

      if (num_blocks < blocks_cutoff) {
         // don't use device cache
         func = (void*)&::detail::block_read_device<::detail::MultiBuffer::shared_buffer_type>;
      } else {
         // use device cache
         func = (void*)&::detail::block_read_device<::detail::MultiBuffer::shared_buffer_type>;
      }
      cudaCheck(cudaLaunchCooperativeKernel(func, num_blocks, blocksize,
                                            args, 0, stream));
      getLaunched() = true;
   }
}

void stop(::detail::MultiBuffer& mb, cudaStream_t stream)
{
   if (getLaunched()) {
     mb.done_packing();
     getLaunched() = false;
   }
}

} // namespace detail

// Start the current batch (launches kernel)
void force_start(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (!detail::getLaunched()) {
      detail::launch(detail::getMultiBuffer(), stream);
   }
}

// Ensure current batch launched (does nothing)
void force_complete(cudaStream_t stream)
{
   if (detail::getLaunched()) {
      detail::stop(detail::getMultiBuffer(), stream);
   }
}

// Wait for all batches to finish running
void synchronize(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   force_complete(stream);

   // perform synchronization
   cudaCheck(cudaDeviceSynchronize());
}

} // namespace persistent_launch

} // namespace cuda

