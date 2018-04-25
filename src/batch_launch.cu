
#include "batch_launch.cuh"

#include <cooperative_groups.h>

namespace cuda {

namespace batch_launch {

namespace detail {

// Launches a batch kernel and cycles to next buffer
void launch(::detail::MultiBuffer& mb, cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (getMaxN() > 0) {

      ::detail::MultiBuffer::buffer_type* device_buffer = mb.done_packing();

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

      int num_blocks = (getMaxN()+(blocksize-1))/blocksize;
      if (num_blocks > ::detail::cuda::get_num_sm()) {
         num_blocks = ::detail::cuda::get_num_sm();
      }

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
      getMaxN() = 0;
   }
}


} // namespace detail

// Ensure the current batch launched (actually launches batch)
void force_launch(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   if (detail::getMaxN() > 0) {
      detail::launch(detail::getMultiBuffer(), stream);
   }
}

// Wait for all batches to finish running
void synchronize(cudaStream_t stream)
{
   // NVTX_RANGE_COLOR(NVTX_CYAN)
   force_launch(stream);

   // perform synchronization
   cudaCheck(cudaDeviceSynchronize());
}

} // namespace batch_launch

} // namespace cuda

