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

#ifndef _BATCH_EXEC_CUH
#define _BATCH_EXEC_CUH

// this file has no includes
// it is designed to be included in other files in other namespaces
// this is done to workaround a problem in clangcuda
// where templated global functions create duplicate symbols

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
  using ::detail::device_wrapper_ptr;
  using ::detail::event_complete;
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
    FGPRINTF(FileGroup::proc, "block %6i thread %6i Device Buffer get %p\n", blockIdx.x, threadIdx.x, dev_buf);
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
          FGPRINTF(FileGroup::proc, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
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
                FGPRINTF(FileGroup::proc, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
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
            FGPRINTF(FileGroup::proc, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
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
              FGPRINTF(FileGroup::proc, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
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
              FGPRINTF(FileGroup::proc, "block %6i thread %6i Device Buffer read %p\n", blockIdx.x, threadIdx.x, dev_buf);
            }

            if (threadIdx.x+1 == blockDim.x) {
              FGPRINTF(FileGroup::proc, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
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
  using ::detail::device_wrapper_ptr;
  using ::detail::event_complete;
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
    FGPRINTF(FileGroup::proc, "block %6i thread %6i Device Buffer get %p\n", blockIdx.x, threadIdx.x, dev_buf);
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
          FGPRINTF(FileGroup::proc, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
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
                FGPRINTF(FileGroup::proc, "block %6i thread %6i read fnc %p\n", blockIdx.x, threadIdx.x, fnc);
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
            FGPRINTF(FileGroup::proc, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
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
              FGPRINTF(FileGroup::proc, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
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
              FGPRINTF(FileGroup::proc, "block %6i thread %6i Device Buffer read %p\n", blockIdx.x, threadIdx.x, dev_buf);
            }

            if (threadIdx.x+1 == blockDim.x) {
              FGPRINTF(FileGroup::proc, "block %6i thread %6i grid sync\n", blockIdx.x, threadIdx.x);
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

#endif // _BATCH_EXEC_CUH

