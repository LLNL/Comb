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

#ifndef _MULTIBUFFER_CUH
#define _MULTIBUFFER_CUH

#include <atomic>

#include "batch_utils.hpp"

namespace detail {

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

   static_assert(sizeof(shared_buffer_type) == buffer_capacity, "Shared Static Buffer size not buffer_capacity");
   static_assert(sizeof(buffer_type) == buffer_capacity, "Static Buffer size not buffer_capacity");

   static const size_t useable_capacity = buffer_type::capacity_bytes - 2*sizeof(device_wrapper_ptr);

   MultiBuffer();

   // pack a function pointer, associated arguments
   // returns false if unable to pack
   template < typename kernel_type >
   bool pack(device_wrapper_ptr fnc, kernel_type* data)
   {
      static_assert(sizeof(device_wrapper_ptr) + ::detail::aligned_sizeof<kernel_type, sizeof(device_wrapper_ptr)>::value <= useable_capacity, "Type could never fit into useable capacity");

      buffer_size_type nbytes = m_info_cur->buffer_device->write_bytes( buffer_type::dynamic_buffer_type<device_wrapper_ptr>(1, &fnc)
                                                                      , buffer_type::dynamic_buffer_type<kernel_type>(1, data) );

      char extra_bytes[sizeof(device_wrapper_ptr)];
      buffer_size_type aligned_nbytes = nbytes;
      if (aligned_nbytes % sizeof(device_wrapper_ptr) != 0) {
         aligned_nbytes += sizeof(device_wrapper_ptr) - (aligned_nbytes % sizeof(device_wrapper_ptr));
      }
      assert(aligned_nbytes <= useable_capacity);

      assert(aligned_nbytes == m_info_cur->buffer_device->write_bytes( buffer_type::dynamic_buffer_type<device_wrapper_ptr>(1, &fnc)
                                                                     , buffer_type::dynamic_buffer_type<kernel_type>(1, data)
                                                                     , buffer_type::dynamic_buffer_type<char>(aligned_nbytes - nbytes, &extra_bytes[0]) ));

      if (!cur_buffer_writeable()) {
         return false;
      }

      if (m_info_cur->buffer_pos + aligned_nbytes <= useable_capacity) {

         //FPRINTF(stdout, "pack writing fnc %p\n", fnc);
         m_info_cur->buffer_pos += aligned_nbytes;
         assert(m_info_cur->buffer_pos <= useable_capacity);
         bool success = m_info_cur->buffer_device->write( buffer_type::dynamic_buffer_type<device_wrapper_ptr>(1, &fnc)
                                                        , buffer_type::dynamic_buffer_type<kernel_type>(1, data)
                                                        , buffer_type::dynamic_buffer_type<char>(aligned_nbytes - nbytes, &extra_bytes[0]) );
         assert(success);
         return true;
      } else if (next_buffer_empty()) {

         continue_and_next_buffer();

         //FPRINTF(stdout, "pack writing fnc %p\n", fnc);
         m_info_cur->buffer_pos += aligned_nbytes;
         assert(m_info_cur->buffer_pos <= useable_capacity);
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
      } else if (m_info_cur->buffer_pos == buffer_capacity) {
         if (m_info_cur->buffer_device->reinit()) {
           //FPRINTF(stdout, "cur_buffer_empty reinit-ed %p\n", m_info_cur->buffer_device);
           m_info_cur->buffer_pos = 0;
           return true;
         }
      } else if (m_info_cur->buffer_pos > useable_capacity) {
         assert(0);
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
      if (0 <= m_info_cur->buffer_pos && m_info_cur->buffer_pos <= useable_capacity) {
         return true;
      } else if (m_info_cur->buffer_pos == buffer_capacity) {
         if (m_info_cur->buffer_device->reinit()) {
           //FPRINTF(stdout, "cur_buffer_writeable reinit-ed %p\n", m_info_cur->buffer_device);
           m_info_cur->buffer_pos = 0;
           return true;
         }
      } else {
         assert(0);
      }
      return false;
   }

   bool next_buffer_empty()
   {
      if (m_info_cur->next->buffer_pos == 0) {
         return true;
      } else if (m_info_cur->next->buffer_pos == buffer_capacity) {
         if (m_info_cur->next->buffer_device->reinit()) {
           //FPRINTF(stdout, "next_buffer_empty reinit-ed %p\n", m_info_cur->next->buffer_device);
           m_info_cur->next->buffer_pos = 0;
           return true;
         }
      } else if (m_info_cur->next->buffer_pos > useable_capacity) {
         assert(0);
      }
      return false;
   }

   buffer_type* stop_and_next_buffer()
   {
      void* ptrs[2] {(void*)comb_detail_fnc_null_val, nullptr};
      //FPRINTF(stdout, "stop_and_next_buffer cur %p writing %p %p\n", m_info_cur->buffer_device, ptrs[0], ptrs[1]);
      m_info_cur->buffer_pos = buffer_capacity;
      bool wrote = m_info_cur->buffer_device->write( buffer_type::dynamic_buffer_type<void*>(2, &ptrs[0]) );
      assert(wrote);
      m_info_cur = m_info_cur->next;
      buffer_type* buffer = get_buffer();
      m_info_first = m_info_cur;
      return buffer;
   }

   void continue_and_next_buffer()
   {
      void* ptrs[2] {(void*)comb_detail_fnc_null_val, (void*)m_info_cur->next->buffer_device};
      //FPRINTF(stdout, "continue_and_next_buffer cur %p writing %p %p\n", m_info_cur->buffer_device, ptrs[0], ptrs[1]);
      m_info_cur->buffer_pos = buffer_capacity;
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
      buffer_size_type buffer_pos;
   };

   // current buffer
   internal_info* m_info_first;
   internal_info* m_info_cur;

   internal_info* m_info_arr;
   buffer_type* m_buffer;

   void print_state(const char* func)
   {
      // FPRINTF(stdout, "MultiBuffer(%p) %20s\tpos %4i max_n %7i\n", m_info->buffer_host, func, m_info->buffer_pos);
   }
};


class EventBuffer {
public:

   explicit EventBuffer(size_t capacity)
      : m_event_stack( nullptr )
      , m_event_stack_capacity( 0 )
      , m_event_stack_top( 0 )
      , m_event_buffer( nullptr )
   {
      m_event_stack_capacity = capacity;
      m_event_stack_top = m_event_stack_capacity;
      m_event_stack = (batch_event_type_ptr*)malloc(m_event_stack_capacity*sizeof(batch_event_type_ptr));
      cudaCheck(cudaHostAlloc(&m_event_buffer, m_event_stack_capacity*sizeof(batch_event_type), cudaHostAllocDefault));
      for (size_t i = 0; i < m_event_stack_capacity; ++i) {
         m_event_buffer[i] = event_init_val;
         m_event_stack[i] = &m_event_buffer[i];
      }
   }

   batch_event_type_ptr createEvent()
   {
      batch_event_type_ptr event = nullptr;
      assert(m_event_stack_top != 0);
      if (m_event_stack_top != 0) {
         event = m_event_stack[--m_event_stack_top];
         *event = event_complete_val;
      }
      return event;
   }

   bool recordEvent(batch_event_type_ptr event, MultiBuffer& mb)
   {
      *event = event_init_val;
      // relies on pack to perform thread fence
      return mb.pack(comb_detail_fnc_event_val, &event);
   }

   bool queryEvent(batch_event_type_ptr event)
   {
      if (*event == event_complete_val) {
         std::atomic_thread_fence(std::memory_order_acquire);
         return true;
      } else {
         return false;
      }
   }

   void waitEvent(batch_event_type_ptr event)
   {
      while (!queryEvent(event));
   }

   void destroyEvent(batch_event_type_ptr event)
   {
      assert(m_event_stack_top != m_event_stack_capacity);
      if (m_event_stack_top != m_event_stack_capacity) {
         m_event_stack[m_event_stack_top++] = event;
      }
   }

   ~EventBuffer()
   {
      free(m_event_stack);
      // cudaCheck(cudaFreeHost(m_event_buffer));
   }
private:

   batch_event_type_ptr* m_event_stack;
   size_t m_event_stack_capacity;
   size_t m_event_stack_top;
   batch_event_type* m_event_buffer;
};

} // namespace detail

#endif // _MULTIBUFFER_CUH

