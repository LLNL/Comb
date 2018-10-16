//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

#ifndef _PROFILING_HPP
#define _PROFILING_HPP

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include <atomic>

#ifdef COMB_HAVE_CUDA
#include <cuda.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#endif

#include <mpi.h>

struct Timer {

  struct TimePoint {
    using type = std::chrono::high_resolution_clock::time_point;
    type tp;
  };

  TimePoint* times;
  const char** names;
  IdxT idx;
  IdxT size;

  Timer(IdxT size_)
    : times(nullptr)
    , names(nullptr)
    , idx(0)
    , size(0)
  {
    resize(size_);
  }

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  void start(const char* str) {
    if (idx < size) {
      record_time();
      names[idx] = str;
      ++idx;
    }
  }

  void restart(const char* str) {
    if (idx < size) {
      start(str);
    } else if (idx == size) {
      stop();
    }
  }

  void stop() {
    if (idx <= size) {
      record_time();
      if (idx < size) names[idx] = nullptr;
      ++idx;
    }
  }


  void print(const char* prefix = "") {
    // gather bsic per name statistics in first seen order
    struct stats {
      std::string name;
      double sum;
      double min;
      double max;
      long   num;
    };
    std::vector<std::string> name_order;
    using map_type = std::unordered_map<std::string, stats>;
    map_type name_map;

    for (IdxT i = 1; i < idx; ++i) {
      if (names[i-1] != nullptr) {
        std::string name{names[i-1]};
        double time_s = std::chrono::duration<double>(times[i].tp - times[i-1].tp).count();
        auto item = name_map.find(name);
        if (item == name_map.end()) {
          auto ins = name_map.insert(map_type::value_type{name, stats{name, time_s, time_s, time_s, 1}});
          assert(ins.second);
          item = ins.first;
          name_order.emplace_back(name);
        } else {
          item->second.sum += time_s;
          item->second.min = std::min(item->second.min, time_s);
          item->second.max = std::max(item->second.max, time_s);
          item->second.num += 1;
        }
      }
    }

    for (IdxT i = 0; i < name_order.size(); ++i) {
      auto item = name_map.find(name_order[i]);
      assert(item != name_map.end());
      FPRINTF(stdout, "%s%s: num %ld sum %.9f s min %.9f s max %.9f s\n", prefix, item->second.name.c_str(), item->second.num, item->second.sum, item->second.min, item->second.max);
    }
  }

  void resize(IdxT size_) {
    clean();
    times = (TimePoint*)malloc((size_+1)*sizeof(TimePoint));
    names = (const char**)malloc(size_*sizeof(const char*));
    size = size_;
  }

  void clear() {
    for (IdxT i = 0; i < idx; ++i) {
      times[i].~TimePoint();
    }
    idx = 0;
  }

  void clean() {
    clear();
    if (times != nullptr) { free(times); times = nullptr; }
    if (names != nullptr) { free(names); names = nullptr; }
  }

  ~Timer() { clean(); }
private:
  void record_time() {

    std::atomic_thread_fence(std::memory_order_release);

    new(&times[idx]) TimePoint{std::chrono::high_resolution_clock::now()};

    std::atomic_thread_fence(std::memory_order_acquire);

  }
};

struct Range {
  static const uint32_t green    = 0x0000FF00;
  static const uint32_t red      = 0x00FF0000;
  static const uint32_t blue     = 0x000000FF;
  static const uint32_t yellow   = 0x00FFFF00;
  static const uint32_t cyan     = 0x0000FFFF;
  static const uint32_t indigo   = 0x004B0082;
  static const uint32_t magenta  = 0x00FF00FF;
  static const uint32_t orange   = 0x00FFA500;
  static const uint32_t pink     = 0x00FF69B4;

  const char* name;
#ifdef COMB_HAVE_CUDA
  nvtxRangeId_t id;
#endif

  Range(const char* name_, uint32_t color)
    : name(nullptr)
  {
    start(name_, color);
  }

  void start(const char* name_, uint32_t color) {
    if (name_ != nullptr) {
#ifdef COMB_HAVE_CUDA
      nvtxEventAttributes_t eventAttrib = {0};
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = color;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = name_;
      id = nvtxRangeStartEx(&eventAttrib);
#endif
      name = name_;
    }
  }

  void stop()
  {
    if(name != nullptr) {
#ifdef COMB_HAVE_CUDA
      nvtxRangeEnd(id);
#endif
      name = nullptr;
    }
  }

  void restart(const char* name_, uint32_t color) {
    stop();
    start(name_, color);
  }

  ~Range()
  {
    stop();
  }
};

#endif // _PROFILING_HPP

