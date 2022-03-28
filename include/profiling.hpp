//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
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

#include "config.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>

#include "print.hpp"
#include "comm_utils_mpi.hpp"
#include "exec_utils_cuda.hpp"
#include "exec_utils_hip.hpp"

#include "ExecContext.hpp"

#ifdef COMB_ENABLE_CALIPER
#include <caliper/cali.h>
#endif

struct Timer {

  enum {
    unused
   ,cpu
#ifdef COMB_ENABLE_CUDA
   ,cuda
#endif
#ifdef COMB_ENABLE_HIP
   ,hip
#endif
  };

  struct TimePoint
  {
    static double duration(TimePoint const& t0, TimePoint const& t1)
    {
      double time = -1.0;
      if (t0.type == cpu && t1.type == cpu) {
        time = std::chrono::duration<double>(t1.tp_cpu - t0.tp_cpu).count();
#ifdef COMB_ENABLE_CUDA
      } else if (t0.type == cuda && t1.type == cuda) {
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, t0.tp_cuda, t1.tp_cuda));
        time = static_cast<double>(ms) / 1000.0;
#endif
#ifdef COMB_ENABLE_HIP
      } else if (t0.type == hip && t1.type == hip) {
        float ms;
        hipCheck(hipEventElapsedTime(&ms, t0.tp_hip, t1.tp_hip));
        time = static_cast<double>(ms) / 1000.0;
#endif
      } else {
        assert(0 && "TimePoint::duration type mismatch");
      }
      return time;
    }

    std::chrono::high_resolution_clock::time_point tp_cpu;
#ifdef COMB_ENABLE_CUDA
    cudaEvent_t tp_cuda;
#endif
#ifdef COMB_ENABLE_HIP
    hipEvent_t tp_hip;
#endif
    int type;

    TimePoint()
      : type(unused)
    {
#ifdef COMB_ENABLE_CUDA
      cudaCheck(cudaEventCreateWithFlags(&tp_cuda, cudaEventDefault));
#endif
#ifdef COMB_ENABLE_HIP
      hipCheck(hipEventCreateWithFlags(&tp_hip, hipEventDefault));
#endif
    }

    void record(CPUContext&)
    {
      tp_cpu = std::chrono::high_resolution_clock::now();
      type = cpu;
    }

#ifdef COMB_ENABLE_MPI
    void record(MPIContext&)
    {
      tp_cpu = std::chrono::high_resolution_clock::now();
      type = cpu;
    }
#endif

#ifdef COMB_ENABLE_CUDA
    void record(CudaContext& con)
    {
      cudaCheck(cudaEventRecord(tp_cuda, con.stream()));
      type = cuda;
    }
#endif

#ifdef COMB_ENABLE_HIP
    void record(HipContext& con)
    {
      hipCheck(hipEventRecord(tp_hip, con.stream()));
      type = hip;
    }
#endif

    ~TimePoint()
    {
#ifdef COMB_ENABLE_CUDA
      cudaCheck(cudaEventDestroy(tp_cuda));
#endif
#ifdef COMB_ENABLE_HIP
      hipCheck(hipEventDestroy(tp_hip));
#endif
    }
  };

  struct Stats {
    std::string name;
    double sum;
    double min;
    double max;
    long   num;
  };

  std::vector<TimePoint> times;
  std::vector<const char*> names;
  size_t idx;

  Timer(IdxT size)
    : idx(0)
  {
    resize(size);
  }

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  template < typename Context >
  void start(Context& con, const char* str) {
    if (idx >= times.size()) {
      resize(2*idx+2);
      assert(idx < times.size());
    }
    times[idx].record(con);
    names[idx] = str;
    ++idx;
  }

  template < typename Context >
  void restart(Context& con, const char* str) {
    start(con, str);
  }

  template < typename Context >
  void stop(Context& con) {
    start(con, nullptr);
  }

  std::vector<std::pair<std::string, double>> get_times()
  {
    std::vector<std::pair<std::string, double>> items;

    for (size_t i = 1; i < idx; ++i) {
      if (names[i-1] != nullptr) {

        std::string name{names[i-1]};
        double time = TimePoint::duration(times[i-1], times[i]);

        items.emplace_back(std::pair<std::string, double>{name, time});
      }
    }

    return items;
  }

  std::vector<Stats> getStats() {
    // gather basic per name statistics in first seen order
    std::vector<std::string> name_order;
    using map_type = std::unordered_map<std::string, Stats>;
    map_type name_map;

    for (auto& time : get_times()) {
      std::string& name{time.first};
      double time_s = time.second;
      auto item = name_map.find(name);
      if (item == name_map.end()) {
        auto ins = name_map.insert(map_type::value_type{name, Stats{name, time_s, time_s, time_s, 1}});
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

    std::vector<Stats> res;

    for (size_t i = 0; i < name_order.size(); ++i) {
      auto item = name_map.find(name_order[i]);
      assert(item != name_map.end());
      res.emplace_back(item->second);
    }

    return res;
  }

  void print(const char* prefix = "") {

    std::vector<Stats> res = getStats();

    int max_name_len = 0;

    for (auto& stat : res) {
      max_name_len = std::max(max_name_len, (int)stat.name.size());
    }

    for (auto& stat : res) {
      int padding = max_name_len - stat.name.size();
      fgprintf(FileGroup::proc, "%s%s:%*s num %ld avg %.9f s min %.9f s max %.9f s\n",
                    prefix, stat.name.c_str(), padding, "", stat.num, stat.sum/stat.num, stat.min, stat.max);
    }
  }

  void resize(size_t size) {
    if (idx > size) idx = size;
    times.resize(size);
    names.resize(size);
  }

  void clear() {
    idx = 0;
  }

  void clean() {
    clear();
    times.clear();
    names.clear();
  }

  ~Timer() { clean(); }
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
#ifdef COMB_ENABLE_CALIPER
  cali::Annotation ann;
#endif

  Range(const char* name_, uint32_t color)
    : name(nullptr)
#ifdef COMB_ENABLE_CALIPER
      , ann("comb.range", CALI_ATTR_NESTED)
#endif
  {
    start(name_, color);
  }

  void start(const char* name_, uint32_t color) {
    COMB::ignore_unused(color);
    if (name_ != nullptr) {
      LOGPRINTF("%p Range::start name %s\n", this, name_);
#ifdef COMB_ENABLE_CUDA
      nvtxEventAttributes_t eventAttrib = {0};
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = color;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = name_;
      nvtxRangePushEx(&eventAttrib);
#endif
#ifdef COMB_ENABLE_HIP
      roctxRangePush(name_);
#endif
#ifdef COMB_ENABLE_CALIPER
      ann.begin(name_);
#endif
      name = name_;
    }
  }

  void stop()
  {
    if(name != nullptr) {
      LOGPRINTF("%p Range::stop name %s\n", this, name);
#ifdef COMB_ENABLE_CALIPER
      ann.end();
#endif
#ifdef COMB_ENABLE_CUDA
      nvtxRangePop();
#endif
#ifdef COMB_ENABLE_HIP
      roctxRangePop();
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

