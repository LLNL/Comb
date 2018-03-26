#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <cuda.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <mpi.h>

#ifndef _PROFILING_CUH
#define _PROFILING_CUH

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
    for (IdxT i = 1; i < idx; ++i) {
      if (names[i-1] != nullptr) {
        std::chrono::duration<double> time_s = times[i].tp - times[i-1].tp;
        printf("%s%s: %.9f s\n", prefix, names[i-1], time_s.count());
      }
    }
    fflush(stdout);
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
      new(&times[idx]) TimePoint{std::chrono::high_resolution_clock::now()};
  }
};

struct Range {
  static const uint32_t green    = 0x0000FF00;
  static const uint32_t red      = 0x00FF0000;
  static const uint32_t blue     = 0x000000FF;
  static const uint32_t yellow   = 0x00FFFF00;
  static const uint32_t cyan     = 0x0000FFFF;
  static const uint32_t magenta  = 0x00FF00FF;
  static const uint32_t orange   = 0x00FFA500;
  static const uint32_t pink     = 0x00FF69B4;

  const char* name;
  nvtxRangeId_t id;

  Range(const char* name_, uint32_t color)
    : name(nullptr)
  {
    start(name_, color);
  }

  void start(const char* name_, uint32_t color) {
    if (name_ != nullptr) {
      nvtxEventAttributes_t eventAttrib = {0};
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = color;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = name_;
      id = nvtxRangeStartEx(&eventAttrib);
      name = name_;
    }
  }

  void stop()
  {
    if(name != nullptr) {
      nvtxRangeEnd(id);
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

#endif // _PROFILING_CUH

