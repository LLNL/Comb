//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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

#include "print.hpp"
#include "comm_utils_mpi.hpp"
#include "utils_cuda.hpp"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <vector>

int mpi_rank = 0;
FILE* comb_out_file = stdout;
FILE* comb_err_file = stderr;
FILE* comb_proc_file = nullptr;
FILE* comb_summary_file = nullptr;

void comb_setup_files()
{
#ifdef COMB_ENABLE_MPI
  mpi_rank = detail::MPI::Comm_rank(MPI_COMM_WORLD);
#endif

  int run_num = 0;

  if (mpi_rank == 0) {
    while (1) {

      char summary_fname[256];
      snprintf(summary_fname, 256, "Comb_%02i_summary", run_num);

      // try to open summary_fname
      FILE* check_file = fopen(summary_fname, "r");

      if (check_file == nullptr) {
        // run number not used
        comb_summary_file = fopen(summary_fname, "w");
        assert(comb_summary_file != nullptr);
        break;
      } else {
        // already have a run with that number
        fclose(check_file); check_file = nullptr;
        run_num += 1;

        if (run_num >= 100) {
          fprintf(stderr, "Comb could not find an unused file name of the form \"Comb_??_summary\".\n");
          abort();
        }
      }
    }
  }

#ifdef COMB_ENABLE_MPI
  detail::MPI::Bcast(&run_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  char proc_fname[256];
  snprintf(proc_fname, 256, "Comb_%02i_proc%04i", run_num, mpi_rank);
  comb_proc_file = fopen(proc_fname, "w");
  assert(comb_proc_file != nullptr);
}

void comb_teardown_files()
{
  mpi_rank = 0;
  comb_out_file = stdout;
  comb_err_file = stderr;
  if (comb_proc_file != nullptr) {
    fclose(comb_proc_file);
    comb_proc_file = nullptr;
  }
  if (comb_summary_file != nullptr) {
    fclose(comb_summary_file);
    comb_summary_file = nullptr;
  }
}

void fgprintf(FileGroup fg, const char* fmt, ...)
{
  va_list args1;
  va_start(args1, fmt);

  va_list args2;
  va_copy(args2, args1);

  int len = vsnprintf(nullptr, 0, fmt, args1);
  va_end(args1);

  char* msg = (char*)malloc(len+1);
  vsnprintf(msg, len+1, fmt, args2);
  va_end(args2);

  bool should_have_printed = false;
  bool printed = false;

  // print to out file
  if ( fg == FileGroup::out_any ||
       ((fg == FileGroup::out_master || fg == FileGroup::all) && mpi_rank == 0) ) {
    FILE* f = (comb_out_file != nullptr) ? comb_out_file : stdout;
    if (f != nullptr) {
      fprintf(f, "%s", msg);
      fflush(f);
      printed = true;
    }
    should_have_printed = true;
  }

  // print to err file
  if ( fg == FileGroup::err_any ||
       (fg == FileGroup::err_master && mpi_rank == 0) ) {
    FILE* f = (comb_err_file != nullptr) ? comb_err_file : stderr;
    if (f != nullptr) {
      fprintf(f, "%s", msg);
      fflush(f);
      printed = true;
    }
    should_have_printed = true;
  }

  // print to proc file
  if ( fg == FileGroup::proc ||
       fg == FileGroup::all ) {
    FILE* f = comb_proc_file;
    if (f == nullptr && !printed) f = stdout;
    if (f != nullptr) {
      fprintf(f, "%s", msg);
      fflush(f);
      printed = true;
    }
    should_have_printed = true;
  }

  // print to summary file
  if ( (fg == FileGroup::summary || fg == FileGroup::all) && mpi_rank == 0 ) {
    FILE* f = comb_summary_file;
    if (f == nullptr && !printed) f = stdout;
    if (f != nullptr) {
      fprintf(f, "%s", msg);
      fflush(f);
      printed = true;
    }
    should_have_printed = true;
  }

  if (should_have_printed && !printed) {
    printf("%s", msg);
  }

  free(msg);
}

void print_proc_memory_stats()
{
  // print /proc/self/stat to per proc file
  std::vector<char> stat;
  const char fstat_name[] = "/proc/self/stat";
  FILE* fstat = fopen(fstat_name, "r");
  if (fstat) {
    // fprintf(f, "%s ", fstat_name);
    int c = fgetc(fstat);
    int clast = c;
    while(c != EOF) {
      stat.push_back(c);
      // fputc(c, f);
      clast = c;
      c = fgetc(fstat);
    }
    fclose(fstat);
    if (clast != '\n') {
      stat.push_back('\n');
      // fprintf(f, "\n");
    }
    stat.push_back('\0');
  }
  fgprintf(FileGroup::proc, "/proc/self/stat: %s", &stat[0]);

#if defined(COMB_ENABLE_CUDA)
  // print cuda device memory usage to per proc file
  size_t free_mem, total_mem;
  cudaCheck(cudaMemGetInfo(&free_mem, &total_mem));
  fgprintf(FileGroup::proc, "cuda device memory usage: %12zu\n", total_mem - free_mem);
#endif
}
