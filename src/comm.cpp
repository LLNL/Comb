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

#include "comm.hpp"

FILE* comb_out_file = nullptr;
FILE* comb_err_file = nullptr;
FILE* comb_proc_file = nullptr;
FILE* comb_summary_file = nullptr;

void comb_setup_files(int rank)
{
  comb_out_file = stdout;
  comb_err_file = stderr;

  int run_num = 0;

  if (rank == 0) {
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

  detail::MPI::Bcast(&run_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

  char proc_fname[256];
  snprintf(proc_fname, 256, "Comb_%02i_proc%04i", run_num, rank);
  comb_proc_file = fopen(proc_fname, "w");
  assert(comb_proc_file != nullptr);
}

void comb_teardown_files()
{
  comb_out_file = nullptr;
  comb_err_file = nullptr;
  if (comb_proc_file != nullptr) {
    fclose(comb_proc_file);
    comb_proc_file = nullptr;
  }
  if (comb_summary_file != nullptr) {
    fclose(comb_summary_file);
    comb_summary_file = nullptr;
  }
}

void CommInfo::print(FileGroup fg, const char* fmt, ...)
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

  // print to out file
  if ( fg == FileGroup::out_any ||
       ((fg == FileGroup::out_master || fg == FileGroup::all) && rank == 0) ) {
    fprintf(comb_out_file, "%s", msg);
    fflush(comb_out_file);
  }

  // print to err file
  if ( fg == FileGroup::err_any ||
       (fg == FileGroup::err_master && rank == 0) ) {
    fprintf(comb_err_file, "%s", msg);
    fflush(comb_err_file);
  }

  // print to proc file
  if ( fg == FileGroup::proc ||
       fg == FileGroup::all ) {
    fprintf(comb_proc_file, "%s", msg);
    fflush(comb_proc_file);
  }

  // print to summary file
  if ( (fg == FileGroup::summary || fg == FileGroup::all) && rank == 0 ) {
    fprintf(comb_summary_file, "%s", msg);
      fflush(comb_summary_file);
  }

  free(msg);
}
