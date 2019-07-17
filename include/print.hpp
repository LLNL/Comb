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

#ifndef _PRINT_HPP
#define _PRINT_HPP

#include "config.hpp"

#include <cstdio>

enum struct FileGroup
{ out_any    // stdout, any proc
, out_master // stdout, master only
, err_any    // stderr, any proc
, err_master // stderr, master only
, proc       // per process file, any proc
, summary    // per run summary file, master only
, all        // out_master, proc, summary
};

extern int   mpi_rank;
extern FILE* comb_out_file;
extern FILE* comb_err_file;
extern FILE* comb_proc_file;
extern FILE* comb_summary_file;

extern void comb_setup_files();
extern void comb_teardown_files();

extern void print(FileGroup fg, const char* fmt, ...);
extern void print_proc_memory_stats();

#endif // _PRINT_HPP
