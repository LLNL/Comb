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

#include "config.hpp"

#include "comb.hpp"

namespace COMB {

void print_timer(CommInfo& comminfo, Timer& tm, const char* prefix) {

  auto res = tm.getStats();

  int max_name_len = 0;

  for (auto& stat : res) {
    max_name_len = std::max(max_name_len, (int)stat.name.size());
  }

  double* sums = new double[res.size()];
  double* mins = new double[res.size()];
  double* maxs = new double[res.size()];
  long  * nums = new long  [res.size()];

  for (int i = 0; i < (int)res.size(); ++i) {
    sums[i] = res[i].sum;
    mins[i] = res[i].min;
    maxs[i] = res[i].max;
    nums[i] = res[i].num;
  }

  double* final_sums = nullptr;
  double* final_mins = nullptr;
  double* final_maxs = nullptr;
  long  * final_nums = nullptr;
  if (comminfo.rank == 0) {
    final_sums = new double[res.size()];
    final_mins = new double[res.size()];
    final_maxs = new double[res.size()];
    final_nums = new long  [res.size()];
  }

  MPI_Reduce(sums, final_sums, res.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(mins, final_mins, res.size(), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(maxs, final_maxs, res.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(nums, final_nums, res.size(), MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);

  if (comminfo.rank == 0) {

    for (int i = 0; i < (int)res.size(); ++i) {
      int padding = max_name_len - res[i].name.size();
      fgprintf(FileGroup::summary, "%s%s:%*s num %ld avg %.9f s min %.9f s max %.9f s\n",
                             prefix, res[i].name.c_str(), padding, "", final_nums[i], final_sums[i]/final_nums[i], final_mins[i], final_maxs[i]);
    }

    delete[] final_sums;
    delete[] final_mins;
    delete[] final_maxs;
    delete[] final_nums;
  }

  for (int i = 0; i < (int)res.size(); ++i) {
    int padding = max_name_len - res[i].name.size();
    fgprintf(FileGroup::proc, "%s%s:%*s num %ld avg %.9f s min %.9f s max %.9f s\n",
                        prefix, res[i].name.c_str(), padding, "", nums[i], sums[i]/nums[i], mins[i], maxs[i]);
  }

  delete[] sums;
  delete[] mins;
  delete[] maxs;
  delete[] nums;
}

} // namespace COMB
