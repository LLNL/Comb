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

#ifndef _EXECCONTEXT_HPP
#define _EXECCONTEXT_HPP

#include "config.hpp"

template < typename exec_pol >
struct ExecContext;

template < typename lhs_pol, typename rhs_pol >
inline bool operator==(ExecContext<lhs_pol> const&, ExecContext<rhs_pol> const&)
{
  return false;
}

template < typename lhs_pol, typename rhs_pol >
inline bool operator!=(ExecContext<lhs_pol> const&lhs, ExecContext<rhs_pol> const&rhs)
{
  return !operator==(lhs, rhs);
}

#endif // _EXECCONTEXT_HPP
