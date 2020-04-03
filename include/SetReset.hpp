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

#ifndef _SETRESET_HPP
#define _SETRESET_HPP

template <typename T>
class SetReset
{
public:
   SetReset(T& var_, T new_val)
      :  var(var_)
      ,  orig_val(var_)
   {
      var = new_val;
   }
   ~SetReset()
   {
      var = orig_val;
   }
private:
   T& var;
   T orig_val;
};

#endif // _SETRESET_HPP
