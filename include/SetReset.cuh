
#ifndef _SETRESET_CUH
#define _SETRESET_CUH

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

#endif // _SETRESET_CUH
