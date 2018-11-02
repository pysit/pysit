#ifndef __FD_REAL__
#define __FD_REAL__

#include "fd_util.hpp"

template < long long int DERIV, long long int NPOINTS, long long int CENTER, long long int POINT >
class CompCoeffReal
{
    public:
    enum { I = POINT  };
    enum { J = NPOINTS  };
    enum { K = DERIV  };
    enum { Z = CENTER };
    static const double value = ((J-Z)*CompCoeffReal<K, J-1, Z, I>::value - K*CompCoeffReal<K-1, J-1, Z, I>::value)/(J-I);
};

template < long long int DERIV, long long int NPOINTS, long long int CENTER >
class CompCoeffReal<DERIV, NPOINTS, CENTER, NPOINTS>
{
    public:
    enum { J = NPOINTS  };
    enum { K = DERIV  };
    enum { Z = CENTER };
    static const double value = ProdN<double, J>::value * (K*CompCoeffReal<K-1, J-1, Z, J-1>::value - (J-1-Z)*CompCoeffReal<K, J-1, Z, J-1>::value)/ProdD<double, J>::value;
};

template <long long int CENTER>
class CompCoeffReal<0LL,0LL,CENTER,0LL>
{
    public:
        static const double value = 1.0;
};

template <long long int A, long long int CENTER, long long int B>
class CompCoeffReal<-1,A,CENTER,B>
{
    public:
        static const double value = 0.0;
};

template <long long int A,long long int CENTER>
class CompCoeffReal<-1,A,CENTER,A>
{
    public:
        static const double value = 0.0;
};

template <long long int A,long long int CENTER>
class CompCoeffReal<A,-1,CENTER,-1>
{
    public:
        static const double value = 0.0;
};

template <long long int A,long long int CENTER, long long int B>
class CompCoeffReal<A,-1,CENTER,B>
{
    public:
        static const double value = 0.0;
};

template <long long int A, long long int B,long long int CENTER>
class CompCoeffReal<A,B,CENTER,-1>
{
    public:
        static const double value = 0.0;
};

template <long long int A,long long int CENTER>
class CompCoeffReal<-1,A,CENTER,-1>
{
    public:
        static const double value = 0.0;
};

template < long long int DERIV, long long int NPOINTS, long long int CENTER, long long int POINT >
class FDWeightReal
{
    public:
        enum { RATIONAL = 0};
        static const double value = CompCoeffReal<DERIV, NPOINTS-1, CENTER, POINT>::value;
};


#endif