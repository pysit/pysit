#ifndef __FD_RATIONAL__
#define __FD_RATIONAL__

#include "fd_util.hpp"

template < long long int DERIV, long long int NPOINTS, long long int CENTER, long long int POINT >
class CompCoeffRational
{
    public:
        enum { I = POINT  };
        enum { J = NPOINTS  };
        enum { K = DERIV  };
        enum { Z = CENTER };
        enum {__N = (J-Z)*static_cast<long long int>(CompCoeffRational<K, J-1, Z, I>::N)*static_cast<long long int>(CompCoeffRational<K-1, J-1, Z, I>::D) - K*static_cast<long long int>(CompCoeffRational<K-1, J-1, Z, I>::N)*static_cast<long long int>(CompCoeffRational<K, J-1, Z, I>::D)};
        enum {__D = (J-I)*static_cast<long long int>(CompCoeffRational<K, J-1, Z, I>::D)*static_cast<long long int>(CompCoeffRational<K-1, J-1, Z, I>::D)};
        enum {__GCD = GCD<__N,__D>::value};
        enum {N = __N/__GCD};
        enum {D = __D/__GCD};
};

template < long long int DERIV, long long int NPOINTS, long long int CENTER >
class CompCoeffRational<DERIV, NPOINTS, CENTER, NPOINTS>
{
    public:
        enum { J = NPOINTS  };
        enum { K = DERIV  };
        enum { Z = CENTER };
        enum {__N = ProdN<long long int, J>::value * (K*static_cast<long long int>(CompCoeffRational<K-1, J-1, Z, J-1>::N)*static_cast<long long int>(CompCoeffRational<K, J-1, Z, J-1>::D) - (J-1 - Z)*static_cast<long long int>(CompCoeffRational<K, J-1, Z, J-1>::N)*static_cast<long long int>(CompCoeffRational<K-1, J-1, Z, J-1>::D))};
        enum {__D = ProdD<long long int, J>::value * static_cast<long long int>(CompCoeffRational<K-1, J-1, Z, J-1>::D)*static_cast<long long int>(CompCoeffRational<K, J-1, Z, J-1>::D)};
        enum {__GCD = GCD<__N,__D>::value};
        enum {N = __N/__GCD};
        enum {D = __D/__GCD};
};

template <long long int CENTER>
class CompCoeffRational<0LL,0LL,CENTER,0LL>
{
    public:
        enum {N = 1};
        enum {D = 1};
};

template <long long int A, long long int CENTER, long long int B>
class CompCoeffRational<-1,A,CENTER,B>
{
    public:
        enum {N = 0};
        enum {D = 1};
};

template <long long int A,long long int CENTER>
class CompCoeffRational<-1,A,CENTER,A>
{
    public:
        enum {N = 0};
        enum {D = 1};
};

template <long long int A,long long int CENTER>
class CompCoeffRational<A,-1,CENTER,-1>
{
    public:
        enum {N = 0};
        enum {D = 1};
};

template <long long int A,long long int CENTER, long long int B>
class CompCoeffRational<A,-1,CENTER,B>
{
    public:
        enum {N = 0};
        enum {D = 1};
};

template <long long int A, long long int B,long long int CENTER>
class CompCoeffRational<A,B,CENTER,-1>
{
    public:
        enum {N = 0};
        enum {D = 1};
};

template <long long int A,long long int CENTER>
class CompCoeffRational<-1,A,CENTER,-1>
{
    public:
        enum {N = 0};
        enum {D = 1};
};

template < long long int DERIV, long long int NPOINTS, long long int CENTER, long long int POINT >
class FDWeightRational
{
    public:
        enum { RATIONAL = 1};
        enum {N = CompCoeffRational<DERIV, NPOINTS-1, CENTER, POINT>::N};
        enum {D = CompCoeffRational<DERIV, NPOINTS-1, CENTER, POINT>::D};
        static const double value = (1.0*N) / (1.0*D);
};

#endif