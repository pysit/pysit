#ifndef __UTIL_HPP__
#define __UTIL_HPP__

template <long long int A, long long int B>
class GCD
{
    public:
    enum {value = GCD<B, A%B>::value};
};

template <long long int A>
class GCD<A,0>
{
    public:
    enum {value = A};
};

template <typename T, int F, int U>
class ProdHelper{ public: static const T value = (F-U)*ProdHelper<T, F,U-1>::value;};

template <typename T, int F>
class ProdHelper<T, F,0>{ public: static const T value = F;};

template <typename T, int J>
class ProdN{ public: static const T value = ProdHelper<T, J-1, J-2>::value;};

template <typename T>
class ProdN<T, 1>{  public: static const T value = static_cast<T>(1);};

template <typename T>
class ProdN<T, 0>{  public: static const T value = static_cast<T>(1);};

template <typename T, int J>
class ProdD{ public: static const T value = ProdHelper<T, J, J-1>::value;};

template <typename T>
class ProdD<T, 0>{  public: static const T value = static_cast<T>(1);};

#endif