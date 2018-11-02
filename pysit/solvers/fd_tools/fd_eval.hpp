#ifndef __FD_EVAL__
#define __FD_EVAL__

/******************************************************************************\
********        General application of finite difference stencil        ********
\******************************************************************************/

template < typename T,
           template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER, long long int POINT>
class FDStencilHelper
{
    public:
        static T apply(T* v, int const& i, int const& s)
        {
            T coeff = FDWEIGHT<DERIV, NPOINTS, CENTER, POINT>::value;
            T retval = FDStencilHelper<T,FDWEIGHT, DERIV, NPOINTS, CENTER, POINT-1>::apply(v, i, s)
                   + v[i+(POINT-CENTER)*s]*coeff;
            //std::cout << coeff << "(" << (POINT-CENTER) << ", " << i+(POINT-CENTER)*s << ") " << v[i+(POINT-CENTER)*s] << " ";
            return retval;
        }
};

template < typename T,
           template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER>
class FDStencilHelper<T, FDWEIGHT, DERIV, NPOINTS, CENTER, 0>
{
    public:
        static T apply(T* v, int const& i, int const& s)
        {
            //T coeff = FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::value;
            //std::cout << coeff << "(" << (0-CENTER) << ", " << i+(0-CENTER)*s  << ") " << v[i+(0-CENTER)*s] << " ";// << std::endl;
            return v[i+(0-CENTER)*s]*FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::value;
        }
};

template < typename T,
           template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER>
class FDStencil
{
    public:
        static T apply(T* v, int const& i, int const& s)
        // v: vector; i: base index; s: stride
        {
            T val = FDStencilHelper<T, FDWEIGHT, DERIV, NPOINTS, CENTER, NPOINTS-1>::apply(v, i, s);
            //std::cout << std::endl << std::endl;
            return val;
        }
};


/******************************************************************************\
********        Application of FD stencil for zero-padded domains       ********
\******************************************************************************/

template < typename T,
           template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV,
           long long int NPOINTS, long long int CENTER,
           long long int START_FD_COEFF_INDEX,
           long long int POINT>
class ZeroPaddedFDStencilHelper
{
    public:
        static T apply(T* v, int const& i, int const& s)
        {
            if (POINT >= START_FD_COEFF_INDEX)
            {
                T coeff = FDWEIGHT<DERIV, NPOINTS, CENTER, POINT>::value;
//              std::cout << v[i+(POINT-CENTER)*s] << "*" << coeff << " + " ;
                return ZeroPaddedFDStencilHelper<T,FDWEIGHT, DERIV, NPOINTS, CENTER, START_FD_COEFF_INDEX, POINT-1>::apply(v, i, s) + v[i+(POINT-CENTER)*s]*coeff;
            }
            return 0;
        }
};

template < typename T,
           template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER, long long int START_FD_COEFF_INDEX >
class ZeroPaddedFDStencilHelper<T, FDWEIGHT, DERIV, NPOINTS, CENTER, START_FD_COEFF_INDEX, 0>
{
    public:
        static T apply(T* v, int const& i, int const& s)
        {
            if(0 == START_FD_COEFF_INDEX)
//              std::cout << v[i+(0-CENTER)*s] << "*" << FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::value << " + " ;
                return v[i+(0-CENTER)*s]*FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::value;
            return 0;
        }
};

template < typename T,
           template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV,
           long long int NPOINTS, long long int CENTER,
           int START_FD_COEFF_INDEX, int END_FD_COEFF_INDEX>
class ZeroPaddedFDStencil
{
    public:
        static T apply(T* v, int const& i, int const& s)
        // v: vector; i: base index; s: stride
        {
            enum {POINT = END_FD_COEFF_INDEX};

            T val = ZeroPaddedFDStencilHelper<T, FDWEIGHT, DERIV, NPOINTS, CENTER, START_FD_COEFF_INDEX, POINT>::apply(v, i, s);
//          std::cout << std::endl;
            return val;
        }
};


#endif