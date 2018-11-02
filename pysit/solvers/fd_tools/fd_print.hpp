#ifndef __FD_PRINT__
#define __FD_PRINT__

template < template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER, long long int POINT>
class CoeffPrinterHelper
{
    public:
        static void print()
        {
            CoeffPrinterHelper<FDWEIGHT, DERIV, NPOINTS, CENTER, POINT-1>::print();
            std::cout << FDWEIGHT<DERIV, NPOINTS, CENTER, POINT>::value<< " ";
        }
        static void print_rational()
        {
            if(not FDWEIGHT<DERIV, NPOINTS, CENTER, POINT>::RATIONAL)
                throw(0);

            CoeffPrinterHelper<FDWEIGHT, DERIV, NPOINTS, CENTER, POINT-1>::print_rational();
            std::cout << FDWEIGHT<DERIV, NPOINTS, CENTER, POINT>::N << "/" << FDWEIGHT<DERIV, NPOINTS, CENTER, POINT>::D << " ";
        }
};

template < template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER>
class CoeffPrinterHelper<FDWEIGHT, DERIV, NPOINTS, CENTER, 0>
{
    public:
        static void print()
        {
            std::cout << FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::value << " ";
        }
        static void print_rational()
        {
            if(not FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::RATIONAL)
                throw(0);
            std::cout << FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::N << "/" << FDWEIGHT<DERIV, NPOINTS, CENTER, 0>::D << " ";
        }
};

template < template < long long int A, long long int B, long long int C, long long int D > class FDWEIGHT,
           long long int DERIV, long long int NPOINTS, long long int CENTER>
class CoeffPrinter
{
    public:
        static void print()
        {
            CoeffPrinterHelper<FDWEIGHT, DERIV, NPOINTS, CENTER, NPOINTS-1>::print();
            std::cout << std::endl << std::endl;
        }
        static void print_rational()
        {
            CoeffPrinterHelper<FDWEIGHT, DERIV, NPOINTS, CENTER, NPOINTS-1>::print_rational();
            std::cout << std::endl << std::endl;
        }
};


#endif