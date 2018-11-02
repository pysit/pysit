#ifndef __FD_MANUAL_HPP__
#define __FD_MANUAL_HPP__


// T=type, A=accuracy, S=shift value, D=derivative
template< typename T, int D, int A, int S > class FD {public: static T apply(T* v, int const& i, int const& s, T const& dv) {return 0;}; };
    //                  DERIV=1  ACC=2
template< typename T> class FD<T, 1, 2, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./2.)*0.0     +0.0+(1./2.)*v[i+  s]) / dv;}; };
template< typename T> class FD<T, 1, 2,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./2.)*v[i-  s]+0.0+(1./2.)*v[i+  s]) / dv;}; };
template< typename T> class FD<T, 1, 2,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./2.)*v[i-  s]+0.0+(1./2.)*0.0     ) / dv;}; };

//                  DERIV=1  ACC=4
template< typename T> class FD<T, 1, 4, -2> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./12.)*0.0     +(-2./3.)*0.0     +0.0+(2./3.)*v[i+  s]+(-1./12.)*v[i+2*s]) / dv;}; };
template< typename T> class FD<T, 1, 4, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./12.)*0.0     +(-2./3.)*v[i-  s]+0.0+(2./3.)*v[i+  s]+(-1./12.)*v[i+2*s]) / dv;}; };
template< typename T> class FD<T, 1, 4,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./12.)*v[i-2*s]+(-2./3.)*v[i-  s]+0.0+(2./3.)*v[i+  s]+(-1./12.)*v[i+2*s]) / dv;}; };
template< typename T> class FD<T, 1, 4,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./12.)*v[i-2*s]+(-2./3.)*v[i-  s]+0.0+(2./3.)*v[i+  s]+(-1./12.)*0.0     ) / dv;}; };
template< typename T> class FD<T, 1, 4,  2> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./12.)*v[i-2*s]+(-2./3.)*v[i-  s]+0.0+(2./3.)*0.0     +(-1./12.)*0.0     ) / dv;}; };

//                  DERIV=1  ACC=6
template< typename T> class FD<T, 1, 6, -3 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*0.0     +(3./20.)*0.0     +(-3./4.)*0.0     +0.0+(3./4.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./60.)*v[i+3*s]) / dv;}; };
template< typename T> class FD<T, 1, 6, -2 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*0.0     +(3./20.)*0.0     +(-3./4.)*v[i-  s]+0.0+(3./4.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./60.)*v[i+3*s]) / dv;}; };
template< typename T> class FD<T, 1, 6, -1 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*0.0     +(3./20.)*v[i-2*s]+(-3./4.)*v[i-  s]+0.0+(3./4.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./60.)*v[i+3*s]) / dv;}; };
template< typename T> class FD<T, 1, 6,  0 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*v[i-3*s]+(3./20.)*v[i-2*s]+(-3./4.)*v[i-  s]+0.0+(3./4.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./60.)*v[i+3*s]) / dv;}; };
template< typename T> class FD<T, 1, 6,  1 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*v[i-3*s]+(3./20.)*v[i-2*s]+(-3./4.)*v[i-  s]+0.0+(3./4.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./60.)*0.0     ) / dv;}; };
template< typename T> class FD<T, 1, 6,  2 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*v[i-3*s]+(3./20.)*v[i-2*s]+(-3./4.)*v[i-  s]+0.0+(3./4.)*v[i+  s]+(-3./20.)*0.0     +(1./60.)*0.0     ) / dv;}; };
template< typename T> class FD<T, 1, 6,  3 > {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((-1./60.)*v[i-3*s]+(3./20.)*v[i-2*s]+(-3./4.)*v[i-  s]+0.0+(3./4.)*0.0     +(-3./20.)*0.0     +(1./60.)*0.0     ) / dv;}; };

//                  DERIV=1  ACC=8
template< typename T> class FD<T, 1, 8, -4> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*0.0     +(-4./105.)*0.0     +(1./5.)*0.0     +(-4./5.)*0.0     +0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*v[i+3*s]+(-1./280.)*v[i+4*s]) / dv;}; };
template< typename T> class FD<T, 1, 8, -3> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*0.0     +(-4./105.)*0.0     +(1./5.)*0.0     +(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*v[i+3*s]+(-1./280.)*v[i+4*s]) / dv;}; };
template< typename T> class FD<T, 1, 8, -2> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*0.0     +(-4./105.)*0.0     +(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*v[i+3*s]+(-1./280.)*v[i+4*s]) / dv;}; };
template< typename T> class FD<T, 1, 8, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*0.0     +(-4./105.)*v[i-3*s]+(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*v[i+3*s]+(-1./280.)*v[i+4*s]) / dv;}; };
template< typename T> class FD<T, 1, 8,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*v[i-4*s]+(-4./105.)*v[i-3*s]+(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*v[i+3*s]+(-1./280.)*v[i+4*s]) / dv;}; };
template< typename T> class FD<T, 1, 8,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*v[i-4*s]+(-4./105.)*v[i-3*s]+(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*v[i+3*s]+(-1./280.)*0.0     ) / dv;}; };
template< typename T> class FD<T, 1, 8,  2> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*v[i-4*s]+(-4./105.)*v[i-3*s]+(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(4./105.)*0.0     +(-1./280.)*0.0     ) / dv;}; };
template< typename T> class FD<T, 1, 8,  3> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*v[i-4*s]+(-4./105.)*v[i-3*s]+(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*v[i+  s]+(-1./5.)*0.0     +(4./105.)*0.0     +(-1./280.)*0.0     ) / dv;}; };
template< typename T> class FD<T, 1, 8,  4> {public: static T apply(T* v, int const& i, int const& s, T const& dv) { return ((1./280.)*v[i-4*s]+(-4./105.)*v[i-3*s]+(1./5.)*v[i-2*s]+(-4./5.)*v[i-  s]+0.0+(4./5.)*0.0     +(-1./5.)*0.0     +(4./105.)*0.0     +(-1./280.)*0.0     ) / dv;}; };

//                  DERIV=2  ACC=2
template< typename T> class FD<T, 2, 2, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return (1.0*0.0     -2.0*v[i]+1.0*v[i+  s]) / dv2;}; };
template< typename T> class FD<T, 2, 2,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return (1.0*v[i-  s]-2.0*v[i]+1.0*v[i+  s]) / dv2;}; };
template< typename T> class FD<T, 2, 2,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return (1.0*v[i-  s]-2.0*v[i]+1.0*0.0     ) / dv2;}; };

//                  DERIV=2  ACC=4
template< typename T> class FD<T, 2, 4, -2> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./12.)*0.0     +(4./3.)*0.0     +(-5./2.)*v[i]+(4./3.)*v[i+  s]+(-1./12.)*v[i+2*s]) / dv2;}; };
template< typename T> class FD<T, 2, 4, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./12.)*0.0     +(4./3.)*v[i-  s]+(-5./2.)*v[i]+(4./3.)*v[i+  s]+(-1./12.)*v[i+2*s]) / dv2;}; };
template< typename T> class FD<T, 2, 4,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./12.)*v[i-2*s]+(4./3.)*v[i-  s]+(-5./2.)*v[i]+(4./3.)*v[i+  s]+(-1./12.)*v[i+2*s]) / dv2;}; };
template< typename T> class FD<T, 2, 4,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./12.)*v[i-2*s]+(4./3.)*v[i-  s]+(-5./2.)*v[i]+(4./3.)*v[i+  s]+(-1./12.)*0.0     ) / dv2;}; };
template< typename T> class FD<T, 2, 4,  2> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./12.)*v[i-2*s]+(4./3.)*v[i-  s]+(-5./2.)*v[i]+(4./3.)*0.0     +(-1./12.)*0.0     ) / dv2;}; };

//                  DERIV=2  ACC=6
template< typename T> class FD<T, 2, 6, -3> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*0.0     +(-3./20.)*0.0     +(3./2.)*0.0     +(-49./18.)*v[i]+(3./2.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./90.)*v[i+3*s]) / dv2;}; };
template< typename T> class FD<T, 2, 6, -2> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*0.0     +(-3./20.)*0.0     +(3./2.)*v[i-  s]+(-49./18.)*v[i]+(3./2.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./90.)*v[i+3*s]) / dv2;}; };
template< typename T> class FD<T, 2, 6, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*0.0     +(-3./20.)*v[i-2*s]+(3./2.)*v[i-  s]+(-49./18.)*v[i]+(3./2.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./90.)*v[i+3*s]) / dv2;}; };
template< typename T> class FD<T, 2, 6,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*v[i-3*s]+(-3./20.)*v[i-2*s]+(3./2.)*v[i-  s]+(-49./18.)*v[i]+(3./2.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./90.)*v[i+3*s]) / dv2;}; };
template< typename T> class FD<T, 2, 6,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*v[i-3*s]+(-3./20.)*v[i-2*s]+(3./2.)*v[i-  s]+(-49./18.)*v[i]+(3./2.)*v[i+  s]+(-3./20.)*v[i+2*s]+(1./90.)*0.0     ) / dv2;}; };
template< typename T> class FD<T, 2, 6,  2> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*v[i-3*s]+(-3./20.)*v[i-2*s]+(3./2.)*v[i-  s]+(-49./18.)*v[i]+(3./2.)*v[i+  s]+(-3./20.)*0.0     +(1./90.)*0.0     ) / dv2;}; };
template< typename T> class FD<T, 2, 6,  3> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((1./90.)*v[i-3*s]+(-3./20.)*v[i-2*s]+(3./2.)*v[i-  s]+(-49./18.)*v[i]+(3./2.)*0.0     +(-3./20.)*0.0     +(1./90.)*0.0     ) / dv2;}; };

//                  DERIV=2  ACC=8
template< typename T> class FD<T, 2, 8, -4> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*0.0     +(8./315.)*0.0     +(-1./5.)*0.0     +(8./5.)*0.0     +(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*v[i+3*s]+(-1./560.)*v[i+4*s]) / dv2;}; };
template< typename T> class FD<T, 2, 8, -3> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*0.0     +(8./315.)*0.0     +(-1./5.)*0.0     +(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*v[i+3*s]+(-1./560.)*v[i+4*s]) / dv2;}; };
template< typename T> class FD<T, 2, 8, -2> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*0.0     +(8./315.)*0.0     +(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*v[i+3*s]+(-1./560.)*v[i+4*s]) / dv2;}; };
template< typename T> class FD<T, 2, 8, -1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*0.0     +(8./315.)*v[i-3*s]+(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*v[i+3*s]+(-1./560.)*v[i+4*s]) / dv2;}; };
template< typename T> class FD<T, 2, 8,  0> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*v[i-4*s]+(8./315.)*v[i-3*s]+(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*v[i+3*s]+(-1./560.)*v[i+4*s]) / dv2;}; };
template< typename T> class FD<T, 2, 8,  1> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*v[i-4*s]+(8./315.)*v[i-3*s]+(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*v[i+3*s]+(-1./560.)*0.0     ) / dv2;}; };
template< typename T> class FD<T, 2, 8,  2> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*v[i-4*s]+(8./315.)*v[i-3*s]+(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*v[i+2*s]+(8./315.)*0.0     +(-1./560.)*0.0     ) / dv2;}; };
template< typename T> class FD<T, 2, 8,  3> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*v[i-4*s]+(8./315.)*v[i-3*s]+(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*v[i+  s]+(-1./5.)*0.0     +(8./315.)*0.0     +(-1./560.)*0.0     ) / dv2;}; };
template< typename T> class FD<T, 2, 8,  4> {public: static T apply(T* v, int const& i, int const& s, T const& dv2) { return ((-1./560.)*v[i-4*s]+(8./315.)*v[i-3*s]+(-1./5.)*v[i-2*s]+(8./5.)*v[i-  s]+(-205./72.)*v[i]+(8./5.)*0.0     +(-1./5.)*0.0     +(8./315.)*0.0     +(-1./560.)*0.0     ) / dv2;}; };

#endif
