#ifndef __IF_UNROLL__
#define __IF_UNROLL__

template < typename T,
           template<typename AA, template<typename G> class A, int B, int C, int D, int E> class ACTION,
           template <typename F> class ACTIONARGUMENT,
           int ACCURACY,
           int BOUNDARYDISTANCE >
class IfUnrollLeft
{
    public:
        static
        void process(ACTIONARGUMENT<T>& arg, int const& idx_boundary_distance, int const& index, int const& stride, bool& processed)
        {
            enum {NPOINTS = ACCURACY + 1};
            enum {CENTER = ACCURACY/2};

            enum {END_FD_COEFF_INDEX = NPOINTS - 1};
            enum {START_FD_COEFF_INDEX = CENTER - (CENTER - BOUNDARYDISTANCE)};

            if (idx_boundary_distance == CENTER-BOUNDARYDISTANCE)
            {
                ACTION< T, ACTIONARGUMENT, NPOINTS, CENTER, START_FD_COEFF_INDEX, END_FD_COEFF_INDEX>::execute(arg, index, stride);
                processed = true;
            }
            else
            {
                IfUnrollLeft<T, ACTION, ACTIONARGUMENT, ACCURACY, BOUNDARYDISTANCE-1>::process(arg, idx_boundary_distance, index, stride, processed);
            }

        }
};

template < typename T,
           template<typename AA, template<typename G> class A, int B, int C, int D, int E> class ACTION,
           template <typename F> class ACTIONARGUMENT,
           int ACCURACY >
class IfUnrollLeft<T, ACTION, ACTIONARGUMENT, ACCURACY, 0>
{
    public:
        static
        void process(ACTIONARGUMENT<T>& arg, int const& idx_boundary_distance, int const& index, int const& stride, bool& processed)
        {
            enum {NPOINTS = ACCURACY + 1};
            enum {CENTER = ACCURACY/2};

            enum {END_FD_COEFF_INDEX = NPOINTS - 1};
            enum {START_FD_COEFF_INDEX = CENTER};

            if (idx_boundary_distance == 0)
            {
                ACTION< T, ACTIONARGUMENT, NPOINTS, CENTER, START_FD_COEFF_INDEX, END_FD_COEFF_INDEX>::execute(arg, index, stride);
                processed = true;
            }
            else
            {
                processed = false;
            }

        }
};


template < typename T,
           template<typename AA, template<typename G> class A, int B, int C, int D, int E> class ACTION,
           template <typename F> class ACTIONARGUMENT,
           int ACCURACY,
           int BOUNDARYDISTANCE >
class IfUnrollRight
{
    public:
        static
        void process(ACTIONARGUMENT<T>& arg, int const& idx_boundary_distance, const int& index, int const& stride, bool& processed)
        {
            enum {NPOINTS = ACCURACY + 1};
            enum {CENTER = ACCURACY/2};

            enum {END_FD_COEFF_INDEX = CENTER + BOUNDARYDISTANCE};
            enum {START_FD_COEFF_INDEX = 0};

            if (idx_boundary_distance == BOUNDARYDISTANCE)
            {
                ACTION< T, ACTIONARGUMENT, NPOINTS, CENTER, START_FD_COEFF_INDEX, END_FD_COEFF_INDEX>::execute(arg, index, stride);
                processed = true;
            }
            else
            {
                IfUnrollRight<T, ACTION, ACTIONARGUMENT, ACCURACY, BOUNDARYDISTANCE-1>::process(arg, idx_boundary_distance, index, stride, processed);
            }

        }
};

template < typename T,
           template<typename AA, template<typename G> class A, int B, int C, int D, int E> class ACTION,
           template <typename F> class ACTIONARGUMENT,
           int ACCURACY >
class IfUnrollRight<T, ACTION, ACTIONARGUMENT, ACCURACY, 0>
{
    public:
        static
        void process(ACTIONARGUMENT<T>& arg, int const& idx_boundary_distance, const int& index, int const& stride, bool& processed)
        {
            enum {NPOINTS = ACCURACY + 1};
            enum {CENTER = ACCURACY/2};

            enum {END_FD_COEFF_INDEX = CENTER};
            enum {START_FD_COEFF_INDEX = 0};

            if (idx_boundary_distance == 0)
            {
                ACTION< T, ACTIONARGUMENT, NPOINTS, CENTER, START_FD_COEFF_INDEX, END_FD_COEFF_INDEX>::execute(arg, index, stride);
                processed = true;
            }
            else
            {
                processed = false;
            }

        }
};


#endif
