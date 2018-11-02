#ifndef __CDA_TIME_SCALAR_1D__
#define __CDA_TIME_SCALAR_1D__

#include <iostream>

#include "fd_manual.hpp"

template <typename T>
class FDArg1D
{
    public:
        T* U;
        T* Phi;

        T delta;
        T delta2;

        T& dU;
        T& dPhi;
        T& lapU;

        FDArg1D<T>(T* u, T* pPhi, T& du, T& dP, T& lapu, T const& del) : U(u),Phi(pPhi),dU(du),dPhi(dP),lapU(lapu),delta(del){delta2 = delta*delta;};

        void reset()
        {
            lapU = 0.0;
        };
};

template <typename T, int ACCURACY, int SHIFT>
class FDAction1D
{
    public:
        static
        void execute(FDArg1D<T>& arg, int const& idx, int const& stride)
        {
            arg.dU    = FD<T,1,ACCURACY,SHIFT>::apply(arg.U, idx, stride, arg.delta);
            arg.dPhi  = FD<T,1,ACCURACY,SHIFT>::apply(arg.Phi, idx, stride, arg.delta);
            arg.lapU += FD<T,2,ACCURACY,SHIFT>::apply(arg.U, idx, stride, arg.delta2);
        };
};


template< typename T, int ACCURACY >
void cda_time_scalar_1D(      T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                              T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              double const& dt,                            // in
                              double const& dz,                            // in
                              int const& nz,                               // in
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    enum {MAX_FD_SHIFT = ACCURACY/2};

    // Derivative variables
    T dUdz = 0.0;
    T dPhizdz = 0.0;
    T lapU = 0.0;

    // PML variable
    T sigmaz = 0.0;

    // Time delta variables
    T dt2 = dt*dt;

    // Loop/index variables
    int idx;
    int zstride=1;

    // Wrapper class for the arrays and references to the derivative variables.
    FDArg1D<T> z_args(k_u, k_Phiz, dUdz, dPhizdz, lapU, dz);

    for(int k=0; k < nz; k++)
    {
        idx = k;
        kp1_Phiz[idx] = 0.0;
        kp1_u[idx]    = 0.0;

        // Don't compute updates in the padded region.
        // Keep looping over so that the indices match the data array nicely.
        // This handles homogeneous Dirichlet BCs and non-updating in ghost regions.
        if ((k == 0) || (k == nz-1)) continue;

        // Reset the derivative values
        z_args.reset();

        // Do the Z direction
        // Left side
        if     ((ACCURACY >= 2) && (k == 0)) FDAction1D<T, ACCURACY, -MAX_FD_SHIFT+0>::execute(z_args, idx, zstride);
        else if((ACCURACY >= 4) && (k == 1)) FDAction1D<T, ACCURACY, -MAX_FD_SHIFT+1>::execute(z_args, idx, zstride);
        else if((ACCURACY >= 6) && (k == 2)) FDAction1D<T, ACCURACY, -MAX_FD_SHIFT+2>::execute(z_args, idx, zstride);
        else if((ACCURACY >= 8) && (k == 3)) FDAction1D<T, ACCURACY, -MAX_FD_SHIFT+3>::execute(z_args, idx, zstride);
        // Right side
        else if((ACCURACY >= 2) && (k == nz-1)) FDAction1D<T, ACCURACY, MAX_FD_SHIFT-0>::execute(z_args, idx, zstride);
        else if((ACCURACY >= 4) && (k == nz-2)) FDAction1D<T, ACCURACY, MAX_FD_SHIFT-1>::execute(z_args, idx, zstride);
        else if((ACCURACY >= 6) && (k == nz-3)) FDAction1D<T, ACCURACY, MAX_FD_SHIFT-2>::execute(z_args, idx, zstride);
        else if((ACCURACY >= 8) && (k == nz-4)) FDAction1D<T, ACCURACY, MAX_FD_SHIFT-3>::execute(z_args, idx, zstride);
        // Bulk
        else FDAction1D<T, ACCURACY, 0>::execute(z_args, idx, zstride);

        sigmaz = 0.0;
        // Check if in left PML
        if((n_zlpml>0) && (k < n_zlpml))
        {
            sigmaz = zlpml[k];
        }
        // Check if in right PML
        else if((n_zrpml>0) && (k >= nz-n_zrpml))
        {
            sigmaz = zrpml[n_zrpml-((nz-1)-k)]; // 0th element of the right pml array corresponds to n_zrpml'th node from the right boundary.
        }
        if(sigmaz != 0.0)
        {
            kp1_Phiz[idx] = k_Phiz[idx] - dt * sigmaz*(k_Phiz[idx] + dUdz);

            T fac1 = (2.0*dt2 / (2.0 + dt*sigmaz));
            T fac2 = (C[idx]*C[idx])*(rhs[idx]+lapU+dPhizdz) - (km1_u[idx]-2.0*k_u[idx])/dt2 + sigmaz*km1_u[idx]/(2.0*dt);
            kp1_u[idx] = fac1 * fac2;
        }
        else
        {
            kp1_Phiz[idx] =  k_Phiz[idx];
            kp1_u[idx] = dt2*(C[idx]*C[idx])*(rhs[idx]+lapU+dPhizdz) - (km1_u[idx]-2.0*k_u[idx]);
        }

    }
};


#endif
