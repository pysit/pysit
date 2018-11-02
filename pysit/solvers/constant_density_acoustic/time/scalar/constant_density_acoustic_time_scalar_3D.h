#ifndef __CDA_TIME_SCALAR_3D__
#define __CDA_TIME_SCALAR_3D__

#include <iostream>

#include "fd_manual.hpp"

template <typename T>
class FDArg3D
{
    public:
        T* U;
        T* Phi;
        T* psi;

        T delta;
        T delta2;

        T& dU;
        T& dPhi;
        T& dpsi;
        T& lapU;

        FDArg3D<T>(T* u, T* pPhi, T* ppsi, T& du, T& dP, T& dp, T& lapu, T const& del) : U(u),Phi(pPhi),psi(ppsi),dU(du),dPhi(dP),dpsi(dp),lapU(lapu),delta(del){delta2 = delta*delta;};

        void reset()
        {
            lapU = 0.0;
        };
};

template <typename T, int ACCURACY, int SHIFT>
class FDAction3D
{
    public:
        static
        void execute(FDArg3D<T>& arg, int const& idx, int const& stride)
        {
            arg.dU    = FD<T,1,ACCURACY,SHIFT>::apply(arg.U, idx, stride, arg.delta);
            arg.dPhi  = FD<T,1,ACCURACY,SHIFT>::apply(arg.Phi, idx, stride, arg.delta);
            arg.dpsi  = FD<T,1,ACCURACY,SHIFT>::apply(arg.psi, idx, stride, arg.delta);
            arg.lapU += FD<T,2,ACCURACY,SHIFT>::apply(arg.U, idx, stride, arg.delta2);
        };
};

template< typename T, int ACCURACY >
void cda_time_scalar_3D(      T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                              T* k_Phiy, int nr_k_Phiy, int nc_k_Phiy,     // in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                              T* k_psi,  int nr_k_psi,  int nc_k_psi,      // in
                              T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* ylpml,  int n_ylpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* yrpml,  int n_yrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              double const& dt,                            // in
                              double const& dx,                            // in
                              double const& dy,                            // in
                              double const& dz,                            // in
                              int const& nx,                               // in
                              int const& ny,                               // in
                              int const& nz,                               // in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                              T* kp1_Phiy, int nr_kp1_Phiy,  int nc_kp1_Phiy,  // out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                              T* kp1_psi,  int nr_kp1_psi,   int nc_kp1_psi,   // out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    enum {MAX_FD_SHIFT = ACCURACY/2};

    // Derivative variables
    T dUdx = 0.0;
    T dUdy = 0.0;
    T dUdz = 0.0;
    T dPhixdx = 0.0;
    T dPhiydy = 0.0;
    T dPhizdz = 0.0;
    T dpsidx = 0.0;
    T dpsidy = 0.0;
    T dpsidz = 0.0;
    T lapU = 0.0;

    // PML variable
    T sigmax = 0.0;
    T sigmay = 0.0;
    T sigmaz = 0.0;

    // Time delta variables
    T dt2 = dt*dt;

    // Loop/index variables
    int idx;
    int xstride = nz*ny;
    int ystride = nz;
    int zstride = 1;

    // Wrapper class for the arrays and references to the derivative variables.
    FDArg3D<T> x_args(k_u, k_Phix, k_psi, dUdx, dPhixdx, dpsidx, lapU, dx);
    FDArg3D<T> y_args(k_u, k_Phiy, k_psi, dUdy, dPhiydy, dpsidy, lapU, dy);
    FDArg3D<T> z_args(k_u, k_Phiz, k_psi, dUdz, dPhizdz, dpsidz, lapU, dz);

    for(int i=0; i < nx; ++i)
    {
        for(int j=0; j < ny; ++j)
        {
            for(int k=0; k < nz; k++)
            {
                idx = i*xstride + j*ystride + k;

                kp1_u[idx]    = 0.0;
                kp1_Phix[idx] = 0.0;
                kp1_Phiy[idx] = 0.0;
                kp1_Phiz[idx] = 0.0;
                kp1_psi[idx] = 0.0;

                // This handles homogeneous Dirichlet BCs and non-updating in ghost regions.
                if ((i == 0) || (i == nx-1)) continue;
                if ((j == 0) || (j == ny-1)) continue;
                if ((k == 0) || (k == nz-1)) continue;

                // Reset the derivative values
                x_args.reset();
                y_args.reset();
                z_args.reset();

                // Do the X direction
                // Left side
                if     ((ACCURACY >= 2) && (i == 0)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+0>::execute(x_args, idx, xstride);
                else if((ACCURACY >= 4) && (i == 1)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+1>::execute(x_args, idx, xstride);
                else if((ACCURACY >= 6) && (i == 2)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+2>::execute(x_args, idx, xstride);
                else if((ACCURACY >= 8) && (i == 3)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+3>::execute(x_args, idx, xstride);
                // Right side
                else if((ACCURACY >= 2) && (i == nx-1)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-0>::execute(x_args, idx, xstride);
                else if((ACCURACY >= 4) && (i == nx-2)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-1>::execute(x_args, idx, xstride);
                else if((ACCURACY >= 6) && (i == nx-3)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-2>::execute(x_args, idx, xstride);
                else if((ACCURACY >= 8) && (i == nx-4)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-3>::execute(x_args, idx, xstride);
                // Bulk
                else FDAction3D<T, ACCURACY, 0>::execute(x_args, idx, xstride);

                // Do the Y direction
                // Left side
                if     ((ACCURACY >= 2) && (j == 0)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+0>::execute(y_args, idx, ystride);
                else if((ACCURACY >= 4) && (j == 1)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+1>::execute(y_args, idx, ystride);
                else if((ACCURACY >= 6) && (j == 2)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+2>::execute(y_args, idx, ystride);
                else if((ACCURACY >= 8) && (j == 3)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+3>::execute(y_args, idx, ystride);
                // Right side
                else if((ACCURACY >= 2) && (j == ny-1)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-0>::execute(y_args, idx, ystride);
                else if((ACCURACY >= 4) && (j == ny-2)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-1>::execute(y_args, idx, ystride);
                else if((ACCURACY >= 6) && (j == ny-3)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-2>::execute(y_args, idx, ystride);
                else if((ACCURACY >= 8) && (j == ny-4)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-3>::execute(y_args, idx, ystride);
                // Bulk
                else FDAction3D<T, ACCURACY, 0>::execute(y_args, idx, ystride);

                // Do the Z direction
                // Left side
                if     ((ACCURACY >= 2) && (k == 0)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+0>::execute(z_args, idx, zstride);
                else if((ACCURACY >= 4) && (k == 1)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+1>::execute(z_args, idx, zstride);
                else if((ACCURACY >= 6) && (k == 2)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+2>::execute(z_args, idx, zstride);
                else if((ACCURACY >= 8) && (k == 3)) FDAction3D<T, ACCURACY, -MAX_FD_SHIFT+3>::execute(z_args, idx, zstride);
                // Right side
                else if((ACCURACY >= 2) && (k == nz-1)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-0>::execute(z_args, idx, zstride);
                else if((ACCURACY >= 4) && (k == nz-2)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-1>::execute(z_args, idx, zstride);
                else if((ACCURACY >= 6) && (k == nz-3)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-2>::execute(z_args, idx, zstride);
                else if((ACCURACY >= 8) && (k == nz-4)) FDAction3D<T, ACCURACY, MAX_FD_SHIFT-3>::execute(z_args, idx, zstride);
                // Bulk
                else FDAction3D<T, ACCURACY, 0>::execute(z_args, idx, zstride);

                sigmax = 0.0;
                sigmay = 0.0;
                sigmaz = 0.0;

                // Check if in left PML-X
                if((n_xlpml>0) && (i < n_xlpml))
                {
                    sigmax = xlpml[i];
                }
                // Check if in right PML-X
                else if((n_xrpml>0) && (i >= nx-n_xrpml))
                {
                    sigmax = xrpml[n_xrpml-((nx-1)-i)];
                }

                // Check if in left PML-Y
                if((n_ylpml>0) && (j < n_ylpml))
                {
                    sigmay = ylpml[j];
                }
                // Check if in right PML-Y
                else if((n_yrpml>0) && (j >= ny-n_yrpml))
                {
                    sigmay = yrpml[n_yrpml-((ny-1)-j)];
                }

                // Check if in left PML-Z
                if((n_zlpml>0) && (k < n_zlpml))
                {
                    sigmaz = zlpml[k];
                }
                // Check if in right PML-Z
                else if((n_zrpml>0) && (k >= nz-n_zrpml))
                {
                    sigmaz = zrpml[n_zrpml-((nz-1)-k)]; // 0th element of the right pml array corresponds to n_zrpml'th node from the right boundary.
                }

                if((sigmaz != 0.0) || (sigmay != 0.0) || (sigmax != 0.0))
                {
                    kp1_Phix[idx] = k_Phix[idx] - dt*sigmax*k_Phix[idx] + dt*(sigmay+sigmaz-sigmax)*dUdx + dt*(sigmay*sigmaz*dpsidx);
                    kp1_Phiy[idx] = k_Phiy[idx] - dt*sigmay*k_Phiy[idx] + dt*(sigmaz+sigmax-sigmay)*dUdy + dt*(sigmaz*sigmax*dpsidy);
                    kp1_Phiz[idx] = k_Phiz[idx] - dt*sigmaz*k_Phiz[idx] + dt*(sigmax+sigmay-sigmaz)*dUdz + dt*(sigmax*sigmay*dpsidz);

                    kp1_psi[idx] = k_psi[idx] + dt * k_u[idx];

                    T fac1 = (2.0*dt2 / (2.0 + dt*(sigmax+sigmay+sigmaz)));
                    T fac2 = (C[idx]*C[idx]) * (rhs[idx] + lapU + dPhixdx + dPhiydy + dPhizdz - (sigmax*sigmay*sigmaz)*k_psi[idx])
                           - (km1_u[idx]-2.0*k_u[idx])/dt2
                           + (sigmax+sigmay+sigmaz)*km1_u[idx]/(2.0*dt)
                           - (sigmax*sigmay + sigmay*sigmaz + sigmaz*sigmax)*k_u[idx];
                    kp1_u[idx] = fac1 * fac2;
                }
                else
                {
                    kp1_Phix[idx] = k_Phix[idx];
                    kp1_Phiy[idx] = k_Phiy[idx];
                    kp1_Phiz[idx] = k_Phiz[idx];
                    kp1_psi[idx]  = k_psi[idx] + dt * k_u[idx];
                    kp1_u[idx] = dt2*(C[idx]*C[idx])*(rhs[idx]+lapU+dPhixdx+dPhiydy+dPhizdz) - (km1_u[idx]-2.0*k_u[idx]);
                }
            }
        }
    }
};


#endif
