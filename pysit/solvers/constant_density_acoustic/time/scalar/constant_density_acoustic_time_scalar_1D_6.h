#ifndef __CDA_TIME_SCALAR_1D_6__
#define __CDA_TIME_SCALAR_1D_6__

#include <stdlib.h>


template< typename T, int ACCURACY >
void cda_time_scalar_1D_6(    T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
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


    // PML variable
    T sigmaz = 0.0;

    // Time delta variables
    T dt2 = dt*dt;

    // Loop/index variables
    int idx;
    int zstride=1;
    int s = zstride;


    // Loop public variables
    T dv = dz;
    T dv2 = dz*dz;
    
    // Loop private variables
        // derivatives
    T dU;
    T dPhi;
    T lapU = 0.0;
        // non derivatives
    T fac1;
    T fac2;

    // assignin the NUMBER of threads
    char* NUM = getenv("OMP_NUM_THREADS");
    int Num_Th = atoi (NUM);
    
    #pragma omp parallel num_threads(Num_Th) private(dU, dPhi, lapU, sigmaz, idx, fac1, fac2) shared(dv, dv2, s, k_u,k_Phiz,kp1_Phiz, kp1_u, rhs, C, dt2, dt, km1_u, zlpml, n_zrpml)
    {
        #pragma omp for
        for(int k=0; k < nz; k++)
        {
            idx = k;
            kp1_Phiz[idx] = 0.0;
            kp1_u[idx]    = 0.0;

            if ((k == 0) || (k == nz-1)) continue;
            lapU = 0.0;

            if (k==0)
            {
                //decentered derivative 3 ranks on the right
                dU = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*0.0+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dv;
                dPhi = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*0.0+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dv;
                lapU += ((1./90.)*0.0+(-3./20.)*0.0+(3./2.)*0.0+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dv2;

            }
            else if (k == 1)
            {
                //decentered derivative 2 rank on the right
                dU = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dv;
                dPhi = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dv;
                lapU += ((1./90.)*0.0+(-3./20.)*0.0+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dv2;

            }
            else if (k == 2)
            {
                //decentered derivative 1 rank on the right
                dU = ((-1./60.)*0.0+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dv;
                dPhi = ((-1./60.)*0.0+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dv;
                lapU += ((1./90.)*0.0+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dv2;

            }
            else if (k == nz-1)
            {
                //decentered derivative 3 ranks on the left
                dU = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*0.0+(-3./20.)*0.0+(1./60.)*0.0)/dv;
                dPhi = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*0.0+(-3./20.)*0.0+(1./60.)*0.0)/dv;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*0.0+(-3./20.)*0.0+(1./90.)*0.0)/dv2;

            }
            else if (k == nz-2)
            {
                //decentered derivative 2 ranks on the left
                dU = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*0.0+(1./60.)*0.0)/dv;
                dPhi = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*0.0+(1./60.)*0.0)/dv;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*0.0+(1./90.)*0.0)/dv2;

            }
            else if (k == nz-3)
            {
                //decentered derivative 1 rank on the left
                dU = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*0.0)/dv;
                dPhi = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*0.0)/dv;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*0.0)/dv2;

            }
            else
            {
                //classic centered derivative
                dU = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dv;
                dPhi = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dv;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dv2;
            }

            sigmaz = 0.0;
            if((n_zlpml>0) && (k < n_zlpml))
            {
                sigmaz = zlpml[k];
            }
            else if((n_zrpml>0) && (k >= nz-n_zrpml))
            {
                sigmaz = zrpml[n_zrpml-((nz-1)-k)];
            }

            if(sigmaz != 0.0)
            {
                kp1_Phiz[idx] = k_Phiz[idx] - dt * sigmaz*(k_Phiz[idx] + dU);
                fac1 = (2.0*dt2 / (2.0 + dt*sigmaz));
                fac2 = (C[idx]*C[idx])*(rhs[idx]+lapU+dPhi) - (km1_u[idx]-2.0*k_u[idx])/dt2 + sigmaz*km1_u[idx]/(2.0*dt);
                kp1_u[idx] = fac1 * fac2;

             }
            else
            {
                kp1_Phiz[idx] =  k_Phiz[idx];
                kp1_u[idx] = dt2*(C[idx]*C[idx])*(rhs[idx]+lapU+dPhi) - (km1_u[idx]-2.0*k_u[idx]);
            }
          }
        }
};


#endif
