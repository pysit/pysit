#ifndef __CDA_TIME_SCALAR_2D_6__
#define __CDA_TIME_SCALAR_2D_6__

#include <stdlib.h>


template< typename T, int ACCURACY >
void cda_time_scalar_2D_6(      T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
                              T* k_Phix, int nr_k_Phix, int nc_k_Phix,     // in - padded wavefield shape
                              T* k_Phiz, int nr_k_Phiz, int nc_k_Phiz,     // in - padded wavefield shape
                              T* k_u,    int nr_k_u,    int nc_k_u,        // in - padded wavefield shape
                              T* C,      int nr_C,      int nc_C,          // in - padded wavefield shape
                              T* rhs,    int nr_rhs,    int nc_rhs,        // in - padded wavefield shape
                              T* xlpml,  int n_xlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* xrpml,  int n_xrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zlpml,  int n_zlpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              T* zrpml,  int n_zrpml,                      // in - length is the number of nodes inside the padding that the pml value is defined.
                              double const& dt,                            // in
                              double const& dx,                            // in
                              double const& dz,                            // in
                              int const& nx,                               // in
                              int const& nz,                               // in
                              T* kp1_Phix, int nr_kp1_Phix,  int nc_kp1_Phix,  // out
                              T* kp1_Phiz, int nr_kp1_Phiz,  int nc_kp1_Phiz,  // out
                              T* kp1_u,    int nr_kp1_u,     int nc_kp1_u   )  // out
{
    enum {MAX_FD_SHIFT = ACCURACY/2};

    T lapU = 0.0;

    // PML variable
    T sigmax = 0.0;
    T sigmaz = 0.0;

    // Time delta variables
    T dt2 = dt*dt;

    // Loop/index variables
    int idx;
    int zstride = 1;
    int xstride = nz;
    int s = zstride;
    int i, k;

    // shared space step square variable
    T dx2 = dx*dx;
    T dz2 = dz*dz;
    
    // private variables
        //non derivatives 
    T fac1;
    T fac2;
        //derivatives
    T dux , duz;
    T dPhix, dPhiz;

    char* NUM = getenv("OMP_NUM_THREADS");
    int Num_Th = atoi (NUM);

    #pragma omp parallel for private(sigmaz, sigmax, i, k, idx, dux, duz, dPhix, dPhiz, lapU, fac1, fac2) shared(dx, dx2, dz, dz2, nz, nx, kp1_Phix, kp1_Phiz, k_Phix, k_Phiz, n_zrpml, n_zlpml, n_xrpml, xrpml, xlpml, zrpml, zlpml, s, rhs, C, dt, dt2, km1_u, k_u, kp1_u) num_threads(Num_Th) collapse(2) 
    for(i=0; i < nx; ++i)
    {        
        for(k=0; k < nz; k++)
        {
            idx = i*xstride + k;

            kp1_Phix[idx] = 0.0;
            kp1_Phiz[idx] = 0.0;
            kp1_u[idx]    = 0.0;

            // This handles homogeneous Dirichlet BCs and non-updating in ghost regions.
            if ((i == 0) || (i == nx-1)) continue;
            if ((k == 0) || (k == nz-1)) continue;
            lapU = 0.0;
            
            
            // Do the X direction
            // Left side
            if (i==0)
            {
                //decentered derivative 3 ranks on the right
                dux = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*0.0+0.0+(3./4.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./60.)*k_u[idx+3*nz])/dx;
                dPhix = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*0.0+0.0+(3./4.)*k_Phix[idx+nz]+(-3./20.)*k_Phix[idx+2*nz]+(1./60.)*k_Phix[idx+3*nz])/dx;
                lapU += ((1./90.)*0.0+(-3./20.)*0.0+(3./2.)*0.0+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./90.)*k_u[idx+3*nz])/dx2;

            }
            else if (i == 1)
            {
                //decentered derivative 2 rank on the right
                dux = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*k_u[idx-nz]+0.0+(3./4.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./60.)*k_u[idx+3*nz])/dx;
                dPhix = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*k_Phix[idx-nz]+0.0+(3./4.)*k_Phix[idx+nz]+(-3./20.)*k_Phix[idx+2*nz]+(1./60.)*k_Phix[idx+3*nz])/dx;
                lapU += ((1./90.)*0.0+(-3./20.)*0.0+(3./2.)*k_u[idx-nz]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./90.)*k_u[idx+3*nz])/dx2;
            }
            else if (i == 2)
            {
                //decentered derivative 1 rank on the right
                dux = ((-1./60.)*0.0+(3./20.)*k_u[idx-2*nz]+(-3./4.)*k_u[idx-nz]+0.0+(3./4.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./60.)*k_u[idx+3*nz])/dx;
                dPhix = ((-1./60.)*0.0+(3./20.)*k_Phix[idx-2*nz]+(-3./4.)*k_Phix[idx-nz]+0.0+(3./4.)*k_Phix[idx+nz]+(-3./20.)*k_Phix[idx+2*nz]+(1./60.)*k_Phix[idx+3*nz])/dx;
                lapU += ((1./90.)*0.0+(-3./20.)*k_u[idx-2*nz]+(3./2.)*k_u[idx-nz]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./90.)*k_u[idx+3*nz])/dx2;

            }
            else if (i == nx-1)
            {
                //decentered derivative 3 ranks on the left
                dux = ((-1./60.)*k_u[idx-3*nz]+(3./20.)*k_u[idx-2*nz]+(-3./4.)*k_u[idx-nz]+0.0+(3./4.)*0.0+(-3./20.)*0.0+(1./60.)*0.0)/dx;
                dPhix = ((-1./60.)*k_Phix[idx-3*nz]+(3./20.)*k_Phix[idx-2*nz]+(-3./4.)*k_Phix[idx-nz]+0.0+(3./4.)*0.0+(-3./20.)*0.0+(1./60.)*0.0)/dx;
                lapU += ((1./90.)*k_u[idx-3*nz]+(-3./20.)*k_u[idx-2*nz]+(3./2.)*k_u[idx-nz]+(-49./18.)*k_u[idx]+(3./2.)*0.0+(-3./20.)*0.0+(1./90.)*0.0)/dx2;
            }
            else if (i == nx-2)
            {
                //decentered derivative 2 ranks on the left
                dux = ((-1./60.)*k_u[idx-3*nz]+(3./20.)*k_u[idx-2*nz]+(-3./4.)*k_u[idx-nz]+0.0+(3./4.)*k_u[idx+nz]+(-3./20.)*0.0+(1./60.)*0.0)/dx;
                dPhix = ((-1./60.)*k_Phix[idx-3*nz]+(3./20.)*k_Phix[idx-2*nz]+(-3./4.)*k_Phix[idx-nz]+0.0+(3./4.)*k_Phix[idx+nz]+(-3./20.)*0.0+(1./60.)*0.0)/dx;
                lapU += ((1./90.)*k_u[idx-3*nz]+(-3./20.)*k_u[idx-2*nz]+(3./2.)*k_u[idx-nz]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+nz]+(-3./20.)*0.0+(1./90.)*0.0)/dx2;
            }
            else if (k == nx-3)
            {
                //decentered derivative 1 rank on the left
                dux = ((-1./60.)*k_u[idx-3*nz]+(3./20.)*k_u[idx-2*nz]+(-3./4.)*k_u[idx-nz]+0.0+(3./4.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./60.)*0.0)/dx;
                dPhix = ((-1./60.)*k_Phix[idx-3*nz]+(3./20.)*k_Phix[idx-2*nz]+(-3./4.)*k_Phix[idx-nz]+0.0+(3./4.)*k_Phix[idx+nz]+(-3./20.)*k_Phix[idx+2*nz]+(1./60.)*0.0)/dx;
                lapU += ((1./90.)*k_u[idx-3*nz]+(-3./20.)*k_u[idx-2*nz]+(3./2.)*k_u[idx-nz]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./90.)*0.0)/dx2;

            }
            else
            {
                //classic centered derivative
                dux = ((-1./60.)*k_u[idx-3*nz]+(3./20.)*k_u[idx-2*nz]+(-3./4.)*k_u[idx-nz]+0.0+(3./4.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./60.)*k_u[idx+3*nz])/dx;
                dPhix = ((-1./60.)*k_Phix[idx-3*nz]+(3./20.)*k_Phix[idx-2*nz]+(-3./4.)*k_Phix[idx-nz]+0.0+(3./4.)*k_Phix[idx+nz]+(-3./20.)*k_Phix[idx+2*nz]+(1./60.)*k_Phix[idx+3*nz])/dx;
                lapU += ((1./90.)*k_u[idx-3*nz]+(-3./20.)*k_u[idx-2*nz]+(3./2.)*k_u[idx-nz]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+nz]+(-3./20.)*k_u[idx+2*nz]+(1./90.)*k_u[idx+3*nz])/dx2;
            }

            // Do the Z direction
            // Left side
           
           if (k==0)
            {
                //decentered derivative 3 ranks on the right
                duz = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*0.0+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dz;
                dPhiz = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*0.0+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dz;
                lapU += ((1./90.)*0.0+(-3./20.)*0.0+(3./2.)*0.0+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dz2;
            }
            else if (k == 1)
            {
                //decentered derivative 2 rank on the right
                duz = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dz;
                dPhiz = ((-1./60.)*0.0+(3./20.)*0.0+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dz;
                lapU += ((1./90.)*0.0+(-3./20.)*0.0+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dz2;
            }
            else if (k == 2)
            {
                //decentered derivative 1 rank on the right
                duz = ((-1./60.)*0.0+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dz;
                dPhiz = ((-1./60.)*0.0+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dz;
                lapU += ((1./90.)*0.0+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dz2;
            }
            else if (k == nz-1)
            {
                //decentered derivative 3 ranks on the left
                duz = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*0.0+(-3./20.)*0.0+(1./60.)*0.0)/dz;
                dPhiz = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*0.0+(-3./20.)*0.0+(1./60.)*0.0)/dz;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*0.0+(-3./20.)*0.0+(1./90.)*0.0)/dz2;
            }
            else if (k == nz-2)
            {
                //decentered derivative 2 ranks on the left
                duz = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*0.0+(1./60.)*0.0)/dz;
                dPhiz = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*0.0+(1./60.)*0.0)/dz;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*0.0+(1./90.)*0.0)/dz2;
            }
            else if (k == nz-3)
            {
                //decentered derivative 1 rank on the left
                duz = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*0.0)/dz;
                dPhiz = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*0.0)/dz;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*0.0)/dz2;
            }
            else
            {
                //classic centered derivative
                duz = ((-1./60.)*k_u[idx-3*s]+(3./20.)*k_u[idx-2*s]+(-3./4.)*k_u[idx-s]+0.0+(3./4.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./60.)*k_u[idx+3*s])/dz;
                dPhiz = ((-1./60.)*k_Phiz[idx-3*s]+(3./20.)*k_Phiz[idx-2*s]+(-3./4.)*k_Phiz[idx-s]+0.0+(3./4.)*k_Phiz[idx+s]+(-3./20.)*k_Phiz[idx+2*s]+(1./60.)*k_Phiz[idx+3*s])/dz;
                lapU += ((1./90.)*k_u[idx-3*s]+(-3./20.)*k_u[idx-2*s]+(3./2.)*k_u[idx-s]+(-49./18.)*k_u[idx]+(3./2.)*k_u[idx+s]+(-3./20.)*k_u[idx+2*s]+(1./90.)*k_u[idx+3*s])/dz2;
            }

            sigmax = 0.0;
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

            if((sigmaz != 0.0) || (sigmax != 0.0))
            {
                kp1_Phix[idx] = k_Phix[idx] - dt*sigmax*k_Phix[idx] + dt*(sigmaz-sigmax)*dux;
                kp1_Phiz[idx] = k_Phiz[idx] - dt*sigmaz*k_Phiz[idx] + dt*(sigmax-sigmaz)*duz;


                fac1 = (2.0*dt2 / (2.0 + dt*(sigmax+sigmaz)));
                fac2 = (C[idx]*C[idx])*(rhs[idx]+lapU+dPhix+dPhiz) - (km1_u[idx]-2.0*k_u[idx])/dt2 + (sigmax+sigmaz)*km1_u[idx]/(2.0*dt) - (sigmax*sigmaz)*k_u[idx];
                kp1_u[idx] = fac1 * fac2;
            }
            else
            {
                kp1_Phix[idx] = k_Phix[idx];
                kp1_Phiz[idx] = k_Phiz[idx];
                kp1_u[idx] = dt2*(C[idx]*C[idx])*(rhs[idx]+lapU+dPhix+dPhiz) - (km1_u[idx]-2.0*k_u[idx]);
            }


        }
    }
};


#endif
