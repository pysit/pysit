#ifndef __CDA_TIME_SCALAR_3D_4__
#define __CDA_TIME_SCALAR_3D_4__

#include <stdlib.h> 


template< typename T, int ACCURACY >
void cda_time_scalar_3D_4(      T* km1_u,  int nr_km1_u,  int nc_km1_u,      // in - padded wavefield shape
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
    int s = zstride;
    int i, k, j;

    // shared space step square variable
    T dx2 = dx*dx;
    T dz2 = dz*dz;
    T dy2 = dy*dy;
    
    // private variables
        //non derivatives 
    T fac1;
    T fac2;
        //derivatives
    T dux , duz, duy;
    T dPhix, dPhiz, dPhiy;
    T dPsix, dPsiz, dPsiy;

    char* NUM = getenv("OMP_NUM_THREADS");
    int Num_Th = atoi (NUM);

    #pragma omp parallel for private(sigmaz, sigmax, sigmay, i, k, j, idx, dux, duz, duy, dPhix, dPhiz, dPhiy, lapU, fac1, fac2, dPsix, dPsiy, dPsiz) shared(dx, dx2, dz, dz2, dy, dy2, xstride, zstride, ystride, kp1_Phix, kp1_Phiz, kp1_Phiy, k_Phix, k_Phiz, k_Phiy, n_zrpml, n_zlpml, n_yrpml, n_ylpml, n_xrpml, n_xlpml, xrpml, xlpml, zrpml, zlpml, yrpml, ylpml, s, rhs, C, dt, dt2, km1_u, k_u, kp1_u) num_threads(Num_Th) collapse(3)
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

                lapU = 0.0;

                // Do the X direction
                if (i==0)
            {
                //decentered derivative 2 ranks on the right
                dux = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_u[idx+xstride]+(-1./12.)*k_u[idx+2*xstride])/dx;
                dPsix = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_psi[idx+xstride]+(-1./12.)*k_psi[idx+2*xstride])/dx;
                dPhix = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_Phix[idx+xstride]+(-1./12.)*k_Phix[idx+2*xstride])/ dx;
                lapU += ((-1./12.)*0.0+(4./3.)*0.0+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+xstride]+(-1./12.)*k_u[idx+2*xstride])/ dx2;

            }
            else if (i == 1)
            {
                //decentered derivative 1 rank on the right
                dux = ((1./12.)*0.0 +(-2./3.)*k_u[idx-xstride]+0.0+(2./3.)*k_u[idx+xstride]+(-1./12.)*k_u[idx+2*xstride])/ dx;
                dPsix = ((1./12.)*0.0 +(-2./3.)*k_psi[idx-xstride]+0.0+(2./3.)*k_psi[idx+xstride]+(-1./12.)*k_psi[idx+2*xstride])/ dx;
                dPhix = ((1./12.)*0.0+(-2./3.)*k_Phix[idx-xstride]+0.0+(2./3.)*k_Phix[idx+xstride]+(-1./12.)*k_Phix[idx+2*xstride])/ dx;
                lapU += ((-1./12.)*0.0+(4./3.)*k_u[idx-xstride]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+xstride]+(-1./12.)*k_u[idx+2*xstride])/ dx2;
            // Right side
            }
            else if (i == nx-1)
            {
                //decentered derivative 2 ranks on the left
                dux = ((1./12.)*k_u[idx-2*xstride]+(-2./3.)*k_u[idx-xstride]+0.0+(2./3.)*0.0 +(-1./12.)*0.0)/ dx;
                dPsix = ((1./12.)*k_psi[idx-2*xstride]+(-2./3.)*k_psi[idx-xstride]+0.0+(2./3.)*0.0 +(-1./12.)*0.0)/ dx;
                dPhix = ((1./12.)*k_Phix[idx-2*xstride]+(-2./3.)*k_Phix[idx-xstride]+0.0+(2./3.)*0.0+(-1./12.)*0.0) / dx;
                lapU += ((-1./12.)*k_u[idx-2*xstride]+(4./3.)*k_u[idx-xstride]+(-5./2.)*k_u[idx]+(4./3.)*0.0+(-1./12.)*0.0)/ dx2;
            }
            else if (i == nx-2)
            {
                //decentered derivative 1 ranks on the left
                dux = ((1./12.)*k_u[idx-2*xstride]+(-2./3.)*k_u[idx-xstride]+0.0+(2./3.)*k_u[idx+xstride]+(-1./12.)*0.0)/ dx;
                dPsix = ((1./12.)*k_psi[idx-2*xstride]+(-2./3.)*k_psi[idx-xstride]+0.0+(2./3.)*k_psi[idx+xstride]+(-1./12.)*0.0)/ dx;
                dPhix = ((1./12.)*k_Phix[idx-2*xstride]+(-2./3.)*k_Phix[idx-xstride]+0.0+(2./3.)*k_Phix[idx+xstride]+(-1./12.)*0.0)/ dx;
                lapU += ((-1./12.)*k_u[idx-2*xstride]+(4./3.)*k_u[idx-xstride]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+xstride]+(-1./12.)*0.0)/ dx2;
            }
            else
            {
                //classic centered derivative
                dux = ((1./12.)*k_u[idx-2*xstride]+(-2./3.)*k_u[idx-xstride]+0.0+(2./3.)*k_u[idx+xstride]+(-1./12.)*k_u[idx+2*xstride])/ dx;
                dPsix = ((1./12.)*k_psi[idx-2*xstride]+(-2./3.)*k_psi[idx-xstride]+0.0+(2./3.)*k_psi[idx+xstride]+(-1./12.)*k_psi[idx+2*xstride])/ dx;
                dPhix = ((1./12.)*k_Phix[idx-2*xstride]+(-2./3.)*k_Phix[idx-xstride]+0.0+(2./3.)*k_Phix[idx+xstride]+(-1./12.)*k_Phix[idx+2*xstride])/ dx;
                lapU += ((-1./12.)*k_u[idx-2*xstride]+(4./3.)*k_u[idx-xstride]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+xstride]+(-1./12.)*k_u[idx+2*xstride])/ dx2;
            }

                // Do the Y direction
                if (j==0)
            {
                //decentered derivative 2 ranks on the right
                duy = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_u[idx+ystride]+(-1./12.)*k_u[idx+2*ystride])/dy;
                dPsiy = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_psi[idx+ystride]+(-1./12.)*k_psi[idx+2*ystride])/dy;
                dPhiy = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_Phiy[idx+ystride]+(-1./12.)*k_Phiy[idx+2*ystride])/ dy;
                lapU += ((-1./12.)*0.0+(4./3.)*0.0+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+ystride]+(-1./12.)*k_u[idx+2*ystride])/ dy2;

            }
            else if (j == 1)
            {
                //decentered derivative 1 rank on the right
                duy = ((1./12.)*0.0 +(-2./3.)*k_u[idx-ystride]+0.0+(2./3.)*k_u[idx+ystride]+(-1./12.)*k_u[idx+2*ystride])/ dy;
                dPsiy = ((1./12.)*0.0 +(-2./3.)*k_psi[idx-ystride]+0.0+(2./3.)*k_psi[idx+ystride]+(-1./12.)*k_psi[idx+2*ystride])/ dy;
                dPhiy = ((1./12.)*0.0+(-2./3.)*k_Phiy[idx-ystride]+0.0+(2./3.)*k_Phiy[idx+ystride]+(-1./12.)*k_Phiy[idx+2*ystride])/ dy;
                lapU += ((-1./12.)*0.0+(4./3.)*k_u[idx-ystride]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+ystride]+(-1./12.)*k_u[idx+2*ystride])/ dy2;
            // Right side
            }
            else if (j== nx-1)
            {
                //decentered derivative 2 ranks on the left
                duy = ((1./12.)*k_u[idx-2*ystride]+(-2./3.)*k_u[idx-ystride]+0.0+(2./3.)*0.0 +(-1./12.)*0.0)/ dy;
                dPsiy = ((1./12.)*k_psi[idx-2*ystride]+(-2./3.)*k_psi[idx-ystride]+0.0+(2./3.)*0.0 +(-1./12.)*0.0)/ dy;
                dPhiy = ((1./12.)*k_Phiy[idx-2*ystride]+(-2./3.)*k_Phiy[idx-ystride]+0.0+(2./3.)*0.0+(-1./12.)*0.0) / dy;
                lapU += ((-1./12.)*k_u[idx-2*ystride]+(4./3.)*k_u[idx-ystride]+(-5./2.)*k_u[idx]+(4./3.)*0.0+(-1./12.)*0.0)/ dy2;
            }
            else if (j == nx-2)
            {
                //decentered derivative 1 ranks on the left
                duy = ((1./12.)*k_u[idx-2*ystride]+(-2./3.)*k_u[idx-ystride]+0.0+(2./3.)*k_u[idx+ystride]+(-1./12.)*0.0)/ dy;
                dPsiy = ((1./12.)*k_psi[idx-2*ystride]+(-2./3.)*k_psi[idx-ystride]+0.0+(2./3.)*k_psi[idx+ystride]+(-1./12.)*0.0)/ dy;
                dPhiy = ((1./12.)*k_Phiy[idx-2*ystride]+(-2./3.)*k_Phiy[idx-ystride]+0.0+(2./3.)*k_Phiy[idx+ystride]+(-1./12.)*0.0)/ dy;
                lapU += ((-1./12.)*k_u[idx-2*ystride]+(4./3.)*k_u[idx-ystride]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+ystride]+(-1./12.)*0.0)/ dy2;
            }
            else
            {
                //classic centered derivative
                duy = ((1./12.)*k_u[idx-2*ystride]+(-2./3.)*k_u[idx-ystride]+0.0+(2./3.)*k_u[idx+ystride]+(-1./12.)*k_u[idx+2*ystride])/ dy;
                dPsiy = ((1./12.)*k_psi[idx-2*ystride]+(-2./3.)*k_psi[idx-ystride]+0.0+(2./3.)*k_psi[idx+ystride]+(-1./12.)*k_psi[idx+2*ystride])/ dy;
                dPhiy = ((1./12.)*k_Phiy[idx-2*ystride]+(-2./3.)*k_Phiy[idx-ystride]+0.0+(2./3.)*k_Phiy[idx+ystride]+(-1./12.)*k_Phiy[idx+2*ystride])/ dy;
                lapU += ((-1./12.)*k_u[idx-2*ystride]+(4./3.)*k_u[idx-ystride]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+ystride]+(-1./12.)*k_u[idx+2*ystride])/ dy2;
            }


                // Do the Z direction
                // Left side
            if (k==0)
            {
                //decentered derivative 2 ranks on the right
                duz = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_u[idx+s]+(-1./12.)*k_u[idx+2*s])/dz;
                dPsiz = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_psi[idx+s]+(-1./12.)*k_psi[idx+2*s])/dz;
                dPhiz = ((1./12.)*0.0+(-2./3.)*0.0+0.0+(2./3.)*k_Phiz[idx+s]+(-1./12.)*k_Phiz[idx+2*s])/ dz;
                lapU += ((-1./12.)*0.0+(4./3.)*0.0+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+s]+(-1./12.)*k_u[idx+2*s])/ dz2;

            }
            else if (k == 1)
            {
                //decentered derivative 1 rank on the right
                duz = ((1./12.)*0.0 +(-2./3.)*k_u[idx-s]+0.0+(2./3.)*k_u[idx+s]+(-1./12.)*k_u[idx+2*s])/ dz;
                dPsiz = ((1./12.)*0.0 +(-2./3.)*k_psi[idx-s]+0.0+(2./3.)*k_psi[idx+s]+(-1./12.)*k_psi[idx+2*s])/ dz;
                dPhiz = ((1./12.)*0.0+(-2./3.)*k_Phiz[idx-s]+0.0+(2./3.)*k_Phiz[idx+s]+(-1./12.)*k_Phiz[idx+2*s])/ dz;
                lapU += ((-1./12.)*0.0+(4./3.)*k_u[idx-s]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+s]+(-1./12.)*k_u[idx+2*s])/ dz2;

            }
            else if (k == nz-1)
            {
                //decentered derivative 2 ranks on the left
                duz = ((1./12.)*k_u[idx-2*s]+(-2./3.)*k_u[idx-s]+0.0+(2./3.)*0.0 +(-1./12.)*0.0)/ dz;
                dPsiz = ((1./12.)*k_psi[idx-2*s]+(-2./3.)*k_psi[idx-s]+0.0+(2./3.)*0.0 +(-1./12.)*0.0)/ dz;
                dPhiz = ((1./12.)*k_Phiz[idx-2*s]+(-2./3.)*k_Phiz[idx-s]+0.0+(2./3.)*0.0+(-1./12.)*0.0) / dz;
                lapU += ((-1./12.)*k_u[idx-2*s]+(4./3.)*k_u[idx-s]+(-5./2.)*k_u[idx]+(4./3.)*0.0+(-1./12.)*0.0)/ dz2;
            }
            else if (k == nz-2)
            {
                //decentered derivative 1 ranks on the left
                duz = ((1./12.)*k_u[idx-2*s]+(-2./3.)*k_u[idx-s]+0.0+(2./3.)*k_u[idx+s]+(-1./12.)*0.0)/ dz;
                dPsiz = ((1./12.)*k_psi[idx-2*s]+(-2./3.)*k_psi[idx-s]+0.0+(2./3.)*k_psi[idx+s]+(-1./12.)*0.0)/ dz;
                dPhiz = ((1./12.)*k_Phiz[idx-2*s]+(-2./3.)*k_Phiz[idx-s]+0.0+(2./3.)*k_Phiz[idx+s]+(-1./12.)*0.0)/ dz;
                lapU += ((-1./12.)*k_u[idx-2*s]+(4./3.)*k_u[idx-s]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+s]+(-1./12.)*0.0)/ dz2;
            }
            else
            {
                //classic centered derivative
                duz = ((1./12.)*k_u[idx-2*s]+(-2./3.)*k_u[idx-s]+0.0+(2./3.)*k_u[idx+s]+(-1./12.)*k_u[idx+2*s])/ dz;
                dPsiz = ((1./12.)*k_psi[idx-2*s]+(-2./3.)*k_psi[idx-s]+0.0+(2./3.)*k_psi[idx+s]+(-1./12.)*k_psi[idx+2*s])/ dz;
                dPhiz = ((1./12.)*k_Phiz[idx-2*s]+(-2./3.)*k_Phiz[idx-s]+0.0+(2./3.)*k_Phiz[idx+s]+(-1./12.)*k_Phiz[idx+2*s])/ dz;
                lapU += ((-1./12.)*k_u[idx-2*s]+(4./3.)*k_u[idx-s]+(-5./2.)*k_u[idx]+(4./3.)*k_u[idx+s]+(-1./12.)*k_u[idx+2*s])/ dz2;
            }
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
                    kp1_Phix[idx] = k_Phix[idx] - dt*sigmax*k_Phix[idx] + dt*(sigmay+sigmaz-sigmax)*dux + dt*(sigmay*sigmaz*dPsix);
                    kp1_Phiy[idx] = k_Phiy[idx] - dt*sigmay*k_Phiy[idx] + dt*(sigmaz+sigmax-sigmay)*duy + dt*(sigmaz*sigmax*dPsiy);
                    kp1_Phiz[idx] = k_Phiz[idx] - dt*sigmaz*k_Phiz[idx] + dt*(sigmax+sigmay-sigmaz)*duz + dt*(sigmax*sigmay*dPsiz);

                    kp1_psi[idx] = k_psi[idx] + dt * k_u[idx];

                    fac1 = (2.0*dt2 / (2.0 + dt*(sigmax+sigmay+sigmaz)));
                    fac2 = (C[idx]*C[idx]) * (rhs[idx] + lapU + dPhix + dPhiy + dPhiz - (sigmax*sigmay*sigmaz)*k_psi[idx])
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
                    kp1_u[idx] = dt2*(C[idx]*C[idx])*(rhs[idx]+lapU+dPhix+dPhiy+dPhiz) - (km1_u[idx]-2.0*k_u[idx]);
                }
            }
        }
    }
};


#endif
