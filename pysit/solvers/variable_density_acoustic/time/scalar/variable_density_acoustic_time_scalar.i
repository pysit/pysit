#ifndef __constant_density_acoustic_time_scalar_i__
#define __constant_density_acoustic_time_scalar_i__

%{
#include "constant_density_acoustic_time_scalar_1D.h"
#include "constant_density_acoustic_time_scalar_2D.h"
#include "constant_density_acoustic_time_scalar_3D.h"
%}

%define typemaps_constant_density_acoustic_scalar( DATA_TYPE )
%apply ( DATA_TYPE* IN_ARRAY2, int DIM1, int DIM2 ) {
    ( DATA_TYPE* km1_u,  int nr_km1_u,  int nc_km1_u  ),
	( DATA_TYPE* k_Phix, int nr_k_Phix, int nc_k_Phix ),
    ( DATA_TYPE* k_Phiy, int nr_k_Phiy, int nc_k_Phiy ),
    ( DATA_TYPE* k_Phiz, int nr_k_Phiz, int nc_k_Phiz ),
    ( DATA_TYPE* k_psi,  int nr_k_psi,  int nc_k_psi  ),
    ( DATA_TYPE* k_u,    int nr_k_u,    int nc_k_u    ),
    ( DATA_TYPE* C,      int nr_C,      int nc_C      ),
    ( DATA_TYPE* rhs,    int nr_rhs,    int nc_rhs    )
};
%apply ( DATA_TYPE* IN_ARRAY1, int DIM1 ) {
    ( DATA_TYPE* xlpml, int n_xlpml ),
	( DATA_TYPE* xrpml, int n_xrpml ),
    ( DATA_TYPE* ylpml, int n_ylpml ),
    ( DATA_TYPE* yrpml, int n_yrpml ),
    ( DATA_TYPE* zlpml, int n_zlpml ),
    ( DATA_TYPE* zrpml, int n_zrpml )
};
%apply ( DATA_TYPE* INPLACE_ARRAY2, int DIM1, int DIM2 ) {
	( DATA_TYPE* kp1_Phix, int nr_kp1_Phix, int nc_kp1_Phix ),
    ( DATA_TYPE* kp1_Phiy, int nr_kp1_Phiy, int nc_kp1_Phiy ),
    ( DATA_TYPE* kp1_Phiz, int nr_kp1_Phiz, int nc_kp1_Phiz ),
    ( DATA_TYPE* kp1_psi,  int nr_kp1_psi,  int nc_kp1_psi  ),
    ( DATA_TYPE* kp1_u,    int nr_kp1_u,    int nc_kp1_u    )
};
%enddef

typemaps_constant_density_acoustic_scalar( float  )
typemaps_constant_density_acoustic_scalar( double  )

%include "constant_density_acoustic_time_scalar_1D.h"
%include "constant_density_acoustic_time_scalar_2D.h"
%include "constant_density_acoustic_time_scalar_3D.h"

%define INSTANTIATE_CDA_SCALAR( out_fname, in_fname , order )
%template(out_fname) in_fname<float, order>;
%template(out_fname) in_fname<double, order>;
%enddef

# 1D
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_1D_2os, cda_time_scalar_1D, 2)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_1D_4os, cda_time_scalar_1D, 4)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_1D_6os, cda_time_scalar_1D, 6)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_1D_8os, cda_time_scalar_1D, 8)

# 2D
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_2D_2os, cda_time_scalar_2D, 2)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_2D_4os, cda_time_scalar_2D, 4)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_2D_6os, cda_time_scalar_2D, 6)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_2D_8os, cda_time_scalar_2D, 8)

# 3D
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_3D_2os, cda_time_scalar_3D, 2)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_3D_4os, cda_time_scalar_3D, 4)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_3D_6os, cda_time_scalar_3D, 6)
INSTANTIATE_CDA_SCALAR(constant_density_acoustic_time_scalar_3D_8os, cda_time_scalar_3D, 8)

#endif
