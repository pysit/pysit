%module constant_density_acoustic_time_scalar_cpp

%{
#define SWIG_FILE_WITH_INIT
%}

%include "../../../numpy.i"
%init %{
import_array();
%}

%include "constant_density_acoustic_time_scalar.i"
# %include "constant_density_acoustic_time_ode.i"
# %include "constant_density_acoustic_time_vector.i"

