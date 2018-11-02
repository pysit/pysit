%module wave_solvers_cpp

%{
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"
%init %{
import_array();
%}

%include "wave_constant_density_acoustic_first_order.i"
%include "wave_constant_density_acoustic_second_order.i"
%include "wave_constant_density_acoustic_ode.i"

