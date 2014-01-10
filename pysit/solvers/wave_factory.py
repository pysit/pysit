from .solver_factory import SolverFactory

from .constant_density_acoustic import(ConstantDensityAcousticTimeODE_1D,
                                       ConstantDensityAcousticTimeODE_2D,
                                       ConstantDensityAcousticTimeODE_3D,
                                       ConstantDensityAcousticTimeScalar_1D_numpy,
                                       ConstantDensityAcousticTimeScalar_1D_cpp,
                                       ConstantDensityAcousticTimeScalar_2D_numpy,
                                       ConstantDensityAcousticTimeScalar_2D_cpp,
                                       ConstantDensityAcousticTimeScalar_3D_numpy,
                                       ConstantDensityAcousticTimeScalar_3D_cpp)

__all__ = ['ConstantDensityAcousticWave']

__docformat__ = "restructuredtext en"


# Setup the constant density acoustic wave factory

class ConstantDensityAcousticWaveFactory(SolverFactory):

    supports_equation_physics = 'constant-density-acoustic'
    supports_equation_dynamics = 'time'

    # These are the arguments that users can specify.  Other validations are
    # initrinsic to the mesh and solver.
    default_kwargs = {'equation_formulation': 'scalar',
                      'temporal_integrator': 'leap-frog',
                      'temporal_accuracy_order': 2,
                      'spatial_discretization': 'finite-difference',
                      'spatial_accuracy_order': 2,
                      'kernel_implementation': 'numpy'}


ConstantDensityAcousticWave = ConstantDensityAcousticWaveFactory()

ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_1D_numpy)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_1D_cpp)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_2D_numpy)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_2D_cpp)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_3D_numpy)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_3D_cpp)

ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeODE_1D)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeODE_2D)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeODE_3D)
