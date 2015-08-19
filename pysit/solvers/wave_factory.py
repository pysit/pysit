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
from .variable_density_acoustic import(VariableDensityAcousticTimeODE_1D,
                                       VariableDensityAcousticTimeODE_2D,
                                       VariableDensityAcousticTimeODE_3D,
                                       VariableDensityAcousticTimeScalar_1D_numpy,
                                       VariableDensityAcousticTimeScalar_2D_numpy,
                                       VariableDensityAcousticTimeScalar_3D_numpy)

__all__ = ['ConstantDensityAcousticWave','VariableDensityAcousticWave']

__docformat__ = "restructuredtext en"


# Setup the constant density acoustic wave factory
class ConstantDensityAcousticWaveFactory(SolverFactory):

    supports = {'equation_physics': 'constant-density-acoustic',
                'equation_dynamics': 'time'}

# Setup the variable density acoustic wave factory
class VariableDensityAcousticWaveFactory(SolverFactory):

    supports = {'equation_physics': 'variable-density-acoustic',
                'equation_dynamics': 'time'}


ConstantDensityAcousticWave = ConstantDensityAcousticWaveFactory()
VariableDensityAcousticWave = VariableDensityAcousticWaveFactory()
# Partial matches are resolved in the order of registration.  Therefore, the
# default situation takes scalar, leapfrog, spatially fd, numpy kernels.  Other
# defaults (such as accuracy) are minimally specified in the constructor, if
# they are not specified to the factory call.

# Constant Density Register:
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_1D_numpy)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_1D_cpp)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_2D_numpy)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_2D_cpp)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_3D_numpy)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeScalar_3D_cpp)

ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeODE_1D)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeODE_2D)
ConstantDensityAcousticWave.register(ConstantDensityAcousticTimeODE_3D)

# Variable Density Register:
VariableDensityAcousticWave.register(VariableDensityAcousticTimeScalar_1D_numpy)
VariableDensityAcousticWave.register(VariableDensityAcousticTimeScalar_2D_numpy)
VariableDensityAcousticWave.register(VariableDensityAcousticTimeScalar_3D_numpy)

VariableDensityAcousticWave.register(VariableDensityAcousticTimeODE_1D)
VariableDensityAcousticWave.register(VariableDensityAcousticTimeODE_2D)
VariableDensityAcousticWave.register(VariableDensityAcousticTimeODE_3D)