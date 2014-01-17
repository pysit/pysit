from .solver_factory import SolverFactory

from .constant_density_acoustic import(ConstantDensityAcousticFrequencyScalar_1D,
                                       ConstantDensityAcousticFrequencyScalar_2D,
                                       ConstantDensityAcousticFrequencyScalar_3D)

__all__ = ['ConstantDensityHelmholtz']

__docformat__ = "restructuredtext en"


# Setup the constant density acoustic wave factory
class ConstantDensityHelmholtzFactory(SolverFactory):

    supports = {'equation_physics': 'constant-density-acoustic',
                'equation_dynamics': 'time'}


ConstantDensityHelmholtz = ConstantDensityHelmholtzFactory()

# Partial matches are resolved in the order of registration.  Therefore, the
# default situation takes scalar, spatially fd solvers.  Other defaults (such
# as accuracy) are minimally specified in the constructor, if they are not
# specified to the factory call.

ConstantDensityHelmholtz.register(ConstantDensityAcousticFrequencyScalar_1D)
ConstantDensityHelmholtz.register(ConstantDensityAcousticFrequencyScalar_2D)
ConstantDensityHelmholtz.register(ConstantDensityAcousticFrequencyScalar_3D)
