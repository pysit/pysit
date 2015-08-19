from .solver_factory import SolverFactory

from .constant_density_acoustic import(ConstantDensityAcousticFrequencyScalar_1D,
                                       ConstantDensityAcousticFrequencyScalar_2D,
                                       ConstantDensityAcousticFrequencyScalar_3D)

from .variable_density_acoustic import(VariableDensityAcousticFrequencyScalar_1D,
                                       VariableDensityAcousticFrequencyScalar_2D,
                                       VariableDensityAcousticFrequencyScalar_3D)
__all__ = ['ConstantDensityHelmholtz','VariableDensityHelmholtz']

__docformat__ = "restructuredtext en"


# Setup the constant density acoustic wave factory
class ConstantDensityHelmholtzFactory(SolverFactory):

    supports = {'equation_physics': 'constant-density-acoustic',
                'equation_dynamics': 'frequency'}

# Setup the variable density acoustic wave factory
class VariableDensityHelmholtzFactory(SolverFactory):

    supports = {'equation_physics': 'variable-density-acoustic',
                'equation_dynamics': 'frequency'}

ConstantDensityHelmholtz = ConstantDensityHelmholtzFactory()
VariableDensityHelmholtz = VariableDensityHelmholtzFactory()
# Partial matches are resolved in the order of registration.  Therefore, the
# default situation takes scalar, spatially fd solvers.  Other defaults (such
# as accuracy) are minimally specified in the constructor, if they are not
# specified to the factory call.

ConstantDensityHelmholtz.register(ConstantDensityAcousticFrequencyScalar_1D)
ConstantDensityHelmholtz.register(ConstantDensityAcousticFrequencyScalar_2D)
ConstantDensityHelmholtz.register(ConstantDensityAcousticFrequencyScalar_3D)

VariableDensityHelmholtz.register(VariableDensityAcousticFrequencyScalar_1D)
VariableDensityHelmholtz.register(VariableDensityAcousticFrequencyScalar_2D)
VariableDensityHelmholtz.register(VariableDensityAcousticFrequencyScalar_3D)