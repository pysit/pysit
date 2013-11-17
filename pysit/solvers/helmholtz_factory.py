from __future__ import absolute_import
from .constant_density_acoustic import(ConstantDensityAcousticFrequencyScalar_1D,
                                      ConstantDensityAcousticFrequencyScalar_2D,
                                      ConstantDensityAcousticFrequencyScalar_3D,
                                      )

__all__=['ConstantDensityHelmholtz']

__docformat__ = "restructuredtext en"

class ConstantDensityHelmholtz(object):
	defined_solvers = { 'structured-cartesian': {
	                        # dimension
	                        1: ConstantDensityAcousticFrequencyScalar_1D,
	                        2: ConstantDensityAcousticFrequencyScalar_2D,
	                        3: ConstantDensityAcousticFrequencyScalar_3D,
	                     },
	                  }
	def __new__(cls, mesh, *args, **kwargs):
		if cls is ConstantDensityHelmholtz:
			try:
				solver = ConstantDensityHelmholtz.defined_solvers[mesh.type][mesh.dim]
			except KeyError as e:
				print "Solver for '{0}'D constant density Helmholtz equation does not exist for '{1}' type domains.".format(mesh.dim, mesh.discretization)
				raise e
#			return super(ConstantDensityHelmholtz, cls).__new__(solver, mesh)
			return solver(mesh, *args, **kwargs)
		else:
			return super(ConstantDensityHelmholtz, cls).__new__(cls, mesh, *args, **kwargs)

	def __init__(self, mesh, *args, **kwargs):
		raise NotImplementedError("ConstantDensityHelmholtz factory __init__ should never be called.")

