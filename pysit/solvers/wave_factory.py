from .constant_density_acoustic import(ConstantDensityAcousticTimeODE_1D,
                                      ConstantDensityAcousticTimeODE_2D,
                                      ConstantDensityAcousticTimeODE_3D,
                                      ConstantDensityAcousticTimeScalar_1D,
                                      ConstantDensityAcousticTimeScalar_2D,
                                      ConstantDensityAcousticTimeScalar_3D,
                                      )

__all__=['ConstantDensityAcousticWave']

__docformat__ = "restructuredtext en"

class ConstantDensityAcousticWave(object):
	defined_solvers = { 'structured-cartesian': {
	                        'vector': {
	                        	      # dimension
#	                                  1: ConstantDensityAcousticTimeVector_1D,
#	                                  2: ConstantDensityAcousticTimeVector_2D,
#	                                  3: ConstantDensityAcousticTimeVector_3D,
	                        	},
	                        
	                        'scalar': {
	                        	      # dimension
	                                  1: ConstantDensityAcousticTimeScalar_1D,
	                                  2: ConstantDensityAcousticTimeScalar_2D,
	                                  3: ConstantDensityAcousticTimeScalar_3D,
	                        	},
	                        
	                        'ode': {
	                        	      # dimension
	                                  1: ConstantDensityAcousticTimeODE_1D,
	                                  2: ConstantDensityAcousticTimeODE_2D,
	                                  3: ConstantDensityAcousticTimeODE_3D,
	                        	},
	                     },
	                  }
	def __new__(cls, mesh, formulation=None, *args, **kwargs):
		if cls is ConstantDensityAcousticWave:
			try:
				solver = ConstantDensityAcousticWave.defined_solvers[mesh.type][formulation][mesh.dim]
			except KeyError as e:
				print "Solver for '{0}'D constant density acoustic wave equation for the '{1}' formulation system does not exist for '{2}' type domains.".format(mesh.dim, formulation, mesh.type)
				raise e
			return solver(mesh, *args, **kwargs)
#			return super(ConstantDensityAcousticWave, cls).__new__(solver, mesh, *args, **kwargs)
		else:
			return super(ConstantDensityAcousticWave, cls).__new__(cls, mesh, *args, **kwargs)

	def __init__(self, mesh, *args, **kwargs):
		raise NotImplementedError("ConstantDensityAcousticWave factory __init__ should never be called.")

