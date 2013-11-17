import numpy as np
import scipy.sparse as spsp

__all__ = ['build_sigma', 'make_diag_mtx']
	
def build_sigma(mesh, dim):
	s = np.zeros(mesh.shape(include_bc=True))
	if dim.lbc.type == 'pml':
		s += dim.lbc.eval_on_mesh()
	if dim.rbc.type == 'pml':
		s += dim.rbc.eval_on_mesh()
		
	return s.reshape(-1,)

def make_diag_mtx(vec):
	dof = vec.size
	return spsp.spdiags(vec,[0],dof,dof)