import warnings
import math

import numpy as np
from pyamg.gallery import stencil_grid

__all__ = ['cd_coeffs', 'fd_stencil', 'stencil_grid', 'fd_coeffs', 'build_1D_fd']
	
cd_coeffs = {
	1 : { 1 : None,
	      2 : [-0.5, 0, 0.5],
		  3 : None,
		  4 : [1./12, -2./3, 0., 2./3, -1./12],
		  5 : None,
		  6 : [-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0],
		  7 : None,
		  8 : [1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0],
	    },
	2 : { 1 : None,
	      2 : [1.0, -2.0, 1.0],
		  3 : None,
		  4 : [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0],
		  5 : None,
		  6 : [1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0],
		  7 : None,
		  8 : [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0],
	    }
}

def fd_stencil(base_stencil, dim, axis='all'):
	
	if axis == 'all':
		axes = range(dim)
	else:
		if axis >= dim: raise ValueError()
		axes = [axis]

	ln = len(base_stencil)
	mid = int(np.floor(ln/2))
	
	stencil = np.zeros([ln for x in xrange(dim)])
	
	for axis in axes:

		# shift the axes around so that they match our coordinate system (see 
		# domain.py for more details)
		if dim == 3:
			warnings.warn('Behavior for 3D problems is not confirmed to be proper!!!')
#		ax = np.mod(axis+(dim-1),dim)
		ax = axis
			
		stencil[[mid if x!=ax else slice(None) for x in xrange(dim)]] += base_stencil
	
	return stencil

def fd_coeffs(derivative, params):
	"""
	
	%---------------------------------
	% finite-difference weights 
	% (Fornberg algorithm) 
	%
	% z:  expansion point 
	% x:  vector of evaluation points 
	% m:  order of derivative 
	%
	% Example: cwei = FDweights(0,[0 1 2],1); 
	% gives    cwei = [-3/2  2  -1/2]
	%
	% h f'_0 = -3/2 f_0 + 2 f_1 - 1/2 f_2 
	%
	%---------------------------------
	"""
	
	if np.iterable(params[0]):
		x = params[0]
		z = params[1]
	else:
		x = np.arange(params[0])
		z = params[1]
	
	m = derivative
	
	x = np.asarray(x)
	z = float(z)
	
	n = len(x)-1
	
	c1 = 1.
	c4 = x[0]-z
	
	C = np.zeros((len(x),m+1))
	C[0,0] = 1.
	
	for i in xrange(1,n+1):
		mn = min(i,m)
		c2 = 1.
		c5 = c4
		c4 = x[i]-z
		
		for j in xrange(0,i):
			
			c3 = x[i]-x[j]
			c2 *= c3
			
			if j == i-1:
				for k in xrange(mn,0,-1):
					C[i,k] = c1*(k*C[i-1,k-1]-c5*C[i-1,k])/c2
					
				C[i,0] = -c1*c5*C[i-1,0]/c2
			
			for k in xrange(mn,0,-1):
				C[j,k] = (c4*C[j,k]-k*C[j,k-1])/c3
				
			C[j,0] = c4*C[j,0]/c3
		
		c1 = c2
	
	C[np.abs(C) < 1e-16] = 0.0
		
	return C[:,-1].flatten()
		
def build_1D_fd(deriv, order, length, delta, lbc=None, rbc=None, limit_boundary=True):
	""" Builds the finite difference stencil matrix in 1D that can be kroncker producted to build higher dimensional operators.
	
	None in the BC slot leaves the purest form of the operator.
	
	"""
	
	bulk_npoints = deriv + order - (1 if not deriv%2 else 0)
	bulk_center = int(math.floor(bulk_npoints/2))
	
	boundary_npoints = deriv + order
	
	stencil = fd_coeffs(deriv, (bulk_npoints, bulk_center))
	stencil[np.abs(stencil) < 1e-12] = 0.0
	L = stencil_grid(stencil, (length,), format='lil')
	
	if not limit_boundary:
		L /= (delta**deriv)	
		return L.tocsr()
		
	# left side
	for i in xrange(bulk_center):
		boundary_center = i
		if i == 0:
			if lbc != 'dirichlet':
				warnings.warn('Only Dirichlet boundaries are supported in this matrix construction.')
			L[i,:] = 0.0
			L[0,0]=1.0
#			else: #lbc == 'neumann'
#				# Not sure that this is correct...neumann likely need to be patched after the time step...
#				L[i,:] = 0.0
#				coeffs = -fd_coeffs(1, (1+order,boundary_center))
#				coeffs /= coeffs[0]
#				coeffs[0] = 0.0
#				L[i,0:(1+order)] = coeffs
		else:	
			L[i,:] = 0
			stencil = fd_coeffs(deriv, (boundary_npoints,boundary_center))
			stencil[np.abs(stencil) < 1e-12] = 0.0
			L[i,0:boundary_npoints] = stencil
			
	# right side
	print boundary_npoints-bulk_center-1
	
	for i in xrange(-1, -(boundary_npoints-bulk_center-deriv+1), -1):
		boundary_center = boundary_npoints + i
		idx = i
		print i, boundary_center, idx
		if idx == -1:
			if lbc != 'dirichlet':
				warnings.warn('Only Dirichlet boundaries are supported in this matrix construction.')
			L[idx,:] = 0.0
			L[-1,-1] = 1.0
#			else: #lbc == 'neumann'
#				# Not sure that this is correct...neumann likely need to be patched after the time step...
#				L[i,:] = 0.0
#				coeffs = -fd_coeffs(1, (1+order,boundary_center))
#				coeffs /= coeffs[0]
#				coeffs[0] = 0.0
#				L[i,0:(1+order)] = coeffs
		else:	
			L[idx,:] = 0
			stencil = fd_coeffs(deriv, (boundary_npoints,boundary_center))
			stencil[np.abs(stencil)<1e-12] = 0.0
			L[idx,-boundary_npoints::] = stencil
					
	L /= (delta**deriv)		
	
	return L.tocsr()
	
	
#	
#
#if __name__=='__main__':
#	
#	from pysit import Domain, PML
#	
#	pml = PML(0.0, 100,ftype='polynomial')
#	
#	x_config = (0.0, 3.0, 3, pml, pml)
#	y_config = (0.0, 3.0, 3, pml, pml)
#	z_config = (0.0, 3.0, 3, pml, pml)
#	
#	d = Domain( (x_config, y_config, z_config) )
#	
#	sten = cd_coeffs[2][2]
#	
#	sx = fd_stencil(sten, 3, 0)
#	sy = fd_stencil(sten, 3, 1)
#	sz = fd_stencil(sten, 3, 2)
#	
#	gx = stencil_grid(sx, (3,3,3)).todense()
#	gy = stencil_grid(sy, (3,3,3)).todense()
#	gz = stencil_grid(sz, (3,3,3)).todense()
#	
#	print gx
#	print gy
#	print gz

def test_1st():
	
	L = build_1D_fd(1, 4, 7, 1.0).todense()
	
	correct = np.array([[ 1.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ],
                        [-0.25              , -0.8333333333333334,  1.5               , -0.5               ,  0.0833333333333333,  0.                ,  0.                ],
                        [ 0.0833333333333333, -0.6666666666666666,  0.                ,  0.6666666666666666, -0.0833333333333333,  0.                ,  0.                ],
                        [ 0.                ,  0.0833333333333333, -0.6666666666666666,  0.                ,  0.6666666666666666, -0.0833333333333333,  0.                ],
                        [ 0.                ,  0.                ,  0.0833333333333333, -0.6666666666666666,  0.                ,  0.6666666666666666, -0.0833333333333333],
                        [ 0.                ,  0.                , -0.0833333333333333,  0.5               , -1.5               ,  0.8333333333333333,  0.25              ],
                        [ 0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  1.                ]])
	
	assert np.linalg.norm(L-correct) < 1e-14

def test_2nd():
	
	L = build_1D_fd(2, 4, 7, 1.0).todense()
	
	correct = np.array([[ 1.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ],
                        [ 0.8333333333333333, -1.25              , -0.3333333333333333,  1.1666666666666665, -0.5               ,  0.0833333333333333,  0.                ],
                        [-0.0833333333333333,  1.3333333333333333, -2.5               ,  1.3333333333333335, -0.0833333333333333,  0.                ,  0.                ],
                        [ 0.                , -0.0833333333333333,  1.3333333333333333, -2.5               ,  1.3333333333333335, -0.0833333333333333,  0.                ],
                        [ 0.                ,  0.                , -0.0833333333333333,  1.3333333333333333, -2.5               ,  1.3333333333333333, -0.0833333333333333],
                        [ 0.                ,  0.0833333333333333, -0.5000000000000001,  1.1666666666666667, -0.333333333333333 , -1.2499999999999996,  0.8333333333333333],
                        [ 0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  1.                ]])
	
	assert  np.linalg.norm(L-correct) < 1e-14
	
if __name__ == '__main__':
	pass
#	test_1st()
#	test_2nd()
