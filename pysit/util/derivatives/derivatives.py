from __future__ import absolute_import

import numpy as np
import scipy.sparse as spsp

import pyamg
from pyamg.gallery import stencil_grid

from pysit.util.derivatives.fdweight import *

__all__ = ['build_derivative_matrix']

def build_derivative_matrix(mesh, 
                            derivative, order_accuracy,  
                            **kwargs):
	
	if mesh.type == 'structured-cartesian':
		return _build_derivative_matrix_structured_cartesian(mesh, derivative, order_accuracy, **kwargs)
	else:
		raise NotImplementedError('Derivative matrix builder not available (yet) for {0} meshes.'.format(mesh.discretization))

def _set_bc(bc):
	if bc.type == 'pml':     
		return bc.boundary_type
	elif bc.type == 'ghost': 
		return ('ghost', bc.n)
	else:
		return bc.type
	
	
		
def _build_derivative_matrix_structured_cartesian(mesh, 
                                                  derivative, order_accuracy, 
                                                  dimension='all',
                                                  use_shifted_differences=False,
                                                  return_1D_matrix=False,
                                                  **kwargs):
	
	dims = list()
	if type(dimension) is str:
		dimension = [dimension]
	if 'all' in dimension:
		if mesh.dim > 1: 
			dims.append('x')
		if mesh.dim > 2: 
			dims.append('y')
		dims.append('z')
	else:
		for d in dimension:
			dims.append(d)
		
	# sh[-1] is always 'z'
	# sh[0] is always 'x' if in 2 or 3d
	# sh[1] is always 'y' if dim > 2
	sh = mesh.shape(include_bc = True, as_grid = True)
		
	if mesh.dim > 1:
		if 'x' in dims:
			lbc = _set_bc(mesh.x.lbc)
			rbc = _set_bc(mesh.x.rbc)
			delta = mesh.x.delta
			Dx = _build_derivative_matrix_part(sh[0], derivative, order_accuracy, h=delta, lbc=lbc, rbc=rbc, use_shifted_differences=use_shifted_differences)
		else:
			Dx = spsp.csr_matrix((sh[0],sh[0]))  
	if mesh.dim > 2:
		if 'y' in dims:
			lbc = _set_bc(mesh.y.lbc)
			rbc = _set_bc(mesh.y.rbc)
			delta = mesh.y.delta
			Dy = _build_derivative_matrix_part(sh[1], derivative, order_accuracy, h=delta, lbc=lbc, rbc=rbc, use_shifted_differences=use_shifted_differences)
		else:
			Dy = spsp.csr_matrix((sh[1],sh[1])) 
	
	if 'z' in dims:
		lbc = _set_bc(mesh.z.lbc)
		rbc = _set_bc(mesh.z.rbc)
		delta = mesh.z.delta
		Dz = _build_derivative_matrix_part(sh[-1], derivative, order_accuracy, h=delta, lbc=lbc, rbc=rbc, use_shifted_differences=use_shifted_differences)
	else:
		Dz = spsp.csr_matrix((sh[-1],sh[-1]))  

	if return_1D_matrix and 'all' not in dims:
		if 'z' in dims:
			mtx = Dz
		elif 'y' in dims:
			mtx = Dy
		elif 'x' in dims:
			mtx = Dx
	else:					
		if mesh.dim == 1:
			mtx = Dz.tocsr()
		if mesh.dim == 2:
			# kronsum in this order because wavefields are stored with 'z' in row 
			# and 'x' in columns, then vectorized in 'C' order
			mtx = spsp.kronsum(Dz, Dx, format='csr')
		if mesh.dim == 3:
			mtx = spsp.kronsum(Dz, spsp.kronsum(Dy,Dx, format='csr'), format='csr')
	
	return mtx
	
def _build_derivative_matrix_part(npoints, derivative, order_accuracy, h=1.0, lbc='d', rbc='d', use_shifted_differences=False):
	
	if order_accuracy%2:
		raise ValueError('Only even accuracy orders supported.')
		
	centered_coeffs = centered_difference(derivative, order_accuracy)/(h**derivative)
	
	mtx = stencil_grid(centered_coeffs, (npoints, ), format='lil')
	
	max_shift= order_accuracy/2
	
	if use_shifted_differences:
		# Left side
		odd_even_offset = 1-derivative%2
		for i in xrange(0, max_shift):
			coeffs = shifted_difference(derivative, order_accuracy, -(max_shift+odd_even_offset)+i)
			mtx[i,0:len(coeffs)] = coeffs/(h**derivative)
		
		# Right side
		for i in xrange(-1, -max_shift-1,-1):
			coeffs = shifted_difference(derivative, order_accuracy, max_shift+i+odd_even_offset)
			mtx[i,slice(-1, -(len(coeffs)+1),-1)] = coeffs[::-1]/(h**derivative)
					
	if 'd' in lbc: #dirichlet
		mtx[0,:] = 0
		mtx[0,0] = 1.0
	elif 'n' in lbc: #neumann
		mtx[0,:] = 0
		coeffs = shifted_difference(1, order_accuracy, -max_shift)/h
		coeffs /= (-1*coeffs[0])
		coeffs[0] = 0.0
		mtx[0,0:len(coeffs)] = coeffs
	elif type(lbc) is tuple and 'g' in lbc[0]: #ghost
		n_ghost_points = int(lbc[1])
		mtx[0:n_ghost_points,:] = 0
		for i in xrange(n_ghost_points):
			mtx[i,i] = 1.0

	if 'd' in rbc:
		mtx[-1,:] = 0
		mtx[-1,-1] = 1.0
	elif 'n' in rbc:
		mtx[-1,:] = 0
		coeffs = shifted_difference(1, order_accuracy, max_shift)/h
		coeffs /= (-1*coeffs[-1])
		coeffs[-1] = 0.0
		mtx[-1,slice(-1, -(len(coeffs)+1),-1)] = coeffs[::-1]
	elif type(rbc) is tuple and 'g' in rbc[0]:
		n_ghost_points = int(rbc[1])
		mtx[slice(-1,-(n_ghost_points+1), -1),:] = 0
		for i in xrange(n_ghost_points):
			mtx[-i-1,-i-1] = 1.0
		
	return mtx.tocsr()
	
	
def apply_derivative(mesh, derivative, order_accuracy, vector, **kwargs):
	A = build_derivative_matrix(mesh, derivative, order_accuracy, **kwargs)
	return A*vector


if __name__ == '__main__':
	
	from pysit import *
	from pysit.gallery import horizontal_reflector
	
	bc = Dirichlet()
	
	dim = 2
	deriv = 1 # 2
	order = 4
	
	if dim == 1:
		z_config = (0.0, 7.0, bc, bc)
		
		d = RectangularDomain(z_config)
	
		m = CartesianMesh(d, 7)		
		#	Generate true wave speed
		C, C0 = horizontal_reflector(m)
		
		
		D = build_derivative_matrix(m, deriv, order, dimension='all').todense()
		
		Dz = build_derivative_matrix(m, deriv, order, dimension='z').todense()
		
	if dim == 2:
		
		x_config = (0.0, 7.0, bc, bc)
		z_config = (0.0, 7.0, bc, bc)

		d = RectangularDomain(x_config, z_config)
		
		m = CartesianMesh(d, 7, 7)
		
		#	Generate true wave speed
		C, C0 = horizontal_reflector(m)
		
		D = build_derivative_matrix(m, deriv, order, dimension='all').todense()
		
		Dx = build_derivative_matrix(m, deriv, order, dimension='x').todense()
		Dz = build_derivative_matrix(m, deriv, order, dimension='z').todense()
		

	if dim == 3:
				
		x_config = (0.0, 7.0, bc, bc)
		y_config = (0.0, 7.0, bc, bc)
		z_config = (0.0, 7.0, bc, bc)

		d = RectangularDomain(x_config, x_config, z_config)
		
		m = CartesianMesh(d, 7, 7, 7)
		
		#	Generate true wave speed
		C, C0 = horizontal_reflector(m)
		
		
		D = build_derivative_matrix(m, deriv, order, dimension='all').todense()
		
		sh = m.shape(as_grid=True)
		
		Dx = build_derivative_matrix(m, deriv, order, dimension=['x']).todense()
		Dy = build_derivative_matrix(m, deriv, order, dimension=['y']).todense()
		Dz = build_derivative_matrix(m, deriv, order, dimension=['z']).todense()
		
		x=(Dx*C).reshape(sh)
		y=(Dy*C).reshape(sh)
		z=(Dz*C).reshape(sh)
		
		print x[:,:,0] # should have ones all in first and last rows
		print y[:,:,0] # should have ones all in first and last columns
		print z[0,0,:] # should have ones at the ends
	