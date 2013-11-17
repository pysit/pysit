from __future__ import division

import math

import numpy as np

__all__ = ['finite_difference_coefficients', 'centered_difference', 'shifted_difference']

def centered_difference(deriv, order):
	if order%2:
		raise ValueError('Centered differences only defined for even accuracy.')
	npoints = deriv + order + deriv%2 - 1
	center_idx = math.floor(npoints/2)
	return finite_difference_coefficients(center_idx, np.arange(npoints),deriv)

def shifted_difference(deriv, order, shift):
	npoints = deriv+order
	center_idx = math.floor(npoints/2) + shift
	if center_idx < 0 or center_idx > (npoints-1):
		raise ValueError('Shift out of bounds.')
	return finite_difference_coefficients(center_idx, np.arange(npoints),deriv)

def finite_difference_coefficients(z, x, m, tol=1e-10):
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
	
	C[np.abs(C)<tol]=0.0
	return C[:,-1].flatten()
	

	
def rational_weights(center, npoints, deriv, dtype=np.int):
	
	center = int(center)
	
	n = npoints-1
	
	c1 = 1.
	c4 = 0-center
	
	C_num = np.zeros((npoints,deriv+1),dtype=dtype)
	C_den = np.ones((npoints,deriv+1),dtype=dtype)
	C_num[0,0] = 1
	C_den[0,0] = 1
	
	for i in xrange(1,n+1):
		mn = min(i,deriv)
		c2 = 1.
		c5 = c4
		c4 = i-center
		
		for j in xrange(0,i):
			c3 = i-j
			c2 *= c3
			
			if j == i-1:
				for k in xrange(mn,0,-1):
					C_num[i,k] = (c1*(k*C_num[i-1,k-1]*C_den[i-1,k]-c5*C_num[i-1,k]*C_den[i-1,k-1]))
					C_den[i,k] = C_den[i-1,k-1]*C_den[i-1,k]*c2
					
				C_num[i,0] = -c1*c5*C_num[i-1,0]
				C_den[i,0] = C_den[i-1,0]*c2
			
			for k in xrange(mn,0,-1):
				C_num[j,k] = (c4*C_num[j,k]*C_den[j,k-1]-k*C_num[j,k-1]*C_den[j,k])
				C_den[j,k] = (C_den[j,k]*C_den[j,k-1])*c3
				
			C_num[j,0] = c4*C_num[j,0]
			C_den[j,0] = C_den[j,0]*c3
		
		c1 = c2
		
	return C_num, C_den
	
