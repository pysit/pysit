from __future__ import absolute_import

import math

import numpy as np

from pysit.gallery.gallery_base import GeneratedGalleryModel

__all__ = ['PointReflectorModel', 'point_reflector']

def _gaussian_reflector(grid, pos, rad, amp, threshold):
	""" Gaussian function with position, radius at fwhm, and amplitude. """
	# Assume radius is fwhm of Gaussian
	sigma = rad / math.sqrt(2.0*math.log(2.0))
	
	# reduce(X, map(Y, (Grid,Pos)))	nicely handles both 2D and 3D gaussians
	T = amp * np.exp( -1.0* reduce(lambda x,y: x+y, map(lambda x: (x[0]-x[1])**2, zip(grid,pos)))  / (2.0*sigma**2))
	T[np.where(abs(T) < threshold)] = 0
	return T

class PointReflectorModel(GeneratedGalleryModel):
	
	model_name =  "Point Reflector"
		
	valid_dimensions = (1,2,3)
		
	@property #read only
	def dimension(self):
		return self.domain.dim

	supported_physics = ('acoustic',)

	def __init__(self, mesh,  
		               reflector_position=[(0.35, 0.42), (0.65, 0.42)], # as percentage of domain size
		               reflector_radius=[0.05, 0.05],
		               reflector_amplitude=[1.0, 1.0],
	                   background_velocity=1.0,
	                   drop_threshold=1e-7,
	                   ):
		""" Constructor for a constant background model with point reflectors.
		
		Parameters
		----------
		mesh : pysit mesh
			Computational mesh on which to construct the model
		reflector_position : list
			Positions of the reflectors, as a percentage of domain size
		reflector_radius : list
			Radius of the reflectors as FWHM of Gaussian
		reflector_amplitude : list
			Scale of the reflectors
		background_velocity : float
		drop_threshold : float
			Cutoff value for evaluation of reflectors
		
		"""

		GeneratedGalleryModel.__init__(self)
		
		
		self.reflector_position = reflector_position
		self.reflector_radius = reflector_radius
		self.reflector_amplitude = reflector_amplitude
		
		self.background_velocity = background_velocity
		
		self.drop_threshold = drop_threshold
		
		self._mesh = mesh
		self._domain = mesh.domain
				
		# Set _initial_model and _true_model
		self.rebuild_models()
		
	def rebuild_models(self, reflector_position=None, reflector_radius=None, reflector_amplitude=None, background_velocity=None):
		""" Rebuild the true and initial models based on the current configuration."""
		
		if reflector_position is not None:
			self.reflector_position = reflector_position
			
		if reflector_radius is not None:
			self.reflector_radius = reflector_radius
			
		if reflector_amplitude is not None:
			self.reflector_amplitude = reflector_amplitude
			
		if background_velocity is not None:
			self.background_velocity = background_velocity
			
		C0 = self.background_velocity*np.ones(self._mesh.shape())
		
		dC = self._build_reflectors()
		
		self._initial_model = C0
		self._true_model = C0 + dC
		
	def _build_reflectors(self):
		
		mesh = self.mesh
		domain = self.domain
		
		grid = mesh.mesh_coords()
		
		dC = np.zeros(mesh.shape())
	
		for pos, rad, amp in zip(self.reflector_position, self.reflector_radius, self.reflector_amplitude):
			
			p = tuple([domain.parameters[i].lbound + pos[i]*domain.parameters[i].length for i in xrange(domain.dim)])
					
			dC += _gaussian_reflector(grid, p, rad, amp, self.drop_threshold)
			
		return dC

def point_reflector( mesh, **kwargs):
	""" Friendly wrapper for instantiating the point reflector model. """
	
	# Setup the defaults
	model_config = dict(reflector_position=[(0.35, 0.42), (0.65, 0.42)], # as percentage of domain size
		                reflector_radius=[0.05, 0.05],
		                reflector_amplitude=[1.0, 1.0],
	                    background_velocity=1.0,
	                    drop_threshold=1e-7)
	
	# Make any changes
	model_config.update(kwargs)
                          	
	return PointReflectorModel(mesh, **model_config).get_setup()

if __name__ == '__main__':
	
	from pysit import *
	
	#       Define Domain
	pmlx = PML(0.1, 100)
	pmlz = PML(0.1, 100)
	
	x_config = (0.1, 1.0, pmlx, pmlx)
	z_config = (0.1, 0.8, pmlz, pmlz)
	
	d = RectangularDomain(x_config, z_config)
	
	m = CartesianMesh(d, 90, 70)
	
	#       Generate true wave speed
	C, C0, m, d = point_reflector(m)
	
	import matplotlib.pyplot as plt
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()
