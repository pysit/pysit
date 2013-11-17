from __future__ import absolute_import

import numpy as np

from pysit.gallery.gallery_base import GeneratedGalleryModel

__all__ = ['HorizontalReflectorModel', 'horizontal_reflector']		
	
def _gaussian_derivative_pulse(ZZ, threshold, **kwargs):
	""" Derivative of a Gaussian at a specific sigma """
	T = -100.0*ZZ*np.exp(-(ZZ**2)/(1e-4))
	T[np.where(abs(T) < threshold)] = 0
	return T

def _gaussian_pulse(ZZ, threshold, sigma_in_pixels=1.0, **kwargs):
	""" Gaussian function, in Z direction, with sigma specified in terms of pixels """
	zdelta = ZZ[np.where((ZZ-ZZ.min) != 0.0)].min() - ZZ.min()
	sigma = sigma_in_pixels*zdelta
	T = np.exp(-(ZZ**2) / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
	T = T * zdelta
	T[np.where(abs(T) < threshold)] = 0
	return T

_pulse_functions = { 'gaussian_derivative' : _gaussian_derivative_pulse,
	                 'gaussian' : _gaussian_pulse
	               }	


class HorizontalReflectorModel(GeneratedGalleryModel):
	
	""" Gallery model for constant background plus simple horizontal reflectors. """
	
	model_name = "Horizontal Reflector"
		
	valid_dimensions = (1,2,3)
		
	@property
	def dimension(self):
		return self.domain.dim

	supported_physics = ('acoustic',)

	def __init__(self, mesh,  
	                   reflector_depth=[0.45, 0.65], # as percentage of domain
	                   reflector_scaling=[1.0, 1.0],
	                   background_velocity=1.0,
	                   drop_threshold=1e-7,
	                   pulse_style='gaussian_derivative',
	                   pulse_config={},
	                   ):
		""" Constructor for a constant background model with horizontal reflectors.
		
		Parameters
		----------
		mesh : pysit mesh
			Computational mesh on which to construct the model
		reflector_depth : list
			Depths of the reflectors, as a percentage of domain depth
		reflector_scaling : list
			Scale factors for reflectors
		background_velocity : float
		drop_threshold : float
			Cutoff value for evaluation of reflectors
		pulse_style : {'gaussian_derivative', 'gaussian_pulse'}
			Shape of the reflector
		pulse_config : dict
			Configuration of the pulses.
			
		Notes
		-----
		The following options exist for the pulse styles:
		'gaussian_pulse' : 'sigma_in_pixels' : width of the gaussial pulse
		
		"""

		GeneratedGalleryModel.__init__(self)
		
		
		self.reflector_depth = reflector_depth
		self.reflector_scaling = reflector_scaling
		
		self.background_velocity = background_velocity
		
		self.drop_threshold = drop_threshold
		
		self.pulse_style = pulse_style
		self.pulse_config = pulse_config
		
		self._mesh = mesh
		self._domain = mesh.domain
		# Set _initial_model and _true_model
		self.rebuild_models()
		
	def rebuild_models(self, reflector_depth=None, reflector_scaling=None, background_velocity=None):
		""" Rebuild the true and initial models based on the current configuration."""
		
		if reflector_depth is not None:
			self.reflector_depth = reflector_depth
			
		if reflector_scaling is not None:
			self.reflector_scaling = reflector_scaling
			
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
		ZZ = grid[-1]		
		
		dC = np.zeros(mesh.shape())

		# can set any defaults here
		if self.pulse_style == 'gaussian_derivative':
			pulse_config = {}
		elif self.pulse_style == 'gaussian':
			pulse_config = {}
		
		# update to any user defined defaults
		pulse_config.update(self.pulse_config)

		for d,s in zip(self.reflector_depth, self.reflector_scaling):
			
			# depth is a percentage of the length
			depth  = domain.z.lbound + d * domain.z.length
			
			pulse = _pulse_functions[self.pulse_style](ZZ-depth, self.drop_threshold, **pulse_config)
			dC += s*pulse
			
		return dC

def horizontal_reflector( mesh, **kwargs):
	""" Friendly wrapper for instantiating the horizontal reflector model. """
	
	# Setup the defaults
	model_config = dict(reflector_depth=[0.45, 0.65], # as percentage of domain
	                    reflector_scaling=[1.0, 1.0],
	                    background_velocity=1.0,
	                    drop_threshold=1e-7,
	                    pulse_style='gaussian_derivative',
	                    pulse_config={},)
	
	# Make any changes
	model_config.update(kwargs)
                          	
	return HorizontalReflectorModel(mesh, **model_config).get_setup()

#if __name__ == '__main__':
#	
#	from pysit import *
#	
#	#       Define Domain
#	pmlx = PML(0.1, 100)
#	pmlz = PML(0.1, 100)
#	
#	x_config = (0.1, 1.0, pmlx, pmlx)
#	z_config = (0.1, 0.8, pmlz, pmlz)
#	
#	d = RectangularDomain(x_config, z_config)
#	
#	m = CartesianMesh(d, 90, 70)
#	
#	#       Generate true wave speed
#	C, C0, m, d = horizontal_reflector(m)
#	
#	import matplotlib.pyplot as plt
#	
#	fig = plt.figure()
#	fig.add_subplot(2,1,1)
#	vis.plot(C, m)
#	fig.add_subplot(2,1,2)
#	vis.plot(C0, m)
#	plt.show()
