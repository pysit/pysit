from __future__ import absolute_import

import numpy as np

from pysit import RectangularDomain, CartesianMesh, PML
from pysit.util.implicit_surfaces import *

from pysit.gallery.gallery_base import GeneratedGalleryModel

__all__ = ['SonarModel', 'sonar_model', 'Submarine']

class Submarine(object):
	""" Class which defines a submarine as the union of two spheres and a cyllinder. """
	def __init__(self, c, scale=1.0, velocity=0.75):	

		self.c = np.array(c)
		self.scale = scale
		self.velocity = velocity
		
		tube_length = 0.2
		tube_radius = 0.05
		
		end1_c = np.array(c)
		end2_c = np.array(c)
		end1_c[0] -= scale*tube_length/2.0
		end2_c[0] += scale*tube_length/2.0
		
		tube = ImplicitXAlignedCylinder(c=c, r=tube_radius*scale, length=tube_length*scale)
		end1 = ImplicitSphere(c=end1_c, r=tube_radius*scale)
		end2 = ImplicitSphere(c=end2_c, r=tube_radius*scale)
		
		self.implicit_surface = ImplicitUnion(tube, end1, end2)
		
	def __call__(self, grid, asarray=False):
		return self.implicit_surface.interior(grid, asarray)
		
default_submarine_2D = Submarine(c=(1.25,0.75), scale=0.5, velocity=7.5)
default_submarine_3D = Submarine(c=(1.25,0.75,0.75), scale=0.5, velocity=7.5)

class SonarModel(GeneratedGalleryModel):
	
	""" Gallery model for subsurface plus ssubmarine "sonar" model. """
	
	model_name = "Sonar"
		
	valid_dimensions = (2,3)
		
	@property #read only
	def dimension(self):
		return self.domain.dim

	supported_physics = ('acoustic',)

	def __init__(self, n_pixels=(250,150),  
	                   submarine=None,
	                   air_velocity=0.1,
	                   water_velocity=1.0,
	                   rock_velocity=10.0,
	                   **kwargs):
		""" Constructor for the sonar model.
		
		Parameters
		----------
		n_pixels : tuple
			The size, in pixels, of the model
		submarine : none or PySIT Submarine
			Specifies presence of a submarine in the model.			
		air_velocity : float
			Velocity in air.
		water_velocity : float
			Velocity in water.
		rock_velocity : float
			Velocity in rock.
		"""

		GeneratedGalleryModel.__init__(self)
		
		self.air_velocity = air_velocity
		self.water_velocity = water_velocity
		self.rock_velocity = rock_velocity
		
		if len(n_pixels) not in [2,3]:
			raise ValueError('Submarine-sonar model only works for dimensions greater than 1.')
		
		if submarine is None:
			if len(n_pixels) == 2:
				submarine = default_submarine_2D
			else: # len(n_pixels) == 3
				submarine = default_submarine_3D
		self.submarine = submarine
			
		config_list = list()
		
		# Configure X direction
		x_lbc = kwargs['x_lbc'] if ('x_lbc' in kwargs.keys()) else PML(0.1, 100.0)
		x_rbc = kwargs['x_rbc'] if ('x_rbc' in kwargs.keys()) else PML(0.1, 100.0)
		
		xmin, xmax = 0.0, 2.5
		x_config = (xmin, xmax, x_lbc, x_rbc)
		config_list.append(x_config)
	
		
		if len(n_pixels) == 3:
			
			# If it is there, configure Y direction
			y_lbc = kwargs['y_lbc'] if ('y_lbc' in kwargs.keys()) else PML(0.1, 100.0)
			y_rbc = kwargs['y_rbc'] if ('y_rbc' in kwargs.keys()) else PML(0.1, 100.0)
			
			ymin, ymax = 0.0, 2.5
			y_config = (ymin, ymax, y_lbc, y_rbc)
			config_list.append(y_config)
		
		# Configure Z direction
		z_lbc = kwargs['z_lbc'] if ('z_lbc' in kwargs.keys()) else PML(0.1, 100.0)
		z_rbc = kwargs['z_rbc'] if ('z_rbc' in kwargs.keys()) else PML(0.1, 100.0)
		
		zmin, zmax = 0.0, 1.5
		z_config = (zmin, zmax, z_lbc, z_rbc)
		config_list.append(z_config)
					
		domain = RectangularDomain(*config_list)
		
		mesh_args = [domain] + list(n_pixels)
		mesh = CartesianMesh(*mesh_args)
		
		
		self._mesh = mesh
		self._domain = mesh.domain
				
		# Set _initial_model and _true_model
		self.rebuild_models()
		
	def rebuild_models(self):
		""" Rebuild the true and initial models based on the current configuration."""
		
		zmin = self._domain.z.lbound
		zmax = self._domain.z.rbound
		
		xmin = self._domain.x.lbound
		xmax = self._domain.x.rbound
		
		grid = self.mesh.mesh_coords()
		
		# the small number is added to prevent undesireable numerical effects
		air_depth   = (1e-8 + 2.0/15.0)   * (zmax - zmin) + zmin
		rock_bottom = 13.0/15.0  * (zmax - zmin) + zmin
		
		coast_left  = 3.0/25.0   * (xmax - xmin) + xmin
		coast_right = 13.0/25.0  * (xmax - xmin) + xmin
		
		max_depth = zmax
		
		# Set up air layer
		if self._domain.dim == 2:
			n = (0., 1.)
			p = (coast_right, air_depth)
		else: # domain.dim == 3
			n = (0.0, 0.0, 1.0)
			p = (coast_right, coast_right, air_depth)
		
		air_plane = ImplicitPlane(p,n)
		air = air_plane
			
		# Set up rock layer
		if self._domain.dim == 2:
			n = (coast_right - coast_left, -(1.0 - air_depth))
			p = (coast_right, max_depth)
			n2 = (0., -1.)
			p2 = (0., rock_bottom)
		else: # domain.dim == 3
			n = (coast_right - coast_left, 0.0, -(1.0 - air_depth))
			p = (coast_right, 0.0, max_depth)
			n2 = (0., 0., -1.)
			p2 = (0., 0., rock_bottom)
			
		rock_plane = ImplicitPlane(p,n)
		rock_plane2 = ImplicitPlane(p2,n2)
		
		rock = ImplicitDifference(ImplicitUnion(rock_plane, rock_plane2), air_plane)
		
		C0 = air.interior(grid, True)   * self.air_velocity   + \
		     rock.interior(grid, True)  * self.rock_velocity  
		
		C0[np.where(C0 == 0.0)] = self.water_velocity
		     
		submarine = self.submarine
		
		if submarine is not None:
			sub = submarine.implicit_surface
			
			C =  air.interior(grid, True)   * self.air_velocity   + \
				 rock.interior(grid, True)  * self.rock_velocity  + \
				 sub.interior(grid, True)   * submarine.velocity
			
			C[np.where(C == 0.0)] = self.water_velocity
			
		else:
			C = C0.copy()
	
		C.shape = self._mesh.shape()
		C0.shape = self._mesh.shape()
		
		self._true_model = C
		self._initial_model = C0
		
	
def sonar_model( **kwargs ):
	""" Friendly wrapper for instantiating the sonar model. """
	
	# Setup the defaults
	model_config = dict(n_pixels=(250,150),  
	                    submarine=None,
	                    air_velocity=0.1,
	                    water_velocity=1.0,
	                    rock_velocity=10.0)
	
	# Make any changes
	model_config.update(kwargs)
                          	
	return SonarModel(**model_config).get_setup()

if __name__ == '__main__':
	
	from pysit import *
	
	#       Generate true wave speed
	C, C0, m, d = sonar_model()
#	C, C0, m, d = sonar_model(n_pixels=(250,250,150))
	
	import matplotlib.pyplot as plt
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()
	
