import numpy as np

from pysit import *

from pysit.gallery.gallery_base import PrecomputedGalleryModel

__all__ = ['MarmousiModel', 'marmousi']

class MarmousiModel(PrecomputedGalleryModel):
	
	""" Marmousi community velocity model.
	"""
	
	# Names
	model_name = "Marmousi"
	
	# A sanitized name for filesystem work
	fs_full_name = "marmousi"
	fs_short_name = "marm"
	
	# Available data
	supported_physics = ['acoustic', 'variable-density-acoustic']
	supported_physical_parameters = ['density', 'vp']
	
	# Descriptive data
	valid_dimensions = (2,)
	@property #read only
	def dimension(self):
		return 2
	
	# File information                	
	_local_parameter_filenames = { 'vp' : 'velocity_rev1.segy.gz',
		                           'density' : 'density_rev1.segy.gz'}
		                           	
	_parameter_scale_factor = { 'vp' : 1.0,
		                        'density' : 1.0}
	
	_vp_file_sources = ['http://math.mit.edu/~rhewett/pysit/marmousi/velocity_rev1.segy.gz']
	_density_file_sources = ['http://math.mit.edu/~rhewett/pysit/marmousi/density_rev1.segy.gz']
	
	_remote_file_sources = {'vp' : _vp_file_sources,
		                    'density' : _density_file_sources}
	
	_model_transposed = True
	
	# Model specification	
	base_physical_origin =  np.array([0.0, 0.0])
	base_physical_size = np.array([9200.0, 3000.0])
	base_physical_dimensions_units = ('m', 'm')
	
	base_pixels = np.array([2301, 751])
	base_pixel_scale = np.array([4.0, 4.0])
	base_pixel_units = ('m', 'm')

	# Water properties specify the way the water masking is handled
	# (None, ) indicates no water
	# ('depth', <depth: float>) specifies that a preset depth from the base_physical_origin is to be used
	# ('mask', ) indicates that the specified mask name should be used
	water_properties = ('depth', 32.0)
	
	_initial_configs = {'smooth_width': {'sigma':300.0},
		                'smooth_low_pass': {'freq':1./300.},
		                'constant': {'velocity': 3000.0},
		                'gradient': {'min':1500.0, 'max':3000}}
		                	
	_scale_map = {'full':   np.array([ 4.0,  4.0]),
		          'large':  np.array([ 8.0,  8.0]),
		          'medium': np.array([12.0, 12.0]),
		          'small':  np.array([16.0, 16.0]),
		          'mini':   np.array([20.0, 20.0]),}
		          	
	# old marmousi mini-square had dx=24m
	patches = { 'mini-square' : {'origin': np.array([195*24.0, 0.0]), 
		                         'size': np.array([3000.0, 3000.0])},
		      }

def download(parameter='all'):
	
	if parameter == 'all':
		for p in MarmousiModel.supported_physical_parameters:
			MarmousiModel._download_and_prepare(p)
		for p in MarmousiModel.supported_masks:
			MarmousiModel._download_and_prepare(p)
	else:
		MarmousiModel._download_and_prepare(parameter)

def marmousi(patch=None, **kwargs):
	""" Friendly wrapper for instantiating the Marmousi model. """
	
	model_config = dict(physics='acoustic', 
	                    origin=None,
	                    size=None,
	                    pixel_scale='mini', 
	                    pixels=None,
	                    initial_model_style='smooth_low_pass',
	                    initial_config={},
	                    fix_water_layer=True,)
	
	# Make any changes
	model_config.update(kwargs)
	
	if patch in MarmousiModel.patches:
		model_config.update(MarmousiModel.patches[patch])
                          	
	return MarmousiModel(**model_config).get_setup()

if __name__ == '__main__':
	
	import matplotlib.pyplot as plt
	
#	M = MarmousiModel(initial_model_style='smooth_low_pass', pixel_scale='full')	
	
#	
	C, C0, m, d = marmousi(patch='mini-square')
	C, C0, m, d = marmousi(pixel_scale='small', initial_model_style='smooth_width', patch='mini-square')
#	C, C0, m, d = marmousi(pixel_scale='medium', initial_model_style='smooth_low_pass')
#	C, C0, m, d = marmousi(pixel_scale='medium', initial_model_style='constant')
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()
#	C, C0, m, d = marmousi(pixel_scale='medium', initial_model_style='smooth_width')
	C, C0, m, d = marmousi(pixel_scale='medium', initial_model_style='smooth_low_pass', patch='mini-square')
#	C, C0, m, d = marmousi(pixel_scale='medium', initial_model_style='constant')
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()