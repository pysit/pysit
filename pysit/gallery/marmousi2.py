import numpy as np

from pysit import *

from pysit.gallery.gallery_base import PrecomputedGalleryModel

__all__ = ['MarmousiModel2', 'marmousi2']

class MarmousiModel2(PrecomputedGalleryModel):
	
	""" Marmousi II community velocity model.
	"""
	
	# Names
	model_name = "Marmousi2"
	
	# A sanitized name for filesystem work
	fs_full_name = "marmousi2"
	fs_short_name = "marm2"
	
	# Available data
	supported_physics = ['acoustic', 'variable-density-acoustic', 'elastic']
	supported_physical_parameters = ['density', 'vp', 'vs']
	
	# Descriptive data
	valid_dimensions = (2,)
	@property #read only
	def dimension(self):
		return 2
	
	# File information                	
	_local_parameter_filenames = { 'vp' : 'vp_marmousi-ii.segy.gz',
		                           'vs' : 'vs_marmousi-ii.segy.gz',
		                           'density' : 'density_marmousi-ii.segy.gz'}
		                           	
	_parameter_scale_factor = { 'vp' : 1000.0,
		                        'vs' : 1000.0,
		                        'density' : 1.0}
	
	_vp_file_sources = ['http://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz']
	_vs_file_sources = ['http://www.agl.uh.edu/downloads/vs_marmousi-ii.segy.gz']
	_density_file_sources = ['http://www.agl.uh.edu/downloads/density_marmousi-ii.segy.gz']
	
	_remote_file_sources = {'vp' : _vp_file_sources,
		                    'vs' : _vs_file_sources,
		                    'density' : _density_file_sources}
	
	_model_transposed = False
	
	# Model specification	
	base_physical_origin =  np.array([0.0, 0.0])
	base_physical_size = np.array([17000.0, 3500.0])
	base_physical_dimensions_units = ('m', 'm')
	
	base_pixels = np.array([13601, 2801])
	base_pixel_scale = np.array([1.25, 1.25])
	base_pixel_units = ('m', 'm')


	# Water properties specify the way the water masking is handled
	# (None, ) indicates no water
	# ('depth', <depth: float>) specifies that a preset depth from the base_physical_origin is to be used
	# ('mask', ) indicates that the specified mask name should be used
	water_properties = ('depth', 473.0) #m, see Marmousi2, an elasticc upgrade for Marmousi
	
	_initial_configs = {'smooth_width': {'sigma':1000.0},
		                'smooth_low_pass': {'freq':1./1000.},
		                'constant': {'velocity': 3000.0},
		                'gradient': {'min':1500.0, 'max':3000}}
		                	
	_scale_map = {'full':   np.array([ 1.25,  1.25]),
		          'large':  np.array([ 2.50,  2.50]),
		          'medium': np.array([ 5.00,  5.00]),
		          'small':  np.array([10.00, 10.00]),
		          'mini':   np.array([20.00, 20.00]),}
		          	
	# old marmousi mini-square had dx=24m
	patches = { 'original' : {'origin': np.array([3950, 440.0]), 
		                      'size': np.array([9200.0,3000.0])},
		                      	
		        'mini-square' : {'origin': np.array([3950+195*24.0, 440.0]), 
		                         'size': np.array([128*24.0, 128*24.0])},
		                         	
		        'left-layers-square' : {'origin': np.array([1500.0, 0.0]), 
		                                'size': np.array([3000.0, 3000.0])},
		                                	
		        'right-layers-square' : {'origin': np.array([11630.0, 0.0]), 
		                                 'size': np.array([3000.0, 3000.0])},
		                                 	
		        'fracture-square' : {'origin': np.array([8630.0, 0.0]), 
		                             'size': np.array([3000.0, 3000.0])},
		      }

def download(parameter='all'):
	
	if parameter == 'all':
		for p in MarmousiModel2.supported_physical_parameters:
			MarmousiModel2._download_and_prepare(p)
		for p in MarmousiModel2.supported_masks:
			MarmousiModel2._download_and_prepare(p)
	else:
		MarmousiModel2._download_and_prepare(parameter)

def marmousi2(patch=None, **kwargs):
	
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
	
	if patch in MarmousiModel2.patches:
		model_config.update(MarmousiModel2.patches[patch])
                          	
	return MarmousiModel2(**model_config).get_setup()

if __name__ == '__main__':
	
	import matplotlib.pyplot as plt
	
	M = MarmousiModel2(initial_model_style='smooth_low_pass', pixel_scale='mini')	
	
	C, C0, m, d = M.get_setup()
#	C, C0, m, d = marmousi2(patch='mini-square')
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()
	
	M = MarmousiModel2(initial_model_style='smooth_low_pass', pixel_scale='medium')	
	
	C, C0, m, d = M.get_setup()
#	C, C0, m, d = marmousi2(patch='mini-square')
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()