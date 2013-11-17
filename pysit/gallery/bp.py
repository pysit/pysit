import numpy as np

from pysit import *

from pysit.gallery.gallery_base import PrecomputedGalleryModel

__all__ = ['BPModel', 'bp']

class BPModel(PrecomputedGalleryModel):
	
	""" Gallery Model class for the BP Velocity Benchmark.
	
	The benchmark is licensed for open distribution, with limitation. 
	See http://software.seg.org/datasets/2D/2004_BP_Vel_Benchmark/2004_Benchmark_READMES.pdf 
	for more details on licensing. You must agree to the licensing to use this model.
	
	"""
	
	# Names
	model_name = "BP Velocity Benchmark"
	
	# A sanitized name for filesystem work
	fs_full_name = "bp"
	fs_short_name = "bp"
	
	# Available data
	supported_physics = ['acoustic', 'variable-density-acoustic']
	supported_physical_parameters = ['density', 'vp']
	supported_masks = ['salt', 'water']
	
	# Descriptive data
	valid_dimensions = (2,)
	
	@property #read only
	def dimension(self):
		return 2
	
	# File information                	
	_local_parameter_filenames = { 'vp' : 'vel_z6.25m_x12.5m_exact.segy.gz',
		                           'density' : 'density_z6.25m_x12.5m.segy.gz'}
		                           	
	_local_mask_filenames = { 'salt' : 'vel_z6.25m_x12.5m_saltindex.segy.gz', # 0 salt, 1 no salt
		                      'water' : 'vel_z6.25m_x12.5m_wbindex.segy.gz'}  # 1 water, 0 no water
		                           	
	_parameter_scale_factor = { 'vp' : 1.0,
		                        'density' : 1.0}
	
	_vp_file_sources = ['ftp://software.seg.org/pub/datasets/2D/2004_BP_Vel_Benchmark/vel_z6.25m_x12.5m_exact.segy.gz']
	_density_file_sources = ['ftp://software.seg.org/pub/datasets/2D/2004_BP_Vel_Benchmark/density_z6.25m_x12.5m.segy.gz']
	
	_salt_file_sources = ['ftp://software.seg.org/pub/datasets/2D/2004_BP_Vel_Benchmark/vel_z6.25m_x12.5m_saltindex.segy.gz']
	_water_file_sources = ['ftp://software.seg.org/pub/datasets/2D/2004_BP_Vel_Benchmark/vel_z6.25m_x12.5m_wbindex.segy.gz']
	
	_remote_file_sources = {'vp' : _vp_file_sources,
		                    'density' : _density_file_sources,
		                    'salt' : _salt_file_sources,
		                    'water' : _water_file_sources}
	
	_model_transposed = False
	
	# Model specification	
	base_physical_origin =  np.array([0.0, 0.0])
	base_physical_size = np.array([67425.0, 11937.5])
	base_physical_dimensions_units = ('m', 'm')
	
	base_pixels = np.array([5395, 1911])
	base_pixel_scale = np.array([12.5, 6.25])
	base_pixel_units = ('m', 'm')

	# Water properties specify the way the water masking is handled
	# (None, ) indicates no water
	# ('depth', <depth: float>) specifies that a preset depth from the base_physical_origin is to be used
	# ('mask', ) indicates that the specified mask name should be used
	water_properties = ('mask', )
	
	_initial_configs = {'smooth_width': {'sigma':3000.0},
		                'smooth_low_pass': {'freq':1./3000.},
		                'constant': {'velocity': 3000.0},
		                'gradient': {'min':1500.0, 'max':3000}}
		                	
	_scale_map = {'full':   np.array([12.5,  6.25]),
		          'large':  np.array([12.5, 12.5]),
		          'medium': np.array([16.0, 16.0]),
		          'small':  np.array([20.0, 20.0]),
		          'mini':   np.array([25.0, 25.0]),}
		          	
	patches = { 'left' : {'origin': np.array([0.0, 0.0]), 
		                  'size': np.array([21400.0, 11937.5])},
		      }
		      
	license_string = "Use of the BP Velocity Benchmark requires you to read and \nagree with the license before continuing: \nhttp://software.seg.org/datasets/2D/2004_BP_Vel_Benchmark/2004_Benchmark_READMES.pdf"
			
def download(parameter='all'):
	
	if parameter == 'all':
		for p in BPModel.supported_physical_parameters:
			BPModel._download_and_prepare(p)
		for p in BPModel.supported_masks:
			BPModel._download_and_prepare(p)
	else:
		BPModel._download_and_prepare(parameter)

def bp(patch=None, **kwargs):
	""" Friendly wrapper for instantiating the BP Velocity model. """
	
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
	
	if patch in BPModel.patches:
		model_config.update(BPModel.patches[patch])
                          	
	return BPModel(**model_config).get_setup()

if __name__ == '__main__':
	
	import matplotlib.pyplot as plt
	
#	M = BPModel(initial_model_style='smooth', pixel_scale='mini')	
	
#	C, C0, m, d = M.get_setup()
#	C, C0, m, d = bp(patch='left',initial_model_style='smooth_low_pass', pixel_scale='small')
	C, C0, m, d = bp(patch='left',initial_model_style='smooth_low_pass', pixel_scale='mini')
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()
	
	C, C0, m, d = bp(patch='left',initial_model_style='smooth_low_pass', pixel_scale='small')
#	C, C0, m, d = bp(patch='left',initial_model_style='smooth_low_pass', pixel_scale='mini')
	
	fig = plt.figure()
	fig.add_subplot(2,1,1)
	vis.plot(C, m)
	fig.add_subplot(2,1,2)
	vis.plot(C0, m)
	plt.show()