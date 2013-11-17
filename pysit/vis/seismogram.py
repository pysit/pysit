import itertools
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ['plot_seismogram']

def plot_seismogram(shot, axis=None, **kwargs):

	if axis is None:
		ax = plt.gca()
	else:
		ax = axis
	
	domain = shot.sources.domain
	
	data = shot.receivers.data
	#np.array([r.data for r in shot.receiver_list])
	
	plt.imshow(data, aspect='auto', **kwargs)
	seismogram_tickers(shot)
		
	plt.xlabel('X-Position')
	plt.ylabel('Time (s)')
		
	plt.show()
	
class ReceiverFormatterHelper(object):
	def __init__(self, shot):
		self.shot = shot
	def __call__(self, grid_point, pos):
		if ((grid_point >= 0) and (grid_point < len(self.shot.receivers.receiver_list))):
			return '{0:.3}'.format(self.shot.receivers.receiver_list[int(np.floor(grid_point))].position[0])
		else:
			return ''
		
class TimeFormatterHelper(object):
	def __init__(self, shot):
		self.shot=shot
	def __call__(self, grid_point, pos):
		return '{0:.3}'.format(grid_point * self.shot.dt)
		
def seismogram_tickers(shot):
	ax = plt.gca()
	
	xformatter = mpl.ticker.FuncFormatter(ReceiverFormatterHelper(shot))
	ax.xaxis.set_major_formatter(xformatter)
	
	zformatter = mpl.ticker.FuncFormatter(TimeFormatterHelper(shot))
	ax.yaxis.set_major_formatter(zformatter)
		
if __name__ == '__main__':

	from pysit import *
	from pysit.gallery import horizontal_reflector
	
	# Setup

	#	Define Domain
	pmlx = PML(0.1, 100)
	pmlz = PML(0.1, 100)
	
	x_config = (0.1, 1.0, pmlx, pmlx)
	z_config = (0.1, 0.8, pmlz, pmlz)

	d = RectangularDomain(x_config, z_config)
	
	m = CartesianMesh(d, 90, 70)

	#	Generate true wave speed
	C, C0, m, d = horizontal_reflector(m)
	
	# Set up shots
	zmin = d.z.lbound
	zmax = d.z.rbound
	zpos = zmin + (1./9.)*zmax
	
	shots = equispaced_acquisition(m,
	                               RickerWavelet(10.0),
	                               sources=1,
	                               source_depth=zpos,
	                               source_kwargs={},
	                               receivers='max',
	                               receiver_depth=zpos,
	                               receiver_kwargs={},
	                               )
	
	# Define and configure the wave solver
	trange = (0.0,3.0)
	
	solver = ConstantDensityAcousticWave(m,
	#                                    formulation='ode',
	                                     formulation='scalar',
	                                     model_parameters={'C': C}, 
		                                 spatial_accuracy_order=2,
		                                 trange=trange,
		                                 use_cpp_acceleration=True,
		                                 time_accuracy_order=4)
		                                 
	# Generate synthetic Seismic data
	tt = time.time()
	wavefields =  []
	base_model = solver.ModelParameters(m,{'C': C})
	generate_seismic_data(shots, solver, base_model, wavefields=wavefields)
		
	print 'Data generation: {0}s'.format(time.time()-tt)
	
	plot_seismogram(shots[0])
	
