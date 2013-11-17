import numpy as np
import scipy.sparse as spsp
from scipy.interpolate import interp1d

# The names from this namespace that we wish to expose globally go here.
__all__ = ['Shot']

__docformat__ = "restructuredtext en"

class Shot(object):
	""" Container class for a seismic shot.
	
	The `Shot` class provides a logical grouping of seismic sources with
	receivers.  This class may be refactored so that it is a base class for
	subclasses like SuperShot, SyntheticShot, SegyShot, ProductionShot, etc.
	
	Attributes
	----------
	sources : subclass of SourceBase
		Source or set of source objects.
	receivers : subclass of ReceiverBase
		Receiver or set of receiver objects.
	
	"""

	def __init__(self, sources, receivers):
		"""Constructor for the Shot class.
		
		Parameters
		----------
		source : SeismicSource
			Object representing the source emitter.
		receiver_list : list of SeismicReceiver, optional
			Initial list of receivers.
			
		Examples
		--------
		>>> from pysit import *
		>>> d = Domain()
		>>> S = Shot(SeismicSource(d, (0.5,0.5)))
		
		"""
		
		self.receivers = receivers
		receivers.set_shot(self)
		
		self.sources = sources
		sources.set_shot(self)
		
		# # This is a function/function object, not an attribute.
		# self._interpolator = None
		# self._ts = None
		
		
	# def add_receiver(self, r):
		# """Wrapper for list.append.
		
		# Parameters
		# ----------
		# r : SeismicReceiver
			# r is appended to self.receiver_list.
		# """
		
		# self.receiver_list.append(r)
		# r.set_shot(self)
			
	def initialize(self, data_length):
		"""Clear the data from each receiver in the list of receivers.
		
		Parameters
		----------
		data_length : int
			Length of the desired data array.
		"""
		
		self.receivers.clear_data(data_length)
				
	def reset_time_series(self, ts):
		self.sources.reset_time_series(ts)
		self.receivers.reset_time_series(ts)
		
	def compute_data_dft(self, frequencies, force_computation=False):
		""" Precompute the DFT of the data at the given list of frequencies.
		
		Parameters
		----------
		frequencies : float, iterable
			The frequency or frequencies for which to compute the DFT.
		force_computation : bool {optional}
			Force computation of DFT.  By default already computed frequencies are not recomputed.
		
		"""
		
		self.receivers.compute_data_dft(frequencies)
			
	def gather(self, as_array=False, offset=None):
		"""Collect a sub list of receivers or an array of the data from those
		receivers.
		
		Parameters
		----------
		as_array : bool, optional
			Return the data from the selected receivers as an array, rather than
			returning a list of selected receivers. 
		offset : float or int, optional
			Not implemented.  Will eventually allow an offset to be passed so
			that reduced sized gathers can be collected.
			
		Returns
		-------
		sublist : list of SeismicReceiver
			If as_array is False, list of references to the selected receivers.
		A : numpy.ndarray
			If as_array is True, an array of the data from the selected
			receivers.
		
		"""
		
		if offset is not None:
			# sublist = something that sublists by offset
			raise NotImplementedError('Gather by offset not yet implemented.')
		else:
			sublist = self.receivers
		
		if as_array:
			A = sublist.data #np.array([r.data for r in sublist])
			return A
		else:
			return sublist
			
	def serialize_dict(self):
		
		ret = dict()
		
		ret['dt'] = self.dt
		ret['t_start'] = self.trange[0]
		ret['t_end'] = self.trange[1]
		
		ret['sources'] = self.sources.serialize_dict()
		ret['receivers'] = self.receivers.serialize_dict()
		
		return ret
	
	def unserialize_dict(self, d):
		raise NotImplementedError()

	
	
#if __name__ == '__main__':
#
#	from pysit import * #Domain, PML, RickerWavelet, ReceiverSet, PointReceiver, PointSource, WaveSolverAcousticSecondOrder2D, generate_seismic_data
#	from pysit.gallery import horizontal_reflector
#	import time
#	
#	pmlx = PML(0.1, 100)
#	pmlz = PML(0.1, 100)
#	
#	x_config = (0.1, 1.0, 90, pmlx, pmlx)
#	z_config = (0.1, 0.8, 70, pmlz, pmlz)
#
#	d = Domain( (x_config, z_config) )
#	
#	M, C0, C = horizontal_reflector(d)
#	
#	xmax = d.x.rbound_true
#	nx   = d.x.n_true
#	zmin = d.z.lbound_true
#	zmax = d.z.rbound_true
#				
#	f = RickerWavelet(25.0)
#				
#	ws = PointSource(d, (0.5, 0.5), f)
#	ws2 = PointSource(d, (0.5, 0.5), f, source_approximation='delta')
#	
#	Nshots = 1
#	shots = []
#	
#	for i in xrange(Nshots):
#
#		# Define source location and type
#		source = PointSource(d, (xmax*(i+1.0)/(Nshots+1.0), 0.1), RickerWavelet(25.0))
#	
#		# Define set of receivers
#		zpos = zmin + (1./9.)*zmax
#		xpos = np.reshape(d.generate_grid(sparse=True,exclude_pml=True)[0], (nx,))
#		receivers = [PointReceiver(d, (x, zpos)) for x in xpos[::3]] #receivers every 3 nodes
#	
#		# Create and store the shot
#		shot = Shot(source, ReceiverSet(d, receivers))
#		shots.append(shot)
#		
#	solver_fd_cpp = WaveSolverAcousticSecondOrder2D(d, (0.,0.3), model_parameters={'C': C}, gradient_method='fd', time_step_implementation='c++')
#		
#	print('Generating data FD C++...')
#	tt = time.time()
#	# ps_fd_cpp = None 
#	ps_fd_cpp = []
#	generate_seismic_data(shots, solver_fd_cpp, ps=ps_fd_cpp)
#	print 'Data generation: {0}s'.format(time.time()-tt)
#	
#	import pickle
#	
#	with open('foo.pkl','w') as output:
##		pickle.dump(receivers, output)
#		pickle.dump(source, output)
#	
