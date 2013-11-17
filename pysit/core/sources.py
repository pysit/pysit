import itertools

import numpy as np
import scipy.sparse as spsp
from scipy.interpolate import interp1d

from mesh_representation import MeshRepresentationBase, PointRepresentationBase, PlaneRepresentationBase

__all__ = ['PointSource', 'SourceSet']

__docformat__ = "restructuredtext en"

class SourceBase(object):
	"""Base class for representing a source emitter on a grid.
		
	Methods
	-------
	f(t, **kwargs)
		Evaluate w on grid numerically, must be implemented by sub class.
		
	Notes
	-----
	`intensity` could conceivably become a function of time, in the future.
		
	"""
	def __init__(self, **kwargs):
		"""Constructor for the SourceBase class.
		"""
	
		self.shot=None
		
	def get_source_count(self):
		return 1
	source_count = property(get_source_count, None, None, None)
		
	def set_shot(self,shot):
		self.shot=shot
		
	def reset_time_series(self, ts):
		pass
		
	def f(self, t, **kwargs):
		raise NotImplementedError('Evaluation function \'f\' must be implemented by subclass.')
		
	def w(self, *argsw, **kwags):
		raise NotImplementedError('Wavelet function must be implemented at the subclass level.')
		
	# For subclasses to implement.
	def serialize_dict(self, *args, **kwargs):
		raise NotImplementedError()
	
	def unserialize_dict(self, d):
		raise NotImplementedError()
	
class PointSource(PointRepresentationBase, SourceBase):
	"""Subclass of PointRepresentationBase and SourceBase for representing a 
	point source emitter on a grid.
	
	Attributes
	----------
	domain : pysit.Domain
		Inherited from base class.
	position : tuple of float
		Inherited from base class.
	sampling_operator : scipy.sparse matrix
		Inherited from base class.
	adjoint_sampling_operator : scipy.sparse matrix
		Inherited from base class.
	intensity : float, optional
		Intensity of the source wavelet.
	w : function or function object
		Function of time that produces the source wavelet.	
		
	Methods
	-------
	f(t, **kwargs)
		Evaluate w(t)*delta(x-x') numerically.
		
	"""

	def __init__(self, mesh, pos, src_func, intensity=1.0, **kwargs):
		"""Constructor for the PointSource class.
		
		Parameters
		----------
		mesh : pysit.Mesh
			Computation domain on which the source is defined.
		position : tuple of float
			Coordinates of the point in the physical coordinates of the domain.
		src_func : function or function object
			Function of time that produces the source wavelet.	
		intensity : float, optional
			Intensity of the source wavelet.
		**kwargs : dict, optional
			May be used to specify `approximation` and `approximation_width` to
			base class.
		"""
	
		# Populate parameters from the base class.
		PointRepresentationBase.__init__(self, mesh, pos, **kwargs)
		SourceBase.__init__(self, **kwargs)
		
		self.w = src_func
		self.intensity = intensity
		
	def f(self, t=0.0, nu=None, **kwargs):
		"""Evaluate source emitter at time t on numerical grid.
		
		Parameters
		----------
		t : float
			Time at which to evaluate the source wavelet.
		**kwargs : dict, optional
			May pass additional parameters to the source wavelet call.
			
		Returns
		-------
		The function w evaluated on a grid as an ndarray shaped like the domain.
		"""
		if nu is None:
			if self._sample_interp_method == 'sparse':
				return (self.adjoint_sampling_operator*(self.intensity*self.w(t, **kwargs))).toarray().reshape(self.mesh.shape())
			else:	
				return (self.adjoint_sampling_operator*(self.intensity*self.w(t, **kwargs))).reshape(self.mesh.shape())
		else:
			if self._sample_interp_method == 'sparse':
				return (self.adjoint_sampling_operator*(self.intensity*self.w(nu=nu, **kwargs))).toarray().reshape(self.mesh.shape())
			else:	
				return (self.adjoint_sampling_operator*(self.intensity*self.w(nu=nu, **kwargs))).reshape(self.mesh.shape())
		
	def serialize_dict(self, i=None):
		
		ret = dict()
		ret['approximation'] = self.approximation
		ret['approximation_width'] = self.approximation_width
		ret['approximation_deviations'] = self.approximation_deviations
		ret['intesity'] = self.intensity
		ret['position'] = np.array(self.position)
		ret['w_peak_frequency'] = self.w.peak_frequency
		ret['w_t_shift'] = self.w.t_shift
		ret['source_count'] = self.source_count
		return ret
	
	def unserialize_dict(self, d):
		raise NotImplementedError()
	
class PlaneSource(PlaneRepresentationBase, SourceBase):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError()
	
class SourceSet(SourceBase, MeshRepresentationBase):

	def __init__(self, mesh, sources, **kwargs):
		"""Constructor for the SourceSet class.
		
		Parameters
		----------
		mesh : pysit.Mesh
			Physical (and numerical) domain on which the source is defined.
		sources : list of PointSource objects
		**kwargs : dict, optional
			May be used to specify `approximation` and `approximation_width` to
			base class.
		"""
	
		self.source_list = sources
		# Populate parameters from the base class.
		SourceBase.__init__(self, **kwargs)
		MeshRepresentationBase.__init__(self, mesh, **kwargs)
		
		# Create the basis array
		if self._sample_interp_method == 'sparse':
			self.sampling_operator = spsp.vstack([s.sampling_operator for s in self.source_list])
			self.adjoint_sampling_operator = spsp.hstack([s.adjoint_sampling_operator for s in self.source_list])
		else: # dense
			self.sampling_operator = np.vstack([s.sampling_operator for s in self.source_list])
			self.adjoint_sampling_operator = np.hstack([s.adjoint_sampling_operator for s in self.source_list])
		
	def get_source_count(self):
		return sum([r.source_count for r in self.source_list])
	source_count = property(get_source_count, None, None, None)
		
	def w(self, t, **kwargs):
		vec = np.array([s.intensity*s.w(t,**kwargs) for s in self.source_list])
		vec.shape = vec.size,1
		return vec
		
	def set_shot(self,shot):
		self.shot=shot
		for s in self.source_list:
			s.set_shot(shot)
		
		
	def f(self, t=0.0, nu=None, **kwargs):
		"""Evaluate source emitter at time t on numerical grid.
		
		Parameters
		----------
		t : float
			Time at which to evaluate the source wavelet.
		nu : float, optional
			Frequency at which to evaluate the wavelets
		**kwargs : dict, optional
			May pass additional parameters to the source wavelet call.
			
		Returns
		-------
		The function w evaluated on a grid as an ndarray shaped like the domain.
		"""
		v = self.w(t, nu=nu, **kwargs)
		if self._sample_interp_method == 'sparse':
			return (self.adjoint_sampling_operator*v).reshape(self.mesh.shape())
		else:	
			return np.dot(self.adjoint_sampling_operator,v).reshape(self.mesh.shape())
		
	def serialize_dict(self):
		
		ret = dict()
		ret['source_count'] = self.source_count
		
		srcdicts = np.zeros(self.source_count, dtype=np.object)
		
		for src,i in zip(self.source_list, itertools.count()):
			srcdicts[i] = src.serialize_dict(i)
			
		ret['sources'] = srcdicts #np.array(recdicts)
		
		return ret
	
	def unserialize_dict(self, d):
		raise NotImplementedError()
		
#		
#
#if __name__ == '__main__':
#
#	from pysit import Domain, PML, RickerWavelet, Shot
#	
#	# from source_receiver import ReceiverSet, PointSource, PointReceiver
#	
#	pmlx = PML(0.1, 100)
#	pmlz = PML(0.1, 100)
#	
#	x_config = (0.1, 1.0, 90, pmlx, pmlx)
#	z_config = (0.1, 0.8, 70, pmlz, pmlz)
#
#	d = Domain( (x_config, z_config) )
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
#		shot = Shot(source, ReceiverSet(receivers))
#		shots.append(shot)
#		
