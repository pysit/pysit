import itertools

import numpy as np
import scipy.sparse as spsp
from scipy.interpolate import interp1d

from mesh_representation import MeshRepresentationBase, PointRepresentationBase

__all__ = ['PointReceiver', 'ReceiverSet']

__docformat__ = "restructuredtext en"

class ReceiverBase(object):
	
	def __init__(self, time_window=None, directwave_muting=None, **kwargs):
		
		self.ts = None
		self.interpolator = None
		self._data = None
		self.shot=None
		
		# time_window is an n-tuple, n[0] is the type, and any remaining entries are type dependent.
		if time_window is None:
			self._time_window = ('None',)
		else:
			self._time_window = time_window
			
#		# directwave_muting is an n-tuple, n[0] is the type, and any remaining entries are type dependent.
#		if directwave_muting is None:
#			self._directwave_muting = ('None',)
#		else:
#			self._directwave_muting = directwave_muting
		
	def set_shot(self,shot):
		self.shot=shot
	
	def get_receiver_count(self):
		return 1
	receiver_count = property(get_receiver_count, None, None, None)
	
	def get_data(self):
		return self._data
	def set_data(self, ndata):
		self._data = ndata
	data = property(get_data, set_data, None, None)	
	
	def clear_data(self, length):
		raise NotImplementedError('\'clear_data\' method must be implemented by subclass.')
	
	def sample_data_from_array(self, p, k=None, data=None):
		raise NotImplementedError('\'sample_data_from_array\' method must be implemented by subclass.')
		
	def extend_data_to_array(self, k, resid=False, data=None):
		raise NotImplementedError('\'extend_data_to_array\' method must be implemented by subclass.')
		
	def reset_time_series(self, ts):
		"""Rebuilds the interpolation object for a new time series.
		
		Parameters
		----------
		ts : ndarray
			The time series to reset things around.
		
		"""
		
		self.ts = ts
		self.clear_data(len(ts))
		
		self.interpolator = interp1d(ts, np.zeros_like(self.data), axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
	
	def compute_data_dft(self, frequencies):
		raise NotImplementedError('\'compute_data_dft\' method must be implemented by subclass.')
		
	
	def interpolate_data(self, ts):
		"""Interpolates the stored measured data to a new time series.
		
		Parameters
		----------
		ts : float or ndarray
			Time(s) at which to interpolate the measured data to.
		
		"""
		
		# Get a local reference to the shot's interpolation function
		interp = self.interpolator
		
		if interp is not None:			
			# Reset the y data reference for the interpolator to use the data 
			# for the current receiver. The default scipy interpolator uses x 
			# for the time series and y = f(x), so y is the data to interpolate.
			interp.y = self.data.T # this is how things are needed for pre-0.12 scipy
			interp._y = self.data # fix for a regression error in scipy 0.12
			d = interp(ts)
			
			if self._time_window[0] != 'None':
				d *= self.time_window(ts)
#			if self._directwave_muting[0] != 'None':
#				d *= self.directwave_mute(ts)
						
			return d
			
			
		else:
			raise TypeError('Interpolator has not been defined for the current receiver.')
			
	def time_window(self, ts):
	
		window_type = self._time_window[0]
		
		if window_type == 'Box':
			lb = self._time_window[1]
			rb = self._time_window[2]
			if lb is None:
				res = 1.0*(ts < rb)
			elif rb is None:
				res = 1.0*(ts > lb)
			else:
				res = 1.0*((ts > lb) & (ts < rb))
		elif window_type == 'Gaussian':
			t0 = self._time_window[1]
			s = self._time_window[2]
			res = np.exp(-(ts-t0)**2 / s**2)
		else: #if window_type == 'None':
			res = 1.0
			
		return res
			
#	def directwave_mute(self, ts):
#	
#		window_type = self._directwave_muting[0]
#		
#		if window_type == 'Box':
#			lb = self._directwave_muting[1]
#			res = 1.0*(ts > lb)
##		elif window_type == 'Gaussian':
##			t0 = self._time_window[1]
##			s = self._time_window[2]
##			res = np.exp(-(ts-t0)**2 / s**2)
#		else: #if window_type == 'None':
#			res = 1.0
#			
#		return res
		
	# For subclasses to implement.
	def serialize_dict(self, *args, **kwargs):
		raise NotImplementedError()
	
	def unserialize_dict(self, d):
		raise NotImplementedError()
		
			
		

class PointReceiver(PointRepresentationBase, ReceiverBase):
	"""Subclass of PointRepresentationBase and ReceiverBase for representing a 
	seismic receiver on a grid.
	
	Attributes
	----------
	mesh : pysit.Mesh
		Inherited from base class.
	domain : pysit.Domain
		Inherited from base class.
	position : tuple of float
		Inherited from base class.
	sampling_operator : scipy.sparse matrix
		Inherited from base class.
	adjoint_sampling_operator : scipy.sparse matrix
		Inherited from base class.
	data : numpy.ndarray
		Array of seismic data.
	
	Methods
	-------
	sample_data_from_array(p, k, m, data=None, record_interpolation='nearest')
		Record data from array p at point self.pos at time index k.
	extend_data_to_array(k, resid=False, data=None)
		Puts `self.data[k]` or `data[k]` or `self.data[k]-data[k]` on the grid.
		
	"""

	def __init__(self, mesh, pos, **kwargs):
		"""Constructor for the PointReceiver class.
		
		Parameters
		----------
		mesh : pysit.Mesh
			Computation domain on which the source is defined.
		position : tuple of float
			Coordinates of the point in the physical coordinates of the domain.	
		**kwargs : dict, optional
			May be used to specify `approximation` and `approximation_width` to
			base class.
		"""
	
		# Populate parameters from the base class.
		PointRepresentationBase.__init__(self, mesh, pos, **kwargs)
		ReceiverBase.__init__(self, **kwargs)
		
		self.data_dft = dict()
		
	def clear_data(self, data_length):
		"""Generate an empty data array of appropriate length.
		
		Parameters
		----------
		data_length : int
			Length of the data array.
		"""
		self.data = np.zeros((data_length,1))
		self.data_dft = dict()
		
	def sample_data_from_array(self, arr, k=None, nu=None, data=None):
		"""Generate an empty data array of appropriate length.
		
		Parameters
		----------
		arr : numpy.ndarray
			Array of values on domain.
		k : int, optional
			Time-index of data to record.  If none, recored data are returned.
		data : numpy.ndarray, optional
			Optional storage location for recorded data.
			
		Notes
		-----
		`k` is the index into the data array.  It is up to the programmer
		accessing this to ensure that `k` corresponds to the correct `t`.
		
		Providing the optional data argument will store the result in the
		provided array, rather than self.data.
		
		"""

		if self._sample_interp_method == 'sparse':
			v = self.sampling_operator * arr
			v = v[0,0]
		else:	
			v = np.dot(self.sampling_operator, arr) # for numpy arrays
			v = v[0,0]
		
		time_ = k is not None and nu is None 
		freq_ = k is None and nu is not None
		bail_ = k is not None and nu is not None
		
		if bail_:
			ValueError('Both k and nu cannot be specified.')
		
		if time_:
			if k is not None:
				if data is None:
					self.data[k] = v
				else:
					data[k] = v
		elif freq_:
			if data is None:
				self.data_dft[nu] = v
			else:
				data[nu] = v
		else:
			return v
			
	def extend_data_to_array(self, k=None, resid=False, data=None):
		"""Place data on the grid using the specified numerical delta
		approximation.
		
		This routine is designed to help form right-hand-side vectors for the
		wave equation.
		
		Parameters
		----------
		k : int
			Time-index of data to place on the grid.
		resid : bool, optional
			Compute with self.data - data.
		data : numpy.ndarray, optional
			Optional source location for data.
			
		Notes
		-----
		`k` is the index into the data array.  It is up to the programmer
		accessing this to ensure that `k` corresponds to the correct `t`.
		
		Providing the optional data argument will read the source from the
		provided array, rather than self.data.
		
		Setting `resid` to `True` will place the difference of data and
		self.data on the grid, rather than data or self.data.
		
		"""
		if k is not None:
			if(data is None):
				d = self.data[k]
			else:
				if(resid==True):
					d = (self.data[k] - data[k])
				else:
					d = data[k]
		elif data is not None:
			d = data
		
		if self._time_window[0] != 'None':
			d *= self.time_window(self.ts[k])
#		if self._directwave_muting[0] != 'None':
#			d *= self.directwave_mute(self.ts[k])
				
		if self._sample_interp_method == 'sparse':
			return (self.adjoint_sampling_operator*d).toarray().reshape(self.mesh.shape())
		else:
			return (self.adjoint_sampling_operator*d).reshape(self.mesh.shape())

	def compute_data_dft(self, frequencies, force_computation=False):
		""" Precompute the DFT of the data at the given list of frequencies.
		
		Parameters
		----------
		frequencies : float, iterable
			The frequency or frequencies for which to compute the DFT.
		force_computation : bool {optional}
			Force computation of DFT.  By default already computed frequencies are not recomputed.
		
		Notes
		-----
		
		* These computed values are not passed down the line to other receivers in the set.
		"""
		
		for nu in frequencies:
			if force_computation or (nu not in self.data_dft):
				self.data_dft[nu] = 0.0
		
		if self.ts is not None:
			dt = self.ts[1]-self.ts[0]
			
			for t,k in zip(self.ts, itertools.count()):
				for nu in frequencies:
					if force_computation or (nu not in self.data_dft):
						self.data_dft[nu] += self.data[k]*self.np.exp(-1j*2*np.pi*nu*t)*dt
	
	def serialize_dict(self, i=None):
		
		ret = dict()
		ret['approximation'] = self.approximation
		ret['approximation_width'] = self.approximation_width
		ret['approximation_deviations'] = self.approximation_deviations
		if i is None:
			ret['data'] = self.data
			ret['data_status'] = 'actual'
			ret['ts'] = self.ts
		else:
			ret['data'] = i
			ret['data_status'] = 'column reference'
		ret['position'] = np.array(self.position)
		ret['receiver_count'] = self.receiver_count
		return ret
	
	def unserialize_dict(self, d):
		raise NotImplementedError()
		

class ReceiverSet(MeshRepresentationBase, ReceiverBase):
	"""Subclass of list and ReceiverBase for representing a set of seismic 
	receivers on a mesh.
	
	This is currently a very simple extension of list.  In the future, this
	must include overrides of all of the basic list functionality, so that 
	operations like appending and inserting correctly handle changes in the
	data storage and operations like __getslice__ returns another receiver
	set.
	
	Attributes
	----------
	sampling_operator : scipy.sparse matrix
		Linear operator describing how the set of receivers is represented on a mesh.
	adjoint_sampling_operator : scipy.sparse matrix
		The adjoint of the sampling operator.
	data : numpy.ndarray
		Array of seismic data.
	
	Methods
	-------
	sample_data_from_array(p, k, m, data=None, record_interpolation='nearest')
		Record data from array p at point self.pos at time index k.
	extend_data_to_array(k, resid=False, data=None)
		Puts `self.data[k]` or `data[k]` or `self.data[k]-data[k]` on the grid.
		
	"""

	def __init__(self, mesh, receivers, **kwargs):
		"""Constructor for the PointSource class.
		
		Parameters
		----------
		mesh : pysit.Mesh
			Computation domain on which the source is defined.
		position : tuple of float
			Coordinates of the point in the physical coordinates of the domain.	
		**kwargs : dict, optional
			May be used to specify `approximation` and `approximation_width` to
			base class.
		"""
	
		self.receiver_list = receivers
		# Populate parameters from the base class.
		ReceiverBase.__init__(self, **kwargs)
		MeshRepresentationBase.__init__(self, mesh, **kwargs)
		
		# time_window is an n-tuple, n[0] is the type, and any remaining entries are type dependent.
		self._time_window = ('Set',) # for special handling of ReceiverSets
#		# directwave_muting is an n-tuple, n[0] is the type, and any remaining entries are type dependent.
#		self._directwave_muting = ('Set',) # for special handling of ReceiverSets		
		
		# Create the basis array
		if self._sample_interp_method == 'sparse':
			self.sampling_operator = spsp.vstack([r.sampling_operator for r in self.receiver_list])
			self.adjoint_sampling_operator = spsp.hstack([r.adjoint_sampling_operator for r in self.receiver_list])
		else: # dense
			self.sampling_operator = np.vstack([r.sampling_operator for r in self.receiver_list])
			self.adjoint_sampling_operator = np.hstack([r.adjoint_sampling_operator for r in self.receiver_list])
			
		self.data_dft = dict()
		
	def set_shot(self,shot):
		self.shot=shot
		for r in self.receiver_list:
			r.set_shot(shot)
		
	def get_receiver_count(self):
		return sum([r.receiver_count for r in self.receiver_list])
	receiver_count = property(get_receiver_count, None, None, None)
	
	def clear_data(self, data_length):
		"""Generate an empty data array of appropriate length.
		
		Parameters
		----------
		data_length : int
			Length of the data array.
		"""
		self.data = np.zeros((data_length, self.receiver_count))
		self.data_dft = dict()
		
	def get_data(self):
		return self._data
	def set_data(self, ndata):
		self._data = ndata
		count = 0
		for r in self.receiver_list:
			r.data = self._data[:,count:count+r.receiver_count]
			count += r.receiver_count
	data = property(get_data, set_data, None, None)
		
	def get_interpolator(self):
		return self._interpolator
	def set_interpolator(self, interpolator):
		self._interpolator = interpolator
		for r in self.receiver_list:
			r.interpolator = interpolator
	interpolator = property(get_interpolator, set_interpolator, None, None)
			
	def sample_data_from_array(self, arr, k=None, nu=None, data=None):
		"""Generate an empty data array of appropriate length.
		
		Parameters
		----------
		arr : numpy.ndarray
			Array of values on domain.
		k : int, optional
			Time-index of data to record.  If none, recored data are returned.
		data : numpy.ndarray, optional
			Optional storage location for recorded data.
			
		Notes
		-----
		`k` is the index into the data array.  It is up to the programmer
		accessing this to ensure that `k` corresponds to the correct `t`.
		
		Providing the optional data argument will store the result in the
		provided array, rather than self.data.
		
		"""

		if self._sample_interp_method == 'sparse':
			v = self.sampling_operator * arr
			v.shape = 1,v.size
		else:	
			v = np.dot(self.sampling_operator, arr) #numpy arrays, not matrices
			v.shape = 1,v.size
		
		time_ = k is not None and nu is None 
		freq_ = k is None and nu is not None
		bail_ = k is not None and nu is not None
		
		if bail_:
			ValueError('Both k and nu cannot be specified.')
		
		if time_:
			if k is not None:
				if data is None:
					self.data[k,:] = v
				else:
					data[k,:] = v
		elif freq_:
			if data is None:
				self.data_dft[nu] = v
			else:
				data[nu] = v
		else:
			return v
			
	def extend_data_to_array(self, k=None, resid=False, data=None):
		"""Place data on the grid using the specified numerical delta
		approximation.
		
		This routine is designed to help form right-hand-side vectors for the
		wave equation.
		
		Parameters
		----------
		k : int
			Time-index of data to place on the grid.
		resid : bool, optional
			Compute with self.data - data.
		data : numpy.ndarray, optional
			Optional source location for data.
			
		Notes
		-----
		`k` is the index into the data array.  It is up to the programmer
		accessing this to ensure that `k` corresponds to the correct `t`.
		
		Providing the optional data argument will read the source from the
		provided array, rather than self.data.
		
		Setting `resid` to `True` will place the difference of data and
		self.data on the grid, rather than data or self.data.
		
		"""
		
		if k is not None:
			if(data is None):
				d = self.data[k,:]
			else:
				if(resid==True):
					d = (self.data[k,:] - data[k,:])
				else:
					d = data[k,:] #.copy()
		elif data is not None:
			d = data
		
		if self._time_window[0] != 'None':
			pass
#			d *= self.time_window(self.ts[k]) ## FIXME
		
#		if self._directwave_muting[0] != 'None':		
#			d *= self.directwave_mute(self.ts[k])
				
		d.shape = d.size,1
		
		if self._sample_interp_method == 'sparse':
			return (self.adjoint_sampling_operator*d).reshape(self.mesh.shape())
		else:
			return np.dot(self.adjoint_sampling_operator,d).reshape(self.mesh.shape())
			
	def time_window(self, ts):
	
		return np.array([r.time_window(ts) for r in self.receiver_list]).T
		
#	def directwave_mute(self, ts):
#		
#		return np.array([r.directwave_mute(ts) for r in self.receiver_list]).T

	def compute_data_dft(self, frequencies, force_computation=False):
		""" Precompute the DFT of the data at the given list of frequencies.
		
		Parameters
		----------
		frequencies : float, iterable
			The frequency or frequencies for which to compute the DFT.
		force_computation : bool {optional}
			Force computation of DFT.  By default already computed frequencies are not recomputed.
		
		Notes
		-----
		
		* These computed values are not passed down the line to other receivers in the set.
		"""
		
		reset=dict()
		for nu in frequencies:
			if force_computation or (nu not in self.data_dft):
				self.data_dft[nu] = np.zeros(self.receiver_count, dtype=np.complex)
				reset[nu] = True
			else:
				reset[nu] = False
		
		if self.ts is not None:
			dt = self.ts[1]-self.ts[0]
			
			for t,k in zip(self.ts, itertools.count()):
				for nu in frequencies:
					if reset[nu]:
						self.data_dft[nu] += self.data[k,:]*np.exp(-1j*2*np.pi*nu*t)*dt

	def serialize_dict(self):
		
		ret = dict()
			
		ret['data'] = self.data
		ret['receiver_count'] = self.receiver_count
		ret['ts'] = self.ts
		
		recdicts = np.zeros(self.receiver_count, dtype=np.object)
		
		for rec,i in zip(self.receiver_list, itertools.count()):
			recdicts[i] = rec.serialize_dict(i)
			
		ret['receivers'] = np.array(recdicts)
		
		return ret
	
	def unserialize_dict(self, d):
		raise NotImplementedError()
