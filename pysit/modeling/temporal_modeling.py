from __future__ import absolute_import

import numpy as np

__all__ = ['TemporalModeling']

__docformat__ = "restructuredtext en"

class TemporalModeling(object):
	"""Class containing a collection of methods needed for seismic inversion in 
	the time domain.
	
	This collection is designed so that a collection of like-methods can be 
	passed to an optimization routine, changing how we compute each part, eg, in 
	time, frequency, or the Laplace domain, without having to reimplement the 
	optimization routines.
	
	A collection of inversion functions must contain a procedure for computing:
	* the foward model: apply script_F (in our notation)
	* migrate: apply F* (in our notation)
	* demigrate: apply F (in our notation)
	* Hessian?
	
	Attributes
	----------
	solver : pysit wave solver object
		A wave solver that inherits from pysit.solvers.WaveSolverBase
	
	"""

	# read only class description
	@property
	def solver_type(self): return "time"
	@property
	def modeling_type(self): return "time"
		
	def __init__(self, solver):
		"""Constructor for the TemporalInversion class.
		
		Parameters
		----------
		solver : pysit wave solver object
			A wave solver that inherits from pysit.solvers.WaveSolverBase
		 
		"""
		
		if self.solver_type == solver.solver_type:
			self.solver = solver
		else:
			raise TypeError("Argument 'solver' type {1} does not match modeling solver type {0}.".format(self.solver_type, solver.solver_type)) 

	def _setup_forward_rhs(self, rhs_array, data):
		return self.solver.mesh.pad_array(data, out_array=rhs_array)
		
	def forward_model(self, shot, m0, return_parameters=[]):
		"""Applies the forward model to the model for the given solver.
		
		Parameters
		----------
		shot : pysit.Shot
			Gives the source signal approximation for the right hand side.
		m0 : solver.ModelParameters
			The parameters upon which to evaluate the forward model.
		return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}
		
		Returns
		-------
		retval : dict
			Dictionary whose keys are return_parameters that contains the specified data.
		
		Notes
		-----
		* u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.  
		* utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.
		
		"""

		# Local references
		solver = self.solver
		solver.model_parameters = m0
		
		mesh = solver.mesh
		
		d = solver.domain
		dt = solver.dt
		nsteps = solver.nsteps
		source = shot.sources

		# Storage for the field
		if 'wavefield' in return_parameters:
			us = list()
		
		# Setup data storage for the forward modeled data
		if 'simdata' in return_parameters:
			simdata = np.zeros((solver.nsteps, shot.receivers.receiver_count))			

		# Storage for the time derivatives of p
		if 'dWaveOp' in return_parameters:
			dWaveOp = list()
		
		# Step k = 0
		# p_0 is a zero array because if we assume the input signal is causal
		# and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
		# uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
		# well. ukm1 is needed to compute the temporal derivative.
		solver_data = solver.SolverData()
		
		rhs_k   = np.zeros(mesh.shape(include_bc=True))
		rhs_kp1 = np.zeros(mesh.shape(include_bc=True))
		
		for k in xrange(nsteps):
		
			uk = solver_data.k.primary_wavefield
			uk_bulk = mesh.unpad_array(uk)
			
			if 'wavefield' in return_parameters:
				us.append(uk_bulk.copy())
				
			# Record the data at t_k
			if 'simdata' in return_parameters:
				shot.receivers.sample_data_from_array(uk_bulk, k, data=simdata)
			
			if k == 0:
				rhs_k = self._setup_forward_rhs(rhs_k, source.f(k*dt))
				rhs_kp1 = self._setup_forward_rhs(rhs_kp1, source.f((k+1)*dt))
			else:
				# shift time forward
				rhs_k, rhs_kp1 = rhs_kp1, rhs_k
			rhs_kp1 = self._setup_forward_rhs(rhs_kp1, source.f((k+1)*dt))
			
			# Note, we compute result for k+1 even when k == nsteps-1.  We need
			# it for the time derivative at k=nsteps-1.
			solver.time_step(solver_data, rhs_k, rhs_kp1)

			# Compute time derivative of p at time k
			# Note that this is is returned as a PADDED array
			if 'dWaveOp' in return_parameters:
				dWaveOp.append(solver.compute_dWaveOp('time', solver_data))
		
			# When k is the nth step, the next step is uneeded, so don't swap 
			# any values.  This way, uk at the end is always the final step
			if(k == (nsteps-1)): break
			
			# Don't know what data is needed for the solver, so the solver data
			# handles advancing everything forward by one time step.
			# k-1 <-- k, k <-- k+1, etc
			solver_data.advance()
			
		retval = dict()
		
		if 'wavefield' in return_parameters:
			retval['wavefield'] = us
		if 'dWaveOp' in return_parameters:
			retval['dWaveOp'] = dWaveOp
		if 'simdata' in return_parameters:
			retval['simdata'] = simdata
		
		return retval

	def migrate_shot(self, shot, m0, 
	                       operand_simdata, operand_dWaveOpAdj=None, operand_model=None, 
	                       dWaveOp=None,
	                       adjointfield=None, dWaveOpAdj=None):
		"""Performs migration on a single shot.
		
		Parameters
		----------
		shot : pysit.Shot
			Shot for which to compute migration.
		operand : darray
			Operand, i.e., b in (F^*)b.
		dWaveOp : list
			Imaging condition components from the forward model for each receiver in the shot.
		qs : list
			Optional return list allowing us to retrieve the adjoint field as desired.
		 
		"""
		
		# If the imaging component has not already been computed, compute it.
		if dWaveOp is None:
			retval = self.forward_model(shot, m0, return_parameters=['dWaveOp'])
			dWaveOp = retval['dWaveOp']
		
		rp = ['imaging_condition']
		if adjointfield is not None: 
			rp.append('adjointfield')
		if dWaveOpAdj is not None:
			rp.append('dWaveOpAdj')
		
		rv = self.adjoint_model(shot, m0, operand_simdata, operand_dWaveOpAdj, operand_model, return_parameters=rp, dWaveOp=dWaveOp)
		
		# If the adjoint field is desired as output.
		if adjointfield is not None:
			adjointfield[:] = rv['adjointfield'][:]
		if dWaveOpAdj is not None:
			dWaveOpAdj[:] = rv['dWaveOpAdj'][:]
		
		# Get the imaging condition part from the result, this is the migrated image.
		ic = rv['imaging_condition']
		
		# imaging condition is padded, but migration yields an unpadded return
		return ic.without_padding()

	def _setup_adjoint_rhs(self, rhs_array, shot, k, operand_simdata, operand_model, operand_dWaveOpAdj):
		
		# basic rhs is always the pseudodata or residual
		rhs_array = self.solver.mesh.pad_array(shot.receivers.extend_data_to_array(k, data=operand_simdata), out_array=rhs_array)
		
		# for Hessians, sometimes there is more to the rhs
		if (operand_dWaveOpAdj is not None) and (operand_model is not None):
			rhs_array += operand_model*operand_dWaveOpAdj[k]
		
		return rhs_array
		
	def adjoint_model(self, shot, m0, operand_simdata, operand_dWaveOpAdj=None, operand_model=None, return_parameters=[], dWaveOp=None):
		"""Solves for the adjoint field.
		
		m*q_tt - lap q = resid
		
		Parameters
		----------
		shot : pysit.Shot
			Gives the receiver model for the right hand side.
		operand_simdata : ndarray
			Right hand side component in the data space, usually the residual.
		operand_dWaveOpAdj : list of ndarray
			Right hand side component in the wave equation space, usually something to do with the imaging component...this needs resolving
		operand_simdata : ndarray
			Right hand side component in the data space, usually the residual.
		return_parameters : list of {'adjointfield', 'ic'}
		dWaveOp : ndarray
			Imaging component from the forward model.
			
		Returns
		-------
		retval : dict
			Dictionary whose keys are return_parameters that contains the specified data.
		
		Notes
		-----
		* q is the adjoint field.
		* ic is the imaging component.  Because this function computes many of 
		the things required to compute the imaging condition, there is an option 
		to compute the imaging condition as we go.  This should be used to save 
		computational effort.  If the imaging condition is to be computed, the
		optional argument utt must be present.
		
		"""
		
		# Local references
		solver = self.solver
		solver.model_parameters = m0
		
		mesh = solver.mesh
		
		d = solver.domain
		dt = solver.dt
		nsteps = solver.nsteps
		source = shot.sources

		if 'adjointfield' in return_parameters:
			qs = list()
			vs = list()
		
		# Storage for the time derivatives of p
		if 'dWaveOpAdj' in return_parameters:
			dWaveOpAdj = list()
		
		# If we are computing the imaging condition, ensure that all of the parts are there and allocate space.
		if dWaveOp is not None:
			ic = solver.model_parameters.perturbation()
			do_ic = True
		elif 'imaging_condition' in return_parameters:
			raise ValueError('To compute imaging condition, forward component must be specified.')
		else:
			do_ic = False

		# Time-reversed wave solver
		solver_data = solver.SolverData()
		
		rhs_k   = np.zeros(mesh.shape(include_bc=True))
		rhs_km1 = np.zeros(mesh.shape(include_bc=True))
		
		if operand_model is not None:
			operand_model = operand_model.with_padding()
		
		# Loop goes over the valid indices backwards
		for k in xrange(nsteps-1, -1, -1): #xrange(int(solver.nsteps)):
			
			# Local reference
			vk = solver_data.k.primary_wavefield
			vk_bulk = mesh.unpad_array(vk)
			
			# When dpdt is not set, store the current q, otherwise compute the
			# relevant gradient portion
			if 'adjointfield' in return_parameters:
				vs.append(vk_bulk.copy())
			
			# can maybe speed up by using only the bulk and not unpadding later
			if do_ic:
				ic += vk*dWaveOp[k]
			
			if k == nsteps-1:
				rhs_k   = self._setup_adjoint_rhs( rhs_k,   shot, k,   operand_simdata, operand_model, operand_dWaveOpAdj)
				rhs_km1 = self._setup_adjoint_rhs( rhs_km1, shot, k-1, operand_simdata, operand_model, operand_dWaveOpAdj)
			else:
				# shift time forward
				rhs_k, rhs_km1 = rhs_km1, rhs_k
			rhs_km1 = self._setup_adjoint_rhs( rhs_km1, shot, k-1, operand_simdata, operand_model, operand_dWaveOpAdj)
			
			solver.time_step(solver_data, rhs_k, rhs_km1)
			
			# Compute time derivative of p at time k
			if 'dWaveOpAdj' in return_parameters:
				dWaveOpAdj.append(solver.compute_dWaveOp('time', solver_data))
		
			# If k is 0, we don't need results for k-1, so save computation and
			# stop early
			if(k == 0): break
				
			# Don't know what data is needed for the solver, so the solver data
			# handles advancing everything forward by one time step.
			# k-1 <-- k, k <-- k+1, etc
			solver_data.advance()
	
		if do_ic:
			ic *= (-1*dt)	
			ic = ic.without_padding() # gradient is never padded		
		
		retval = dict()
		
		if 'adjointfield' in return_parameters:
			# List of qs is built in time reversed order, put them in time forward order
			qs = list(vs)
			qs.reverse()
			retval['adjointfield'] = qs
		if 'dWaveOpAdj' in return_parameters:
			dWaveOpAdj.reverse()
			retval['dWaveOpAdj'] = dWaveOpAdj
		
		if do_ic:
			retval['imaging_condition'] = ic
		
		return retval

	def linear_forward_model(self, shot, m0, m1, return_parameters=[], dWaveOp0=None):
		"""Applies the forward model to the model for the given solver.
		
		Parameters
		----------
		shot : pysit.Shot
			Gives the source signal approximation for the right hand side.
		m0 : solver.ModelParameters
			The parameters upon where to center the linear approximation.
		m1 : solver.ModelParameters
			The parameters upon which to apply the linear forward model to.
		return_parameters : list of {'wavefield1', 'dWaveOp1', 'dWaveOp0', 'simdata'}
			Values to return.
		u0tt : ndarray
			Derivative field required for the imaging condition to be used as right hand side.
		
		
		Returns
		-------
		retval : dict
			Dictionary whose keys are return_parameters that contains the specified data.
		
		Notes
		-----
		* u1 is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.  
		* u1tt is used to generically refer to the derivative of u1 that is needed to compute the imaging condition.
		* If u0tt is not specified, it may be computed on the fly at potentially high expense.
		
		"""

		# Local references
		solver = self.solver
		solver.model_parameters = m0 # this updates dt and the number of steps so that is appropriate for the current model
		
		mesh = solver.mesh
		
		d = solver.domain
		dt = solver.dt
		nsteps = solver.nsteps
		source = shot.sources

		m1_padded = m1.with_padding()

		# Storage for the field
		if 'wavefield1' in return_parameters:
			us = list()
		
		# Setup data storage for the forward modeled data
		if 'simdata' in return_parameters:
			simdata = np.zeros((solver.nsteps, shot.receivers.receiver_count))

		# Storage for the time derivatives of p
		if 'dWaveOp0' in return_parameters:
			dWaveOp0ret = list()
			
		if 'dWaveOp1' in return_parameters:
			dWaveOp1 = list()
		
		# Step k = 0
		# p_0 is a zero array because if we assume the input signal is causal
		# and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
		# uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
		# well. ukm1 is needed to compute the temporal derivative.
		solver_data = solver.SolverData()
		
		if dWaveOp0 is None:
			solver_data_u0 = solver.SolverData()
			
			# For u0, set up the right hand sides
			rhs_u0_k   = np.zeros(mesh.shape(include_bc=True))
			rhs_u0_kp1 = np.zeros(mesh.shape(include_bc=True))				
			rhs_u0_k   = self._setup_forward_rhs(rhs_u0_k,   source.f(0*dt))
			rhs_u0_kp1 = self._setup_forward_rhs(rhs_u0_kp1, source.f(1*dt))
			
			# compute u0_kp1 so that we can compute dWaveOp0_k (needed for u1)
			solver.time_step(solver_data_u0, rhs_u0_k, rhs_u0_kp1)
			
			# compute dwaveop_0 (k=0) and allocate space for kp1 (needed for u1 time step)
			dWaveOp0_k = solver.compute_dWaveOp('time', solver_data_u0)			
			dWaveOp0_kp1 = dWaveOp0_k.copy()
			
			solver_data_u0.advance()
			# from here, it makes more sense to refer to rhs_u0 as kp1 and kp2, because those are the values we need 
			# to compute u0_kp2, which is what we need to compute dWaveOp0_kp1
			rhs_u0_kp1, rhs_u0_kp2 = rhs_u0_k, rhs_u0_kp1 # to reuse the allocated space and setup the swap that occurs a few lines down
			
		else:
			solver_data_u0 = None

		for k in xrange(nsteps):
			uk = solver_data.k.primary_wavefield
			uk_bulk = mesh.unpad_array(uk)
		
			if 'wavefield1' in return_parameters:
				us.append(uk_bulk.copy())
				
			# Record the data at t_k
			if 'simdata' in return_parameters:
				shot.receivers.sample_data_from_array(uk_bulk, k, data=simdata)
			
			# Note, we compute result for k+1 even when k == nsteps-1.  We need
			# it for the time derivative at k=nsteps-1.
			if dWaveOp0 is None:
				# compute u0_kp2 so we can get dWaveOp0_kp1 for the rhs for u1
				rhs_u0_kp1, rhs_u0_kp2 = rhs_u0_kp2, rhs_u0_kp1
				rhs_u0_kp2 = self._setup_forward_rhs(rhs_u0_kp2, source.f((k+2)*dt))
				solver.time_step(solver_data_u0, rhs_u0_kp1, rhs_u0_kp2)
				
				# shift the dWaveOp0's (ok at k=0 because they are equal then)
				dWaveOp0_k, dWaveOp0_kp1 = dWaveOp0_kp1, dWaveOp0_k
				dWaveOp0_kp1 = solver.compute_dWaveOp('time', solver_data_u0)
				
				solver_data_u0.advance()
			else:
				dWaveOp0_k = dWaveOp0[k]
				dWaveOp0_kp1 = dWaveOp0[k+1] if k < (nsteps-1) else dWaveOp0[k] # incase not enough dWaveOp0's are provided, repeat the last one
			
			if 'dWaveOp0' in return_parameters:
				dWaveOp0ret.append(dWaveOp0_k)	
			
			if k == 0:
				rhs_k   = m1_padded*(-1*dWaveOp0_k)
				rhs_kp1 = m1_padded*(-1*dWaveOp0_kp1)
			else:
				rhs_k, rhs_kp1 = rhs_kp1, m1_padded*(-1*dWaveOp0_kp1)
			
			solver.time_step(solver_data, rhs_k, rhs_kp1)

			# Compute time derivative of p at time k
			if 'dWaveOp1' in return_parameters:
				dWaveOp1.append(solver.compute_dWaveOp('time', solver_data))
		
			# When k is the nth step, the next step is uneeded, so don't swap 
			# any values.  This way, uk at the end is always the final step
			if(k == (nsteps-1)): break
			
			# Don't know what data is needed for the solver, so the solver data
			# handles advancing everything forward by one time step.
			# k-1 <-- k, k <-- k+1, etc
			solver_data.advance()
			
		retval = dict()
		
		if 'wavefield1' in return_parameters:
			retval['wavefield1'] = us
		if 'dWaveOp0' in return_parameters:
			retval['dWaveOp0'] = dWaveOp0ret
		if 'dWaveOp1' in return_parameters:
			retval['dWaveOp1'] = dWaveOp1
		if 'simdata' in return_parameters:
			retval['simdata'] = simdata
		
		return retval

def adjoint_test():
#if __name__ == '__main__':
	import numpy as np
	from pysit import PML, RectangularDomain, CartesianMesh, PointSource, ReceiverSet, Shot, ConstantDensityAcousticWave, generate_seismic_data, PointReceiver, RickerWavelet
	from pysit.gallery import horizontal_reflector

	# Setup

	#	Define Domain
	pmlx = PML(0.1, 1000, ftype='quadratic')
	pmlz = PML(0.1, 1000, ftype='quadratic')
	
	x_config = (0.1, 1.0, pmlx, pmlx)
	z_config = (0.1, 0.8, pmlz, pmlz)

	d = RectangularDomain( x_config, z_config )
	
	m = CartesianMesh(d, 90, 70)

	#	Generate true wave speed
	#   (M = C^-2 - C0^-2)
	C0, C = horizontal_reflector(m)

	# Set up shots
	Nshots = 1
	shots = []
	
	xmin = d.x.lbound
	xmax = d.x.rbound
	nx   = m.x.n
	zmin = d.z.lbound
	zmax = d.z.rbound
	
	for i in xrange(Nshots):

		# Define source location and type
#		source = PointSource(d, (xmax*(i+1.0)/(Nshots+1.0), 0.1), RickerWavelet(10.0))
		source = PointSource(m, (.188888, 0.18888), RickerWavelet(10.0))
	
		# Define set of receivers
		zpos = zmin + (1./9.)*zmax
		xpos = np.linspace(xmin, xmax, nx)
		receivers = ReceiverSet(m, [PointReceiver(m, (x, zpos)) for x in xpos])
	
		# Create and store the shot
		shot = Shot(source, receivers)
		shots.append(shot)
	
	# Define and configure the wave solver	
	trange=(0.,3.0)
	solver = ConstantDensityAcousticWave(m, 
	                                     formulation='ode',
#	                                     formulation='scalar',
	                                     model_parameters={'C': C}, 
		                                 spatial_accuracy_order=4,
#		                                 spatial_shifted_differences=True,
#		                                 cfl_safety=0.01,
		                                 trange=trange,
		                                 time_accuracy_order=4)
	
	# Generate synthetic Seismic data
	np.random.seed(1)
	print('Generating data...')
	wavefields=[]
	base_model = solver.ModelParameters(m,{'C': C})
	generate_seismic_data(shots, solver, base_model, wavefields=wavefields)
	
	tools = TemporalModeling(solver)
	m0 = solver.ModelParameters(m,{'C': C0})
	
	m1 = m0.perturbation() 
	m1 += np.random.rand(*m1.data.shape)
	
	fwdret = tools.forward_model(shot, m0, ['wavefield', 'dWaveOp', 'simdata'])
	dWaveOp0 = fwdret['dWaveOp']
	inc_field = fwdret['wavefield']
	data = fwdret['simdata']
#	data += np.random.rand(*data.shape)
	
	linfwdret = tools.linear_forward_model(shot, m0, m1, ['simdata'])
	lindata = linfwdret['simdata']
	
	adjret = tools.adjoint_model(shot, m0, data, return_parameters=['imaging_condition', 'adjointfield'], dWaveOp=dWaveOp0)
	
	adjmodel = adjret['imaging_condition'].asarray()
	adj_field = adjret['adjointfield']
	m1 = m1.asarray()

	print data.shape, solver.nsteps
	print np.sum(data*lindata)*solver.dt
	print np.dot(m1.T, adjmodel).squeeze()*np.prod(m.deltas)
	print np.dot(m1.T, adjmodel).squeeze()*np.prod(m.deltas)-np.sum(data*lindata)*solver.dt

	qs = adj_field

	qhat = 0.0
	dt = solver.dt
	for k in xrange(solver.nsteps):
		t = k * dt
		
		qhat += qs[k]*(np.exp(-1j*2.0*np.pi*10.0*t)*dt)

if __name__ == '__main__':
	
	adjoint_test()			