from __future__ import absolute_import

import time
from itertools import repeat

import numpy as np

__all__ = ['generate_seismic_data', 'generate_shot_data_time', 'generate_shot_data_frequency']
__docformat__ = "restructuredtext en"

def generate_seismic_data(shots, solver, model, verbose=False, frequencies=None, **kwargs):
	"""Given a list of shots and a solver, generates seismic data.
	
	Parameters
	----------
	shots : list of pysit.Shot
		Collection of shots to be processed
	solver : pysit.WaveSolver
		Instance of wave solver to be used.
	**kwargs : dict, optional
		Optional arguments.
		
	Notes
	-----
	`kwargs` may be used to specify `C0` and `wavefields` to	`generate_shot_data`.
	
	"""
	
	if verbose:
		print('Generating data...')
		tt = time.time()
	
	for shot in shots:
		
		if solver.solver_type == "time":
			generate_shot_data_time(shot, solver, model, verbose=verbose, **kwargs)
		elif solver.solver_type == "frequency":
			if frequencies is None:
				raise TypeError('A frequency solver is passed, but no frequencies are given')
			generate_shot_data_frequency(shot, solver, model, frequencies, verbose=verbose, **kwargs)
		else:
			raise TypeError("A time or frequency solver must be specified.")
	
	if verbose:
		data_tt = time.time() - tt
		print 'Data generation: {0}s'.format(data_tt)
		print 'Data generation: {0}s/shot'.format(data_tt/len(shots))

def generate_shot_data_time(shot, solver, model, wavefields=None, wavefields_padded=None, verbose=False, **kwargs):
	"""Given a shots and a solver, generates seismic data at the specified
	receivers.
	
	Parameters
	----------
	shots : list of pysit.Shot
		Collection of shots to be processed
	solver : pysit.WaveSolver
		Instance of wave solver to be used.
	model_parameters : solver.ModelParameters, optional
		Wave equation parameters used for generating data.
	wavefields : list, optional
		List of wave states.
	verbose : boolean
		Verbosity flag.
	
	Notes
	-----
	An empty list passed as `wavefields` will be populated with the state of the wave
	field at each time index.
	
	"""

	solver.model_parameters = model

	# Ensure that the receiver data is empty.  And that interpolator is setup.
	# shot.clear_data(solver.nsteps)
	ts = solver.ts()
	shot.reset_time_series(ts)

	# Populate some stuff that will probably not exist soon anyway.
	shot.dt = solver.dt
	shot.trange = solver.trange

	if solver.solver_type != "time":
		raise TypeError('Solver must be a time solver to generate data.')
		
	if(wavefields is not None):
		wavefields[:] = []
	if(wavefields_padded is not None):
		wavefields_padded[:] = []

	#Frequently used local references are faster than remote references
	mesh = solver.mesh
	dt = solver.dt
	source = shot.sources

	# Step k = 0
	# p_0 is a zero array because if we assume the input signal is causal and we
	# assume that the initial system (i.e., p_(-2) and p_(-1)) is uniformly
	# zero, then the leapfrog scheme would compute that p_0 = 0 as well.
	# This is stored in this solver specific data structure.
	solver_data = solver.SolverData()
		
	rhs_k   = np.zeros(mesh.shape(include_bc=True))
	rhs_kp1 = np.zeros(mesh.shape(include_bc=True))
	
	# k is the t index.  t = k*dt.
	for k in xrange(solver.nsteps):
#		print "  Computing step {0}...".format(k)
		uk = solver_data.k.primary_wavefield
		
		# Extract the primary, non ghost or boundary nodes from the wavefield
		# uk_bulk is a view so no new storage
		# also, this is a bad way to do things.  Somehow the BC padding needs to be better hidden.  Maybe it needs to always be a part of the mesh?
		uk_bulk = mesh.unpad_array(uk)
	
		# Record the data at t_k
		shot.receivers.sample_data_from_array(uk_bulk, k, **kwargs)
		
		if(wavefields is not None):
			wavefields.append(uk_bulk.copy())
		if(wavefields_padded is not None):
			wavefields_padded.append(uk.copy())
	
		# When k is the nth step, the next time step is not needed, so save
		# computation and break out early.
		if(k == (solver.nsteps-1)): break
	
		if k == 0:
			rhs_k = mesh.pad_array(source.f(k*dt), out_array=rhs_k)
			rhs_kp1 = mesh.pad_array(source.f((k+1)*dt), out_array=rhs_kp1)
		else:
			# shift time forward
			rhs_k, rhs_kp1 = rhs_kp1, rhs_k
			rhs_kp1 = mesh.pad_array(source.f((k+1)*dt), out_array=rhs_kp1)
			
		# Given the state at k and k-1, compute the state at k+1
		solver.time_step(solver_data, rhs_k, rhs_kp1)
		
		# Don't know what data is needed for the solver, so the solver data
		# handles advancing everything forward by one time step.
		# k-1 <-- k, k <-- k+1, etc
		solver_data.advance()
		
def generate_shot_data_frequency(shot, solver, model, frequencies, verbose=False, **kwargs):
		"""most of this is copied from frequency_modeling.forward_model
		
		Parameters
		----------
		shot : pysit.Shot
			Gives the source signal approximation for the right hand side.
		frequencies : list of 2-tuples
			2-tuple, first element is the frequency to use, second element the weight.
		return_parameters : list of {'wavefield', 'simdata', 'simdata_time', 'dWaveOp'}
		
		Returns
		-------
		retval : dict
			Dictionary whose keys are return_parameters that contains the specified data.
		
		Notes
		-----
		* u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.  
		* uhat is used to generically refer to the DFT of u that is needed to compute the imaging condition.
		
		"""
	
		# Local references
		solver.model_parameters = model 
		
		mesh = solver.mesh
		
		source = shot.sources

		# Sanitize the input
		if not np.iterable(frequencies):
			frequencies = [frequencies]		

		solver_data = solver.SolverData()
		rhs = solver.WavefieldVector(mesh,dtype=solver.dtype) 
		for nu in frequencies:
			rhs = solver.build_rhs(mesh.pad_array(source.f(nu=nu)), rhs_wavefieldvector=rhs)
			solver.solve(solver_data, rhs, nu)
			uhat = solver_data.k.primary_wavefield
			
			# Record the data at frequency nu
			shot.receivers.sample_data_from_array(mesh.unpad_array(uhat), nu=nu)