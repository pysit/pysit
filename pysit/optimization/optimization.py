from __future__ import print_function
from __future__ import absolute_import

import sys
import time
import copy

import numpy as np

__all__=['OptimizationBase']

__docformat__ = "restructuredtext en"
	
class OptimizationBase(object):
	""" Base class for descent-like optimization routines.
	
	These are stateful algorithms. The current step, as well as the step index 
	are stored so that (in the future) further steps can be taken without 
	repeating computational effort.
	
	The basic structure of a pysit descent algorithm is focused on three 
	computational phases: computation of the residual, the gradient of the 
	objective function, and selection of a step direction based on this 
	information.  The rest of this class specifies the layout of these methods, 
	which will be useful in nearly all descent algorithms.  A subclass, 
	GradientDescent, will implement basic versions of these routines.  Other 
	algorithms should inherit from there to prevent excess code rewriting.
	
	A separate function method is provided for each of the three basic phases to 
	allow for overriding of the behavior of each.  The residual and gradient 
	computation are least likely to be changed, but the step selection may be 
	changed frequently.  For example, a gradient descent algorithm might 
	implement an adjustment that performs a line search along the gradient.  An 
	implementation of Newton's method would simply override the adjust method to 
	solve the Hessian equation.  Algorithms like CG or BFGS can also be 
	implemented in this manner.
	
	Attributes
	----------
	solver : pysit wave solver object
		A wave solver that inherits from pysit.solvers.WaveSolverBase
	ivnersion_methods : class
		A class containing all of the methods required to compute the inversion steps.
	verbose : bool
		Verbosity flag.
	xi : solver.WaveSolverParameters
		Current state of the unknowns.
	i : float
		Current iteration index.
	<blank>_history : list of tuples
		History of property (e.g., step length) with entries like the tuple (i, step_length_i) for index i.
	<blank>_frequency : int
		Iteration frequency at which to store a particular property.
	
	"""
	
	def __init__(self, objective):
		"""Constructor for the BasicDescentAlgorithm class.
		
		Parameters
		----------
		solver : pysit wave solver object
			A wave solver that inherits from pysit.solvers.WaveSolverBase
		inversion_methods : class
			A class containing all of the methods required to compute the inversion steps.
			
		Notes
		-----
		* InversionMethodsType is a data type that takes a wave solver object 
		  as a construction argument.  The collection of inversion methods will 
		  depend on the solver.
		* InversionMethodsType must have member functions that implement the 
		  basic wave imaging procedures, e.g., forward modeling, 
		  adjoint modeling, demigration, etc. 
		 
		"""
		self.objective_function = objective
		self.solver = objective.solver
		self.verbose = False
		
		self.use_parallel = objective.use_parallel()
		
		self.max_linesearch_iterations = 10
		
		self.logfile = sys.stdout
	
		
	def reset(self,
		      append_mode,
			  value_frequency=0,
			  gradient_frequency=0,
			  gradient_length_frequency=0,
			  step_frequency=0,
			  step_length_frequency=0,
			  residual_length_frequency=0,
			  objective_frequency=0,
			  run_time_frequency=0,
			  alpha_frequency=1,
			  *args, **kwargs):
		"""Resets the state of the optimization algorithm.
		
		Parameters
		----------
		value_frequency : int
			Iteration frequency that the value of the solution should be stored.
		gradient_frequency : int
			Iteration frequency that the gradient vector should be stored.
		step_frequency : int
			Iteration frequency that the step vector and step length should be stored.
		objective_frequency : int
			Iteration frequency that the value of the objective function should be stored.
			
		"""
		
		# if we are not appending reset things
		# if things have not been set yet, reset things	
		if not append_mode or not hasattr(self, 'iteration'):
			self.base_model = self.solver.ModelParameters(self.solver.mesh)
			self.iteration = 0
		
			# Reset the history lists
			self.init_history("value",              value_frequency)     
			self.init_history("gradient",           gradient_frequency)  
			self.init_history("gradient_length",    gradient_length_frequency)  
			self.init_history("step",               step_frequency)      
			self.init_history("step_length",        step_length_frequency)
			self.init_history("residual_length",    residual_length_frequency)    # Requires self.residual_norm to be implemented by residual class if L2 is not appropriate
			self.init_history("objective",          objective_frequency) 
			self.init_history("run_time",           run_time_frequency)  
			
			# All methods line search somehow
			self.init_history("alpha",       alpha_frequency)  
		
	
	def init_history(self, arg, freq):
		"""Initializes a history variable.
		
		Creates or overwrites an object attribute named arg_history and
		arg_frequency dynamically.  This allows for storing properties of some 
		descent algorithms that may not be relevant or exist in others.
		
		Parameters
		----------
		arg : string
			String prefix for naming the history and frequency variables.
		freq : int
			Frequency for storing the associated arg.
			
		"""
		
		setattr(self, arg + "_history", {})
		setattr(self, arg + "_frequency", freq)
	
	def query_store_history(self, arg, i):
		f = getattr(self, arg + "_frequency")
		# Only store the history if this index matches the frequency.
		return f and (not np.mod(i,f))
	
	def store_history(self, arg, i, val, force=False):
		"""Stores a data point for a history variable.
		
		To prevent repeated checks to see if a current iteration requires
		history storage, this function both checks to see if data should be 
		stored and actually stores it.
		
		Parameters
		----------
		arg : string
			String prefix for naming the history and frequency variables.
		i : int
			Index of the current iteration.
		val : arbitrary
			Value to be stored.
		force : boolean
			Force storage anyway.
			
		"""
		# only processor 0 should store anything
		if self.use_parallel and (self.objective_function.parallel_wrap_shot.rank != 0):
			return
			
		f = getattr(self, arg + "_frequency")
		# Only store the history if this index matches the frequency.
		if f and (force or not np.mod(i,f)):
			loc = getattr(self, arg + "_history")
#			if not loc.has_key(i):
#				loc[i] = []
#			# Always make a copy of things that are stored
#			loc[i].append(copy.deepcopy(val))
			if not loc.has_key(i):
				loc[i] = None
			# Always make a copy of things that are stored
			loc[i] = copy.deepcopy(val)
	
	def retrieve_history(self, arg):
		"""Convenience routine for extracting a given history.		
		
		Parameters
		----------
		arg : string
			String prefix for naming the history and frequency variables.
		
		Returns
		-------
		iters, data : list of int, list of type(data)
			If the history has been stored.
		None, None
			Otherwise
		
		"""
		f = getattr(self, arg + "_frequency")
		# Only store the history if this index matches the frequency.
		if f:
			hist = getattr(self, arg + "_history")
			return zip(*sorted(hist.items()))
		else:
			return None, None
			
	def _print(self, *args):
		# only processor 0 should store anything
		if self.use_parallel and (self.objective_function.parallel_wrap_shot.rank != 0):
			return
			
		if self.verbose:
			print(*args, file=self.logfile)
			
#   #### Actual optimization stuff below...
		
	def initialize(self, initial_value, **kwargs):		
		"""Handle any optimization loop initialization and verify any preconditions.
		Parameters
		----------
		initial_value : solver.ModelData
			Starting guess.
			
		"""
		# Generally, there will be an initial value, but just in case...
		self.base_model = copy.deepcopy(initial_value)
		
		self.solver.model_parameters = self.base_model
	
	def __call__(self, 
	             shots,
				 initial_value,
				 iteration_parameters,
				 line_search='backtrack',
				 verbose=False,
				 append=False,
				 status_configuration={},
				 **kwargs):
		"""The main function for executing a number of steps of the descent
		algorith.
		
		Most things can be done without directly overriding this function.
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute on.
		initial_value : solver.WaveParameters
			Initial guess for the iteration.
		iteration_parameters : int, iterable
			Loop iteration parameters, like number of steps or frequency sets.
		<blank>_frequency : int, optional kwarg
			Frequency with which to store histories.  Detailed in reset method.
		verbose : bool
			Verbosity flag.
			
		"""
				 
		self.reset(append, **status_configuration)
		
		self.verbose=verbose
		
		self.line_search = line_search	
		if type(line_search) is str:
			self.ls_method = line_search
			self.ls_config = None
		else: #assume line_search is tuple('method', config1, config2, ...)
			self.ls_method = line_search[0]
			self.ls_config = line_search[1:]
						
		self.initialize(initial_value, **kwargs)
		
		# valid ieration parameters:
		# int, e.g., iteration_parameters=4
		# iterable(int), e.g., iteration_parameters=[50,50,50] will run the loop 3 times with 50 iterations each
		# iterable( list(int, arguments)), e.g., iteration_parameters=[(50,[1,2,3,4,5]), (50,[6,7,8,9])] will run the loop twice, 50 times each, for the frequencies listed in arguments
		if np.iterable(iteration_parameters):
			for ip in iteration_parameters:
				if type(ip) in [tuple, list]:
					steps, arguments = ip
				elif type(iteration_parameters) is int:
					steps = ip
					arguments = {}
				else:
					raise ValueError('Invalid iteration parameter {0} detected.'.format(ip))
				
				# Call the inner loop
				self.inner_loop(shots, steps, objective_arguments=arguments, **kwargs)				
		else:
			if type(iteration_parameters) is int:
				# Call the inner loop
				steps=iteration_parameters
				self.inner_loop(shots, steps, **kwargs)				
			else:
				raise ValueError('Singular iteration parameters of type {0} are not permitted at this time.'.format(type(iteration_parameters))) #Floats as a convergence epsilon may happen, but nothing runs to convergence.
			
		# Return the current state at the end of the run
		return self.base_model

	def inner_loop(self, shots, steps, objective_arguments={}, **kwargs):
		"""Inner loop the optimization iteration
		
		This is a separate method so that the workings of the inner loop can be
		overridden without duplicating the wrapper code in the call function.
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute the residual.
		steps : int
			Number of iterations to run.
			
		"""
		
		for step in xrange(steps):
			# Zeroth step is always the initial condition.
			tt = time.time()
			i = self.iteration
			
			self.store_history('value', i, self.base_model)
			
			self._print('Iteration {0}'.format(i))
			
			self.solver.model_parameters = self.base_model
			
			# extra data to try to extract from gradient call
			aux_info = {'objective_value': (True, None),
						'residual_norm': (True, None)}
			
			# Compute the gradient			
			gradient = self.objective_function.compute_gradient(shots, self.base_model, aux_info=aux_info, **objective_arguments)
			objective_value = aux_info['objective_value'][1]
			
			# Process and store meta data about the gradient
			self.store_history('gradient', i, gradient)
			gradient_norm = gradient.norm()
			self._print('  gradnorm {0}'.format(gradient_norm))
			self.store_history('gradient_length', i, gradient_norm)
			
			if aux_info['objective_value'][1] is not None:
				self.store_history('objective', i, aux_info['objective_value'][1])
				self._print('  objective {0}'.format(aux_info['objective_value'][1]))
			
			if aux_info['residual_norm'][1] is not None:
				self.store_history('residual_length', i, aux_info['residual_norm'][1])
				self._print('  residual {0}'.format(aux_info['residual_norm'][1]))
			
			# Compute step modifier
			step = self._select_step(shots, objective_value, gradient, i, objective_arguments, **kwargs)
			
			# Process and store meta data about the step
			step_len = step.norm()
			self.store_history('step_length', i, step_len)
			self.store_history('step', i, step)
			
			# Apply new step
			self.base_model += step
			
			ttt = time.time()-tt
			self.store_history('run_time', i, ttt)
			
			self.iteration += 1
			
			self._print('  run time {0}s'.format(ttt))
	
	def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
		raise NotImplementedError("_select_step must be implemented by a subclass.")
		
	def select_alpha(self, shots, gradient, direction, objective_arguments, **kwargs):
		"""Resets the state of the optimization algorithm.
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute on.
		gradient : Solver.ModelData
			The gradient in model space.
		direction : Solver.ModelData
			The search direction in model space.
		method : {'constant', 'linear', 'quadratic', 'linesearch'}, optional
			The technique used to select alpha.
		alpha : float, optional
			The returned value for 'constant'.
			
		Returns
		-------
		alpha : float
			Line search parameter.		
			
		"""
		
		if self.ls_method == 'constant':
			return self._constant_line_search()
			
		elif self.ls_method == 'linear':
			return self._linear_line_search(shots, gradient, direction, objective_arguments, **kwargs)

		elif self.ls_method == 'backtrack':
			return self._backtrack_line_search(shots, gradient, direction, objective_arguments, **kwargs)
			
		else:
			raise ValueError('Alpha selection method {0} invalid'.format(self.ls_method))
			
	def _constant_line_search(self):
		alpha = self.ls_config[0]
		return alpha

	def _linear_line_search(self, shots, gradient, direction, objective_arguments, **kwargs):
		raise NotImplementedError('Linear selection of alpha is an objective function dependent operation.')
			
#		# \int{gradient*s}dx = -\int{gradient^2} = -\int{s^2}
#		d_norm = -1*np.linalg.norm(direction) * np.prod(self.solver.mesh.deltas)
#		
#		
#		# The commented out bit is probably the correct way to do things, 
#		# but it does not generalize between time and frequency due to 
#		# differences in the way the data are stored (eg, array, dict of 
#		# arrays, etc).  Also, the "linear" test is 
##			res = map(lambda x: self.objective_function.modeling_tools.linear_forward_model(x, self.base_model, direction, return_parameters=['pseudodata'], **kwargs), shots)
##			pds = [np.linalg.norm(r['pseudodata'])**2 for r in res]
##			denominator = np.sum(pds) * self.solver.dt
#		
#		res = self.objective_function.apply_hessian(shots, self.base_model, direction, hessian_mode='approximate', **objective_arguments)
##			res = self.objective_function.apply_hessian(shots, direction, hessian_mode='full', **objective_arguments)
#		denominator = np.dot(direction.T, res).squeeze() * np.prod(self.solver.mesh.deltas)
#		
#		numerator = d_norm**2
#		
#		return numerator / denominator
		
	def _backtrack_line_search(self, shots, gradient, direction, objective_arguments, 
	                                    current_objective_value=None, 
	                                    alpha0_kwargs={}, **kwargs):
		
		geom_fac = 0.8
		geom_fac_up = 0.7
		goldstein_c = 1e-3 #1e-4
		
		fp_comp = 1e-6
		
		if current_objective_value is None:
			fk = self.objective_function.evaluate(shots, self.base_model, **objective_arguments)
		else:
			fk = current_objective_value
		
		myalpha0_kwargs = dict()
		myalpha0_kwargs.update(alpha0_kwargs)
		myalpha0_kwargs.update({'upscale_factor' : geom_fac_up})
		
		alpha = self._compute_alpha0(current_objective_value, gradient, **myalpha0_kwargs)
		
		stop = False
		itercnt = 1
		self._print("  Starting: ".format(itercnt), alpha, fk)
		while not stop:
			
			# Cut the initial alpha until it is as large as can be and still satisfy the valid conditions for an updated model.
			valid=False
			alpha *= 2
			cnt = 0
			while not valid:
				alpha/=2
				tdir = alpha*direction
				model = self.base_model + tdir
				cnt +=1
				valid = model.validate()

			self.solver.model_parameters = model
			
			fkp1 = self.objective_function.evaluate(shots, model, **objective_arguments)
			
			cmpval = fk + alpha * goldstein_c * gradient.inner_product(tdir)
			
			self._print("  Pass {0}: a:{1}; {2} ?<= {3}".format(itercnt, alpha, fkp1, cmpval))
			
			if (fkp1 <= cmpval) or ((abs(fkp1-cmpval)/abs(fkp1)) <= fp_comp): # reasonable floating point cutoff
				stop = True
			elif itercnt > self.max_linesearch_iterations:
				stop = True
				self._print('Too many passes ({0}), attempting to use current alpha ({1}).'.format(itercnt, alpha))
			else:
				itercnt += 1
				alpha = alpha * geom_fac
				
		self.prev_alpha = alpha
		
		return alpha		
		
		