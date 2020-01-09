

import itertools
import math
import copy

import numpy as np

__all__ = ['HybridModeling']

__docformat__ = "restructuredtext en"

class HybridModeling(object):
    """Class containing a collection of methods needed for seismic inversion in
    the frequency domain.

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
    def modeling_type(self): return "frequency"

    def __init__(self, solver, dft_points_per_period=12.0, adjoint_energy_threshold=1e-5):
        """Constructor for the FrequencyInversion class.

        Parameters
        ----------
        solver : pysit wave solver object
            A wave solver that inherits from pysit.solvers.WaveSolverBase

        """
        if self.solver_type == solver.supports['equation_dynamics']:
            self.solver = solver
        else:
            raise TypeError("Argument 'solver' type {1} does not match modeling solver type {0}.".format(self.solver_type, solver.supports['equation_dynamics']))

        if dft_points_per_period < 2:
            raise ValueError("Must have at least 2 points per period for DFT.")
        self.dft_points_per_period = dft_points_per_period

        self.adjoint_energy_threshold = adjoint_energy_threshold

    def _setup_forward_rhs(self, rhs_array, data):
        return self.solver.mesh.pad_array(data, out_array=rhs_array)

    def _compute_subsample_indices(self, frequencies):

        dt = self.solver.dt

        subsample_indices = dict()
        for nu in frequencies:
            max_dt = 1./(self.dft_points_per_period*nu) # nyquist is 1/2nu
            ratio = max_dt/dt
            idx = max(int(math.floor(ratio)),1)
            if idx*dt > max_dt:
                raise ValueError("Something went wrong in determinining large DFT time steps.")
            subsample_indices[nu] = idx

        return subsample_indices

    def forward_model(self, shot, m0, frequencies, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

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
        solver = self.solver
        solver.model_parameters = m0 # this updates dt and the number of steps so that is appropriate for the current model

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        nsteps = solver.nsteps
        source = shot.sources

        # Sanitize the input
        if not np.iterable(frequencies):
            frequencies = [frequencies]

        # Setup data storage for the forward modeled data
        if 'simdata' in return_parameters:
            simdata = dict()
            for nu in frequencies:
                simdata[nu] = np.zeros(shot.receivers.receiver_count)

        # Setup data storage for the forward modeled data (in time, if it is needed, and it frequently is)
        if 'simdata_time' in return_parameters:
            simdata_time = np.zeros((solver.nsteps, shot.receivers.receiver_count))

        # Storage for the derivative of the propagation operator with respect to the model \frac{d\script{L}}{dm}
        if 'dWaveOp' in return_parameters:
            dWaveOp = dict()
            for nu in frequencies:
                dWaveOp[nu] = 0.0

        # Initialize the DFT components
        uhats = dict()
        for nu in frequencies:
            uhats[nu] = 0.0

        subsample_indices = self._compute_subsample_indices(frequencies)

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k   = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        for k in range(nsteps):

            # Local reference

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            # Record the data at t_k
            if 'simdata_time' in return_parameters:
                shot.receivers.sample_data_from_array(uk_bulk, k, data=simdata_time)

            t = k*dt

            for nu in frequencies:
                idx = subsample_indices[nu]
                if np.mod(k, idx) == 0:
                    uhats[nu] += uk*(np.exp(-1j*2*np.pi*nu*t)*dt*idx)

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

            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == (nsteps-1)): break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        # Record the data at t_k
        if 'simdata' in return_parameters:
            for nu in frequencies:
                simdata[nu] = shot.receivers.sample_data_from_array(mesh.unpad_array(uhats[nu]))

        # Compute time derivative of p at time k
        if 'dWaveOp' in return_parameters:
            for nu in frequencies:
                dWaveOp[nu] += solver.compute_dWaveOp('frequency', uhats[nu], nu)

        retval = dict()

        if 'dWaveOp' in return_parameters:
            retval['dWaveOp'] = dWaveOp
        if 'simdata' in return_parameters:
            retval['simdata'] = simdata
        if 'wavefield' in return_parameters:
            _uhats = dict()
            _uhats = {nu: mesh.unpad_array(uhats[nu], copy=True) for nu in frequencies}
            retval['wavefield'] = _uhats
        if 'simdata_time' in return_parameters:
            retval['simdata_time'] = simdata_time

        return retval

    def migrate_shot(self, shot, m0, operand_simdata, frequencies,
                           operand_dWaveOpAdj=None, operand_model=None,
                           frequency_weights=None,
                           dWaveOp=None,
                           adjointfield=None, dWaveOpAdj=None):
        """Performs migration on a single shot.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute migration.
        operand_simdata : ndarray
            Operand, i.e., b in F*b. This data is in TIME to properly compute the adjoint.
        frequencies : list of 2-tuples
            2-tuple, first element is the frequency to use, second element the weight.
        utt : list
            Imaging condition components from the forward model for each receiver in the shot.
        qs : list
            Optional return list allowing us to retrieve the adjoint field as desired.

        """

        # If the imaging component has not already been computed, compute it.
        prep_rp = list()
        if dWaveOp is None:
            prep_rp.append('dWaveOp')
            dWaveOp = dict()

        if len(prep_rp) > 0:
            retval = self.forward_model(shot, m0, frequencies, return_parameters=prep_rp)
            if 'dWaveOp' in prep_rp:
                for nu in frequencies:
                    dWaveOp[nu] = retval['dWaveOp'][nu]


        rp = ['imaging_condition']
        if adjointfield is not None:
            rp.append('adjointfield')
        if dWaveOpAdj is not None:
            rp.append('dWaveOpAdj')

        rv = self.adjoint_model(shot, m0, operand_simdata, frequencies, operand_dWaveOpAdj=operand_dWaveOpAdj, operand_model=operand_model, frequency_weights=frequency_weights, return_parameters=rp, dWaveOp=dWaveOp)

        # If the adjoint field is desired as output.
        for nu in frequencies:
            if adjointfield is not None:
                adjointfield[nu] = rv['adjointfield'][nu]
            if dWaveOpAdj is not None:
                dWaveOpAdj[nu] = rv['dWaveOpAdj'][nu]

        # Get the imaging condition part from the result, this is the migrated image.
        ic = rv['imaging_condition']

        return ic

    def _setup_adjoint_rhs(self, rhs_array, shot, k, operand_simdata, operand_model, operand_dWaveOpAdj):

        # basic rhs is always the pseudodata or residual
        rhs_array = self.solver.mesh.pad_array(shot.receivers.extend_data_to_array(k, data=operand_simdata), out_array=rhs_array)

        # for Hessians, sometimes there is more to the rhs
        if (operand_dWaveOpAdj is not None) and (operand_model is not None):
            rhs_array += operand_model*operand_dWaveOpAdj[k]

        return rhs_array

    def adjoint_model(self, shot, m0,
                           operand_simdata, frequencies,
                           operand_dWaveOpAdj=None, operand_model=None,
                           frequency_weights=None,
                           return_parameters=[],
                           dWaveOp=None):
        """Solves for the adjoint field in frequency.

        m*q_tt - lap q = resid

        Parameters
        ----------
        shot : pysit.Shot
            Gives the receiver model for the right hand side.
        operand : ndarray
            Right hand side, usually the residual.
        frequencies : list of 2-tuples
            2-tuple, first element is the frequency to use, second element the weight.
        return_parameters : list of {'q', 'qhat', 'ic'}
        dWaveOp : ndarray
            Imaging component from the forward model (in frequency).

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * q is the adjoint field.
        * qhat is the DFT of oq at the specified frequencies
        * ic is the imaging component.  Because this function computes many of
          the things required to compute the imaging condition, there is an option
          to compute the imaging condition as we go.  This should be used to save
          computational effort.  If the imaging condition is to be computed, the
          optional argument utt must be present.

        """

        # Sanitize the input
        if not np.iterable(frequencies):
            frequencies = [frequencies]

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        nsteps = solver.nsteps
        source = shot.sources

        # Sanitize the input
        if not np.iterable(frequencies):
            frequencies = [frequencies]

        qhats = dict()
        vhats = dict()
        for nu in frequencies:
            vhats[nu] = 0.0

        subsample_indices = self._compute_subsample_indices(frequencies)

        if 'dWaveOpAdj' in return_parameters:
            dWaveOpAdj = dict()
            for nu in frequencies:
                dWaveOpAdj[nu] = 0.0

        # If we are computing the imaging condition, ensure that all of the parts are there.
        if dWaveOp is None and 'imaging_condition' in return_parameters:
            raise ValueError('To compute imaging condition, forward component must be specified.')

        if operand_model is not None:
            operand_model = operand_model.with_padding()

        # Time-reversed wave solver
        solver_data = solver.SolverData()

        rhs_k   = np.zeros(mesh.shape(include_bc=True))
        rhs_km1 = np.zeros(mesh.shape(include_bc=True))

        max_energy = 0.0

        # Loop goes over the valid indices backwards
        for k in range(nsteps-1, -1, -1): #xrange(int(solver.nsteps)):

            # Local references
            vk = solver_data.k.primary_wavefield

            max_energy = max(max_energy, np.linalg.norm(vk, np.inf))

            t = k*dt

            # When dpdt is not set, store the current q, otherwise compute the
            # relevant gradient portion

            for nu in frequencies:
                # Note, this compuation is the DFT, but we need the conjugate later, so rather than exp(-1j...) we use exp(1j...) to compute the conjugate now.
                idx = subsample_indices[nu]
                if np.mod(k, idx) == 0:
                    vhats[nu] += vk*(np.exp(-1j*2*np.pi*nu*(solver.tf-t))*dt*idx)

            if k == nsteps-1:
                rhs_k   = self._setup_adjoint_rhs( rhs_k,   shot, k,   operand_simdata, operand_model, operand_dWaveOpAdj)
                rhs_km1 = self._setup_adjoint_rhs( rhs_km1, shot, k-1, operand_simdata, operand_model, operand_dWaveOpAdj)
            else:
                # shift time forward
                rhs_k, rhs_km1 = rhs_km1, rhs_k
            rhs_km1 = self._setup_adjoint_rhs( rhs_km1, shot, k-1, operand_simdata, operand_model, operand_dWaveOpAdj)

            solver.time_step(solver_data, rhs_k, rhs_km1)

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        # When computing the adjoint field by DFT, the field, as a function of
        # time, must have finite support.  To achieve this, the wave must be
        # given sufficient time to die out.  In practice, an additional solver.tf
        # seconds appears to be sufficient, though it may be excessive.
        # As of now, no data about the wavefields are stored in this function,
        # so this part simply does the DFT on the conjugate (time-reversed)
        # adjoint field vk.  The right-hand-side should be zero.
        rhs_k *= 0
        for k in range(1,nsteps):

            vk = solver_data.k.primary_wavefield
            t = -k*dt

            if np.abs(np.linalg.norm(vk, np.inf)/max_energy) < self.adjoint_energy_threshold:
#               print "Breaking early:", nsteps + k, k
                break

            for nu in frequencies:
                idx = subsample_indices[nu]
                if np.mod(k, idx) == 0:
                    vhats[nu] += vk*(np.exp(-1j*2*np.pi*nu*(solver.tf-t))*dt*idx)

            solver.time_step(solver_data, rhs_k, rhs_k)
            solver_data.advance()

        retval = dict()

        for nu in frequencies:
            qhats[nu] = np.conj(vhats[nu],vhats[nu])
            # The next line accounts for the fact that not all frequencies are
            # integer, in the relationship between the adjoint field q and the
            # conjugate adjoint field v.
            qhats[nu] *= np.exp(-1j*2*np.pi*nu*solver.tf)

        if 'adjointfield' in return_parameters:
            _qhats = dict()
            _qhats = {nu: mesh.unpad_array(qhats[nu], copy=True) for nu in frequencies}
            retval['adjointfield'] = _qhats
        if 'dWaveOpAdj' in return_parameters:
            for nu in frequencies:
                dWaveOpAdj[nu] = solver.compute_dWaveOp('frequency', qhats[nu],nu)
            retval['dWaveOpAdj'] = dWaveOpAdj

        # If the imaging component needs to be computed, do it
        if 'imaging_condition' in return_parameters:
            ic = solver.model_parameters.perturbation(dtype=np.complex)

            if frequency_weights is None:
                frequency_weights = itertools.repeat(1.0)

            for nu,weight in zip(frequencies,frequency_weights):
                # note, no dnu here because the nus are not generally the complete set, so dnu makes little sense, otherwise dnu = 1./(nsteps*dt)
                ic -= weight*qhats[nu]*np.conj(dWaveOp[nu])

            retval['imaging_condition'] = ic.without_padding()

        return retval

    def linear_forward_model(self, shot, m0, m1, frequencies, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m1 : solver.ModelParameters
        frequencies : list of 2-tuples
            2-tuple, first element is the frequency to use, second element the weight.
        return_parameters : list of {'dWaveOp0', 'wavefield1', 'dWaveOp1', 'simdata', 'simdata_time'}, optional
            Values to return.


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

        # Sanitize the input
        if not np.iterable(frequencies):
            frequencies = [frequencies]

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
        u1hats = dict()
        for nu in frequencies:
            u1hats[nu] = 0.0

        # Setup data storage for the forward modeled data
        if 'simdata' in return_parameters:
            simdata = dict()

        # Setup data storage for the forward modeled data (in time, if it is needed, and it frequently is)
        if 'simdata_time' in return_parameters:
            simdata_time = np.zeros((solver.nsteps, shot.receivers.receiver_count))

        # Storage for the time derivatives of p
        if 'dWaveOp0' in return_parameters:
            dWaveOp0 = dict()
            u0hats = dict()
            for nu in frequencies:
                dWaveOp0[nu] = 0.0
                u0hats[nu] = 0.0

        # Storage for the time derivatives of p
        if 'dWaveOp1' in return_parameters:
            dWaveOp1 = dict()
            for nu in frequencies:
                dWaveOp1[nu] = 0.0

        subsample_indices = self._compute_subsample_indices(frequencies)

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        # (***) Given that these modeling tools are for frequency methods, we do not
        # have the time derivatives / wave operator derivatives (aka dWaveOp) in
        # time available.  This saves space, but as a result we have to recompute
        # it.
        # Also, because implicit and some ODE methods require uhat_1 at times k
        # and k+1, we need uhat_0 at k, k+1, and k+2, so all of this rigamaroll
        # is to get that.
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

        for k in range(nsteps):

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            t = k*dt

            # Record the data at t_k
            if 'simdata_time' in return_parameters:
                shot.receivers.sample_data_from_array(uk_bulk, k, data=simdata_time)

            for nu in frequencies:
                idx = subsample_indices[nu]
                if np.mod(k, idx) == 0:
                    u1hats[nu] += uk*(np.exp(-1j*2*np.pi*nu*t)*dt*idx)

            if 'dWaveOp0' in return_parameters:
                for nu in frequencies:
                    idx = subsample_indices[nu]
                    if np.mod(k, idx) == 0:
                        u0hats[nu] += solver_data_u0.k.primary_wavefield*(np.exp(-1j*2*np.pi*nu*t)*dt*idx)

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.

            # See comment (***) above.
            # compute u0_kp2 so we can get dWaveOp0_kp1 for the rhs for u1
            rhs_u0_kp1, rhs_u0_kp2 = rhs_u0_kp2, rhs_u0_kp1
            rhs_u0_kp2 = self._setup_forward_rhs(rhs_u0_kp2, source.f((k+2)*dt))
            solver.time_step(solver_data_u0, rhs_u0_kp1, rhs_u0_kp2)

            # shift the dWaveOp0's (ok at k=0 because they are equal then)
            # The derivative component is computed after the time step so that
            # information from time k+1 can be used to compute the derivative.
            dWaveOp0_k, dWaveOp0_kp1 = dWaveOp0_kp1, dWaveOp0_k
            dWaveOp0_kp1 = solver.compute_dWaveOp('time', solver_data_u0)

            solver_data_u0.advance()

            if k == 0:
                rhs_k   = m1_padded*(-1*dWaveOp0_k)
                rhs_kp1 = m1_padded*(-1*dWaveOp0_kp1)
            else:
                rhs_k, rhs_kp1 = rhs_kp1, m1_padded*(-1*dWaveOp0_kp1)

            solver.time_step(solver_data, rhs_k, rhs_kp1)

            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == (nsteps-1)): break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        # Compute time derivative of p at time k
        if 'dWaveOp0' in return_parameters:
            for nu in frequencies:
                dWaveOp0[nu] = solver.compute_dWaveOp('frequency', u0hats[nu],nu)

        # Compute time derivative of p at time k
        if 'dWaveOp1' in return_parameters:
            for nu in frequencies:
                dWaveOp1[nu] = solver.compute_dWaveOp('frequency', u1hats[nu],nu)

        # Record the data at t_k
        if 'simdata' in return_parameters:
            for nu in frequencies:
                simdata[nu] = shot.receivers.sample_data_from_array(mesh.unpad_array(u1hats[nu]))

        retval = dict()

        if 'dWaveOp0' in return_parameters:
            retval['dWaveOp0'] = dWaveOp0
        if 'wavefield1' in return_parameters:
            _u1hats = dict()
            _u1hats = {nu: mesh.unpad_array(u1hats[nu], copy=True) for nu in frequencies}
            retval['wavefield1'] = _u1hats
        if 'dWaveOp1' in return_parameters:
            retval['dWaveOp1'] = dWaveOp1
        if 'simdata' in return_parameters:
            retval['simdata'] = simdata
        if 'simdata_time' in return_parameters:
            retval['simdata_time'] = simdata_time

        return retval

def adjoint_test(frequencies=[10.0, 10.5, 10.1413515123], plots=False, data_noise=0.0, purefrequency=False):
    # default frequencies are enough to indicate a bug due to integer offsets
    import numpy as np
    import matplotlib.pyplot as plt

    from pysit import PML, RectangularDomain, CartesianMesh, PointSource, ReceiverSet, Shot, ConstantDensityAcousticWave, generate_seismic_data, PointReceiver, RickerWavelet, FrequencyModeling, ConstantDensityHelmholtz, vis
    from pysit.gallery import horizontal_reflector

    #   Define Domain
    pmlx = PML(0.3, 100, ftype='quadratic')
    pmlz = PML(0.3, 100, ftype='quadratic')

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain( x_config, z_config )

    m = CartesianMesh(d, 90, 70)

    #   Generate true wave speed
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

    point_approx = 'delta'

    for i in range(Nshots):

        # Define source location and type
        source = PointSource(m, (.188888, 0.18888), RickerWavelet(10.0), approximation=point_approx)

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
#                                        formulation='ode',
                                         formulation='scalar',
                                         model_parameters={'C': C},
                                         spatial_accuracy_order=4,
#                                        spatial_shifted_differences=True,
#                                        cfl_safety=0.01,
                                         trange=trange,
                                         time_accuracy_order=4)

    tools = HybridModeling(solver)
    m0 = solver.ModelParameters(m,{'C': C0})


    solver_frequency = ConstantDensityHelmholtz(m,
                                                model_parameters={'C': C0},
                                                spatial_shifted_differences=True,
                                                spatial_accuracy_order=4)
    frequencytools = FrequencyModeling(solver_frequency)
    m0_freq = solver_frequency.ModelParameters(m,{'C': C0})

    np.random.seed(0)

    m1 = m0.perturbation()
    pert = np.random.rand(*m1.data.shape)
    m1  += pert

#   freqs = [10.5514213] #[3.0, 5.0, 10.0]
#   freqs = [10.5]
#   freqs = np.linspace(3,19,8)
    freqs = frequencies

    fwdret = tools.forward_model(shot, m0, freqs, ['wavefield', 'dWaveOp', 'simdata_time'])
    dWaveOp0 = fwdret['dWaveOp']
    data = fwdret['simdata_time']
    u0hat = fwdret['wavefield'][freqs[0]]

    data += data_noise*np.random.rand(*data.shape)

    dhat = dict()
    for nu in freqs: dhat[nu]=0
    assert data.shape[0] == solver.nsteps
    for k in range(solver.nsteps):
        t = k*solver.dt
        for nu in freqs:
            dhat[nu] += data[k,:]*np.exp(-1j*2*np.pi*nu*t)*solver.dt

    print("Hybrid:")
    linfwdret = tools.linear_forward_model(shot, m0, m1, freqs, ['simdata','wavefield1','simdata_time'])
    lindata = linfwdret['simdata']
    lindata_time = linfwdret['simdata_time']
    u1hat = linfwdret['wavefield1'][freqs[0]]

    adjret = tools.adjoint_model(shot, m0, data, freqs, return_parameters=['imaging_condition', 'adjointfield'], dWaveOp=dWaveOp0)
    qhat = adjret['adjointfield'][freqs[0]]
    adjmodel = adjret['imaging_condition'].asarray()

    m1_ = m1.asarray()

    temp_data_prod = 0.0
    for nu in freqs:
        temp_data_prod += np.dot(lindata[nu].reshape(dhat[nu].shape), np.conj(dhat[nu]))

    print(temp_data_prod)
    print(np.dot(m1_.T, np.conj(adjmodel)).squeeze()*np.prod(m.deltas))
    print(np.dot(m1_.T, np.conj(adjmodel)).squeeze()*np.prod(m.deltas) - temp_data_prod)

    if purefrequency:
        print("Frequency:")
        linfwdret_freq = frequencytools.linear_forward_model(shot, m0, m1, freqs, ['simdata','wavefield1', 'dWaveOp0'])
        lindata_freq = linfwdret_freq['simdata']
        u1hat_freq = linfwdret_freq['wavefield1'][freqs[0]]
        dWaveOp0_freq = linfwdret_freq['dWaveOp0']

        adjret_freq = frequencytools.adjoint_model(shot, m0, dhat, freqs, return_parameters=['imaging_condition', 'adjointfield'], dWaveOp=dWaveOp0_freq)
        qhat_freq = adjret_freq['adjointfield'][freqs[0]]
        adjmodel_freq = adjret_freq['imaging_condition'].asarray()

        temp_data_prod = 0.0
        for nu in freqs:
            temp_data_prod += np.dot(lindata_freq[nu].reshape(dhat[nu].shape).T, np.conj(dhat[nu]))

        print(temp_data_prod.squeeze())
        print(np.dot(m1_.T, np.conj(adjmodel_freq)).squeeze()*np.prod(m.deltas))
        print(np.dot(m1_.T, np.conj(adjmodel_freq)).squeeze()*np.prod(m.deltas) - temp_data_prod.squeeze())

    if plots:

        xx, zz = d.generate_grid()
        sl = [(xx>=0.1) & (xx<=0.99) & (zz>=0.1) & (zz<0.8)]

        pml_null = PML(0.0,100)
        x_bulk = (0.1, 1.0, 90, pml_null, pml_null)
        z_bulk = (0.1, 0.8, 70, pml_null, pml_null)
        d_bulk = Domain( (x_bulk, z_bulk) )

        def clims(*args):
            rclim = min([np.real(x).min() for x in args]), max([np.real(x).max() for x in args])
            iclim = min([np.imag(x).min() for x in args]), max([np.imag(x).max() for x in args])
            return rclim, iclim

        qrclim, qiclim = clims(qhat, qhat_freq)
        u1rclim, u1iclim = clims(u1hat, u1hat_freq)

        plt.figure()
        plt.subplot(2,3,1)
        display_on_grid(np.real(u0hat[sl]), d_bulk)
        plt.title(r're(${\hat u_0}$)')
        plt.subplot(2,3,4)
        display_on_grid(np.imag(u0hat[sl]), d_bulk)
        plt.title(r'im(${\hat u_0}$)')
        plt.subplot(2,3,2)
        display_on_grid(np.real(qhat[sl]), d_bulk, clim=qrclim)
        plt.title(r're(${\hat q}$) H')
        plt.subplot(2,3,5)
        display_on_grid(np.imag(qhat[sl]), d_bulk, clim=qiclim)
        plt.title(r'im(${\hat q}$) H')
        plt.subplot(2,3,3)
        display_on_grid(np.real(u1hat[sl]), d_bulk, clim=u1rclim)
        plt.title(r're(${\hat u_1}$) H')
        plt.subplot(2,3,6)
        display_on_grid(np.imag(u1hat[sl]), d_bulk, clim=u1iclim)
        plt.title(r'im(${\hat u_1}$) H')
        plt.show()

        plt.figure()
        plt.subplot(2,3,1)
        display_on_grid(np.real(u0hat[sl]), d_bulk)
        plt.title(r're(${\hat u_0}$)')
        plt.subplot(2,3,4)
        display_on_grid(np.imag(u0hat[sl]), d_bulk)
        plt.title(r'im(${\hat u_0}$)')
        plt.subplot(2,3,2)
        display_on_grid(np.real(qhat_freq[sl]), d_bulk, clim=qrclim)
        plt.title(r're(${\hat q}$) P')
        plt.subplot(2,3,5)
        display_on_grid(np.imag(qhat_freq[sl]), d_bulk, clim=qiclim)
        plt.title(r'im(${\hat q}$) P')
        plt.subplot(2,3,3)
        display_on_grid(np.real(u1hat_freq[sl]), d_bulk, clim=u1rclim)
        plt.title(r're(${\hat u_1}$) P')
        plt.subplot(2,3,6)
        display_on_grid(np.imag(u1hat_freq[sl]), d_bulk, clim=u1iclim)
        plt.title(r'im(${\hat u_1}$) P')
        plt.show()


if __name__ == '__main__':

    #adjoint_test(purefrequency=True, frequencies=[10.0, 10.5, 10.1413515123])

    import time

    import numpy as np
    import matplotlib.pyplot as plt

    import pysit
    import pysit.vis as vis
    from pysit import PML, RectangularDomain, CartesianMesh, PointSource, ReceiverSet, Shot, ConstantDensityAcousticWave, generate_seismic_data, PointReceiver, RickerWavelet, FrequencyModeling, ConstantDensityHelmholtz, vis
    from pysit.gallery import horizontal_reflector

    #   Define Domain
    pmlx = PML(0.3, 100, ftype='quadratic')
    pmlz = PML(0.3, 100, ftype='quadratic')

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain( x_config, z_config )

    m = CartesianMesh(d, 2*90, 2*70)
    m = CartesianMesh(d, 90, 70)

    #   Generate true wave speed
    #   (M = C^-2 - C0^-2)
    C0, C = horizontal_reflector(m)

    # Set up shots
    shots = list()

    xmin = d.x.lbound
    xmax = d.x.rbound
    nx   = m.x.n
    zmin = d.z.lbound
    zmax = d.z.rbound

    point_approx = 'delta'

    # Define source location and type
    source = PointSource(m, (.188888, 0.18888), RickerWavelet(10.0), approximation=point_approx)

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
                                         formulation='scalar',
                                         model_parameters={'C': C},
                                         spatial_accuracy_order=4,
                                         use_cpp_acceleration=True,
                                         trange=trange,)

    class Experiment(object):

        def __init__(self, shot, tools, m0, m1, name='', data_noise=0.0):

            self.shot = shot

            self.tools = tools

            self.m0 = m0
            self.m1 = m1

            self.results_fwd = None

            self.name = name

            self.data_noise = data_noise

        def run_fwd(self, freqs):

            tt = time.time()

            self.fwd_results = self.tools.forward_model(self.shot, self.m0, freqs, ['wavefield', 'dWaveOp', 'simdata_time'])

            self.fwd_time = time.time() - tt

            print(self.name + ": fwd run time ({0} frequency) -- {1:.6f}s".format(len(freqs), self.fwd_time))

            np.random.seed(1)
            data = self.fwd_results['simdata_time']
            self.data = data + self.data_noise*np.random.rand(*data.shape)

            dhat = dict()
            for nu in freqs: dhat[nu]=0
            assert data.shape[0] == self.tools.solver.nsteps
            for k in range(self.tools.solver.nsteps):
                t = k*self.tools.solver.dt
                for nu in freqs:
                    dhat[nu] += data[k,:]*np.exp(-1j*2*np.pi*nu*t)*self.tools.solver.dt
            self.dhat = dhat

        def run_lin_fwd(self,freqs):

            tt = time.time()

            self.lin_results = self.tools.linear_forward_model(self.shot, self.m0, self.m1, freqs, ['simdata','wavefield1','simdata_time'])

            self.lin_time = time.time() - tt

            print(self.name + ": lin run time ({0} frequency) -- {1:.6f}s".format(len(freqs), self.lin_time))

        def run_adj(self,freqs):

            tt = time.time()

            self.adj_results = self.tools.adjoint_model(self.shot, self.m0, self.data, freqs, return_parameters=['imaging_condition', 'adjointfield'], dWaveOp=self.fwd_results['dWaveOp'])

            self.adj_time = time.time() - tt

            print(self.name + ": adj run time ({0} frequency) -- {1:.6f}s".format(len(freqs), self.adj_time))





    def compare_fwd(exp1, exp2, freqs, plot=True):

        for nu in freqs:

            uhat0_1 = exp1.fwd_results['wavefield'][nu]
            uhat0_2 = exp2.fwd_results['wavefield'][nu]

            diff = uhat0_1 - uhat0_2

            print("Error norm ({0} - {1}) {3: 09.4f}Hz: {2:.4e}".format(exp1.name, exp2.name, np.linalg.norm(diff)/np.linalg.norm(uhat0_1), nu))

            if plot:

                clim = min(uhat0_1.min(), uhat0_2.min()), max(uhat0_1.max(), uhat0_2.max())

                plt.figure()
                plt.subplot(3,2,1)
                vis.plot(uhat0_1.real, m,clim=clim)
                plt.colorbar()
                plt.subplot(3,2,3)
                vis.plot(uhat0_2.real, m,clim=clim)
                plt.colorbar()
                plt.subplot(3,2,5)
                vis.plot(diff.real, m,diff)
                plt.colorbar()

                plt.subplot(3,2,2)
                vis.plot(uhat0_1.imag, m,clim=clim)
                plt.colorbar()
                plt.subplot(3,2,4)
                vis.plot(uhat0_2.imag, m,clim=clim)
                plt.colorbar()
                plt.subplot(3,2,6)
                vis.plot(diff.imag, m,diff)
                plt.colorbar()

                plt.show()

        nsteps = exp1.tools.solver.nsteps

        print("\nTime steps: {0}".format(nsteps))
        print("Per step improvement (fwd): {0: .4e} ({1:.4f}x)".format((exp1.fwd_time - exp2.fwd_time)/nsteps, exp1.fwd_time/exp2.fwd_time))
        print("Per step improvement (lin): {0: .4e} ({1:.4f}x)".format((exp1.lin_time - exp2.lin_time)/nsteps, exp1.lin_time/exp2.lin_time))
        print("Per step improvement (adj): {0: .4e} ({1:.4f}x)".format((exp1.adj_time - exp2.adj_time)/nsteps, exp1.adj_time/exp2.adj_time))
        print("")



    def test_adjoints(exp, freqs):

        deltas = exp.m0.mesh.deltas

        m1_ = exp.m1.asarray()

        lindata = exp.lin_results['simdata']
        dhat = exp.dhat

        adjmodel = exp.adj_results['imaging_condition'].asarray()

        temp_data_prod = 0.0
        for nu in freqs:
            temp_data_prod += np.dot(lindata[nu].reshape(dhat[nu].shape), np.conj(dhat[nu]))

        pt1 = temp_data_prod

        pt2 = np.dot(m1_.T, np.conj(adjmodel)).squeeze()*np.prod(deltas)

        print("{0}: ".format(exp.name))
        print("<Fm1, d>             = {0: .4e} ({1:.4e})".format(pt1, np.linalg.norm(pt1)))
        print("<m1, F*d>            = {0: .4e} ({1:.4e})".format(pt2, np.linalg.norm(pt2)))
        print("<Fm1, d> - <m1, F*d> = {0: .4e} ({1:.4e})".format(pt1-pt2, np.linalg.norm(pt1-pt2)))

        print("Relative error       = {0: .4e}\n".format(np.linalg.norm(pt1-pt2)/np.linalg.norm(pt1)))

    tools_old = pysit.modeling.HybridModeling(solver)
    tools_new = HybridModeling(solver, adjoint_energy_threshold=1e-3)

    np.random.seed(0)

    m0 = solver.ModelParameters(m,{'C': C0})

    m1 = m0.perturbation()
    pert = np.random.rand(*m1.data.shape)
    m1  += pert

    freqs = [3.0, 5.0, 10.0, 10.5, 10.5514213] #[3.0, 5.0, 10.0]
#   freqs = [10.5]
    freqs = np.linspace(3,19,8)
#   freqs = [20.0]
#   freqs = [3.0]

    shot_old = copy.deepcopy(shot)
    shot_new = copy.deepcopy(shot)

    old = Experiment(shot_old, tools_old, m0, m1, 'old')
    new = Experiment(shot_new, tools_new, m0, m1, 'new')

    old.run_fwd(freqs)
    old.run_lin_fwd(freqs)
    old.run_adj(freqs)
    print("")

    new.run_fwd(freqs)
    new.run_lin_fwd(freqs)
    new.run_adj(freqs)
    print("")

    compare_fwd(old, new, freqs, plot=False)

    test_adjoints(old, freqs)
    test_adjoints(new, freqs)