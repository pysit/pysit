

import itertools

import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.frequency_modeling import FrequencyModeling

__all__ = ['FrequencyLeastSquares']

__docformat__ = "restructuredtext en"

class FrequencyLeastSquares(ObjectiveFunctionBase):

    def __init__(self, solver, parallel_wrap_shot=ParallelWrapShotNull()):
        self.solver = solver
        self.modeling_tools = FrequencyModeling(solver)

        self.parallel_wrap_shot = parallel_wrap_shot

    def _residual_list(self, shots_list, m0, frequencies=None, frequency_weights=None, dWaveOp=None, wavefield=None, **kwargs):
        """Computes residual in the usual sense for a list of shots

        Parameters
        ----------
        shots_list : list of pysit.Shot
            Shot for which to compute the residual.
        utt : list of ndarray (optional)
            An empty list for returning the derivative term required for
            computing the imaging condition.

        """

        if frequencies is None:
            raise ValueError('A set of frequencies must be specified.')

        # If we will use the second derivative info later (and this is usually
        # the case in inversion), tell the solver to store that information, in
        # addition to the solution as it would be observed by the receivers in
        # this shot (aka, the simdata).
        if dWaveOp is not None:
            rp = ['simdata','dWaveOp']
        else:
            rp = ['simdata']
        # If we are dealing with variable density, we want the wavefield returned as well.
        if wavefield is not None:
            rp.append('wavefield')
        # Run the forward modeling step
        retval = self.modeling_tools.forward_model_list(shots_list, m0, frequencies, return_parameters=rp, **kwargs)
        resid = 0.0
        resid_list = dict()
        for i in range(len(shots_list)):
            shots_list[i].receivers.compute_data_dft(frequencies)
            resid_list[i] = dict()
            for nu,weight in zip(frequencies,frequency_weights):
                resid_list[i][nu] = shots_list[i].receivers.data_dft[nu] - retval['simdata'][i][nu]
                resid += weight*np.linalg.norm(resid_list[i][nu])**2

            if dWaveOp is not None:
                for nu in frequencies:
                    dWaveOp[i][nu]  = retval['dWaveOp'][i][nu]
            if wavefield is not None:
                for nu in frequencies:
                    wavefield[i][nu] = retval['wavefield'][i][nu]        

        return resid , resid_list

    def _residual(self, shot, m0, frequencies=None, dWaveOp=None, wavefield=None):
        """Computes residual in the usual sense.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.
        utt : list of ndarray (optional)
            An empty list for returning the derivative term required for
            computing the imaging condition.

        """

        if frequencies is None:
            raise ValueError('A set of frequencies must be specified.')

        # If we will use the second derivative info later (and this is usually
        # the case in inversion), tell the solver to store that information, in
        # addition to the solution as it would be observed by the receivers in
        # this shot (aka, the simdata).
        if dWaveOp is not None:
            rp = ['simdata','dWaveOp']
        else:
            rp = ['simdata']
        # If we are dealing with variable density, we want the wavefield returned as well.
        if wavefield is not None:
            rp.append('wavefield')
        # Run the forward modeling step
        retval = self.modeling_tools.forward_model(shot, m0, frequencies, return_parameters=rp)

        resid = dict()
        for nu in frequencies:
            resid[nu] = shot.receivers.data_dft[nu] - retval['simdata'][nu]

        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            for nu in frequencies:
                dWaveOp[nu]  = retval['dWaveOp'][nu]
        if wavefield is not None:
            for nu in frequencies:
                wavefield[nu] = retval['wavefield'][nu]

        return resid

    def evaluate(self, shots, m0, frequencies=None, frequency_weights=None, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""
        if frequencies is None:
            raise ValueError('A set of frequencies must be specified.')

        if frequency_weights is not None and (len(frequencies) != len(frequency_weights)):
            raise ValueError('Weights and frequencies must be the same length.')

        if frequency_weights is None:
            frequency_weights = itertools.repeat(1.0)

        if 'petsc' in kwargs:
            r_norm2, resid_list = self._residual_list(shots, m0, frequencies, frequency_weights, **kwargs)
        else:
            r_norm2 = 0
            for shot in shots:
                # ensure that the dft of the data exists
                shot.receivers.compute_data_dft(frequencies)
                r = self._residual(shot, m0, frequencies)
                for nu,weight in zip(frequencies,frequency_weights):
                    r_norm2 += weight*np.linalg.norm(r[nu])**2

            # sum-reduce and communicate result
            if self.parallel_wrap_shot.use_parallel:
                # Allreduce wants an array, so we give it a 0-D array
                new_r_norm2 = np.array(0.0)
                self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
                r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

        return 0.5*r_norm2 # *d_omega which does not exist for this problem !!! check if a dt needed

    def _gradient_helper(self, shot, m0, frequencies, frequency_weights, ignore_minus=False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        single shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.

        """
        
        dWaveOp = dict()

        # If this is true, then we are dealing with variable density. In this case, we want our forward solve
        # To also return the wavefield, because we need to take gradients of the wavefield in the adjoint model
        # Step to calculate the gradient of our objective in terms of m2 (ie. 1/rho)
        if hasattr(m0, 'kappa') and hasattr(m0,'rho'):
            wavefield=dict()
        else:
            wavefield=None

        # Compute the residual vector and its norm
        r = self._residual(shot, m0, frequencies, dWaveOp=dWaveOp, wavefield=wavefield)

        # Perform the migration or F* operation to get the gradient component
        g = self.modeling_tools.migrate_shot(shot, m0, r, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp, wavefield=wavefield)
        g.toreal()

        if ignore_minus:
            return g, r
        else:
            return -1*g, r

    def _gradient_helper_list(self, shots_list, m0, frequencies, frequency_weights, ignore_minus=False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        list of shots shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shots_list :  list of pysit.Shot
            Shot for which to compute the residual.

        """
        
        dWaveOp = dict()

        for i in range(len(shots_list)):
            dWaveOp[i] = dict()

        # If this is true, then we are dealing with variable density. In this case, we want our forward solve
        # To also return the wavefield, because we need to take gradients of the wavefield in the adjoint model
        # Step to calculate the gradient of our objective in terms of m2 (ie. 1/rho)
        if hasattr(m0, 'kappa') and hasattr(m0,'rho'):
            wavefield=dict()
            for i in range(len(shots_list)):
                wavefield[i] = dict()
        else:
            wavefield=None

        # Compute the residual list and its norm
        r, resid_list = self._residual_list(shots_list, m0, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs)

        g = self.modeling_tools.migrate_shot_list(shots_list, m0, resid_list, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs) 

        for i in range(len(g)):
            if ignore_minus:
                g[i].toreal()
            else:
                g[i].toreal()
                g[i] = - g[i]

        return g, r

    def compute_gradient(self, shots, m0, frequencies=None, frequency_weights=None, aux_info={}, **kwargs):
        """Compute the gradient for a set of shots.

        Computes the gradient as
            -F*(d - scriptF[m0]) = sum(F*_s(d - scriptF_s[m0])) for s in shots

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute the gradient.

        """
        if frequencies is None:
            raise ValueError('A set of frequencies must be specified.')

        if frequency_weights is not None and (len(frequencies) != len(frequency_weights)):
            raise ValueError('Weights and frequencies must be the same length.')
        if 'petsc' in kwargs and kwargs['petsc'] is not None:
            if frequency_weights is None:
                frequency_weights = itertools.repeat(1.0)

            grad = m0.perturbation()
            g, r_norm2 = self._gradient_helper_list(shots, m0, frequencies, frequency_weights, ignore_minus=True, **kwargs)
        
            for i in range(len(g)):
                grad -= g[i]

        else:
            # compute the portion of the gradient due to each shot
            grad = m0.perturbation()
            r_norm2 = 0.0
            for shot in shots:
                # ensure that the dft of the data exists
                shot.receivers.compute_data_dft(frequencies)

                g, r = self._gradient_helper(shot, m0, frequencies, frequency_weights, ignore_minus=True, **kwargs)
                grad -= g

                if frequency_weights is None:
                    frequency_weights = itertools.repeat(1.0)

                for nu,weight in zip(frequencies,frequency_weights):
                    r_norm2 += weight*np.linalg.norm(r[nu])**2

            # sum-reduce and communicate result
            if self.parallel_wrap_shot.use_parallel:
                # Allreduce wants an array, so we give it a 0-D array
                new_r_norm2 = np.array(0.0)
                self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
                r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

                ngrad = np.zeros_like(grad.asarray())
                self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
                grad=m0.perturbation(data=ngrad)

        # store any auxiliary info that is requested
        if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
            aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
        if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
            aux_info['objective_value'] = (True, 0.5*r_norm2)
        return grad

    def apply_hessian(self, shots, m0, m1,
                            frequencies=None, frequency_weights=None,
                            hessian_mode='approximate', levenberg_mu=0.0, **kwargs):

        modes = ['approximate', 'full', 'levenberg']
        if hessian_mode not in modes:
            raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))

        result = m0.perturbation()

        if hessian_mode in ['approximate', 'levenberg']:
            for shot in shots:

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, frequencies, return_parameters=['simdata', 'dWaveOp0'])
                dWaveOp0 = linear_retval['dWaveOp0']

                d1 = linear_retval['simdata']
                result += self.modeling_tools.migrate_shot(shot, m0, d1, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp0)

                result.toreal()

        elif hessian_mode == 'full':
            for shot in shots:
                # Run the forward modeling step
                dWaveOp0 = dict() # wave operator derivative wrt model for u_0
                r0 = self._residual(shot, m0, frequencies, dWaveOp=dWaveOp0, **kwargs)

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, frequencies, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
                d1 = linear_retval['simdata']
                dWaveOp1 = linear_retval['dWaveOp1']

                # <q, u1tt>, first adjointy bit
                dWaveOpAdj1=dict()
                res1 = self.modeling_tools.migrate_shot( shot, m0, r0, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)
                result += res1

                # <p, u0tt>
                res2 = self.modeling_tools.migrate_shot(shot, m0, d1, frequencies, frequency_weights=frequency_weights, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
                result += res2

                result.toreal()

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:

            nresult = np.zeros_like(result.asarray())
            self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
            result = m0.perturbation(data=nresult)

        # Note, AFTER the application has been done in parallel do this.
        if hessian_mode == 'levenberg':
            result += levenberg_mu*m1

        return result