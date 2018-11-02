import numpy as np

from .solver_data import SolverDataBase

from pysit.util.basic_registration_factory import DoesNotMatch
from pysit.util.basic_registration_factory import CompleteMatch
from pysit.util.basic_registration_factory import IncompleteMatch
from pysit.util.solvers import supports
from pysit.util.solvers import inherit_dict

__all__ = ['SolverBase']

__docformat__ = "restructuredtext en"


class NullModelParameters(object):
    def __init__(self, *args, **kwargs):
        raise TypeError("NullModelParameters should never be instantiated.")


@inherit_dict('supports', '_local_support_spec')
class SolverBase(object):
    """ Base class for pysit solvers. (e.g., wave, helmholtz, and laplace-domain)

    This class serves as a base class for wave equation solvers in pysit.  It
    defines some required interface items.

    Attributes
    ----------
    mesh : pysit.Mesh
        Computational domain on which the source is defined.
    domain : pysit.Domain
        Physical (and numerical) domain on which the solver is operates.
    model_parameters : self.WaveEquationParamters
        Object containing the relevant parameters for a given wave equation.

    """

    # These must be set in a subclass.  Ideally it should happen in once place
    # and they can be inherited.
    _local_support_spec = {'equation_physics': None,  # e.g., 'constant-density-acoustic', 'elastic'
                           'equation_dynamics': None}  # e.g, 'time', 'frequency', 'laplace'

    def __init__(self,
                 mesh,
                 precision='double',
                 **kwargs):
        """Constructor for the WaveSolverBase class.

        Parameters
        ----------
        mesh : pysit.Mesh
            Computational domain on which the source is defined.
        model_parameters : dict
            Dictionary of initial wave parameters for the solver.
        """

        self.mesh = mesh
        self.domain = mesh.domain

        # This _should_ always be true, if the solver is constructed through
        # the appropriate factory, as 'precision' is a support value.
        if precision in self.supports['precision']:
            self.precision = precision

            if self.supports['equation_dynamics'] == 'time':
                self.dtype = np.double if precision == 'double' else np.single
            else:
                self.dtype = np.complex128 if precision == 'double' else np.complex64

        self._mp = None
        self._model_change_count = 0
        # self.model_parameters = self.ModelParameters(mesh, inputs=model_parameters)

    @classmethod
    def _factory_validation_function(cls, mesh, *args, **kwargs):

        complete_match = True
        incomplete_match = True

        for parameter, values in list(cls.supports.items()):
            if parameter in kwargs:
                if not supports(kwargs[parameter], values):
                    return DoesNotMatch
            elif parameter == 'boundary_conditions':
                valid_bc = True
                for i in range(mesh.dim):
                    L = supports(mesh.parameters[i].lbc.type, values)
                    R = supports(mesh.parameters[i].rbc.type, values)
                    valid_bc &= L and R
                if not valid_bc:
                    return DoesNotMatch
            elif parameter == 'spatial_dimension':
                if not supports(mesh.dim, values):
                    return DoesNotMatch
            else:
                complete_match = False

        if complete_match:
            return CompleteMatch
        elif incomplete_match:
            return IncompleteMatch

    @property #getter
    def model_parameters(self): return self._mp

    @model_parameters.setter
    def model_parameters(self, mp):
        if type(mp) is self.ModelParameters:
            if self._mp is None or np.linalg.norm(self._mp.without_padding().data - mp.data) != 0.0:
                self._mp = mp.with_padding(padding_mode='edge')

                self._process_mp_reset()

                self._model_change_count += 1
        else:
            raise TypeError('{0} is not of type {1}'.format(type(mp), self.ModelParameters))

    def _process_mp_reset(self, *args, **kwargs):
        raise NotImplementedError('_process_mp_reset() must be implemented by a subclass.')


    def compute_dWaveOp(self, regime, *args):
        return self.__getattribute__('_compute_dWaveOp_{0}'.format(regime))(*args)

    ModelParameters = None

    WavefieldVector = None

    _SolverData = SolverDataBase

    def SolverData(self, **kwargs):
        return self._SolverData(self, **kwargs)


