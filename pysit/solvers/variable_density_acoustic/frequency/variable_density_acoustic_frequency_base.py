import scipy.sparse.linalg as spspla

from pysit.util import ConstructableDict

from ..variable_density_acoustic_base import *

from pysit.util.solvers import inherit_dict

from pysit.util.wrappers.petsc import PetscWrapper

__all__ = ['VariableDensityAcousticFrequencyBase']

__docformat__ = "restructuredtext en"

solver_style_map = {'sparseLU': '_build_sparseLU_solver',
                    'amg': '_build_amg_solver',
                    'iterative': '_build_iterative_solver',
                    'petsc_mumps':'_build_petsc_mumps_solver',
                    'petsc_superlu_dist':'_build_petsc_superlu_dist_solver',
                    'petsc_mkl_pardiso':'_build_petsc_mkl_pardiso_solver'}
# need to add the Petsc solver and we need to add the different possible linear algebra solvers

@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticFrequencyBase(VariableDensityAcousticBase):

    _local_support_spec = {'equation_dynamics': 'frequency',
                           # These should be defined by subclasses.
                           'spatial_discretization': None,
                           'spatial_accuracy_order': None,
                           'spatial_dimension': None,
                           'boundary_conditions': None,
                           'precision': None}

    def __init__(self, mesh, solver_style='sparseLU', **kwargs):

        # A dictionary that holds the helmholtz operators as a function of nu
        self.linear_operators = ConstructableDict(self._build_helmholtz_operator)

        # A dictionary that holds the helmholtz solver as a function of nu
        solver_builder = self.__getattribute__(solver_style_map[solver_style])
        self.solvers = ConstructableDict(solver_builder)

        VariableDensityAcousticBase.__init__(self,
                                             mesh,
                                             solver_style=solver_style,
                                             **kwargs)

    def _process_mp_reset(self, *args, **kwargs):

        self.linear_operators.clear()
        self.solvers.clear()
        self._rebuild_operators()

    def _rebuild_operators(self):
        raise NotImplementedError("'_rebuild_operators' must be implemented in a subclass")

    def _build_sparseLU_solver(self, nu):
        return spspla.factorized(self.linear_operators[nu])

    def _build_petsc_mumps_solver(self, nu):
        dummy_wrapper = PetscWrapper()
        return dummy_wrapper.factorize(self.linear_operators[nu], 'mumps')

    def _build_petsc_superlu_dist_solver(self, nu):
        dummy_wrapper = PetscWrapper()
        return dummy_wrapper.factorize(self.linear_operators[nu], 'superlu_dist')

    def _build_petsc_mkl_pardiso_solver(self, nu):
        dummy_wrapper = PetscWrapper()
        return dummy_wrapper.factorize(self.linear_operators[nu], 'mkl_pardiso')

    def _build_amg_solver(self, nu):
        raise NotImplementedError('AMG solver for helmholtz is not implemented.')

    def _build_iterative_solver(self, nu):
        raise NotImplementedError('Iterative solver for helmholtz is not implemented.')

    def solve(self, *args, **kwargs):
        """Framework for a single execution of the solver at a given frequency. """
        raise NotImplementedError("Function 'solve' Must be implemented by subclass.")

    def solve_petsc(self, *args, **kwargs):
        """Framework for a single execution of the solver at a given frequency. """

    def solve_petsc_uhat(self, *args, **kwargs):
        """Framework for a single execution of the solver at a given frequency. """
    