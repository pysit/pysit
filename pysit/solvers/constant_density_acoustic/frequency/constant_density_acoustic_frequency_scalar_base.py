import numpy as np
import sys

from constant_density_acoustic_frequency_base import *
from pysit.solvers.solver_data import SolverDataFrequencyBase

from pysit.util.solvers import inherit_dict

__all__ = ['ConstantDensityAcousticFrequencyScalarBase']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticFrequencyScalarBase(ConstantDensityAcousticFrequencyBase):

    _local_support_spec = {'equation_formulation': 'scalar'}

    _SolverData = SolverDataFrequencyBase

    def __init__(self, mesh, **kwargs):

        self.M = None
        self.C = None
        self.K = None

        ConstantDensityAcousticFrequencyBase.__init__(self, mesh, **kwargs)

    def solve(self, solver_data, rhs, nu, *args, **kwargs):
        if type(rhs) is self.WavefieldVector:
            _rhs = rhs.data.reshape(-1)
        else:
            _rhs = rhs.reshape(-1)

        u = self.solvers[nu](_rhs)
        u.shape = solver_data.k.data.shape

        solver_data.k.data = u

    def solve_petsc(self, solver_data_list, rhs_list, nu, *args, **kwargs ):
        #try catch for the petsc4py use in multiple rhs solve
        try:
            import petsc4py
            petsc4py.init(sys.argv)
            from petsc4py import PETSc
            from pysit.util.wrappers.petsc import PetscWrapper

        except ImportError:
            raise ImportError('petsc4py is not installed, please install it and try again')


        petsc = kwargs['petsc']
        if petsc is not None:
            if len(solver_data_list) != len(rhs_list):
                raise ValueError('solver and right hand side list must be the same size')
            else:
                #Building the Helmholtz operator for petsc
                H = self._build_helmholtz_operator(nu).tocsr()
                ndof = H.shape[1]
                nshot = len(rhs_list)

                # creating the B rhs Matrix            
                B = PETSc.Mat().createDense([ndof, nshot])
                B.setUp()
                for i in range(nshot):
                    B.setValues(range(0, ndof), [i], rhs_list[i])

                B.assemblyBegin()
                B.assemblyEnd()

                # call the wrapper to solve the system
                wrapper = PetscWrapper()
                try:
                    linear_solver = wrapper.factorize(H, petsc, PETSc.COMM_WORLD)
                    Uhat = linear_solver(B.getDenseArray())
                except:
                    raise SyntaxError('petsc = '+str(petsc)+' is not a correct solver you can only use \'superlu_dist\', \'mumps\' or \'mkl_pardiso\' ')
                

                numb = 0
                for solver_data in solver_data_list:
                    u = Uhat[:,numb]
                    u.shape = solver_data.k.data.shape
                    solver_data.k.data = u
                    numb += 1

    def solve_petsc_uhat(self, solver, rhs_list, frequency, petsc='mkl_pardiso', *args, **kwargs):
        #try catch for the petsc4py use in multiple rhs solve
        try:
            import petsc4py
            petsc4py.init(sys.argv)
            from petsc4py import PETSc
            from pysit.util.wrappers.petsc import PetscWrapper
        except ImportError:
            raise ImportError('petsc4py is not installed, please install it and try again')

        #Building the Helmholtz operator for petsc
        H = self._build_helmholtz_operator(frequency)
        
        ndof = H.shape[1]
        nshot = len(rhs_list)
        nwfield = len(self.WavefieldVector.aux_names) + 1
        usize = ndof/nwfield

        # creating the B rhs Matrix
        B = PETSc.Mat().createDense([ndof, nshot])
        B.setUp()
        for i in range(nshot):
            B.setValues(range(0, ndof), [i], rhs_list[i])

        B.assemblyBegin()
        B.assemblyEnd()

        # call the wrapper to solve the system
        wrapper = PetscWrapper()
        try:
            linear_solver = wrapper.factorize(H, petsc, PETSc.COMM_WORLD)
            Uhat = linear_solver(B.getDenseArray())
        except:
            raise SyntaxError('petsc = '+str(petsc)+' is not a correct solver you can only use \'superlu_dist\', \'mumps\' or \'mkl_pardiso\' ')               
        
        Uhat = Uhat[xrange(usize),:]

        return Uhat


    def build_rhs(self, fhat, rhs_wavefieldvector=None):


        if rhs_wavefieldvector is None:
            rhs_wavefieldvector = self.WavefieldVector(self.mesh, dtype=self.dtype)
        elif type(rhs_wavefieldvector) is not self.WavefieldVector:
            raise TypeError('Input rhs array must be a WavefieldVector.')
        else:
            rhs_wavefieldvector.data *= 0
        rhs_wavefieldvector.u = fhat

        return rhs_wavefieldvector

    def _build_helmholtz_operator(self, nu):
        omega = 2*np.pi*nu
        return (-(omega**2)*self.M + omega*1j*self.C + self.K).tocsc() # csc is used for the sparse solvers right now
