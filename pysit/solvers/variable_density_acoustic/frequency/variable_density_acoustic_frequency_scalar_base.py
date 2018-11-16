import numpy as np
import sys
from pysit.util.matrix_helpers import make_diag_mtx

from .variable_density_acoustic_frequency_base import *
from pysit.solvers.solver_data import SolverDataFrequencyBase

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticFrequencyScalarBase']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticFrequencyScalarBase(VariableDensityAcousticFrequencyBase):

    _local_support_spec = {'equation_formulation': 'scalar'}

    _SolverData = SolverDataFrequencyBase

    def __init__(self, mesh, **kwargs):

        self.M = None
        self.C = None
        self.K = None

        VariableDensityAcousticFrequencyBase.__init__(self, mesh, **kwargs)

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
                H = self._build_helmholtz_operator(nu)
                ndof = H.shape[1]
                nshot = len(rhs_list)

                # creating the B rhs Matrix            
                B = PETSc.Mat().createDense([ndof, nshot])
                B.setUp()
                for i in range(nshot):
                    B.setValues(list(range(0, ndof)), [i], rhs_list[i])

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

    def solve_petsc_uhat(self, solver, rhs_list, frequency, petsc='mumps', *args, **kwargs):
        #try catch for the petsc4py use in multiple rhs solve
        #use only in the data generation where we do not need to compute
        #the system for the whole auxiliary fields
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
        # if the compact operator is used we do not need to slice the solution
        if self.compact:
            usize = ndof
        else:
            nwfield = len(self.WavefieldVector.aux_names) + 1
            usize = ndof//nwfield

        # creating the B rhs Matrix
        B = PETSc.Mat().createDense([ndof, nshot])
        B.setUp()
        for i in range(nshot):
            B.setValues(list(range(0, ndof)), [i], rhs_list[i])

        B.assemblyBegin()
        B.assemblyEnd()

        # call the wrapper to solve the system
        wrapper = PetscWrapper()
        try:
            linear_solver = wrapper.factorize(H, petsc, PETSc.COMM_WORLD)
            Uhat = linear_solver(B.getDenseArray())
        except:
            raise SyntaxError('petsc = '+str(petsc)+' is not a correct solver you can only use \'superlu_dist\', \'mumps\' or \'mkl_pardiso\' ')               
        
        Uhat = Uhat[range(usize),:]

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
        # we build the right helmholtz operator compact or not
        if self._local_support_spec['spatial_dimension']==1:
            # 1D the compact is not implemented so we raise a warming and use the auxiliary field PML
            return (-(omega**2)*self.M + omega*1j*self.C + self.K).tocsc()
        elif self.mesh.x.lbc.type == 'pml':
            if self.mesh.x.lbc.domain_bc.compact:
                # building the compact operator
                oc  = self.operator_components
                if self._local_support_spec['spatial_dimension']==2:
                    dx, dz = self.mesh.deltas
                    length_pml_x = dx * (self.mesh.x.lbc.n -1 )
                    length_pml_z = dz * (self.mesh.z.lbc.n -1 )
                    H1 = -(omega**2)*oc.M
                    H2 = make_diag_mtx(-1j/(omega*length_pml_x) * oc.sxp / (1 - 1j/omega * oc.sx)**3).dot(oc.Dx)
                    H3 = make_diag_mtx(-1j/(omega*length_pml_z) * oc.szp / (1 - 1j/omega * oc.sz)**3).dot(oc.Dz)
                    H4 = - make_diag_mtx(1.0/(1 - 1j/omega * oc.sx)**2).dot(oc.Dxx)
                    H5 = - make_diag_mtx(1.0/(1 - 1j/omega * oc.sz)**2).dot(oc.Dzz)
                    return (H1 + H2 + H3 + H4 + H5).tocsc() # csc is used for the sparse solvers right now
                if self._local_support_spec['spatial_dimension']==3:
                    dx, dy, dz = self.mesh.deltas
                    length_pml_x = dx * (self.mesh.x.lbc.n -1 )
                    length_pml_y = dy * (self.mesh.y.lbc.n -1 )
                    length_pml_z = dz * (self.mesh.z.lbc.n -1 )
                    H1 = -(omega**2)*oc.M
                    H2 = make_diag_mtx(-1j/(omega*length_pml_x) * oc.sxp / (1 - 1j/omega * oc.sx)**3).dot(oc.Dx)
                    H3 = make_diag_mtx(-1j/(omega*length_pml_y) * oc.syp / (1 - 1j/omega * oc.sy)**3).dot(oc.Dy)
                    H4 = make_diag_mtx(-1j/(omega*length_pml_z) * oc.szp / (1 - 1j/omega * oc.sz)**3).dot(oc.Dz)
                    H5 = - make_diag_mtx(1.0/(1 - 1j/omega * oc.sx)**2).dot(oc.Dxx)
                    H6 = - make_diag_mtx(1.0/(1 - 1j/omega * oc.sy)**2).dot(oc.Dyy)
                    H7 = - make_diag_mtx(1.0/(1 - 1j/omega * oc.sz)**2).dot(oc.Dzz)
                    return (H1 + H2 + H3 + H4 + H5 + H6 + H7).tocsc() 
            else:
                return (-(omega**2)*self.M + omega*1j*self.C + self.K).tocsc()
        else:
            return (-(omega**2)*self.M + omega*1j*self.C + self.K).tocsc() # csc is used for the sparse solvers right now
        
