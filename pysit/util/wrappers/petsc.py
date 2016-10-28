import time

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    haspetsc = True
except:
    haspetsc = False

import scipy.sparse as spsp


class PetscWrapper():

    def __init__(self):

        if not haspetsc:
            raise NotImplementedError("petsc4py not supported")

        self.solverType = []
        self.H_inv = []

    def factorize(self, H, solverType, comm=None):
        
        if comm is None:
            comm = PETSc.COMM_WORLD
        
        self.solverType = solverType
        
        # check if H is a csr matrix otherwise converts H into csr format
        if type(H) != spsp.csr.csr_matrix:
            H_csr = H.tocsr()
        else:
            H_csr = H

        # converting the scipy matrix to a PETsc Matrix
        H_petsc = PETSc.Mat().createAIJ(size=H_csr.shape, csr=(H_csr.indptr, H_csr.indices, H_csr.data))

        # Assembling the matrices for PETSC
        H_petsc.assemblyBegin()
        H_petsc.assemblyEnd()

        # we use the PETSc Krylov solvers interface but we only use the preconditioner.
        ksp = PETSc.KSP()
        ksp.create(comm)
        ksp.setOperators(H_petsc)
        ksp.setType('preonly') # setting the solver to only use the preconditioner

        # extracting the preconditioner routine from the Krylov space routine
        pc = ksp.getPC()
        pc.setType('lu') # setting the preconditioner to use an LU factorization

        pc.setFactorSolverPackage(self.solverType) # using mumps as a sparse algorithm

        # setting the solvers from the options
        pc.setFromOptions()
        ksp.setFromOptions()

        # factorizing the matrix
        pc.setUp()
        

        # getting the factorized matrix
        self.H_inv = pc.getFactorMatrix()

        return self.solve

    def solve(self, B):
        
        if B.shape[0] == self.H_inv.size[0]:
            if B.ndim == 1:
                nrhs = 1
                N = B.shape[0]
                B = B.reshape((N,nrhs))
            else :
                nrhs = B.shape[1]
                N = B.shape[0]
        else:
            raise ValueError("Matrix dimension does not match the length of right hand side")

        # Allocating the matrices in Petsc
        BPetsc = PETSc.Mat().createDense((N, nrhs))
        BPetsc.setUp()

        # coping the RHS matrix to ta Petsc matrix
        for ii in xrange(0, nrhs, 1):
            BPetsc.setValues(range(0, N), [ii], B[:, ii])

        # BPetsc.setValues(range(0, B.shape[0]), range(0, B.shape[1]), B)

        # assembling the RHS matrix
        BPetsc.assemblyBegin()
        BPetsc.assemblyEnd()

        # defining and allocating the solution matrix
        X = PETSc.Mat().createDense((N, nrhs))
        X.setUp()

        # assembling the solution matrix
        X.assemblyBegin()
        X.assemblyEnd()

        # solving the system by calling PetSc
        self.H_inv.matSolve(BPetsc, X)

        # somehow it doesn't like it if I don't force the np.array
        return np.array(X.getDenseArray())
