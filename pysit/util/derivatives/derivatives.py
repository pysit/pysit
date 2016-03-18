from __future__ import absolute_import

import numpy as np
import scipy.sparse as spsp

import pyamg
from pyamg.gallery import stencil_grid

from pysit.util.derivatives.fdweight import *

__all__ = ['build_derivative_matrix','build_heterogenous_laplacian','build_heterogenous_matrices','build_permutation_matrix']

def build_derivative_matrix(mesh,
                            derivative, order_accuracy,
                            **kwargs):

    if mesh.type == 'structured-cartesian':
        return _build_derivative_matrix_structured_cartesian(mesh, derivative, order_accuracy, **kwargs)
    else:
        raise NotImplementedError('Derivative matrix builder not available (yet) for {0} meshes.'.format(mesh.discretization))

def _set_bc(bc):
    if bc.type == 'pml':
        return bc.boundary_type
    elif bc.type == 'ghost':
        return ('ghost', bc.n)
    else:
        return bc.type



def _build_derivative_matrix_structured_cartesian(mesh,
                                                  derivative, order_accuracy,
                                                  dimension='all',
                                                  use_shifted_differences=False,
                                                  return_1D_matrix=False,
                                                  **kwargs):

    dims = list()
    if type(dimension) is str:
        dimension = [dimension]
    if 'all' in dimension:
        if mesh.dim > 1:
            dims.append('x')
        if mesh.dim > 2:
            dims.append('y')
        dims.append('z')
    else:
        for d in dimension:
            dims.append(d)

    # sh[-1] is always 'z'
    # sh[0] is always 'x' if in 2 or 3d
    # sh[1] is always 'y' if dim > 2
    sh = mesh.shape(include_bc = True, as_grid = True)

    if mesh.dim > 1:
        if 'x' in dims:
            lbc = _set_bc(mesh.x.lbc)
            rbc = _set_bc(mesh.x.rbc)
            delta = mesh.x.delta
            Dx = _build_derivative_matrix_part(sh[0], derivative, order_accuracy, h=delta, lbc=lbc, rbc=rbc, use_shifted_differences=use_shifted_differences)
        else:
            Dx = spsp.csr_matrix((sh[0],sh[0]))
    if mesh.dim > 2:
        if 'y' in dims:
            lbc = _set_bc(mesh.y.lbc)
            rbc = _set_bc(mesh.y.rbc)
            delta = mesh.y.delta
            Dy = _build_derivative_matrix_part(sh[1], derivative, order_accuracy, h=delta, lbc=lbc, rbc=rbc, use_shifted_differences=use_shifted_differences)
        else:
            Dy = spsp.csr_matrix((sh[1],sh[1]))

    if 'z' in dims:
        lbc = _set_bc(mesh.z.lbc)
        rbc = _set_bc(mesh.z.rbc)
        delta = mesh.z.delta
        Dz = _build_derivative_matrix_part(sh[-1], derivative, order_accuracy, h=delta, lbc=lbc, rbc=rbc, use_shifted_differences=use_shifted_differences)
    else:
        Dz = spsp.csr_matrix((sh[-1],sh[-1]))

    if return_1D_matrix and 'all' not in dims:
        if 'z' in dims:
            mtx = Dz
        elif 'y' in dims:
            mtx = Dy
        elif 'x' in dims:
            mtx = Dx
    else:
        if mesh.dim == 1:
            mtx = Dz.tocsr()
        if mesh.dim == 2:
            # kronsum in this order because wavefields are stored with 'z' in row
            # and 'x' in columns, then vectorized in 'C' order
            mtx = spsp.kronsum(Dz, Dx, format='csr')
        if mesh.dim == 3:
            mtx = spsp.kronsum(Dz, spsp.kronsum(Dy,Dx, format='csr'), format='csr')

    return mtx

def _build_derivative_matrix_part(npoints, derivative, order_accuracy, h=1.0, lbc='d', rbc='d', use_shifted_differences=False):

    if order_accuracy%2:
        raise ValueError('Only even accuracy orders supported.')

    centered_coeffs = centered_difference(derivative, order_accuracy)/(h**derivative)

    mtx = stencil_grid(centered_coeffs, (npoints, ), format='lil')

    max_shift= order_accuracy/2

    if use_shifted_differences:
        # Left side
        odd_even_offset = 1-derivative%2
        for i in xrange(0, max_shift):
            coeffs = shifted_difference(derivative, order_accuracy, -(max_shift+odd_even_offset)+i)
            mtx[i,0:len(coeffs)] = coeffs/(h**derivative)

        # Right side
        for i in xrange(-1, -max_shift-1,-1):
            coeffs = shifted_difference(derivative, order_accuracy, max_shift+i+odd_even_offset)
            mtx[i,slice(-1, -(len(coeffs)+1),-1)] = coeffs[::-1]/(h**derivative)

    if 'd' in lbc: #dirichlet
        mtx[0,:] = 0
        mtx[0,0] = 1.0
    elif 'n' in lbc: #neumann
        mtx[0,:] = 0
        coeffs = shifted_difference(1, order_accuracy, -max_shift)/h
        coeffs /= (-1*coeffs[0])
        coeffs[0] = 0.0
        mtx[0,0:len(coeffs)] = coeffs
    elif type(lbc) is tuple and 'g' in lbc[0]: #ghost
        n_ghost_points = int(lbc[1])
        mtx[0:n_ghost_points,:] = 0
        for i in xrange(n_ghost_points):
            mtx[i,i] = 1.0

    if 'd' in rbc:
        mtx[-1,:] = 0
        mtx[-1,-1] = 1.0
    elif 'n' in rbc:
        mtx[-1,:] = 0
        coeffs = shifted_difference(1, order_accuracy, max_shift)/h
        coeffs /= (-1*coeffs[-1])
        coeffs[-1] = 0.0
        mtx[-1,slice(-1, -(len(coeffs)+1),-1)] = coeffs[::-1]
    elif type(rbc) is tuple and 'g' in rbc[0]:
        n_ghost_points = int(rbc[1])
        mtx[slice(-1,-(n_ghost_points+1), -1),:] = 0
        for i in xrange(n_ghost_points):
            mtx[-i-1,-i-1] = 1.0

    return mtx.tocsr()


def apply_derivative(mesh, derivative, order_accuracy, vector, **kwargs):
    A = build_derivative_matrix(mesh, derivative, order_accuracy, **kwargs)
    return A*vector

def build_heterogenous_laplacian(sh,alpha,deltas):
    # This is a 2D "Heteogenous Laplacian" matrix operator constructor.
    # It takes a mesh (including BC), and returns a square, sparse matrix of size 
    # mesh.x.n*mesh.z.n. The operator is div(alpha grad), and is second order accurate.
    # for our purposes, alpha usually is (1/rho), or, model m2. 
    nz = sh[-1]
    nx = sh[0]
    P = build_permutation_matrix(nz,nx)
    P_inv = build_permutation_matrix(nx,nz)

    alpha_x, alpha_z = build_offcentered_alpha(sh,alpha)  # alpha is passed into this routine centered on the computational nodes.
                                                          # build_offcentered_alpha returns alpha cenetered on the midpoints of the nodes.
                                                          
    # Builds x part of laplacian, assuming dirichlet boundaries at the edge of the PML
    km1, k, kp1 = np.zeros(nx*nz-1), np.zeros(nx*nz), np.zeros(nx*nz-1)
    for i in xrange(nz):
        for j in xrange(nx-1):
            if j!=(nx-2):
                km1[i*nx+j]=alpha_x[i][j+1][0]
            else:
                km1[i*nx+j]=0.0
            if j!=0:
                k[i*nx+j]=-(alpha_x[i][j][0]+alpha_x[i][j+1][0])
                kp1[i*nx+j]=alpha_x[i][j+1][0]
            else:
                k[i*nx+j]=deltas[0]**2    # this is so later, when we divice by delta[0]**2,
                kp1[i*nx+j]=0.0           # this value turns to 1.0
        k[i*nx+(nx-1)]=deltas[0]**2       
        if i!=(nz-1):
            km1[i*nx+(nx-1)]=0.0
            kp1[i*nx+(nx-1)]=0.0

    Lx=spsp.diags([km1,k,kp1],[-1,0,1],dtype='float')
    Lx/=deltas[0]**2

    # Builds z part of laplacian, assuming dirichlet boundaries at the edge of the PML
    km1,k,kp1=np.zeros(nx*nz-1),np.zeros(nx*nz),np.zeros(nx*nz-1)
    for i in xrange(nx):
        for j in xrange(nz-1):
            if j!=(nz-2):
                km1[i*nz+j]=alpha_z[i][j+1][0]
            else:
                km1[i*nz+j]=0.0
            if j!=0:
                k[i*nz+j]=-(alpha_z[i][j][0]+alpha_z[i][j+1][0])
                kp1[i*nz+j]=alpha_z[i][j+1][0]
            else:
                k[i*nz+j]=deltas[1]**2      # this is so later, when we divice by delta[0]**2,
                kp1[i*nz+j]=0.0             # this value turns to 1.0
        k[i*nz+(nz-1)]=deltas[1]**2         

        if i!=(nx-1):
            km1[i*nz+(nz-1)]=0.0
            kp1[i*nz+(nz-1)]=0.0

    Lz=spsp.diags([km1,k,kp1],[-1,0,1],dtype='float')
    Lz/=deltas[1]**2
    
    # the permutation matrix (and following inverse permutation)
    # allows us to use Lx against the same type of vector Lz acts against,
    # because the permutation corrects the arrangment of the entries in 
    # the vector Lx is applied against. 

    Lap=Lz+P_inv*Lx*P

    return Lap

def build_permutation_matrix(nz,nx):
    # This creates a permutation matrix which transforms a column vector of nx
    # "component" columns of size nz, to the corresponding column vector of nz
    # "component" columns of size nx.
    
    def generate_matrix(nz, nx): #local function
        P = spsp.lil_matrix((nz*nx,nz*nx)) 
        for i in xrange(nz): #Looping is not efficient, but we only need to do it once as setup
            for j in xrange(nx):
                P[nx*i+j,i+j*nz]=1
    
        return P.tocsr()
    
    #Start body of code for 'build_permutation_matrix'
    try: #See if there are already stored results from previous calls to this function
        current_storage_dict = build_permutation_matrix.storage_dict
    except: #If not, initialize
        current_storage_dict = dict()
        build_permutation_matrix.storage_dict = current_storage_dict
    
    if (nz,nx) not in current_storage_dict.keys(): #Have not precomputed this!
        mat = generate_matrix(nz,nx)
        current_storage_dict[nz,nx] = mat
 
    return current_storage_dict[nz,nx]

def build_offcentered_alpha(sh,alpha):
    # This computes the midpoints of alpha which will be used in the heterogenous laplacian
    nz=sh[-1]
    nx=sh[0]
    
    v1z,v2z,v3z=np.ones(nz),np.ones(nz-1),np.zeros(nz)
    v1z[-1],v3z[0]=2.0,2.0
    v1x,v2x,v3x=np.ones(nx),np.ones(nx-1),np.zeros(nx)
    v1x[-1],v3x[0]=2.0,2.0
    v3z=v3z.reshape(1,nz)
    v3x=v3x.reshape(1,nx)
    Lz1=np.array(spsp.diags([v1z,v2z],[0,1]).todense())
    Lx1=np.array(spsp.diags([v1x,v2x],[0,1]).todense())
    Lz=np.matrix(0.5*np.concatenate((v3z,Lz1),axis=0))
    Lx=np.matrix(0.5*np.concatenate((v3x,Lx1),axis=0))
    # Lz and Lx simply (of length nz and nx respectively) act on a vector and return one which is one entry larger than before,
    # with each entry being a weighted sum of the two adjacent entries. Boundary values are preserved.
    P=build_permutation_matrix(nz,nx)
    
    alpha_perm=P*alpha
    alpha_z,alpha_x=list(),list()
    for i in xrange(nx):
        alpha_z.append(Lz*alpha[nz*i:nz*(i+1)])
    for i in xrange(nz):
        alpha_x.append(Lx*alpha_perm[nx*i:nx*(i+1)])
    return alpha_x, alpha_z

def build_heterogenous_matrices(sh,deltas,alpha=None,rp=None):
    # This builds 1st order, forward and backward derivative matrices. 
    # alpha is a vector which goes inside of the operator, div (alpha grad)
    # It can also build a hetergenous laplacian (if rp is not None),which differs from the above
    # heterogenous laplacian only in its boundary conditions.
    
    nz=sh[-1]
    nx=sh[0]
    
    #builds z derivative matrix
    v=-np.ones(nx*nz)/deltas[-1]
    v1=np.ones(nx*nz-1)/deltas[-1]
    v1[range(nz-1,nz*nx-1,nz)]=0.0  # repair boundary terms.
    D2=spsp.diags([v,v1],[0,1])
    
    D2_tilda=-1.0*D2.T

    #builds x derivative matrix
    p=-np.ones(nx*nz)/deltas[0]
    p1=np.ones(nx*nz-1)/deltas[0]
    #p[range(nx-1,nz*nx,nx)]=-1.0
    p1[range(nx-1,nz*nx-1,nx)]=0.0
    D1=spsp.diags([p,p1],[0,1])
    
    D1_tilda=-1.0*D1.T

    P=build_permutation_matrix(nz,nx)
    P_inv=build_permutation_matrix(nx,nz)

    #builds exact adjoint gradient for z.
    v=-np.ones(nx*nz)/deltas[-1]
    v1=np.ones(nx*nz-1)/deltas[-1]
    v1[range(nz-1,nz*nx-1,nz)]=0.0
    v1[range(0,nz*nx-1,nz)]=0.0
    D2_adj=spsp.diags([v,v1],[0,1])

    #builds exact adjoint gradient for x.
    p=-np.ones(nx*nz)/deltas[0]
    p1=np.ones(nx*nz-1)/deltas[0]
    p1[range(nx-1,nz*nx-1,nx)]=0.0
    p1[range(0,nz*nx-1,nx)]=0.0
    D1_adj=spsp.diags([p,p1],[0,1])

    if rp is not None:
        A=spsp.diags([alpha],[0])
        Lap = D2_tilda*A*D2+P_inv*D1_tilda*P*A*P_inv*D1*P
        return Lap
    else:
        D1=P_inv*D1*P
        D1_adj=P_inv*D1_adj*P
        return [D1,D1_adj],[D2,D2_adj] 

if __name__ == '__main__':

    from pysit import *
    from pysit.gallery import horizontal_reflector

    bc = Dirichlet()

    dim = 2
    deriv = 1 # 2
    order = 4

    if dim == 1:
        z_config = (0.0, 7.0, bc, bc)

        d = RectangularDomain(z_config)

        m = CartesianMesh(d, 7)
        #   Generate true wave speed
        C, C0 = horizontal_reflector(m)


        D = build_derivative_matrix(m, deriv, order, dimension='all').todense()

        Dz = build_derivative_matrix(m, deriv, order, dimension='z').todense()

    if dim == 2:

        x_config = (0.0, 7.0, bc, bc)
        z_config = (0.0, 7.0, bc, bc)

        d = RectangularDomain(x_config, z_config)

        m = CartesianMesh(d, 7, 7)

        #   Generate true wave speed
        C, C0 = horizontal_reflector(m)

        D = build_derivative_matrix(m, deriv, order, dimension='all').todense()

        Dx = build_derivative_matrix(m, deriv, order, dimension='x').todense()
        Dz = build_derivative_matrix(m, deriv, order, dimension='z').todense()


    if dim == 3:

        x_config = (0.0, 7.0, bc, bc)
        y_config = (0.0, 7.0, bc, bc)
        z_config = (0.0, 7.0, bc, bc)

        d = RectangularDomain(x_config, x_config, z_config)

        m = CartesianMesh(d, 7, 7, 7)

        #   Generate true wave speed
        C, C0 = horizontal_reflector(m)


        D = build_derivative_matrix(m, deriv, order, dimension='all').todense()

        sh = m.shape(as_grid=True)

        Dx = build_derivative_matrix(m, deriv, order, dimension=['x']).todense()
        Dy = build_derivative_matrix(m, deriv, order, dimension=['y']).todense()
        Dz = build_derivative_matrix(m, deriv, order, dimension=['z']).todense()

        x=(Dx*C).reshape(sh)
        y=(Dy*C).reshape(sh)
        z=(Dz*C).reshape(sh)

        print x[:,:,0] # should have ones all in first and last rows
        print y[:,:,0] # should have ones all in first and last columns
        print z[0,0,:] # should have ones at the ends
