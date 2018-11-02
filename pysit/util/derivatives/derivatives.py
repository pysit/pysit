

import numpy as np
import scipy.sparse as spsp

import pyamg
from pyamg.gallery import stencil_grid

from pysit.util.derivatives.fdweight import *
from pysit.util.matrix_helpers import make_diag_mtx

__all__ = ['build_derivative_matrix','build_derivative_matrix_VDA', 'build_heterogenous_matrices','build_permutation_matrix','_build_staggered_first_derivative_matrix_part', 'build_linear_interpolation_matrix_part']

def build_derivative_matrix(mesh,
                            derivative, order_accuracy,
                            **kwargs):

    if mesh.type == 'structured-cartesian':
        return _build_derivative_matrix_structured_cartesian(mesh, derivative, order_accuracy, **kwargs)
    else:
        raise NotImplementedError('Derivative matrix builder not available (yet) for {0} meshes.'.format(mesh.discretization))

def build_derivative_matrix_VDA(mesh, derivative, order_accuracy, alpha = None, **kwargs): #variable density acoustic
    if mesh.type == 'structured-cartesian':
        return _build_derivative_matrix_staggered_structured_cartesian(mesh, derivative, order_accuracy, alpha=alpha, **kwargs)
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

    max_shift= order_accuracy//2

    if use_shifted_differences:
        # Left side
        odd_even_offset = 1-derivative%2
        for i in range(0, max_shift):
            coeffs = shifted_difference(derivative, order_accuracy, -(max_shift+odd_even_offset)+i)
            mtx[i,0:len(coeffs)] = coeffs/(h**derivative)

        # Right side
        for i in range(-1, -max_shift-1,-1):
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
        for i in range(n_ghost_points):
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
        for i in range(n_ghost_points):
            mtx[-i-1,-i-1] = 1.0

    return mtx.tocsr()

def _build_derivative_matrix_staggered_structured_cartesian(mesh,
                                                            derivative, order_accuracy,
                                                            dimension='all',
                                                            alpha = None,
                                                            return_1D_matrix=False,
                                                            **kwargs):

    #Some of the operators could be cached the same way I did to make 'build_permutation_matrix' faster.
    #Could be considered if the current speed is ever considered to be insufficient.

    import time
    tt = time.time()

    if return_1D_matrix:
        raise Exception('Not yet implemented')

    if derivative < 1 or derivative > 2:
        raise ValueError('Only defined for first and second order right now')

    if derivative == 1 and dimension not in ['x', 'y', 'z']:
        raise ValueError('First derivative requires a direciton')

    sh = mesh.shape(include_bc = True, as_grid = True) #Will include PML padding
    if len(sh) != 2: raise Exception('currently hardcoded 2D implementation, relatively straight-forward to change. Look at the function build_derivative_matrix to get a more general function.')
    nx = sh[0]
    nz = sh[-1]

    #Currently I am working with density input on the regular grid.
    #In the derivation of the variable density solver we only require density at the stagger points
    #For now I am just interpolating density defined on regular points towards the stagger points and use that as 'density model'.
    #Later it is probably better to define the density directly on the stagger points (and evaluate density gradient there to update directly at these points?)

    if type(alpha) == None: #If no alpha is given, we set it to a uniform vector. The result should be the homogeneous Laplacian.
        alpha = np.ones(nx*nz)

    alpha = alpha.flatten() #make 1D

    dx = mesh.x.delta
    dz = mesh.z.delta

    #Get 1D linear interpolation matrices
    Jx_1d = build_linear_interpolation_matrix_part(nx)
    Jz_1d = build_linear_interpolation_matrix_part(nz)


    #Get 1D derivative matrix for first spatial derivative using the desired order of accuracy
    lbc_x = _set_bc(mesh.x.lbc)
    rbc_x = _set_bc(mesh.x.rbc)

    lbc_z = _set_bc(mesh.z.lbc)
    rbc_z = _set_bc(mesh.z.rbc)

    Dx_1d = _build_staggered_first_derivative_matrix_part(nx, order_accuracy, h=dx, lbc = lbc_x, rbc = rbc_x)
    Dz_1d = _build_staggered_first_derivative_matrix_part(nz, order_accuracy, h=dz, lbc = lbc_z, rbc = rbc_z)

    #Some empty matrices of the right shape so we can use kronsum to get the proper 2D matrices for the operations we want.
    #The same is used in the homogeneous 'build_derivative_matrix' function.
    Ix = spsp.eye(nx)
    Iz = spsp.eye(nz)



    Dx_2d = spsp.kron(Dx_1d, Iz, format='csr')
    if dimension == 'x' and derivative == 1:
        return Dx_2d

    Dz_2d = spsp.kron(Ix, Dz_1d, format='csr')
    if dimension == 'z' and derivative == 1:
        return Dz_2d

    #If we are evaluating this we want to make the heterogeneous Laplacian
    Jx_2d = spsp.kron(Jx_1d, Iz, format='csr')
    Jz_2d = spsp.kron(Ix, Jz_1d, format='csr')

    #alpha interpolated to x stagger points. Make diag mat
    diag_alpha_x = make_diag_mtx(Jx_2d*alpha)

    #alpha interpolated to z stagger points. Make diag mat
    diag_alpha_z = make_diag_mtx(Jz_2d*alpha)

    #Create laplacian components
    #The negative transpose of Dx and Dz takes care of the divergence term of the heterogeneous laplacian
    Dxx_2d = -Dx_2d.T*diag_alpha_x*Dx_2d
    Dzz_2d = -Dz_2d.T*diag_alpha_z*Dz_2d

    #Correct the Laplacian around the boundary. This is also done in the homogeneous Laplacian
    #I want the heterogeneous Laplacian to be the same as the homogeneous Laplacian when alpha is uniform
    #This is the only part of the Laplacian that deviates from symmetry, just as in the homogeneous case.
    #But because of these conditions on the dirichlet boundary the wavefield will always equal 0 there and this deviation from symmetry is fine.

    #For indexing, get list of all boundary node numbers
    left_node_nrs  = np.arange(nz)
    right_node_nrs = np.arange((nx-1)*nz,nx*nz)
    top_node_nrs   = np.arange(nz,(nx-1)*nz,nz) #does not include left and right top node
    bot_node_nrs   = top_node_nrs + nz - 1    #does not include left and right top node
    all_boundary_node_nrs   = np.concatenate((left_node_nrs, right_node_nrs, top_node_nrs, bot_node_nrs))
    nb             = all_boundary_node_nrs.size
    L = Dxx_2d + Dzz_2d

    all_node_numbers = np.arange(0,(nx*nz), dtype='int32')
    internal_node_numbers = list(set(all_node_numbers) - set(all_boundary_node_nrs))


    L = L.tocsr() #so we can extract rows efficiently

    #Operation below fixes the boundary rows quite efficiently.
    L_fixed = _turn_sparse_rows_to_identity(L, internal_node_numbers, all_boundary_node_nrs)

    return L_fixed.tocsr()

def _turn_sparse_rows_to_identity(A, rows_to_keep, rows_to_change):
    #Convenience function for removing some rows from the sparse laplacian
    #Had some major performance problems by simply slicing in all the matrix formats I tried.

    nr,nc = A.shape
    if nr != nc:
        raise Exception('assuming square matrix')

    #Create diagonal matrix. When we multiply A by this matrix we can remove rows
    rows_to_keep_diag = np.zeros(nr, dtype='int32')
    rows_to_keep_diag[rows_to_keep] = 1
    diag_mat_remove_rows = make_diag_mtx(rows_to_keep_diag)

    #The matrix below has the rows we want to turn into identity turned to 0
    A_with_rows_removed = diag_mat_remove_rows*A

    #Make diag matrix that has diagonal entries in the rows we want to be identity
    rows_to_change_diag = np.zeros(nr, dtype='int32')
    rows_to_change_diag[rows_to_change] = 1
    A_with_identity_rows = make_diag_mtx(rows_to_change_diag)

    A_modified = A_with_rows_removed + A_with_identity_rows
    return A_modified


def _build_staggered_first_derivative_matrix_part(npoints, order_accuracy, h=1.0, lbc='d', rbc='d'):
    #npoints is the number of regular grid points.

    if order_accuracy%2:
        raise ValueError('Only even accuracy orders supported.')

    #coefficients for the first derivative evaluated in between two regular grid points.
    stagger_coeffs = staggered_difference(1, order_accuracy)/h
    #Use the old 'stencil_grid' routine.
    #Because we do a staggered grid we need to shift the coeffs one entry and the matrix will not be square
    incorrect_mtx = stencil_grid(np.insert(stagger_coeffs,0,0), (npoints, ), format='lil')
    #Get rid of the last row which we dont want in our staggered approach
    mtx = incorrect_mtx[0:-1,:]

    if 'n' in lbc or 'n' in rbc:
        raise ValueError('Did not yet implement Neumann boundaries. Perhaps looking at the centered grid implementation would be a good start?')

    if 'g' in lbc or 'g' in rbc:
        raise ValueError('Did not yet implement this boundary condition yet. Perhaps looking at the centered grid implementation would be a good start?')

    #For dirichlet we don't need to alter the matrix for the first derivative for the boundary nodes as is done in the centered approach
    #The reason is that the first staggered point we evaluate at is in the interior of the domain.
    return mtx.tocsr()

def build_linear_interpolation_matrix_part(npoints):
    #same logic as in function 'build_staggered_first_derivative_matrix_part
    coeffs = np.array([0.5, 0.5])
    incorrect_mtx = stencil_grid(np.insert(coeffs,0,0), (npoints, ), format='lil')
    mtx = incorrect_mtx[0:-1,:]
    return mtx.tocsr()


def apply_derivative(mesh, derivative, order_accuracy, vector, **kwargs):
    A = build_derivative_matrix(mesh, derivative, order_accuracy, **kwargs)
    return A*vector

def build_permutation_matrix(nz,nx):
    # This creates a permutation matrix which transforms a column vector of nx
    # "component" columns of size nz, to the corresponding column vector of nz
    # "component" columns of size nx.

    def generate_matrix(nz, nx): #local function
        P = spsp.lil_matrix((nz*nx,nz*nx))
        for i in range(nz): #Looping is not efficient, but we only need to do it once as setup
            for j in range(nx):
                P[nx*i+j,i+j*nz]=1

        return P.tocsr()

    #Start body of code for 'build_permutation_matrix'
    try: #See if there are already stored results from previous calls to this function
        current_storage_dict = build_permutation_matrix.storage_dict
    except: #If not, initialize
        current_storage_dict = dict()
        build_permutation_matrix.storage_dict = current_storage_dict

    if (nz,nx) not in list(current_storage_dict.keys()): #Have not precomputed this!
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
    for i in range(nx):
        alpha_z.append(Lz*alpha[nz*i:nz*(i+1)])
    for i in range(nz):
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
    v1[list(range(nz-1,nz*nx-1,nz))]=0.0  # repair boundary terms.
    D2=spsp.diags([v,v1],[0,1])

    D2_tilda=-1.0*D2.T

    #builds x derivative matrix
    p=-np.ones(nx*nz)/deltas[0]
    p1=np.ones(nx*nz-1)/deltas[0]
    #p[range(nx-1,nz*nx,nx)]=-1.0
    p1[list(range(nx-1,nz*nx-1,nx))]=0.0
    D1=spsp.diags([p,p1],[0,1])

    D1_tilda=-1.0*D1.T

    P=build_permutation_matrix(nz,nx)
    P_inv=build_permutation_matrix(nx,nz)

    #builds exact adjoint gradient for z.
    v=-np.ones(nx*nz)/deltas[-1]
    v1=np.ones(nx*nz-1)/deltas[-1]
    v1[list(range(nz-1,nz*nx-1,nz))]=0.0
    v1[list(range(0,nz*nx-1,nz))]=0.0
    D2_adj=spsp.diags([v,v1],[0,1])

    #builds exact adjoint gradient for x.
    p=-np.ones(nx*nz)/deltas[0]
    p1=np.ones(nx*nz-1)/deltas[0]
    p1[list(range(nx-1,nz*nx-1,nx))]=0.0
    p1[list(range(0,nz*nx-1,nx))]=0.0
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

        print(x[:,:,0]) # should have ones all in first and last rows
        print(y[:,:,0]) # should have ones all in first and last columns
        print(z[0,0,:]) # should have ones at the ends
