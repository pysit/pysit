import itertools

import numpy as np
import scipy
import scipy.signal as signal

__all__ = ['gaussian_kernel', 'blur_image', 'resample_array']

def gaussian_kernel(size, sigma, mesh_deltas=None):
    """ Returns a normalized gauss kernel array for convolutions.

    Parameters
    ----------
    size : iterable
        Number of pixels per standard deviation
    sigma : float or iterable
        Standard deviation of gaussian in physical units
    mesh_deltas : iterable, optional
        Grid size in physical units.  Defaults to array of ones (pixel units).

    """
    imsize = size.astype(int)

    if mesh_deltas is None:
        mesh_deltas = np.ones(imsize)

    # Use numpy index tricks to generate a meshgrid like array for each dimension.
    # Product with the delta ensures that the indices are in the physical units
    # specified.
    idx = tuple([slice(-s, s+1) for s in imsize])
    grid = [g.astype(np.float)*dx for g,dx in zip(np.mgrid[idx], mesh_deltas)]

    # Create a different kernel if sigma is specifed differently for different axes.
    if np.iterable(sigma) and len(sigma) == len(grid):
        arg = sum([dim.astype(float)**2/(2*(s**2)) for dim,s in zip(grid,sigma)])
    else:
        arg = sum([dim**2/(2*float(sigma)**2) for dim in grid])

    g = np.exp(-(arg))

    # Return the normalized Gaussian
    return g/g.sum()

def blur_image(im, sigma=None, freq=None, mesh_deltas=None, n_sigma=1.0):
    """ Returns a blurred image by convolving with a gaussian kernel.

    Parameters
    ----------
    im : ndarray
        Input image
    sigma : float or ndarray
        Standard deviation of blur kernel
    freq :
        Cutoff (spatial) frequency of the blur kernel.  Equivalent to 1/sigma.
    mesh_deltas : iterable, optional
        Grid size in physical units.  Defaults to array of ones (pixel units).
    n_sigma : float
        Width of the blur kernel in standard deviations.

    Notes
    -----
    * One of `sigma` or `freq` must be specified.  If both are specified,
      `freq` takes priority.
    """

    if sigma is None and freq is None:
        raise ValueError('Either sigma or frequency must be set.')

    if mesh_deltas is None:
        mesh_deltas = np.ones(im.ndim)

    if not np.iterable(mesh_deltas):
        mesh_deltas = np.array([mesh_deltas])
    else:
        mesh_deltas = np.asarray(mesh_deltas)

    # frequency takes priority if both freq and sigma are specified.
    if freq is not None:
        sigma = 1./freq

    # determine the size, in pixels, of the kernel
    kernel_size_pixel = (np.ceil(n_sigma*sigma / mesh_deltas)).astype('int32')

    kernel = gaussian_kernel(kernel_size_pixel, sigma, mesh_deltas)

    # Pad the image by the kernel size on all sides, to get smoothing without
    # edge effects
    im_ = np.pad(im, list(zip(kernel_size_pixel,kernel_size_pixel)), mode='edge')

    improc = signal.fftconvolve(im_, kernel, mode='valid')

    #return a copy to ensure a contiguous array
    return improc.copy()

def resample_array(arr, new_size, mode='nearest'):
    """ Returns a resampled array at new resolution.

    Parameters
    ----------
    arr : ndarray
        Input array
    new_size : iterable
        Size of the output array
    mode : {'nearest', 'linear'}
        Interpolation mode.
    """

    sh = arr.shape

    y = arr

    # For each dimension, construct a 1D interpolator and interpolate.
    # new size, old size
    for nsz, osz, i in zip(new_size, sh, itertools.count()):

        x = np.arange(osz, dtype=np.float)
        I = scipy.interpolate.interp1d(x, y, copy=False, axis=i, kind=mode)
        new_x = np.linspace(0, osz-1, nsz)
        y = I(new_x)

    return y

