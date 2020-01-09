import math

import numpy as np

__all__ = ['SourceWaveletBase', 'DerivativeGaussianPulse', 'RickerWavelet', 'GaussianPulse', 'WhiteNoiseSource']

_sqrt2 = math.sqrt(2.0)


def _arrayify(arg):
    if not np.iterable(arg):
        return True, np.array([arg])
    else:
        return False, np.asarray(arg)


class SourceWaveletBase(object):
    """ Base class for source wavelets or profile functions.

    This is implemented as a function object, so the magic happens in the
    `__call__` member function.

    Methods
    -------

    __call__(self, t=None, nu=None, **kwargs)

    """

    @property
    def time_source(self):
        """bool, Indicates if wavelet is defined in time domain."""
        return False

    @property
    def frequency_source(self):
        """bool, Indicates if wavelet is defined in frequency domain."""
        return False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('')

    def __call__(self, t=None, nu=None, **kwargs):
        """Callable object method for the seismic sources.

        Parameters
        ----------
        t : float, array-like
            Time(s) at which to evaluate wavelet.
        nu : float, array-like, optional
            Frequency(ies), in Hz, at which to evaluate wavelet.

        """

        if t is not None:
            if self.time_source:
                return self._evaluate_time(t)
            else:
                raise TypeError('Sources of type {0} are not time-domain sources.'.format(self.__class__.__name__))
        elif nu is not None:
            if self.frequency_source:
                return self._evaluate_frequency(nu)
            else:
                raise TypeError('Sources of type {0} are not time-domain sources.'.format(self.__class__.__name__))
        else:
            raise ValueError('Either a time or frequency must be provided.')


class DerivativeGaussianPulse(SourceWaveletBase):
    """ Pulse as the n-th derivative of a Gaussian.

    Defined in the notation of the Ricker wavelet shown here [1]_.  Provides
    arbitrary wavelets as the n-th derivative of Gaussian functions in both
    time and in frequency.

    Attributes
    ----------
    peak_frequency : float
        Frequency at which the Ricker wavelet is centered.
    order : integer, optional
        Specifies which derivative to use, 0th is default.
    threshold : float, optional
        Drop tolerance for evaluation.
    shift_deviations : float, optional
        Number of standard deviations of the base Gaussian to time shift wavelet.
    t_shift : float, optional
        Time shift from zero; overrides shift_deviations.

    Notes
    -----

    The notation is slightly non-standard in order to preserve correspondance
    with [1]_.  However, the definitions in time and in frequency are indeed
    internally consistent.

    References
    ----------

    .. [1] N.  Ricker, "The form and laws of propagation of seismic wavelets,"
       Geophysics, vol. 18, pp. 10-40, 1953.

    """

    precomputed_values_t = dict()   #Shared amongst instances of class. Right now only for time domain, because this computes it at every timestep.
                                    #This vastly improves the performance of source encoded supershots, as many source evaluations are required each timestep.
    @property
    def time_source(self):
        """bool, Indicates if wavelet is defined in time domain."""
        return True

    @property
    def frequency_source(self):
        """bool, Indicates if wavelet is defined in frequency domain."""
        return True

    def __init__(self, peak_frequency, order=0, threshold=1e-6, shift_deviations=6, t_shift=None):
        self.order = order
        self.peak_frequency = peak_frequency
        self.threshold = threshold
        self.shift_deviations = shift_deviations

        nu = peak_frequency

        self.sigma = 1/(math.pi*nu*_sqrt2)

        if t_shift is None:
            self.t_shift = self.shift_deviations*self.sigma
        else:
            self.t_shift = t_shift

        poly_coeffs = (order)*[0.0]+[1.0]
        self._hermite = np.polynomial.Hermite(poly_coeffs)

    def _evaluate_time(self, ts):

        # Vectorize the time list
        ts_was_not_array, ts = _arrayify(ts)

        n = self.order

        v = []
        for t in ts:
            if (t,self.sigma,self.t_shift,self.threshold) not in DerivativeGaussianPulse.precomputed_values_t: #Not precomputed
                x = (t-self.t_shift)/(_sqrt2*self.sigma)
                c = (-1/_sqrt2)**n
                _v = c*self._hermite(x)*np.exp(-(x**2))
                if np.abs(_v) < self.threshold: _v = 0.0
                DerivativeGaussianPulse.precomputed_values_t[t,self.sigma,self.t_shift,self.threshold] = _v
            v.append(DerivativeGaussianPulse.precomputed_values_t[t,self.sigma,self.t_shift,self.threshold])

        return v[0] if ts_was_not_array else np.array(v)

    def _evaluate_frequency(self, nus):

        # Vectorize the frequency list
        nus_was_not_array, nus = _arrayify(nus)

        omegas = 2*np.pi*nus
        n = self.order

        shift = np.exp(-1j*2*np.pi*nus*self.t_shift)

        a = (-1)**n
        b = (1j*omegas)**n
        c = self.sigma**(n+1)
        d = math.sqrt(2*np.pi)
        v = d*a*b*c*np.exp(-0.5*(self.sigma**2) * omegas**2)*shift

        v[np.abs(v) < self.threshold] = 0.0

        return v[0] if nus_was_not_array else v


class RickerWavelet(DerivativeGaussianPulse):
    """ Canonical example source wavelet.

    The Ricker wavelet is the negative 2nd derivative of a Gaussian [1]_.

    References
    ----------

    .. [1] N.  Ricker, "The form and laws of propagation of seismic wavelets,"
       Geophysics, vol. 18, pp. 10-40, 1953.

    """

    # Not allowed to change the order for the RickerWavelet.
    @property
    def order(self):
        return 2

    @order.setter
    def order(self, n):
        pass

    def __init__(self, nu, **kwargs):
        DerivativeGaussianPulse.__init__(self, nu, order=self.order, **kwargs)

    def _evaluate_time(self, ts):
        return -1*DerivativeGaussianPulse._evaluate_time(self, ts)

    def _evaluate_frequency(self, nus):
        return -1*DerivativeGaussianPulse._evaluate_frequency(self, nus)


class GaussianPulse(DerivativeGaussianPulse):

    """ 0th derivative of the Gaussian"""

    # Not allowed to change the order for the RickerWavelet.
    @property
    def order(self):
        return 0

    @order.setter
    def order(self, n):
        pass

    def __init__(self, nu, **kwargs):
        DerivativeGaussianPulse.__init__(self, nu, order=self.order, **kwargs)


class WhiteNoiseSource(SourceWaveletBase):

    """ Random wavelet.

    Notes
    -----

    Do not use for both time and frequency simultaneously, as realizations are
    not coherent.

    """

    @property
    def time_source(self):
        """bool, Indicates if wavelet is defined in time domain."""
        return True

    @property
    def frequency_source(self):
        """bool, Indicates if wavelet is defined in frequency domain."""
        return True

    def __init__(self, seed=None, variance=1.0, **kwargs):

        # time domain storage, of dubious merit for implementing in this manner.
        self._f = dict()
        # frequency domain storage
        self._f_hat = dict()

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # This is currently ignored.
        self.variance = variance

    def _evaluate_time(self, ts):

        # Vectorize the time list
        ts_was_not_array, ts = _arrayify(ts)

        v = list()
        for t in ts:
            if t not in self._f:
                self._f[t] = self.variance*(np.random.randn())
            v.append(self._f[t])

        return v[0] if ts_was_not_array else np.array(v)

    def _evaluate_frequency(self, nus):

        # Vectorize the frequency list
        nus_was_not_array, nus = _arrayify(nus)

        v = list()
        for nu in nus:
            if nu not in self._f_hat:
                self._f_hat[nu] = self.variance*(np.random.randn() + np.random.randn()*1j)
            v.append(self._f_hat[nu])

        return v[0] if nus_was_not_array else np.array(v)
