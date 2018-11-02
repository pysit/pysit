****************************
Part 5: The Frequency Domain
****************************

In this exercise, you will implement solutions to the same problem, except in
the frequency domain.  Many of your existing functions will be very useful.

In the frequency domain, the (time-shifted) Ricker wavelet is defined as

.. math::

   \hat w(\nu) = \frac{2}{\sqrt{\pi}} \frac{\nu^2}{\nu_0^3}\exp\{-\frac{\nu^2}{\nu_0^2}\}\exp\{-i2\pi\nu t_0\}

where :math:`\nu` is the frequency in Hz.


.. topic:: Problem 5.1

    Implement a frequency version of your ``ricker_wavelet`` function. Plot
    the absolute value of :math:`\hat w(\nu)` for a reasonable range of
    :math:`\nu` and verify that the peak occurs at :math:`\pm \nu_0`.


.. topic:: Problem 5.2

    Derive the Helmholtz (frequency domain) differential equation equivalent
    to the time domain wave equation from the first problem set. How do the
    mass, attenuation, and stiffness matrices relate to this equation?

    .. note::

        Be careful of the Fourier transform convention. The Ricker wavelet
        above uses :math:`\hat f(\nu) = \int_0^T f(t)\exp\{-i2\pi\nu
        t\}\textrm{d}t` as the convention. Then, :math:`f(t) =
        \int_{-\infty}^{\infty} \hat f(\nu)\exp\{i2\pi\nu t\}\textrm{d}\nu` is
        the inverse transform.


.. topic:: Problem 5.3

    Using your ``point_source`` and ``construct_matrices`` functions, write a
    function ``helmholtz_solver(C, source, nu, config)`` which solves the
    Helmholtz equation, that you developed in Problem 5.2, at a single
    frequency ``nu``. Write a function ``forward_operator_frequency`` which
    behaves like the time-domain ``forward_operator`` function, except that it
    implements

    .. math::

       \hat{\mathcal{F}[m]} = \hat u,

    and

    .. math::

       S\hat{\mathcal{F}[m]} = \hat d.

    Your ``record_data`` function should also be useful.


.. topic:: Problem 5.4

    Derive the equivalent adjoint state method for the frequency domain. What
    is the adjoint equation? What are the adjoint sources? Write a function
    ``adjoint_operator_frequency`` which implements the adjoint state method,
    using your ``helmholtz_solver``,

    .. math::

       \hat{F^*}\hat r_\text{ext} = \delta m.

    .. admonition:: Hint

        Be very careful with your conjugates.


.. topic:: Problem 5.5

    Derive the equivalent linear forward model, and using your
    ``helmholtz_solver`` function, implement a
    ``linear_forward_operator_frequency`` function for applying

    .. math::

       \hat F\delta m = \hat u_1.


.. topic:: Problem 5.6

    Verify that the adjoint relationship is satisfied for these operators.
    Again, be careful with your conjugates.

.. topic:: Problem 5.7

    Implement gradient descent for the frequency domain. Why don't you get
    useful answers using a single frequency (this is exceptional for 1D)? 

    What happens if you formulate the optimization such that you are using
    multiple frequencies at once? How do your results compare to those from
    the time domain?


Bonus Problems
==============

**Bonus Problem 5.8:** Derive the frequecy domain version of the Ricker
wavelet.

**Bonus Problem 5.9:** Derive how you can implement these same frequency
domain operations using only time-domain solvers. Can you implement this? What
extra 'tweaks' do you need to do to ensure that your time-based frequency
operators pass the adjoint test?

**Bonus Problem 5.10:** Numerically verify that your time- and
frequency-domain solvers satisfy the discrete Plancharel's theorem.
