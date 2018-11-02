***********************************
Part 4: Gradient-based Optimization
***********************************

Finally, you will implement a simple gradient descent algorithm for minimizing
the objective function

.. math::

   J(m) = \frac{1}{2}\int_0^T||d_{x_r}(t) - \mathbf{S}\mathcal{F}[m](t)||_2^2
   \textrm{d}t,

to 'solve' the seismic imaging problem.

Gradient Descent
================

Full waveform inversion is the geophysics terminology for minimizing the
objective

.. math::

   J(m) = \frac{1}{2}\int_0^T||d_{x_r}(t) - \mathbf{S}\mathcal{F}[m]||_2^2 \textrm{d}t.

The term arises because we will be using the wave equation simulation to match
the seismic data to 'invert' or solve the seismic inverse problem.

From the `notes <http://math.mit.edu/icg/resources/notes325.pdf>`_, we have
that the gradient of the objective at a point :math:`m_0` is

.. math::

   \nabla J[m_0] = -F^*(d_{x_r} - \mathbf{S}\mathcal{F}[m_0] ) = -F^*r.

The method of gradient (or steepest) descent attempts to solve the
optimization problem using the iteration

.. math::

   m_{k+1} = m_k -\beta_k \nabla J[m_k],

where :math:`\beta_k` is a step parameter, best found via a 'line search,'
that is necessary because the gradient provides no notion of scale. With a
sufficiently sized :math:`\beta_k`, this method can recover the true modem
:math:`m` approximately.

.. note::

    Most texts use :math:`\alpha` for the line search parameter, but to avoid
    confusion with the :math:`\alpha` in the CFL condition, we use
    :math:`\beta` instead.

.. topic:: Problem 4.1

    Write a function ``gradient_descent(C0, d, k, config)`` which takes a
    starting model ``C0``, 'measured' data ``d``, and a number of iterations
    ``k`` as arguments and returns the sequence of ``k`` estimates of the true
    model, as well as the values of the objective function :math:`J` at each
    of those points.

    How might you select :math:`\beta_k`? How does :math:`m_k` compare to
    :math:`m`? Why? How many iterations do you need before you start to
    recover your model? The model recovery is not perfect. Explain the errors.

    .. warning::

        Be very careful to update the model correctly, as :math:`\delta m \ne
        c - c_0 = \delta c`! How does :math:`\delta m` relate to :math:`\delta
        c`?

    Plot the convergence of your gradient descent method using a log-scaled plot.


.. topic:: Problem 4.2

    Import the model function ``gauss_model`` from :download:`models.py
    <models.py>`. Use your ``gradient_descent`` function to try to recover the
    new Gaussian model. How do your results change? Explain the errors in the
    recovery.


Bonus Problems
==============

**Bonus Problem 4.3:** Gradient descent is going to converge slowly. How might
you accelerate this process?
