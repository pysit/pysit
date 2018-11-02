*****************************************
Part 3: The Linear Problem and Validation
*****************************************

In this exercise you will solve the *linear* forward modeling equations and
verify that the adjoint conditions hold for your code. The definition of the
adjoint of a linear operator states that

.. math::

   \left< F\delta m, d \right> _{\mathcal{D}} = \left< \delta m, F^{*}d
   \right> _{\mathcal{M}},

where :math:`\left< \cdot, \cdot \right> _{\mathcal{D}}` and :math:`\left<
\cdot, \cdot \right> _{\mathcal{M}}` are inner products in the data and model
spaces, respectively. One test to see if your migration operator :math:`F^*`
is working correctly is to test if this relationship holds to high precision
for any pair of :math:`d` and :math:`\delta m`.

Linear Forward Operator
=======================

To implement the adjoint test, you will need to solve the linear modeling
equations, which are derived in the `notes
<http://math.mit.edu/icg/resources/notes325.pdf>`_,

.. math::

   (\frac{1}{c_0(x)^2}\partial_{tt}-\partial_{xx})u_1(x,t) & = -\delta m(x) \partial_{tt}u_0(x,t),  \\
   (\frac{1}{c_0(x)}\partial_t-\partial_x)u_1(0,t) & = 0, \\
   (\frac{1}{c_0(x)}\partial_t+\partial_x)u_1(1,t) & = 0, \\
   u_1(x,t) & = 0 \quad\text{for}\quad t \le 0,

where :math:`u_1` is the Born scattered field and equivalently :math:`F\delta
m = u_1`.

.. topic:: Problem 3.1

    Write a function ``linear_sources(dm, u0s, config)`` that takes an
    arbitrary model perturbation ``dm`` and a time sequence of incident
    wavefields ``u0s`` and generates the linear source wavefields (the
    right-hand-sides of the linear modeling equations). Functions you have
    previously written should be useful.

.. topic:: Problem 3.2

    Use your ``leap_frog`` function to solve for the linear forward wavefield
    :math:`u_1` due to a perturbation :math:`\delta m`. Use this code to write
    a function ``linear_forward_operator(C0, dm, config)`` which returns a
    tuple containing the wavefields and the sampled data.


Adjoint Validation
==================

When sampling is accounted for, the adjoint condition requires that,

.. math::

   \left< \mathbf{S}F\delta m, d \right> _{\mathcal{D}} = \left< \delta m,
   F^{*}\mathbf{S}^*d \right> _{\mathcal{M}},

hold to high precision.

.. topic:: Problem 3.3

    Verify that the adjoint condition is satisfied by your implementation of
    the linear forward modeling and migration operators. Be careful to take
    into account the differences in the model and data inner product. Write a
    function ``adjoint_condition(C0, config)`` that implements this test. How
    accurate is the relationship? It should be accurate to machine precision
    for random values of the data :math:`d` and the model perturbation
    :math:`\delta m`. Be sure that you define :math:`\delta m(0) = \delta m(1)
    = 0`, as it is nonphysical to have sources on an absorbing boundary.

    Consider modifying your ``construct_matrices`` function to accept a
    ``config`` key ``'bc'``, which allows you to toggle between absorbing and
    homogeneous Dirichlet boundaries. This result should still hold.


    .. code:: python

        print "Absorbing BC"
        adjoint_condition(C0, config)

        print "Dirichlet BC"
        config['bc'] = 'dirichlet'
        adjoint_condition(C0, config)



Bonus Problems
==============

**Bonus Problem 3.4:** It is compuationally intractible to compute the
linear forward operator :math:`F` directly. Why? If you wanted to
explicitely compute this operator, how would you do it?
