
import numpy as np
import copy

__all__ = ['TemporalLeastSquares']

__docformat__ = "restructuredtext en"


class GradientTest(object):

    """Class for Gradient test"""

    def __init__(self, objective, model_perturbation=None, length_ratio=None):
        """Construct for the GradientTest class
        Considering a function x --> f(x), x \in R^{n}, according to the Taylor
        expansion, we have the following properties given a point x0:

        (1). f(x) = f(x0) + O(|x-x0|)
        (2). f(x) = f(x0)) + g(x0)^{T}(x-x0) + O(|x-x0|^2).

        Therefore, if the gradient g(x0) is correct, we should have:

        (1) |f(x) - f(x0)| = O(|x-x0|)
        (2) |f(x) - f(x0) - g(x0)^{T}(x-x0)| = O(|x-x0|^2).

        To verify this property, we first select a random perturbation direction
        p, and set up a step vector alpha. One simple choice for alpha can be

              alpha = 2 ** [-10, -9, -8, -7, -6, -5, -4, -3, -2 , -1].

        Therefore, if our gradient is correct, the zero order difference

                   df0(alpha) = |f(x0) + alpha * p) - f(x0)|

        should be parallel with the line of alpha in the loglog plot.

        Meanwhile, the first order difference

         df1(alpha) = |f(x0) + alpha * p) - f(x0) - alpha * g(x0)^{T} p|

        should be parallel with the line of alpha^2 in the loglog plot.


        Inputs:
        objective : the objective function. It should belong to the class of
                    objective_function. It should be able to return the objective
                    value and gradient.

        model_perturbation : the model perturbation direction corresponds to the
                             vector of p in the explanation. It should belong to
                             the class of model_parameter

        length_ratio : a vector of the step size corresponds to the alpha in the
                       explanation

        Parameters
        ----------
        model_perturbation : the model perturbation direction corresponds to the
                             vector of p in the explanation. It should belong to
                             the class of model_parameter

        length_ratio : a vector of the step size corresponds to the alpha in the
                       explanation

        objective_value : a vector of objective value f(x0 + alpha * p) corresponding
                          to the elenments in alpha

        zero_order_difference : a vector of the zero order difference df0(alpha)

        first_order_difference : a vector of the first order difference df1(alpha)

        base_model : the base model x0. It should belong to the class of
                     model_parameter

        """

        self.objective_function = objective
        self.solver = objective.solver
        self.use_parallel = objective.use_parallel()
        self.model_perturbation = model_perturbation
        self.length_ratio = length_ratio
        self.objective_value = []
        self.zero_order_difference = []
        self.first_order_difference = []
        self.base_model = []

    def __call__(self, shots):
        aux_info = {'objective_value': (True, None),
                    'residual_norm': (True, None)}
        model_perturbation = self.model_perturbation
        length_ratio = self.length_ratio
        n_ratio = len(length_ratio)

        # Compute the gradient g(x0)
        gradient = self.objective_function.compute_gradient(
            shots, self.base_model, aux_info=aux_info)

        # Compute the objective value f(x0)
        objective_value_original = self.objective_function.evaluate(shots,
                                                                    self.base_model)

        for i in range(0, n_ratio):
            ratio_i = length_ratio[i]
            model_perturbation_i = copy.deepcopy(model_perturbation)
            model = copy.deepcopy(self.base_model)
            model_perturbation_i.data = model_perturbation_i.data * ratio_i
            model.data = self.base_model.data + model_perturbation_i.data

            # Compute the objective value f(x0 + alpha * p)
            fi0 = self.objective_function.evaluate(shots,
                                                   model)

            # Compute f(x0) + alpha * g^{T}p
            fi1 = objective_value_original + \
                np.dot(ratio_i * model_perturbation.data.ravel(), gradient.data.ravel())

            # Compute the zero order difference df0
            diff_f0 = abs(fi0 - objective_value_original)

            # Compute the first order difference df1
            diff_f1 = abs(fi0 - fi1)

            # Store all the values
            self.objective_value.append(fi0)
            self.zero_order_difference.append(diff_f0)
            self.first_order_difference.append(diff_f1)
