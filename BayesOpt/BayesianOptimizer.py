# %%
import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize, linprog
from itertools import product
from collections.abc import Callable
from tqdm import tqdm

import scipy.stats as scis 
from BOlite.BayesOpt.GaussianProcess import GaussianProcess
from BOlite.BayesOpt.utils.plot import * 
from BOlite.BayesOpt.utils.logging import *

npr.seed(42)
matplotlib.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False
    })

# %%

class BayesianOptimizer:
    """
    This class implements a classic Bayesian Optimization algorithm, using a Gaussian Process
    as surrogate model and one of three acquisition functions to evaluate new query points:
    Expected Improvement, Probability of Improvement and Upper Confidence bound. New queries 
    are considered over a Sobol-sampled grid of possible points, and hyperparameters of the
    Gaussian Process are iteratively optimized using A Nelder-Mead derivative-free algorithm.

    :param surrogate: Surrogate function, ought to be a Gaussian Process instance.
    :param initial_queries: D-dimensional array of N initial queries. [N, D]
    :param initial_observations: Array of N initial observations. [N, 1]
    :param oracle: True callable function.
    :param box: Tuple of D-dimensional list of bounds. These are used to sample within the span
        limits of the function input space.
    """
    def __init__(
        self, 
        surrogate: GaussianProcess, 
        initial_queries: np.ndarray,
        initial_observations: np.ndarray, 
        oracle: Callable, 
        box: list
    ):
        self.surrogate = surrogate
        self.surrogate_kwargs = self.surrogate.kernel_kwargs
        self.oracle = oracle
        self.box = box
        self.X = initial_queries
        self.y = initial_observations
        self.y_min = np.min(self.y)
        self.surrogate.fit(self.X, self.y)
        self.ACQ_FNS = {
            "ei": self._get_expected_improvement,
            "poi": self._get_probability_improvement,
            "ucb": self._get_upper_confidence_bound
        }

    def loglik(self, y: np.ndarray, K: np.ndarray, cholesky: bool = True):
        """
        Gaussian log-likelihood of the observed points and their corresponding
        input queries. Standard matrix inversion or Choleskly decomposition approaches
        are both implemented, and the Gram matrix K includes possible noise variance `sigma_n`.

        :param y: True function observations.
        :param K: Gram matrix built using the surrogate's kernel. Uses the function's
            input queries.
        :param cholesky: Boolean to use Cholesky decomposition and avoid O(N^3) 
            matrix inversion. Defaults to True.

        :return ll: Log-likelihood value.
        """
        if not cholesky:
            ll = (
                -1/2 * y.T @ np.linalg.inv(K) @ y -
                1/2 * np.log(np.linalg.det(K)) - 
                K.shape[0]/2 * np.log(2*np.pi)
            )
        elif cholesky:
            ll = (
                -1/2 * y.T @ self.alpha_ -
                np.sum(np.log(self.L_.diagonal())) - 
                K.shape[0]/2 * np.log(2*np.pi)
            )
        return ll

    def margloglikelihood(self, parameters: list):
        """
        Marginal log likelihood over the function values f. Function is meant to be used
        with `scipy.optimize` algorithms, which inputs a list of parameters to optimize
        over. Function internally constructs the `kernel_kwargs` dictionary, builds the 
        kernel Gram matrix and computes the relevant Cholesky decomposition parameters.

        :param parameters: List of parameters. If expontential kernel is used, this defaults
            to `lengthscale`, `sigma_n` and `sigma_f`.

        :return -ll: Returns the negative log likelihood using the Cholesky decomposition
            approach.
        """
        self.surrogate.kernel_kwargs = dict(zip(self.surrogate_kwargs.keys(), parameters))
        K = self.surrogate.kernel_matrix(self.X, self.X, update=False)
        self.L_, self.alpha_ = self.surrogate.cholesky_parameters(K, update=False)
        return - self.loglik(self.y, K, cholesky=True)

    def optimize_surrogate(
        self, 
        method: str = "Nelder-Mead", 
        maxfev: float = 1e5, 
        tol: float = 1e-2
    ):
        """
        Optimization step of the surrogate function, updates Kernel hyperparameters
        `lengthscale`, `sigma_n`, `sigma_f` when new points are queried in the BO schedule.
        The default approach uses the Nelder-Mead simplex method, a gradient-free algorithm 
        that traverses the parameter space in search of a minimum. Alternative methods need be
        derivative free.

        The function automatically updates the class internal kernel arguments and re-fits
        the Gaussian process using the new optimal parameters.

        :param method: Method to use in the optimization algorithm. See `scipy.optimize`
            documentation for derivative-free options.
        :param maxfev: Maximum number of function evaluations, used to control runtime at
            the cost of precision in finding optimal parameters.
        :param tol: Tolerance parameter of changes in iterative optimal solution which 
            may terminate the process. See `scipy.optimize` documentation for details.
        """
        self.surrogate.fit(self.X, self.y)
        min = minimize(
            self.margloglikelihood, 
            x0=list(self.surrogate_kwargs.values()), 
            method=method, 
            options={"maxfev": maxfev, "fatol": tol, "xatol": tol}
        )
        self.surrogate_kwargs = dict(zip(self.surrogate_kwargs.keys(), min.x))
        self.surrogate.fit(self.X, self.y)            

    def query_new_point(self, acquisition: str, num_points: int, **kwargs):
        """
        Uses the surrogate optimization schedule to update optimal Gaussian process 
        hyperparameters using the existing array of queries and observations. It also
        uses Sobol sampling to draw new possible query point candidates, and iteratively
        evaluates these using a choice of acquisition function. 

        :param acquisition: Choice of acquisition function. Valid choices 
            are `ei`, `poi`, `ucb`.
        :param num_points: Number of points to evaluate as candidates. To use Sobol sampling
            and guarantee balance, a number factorizable with base 2 ought to be chosen.

        :return results, xqueries: Acquisition function outputs and candidate queries.
        """
        self.optimize_surrogate(method="Nelder-Mead", maxfev=50, tol=1e-2)
        xqueries = np.expand_dims(self._sobol_sample(num_points, self.box), axis=1)
        results = np.array([acquisition(x, **kwargs) for x in xqueries]) # array operations worked slower?
        return results, xqueries

    def optimize(self, num_steps, num_points, acquisition_fn, **kwargs):
        """
        Optimization function for the Bayesian Optimization schedule. The function considers
        a total number of steps to iterate over, a number of points to evaluate at each step
        and the choice of acquisition function to use thorought. Additional arguments can be
        employed to control parameters of the acquisition function.

        In each step, the algorithm considers `num_points` candidates, and selects the point
        that according to the acqusition function may be a possible function minimum. Internal
        arrays of query points and observations are then updated, as well as the global minimum
        found.

        :param num_steps: Number of steps to consider in the BO schedule.
        :param num_points: Number of points per step to evaluate on.
        :param acquisition: Acquisition function. Valid choices are `ei`, `poi`, `ucb`.

        :return x_min, y_min: Best query point and corresponding function evaluation output.
        """
        assert acquisition_fn in ["ei", "poi", "ucb"], (
            "Invalid choice of Acquisition function. Valid choices are `ei`, `poi`, `ucb`."
        )
        acquisition = self.ACQ_FNS[acquisition_fn]
        for _ in tqdm(range(num_steps)):
            acqout, xqueries = self.query_new_point(acquisition, num_points=num_points, **kwargs)
            query_point = xqueries[np.argmin(acqout)]
            yobs = self.oracle(*query_point.T)
            self.X = np.vstack((self.X, query_point))
            self.y = np.hstack((self.y, yobs))
            self.y_min = np.min(self.y)
        return self.X[np.argmin(self.y)], self.y_min

    def _get_expected_improvement(self, Xstar, eps=0) -> float:
        """
        Expected Improvement acquisition function. The method uses posterior predictions
        on new datapoints and compares the point estimates to the current best output.

        :param Xstar: New input values to evaluate.
        :param eps: Hyperparameter that encourages exploration (if higher) or 
            exploitation (if lower).
        
        :return EI: Expected Improvement estimate for the given input value.
        """
        assert self.surrogate.fitted is True, "Not true"
        pred, var = self.surrogate.predict(Xstar, cholesky=True)
        impr = (pred - self.y_min - eps) / var**0.5
        return (
            (pred - self.y_min - eps) * scis.norm.cdf(impr) + 
            var**0.5 * scis.norm.pdf(impr) 
        )

    def _get_probability_improvement(self, Xstar, eps=0) -> float:
        """
        Probability of Improvement acquisition function. The method uses posterior 
        predictions on new datapoints and compares the point estimates to the current 
        best output. Unlike in Expected Improvement, this approach ignores the size of
        the expected improvement when making a decision.

        :param Xstar: New input values to evaluate.
        :param eps: Hyperparameter that encourages exploration (if higher) or 
            exploitation (if lower).
        
        :return PoI: Probability of Improvement estimate for the given input value.
        """
        assert self.surrogate.fitted is True, "Not true"
        pred, var = self.surrogate.predict(Xstar, cholesky=True)
        return scis.norm.cdf((pred - self.y_min - eps) / var**0.5)

    def _get_upper_confidence_bound(self, Xstar, eps=1) -> float:
        """
        Upper Confidence Bound acquisition function. The method uses posterior
        estimates of the mean and variance to select the new query based on the
        relative potential to yield a new optimum.

        :param Xstar: New input values to evaluate.
        :param eps: Hyperparameter that encourages exploration (if higher than 1) or
            exploitation (if lower than 1).
        
        :return GP-UCB: UCB estimate for the given input value.  
        """
        assert self.surrogate.fitted is True, "Not true"
        pred, var = self.surrogate.predict(Xstar, cholesky=True)
        return pred - eps*var**0.5

    def _sobol_sample(self, num_draws, box):
        sampler = scis.qmc.Sobol(d=2, scramble=True)
        return box[0] + (box[1] - box[0]) * sampler.random(n=num_draws)



# %%
if __name__ == "__main__":
    #----------% 3D visualization routine %----------#
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # True minimum output (at three different input coordinate sets)
    TRUEMIN = 0.397887

    # Branin oracle function
    def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
        return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s

    # Sobol sample, to set the initial combination of points.
    def sobol_sample(num_draws, box):
        sampler = scis.qmc.Sobol(d=2, scramble=True)
        return box[0] + (box[1] - box[0]) * sampler.random(n=num_draws)

    # Number of initial points, number of BO steps
    n_init = 4
    n_steps = 20

    # Input space box and initial queries + observations
    box = np.array([[-5, 0], [10, 15]])
    xinit = sobol_sample(n_init, box)
    yinit = branin(xinit[:, 0], xinit[:, 1])

    # Instance of Bayesian Optimization. Test on smaller number of points + iters
    bo = BayesianOptimizer(
        surrogate=GaussianProcess(), 
        initial_queries=xinit, 
        initial_observations=yinit,
        oracle=branin,
        box=box
    )
    # Optimization schedule
    optx, opty = bo.optimize(num_steps=n_steps, num_points=2**5, acquisition_fn="ei", eps=-8) 

    print(f"Min. query point: {optx}")
    print(f"Min. observation: {opty}")
    print(f"True minimum: {TRUEMIN}")

    # 3D figure of exploration, brighter colors reflect more recent queries
    fig = plot_function(branin, box[0], box[1], 200, title="me", xlabel="this", ylabel="that")

    fig = add_points(
        x=bo.X[:, 0],
        y=bo.X[:, 1],
        z=bo.y,
        num_init=n_init,
        fig=fig
    )
    fig.write_html("figures/exampleBOschedule.html")
    fig.show()

    #----------% Main routine %----------#
    # Setup params
    num_points = 2**5
    num_steps = 100
    num_init = 5
    xinit = sobol_sample(num_init, box)
    yinit = branin(xinit[:, 0], xinit[:, 1])

    fig_configs = [
        {"acquisition_fn": "ei", "eps": -0.5},
        {"acquisition_fn": "ei", "eps": 0},
        {"acquisition_fn": "ei", "eps": 0.5},
        {"acquisition_fn": "poi", "eps": -0.5},
        {"acquisition_fn": "poi", "eps": 0},
        {"acquisition_fn": "poi", "eps": 0.5},
        {"acquisition_fn": "ucb", "eps": 0.5},
        {"acquisition_fn": "ucb", "eps": 1},
        {"acquisition_fn": "ucb", "eps": 1.5}
    ]

    for config in fig_configs:
        name = "_".join(f'{k}_{v}' for k,v in config.items()) + ".pkl"

        # bo instance
        bo = BayesianOptimizer(
            surrogate=GaussianProcess(), 
            initial_queries=xinit, 
            initial_observations=yinit,
            oracle=branin,
            box=box
        )

        # BO schedule
        if not check_log("logs", name):
            optx, opty = bo.optimize(num_steps=num_steps, num_points=num_points, **config)
            config["X"], config["y"] = bo.X, bo.y
            config["xmin"], config["ymin"] = optx, opty
            create_log(
                dirpath="logs", 
                filename=name, 
                outputs=config
            )
        else:
            print(f"[FILE]: {name} already exists. Skipped BO schedule.")


    output_dict = {"ei":{}, "poi":{}, "ucb":{}}
    outputs = [file for file in os.listdir("logs") if ".pkl" in file]
    for output in outputs:
        with open(os.path.join("logs", output), "rb") as file:
            tmp = pickle.load(file)
            mse = np.power(tmp["y"] - TRUEMIN, 2)
        output_dict[tmp["acquisition_fn"]][tmp["eps"]] = mse
        
    #----------% Requested Figure %----------#
    main_figure(output_dict)

    """
    The plot shows more exploration early in the BO schedule on higher values of epsilon. 
    Expected Improvement quickly exploits in the vecinity of the best values found in prior
    steps, while Probability of Improvement explores more frequently and with much larger
    average regret results. UCB explores little even on changing parameters of epsilon, and while
    it tallies a higher regret early on, it quickly starts to exploit around the found minimum.
    """
