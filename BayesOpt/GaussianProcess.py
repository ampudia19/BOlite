# %%
import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from collections.abc import Callable
from BayesOpt.utils.plot import * 
from scipy.special import gamma, kv

npr.seed(42)
matplotlib.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False
    })
# %%
class GaussianProcess:
    """
    This class implements a Gaussian process, allowing for prior and posterior 
    trajectories to be drawn. Posterior inference is done using either direct
    matrix inversion or the Cholesky approach described in Rasmussen and Williams' 
    (2006) Algorithm 2.1. The class is updated by using the `fit` method, while 
    posterior predictions are obtained using the `predict` method.
    """
    def __init__(self, kernel_func: Callable = None, kernel_kwargs: dict = {}):
        """
        Sets the kernel function and kernel hyperparameters. The class defaults to
        a squared exponential kernel function, but any alternative kernel can be passed
        as long as it takes two inputs on top of its hyperparameters and outputs a scalar.

        :param kernel_func: Kernel function. Defaults to a Squared Exponential Kernel.
        :param kernel_kwargs: Kernel parameters. Defaults to lengthscale 1, signal 
            variance 1, and random noise 0.1.
        """
        self.kernel_func = kernel_func
        self.kernel_kwargs = kernel_kwargs
        self.X = None
        self.K = None
        self.L = None
        self.alpha = None
        self.fitted = False

        if kernel_func is None:
            self.kernel_func = self.squared_exponential_kernel
        if len(kernel_kwargs) == 0:
            self.kernel_kwargs = {"lengthscale": 1., "sigma_f": 1., "sigma_n": 0.1}

    def kernel_matrix(self, X0: np.ndarray, X1: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Computes the Gram matrix given a kernel function and kernel hyperparameters. It
        considers a small addition to the diagonal to guarantee numeric stability
        when performing inference. It additionally allows for the native Kernel result
        of the class not to be overwritten.

        :param X0, X1: The two x values to use as kernel inputs.
        :param update: Boolean that enables many predictions without updating self.K.
        """
        K = [
            self.kernel_func(x[0], x[1], **self.kernel_kwargs) 
            for x in product(X0, X1)
        ]
        K = np.reshape(K, (X1.shape[0], X1.shape[0]))
        np.fill_diagonal(K, K.diagonal() + 1e-10) # numerical stability
        if "sigma_n" in self.kernel_kwargs.keys():
            K = K + self.kernel_kwargs["sigma_n"]**2 * np.identity(K.shape[0])
        if update:
            self.K = K
        else:
            return K

    @staticmethod
    def squared_exponential_kernel(
        xi: float, xj: float, lengthscale: float, sigma_f: float, sigma_n: float
    ) -> float:
        """
        Default Kernel method.

        :param xi, xj: The two x values to use as kernel inputs.
        :param lengthscale: Lengthscale parameter. Defaults to 1.
        :param sigma_f: Signal variance parameter. Defaults to 1.
        :param sigma_n: Noise variance parameter. Defaults to 0.1.

        :return: Output k(xi,xj) given hyperparameters.
        """
        xi, xj = xi[np.newaxis], xj[np.newaxis]
        return (
            sigma_f**2 * 
            np.exp(-(1/(2*lengthscale**2)) * np.sum((xi - xj)**2)) +
            sigma_n**2 * np.isclose(xi, xj, rtol=1e-5).all()
        )
    
    @staticmethod
    def exponential_kernel(
        xi: float, xj: float, lengthscale: float, sigma_f: float, sigma_n: float
    ) -> float:
        """
        Exponential Kernel method.

        :param xi, xj: The two x values to use as kernel inputs.
        :param lengthscale: Lengthscale parameter. Defaults to 1.
        :param sigma_f: Signal variance parameter. Defaults to 1.
        :param sigma_n: Noise variance parameter. Defaults to 0.1.

        :return: Output k(xi,xj) given hyperparameters.
        """
        xi, xj = xi[np.newaxis], xj[np.newaxis]
        return (
            sigma_f**2 * 
            np.exp(-(1/lengthscale) * np.sum(np.abs(xi - xj))) +
            sigma_n**2 * np.isclose(xi, xj, rtol=1e-5).all()
        )
    
    @staticmethod
    def matern_kernel(
        # xi: float, xj: float, lengthscale: float, sigma_f: float, sigma_n: float
        xi: float, xj: float, lengthscale: float, smoothness: float, sigma_n: float
    ) -> float:
        """
        Matern Kernel method.

        :param xi, xj: The two x values to use as kernel inputs.
        :param lengthscale: Lengthscale parameter. Defaults to 1.
        :param smoothness: Smoothness parameter. Defaults to 1.
        :param sigma_n: Noise variance parameter. Defaults to 0.1.

        :return: Output k(xi,xj) given hyperparameters.
        """
        # Compute distance between points
        d = np.sum(np.abs(xi - xj) / lengthscale)

        # Compute kernel value
        if np.isclose(d, 0):
            k = 1 + sigma_n**2
        else:
            k = ((2**(1 - smoothness) / gamma(smoothness)) *
                d**smoothness *
                kv(smoothness, d) +
                sigma_n**2 * np.isclose(xi, xj).all())

        return k


    @staticmethod
    def periodic_kernel(
        xi: float, xj: float, lengthscale: float, sigma_f: float, sigma_n: float
    ) -> float:
        """
        Periodic Kernel method.

        :param xi, xj: The two x values to use as kernel inputs.
        :param lengthscale: Lengthscale parameter. Defaults to 1.
        :param sigma_f: Signal variance parameter. Defaults to 1.
        :param sigma_n: Noise variance parameter. Defaults to 0.1.

        :return: Output k(xi,xj) given hyperparameters.
        """
        xi, xj = xi[np.newaxis], xj[np.newaxis]
        return (
            sigma_f**2 * 
            np.exp(-2 * np.sin(np.pi * np.sum(np.abs(xi - xj)) / lengthscale)**2) +
            sigma_n**2 * np.isclose(xi, xj, rtol=1e-5).all()
        )

    def get_prior_trajectories(
        self, X: np.ndarray, num_trajectories: int
    ) -> tuple([np.ndarray, np.ndarray]):
        """
        Computes prior trajectories from a Multivariate distribution with
        N-length zero vector mean and a Kernel matrix of dimensions N x N,
        where N is the length of the input vector X. 

        :param X: Input vector X. Method accepts single and multi-dimensional inputs.
        :param num_trajectories: Number of trajectories drawn from prior distribution.

        :return ystar, vstar: Returns point estimates and variance measures at every X
            point.
        """
        X = self._X_shape_check(X)
        ystar = np.zeros((X.shape[0], num_trajectories))
        mu = np.zeros((X.shape[0],))
        self.kernel_matrix(X, X)
        for trajectory in range(num_trajectories):
            ystar[:, trajectory] = npr.multivariate_normal(mu, self.K)
        return ystar, np.diag(self.K)

    def get_posterior_trajectories(
        self, X: np.ndarray, num_trajectories: int, cholesky: bool
    ) -> tuple([np.ndarray, np.ndarray]):
        """
        Computes posterior trajectories from a fitted instance of a Gaussian Process. 
        Both direct matrix inversion or Cholesky decomposition approaches are supported.

        :param X: Input points at which to draw trajectory results.
        :param num_trajectories: Number of trajectories drawn from posterior distribution.
        :param cholesky: Boolean to use Cholesky decomposition to compute posterior Gram
            matrix. The time complexity is $O(n^2)$. If set to False, matrix inversion
            is instead used, at $O(n^3)$ complexity.
        """
        assert self.fitted is True, (
            "Gaussian Process has not been fit. A fitted instance of GP is "
            "necessary to make predictions from the posterior. Use method "
            "`get_trajectories` instead for predictions using the prior."
        )
        X = self._X_shape_check(X)
        ystar = np.zeros((X.shape[0], num_trajectories))
        mu, Kpost = self.get_posterior(X, cholesky)
        for trajectory in range(num_trajectories):
            if len(mu.shape) == 2:
                mu = np.squeeze(mu)
            ystar[:, trajectory] = npr.multivariate_normal(mu, Kpost)
        return ystar, np.diag(Kpost)
        
    def get_posterior(
        self, Xstars: np.ndarray, cholesky: bool = True
    ) -> tuple([np.ndarray, np.ndarray]):
        """
        Returns mean and covariance from the existing X matrix as well as new $X^*$ observations.
        Inference is performed either using direct matrix inversion or the Cholesky Algorithm.
        Note that running predict does not overwrite the existing X matrix. Call `fit` method
        with additional data to update the evaluated set of input points.

        :param Xstars: New input values to draw posterior values on.
        :param cholesky: Boolean that sets behavior of inference approach.

        :return mu: Point estimates (means) of the posterior distribution given X.
        :return Kpost: Posterior Kernel matrix.
        """
        Xstars = self._X_shape_check(Xstars)
        newX = np.vstack((self.X, Xstars))
        K = self.kernel_matrix(newX, newX, update=False) # expanded K
        if not cholesky:
            mu = K[-Xstars.shape[0]:, :-Xstars.shape[0]] @ np.linalg.inv(self.K) @ self.y
            Kpost = (
                K[-Xstars.shape[0]:, -Xstars.shape[0]:] - 
                (
                    K[-Xstars.shape[0]:, :-Xstars.shape[0]] @ # K(x*,x)
                    np.linalg.inv(K[:-Xstars.shape[0], :-Xstars.shape[0]]) @ # K(x,x)
                    K[-Xstars.shape[0]:, :-Xstars.shape[0]].T # K(x,x*)
                )
            )
        elif cholesky:
            self.cholesky_parameters(K=self.K, update=True)
            self.nu = np.linalg.solve(self.L, K[-Xstars.shape[0]:, :-Xstars.shape[0]].T) # v = L \ K(x,x*)
            mu = K[-Xstars.shape[0]:, :-Xstars.shape[0]] @ self.alpha # K(x*,x) * alpha
            Kpost = K[-Xstars.shape[0]:, -Xstars.shape[0]:] - np.dot(self.nu.T, self.nu) # K(x*,x*) - vTv
        return mu, Kpost

    def cholesky_parameters(self, K, update=True):
        """
        Cholesky decomposition and parameter definition using Algorithm 2.1.

        :param K: Kernel Gram matrix, can be either the existing K matrix or a K matrix
            with new values.

        :return L: Cholesky lower triangular matrix
        :return alpha: Equivalent to (K+sigma_n**2)^{-1} * y)
        """
        L = np.linalg.cholesky(K)
        np.fill_diagonal(L, L.diagonal() + 1e-20) # numerical stability
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        if update:
            self.L, self.alpha = L, alpha
        elif update is False:
            return L, alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Defines the Gram matrix given a kernel function. It also stores current best
        prediction for BO purposes.

        :param X: Input points to define the fitted K matrix.
        :param y: Output values for the given X values.
        """
        X, y = self._X_shape_check(X), self._X_shape_check(y)
        self.y = y
        self.X = X
        self.kernel_matrix(X, X)
        self.fitted = True

    def predict(
        self, Xstars: np.ndarray, cholesky: bool = True
    ) -> tuple([np.ndarray, np.ndarray]):
        """
        Predict method returns posterior means and variances for an array of new values.

        :param Xstars: New input values to predict on.
        :param cholesky: Boolean that sets behavior of inference approach.

        :return mu: Point estimates for new input values.
        : return diag(Kpost): Posterior variance estimates for new input values.  
        """
        Xstars = self._X_shape_check(Xstars)
        mu, Kpost = self.get_posterior(Xstars, cholesky)
        mu, Kpost = mu[-Xstars.shape[0]:], Kpost[-Xstars.shape[0]:, -Xstars.shape[0]:]
        return mu, np.diag(Kpost)

    def _single_predict(self, Xstar: float) -> tuple([np.ndarray, np.ndarray]):
        """
        Legacy function, original approach was to map many individual predictions. Left
        here in case it was somehow useful.
        """
        newX = self._concat(self.X, Xstar)
        self.kernel_matrix(newX, newX)
        self.nu = np.linalg.solve(self.L, self.K[-1,:-1])
        ystar = np.squeeze(self.K[-1,:-1] @ self.alpha)
        vstar = self.K[-1,-1] - self.K[-1,:-1] @ self.nu
        return ystar, vstar

    def _X_shape_check(self, X):
        if isinstance(X, float):
            X = np.array([X])
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        return X