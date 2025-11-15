"""
turbo.py

Author: natelgrw
Last Edited: 11/15/2025

Trust region bayesian optimization (TuRBO) implementation to
explore the parameter space of a gradient.
"""

import numpy as np
import warnings
from typing import Callable, Tuple, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize

# suppress convergence warnings from GP hyperparameter optimization
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.gaussian_process')


# ===== TuRBO Class ===== #


class TuRBO:
    """
    Trust region bayesian optimization class.
    
    Efficiently optimizes expensive black-box functions and
    uses local trust regions and restarts to balance
    exploration and exploitation.
    """
    
    def __init__(self,
                 objective_fn: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 n_init: int = 10,
                 max_evals: int = 100,
                 batch_size: int = 1,
                 trust_region_init: float = 0.8,
                 trust_region_min: float = 0.1,
                 verbose: bool = True):
        """
        Initializes the TuRBO optimizer.
        
        Args:
            objective_fn: function to maximize (takes array, returns scalar)
            dim: dimensionality of search space
            bounds: tuple of (lower_bounds, upper_bounds)
            n_init: number of initial random samples
            max_evals: maximum number of evaluations
            batch_size: number of points to evaluate in parallel
            trust_region_init: initial trust region size (fraction of bounds)
            trust_region_min: minimum trust region size
            verbose: print progress
        """
        self.objective_fn = objective_fn
        self.dim = dim
        self.lower_bounds = bounds[0]
        self.upper_bounds = bounds[1]
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        
        # trust region parameters
        self.tr_length = trust_region_init
        self.tr_length_init = trust_region_init
        self.tr_length_min = trust_region_min
        self.tr_success_counter = 0
        self.tr_fail_counter = 0
        self.success_tol = 3
        self.fail_tol = 5
        
        # data storage
        self.X = np.zeros((0, dim))
        self.y = np.zeros(0)
        self.n_evals = 0
        
        # gp model
        self.gp = None
        
    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes x to [0, 1].
        """
        return (x - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
    
    def _unnormalize_x(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Unnormalizes x from [0, 1] to original bounds.
        """
        return x_norm * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
    
    def _sample_initial(self, n: int) -> np.ndarray:
        """
        Samples n points uniformly from bounds.
        """
        x_norm = np.random.uniform(0, 1, size=(n, self.dim))
        return self._unnormalize_x(x_norm)
    
    def _fit_gp(self):
        """
        Fits a gaussian process to current data.
        """
        if len(self.X) < 2:
            return
        
        # normalize data
        X_norm = self._normalize_x(self.X)
        y_mean = np.mean(self.y)
        y_std = np.std(self.y) + 1e-6
        y_norm = (self.y - y_mean) / y_std
        
        # fit gp with matern kernel
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.dim) * 0.5,
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=False
        )
        
        self.gp.fit(X_norm, y_norm)
        self.gp._y_mean = y_mean
        self.gp._y_std = y_std
    
    def _acquisition_function(self, x_norm: np.ndarray) -> float:
        """
        Upper confidence bound acquisition function.
        
        Args:
            x_norm: normalized input in [0, 1]
            
        Returns:
            Negative acquisition value for minimization.
        """
        if self.gp is None:
            return 0.0
        
        x_norm = x_norm.reshape(1, -1)
        mu, sigma = self.gp.predict(x_norm, return_std=True)
        
        # unnormalize predictions
        mu = mu * self.gp._y_std + self.gp._y_mean
        sigma = sigma * self.gp._y_std
        
        beta = 2.0
        ucb = mu + beta * sigma
        
        return -ucb[0]
    
    def _select_next_point(self, center: np.ndarray) -> np.ndarray:
        """
        Selects the next point to evaluate using the acquisition function
        within the trust region.
        
        Args:
            center: center of trust region (normalized)
            
        Returns:
            Unnormalized next point to evaluate.
        """
        if self.gp is None:
            x_norm = center + np.random.uniform(-self.tr_length/2, self.tr_length/2, self.dim)
            x_norm = np.clip(x_norm, 0, 1)
            return self._unnormalize_x(x_norm)
        
        # optimize acquisition function within trust region
        bounds_norm = []
        for i in range(self.dim):
            lb = max(0.0, center[i] - self.tr_length / 2)
            ub = min(1.0, center[i] + self.tr_length / 2)
            bounds_norm.append((lb, ub))
        
        # multi-start optimization
        best_x = None
        best_acq = float('inf')
        
        for _ in range(10):
            x0 = np.array([np.random.uniform(lb, ub) for lb, ub in bounds_norm])
            
            result = minimize(
                self._acquisition_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds_norm
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return self._unnormalize_x(best_x)
    
    def _update_trust_region(self, y_new: float):
        """
        Update trust region based on improvement.
        """
        if len(self.y) < 2:
            return
        
        y_best_prev = np.max(self.y[:-1])
        
        if y_new > y_best_prev:
            self.tr_success_counter += 1
            self.tr_fail_counter = 0
            
            if self.tr_success_counter >= self.success_tol:
                self.tr_length = min(2 * self.tr_length, 1.0)
                self.tr_success_counter = 0
                if self.verbose:
                    print(f"  trust region expanded to {self.tr_length:.3f}")
        else:
            self.tr_fail_counter += 1
            self.tr_success_counter = 0
            
            if self.tr_fail_counter >= self.fail_tol:
                self.tr_length = max(0.5 * self.tr_length, self.tr_length_min)
                self.tr_fail_counter = 0
                if self.verbose:
                    print(f"  trust region shrunk to {self.tr_length:.3f}")
        
        # restart if trust region too small
        if self.tr_length <= self.tr_length_min and self.n_evals < self.max_evals:
            if self.verbose:
                print(f"  trust region restart at eval {self.n_evals}")
            self.tr_length = self.tr_length_init
            self.tr_success_counter = 0
            self.tr_fail_counter = 0
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Runs the optimization.
            
        Returns:
            best_x: best parameters found
            best_y: best objective value
        """
        # initial random sampling
        if self.verbose:
            print(f"initial sampling: {self.n_init} points")
        
        X_init = self._sample_initial(self.n_init)
        for x in X_init:
            y = self.objective_fn(x)
            self.X = np.vstack([self.X, x])
            self.y = np.append(self.y, y)
            self.n_evals += 1
            
            if self.verbose and (self.n_evals % 10 == 0 or self.n_evals == self.n_init):
                print(f"  eval {self.n_evals}/{self.max_evals}: y = {y:.4f}")
        
        # bayesian optimization loop
        while self.n_evals < self.max_evals:
            self._fit_gp()
            
            best_idx = np.argmax(self.y)
            center = self._normalize_x(self.X[best_idx])
            
            x_next = self._select_next_point(center)
            
            y_next = self.objective_fn(x_next)
            self.X = np.vstack([self.X, x_next])
            self.y = np.append(self.y, y_next)
            self.n_evals += 1
            
            if self.verbose and (self.n_evals % 10 == 0 or self.n_evals == self.max_evals):
                best_y = np.max(self.y)
                print(f"  eval {self.n_evals}/{self.max_evals}: y = {y_next:.4f} (best = {best_y:.4f})")
            
            self._update_trust_region(y_next)
        
        # return best found
        best_idx = np.argmax(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]
        
        if self.verbose:
            print(f"\noptimization complete")
            print(f"best objective value: {best_y:.4f}")
            print(f"best parameters: {best_x}")
        
        return best_x, best_y

