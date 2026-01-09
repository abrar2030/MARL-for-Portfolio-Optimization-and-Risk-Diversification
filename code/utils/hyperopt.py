"""Hyperparameter Optimization using Optuna"""

import optuna
from typing import Dict, Callable, Any
import warnings

warnings.filterwarnings("ignore")


class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 50, direction: str = "maximize"):
        self.n_trials = n_trials
        self.direction = direction
        self.study = None

    def optimize(self, objective_func: Callable, param_space: Dict[str, Any]) -> Dict:
        """Run Optuna optimization"""
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective_func, n_trials=self.n_trials)
        return self.study.best_params

    def get_best_params(self) -> Dict:
        """Get best parameters"""
        if self.study is None:
            return {}
        return self.study.best_params
