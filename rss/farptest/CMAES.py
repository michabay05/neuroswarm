from typing import Callable

import cma
import numpy as np
from numpy.typing import NDArray


class CMAES:
    def __init__(self,
        fitness: Callable,
        target: float,
        seed: int = 20,
        genome_size: int = 4,
        pop_size: int = 10,
        max_iters: int | None = 10
    ):
        assert callable(fitness), "The fitness function must be callable"
        self.fitness = fitness
        self.x0 = [0.5 for _ in range(genome_size)]
        self.sigma0 = 0.25
        self.pop = pop_size
        self.max_iters = max_iters
        self.target = target
        self.seed = seed
        self.bounds = [[0.0 for _ in range(genome_size)], [1.0 for _ in range(genome_size)]]

        self.all_run_stats = []
        self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, self.options())

    def options(self):
        return {
            "bounds": self.bounds,
            "ftarget": self.target,

            # Force a larger population of 20 per batch
            # "popsize": 20,

            # Scale step sizes for each dimension
            # "CMA_stds": [1.0, 100.0, 0.01],

            # Stop after 500 generations no matter what
            "maxiter": self.max_iters,

            # Make the run perfectly reproducible
            "seed": self.seed,

            # Only print to console every 50 loops
            # "verb_disp": 50
        }

    def evolve(self, n: int, seeds: NDArray[np.int64]):
        while not self.es.stop():
            solutions = self.es.ask()
            stats, succ_rates = self.fitness(n, solutions, seeds)
            self.all_run_stats.extend(stats)
            self.es.tell(solutions, succ_rates)

            self.es.disp()

        self.es.result_pretty()
        return self.es.result.xbest, self.es.result.fbest
