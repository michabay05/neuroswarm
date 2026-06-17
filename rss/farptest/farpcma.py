import argparse
from pathlib import Path
import datetime as dt
import json
import time

import numpy as np
from numpy.typing import NDArray
from collections import Counter

from swarmsim.util.processing.multicoreprocessing import process_map
from swarmsim.world import config_from_yaml
from swarmsim.world.simulate import main as simulate
from OptimVar import CMAESVarSet
from CMAES import CMAES

cwd = Path(__file__).resolve().parent
# arbitrary linear and angular velocity values; can be changed
V_MAX, W_MAX = 0.3, 0.6
# NOTE: Perfect score implies 100% (0.0) failure rate; negating here to
# turn the success rate maximization problem into a minimization one for CMA-ES
PERFECT_SCORE = -1.0
VAR_CONFIGS = {
    "forward_rate_0": [-V_MAX, V_MAX],
    "turning_rate_0": [-W_MAX, W_MAX],  # Radians / second
    "forward_rate_1": [-V_MAX, V_MAX],
    "turning_rate_2": [-W_MAX, W_MAX],  # Radians / second
}
DECISION_VARS = CMAESVarSet(VAR_CONFIGS)

def gene_to_world(unnorm_genome, seed, blue_n=6):
    return config_from_yaml(
        cwd / "world.yaml", m="ttc", evader="pid",
        blue_n=blue_n, seed=seed, g=unnorm_genome
    )

def fitness_single_OLD(
    config: tuple[NDArray[np.float64], list[int]]
) -> tuple[Counter, float]:
    norm_genome, seeds = config

    unnorm_genome = DECISION_VARS.from_unit_to_scaled(norm_genome)
    success = 0
    stats: list[Counter] = []
    for seed in seeds:
        world_conf = gene_to_world(unnorm_genome, seed)
        world = simulate(world_conf, show_gui=False, start_paused=False)

        # TODO: The index here is hard-coded; make this more robust by not
        # hard-coding the index for the EntityLen metric
        if world.metrics[3] != 0:
            success += 1

        stat = Counter()
        for m in world.metrics:
            stat[m.name] += m.value

        stats.append(stat)

    success_rate = success / len(seeds)
    return sum(stats, start=Counter()), success_rate

def fitness_mp_OLD(norm_genomes: NDArray[np.float64], seeds: list[int]):
    configs = [(norm_genome, seeds) for norm_genome in norm_genomes]
    return process_map(fitness_single, configs)

def fitness_single(config: tuple[NDArray[np.float64], int]):
    norm_genome, seed = config
    assert len(norm_genome.shape) == 1

    unnorm_genome = DECISION_VARS.from_unit_to_scaled(norm_genome)
    world_conf = gene_to_world(unnorm_genome, seed)
    world = simulate(world_conf, show_gui=False, start_paused=False)

    # TODO: The index here is hard-coded; make this more robust by not
    # hard-coding the index for the EntityLen metric
    success = 1 if world.metrics[3].value == 0 else 0

    stat = Counter()

    blue_n = 0
    for agent in world.population:
        if agent.team == "blue":
            blue_n += 1

    stat["blue_n"] = blue_n
    stat["seed"] = int(seed)
    stat["unnorm_genome"] = unnorm_genome
    for m in world.metrics:
        stat[m.name] += m.value

    return stat, success

def fitness_mp(norm_genomes: NDArray[np.float64], seeds: list[int]):
    succ_rates = []
    all_stats = []
    for norm_genome in norm_genomes:
        configs = [(norm_genome, seed) for seed in seeds]
        ret_arr = process_map(fitness_single, configs)
        stats, successes = zip(*ret_arr)

        all_stats.extend(stats)
        # NOTE: Negating here to turn the success rate maximization problem into
        # a minimization one for CMA-ES
        succ_rates.append(-sum(successes) / len(seeds))

    return all_stats, succ_rates

def test_cma(rng_seed=20, trial_seeds_count=10, pop_size=15, max_iters=30):
    trial_seeds = np.random.default_rng(rng_seed).integers(
        0, 2**31, size=trial_seeds_count, dtype=np.int64)

    cmaes = CMAES(
        fitness=fitness_mp, target=PERFECT_SCORE, seed=rng_seed,
        genome_size=4, pop_size=pop_size, max_iters=max_iters
    )
    try:
        _, _ = cmaes.evolve(trial_seeds)
    except KeyboardInterrupt:
        print("Detected <C-c>; stopping now...")
    finally:
        cmaes.es.result_pretty()

        dt_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"results/cmaes_{dt_str}.json", "w") as f:
            json.dump({
                "rng_seed": rng_seed,
                "pop_size": pop_size,
                "max_iters": max_iters,
                "var_configs": VAR_CONFIGS,
                "runs": cmaes.all_run_stats
            }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rs", "--rng_seed", type=int, required=True, help="Seed for RNG",
    )
    parser.add_argument(
        "-t", "--trials", type=int, required=True, help="Number of trials",
    )
    parser.add_argument(
        "-p", "--pop_size", type=int, required=True, help="Population size",
    )
    parser.add_argument(
        "-mi", "--max_iters", type=int, required=True, help="Maximum number of iterations",
    )

    args = parser.parse_args()

    start = time.time()
    test_cma(
        rng_seed=args.rng_seed, trial_seeds_count=args.trials,
        pop_size=args.pop_size, max_iters=args.max_iters
    )
    print(f"Took {time.time() - start} seconds")
