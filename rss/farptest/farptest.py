import copy
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map
from matplotlib import pyplot as plt

from swarmsim.config import register_dictlike_type, register_agent_type
from swarmsim.agent.MazeAgent import MazeAgentConfig
from swarmsim.world.spawners.DonutSpawner import DonutAgentSpawner
from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber
from swarmsim.world.simulate import main as simulator
from swarmsim.world import config_from_yaml

from CMAES import CMAES
from OptimVar import CMAESVarSet

cwd = Path(__file__).resolve().parent
config = config_from_yaml(
    cwd / "world.yaml",
    m='ttc',
    evader='pid',
    g=[0.2, 0.2, 0.2, 0]
)

# NOTE: Max vel found   - 0.35 m/s
# NOTE: Max omega found - 143.292 deg/s
V_MAX, W_MAX = 0.3, np.deg2rad(90)
PERFECT_SCORE = 0
SCALE = 1
DECISION_VARS = CMAESVarSet(
    {
        "forward_rate_0": [-V_MAX, V_MAX],
        "turning_rate_0": [-W_MAX, W_MAX],  # Radians / second
        "forward_rate_1": [-V_MAX, V_MAX],
        "turning_rate_2": [-W_MAX, W_MAX],  # Radians / second
    }
)


# gui = TennlabGUI(x=0, y=0, h=0, w=300)
# gui.position = "sidebar_right"

def test_single(config):
    tempconfig = copy.deepcopy(config)
    # config.seed += i
    stats = Counter()
    world = simulator(
        world_config=tempconfig,
        subscribers=[],
        # gui=gui,
        show_gui=False,
        start_paused=False,
        framerate_limit=20,
    )  # run simulator
    for m in world.metrics:
        stats[m.name] += m.value
    out = world.metrics[0].value
    return stats, out


def test_mp(samples=100, n_range=None):
    seeds = np.random.default_rng(config.seed).integers(0, 2**31, size=samples)
    results = []
    for n in n_range or range(1, 10):
        configs = []
        for i in range(samples):
            tempconfig = copy.deepcopy(config)
            tempconfig.seed = seeds[i]
            tempconfig.spawners[0]['n'] = n
            configs.append(tempconfig)
        ret_arr = process_map(test_single, configs)
        stats, ttcs = zip(*ret_arr)
        print('n: ', n, sum(stats, Counter()))
        for stat in stats:
            results.append({
                'n': n,
                **stat
            })
    # ttcs = np.array(ttcs)
    # sns.histplot(ttcs)
    # plt.show()
    print(results)
    return results


def test_seq(samples=10):
    seeds = np.random.default_rng(config.seed).integers(0, 2**31, size=samples)
    results = []
    configs = []
    n = 6
    for i in range(samples):
        tempconfig = copy.deepcopy(config)
        tempconfig.seed = seeds[i]
        tempconfig.spawners[0]['n'] = n
        configs.append(tempconfig)
    for i, cfg in enumerate(configs):
        print(i)
        ret_arr = test_single(cfg)
    stats, ttcs = zip(*ret_arr)
    print('n: ', n, sum(stats, Counter()))
    for stat in stats:
        results.append({
            'n': n,
            **stat
        })
    # ttcs = np.array(ttcs)
    # sns.histplot(ttcs)
    # plt.show()
    print(results)

    return results


def test_grid(samples=100):
    seeds = np.random.default_rng(getattr(config, "seed", None)).integers(0, 2**31, size=samples)
    results = []
    x, y = np.meshgrid(
        np.linspace(0.0, 0.3, 12),
        np.linspace(0.0, 0.6, 7),
    )
    configs = []
    n = 6
    for v, w in zip(x.flatten(), y.flatten()):
        for i in range(samples):
            tempconfig = copy.deepcopy(config)
            tempconfig.seed = seeds[i]
            tempconfig.spawners[0]['n'] = n
            controller = tempconfig.spawners[0]['agent']['controller']
            controller['a'] = [v, w]
            controller['b'] = [v, -w]
            configs.append(tempconfig)
    ret_arr = process_map(test_single, configs)
    stats, ttcs = zip(*ret_arr)
    print('n: ', n, sum(stats, Counter()))
    for stat, cfg in zip(stats, configs):
        results.append({
            'n': n,
            'v': cfg.spawners[0]['agent']['controller']['a'][0],
            'w': cfg.spawners[0]['agent']['controller']['a'][1],
            **stat
        })
    # ttcs = np.array(ttcs)
    # sns.histplot(ttcs)
    # plt.show()
    print(results)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('grid.csv')
    return results


def run():
    world = simulator(
        world_config=config,
        subscribers=[],
        # gui=gui,
        show_gui=True,
        start_paused=True,
        framerate_limit=20,
    )  # run simulator
    for m in world.metrics:
        print(f"{m.name}: {m.current_value}")
    return world



def get_world_generator(n=6, seed=2023):
    def gene_to_world(genome, hash_val=None):
        world_conf = config_from_yaml(cwd / "world.yaml", m="ttc", evader="pid", n=n, seed=seed, g=genome)
        world_conf.metadata = {"hash": hash(tuple(list(hash_val))) if hash_val is not None else None}
        worlds = [world_conf]

        return worlds

    return gene_to_world

def test_cma(n, seed, iters, pop_size):
    def _fitness(world):
        assert len(world) == 1
        return world[0].metrics[0].value

    cmaes = CMAES(
        _fitness,
        genome_to_world=get_world_generator(n=n, seed=seed),
        dvars=DECISION_VARS,
        num_processes=None,
        show_each_step=False,
        target=PERFECT_SCORE,
        experiment=None,
        max_iters=iters,
        pop_size=pop_size,
        round_to_every=None
    )

    result, _ = cmaes.minimize()
    unnormalized_genome = DECISION_VARS.unit_unnormalize(result.best_feasible["x"])
    best_conf = get_world_generator(n=n)(unnormalized_genome)
    return test_single(best_conf[0])

def test_mp_w_cma(samples=100, n_range=None, iters=5, pop_size=10):
    seeds = np.random.default_rng(config.seed).integers(0, 2**31, size=samples)
    results = []
    for n in n_range or range(1, 10):
        print(f"\n\n************************\n\n      n = {n}\n\n************************\n\n")
        # configs = [(n, seeds[i]) for i in range(samples)]
        # ret_arr = process_map(test_cma, configs)
        ret_arr = []
        for i in range(samples):
            ret_arr.append(test_cma(n, seeds[i], iters=iters, pop_size=pop_size))

        stats, ttcs = zip(*ret_arr)
        print('n: ', n, sum(stats, Counter()))
        for stat in stats:
            results.append({
                'n': n,
                **stat
            })


    # ttcs = np.array(ttcs)
    # sns.histplot(ttcs)
    # plt.show()
    print(results)

    import json
    with open("test.json", "w") as f:
        json.dump(results, f)

    return results

if __name__ == "__main__":
    # test_mp()
    # # test_grid()
    # # test_seq()
    # run()

    # import time
    # start = time.time()
    # test_cma((8, 2023))
    # print(f"Took {time.time() - start} seconds")

    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, default=1000, help="Environment Horizon")
    parser.add_argument("-s", "--samples", type=int, default=30, help="Number of samples")
    parser.add_argument("-i", "--iters", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("-p", "--pop_size", type=int, default=10, help="Population size")

    start = time.time()
    args = parser.parse_args()
    test_mp_w_cma(samples=args.samples, n_range=[2, 3, 4, 5], iters=args.iters, pop_size=args.pop_size)
    print(f"Took {time.time() - start} seconds")
