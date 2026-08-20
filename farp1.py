import os
import sys
from pathlib import Path
from functools import partial
from itertools import product

# Add proj-neuro to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'turtwig'))

import experiment_tenn2 as t2
import tqdm
from tqdm.contrib.concurrent import process_map
import pandas as pd

# typing
from typing import Sequence


def parse_rangelist(s: str | range | Sequence[int]):
    """parse a comma-separated list of integers or ranges of integers"""
    if isinstance(s, Sequence) and not isinstance(s, str):
        return list(s)
    segments = [seg.lstrip() for seg in s.strip().split(',')]
    li = []
    for seg in segments:
        if '-' in seg:
            start, end = seg.split('-')
            li.extend(range(int(start), int(end) + 1))
        else:
            li.append(int(seg))
    return list(dict.fromkeys(li))  # remove duplicates


class FARP1Experiment(t2.ConnorMillingExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    # def __init__(self, args):
    #     super(t2.ConnorMillingExperiment, self).__init__(args)
    #     self.world_yaml = args.world_yaml
    #     self.run_info = None

    #     self.track_history = args.track_history or args.log_trajectories
    #     self.log_trajectories = args.log_trajectories
    #     self.use_caspian = getattr(args, 'caspian', True)

    #     if self.agents is None and self.args.action != 'train':
    #         try:
    #             self.agents = self.p.experiment['agents']
    #         except (KeyError, IndexError, FileNotFoundError, AttributeError):
    #             pass

    #     self.start_paused = getattr(args, 'start_paused', False)

    #     self.n_inputs, self.n_outputs, _, _ = self.bootstrap_controller_encoders()

    #     self.log("initialized farp1")

    pass


def get_parsers(parser, subpar):
    parser, subpar = t2.get_parsers(parser, subpar)
    sp = subpar.parsers

    sp['test'].add_argument('--rng_seed', type=int, default=...,
                                help="rng seed for the app")
    sp['test'].add_argument('--Nrange', type=str, default=range(1, 10),
                                help="rng seed for the app")
    sp['test'].add_argument('--trials', type=int, default=10,
                                help="number of trials to run. Set to None to run one trial with world.yaml[seed]."
                                " Values greater than 0 will use the world.yaml[seed] to generate more seeds.")
    return parser, subpar


def single_fitness(self, seed=None, n=None, init_callback=None):
    def modify(self, simargs, seed):
        simargs['world_config'].seed = seed
        if n is not None:
            simargs['world_config'].spawners[0]['n'] = n
        return init_callback(self, simargs) if init_callback else simargs

    # seed = self.fetch_world_config().seed if seed is None else seed
    world_final_state = self.simulate(None, self.net, partial(modify, seed=seed))
    metric = self.pick_metric(world_final_state, self.args.behavior)
    return {
        'fitness': self.extract_fitness(world_final_state, metric),
        'metric': metric.name,
        'seed': world_final_state.seed,
        'n': world_final_state.config.spawners[0]['n'],
    }


def mp_fitness(bundle):
    app, n, seed = bundle
    return single_fitness(app, n=n, seed=seed)


def test(app: FARP1Experiment, args):

    # Set up simulator and network

    if args.stdin == "stdin":
        proc = None
        net = None
    else:
        # proc = caspian.Processor(app.processor_params)
        proc = None
        net = app.net

    ns = parse_rangelist(args.Nrange)
    print(ns)
    print(args.trials)
    seeds = app.seeds or [None] * args.trials
    print(seeds)
    bundles = tuple(product([app], ns, seeds))
    print(bundles)

    if args.processes == 1 or (args.processes is None and os.cpu_count() == 1):
        print(f"Using single thread.")
        results = [single_fitness(app, seed=seed, n=n, init_callback=app.init_callback)
                   for _app, n, seed in tqdm.tqdm(bundles)]
    else:
        if args.processes is None:
            print(f"Using {os.cpu_count()} detected CPUs/threads.")
        else:
            print(f"Using {args.processes} threads.")

        # app handles making seeds based on number of trials from args
        results = process_map(mp_fitness, bundles, max_workers=args.processes)

        for res in results:
            print(f"N: {res['n']}\tSeed {res['seed']}\t\tFitness ({res['metric']}): {res['fitness']:8.4f}")

        df = pd.DataFrame(results)
        df.to_csv(f"results/{args.environment}-{args.label}.csv")
        # print(f"Sum: {sum(fitness):8.4f} \t Avg: {sum(fitness) / len(fitness):8.4f} \t Std: {np.std(fitness):8.4f}")
        # print(f"Min: {min(fitness):8.4f} \t Max: {max(fitness):8.4f} \t out of {len(fitness)} trials")

    if args.explore:
        app.p.explore()

    return df


if __name__ == "__main__":
    t2.main(name="farp1-v01",
            cls=FARP1Experiment,
            parser_callback=get_parsers,
            test=test)
