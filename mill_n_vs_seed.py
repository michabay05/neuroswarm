"""Generate data to plot how well an SNN performs across swarm sizes it wasn't trained on."""
import os
import re
import sys
import copy
import glob
import pathlib as pl
from functools import partial
from itertools import product

import tqdm
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

import common.experiment
import experiment_tenn2 as t2
from common.util import parse_rangelist
from common.project import UnzippedProject

try:
    from matplotlib.backends.backend_pgf import LatexManager
    LatexManager()
    use_pgf = True
except RuntimeError:
    use_pgf = False

wd = pl.Path(__file__).parent
cls = t2.ConnorMillingExperiment

# desktop = pl.Path('/mnt/c/Users/kenbl/Desktop').expanduser()
default_save_path = wd / "results/mill_n_vs_seed.csv"

# # folder = desktop / '20260820mill' / 'mill'
# globs = [
#     # "/mnt/c/Users/kenbl/Desktop/four/mill/*/*",
#     "/scratch/kzhu4/mill/*/*",
#     # "/scratch/kzhu4/aggr/*/*",
#     # "/scratch/kzhu4/disp/*/*",
#     # "/scratch/kzhu4/diff/*/*",
# ]
# include = r'mill.*TS1'
# exclude = r'zip|\.7z'

# projects = [UnzippedProject(p) for g in globs for p in glob.glob(g)
#             if (exclude is None or not re.search(exclude, p)
#                 and (include is None or re.search(include, p)))]
# print(*[p.root for p in projects], sep='\n')


def get_parsers(parser, subpar):
    parser, subpar = t2.get_parsers(parser, subpar)
    sp = subpar.parsers

    sp['test'].add_argument('project', nargs='+',
                           help="Specify globs to projects to generate images for.")
    sp['test'].add_argument('--rng_seed', type=int, default=None,
                                help="rng seed for the app")
    sp['test'].add_argument('--Nrange', type=str, default=range(3, 50),
                                help="range of swarm sizes to test")
    sp['test'].add_argument('--trials', type=int, default=100,  # changed default from single
                                help="number of trials to run. Set to None to run one trial with world.yaml[seed]."
                                " Values greater than 0 will use the world.yaml[seed] to generate more seeds.")
    sp['test'].add_argument('--exclude', help="regex to exclude paths. Applied per discovered path")
    sp['test'].add_argument('--include', help="Discovered path(s) must match this regex. Default includes all.")
    sp['test'].add_argument('--csvpath', type=pl.Path, default=default_save_path, help="path to save csv file")
    sp['test'].add_argument('--perproject', action='store_true',
                                help="After computing all data, save relevant data and plots into each project.")
    sp['test'].add_argument('--force', help="Skip confirmation prompts.", action='store_true')
    return parser, subpar


def plot_boxplot(df: pd.DataFrame, max_fit: float | None = None, eons_seed=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    train_n = df['train_n'].values[0] if df['train_n'].nunique() == 1 else None
    metric_name = df['metric'].values[0] if df['metric'].nunique() == 1 else None
    sns.set_theme(style='whitegrid', palette='pastel', context='talk')
    if max_fit is not None:
        plt.axhline(max_fit, color='k', linestyle='--', alpha=0.5)
    sns.boxplot(df, x='test_n', y='fitness', hue='train_n', legend=False)
    sns.despine(offset=2, left=True)
    plt.xlabel('$N_\\mathrm{test}$')
    plt.ylabel('Fitness' + f' ({metric_name})' if metric_name else '')
    seedmsg = '' if eons_seed is None else f', $\\mathrm{{eons\\_seed}}={eons_seed}$'
    if train_n:
        plt.title(f'$N_\\mathrm{{train}}={train_n}$' + seedmsg)
    plt.tight_layout()
    return plt


def save_to_project(project: UnzippedProject, data_path: pl.Path):
    df = pd.read_csv(data_path, index_col=0)
    df = df[df['path'] == str(project.root)]
    df.to_csv(project / 'cross_n.csv')
    try:
        popfits, _times = project.read_popfit_df_wide()
    except FileNotFoundError:
        popfits = None
    es = df['eons_seed']
    plt = plot_boxplot(df,
                       max_fit=float(popfits.max(axis=None)) if popfits is not None else None,
                       eons_seed=es.values[0] if es.nunique() == 1 else None)
    plt.savefig(project / 'cross_n.pdf', backend='pgf' if use_pgf else None)
    plt.close()


def save_to_project_mp(bundle):
    project, data_path = bundle
    save_to_project(project, data_path)


def single_fitness(args, n, seed):
    # seed = self.fetch_world_config().seed if seed is None else seed
    app = cls(args)
    world_final_state = app.simulate(None, app.net, seed=seed, n=n)
    assert world_final_state.config.spawners[0]['n'] == n
    assert app.agents is not None
    assert world_final_state.seed is not None
    metric = app.pick_metric(world_final_state, app.args.behavior)
    try:
        eons_seed = app.p.evolver['eons_params']['seed_eo']
    except KeyError:
        eons_seed = app.p.experiment['args']['eons_seed']
    return {
        'path': app.p.root,
        'eons_seed': eons_seed,
        'train_n': app.agents,
        'test_n': n,
        'seed': world_final_state.seed,
        'metric': metric.name,
        'fitness': app.extract_fitness(world_final_state, metric),
    }


def mp_fitness(bundle):
    app, n, seed = bundle
    return single_fitness(app, n=n, seed=seed)


def test(args, silent=False):
    def prnt(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    projects = [UnzippedProject(p) for g in args.project for p in glob.glob(g)
                if (args.exclude is None or not re.search(args.exclude, p))
                    and (args.include is None or re.search(args.include, p))]

    args_copies = []
    print(f"Matched {len(projects)} projects")
    for project in projects:
        args_copy = copy.deepcopy(args)
        if not project.possibly_valid():
            msg = f"Project {project} is not valid"
            raise RuntimeError(msg)
        args_copy.project = project
        args_copy.root = None
        args_copies.append(args_copy)

    ns = parse_rangelist(args.Nrange)
    if args.trials is not None:
        seeds = np.random.default_rng(args.rng_seed).integers(0, 2**32, size=args.trials)
    else:
        seeds = [args.rng_seed]
    prnt(seeds)
    bundles = tuple(product(args_copies, ns, seeds))
    pd.options.display.max_colwidth = 128
    pd.options.display.max_rows = 200
    pd.options.display.min_rows = 200
    prnt(pd.DataFrame(bundles))
    if not args.force:
        input("Press enter to continue, ctrl-c to cancel.")

    if args.processes == 1 or (args.processes is None and os.cpu_count() == 1):
        prnt(f"Using single thread.")
        results = [single_fitness(*bundle) for bundle in tqdm.tqdm(bundles)]
    else:
        if args.processes is None:
            prnt(f"Using {os.cpu_count()} detected CPUs/threads.")
        else:
            prnt(f"Using {args.processes} threads.")

        # app handles making seeds based on number of trials from args
        results = process_map(mp_fitness, bundles, max_workers=args.processes)

    for res in results:
        prnt(f"{res['test_n']:2d} agents trained with {res['train_n']:2d}\tSeed {res['seed']}"
                f"\tFitness ({res['metric']}): {res['fitness']:8.4f}")

    df = pd.DataFrame(results)
    df.to_csv(args.csvpath)
    prnt(f"Saved to {args.csvpath}")
    if args.perproject:
        prnt("Saving plots/data to each project...")
        bundles = tuple(product(projects, [args.csvpath]))
        if args.processes == 1 or (args.processes is None and os.cpu_count() == 1):
            for bundle in tqdm.tqdm(bundles):
                save_to_project(*bundle)
        else:
            process_map(save_to_project_mp, bundles, max_workers=args.processes)

    return df


if __name__ == "__main__":
    parser, subpar = get_parsers(*t2.get_parsers(*common.experiment.get_parsers()))
    thisfile, *argv = sys.argv
    args = parser.parse_args(['test', *argv])
    args.environment = "mill-n-vs-seed-v01"
    test(args)
