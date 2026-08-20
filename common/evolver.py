import os
import re
import time
import datetime
import numbers
import numpy as np
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import tqdm

import neuro
import eons

from .tennnetwork import make_template
from .application import Application

from dataclasses import dataclass
from typing import ClassVar

# typing
from typing import Tuple, override


DEFAULT_MOA_STATE = [1, 1, 1, 1, 1]


@dataclass
class EpochInfo:
    """Class for the results of an epoch"""
    i: int
    t_start: float
    t_fitness: float
    t_eons: float
    t_end: float
    t_elapsed: float
    best_net_id: int
    best_network: neuro.Network
    best_fitness: float
    best_score: float = None
    validation: float | None = None
    fitnesses: Tuple[float, ...] = ()

    # regex for parsing str, not a field
    RE_EPOCH: ClassVar = re.compile(r'(?P<ts>[1-2]\d{7}\s?[\d:]{0,8}\s?)>>>\s?Epoch\s*(?P<epoch>\d+):\s*(?P<score>-?[\d.]+):\s*(?P<fit>-?[\d.]+)\s\|\sNeurons:\s*(?P<neur>\d+)\sSynapses:\s*(?P<syn>\d+)\s\|\sTime:\s*([\d.]+)s?\s+(?:(?P<eonsms>[\d.]+)ms|(?P<eonsec>[\d.]+s?))\s+([\d.]+)s?')  # noqa: E501

    @property
    def t_total(self) -> float:
        return self.t_end - self.t_start

    def __str__(self):  # type: ignore[reportImplicitOverride]
        # Epoch number and fitness score
        out = f"Epoch {self.i:3d}: {self.best_fitness:11.4f}"
        out += '' if self.best_score is None else f":{self.best_score:11.4f}"
        # Validation score if a validation function is present
        if self.validation is not None:
            out += " {:11.4f}".format(self.validation)
        # Network size (nodes / edges)
        out += f" | Neurons: {self.best_network.num_nodes():3d} Synapses: {self.best_network.num_edges():3d}"
        # Timing information
        eons_ms = self.t_eons * 1000
        out += f" | Time: {self.t_fitness:7.4f}s {eons_ms:5.1f}ms {self.t_total:6.4f}"
        return out

    @property
    def num_neurons(self):
        return self.best_network.num_nodes() if self.best_network else self._num_nodes

    @property
    def num_synapses(self):
        return self.best_network.num_edges() if self.best_network else self._num_edges

    @classmethod
    def from_str(cls, s, error=True):
        match = cls.RE_EPOCH.search(s)
        if not match:
            if error:
                msg = f"Could not parse string {s}"
                raise ValueError(msg)
            return None
        (t_end, epoch, score, fit, neur, syn,
         t_fitness, eonsms, eonsec, t_total) = match.groups()
        t_end = datetime.datetime.strptime(t_end.strip(), "%Y%m%d %H:%M:%S").timestamp()
        new = cls(
            i=int(epoch),
            t_start=t_end - float(t_total),
            t_fitness=float(t_fitness),
            t_eons=float(eonsms) / 1000 if eonsms else float(eonsec),
            t_end=t_end,
            t_elapsed=float(t_total),
            best_net_id=None,
            best_network=None,
            best_fitness=float(fit),
            best_score=float(score),
            validation=None,
        )
        new._num_nodes = int(neur)
        new._num_edges = int(syn)
        return new


class Evolver:

    def __init__(
        self,
        *,
        app,
        eons_params,
        proc_name,
        proc_params,
        stop_fitness=None,
        do_print=True,
        tqdm=None,
    ):

        if not isinstance(eons_params, dict) or not isinstance(proc_params, dict):
            raise TypeError("Passed EONS/Processor parameters must be dictionaries")

        if not isinstance(app, Application):
            raise TypeError("The application must derive from Application")

        # if environment variable $EONS_DISABLE_PRINT is set, override `do_print` to be False
        self.do_print = do_print and not os.environ.get("EONS_DISABLE_PRINT", "").strip()

        self.app = app
        self.eons_params = eons_params
        self.proc_name = proc_name
        self.proc_params = proc_params
        self.sim = proc_name(proc_params)

        if not isinstance(self.sim, neuro.Processor):
            raise TypeError("The processor must derive from neuro.Processor")

        # pop our custom eons params so EONS doesn't complain (with defaults)
        self.penalty = eons_params.pop('penalty', None)

        self.stop_fitness = float('inf') if stop_fitness is None else stop_fitness

        self.eo = eons.EONS(eons_params)
        self.initialize_population()
        self.epoch = 0
        self.best: float = float('-inf')
        self.pocket: EpochInfo | None = None
        self.scores_by_epoch = list()
        self.tqdm = tqdm
        self.t_start = 0

        self.net_callback = lambda net: net

    def initialize_population(self, moa_state: list[int] | None = DEFAULT_MOA_STATE):
        if self.do_print:
            t0 = time.time()

        # Create a template network with the right number of inputs & outputs
        template_net = make_template(self.sim, self.app.n_inputs, self.app.n_outputs, self.eo.rng)
        self.eo.set_template_network(template_net)

        # Generate a new initial population for this EONS instance
        self.pop = self.generate_population(self.eons_params, moa_state)

        if self.do_print:
            print("Initialized population of {} networks in {:8.5f} seconds".format(
                len(self.pop.networks), time.time() - t0))

    def generate_population(self, eons_params, moa_state: list[int] | None = None):
        if moa_state is None:
            return self.eo.generate_population(eons_params)
        elif moa_state == DEFAULT_MOA_STATE:
            try:
                return self.eo.generate_population(eons_params, moa_state)
            except TypeError:
                return self.eo.generate_population(eons_params, 1)
        else:
            return self.eo.generate_population(self.eons_params, moa_state)

    def pre_epoch(self):
        f = getattr(self.app, 'pre_epoch', None)
        if callable(f):
            f(self)

    def post_epoch(self, epoch_info, new_best):
        f = getattr(self.app, 'post_epoch', None)
        if callable(f):
            f(self, epoch_info, new_best)

    def evaluate_population(self, networks):
        generator = (self.app.fitness(self.sim, network, self.epoch + 1) for network in self.net_callback(networks))
        if self.tqdm is True:
            return [x for x in tqdm.tqdm(generator, total=len(networks))]
        if self.tqdm:
            return self.tqdm(generator, total=len(networks))
        else:
            return list(generator)

    def evaluate_validation(self, network):
        return self.app.validation(self.sim, network)

    def fitness_with_penalty(self, fitnesses, networks):
        # get scores from zipped bundles
        # bundle should be list(zip(networks, fitnesses))
        if self.penalty is None:
            scores = fitnesses
        elif callable(self.penalty):
            scores = self.penalty(fitnesses, networks)
        else:
            # penalize networks by the number of nodes and edges.
            c1, c2 = self.penalty  # this is retrieved from eons_params in __init__
            scores = [
                (fitness - (net.num_nodes() * c1 + net.num_edges() * c2))
                for fitness, net in zip(fitnesses, networks)
            ]  # see 4.5 Multi-Objective Optimization
            # "Evolutionary Optimization for Neuromorphic Systems", Schuman et al. 2020
        return scores

    def do_epoch(self, update_params={}, **kwargs):  # noqa: B006
        t_start = time.time()

        if self.epoch == 0:
            self.t_start = t_start

        # Do any pre-epoch operations
        self.pre_epoch()

        # Get the fitness for each network in the population
        t_fs = time.time()
        networks = [nn.network for nn in self.pop.networks]
        self.fitness = self.evaluate_population(networks)
        t_fitness = time.time() - t_fs

        # apply penalty function
        scores = self.fitness_with_penalty(self.fitness, networks)

        # bundle
        bundles = enumerate(zip(networks, self.fitness, scores))
        best = max(bundles, key=lambda b: b[1][-1])  # get best net by score
        topscoring_net_id, (topscoring_net, topscoring_fitness, topscore) = best
        self.scores_by_epoch.append(topscore)

        # Validation score for the best network
        # If an app does not specify a validation score, it should return `None`
        validation = self.evaluate_validation(topscoring_net)

        # Evolve the next population with EONS
        t_es = time.time()
        self.pop = self.eo.do_epoch(self.pop, scores, update_params)
        t_eons = time.time() - t_es

        # Increment our epoch counter
        self.epoch += 1

        t_elapsed = t_eons - self.t_start

        t_end = time.time()

        info = EpochInfo(
            self.epoch,
            t_start,
            t_fitness,
            t_eons,
            t_end,
            t_elapsed,
            topscoring_net_id,
            topscoring_net,
            topscoring_fitness,
            topscore,
            validation,
            tuple(self.fitness),  # every score in the population
        )

        new_best = False
        # save new incumbent solution if it's better
        if topscoring_fitness > self.best:
            new_best = True
            self.best = topscoring_fitness
            self.pocket = info

        if self.do_print:
            print(info)

        # Do any post-epoch operations
        self.post_epoch(info, new_best)

        return info

    def train(self, n_epochs):
        for _epoch in range(n_epochs):
            self.do_epoch()
            if self.stop_fitness is not None and self.best > self.stop_fitness:
                break  # stop early if stopping condition is met
        return self.pocket

    def graph_fitness(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Fitness')
        ax.plot(self.scores_by_epoch)
        plt.show()

    def as_config_dict(self):
        return {
            "app": self.app,
            "eons_params": self.eons_params,
            "proc_name": self.proc_name,
            "proc_params": self.proc_params,
            "stop_fitness": self.stop_fitness,
            "do_print": self.do_print,
            "tqdm": self.tqdm,
            "t_start": self.t_start,
            "net_callback": self.net_callback,
        }


# Helper function for MP Pool mapping
def mp_fitness(bundle):
    app, net, proc_name, proc_params, eons_i = bundle
    sim = proc_name(proc_params)
    return app.fitness(sim, net, eons_i)


class MPEvolver(Evolver):
    def __init__(self, *, pool=None, max_workers=None, **kwargs):
        super().__init__(**kwargs)
        self.pool = Pool() if pool is None else pool
        self.max_workers = max_workers

    @override
    def evaluate_population(self, networks):
        c = os.cpu_count() if not self.max_workers else self.max_workers
        bundles = ((self.app, net, self.proc_name, self.proc_params, self.epoch + 1) for net in networks)
        if self.tqdm is True:
            return process_map(mp_fitness, bundles, total=len(networks),
                               max_workers=c, chunksize=1)
        elif self.tqdm:
            return process_map(mp_fitness, bundles, total=len(networks),
                               max_workers=c, chunksize=1, tqdm_class=self.tqdm)
        return self.pool.map(mp_fitness, bundles)
