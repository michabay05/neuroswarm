import shutil
import getpass
import subprocess
import pandas as pd
import pathlib as pl
from itertools import product
from swarmsim.util.jinja import make_default_jinja_env
from wizards import RunParametrizer


env = make_default_jinja_env(line_statement_prefix='#%:')

thisfile = pl.Path(__file__)
wd = thisfile.parent
template_path = wd / 'behavior.jinja.slurm'
scratch = pl.Path(f"/scratch/{getpass.getuser()}")


with open(template_path, 'r') as f:
    template = env.from_string(f.read())

eons_seeds = [
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035,
    2036,
]
swarm_sizes = [
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
]
rngstrats = [
    'TS1',
    'TSG',
    'TSR',
]
shortnames = {
    'Circliness': 'mill',
    'Aggregation': 'aggr',
    'ExplodingDispersion': 'disp',
    'DelaunayDiffusion': 'diff',
    'MovingMass': 'mvm',
    'Boids': 'bds',
}
behaviors = [
    'Circliness',
    'Aggregation',
    'ExplodingDispersion',
    'DelaunayDiffusion',
    'MovingMass',
    'Boids',
]
controller_modes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
]

# ask the user to choose parameters
behaviors, swarm_sizes, eons_seeds, rngstrats, controller_modes = RunParametrizer({
    'Behavior': behaviors,
    'Number of Agents': swarm_sizes,
    'EONS Seed': eons_seeds,
    'RNG Strategy': rngstrats,
    'Controller Tie Mode': controller_modes,
}).run().values()
# TODO: persisting old parameter selections to file
# TODO: pasting parameter lists from clipboard. need to check types.

configs = []
for behavior, swarm_size, eons_seed, rngstrat, cm in product(
    behaviors, swarm_sizes, eons_seeds, rngstrats, controller_modes
):
    bhvr = shortnames[behavior]
    projname = f"{bhvr}-es{eons_seed}-{rngstrat}-n{swarm_size}-cm{cm}"
    configs.append(dict(
        eons_seed=eons_seed,
        N=swarm_size,
        bhvr=bhvr,
        behavior=behavior,
        projname=projname,
        jobname=projname,
        projpath=scratch / f'{bhvr}/{swarm_size}' / projname,
        rngstrat=rngstrat,
        cm=cm,
    ))

# TODO: textual app to examine generated slurm files, modify slurm parameters like time
# TODO: textual interface check for and delete old runs, inspect .err, etc.

print(pd.DataFrame(configs))
input('Press enter to continue or ctrl-c to cancel.')

existing_runs = [path for cfg in configs for path in [cfg['projpath']]
                 if path.is_dir() and path.glob('*.err')]

if existing_runs:
    print(f"WARNING: Detected existing logs from previous runs for:")
    for p in existing_runs:
        print(p)
    print("\nType 'REMOVE' to delete them, or 'CONTINUE' to continue anyway.")
    response = input(': ')
    if response.strip() == 'REMOVE':
        for p in existing_runs:
            print(f"Removing {p}")
            shutil.rmtree(p)
        print("Done.")
    elif response.strip() == 'CONTINUE':
        pass
    else:
        print("Invalid response received. Aborting.")
        exit(-1)

for d in configs:
    projpath = d['projpath']
    projpath.mkdir(parents=True, exist_ok=True)
    slurm = template.render(**d)
    print(str(projpath))
    with open(f"{projpath}/sbatch.slurm", 'w') as f:
        f.write(slurm)
    subprocess.run(f"sbatch {projpath}/sbatch.slurm", shell=True)
