#!/bin/bash

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=snn_new_metrics_02

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=normal

## Separate output and error messages into 2 files.`
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=mabayneh@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=4G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-10:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 48

MAIN_NEURO=$HOME"/neuromorphic/"
SCRATCH_NEURO=$SCRATCH"/neuromorphic_new_metrics_02/"
TURTWIG_BRANCH="new_metrics"

## Load the relevant modules needed for the job
module load git
module load use.own pyenv

# Setup neuromorphic folders in /scratch
mkdir -p $SCRATCH_NEURO
cd $SCRATCH_NEURO
ln -s $MAIN_NEURO"framework/" framework
ln -s $MAIN_NEURO"RobotSwarmSimulator/" RobotSwarmSimulator
git clone https://github.com/michabay05/neuroswarm.git turtwig

# NOTE: just doing 'cd turtwig' backfired because zoxide was trying to be too smart...
cd ./turtwig
# Ensure `results` folder exists
mkdir -p results

git switch -c $TURTWIG_BRANCH "origin/"$TURTWIG_BRANCH

# Setup correct python version
PYTHON_LATEST="$(pyenv latest 3)"
pyenv global $PYTHON_LATEST
python --version
pip --version
source $MAIN_NEURO"framework/pyframework/bin/activate"

# For debugging purposes...after the zoxide issue
pwd

export MAKE_PARENTS=true

## Run your program or script
python experiment_tenn2.py train \
    --root mich_results \
    -p 48 \
    --save_all \
    --epochs 500 \
    --eons_seed 20 \
    -N 6 \
    --pname_suffix new_mets_es20_n6_trial10 \
    --world rss/turbopi-milling/world.yaml \
    --behavior Aggregation \
    --trial 10
