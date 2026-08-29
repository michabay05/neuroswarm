#!/bin/bash

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=snn_cont_modes_01

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
#SBATCH --array=0-6


## Load the relevant modules needed for the job
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
mkdir -p mich_results
cd turtwig

export MAKE_PARENTS=true

CIRCLINESS_INDEX=0

## Run your program or script
python experiment_tenn2.py train \
    --root mich_results \
    -p 48 \
    -j cm $SLURM_ARRAY_TASK_ID \
    --save_all \
    --epochs 500 \
    --eons_seed 20 \
    -N 6 \
    --pname_suffix cont_modes_es20_n6_trial10 \
    --world rss/turbopi-milling/world.yaml \
    --behavior CIRCLINESS_INDEX \
    --trial 10