#!/bin/bash



## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name={{ jobname }}

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=normal

## Separate output and error messages into 2 files.`
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output={{ projpath }}/%x-%N-%j.out  # Output file`
#SBATCH --error={{ projpath }}/%x-%N-%j.err   # Error file`

## Slurm can send you updates via email
#SBATCH --mail-user=mabayneh@gmu.edu       # Put your GMU email address here
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem={{ mem|default('8G') }}        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-12:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 48


## Load the relevant modules needed for the job
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
cd ~/neuromorphic/turtwig

export MAKE_PARENTS=true
export PROJ_EXIST_OKAY=true

## Run your program or script
python experiment_tenn2.py train {{ projpath }} -p 48 --save_best --epochs 1000 \
 --eons_seed {{ eons_seed }} -N {{ N }} --behavior {{ behavior }} --cycles 1500 \
 --rngstrat {{ rngstrat|default('TS1') }} -j cm {{ cm }}

python experiment_tenn2.py run {{ projpath }} --cycles 1500 --viz_trails 2000x2000 --noviz \
-j cm {{ cm }} --caspian
