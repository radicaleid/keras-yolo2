#!/usr/bin/env bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT --jobname atlas_yolo

MODELDIR=/projects/datascience/parton/atlasml/keras-yolo2

module unload darshan
module load horovod
module load keras
echo PYTHON_VERSION=$(python --version 2>&1 )

source /opt/intel/vtune_amplifier/amplxe-vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/vtune_amplifier/lib64
export PE_RANK=$ALPS_APP_PE
export PMI_NO_FORK=1

env | sort
PPN=2
NTHDS=64
INTER=1
COMBO=${PPN}_${NTHDS}_${INTER}
echo [$SECONDS] run job PPN=$PPN NTHDS=$NTHDS INTER=$INTER COBALT_PARTSIZE=$COBALT_PARTSIZE
aprun -n $(( ${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NTHDS} -j 2 \
   amplxe-cl -collect advanced-hotspots -finalization-mode=none -r ./${COBALT_JOBID}_amplxe -data-limit=0  -- \
   python $MODELDIR/train.py -c config.json --num_intra=$NTHDS --num_inter=$INTER --tb_logdir=logs/${COBALT_JOBID}_${COMBO}_logs --horovod
echo [$SECONDS] exited $?
