#!/usr/bin/env bash
#COBALT -n 4
#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT --jobname atlas_yolo_cray

MODELDIR=/projects/datascience/kristyn/jtchilders/keras-yolo2

module load keras
module use /projects/datascience/kristyn/modulefiles
module load craype-ml-plugin-py3/1.1.2
echo PYTHON_VERSION=$(python --version)

env | sort
PPN=2
NTHDS=64
INTER=1
COMBO=${PPN}_${NTHDS}_${INTER}
echo [$SECONDS] run job PPN=$PPN NTHDS=$NTHDS INTER=$INTER COBALT_PARTSIZE=$COBALT_PARTSIZE
aprun -n $(( ${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -cc depth -d ${NTHDS} -j 2 \
   python ${MODELDIR}/train.py -c config.json --num_intra=$NTHDS --num_inter=$INTER --tb_logdir=logs/${COBALT_JOBID}_${COMBO}_logs --ml_comm
echo [$SECONDS] exited $?
