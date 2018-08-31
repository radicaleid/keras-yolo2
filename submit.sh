#!/usr/bin/env bash
#COBALT -n 2
#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT --jobname atlas_yolo

MODELDIR=/projects/datascience/parton/atlasml/keras-yolo2

module load tensorflow
module load horovod
echo PYTHON_VERSION=$(python --version 2>&1 )

export OMP_NUM_THREADS=128
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
# export KMP_AFFINITY="granularity=fine,compact,1,0"

aprun -n 1 -N 1 python -c "import tensorflow as tf;print('tensorflow version: ',tf.__version__)"

aprun -n 2 -N 1 -d 128 -j 2 -cc depth python $MODELDIR/train.py -c config.json

