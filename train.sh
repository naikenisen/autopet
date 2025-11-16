#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N unet_gang
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
export WANDB_CACHE_DIR=/work/imvia/in156281/.cache/wandb
export WANDB_CONFIG_DIR=/work/imvia/in156281/.config/wandb
python /beegfs/data/work/imvia/in156281/unet_gang/training.py