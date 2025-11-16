#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N unet_gang
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/venv/bin/activate
module load python
python /beegfs/data/work/imvia/in156281/unet_gang/training.py
