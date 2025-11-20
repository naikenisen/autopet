#!/bin/ksh 
#$ -q batch
#$ -o result.out
#$ -j y
#$ -N preprocess
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/venv/lib/python3.9/site-packages:$PYTHONPATH
python /beegfs/data/work/imvia/in156281/autopet/preprocessing.py