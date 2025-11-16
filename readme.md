Instruction pour installer les dépendances dans un environnement virtuel Python sur le CCUB
```bash
module load python
python3 -m venv venv
pip3 install --prefix=/work/imvia/in156281/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/venv/lib/python3.9/site-packages:$PYTHONPATH