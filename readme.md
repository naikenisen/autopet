Instruction pour utiliser l'agent ssh
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/cle
```

Instruction pour installer les dépendances dans un environnement virtuel Python sur le CCUB
```bash
module load python
python3 -m venv venv
pip3 install --prefix=/work/imvia/in156281/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/venv/lib/python3.9/site-packages:$PYTHONPATH
```

Instruction pour configurer les variables d'environnement avant l'exécution
```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Configurer les chemins pour éviter les erreurs de permissions
export PYTHONPATH=/work/imvia/in156281/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
export WANDB_CACHE_DIR=/work/imvia/in156281/.cache/wandb
export WANDB_CONFIG_DIR=/work/imvia/in156281/.config/wandb

# Créer les répertoires si nécessaires
mkdir -p /work/imvia/in156281/.cache/matplotlib
mkdir -p /work/imvia/in156281/.cache/wandb
mkdir -p /work/imvia/in156281/.config/wandb

# Lancer le script d'entraînement
python3 training.py
```
Lancer un job sur le CCUB avec le script `train.sh`

```bash
qsub train.sh
```
Surveiller les jobs en cours d'exécution
```bash
qstat
# supprimer un job
qdel <job_id>
```