import os
import shutil

# Répertoire racine contenant les dossiers PETCT_*
ROOT = "/work/imvia/in156281/data_unet"

# Dossiers de sortie
PET_IMAGES_DIR = os.path.join(ROOT, "pet_images")
CT_IMAGES_DIR = os.path.join(ROOT, "ct_images")
LABELS_DIR = os.path.join(ROOT, "labels")

# Création des dossiers s'ils n'existent pas
os.makedirs(PET_IMAGES_DIR, exist_ok=True)
os.makedirs(CT_IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Parcours des dossiers PETCT_*
for folder in os.listdir(ROOT):
    if not folder.startswith("PETCT_"):
        continue

    folder_path = os.path.join(ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    # Récupère les sous-dossiers et prend uniquement le premier
    subfolders = sorted([
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ])

    if len(subfolders) == 0:
        print(f"[SKIP] Aucun sous-dossier dans {folder}")
        continue

    subfolder_path = os.path.join(folder_path, subfolders[0])

    # Chemins des fichiers PET, CT et SEG
    pet_file = os.path.join(subfolder_path, "PET.nii.gz")
    ct_file = os.path.join(subfolder_path, "CT.nii.gz")
    seg_file = os.path.join(subfolder_path, "SEG.nii.gz")


    # Vérifie la présence des deux fichiers (paire complète)
    if not (os.path.exists(pet_file) and os.path.exists(ct_file) and os.path.exists(seg_file)):
        print(f"[SKIP] Paire PET/CT/SEG incomplète dans {folder}")
        continue

    # Nouveau nom basé sur le nom du dossier parent
    new_name = folder + ".nii.gz"

    # Copie PET → pet_images/
    shutil.copy(pet_file, os.path.join(PET_IMAGES_DIR, new_name))

    # Copie CT → ct_images/
    shutil.copy(ct_file, os.path.join(CT_IMAGES_DIR, new_name))

    # Copie SEG → labels/
    shutil.copy(seg_file, os.path.join(LABELS_DIR, new_name))

    print(f"[OK] Copied PET+CT+SEG for {folder}")