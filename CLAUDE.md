# CUT – Contrastive Unpaired Translation

Tâche : translation de contours anatomiques CT → style IRM, entraînement non-apparié sur coupes coronales 2D.
Pipeline : volumes `.mat` float32 [0,1] → `MatUnalignedDataset` → CUT → `translate_volume.py` → `.mat` float32 [0,1].

## Architecture

| Modèle | Flag | Notes |
|---|---|---|
| CUT | `--CUT_mode CUT` | Modèle complet, NCE identity loss activée |
| FastCUT | `--CUT_mode FastCUT` | Plus léger, pas d'identity loss, 2× moins de mémoire |

**Composants principaux :**
- `models/cut_model.py` — logique d'entraînement (forward, losses, optimizer)
- `models/networks.py` — Generator (ResNet 9-block), Discriminator (PatchGAN), projecteur MLP (netF)
- `models/patchnce.py` — PatchNCE contrastive loss
- `data/mat_unaligned_dataset.py` — dataset principal, chargement direct `.mat` sans preprocessing
- `data/mat_io.py` — chargement `.mat` partagé (legacy scipy + HDF5 v7.3)
- `data/unaligned_dataset.py` — dataset alternatif pour fichiers `.npy` ou images PNG
- `options/` — tous les arguments CLI via argparse

## Workflow complet

### 1 — Lister les volumes

```
# ct_paths.txt        (un chemin absolu par ligne, # pour les commentaires)
/data/ct/patient001.mat
/data/ct/patient002.mat
```

### 2 — Créer le split.json

```bash
python build_split.py \
    --ct_files  ct_paths.txt  \
    --mri_files mri_paths.txt \
    --split_file split.json   \
    --train_ratio 0.85
```

Format produit — **un seul fichier**, deux domaines :
```json
{
  "trainA": ["/data/ct/p001.mat", ...],
  "trainB": ["/data/mri/p001.mat", ...],
  "testA":  ["/data/ct/p_test.mat", ...],
  "testB":  ["/data/mri/p_test.mat", ...]
}
```

Pas de val : CUT n'a pas de boucle de validation. Le suivi se fait via les images générées dans `checkpoints/<name>/web/`. Pour une évaluation manuelle, ajouter des clés `valA`/`valB` et lancer `test.py --phase val`.

**Volumes recommandés :** ≥ 50 par domaine (chaque volume → ~100-130 coupes non-vides à l'axe coronal 2).

### 3 — Entraîner

```bash
python train.py \
    --split_file    split.json     \
    --dataset_mode  mat_unaligned  \
    --name          mon_experience \
    --CUT_mode      CUT            \
    --input_nc      1              \
    --output_nc     1              \
    --coronal_axis  2              \
    --display_id    -1
```

Au démarrage : scan unique de tous les volumes (~40s pour 300 volumes) pour indexer les coupes non-vides, puis cache LRU par worker DataLoader.

### 4 — Inférence → .mat float32

```bash
# Un volume
python translate_volume.py \
    --input  /data/ct/patient042.mat \
    --output /results/patient042_mri.mat \
    --name   mon_experience \
    --coronal_axis 2

# Batch
python translate_volume.py \
    --input_list ct_test_paths.txt \
    --output_dir /results/ \
    --name mon_experience \
    --coronal_axis 2
```

Sortie : float32 [0,1] exact — le générateur sort des tenseurs float32, aucun passage par uint8.

## Options importantes

| Flag | Défaut | Signification |
|---|---|---|
| `--split_file` | `None` | JSON avec les chemins de volumes par split |
| `--dataset_mode` | `unaligned` | Utiliser `mat_unaligned` pour les .mat directs |
| `--coronal_axis` | `1` | Axe coronal dans le volume (vérifier sur ses données) |
| `--mat_key` | auto | Nom de la variable dans le .mat (auto-détecté) |
| `--min_content` | `0.02` | Seuil de coupes non-vides (fraction de pixels > 0) |
| `--input_nc` | `3` | Mettre `1` pour niveaux de gris |
| `--output_nc` | `3` | Mettre `1` pour niveaux de gris |
| `--display_id` | auto | Mettre `-1` pour désactiver visdom |
| `--name` | `experiment_name` | Nom du run, contrôle le dossier checkpoints |
| `--checkpoints_dir` | `./checkpoints` | Où les poids sont sauvegardés |
| `--n_epochs` | `200` | Epochs à LR initiale |
| `--n_epochs_decay` | `200` | Epochs de décroissance LR linéaire |
| `--gpu_ids` | `0` | IDs GPU séparés par virgule ; `-1` pour CPU |
| `--batch_size` | `1` | 1 pour images 256×256 |
| `--lambda_NCE` | `1.0` | Poids de la PatchNCE loss |
| `--nce_layers` | `0,4,8,12,16` | Couches encoder utilisées pour NCE |

## Arborescence

```
.
├── train.py                    # Entraînement
├── test.py                     # Inférence 2D (sorties PNG)
├── translate_volume.py         # Inférence volumique .mat → .mat
├── build_split.py              # Créer split.json depuis listes de volumes
├── prepare_slices.py           # Optionnel : cache .npy si besoin de perf I/O
├── split.json                  # Votre split (ne pas committer)
├── models/
│   ├── cut_model.py
│   ├── networks.py
│   ├── patchnce.py
│   ├── stylegan_networks.py
│   └── base_model.py
├── data/
│   ├── mat_unaligned_dataset.py  # Dataset principal (.mat directs)
│   ├── mat_io.py                 # I/O .mat partagé
│   ├── unaligned_dataset.py      # Dataset alternatif (.npy / PNG)
│   ├── base_dataset.py           # npy_to_tensor() + transforms
│   └── image_folder.py
├── options/
├── util/
└── checkpoints/                # Poids sauvegardés (gitignored)
```

## Conventions

- Modifications réseau → `models/networks.py`
- Nouvelle loss → ajouter à `loss_names` dans le modèle, calculer dans `calculate_NCE_loss` ou `forward`
- Nouveau flag CLI → `modify_commandline_options` de la classe modèle concernée
- Checkpoints : `<epoch>_net_G.pth`, `<epoch>_net_D.pth`, `latest_net_*.pth`
- `opt` est la source unique de vérité pour tous les hyperparamètres
- Pipeline entièrement float32 — aucun passage par uint8 de l'entrée à la sortie
